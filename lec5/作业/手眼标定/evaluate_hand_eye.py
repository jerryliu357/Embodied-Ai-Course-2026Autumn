"""Offline hand-eye fit and consistency checks; never opens hardware.

Every original sample is retained. Repeated poses are excluded from fitting,
and alternative subsets are reported explicitly rather than silently trimmed.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from fk_capture_core import HERE, FK_DIR, load_mapping, rotation_angle
from calibrate_intrinsics import board_points, detect, read_image, write_json
from hand_eye_calibration import load_intrinsics, make_T, valid_R

METHODS = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK,
           "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}


def fit(samples, method):
    A = [np.array(s['T_base_end_effector']) for s in samples]
    B = [np.array(s['T_camera_board']) for s in samples]
    R, t = cv2.calibrateHandEye([a[:3,:3] for a in A], [a[:3,3] for a in A],
                              [b[:3,:3] for b in B], [b[:3,3] for b in B], method=METHODS[method])
    valid_R(R)
    if not np.isfinite(t).all():
        raise ValueError('Nonfinite translation')
    return make_T(R, t)


def board_mean(transforms):
    Rmean = np.mean([t[:3,:3] for t in transforms], axis=0)
    U, _, Vt = np.linalg.svd(Rmean)
    R = U @ np.diag([1, 1, np.linalg.det(U@Vt)]) @ Vt
    return make_T(R, np.mean([t[:3,3] for t in transforms], axis=0))


def board_poses(samples, X):
    return [np.array(s['T_base_end_effector']) @ X @ np.array(s['T_camera_board']) for s in samples]


def errors(transforms, reference):
    return [{'position_mm': float(np.linalg.norm(t[:3,3]-reference[:3,3])*1000),
             'orientation_deg': rotation_angle(reference[:3,:3].T@t[:3,:3])} for t in transforms]


def summary(rows):
    result = {}
    for key in ('position_mm', 'orientation_deg'):
        values = np.array([r[key] for r in rows])
        result[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                           rms=float(np.sqrt(np.mean(values**2))), max=float(values.max()))
    return result


def evaluate(samples, method):
    X = fit(samples, method)
    boards = board_poses(samples, X)
    reference = board_mean(boards)
    residuals = errors(boards, reference)
    validation, transforms = [], []
    for i, sample in enumerate(samples):
        train = samples[:i] + samples[i+1:]
        Xi = fit(train, method)
        transforms.append(Xi)
        predicted = board_poses([sample], Xi)
        validation.extend(errors(predicted, board_mean(board_poses(train, Xi))))
    return dict(T_end_effector_camera=X.tolist(), camera_translation_m=X[:3,3].tolist(),
                T_base_board_mean=reference.tolist(), training=summary(residuals),
                leave_one_out=summary(validation),
                leave_one_out_transform_change=summary(errors(transforms, X)),
                per_sample=[dict(index=s['index'], **r, held_out=v)
                            for s,r,v in zip(samples, residuals, validation)])


def audit(folder):
    meta = json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    K, dist = load_intrinsics(meta['intrinsics_file'])
    data = json.loads((FK_DIR/'calibration.json').read_text(encoding='utf-8-sig'))
    mapping = load_mapping(data)
    samples, duplicates, image_checks = [], [], []
    all_samples = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'samples').glob('sample_*.json'))]
    for s in all_samples:
        A, B = np.array(s['T_base_end_effector']), np.array(s['T_camera_board'])
        valid_R(A[:3,:3]); valid_R(B[:3,:3])
        np.testing.assert_allclose(A, mapping.arm.fk(mapping.to_q(s['raw_positions'])), atol=1e-10)
        frame = read_image(folder/'images'/f"sample_{s['index']:02d}.png")
        if frame.shape != (720,1280,3):
            raise ValueError('Unexpected image resolution')
        corners = detect(frame)
        if corners is None:
            raise ValueError(f"Sample {s['index']} no longer detects")
        rvec, _ = cv2.Rodrigues(B[:3,:3])
        projected, _ = cv2.projectPoints(board_points(25), rvec, B[:3,3], K, dist)
        rms = min(float(np.sqrt(np.mean(np.sum((projected-c)**2, axis=2))))
                  for c in (corners, corners[::-1]))
        image_checks.append(dict(index=s['index'], pnp_rms_px=rms, raw_positions=s['raw_positions']))
        duplicate = next((p['index'] for p in samples
                          if max(abs(np.array(s['raw_positions'])[:5]-np.array(p['raw_positions'])[:5])) < 6), None)
        if duplicate is not None:
            duplicates.append(dict(index=s['index'], repeated_pose_of=duplicate))
        else:
            samples.append(s)
    pairs, rotvecs = [], []
    for a,b in itertools.combinations(samples,2):
        A1,A2 = np.array(a['T_base_end_effector']), np.array(b['T_base_end_effector'])
        B1,B2 = np.array(a['T_camera_board']), np.array(b['T_camera_board'])
        Rrobot = A1[:3,:3].T@A2[:3,:3]
        r = rotation_angle(Rrobot)
        c = rotation_angle(B1[:3,:3].T@B2[:3,:3])
        pairs.append(dict(indices=[a['index'],b['index']], robot_deg=r, camera_deg=c, mismatch_deg=abs(r-c)))
        rotvecs.append(cv2.Rodrigues(Rrobot)[0].ravel())
    subsets = {'all_unique': samples,
               'without_initial_sample_01': [s for s in samples if s['index'] != 1],
               'gui_only': [s for s in samples if s.get('source') == 'six_sliders_fk_gui'],
               'latest_10_onward': [s for s in samples if s['index'] >= 10]}
    fits = {}
    for name, subset in subsets.items():
        if len(subset) < 5:
            continue
        group = dict(indices=[s['index'] for s in subset], methods={})
        for method in METHODS:
            try:
                group['methods'][method] = evaluate(subset,method)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                group['methods'][method] = dict(error=str(exc))
        fits[name] = group
    result = dict(created_at=datetime.now().isoformat(), session=str(folder),
                  original_count=len(all_samples), unique_count=len(samples), duplicates=duplicates,
                  image_checks=image_checks, relative_motion_pairs=pairs,
                  robot_rotation_vector_singular_values_rad=np.linalg.svd(rotvecs,compute_uv=False).tolist(),
                  median_rotation_mismatch_by_sample={str(s['index']): float(np.median([p['mismatch_deg'] for p in pairs if s['index'] in p['indices']])) for s in samples},
                  fits=fits,
                  note='Diagnostic candidates, not an automatically accepted calibration. Board scatter and leave-one-out checks are internal consistency, not independent physical accuracy.')
    write_json(folder/'hand_eye_quality_report.json', result)
    primary = fits['all_unique']['methods']['PARK']
    if 'error' not in primary:
        write_json(folder/'hand_eye_candidate.json', dict(
            status='provisional_requires_validation', approved_for_robot_motion=False,
            method='PARK', created_at=result['created_at'],
            T_end_effector_camera=primary['T_end_effector_camera'],
            transform_definition='p_end_effector = T_end_effector_camera @ p_camera',
            base_frame='base_link', end_effector_frame='gripper_frame_link',
            translation_unit='m', samples=fits['all_unique']['indices'],
            excluded_repeated_poses=duplicates,
            training=primary['training'], leave_one_out=primary['leave_one_out'],
            notes=['Diagnostic estimate only; review consistency before using for robot motion.',
                   'Original samples are unchanged. No initial sample was silently removed.',
                   'Internal validation is not an independent measurement of physical accuracy.']))
    print('Samples:',result['original_count'],'unique:',result['unique_count'],'duplicates:',duplicates)
    print('PnP RMS range:', min(x['pnp_rms_px'] for x in image_checks),max(x['pnp_rms_px'] for x in image_checks))
    print('Motion SVD:',result['robot_rotation_vector_singular_values_rad'])
    print('Median rotation mismatch:',result['median_rotation_mismatch_by_sample'])
    for name, group in fits.items():
        print('\n',name,group['indices'])
        for method, fit_result in group['methods'].items():
            if 'error' in fit_result:
                print(method,fit_result)
                continue
            print(method,'t(mm)',np.round(np.array(fit_result['camera_translation_m'])*1000,1).tolist(),
                  'train(pos RMS/max,rot RMS)',round(fit_result['training']['position_mm']['rms'],2),round(fit_result['training']['position_mm']['max'],2),round(fit_result['training']['orientation_deg']['rms'],2),
                  'LOO(pos RMS/max,rot RMS)',round(fit_result['leave_one_out']['position_mm']['rms'],2),round(fit_result['leave_one_out']['position_mm']['max'],2),round(fit_result['leave_one_out']['orientation_deg']['rms'],2),
                  'Xchange(mm,deg)',round(fit_result['leave_one_out_transform_change']['position_mm']['max'],2),round(fit_result['leave_one_out_transform_change']['orientation_deg']['max'],2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=HERE/'data'/'hand_eye_20261012_202343')
    args = parser.parse_args()
    audit(args.session.resolve())
