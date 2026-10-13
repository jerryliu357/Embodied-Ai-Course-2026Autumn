"""Combine read-only hand-eye sessions and write a separate diagnostic result."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_hand_eye import METHODS, evaluate
from fk_capture_core import FK_DIR, load_mapping, rotation_angle
from hand_eye_calibration import load_intrinsics, valid_R


HERE = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_sessions(folders: list[Path]):
    if len(folders) < 1:
        raise ValueError("至少需要一个会话目录")
    mapping = load_mapping(read_json(FK_DIR / "calibration.json"))
    samples, metadata = [], []
    reference_K = reference_dist = None
    for session_number, folder in enumerate(folders, 1):
        folder = folder.resolve()
        meta = read_json(folder / "session.json")
        K, dist = load_intrinsics(meta["intrinsics_file"])
        if reference_K is None:
            reference_K, reference_dist = K, dist
        elif not np.array_equal(reference_K, K) or not np.array_equal(reference_dist, dist):
            raise ValueError("各会话使用的相机内参不一致")
        paths = sorted((folder / "samples").glob("sample_*.json"))
        metadata.append({"session": str(folder), "sample_count": len(paths)})
        for path in paths:
            sample = read_json(path)
            A = np.asarray(sample["T_base_end_effector"], dtype=float)
            B = np.asarray(sample["T_camera_board"], dtype=float)
            valid_R(A[:3, :3]); valid_R(B[:3, :3])
            expected = mapping.arm.fk(mapping.to_q(sample["raw_positions"]))
            np.testing.assert_allclose(A, expected, atol=1e-10)
            sample["source_session"] = str(folder)
            sample["source_index"] = int(sample["index"])
            sample["index"] = f"S{session_number}-{int(sample['index']):02d}"
            samples.append(sample)
    return samples, metadata


def remove_duplicates(samples):
    unique, duplicates = [], []
    for sample in samples:
        raw = np.asarray(sample["raw_positions"][:5])
        prior = next((x for x in unique
                      if np.max(np.abs(raw - np.asarray(x["raw_positions"][:5]))) < 6), None)
        if prior is None:
            unique.append(sample)
        else:
            duplicates.append({"sample": sample["index"], "repeated_pose_of": prior["index"]})
    return unique, duplicates


def motion_quality(samples):
    vectors, pair_rows = [], []
    for i, first in enumerate(samples):
        for second in samples[i + 1:]:
            A1 = np.asarray(first["T_base_end_effector"])
            A2 = np.asarray(second["T_base_end_effector"])
            B1 = np.asarray(first["T_camera_board"])
            B2 = np.asarray(second["T_camera_board"])
            Ra = A1[:3, :3].T @ A2[:3, :3]
            Rb = B1[:3, :3].T @ B2[:3, :3]
            vectors.append(cv2.Rodrigues(Ra)[0].reshape(3))
            pair_rows.append({
                "samples": [first["index"], second["index"]],
                "robot_rotation_deg": rotation_angle(Ra),
                "camera_rotation_deg": rotation_angle(Rb),
                "rotation_mismatch_deg": abs(rotation_angle(Ra) - rotation_angle(Rb)),
            })
    singular_values = np.linalg.svd(np.asarray(vectors), compute_uv=False).tolist()
    return singular_values, pair_rows


def run(folders: list[Path], output: Path):
    samples, sessions = load_sessions(folders)
    unique, duplicates = remove_duplicates(samples)
    if len(unique) < 12:
        raise ValueError(f"合并后至少需要12个唯一姿态，目前只有{len(unique)}个")
    singular_values, pairs = motion_quality(unique)
    fits = {}
    for name in METHODS:
        try:
            fits[name] = evaluate(unique, name)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            fits[name] = {"error": str(exc)}
    valid = {name: value for name, value in fits.items() if "error" not in value}
    if not valid:
        raise RuntimeError("四种手眼标定方法均失败")
    # Prefer the method whose held-out samples best predict the fixed board.
    chosen_name, chosen = min(valid.items(), key=lambda item: (
        item[1]["leave_one_out"]["position_mm"]["rms"],
        item[1]["leave_one_out"]["orientation_deg"]["rms"],
    ))
    board = np.asarray(chosen["T_base_board_mean"], dtype=float)
    camera = np.asarray(chosen["T_end_effector_camera"], dtype=float)
    result = {
        "created_at": datetime.now().isoformat(),
        "status": "provisional_requires_physical_validation",
        "source_sessions": sessions,
        "original_sample_count": len(samples),
        "unique_sample_count": len(unique),
        "duplicates": duplicates,
        "sample_ids": [sample["index"] for sample in unique],
        "pnp_rms_px": {sample["index"]: sample.get("pnp_rms_px") for sample in samples},
        "robot_rotation_vector_singular_values_rad": singular_values,
        "relative_motion_pairs": pairs,
        "methods": fits,
        "selected_method": chosen_name,
        "T_end_effector_camera": chosen["T_end_effector_camera"],
        "camera_origin_in_end_effector_cm": (camera[:3, 3] * 100).tolist(),
        "camera_to_end_effector_origin_distance_cm": float(np.linalg.norm(camera[:3, 3]) * 100),
        "T_base_board": chosen["T_base_board_mean"],
        "board_origin_in_base_cm": (board[:3, 3] * 100).tolist(),
        "base_to_board_origin_distance_cm": float(np.linalg.norm(board[:3, 3]) * 100),
        "training_consistency": chosen["training"],
        "leave_one_out": chosen["leave_one_out"],
        "approved_for_robot_motion": False,
        "notes": [
            "Original session folders and samples were read only.",
            "The selected method minimizes leave-one-out board-position RMS.",
            "The board coordinates require an independent physical measurement before approval.",
        ],
    }
    write_json(output / "combined_hand_eye_report.json", result)
    write_json(output / "hand_eye_candidate.json", {
        "status": result["status"],
        "approved_for_robot_motion": False,
        "method": chosen_name,
        "T_end_effector_camera": chosen["T_end_effector_camera"],
        "transform_definition": "p_end_effector = T_end_effector_camera @ p_camera",
        "translation_unit": "m",
        "source_sessions": sessions,
        "samples": result["sample_ids"],
    })
    print(f"samples={len(samples)} unique={len(unique)} duplicates={duplicates}")
    print(f"motion_singular_values_rad={np.round(singular_values, 5).tolist()}")
    for name, fit in fits.items():
        if "error" in fit:
            print(name, "ERROR", fit["error"])
        else:
            print(name,
                  "camera_cm", np.round(np.asarray(fit["camera_translation_m"]) * 100, 3).tolist(),
                  "train_rms_mm", round(fit["training"]["position_mm"]["rms"], 2),
                  "loo_rms_mm", round(fit["leave_one_out"]["position_mm"]["rms"], 2))
    print("selected_method=", chosen_name)
    print("board_origin_base_cm=", np.round(board[:3, 3] * 100, 3).tolist())
    print("base_to_board_origin_cm=", round(np.linalg.norm(board[:3, 3]) * 100, 3))
    print("camera_origin_end_effector_cm=", np.round(camera[:3, 3] * 100, 3).tolist())
    print("camera_to_end_effector_origin_cm=", round(np.linalg.norm(camera[:3, 3]) * 100, 3))
    print("report=", output / "combined_hand_eye_report.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.sessions, args.output.resolve())
