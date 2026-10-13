"""One-shot physical validation of an existing eye-in-hand transform.

This program never sends a motion or torque command. It reads a stationary
joint pose, captures one fresh checkerboard image, and evaluates
T_base_board = T_base_end @ T_end_camera @ T_camera_board.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from fk_capture_core import (HERE, FK_DIR, Camera, LeRobotSCSBus, ArmController,
                             JOINTS, find_port, load_mapping, bracket_frame)
from calibrate_intrinsics import detect, write_image, write_json
from hand_eye_calibration import load_intrinsics, solve_pnp, valid_R, PATTERN


def load_inputs(calibration_session: Path):
    meta = json.loads((calibration_session / "session.json").read_text(encoding="utf-8-sig"))
    candidate = json.loads((calibration_session / "hand_eye_candidate.json").read_text(encoding="utf-8-sig"))
    K, dist = load_intrinsics(meta["intrinsics_file"])
    X = np.asarray(candidate["T_end_effector_camera"], dtype=float)
    if X.shape != (4, 4) or not np.isfinite(X).all():
        raise ValueError("手眼矩阵格式无效")
    valid_R(X[:3, :3])
    data = json.loads((FK_DIR / "calibration.json").read_text(encoding="utf-8-sig"))
    return data, load_mapping(data), K, dist, X, meta["intrinsics_file"]


def stable_for_one_second(controller):
    history = []
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        controller.poll(now)
        snapshot = controller.snapshot()
        if not all(s["online"] and s["ratio"] is not None for s in snapshot.values()):
            raise RuntimeError("六个舵机没有全部在线或当前位置超出标定范围")
        history.append((now, np.array([snapshot[sid]["raw"] for sid in JOINTS], dtype=int)))
        history = [item for item in history if item[0] >= now - 1.0]
        if len(history) >= 6 and now - history[0][0] >= .95:
            motion = np.ptp([item[1] for item in history], axis=0)
            if int(motion.max()) <= 3:
                return history[-1][1]
        time.sleep(.1)
    raise RuntimeError("关节在 3 秒内没有稳定，请保持机器人静止后重试")


def capture_once(calibration_session: Path, output_root: Path, port=None):
    data, mapping, K, dist, T_EC, intrinsics = load_inputs(calibration_session)
    folder = output_root / ("board_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    camera = bus = None
    try:
        camera = Camera(folder)
        deadline = time.monotonic() + 10
        while camera.sample() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if camera.sample() is None:
            raise RuntimeError("相机没有画面，请关闭其他相机程序后重试")
        selected_port = port or find_port()
        bus = LeRobotSCSBus(selected_port)
        controller = ArmController(bus, data, lambda: False)
        controller.initialize(time.monotonic())
        stable_for_one_second(controller)
        packet = bracket_frame(controller, camera, lambda: False)
    finally:
        if bus is not None:
            bus.close()
        if camera is not None:
            camera.close()

    raw = np.asarray(packet["raw"], dtype=int)
    q = mapping.to_q(raw)
    T_BE = mapping.arm.fk(q)
    corners = detect(packet["frame"])
    if corners is None:
        raise RuntimeError("没有检测到完整的 6×8 内角点，本次没有生成验证结果")
    T_CM = solve_pnp(corners, K, dist)
    T_BM = T_BE @ T_EC @ T_CM
    valid_R(T_BM[:3, :3])

    # A physically accessible reference: the first/base rotation axis centre.
    # Its position in base_link is fixed by the URDF shoulder_pan joint origin.
    shoulder_axis_in_base_m = np.array([0.0388353, -8.97657e-09, 0.0624])
    origin_base_cm = T_BM[:3, 3] * 100
    origin_from_shoulder_axis_cm = (T_BM[:3, 3] - shoulder_axis_in_base_m) * 100

    rvec, _ = cv2.Rodrigues(T_CM[:3, :3])
    annotated = packet["frame"].copy()
    cv2.drawChessboardCorners(annotated, PATTERN, corners, True)
    cv2.drawFrameAxes(annotated, K, dist, rvec, T_CM[:3, 3], .075, 3)
    point = tuple(np.rint(corners[0, 0]).astype(int))
    cv2.circle(annotated, point, 13, (255, 0, 255), 4)
    cv2.putText(annotated, "M origin: measure this inner corner", (max(10, point[0]-250), max(35, point[1]-22)),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "Red +Xm   Green +Ym   Blue +Zm", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)

    result = {
        "created_at": datetime.now().isoformat(),
        "purpose": "physical validation only; not added to hand-eye calibration samples",
        "read_only_hardware": True,
        "calibration_session": str(calibration_session.resolve()),
        "intrinsics_file": str(Path(intrinsics).resolve()),
        "hand_eye_file": str((calibration_session / "hand_eye_candidate.json").resolve()),
        "raw_positions": raw.tolist(),
        "joint_angles_rad": q.tolist(),
        "T_base_end_effector": T_BE.tolist(),
        "T_end_effector_camera": T_EC.tolist(),
        "T_camera_board": T_CM.tolist(),
        "T_base_board": T_BM.tolist(),
        "estimated_board_origin_in_base_cm": origin_base_cm.tolist(),
        "shoulder_pan_axis_in_base_cm": (shoulder_axis_in_base_m * 100).tolist(),
        "estimated_board_origin_from_shoulder_pan_axis_cm_parallel_to_base_axes": origin_from_shoulder_axis_cm.tolist(),
        "measurement_axes": {"x": "base +X", "y": "base +Y", "z": "base +Z (up)"},
        "checkerboard_origin": "magenta-circled first inner corner in annotated.jpg",
        "camera_frame_age_s": packet["frame_age"],
        "capture_bracket_s": packet["bracket_seconds"],
        "warning": "One observation estimates current board position; it does not independently prove hand-eye accuracy.",
    }
    write_image(folder / "image.png", packet["frame"])
    write_image(folder / "annotated.jpg", annotated)
    write_json(folder / "result.json", result)
    write_json(output_root / "latest_board_validation.json", result | {"folder": str(folder.resolve())})
    print(json.dumps({"folder": str(folder.resolve()),
                      "raw_positions": raw.tolist(),
                      "estimated_board_origin_in_base_cm": np.round(origin_base_cm, 3).tolist(),
                      "estimated_from_shoulder_axis_cm": np.round(origin_from_shoulder_axis_cm, 3).tolist()},
                     ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-session", type=Path,
                        default=HERE / "data" / "hand_eye_20261012_202343")
    parser.add_argument("--output-root", type=Path, default=HERE / "data")
    parser.add_argument("--port")
    parser.add_argument("--check", action="store_true", help="只检查文件和矩阵，不打开硬件")
    args = parser.parse_args()
    if args.check:
        _, mapping, K, dist, X, _ = load_inputs(args.calibration_session.resolve())
        print(f"OK: joints={mapping.arm.names}, K={K.shape}, dist={dist.size}, T_EC={X.shape}")
        return
    capture_once(args.calibration_session.resolve(), args.output_root.resolve(), args.port)


if __name__ == "__main__":
    main()
