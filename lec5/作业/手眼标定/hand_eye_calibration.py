"""采集并计算眼在手上的手眼标定数据。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IK_DIR = (HERE / "robot_control" / "IK_SCS215").resolve()
sys.path.insert(0, str(IK_DIR))
from calibrate_intrinsics import Camera, board_points, detect, write_image, write_json

CAMERA = "GENERAL WEBCAM"
IMAGE_SIZE = (1280, 720)
PATTERN = (6, 8)
SQUARE_MM = 25.0


def make_T(R, t):
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.asarray(R, dtype=float)
    out[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return out


def valid_R(R):
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("旋转矩阵无效")
    if abs(np.linalg.det(R) - 1) > 1e-3 or np.linalg.norm(R.T @ R - np.eye(3)) > 1e-3:
        raise ValueError("旋转矩阵不是合法旋转")


def load_intrinsics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if tuple(data.get("image_size", ())) != IMAGE_SIZE or tuple(data.get("pattern_inner_corners", ())) != PATTERN:
        raise ValueError("内参文件的分辨率或棋盘格规格不匹配")
    K = np.asarray(data["camera_matrix"], dtype=float)
    dist = np.asarray(data["dist_coeffs"], dtype=float).reshape(-1, 1)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError("内参文件有无效数值")
    return K, dist


def solve_pnp(corners, K, dist):
    if corners is None or len(corners) != 48:
        raise ValueError("没有检测到完整48个棋盘格角点")
    ok, rvec, tvec = cv2.solvePnP(board_points(SQUARE_MM), corners, K, dist)
    if not ok:
        raise ValueError("solvePnP失败")
    R, _ = cv2.Rodrigues(rvec)
    valid_R(R)
    return make_T(R, tvec)


def robot_pose(robot):
    raw = robot.read_positions()
    q = robot.mapping.to_q(raw)
    T_BE = np.asarray(robot.mapping.arm.fk(q), dtype=float)
    valid_R(T_BE[:3, :3])
    return raw, q, T_BE


def new_session(intrinsics):
    folder = HERE / "data" / ("hand_eye_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for name in ("images", "corners", "samples"):
        (folder / name).mkdir(parents=True)
    write_json(folder / "session.json", {
        "created_at": datetime.now().isoformat(), "camera_name": CAMERA,
        "image_size": list(IMAGE_SIZE), "pattern_inner_corners": list(PATTERN),
        "square_size_mm": SQUARE_MM, "intrinsics_file": str(Path(intrinsics).resolve()),
        "camera_mount": "eye_in_hand", "board_fixed": True,
    })
    return folder


def capture(robot, camera, K, dist, folder, index):
    raw, q, T_BE = robot_pose(robot)
    latest = camera.sample()
    deadline = time.monotonic() + 2
    while (latest is None or time.monotonic() - latest[0] > .5) and time.monotonic() < deadline:
        time.sleep(.02)
        latest = camera.sample()
    if latest is None or time.monotonic() - latest[0] > .5:
        raise RuntimeError("相机没有新鲜画面")
    stamp, frame = latest
    corners = detect(frame)
    T_CM = solve_pnp(corners, K, dist)
    annotated = frame.copy()
    cv2.drawChessboardCorners(annotated, PATTERN, corners, True)
    name = f"sample_{index:02d}"
    write_image(folder / "images" / (name + ".png"), frame)
    write_image(folder / "corners" / (name + ".jpg"), annotated)
    sample = {
        "index": index, "captured_at": datetime.now().isoformat(),
        "camera_frame_age_s": float(time.monotonic() - stamp),
        "raw_positions": raw.astype(int).tolist(), "joint_angles_rad": q.tolist(),
        "T_base_end_effector": T_BE.tolist(), "T_camera_board": T_CM.tolist(),
    }
    write_json(folder / "samples" / (name + ".json"), sample)
    print(f"已采集第{index}组；末端位置(mm)={np.round(T_BE[:3,3]*1000,1).tolist()}；棋盘格距离(cm)={np.linalg.norm(T_CM[:3,3])*100:.1f}")


def solve_hand_eye(folder, intrinsics, method_name="TSAI"):
    load_intrinsics(intrinsics)
    paths = sorted((Path(folder) / "samples").glob("sample_*.json"))
    if len(paths) < 12:
        raise ValueError(f"至少需要12组样本，目前只有{len(paths)}组")
    R_g2b, t_g2b, R_t2c, t_t2c, pairs = [], [], [], [], []
    for path in paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        T_BE = np.asarray(sample["T_base_end_effector"], dtype=float)
        T_CM = np.asarray(sample["T_camera_board"], dtype=float)
        valid_R(T_BE[:3,:3]); valid_R(T_CM[:3,:3])
        R_g2b.append(T_BE[:3,:3]); t_g2b.append(T_BE[:3,3].reshape(3,1))
        R_t2c.append(T_CM[:3,:3]); t_t2c.append(T_CM[:3,3].reshape(3,1))
        pairs.append((T_BE, T_CM))
    methods = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK,
               "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
    R_EC, t_EC = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=methods[method_name])
    valid_R(R_EC)
    T_EC = make_T(R_EC, t_EC)
    predicted = np.array([(A @ T_EC @ C)[:3,3] for A,C in pairs])
    mean = predicted.mean(axis=0)
    spread = np.linalg.norm(predicted - mean, axis=1)
    result = {
        "calibrated_at": datetime.now().isoformat(), "method": method_name,
        "sample_count": len(paths), "intrinsics_file": str(Path(intrinsics).resolve()),
        "T_end_effector_camera": T_EC.tolist(),
        "camera_in_end_effector_translation_m": T_EC[:3,3].tolist(),
        "predicted_board_in_base_mean_m": mean.tolist(),
        "predicted_board_position_spread_mm": {"mean": float(spread.mean()*1000), "max": float(spread.max()*1000)},
        "samples": [p.name for p in paths],
        "notes": ["T_end_effector_camera 将相机坐标点转换到末端坐标。", "固定棋盘格的预测位置散布用于检查坐标方向和样本一致性。", "散布小不等于实物绝对坐标已验证。"],
    }
    write_json(Path(folder) / "hand_eye_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--port")
    p.add_argument("--session", type=Path, help="继续已有手眼采集目录")
    p.add_argument("--solve", type=Path, help="只计算已有采集目录，不连接机器人")
    p.add_argument("--method", choices=["TSAI", "PARK", "HORAUD", "DANIILIDIS"], default="TSAI")
    p.add_argument("--check", action="store_true", help="只检查内参和依赖，不连接硬件")
    a = p.parse_args()
    if a.check:
        K, _ = load_intrinsics(a.intrinsics.resolve())
        print(f"内参检查通过，K={K.tolist()}，FK目录={IK_DIR}")
        return
    if a.solve:
        solve_hand_eye(a.solve.resolve(), a.intrinsics.resolve(), a.method)
        return
    from robot_adapter import Calibration, Robot, load_config
    from trajectory import Controller
    cfg = load_config(IK_DIR / "config.json")
    if a.port:
        cfg["port"] = a.port
    robot = Robot(Calibration(cfg))
    controller = Controller(robot, realtime=True)
    folder = a.session.resolve() if a.session else new_session(a.intrinsics.resolve())
    camera = None
    try:
        camera = Camera(folder)
        end = time.monotonic() + 12
        while camera.sample() is None and time.monotonic() < end:
            time.sleep(.1)
        if camera.sample() is None:
            raise RuntimeError("相机没有画面，请关闭其他相机应用")
        K, dist = load_intrinsics(a.intrinsics.resolve())
        index = len(list((folder / "samples").glob("sample_*.json"))) + 1
        print("已连接。命令：capture、move up 2、move left 2、status、solve、quit")
        print("第一次move才会使能舵机；先确认路径安全。棋盘格保持固定。")
        while True:
            line = input("hand-eye> ").strip().lower()
            if line in ("quit", "exit"):
                break
            if line == "capture":
                try:
                    capture(robot, camera, K, dist, folder, index); index += 1
                except (ValueError, RuntimeError, OSError) as exc:
                    print(f"未采集：{exc}")
            elif line == "status":
                raw, _, pose = robot_pose(robot)
                print("raw ID1～6:", raw.tolist(), "末端位置mm:", np.round(pose[:3,3]*1000,1).tolist())
            elif line == "solve":
                solve_hand_eye(folder, a.intrinsics.resolve(), a.method)
            elif line.startswith("move "):
                bits = line.split(); directions = {"forward":(1,0,0),"back":(-1,0,0),"left":(0,1,0),"right":(0,-1,0),"up":(0,0,1),"down":(0,0,-1)}
                if len(bits) != 3 or bits[1] not in directions:
                    print("示例：move up 2，方向为up/down/forward/back/left/right，单位cm"); continue
                amount = float(bits[2])
                if not np.isfinite(amount) or not 0 < amount <= 3:
                    print("每次移动请输入0～3 cm"); continue
                report = controller.move(np.asarray(directions[bits[1]]) * amount / 100)
                print("运动结束：", report.get("reason"), "成功=", report.get("success_model_estimate")); time.sleep(.5)
            else:
                print("未知命令：capture、move up 2、status、solve、quit")
    finally:
        if camera is not None: camera.close()
        robot.close()
        print("相机和串口已释放；未自动卸力。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"错误：{exc}"); raise SystemExit(1)
