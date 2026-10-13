"""Direct joint control and synchronized, stationary hand-eye samples. No IK calls."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CONTROL_DIR = HERE / "robot_control"
FK_DIR = CONTROL_DIR / "FK"
MODEL_DIR = CONTROL_DIR / "IK_SCS215"
sys.path.insert(0, str(FK_DIR))
sys.path.insert(0, str(MODEL_DIR))

from arm_backend import ArmController, JOINTS, LeRobotSCSBus, find_port, target_position
from motion import DemoBus, Mailbox
from robot_adapter import Calibration, load_config
from calibrate_intrinsics import Camera, detect, board_points, write_image, write_json
from hand_eye_calibration import load_intrinsics, solve_pnp, new_session, PATTERN


def load_mapping(data):
    mapping = Calibration(load_config())
    for sid in JOINTS:
        joint = data["joints"][str(sid)]
        expected = (mapping.minimum[sid-1], mapping.maximum[sid-1], mapping.zero[sid-1])
        actual = (joint["zero_position"], joint["one_position"], joint["angle_zero_position"])
        if actual != expected:
            raise ValueError(f"ID{sid} 的滑条标定与 FK 零位不一致")
    return mapping


class CaptureController(ArmController):
    """Keep hardware limits intact; use the original bounded manual controller."""
    def _full_range(self, state):
        # The lec4 controller can rewrite EEPROM limits here. This GUI never does.
        return

    def set_target(self, sid, value, speed, now):
        target = target_position(value, self.joints[sid].calibration)
        low, high = self.joints[sid].limits
        if not low <= target <= high:
            raise ValueError(f"ID{sid} 目标超出舵机现有限位 [{low}, {high}]；未发送")
        return super().set_target(sid, value, speed, now)


class Stability:
    def __init__(self):
        self.history = deque()

    def add(self, stamp, snapshot):
        if any(not s["online"] or s["active"] or s["ratio"] is None for s in snapshot.values()):
            self.history.clear()
            return False
        raw = np.array([snapshot[sid]["raw"] for sid in JOINTS])
        self.history.append((stamp, raw))
        while len(self.history) > 1 and self.history[1][0] < stamp - 1.0:
            self.history.popleft()
        return bool(len(self.history) >= 4 and stamp-self.history[0][0] >= 1.0
                    and np.max(np.ptp([x[1] for x in self.history], axis=0)) <= 3)


def bracket_frame(controller, camera, cancelled, timeout=1.5):
    """Read joints, wait for a NEW received frame, then read joints again.

    USB receipt timestamps are not exposure timestamps. Stationary capture is
    required; this is deliberately not advertised as hardware synchronization.
    """
    if camera is None:
        raise ValueError("相机尚未启动")
    before = [int.from_bytes(controller.bus.read(sid, 56, 2), "big") for sid in JOINTS]
    start = time.monotonic()
    latest = None
    while time.monotonic() - start < timeout:
        if cancelled():
            raise ValueError("采集已取消")
        latest = camera.sample()
        if latest is not None and latest[0] > start:
            break
        time.sleep(.01)
    if latest is None or latest[0] <= start:
        raise ValueError("相机未返回新画面；请检查是否被其他应用占用")
    stamp, frame = latest
    after = [int.from_bytes(controller.bus.read(sid, 56, 2), "big") for sid in JOINTS]
    end = time.monotonic()
    if cancelled():
        raise ValueError("采集已取消")
    if np.max(np.abs(np.array(before)-after)) > 3:
        raise ValueError("拍照前后关节发生移动，请停稳后重拍")
    if end - stamp > .5 or end-start > .8:
        raise ValueError("图像或串口反馈延迟过大，请重拍")
    return dict(frame=frame.copy(), raw=after, raw_before=before,
                frame_age=end-stamp, bracket_seconds=end-start,
                captured_at=datetime.now().isoformat(timespec="milliseconds"))


class JointWorker(threading.Thread):
    def __init__(self, data, port, mailbox, output, camera, demo=False):
        super().__init__(daemon=True, name="FK-capture-serial")
        self.data, self.port, self.mailbox, self.output = data, port, mailbox, output
        self.camera, self.demo = camera, demo

    def emit(self, kind, value):
        self.output.put((kind, value))

    def run(self):
        bus = controller = None
        stable = Stability()
        try:
            port = "离线模拟" if self.demo else self.port or find_port()
            bus = DemoBus(self.data) if self.demo else LeRobotSCSBus(port)
            controller = CaptureController(bus, self.data, self.mailbox.cancelled)
            controller.initialize(time.monotonic())
            self.emit("connected", port)
            while not self.mailbox.closing.is_set():
                began = time.monotonic()
                if self.mailbox.emergency.is_set() or self.mailbox.holding.is_set():
                    commands, _, speed = self.mailbox.take()
                    if self.mailbox.emergency.is_set():
                        controller.off(JOINTS)
                    else:
                        controller.hold(JOINTS, speed)
                    self.mailbox.emergency.clear()
                    self.mailbox.holding.clear()
                    stable.history.clear()
                    self.emit("capture_cancelled", None)
                if self.demo:
                    bus.step(.05)
                controller.poll(time.monotonic())
                ready = stable.add(time.monotonic(), controller.snapshot())
                commands, targets, speed = self.mailbox.take()
                for name, payload in commands:
                    if name == "off_joint":
                        controller.off([payload])
                        stable.history.clear()
                        ready = False
                    elif name == "capture":
                        try:
                            if self.demo:
                                raise ValueError("模拟模式不会写入标定样本")
                            if targets or not ready or any(s.active for s in controller.joints.values()):
                                raise ValueError("请松开滑条，等六个关节停稳 1 秒再 Capture")
                            packet = bracket_frame(controller, self.camera, self.mailbox.cancelled)
                            old = stable.history[0][1]
                            if np.max(np.abs(old - packet["raw"])) > 3:
                                raise ValueError("取帧期间姿态变化，请停稳后重拍")
                            self.emit("captured", packet)
                        except (ValueError, OSError, RuntimeError) as exc:
                            self.emit("capture_error", str(exc))
                for sid, value in targets.items():
                    try:
                        controller.set_target(sid, value, speed, time.monotonic())
                        stable.history.clear()
                        ready = False
                    except (ValueError, OSError, RuntimeError) as exc:
                        if not isinstance(exc, ValueError):
                            controller.fault(sid, str(exc))
                        self.emit("notice", str(exc))
                controller.advance(speed, time.monotonic())
                self.emit("snapshot", (controller.snapshot(), ready))
                self.mailbox.closing.wait(max(0, .05-(time.monotonic()-began)))
            if self.mailbox.emergency.is_set():
                controller.off(JOINTS)
            else:
                controller.hold(JOINTS, self.mailbox.speed)
        except Exception as exc:
            if controller is not None:
                controller.off([sid for sid, state in controller.joints.items() if state.owned])
            self.emit("error", str(exc))
        finally:
            if bus is not None:
                bus.close()
            self.emit("closed", None)


def rotation_angle(R):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


class SampleStore:
    def __init__(self, folder, intrinsics, mapping):
        self.folder, self.mapping = Path(folder), mapping
        self.K, self.dist = load_intrinsics(intrinsics)
        meta = json.loads((self.folder / "session.json").read_text(encoding="utf-8-sig"))
        if (meta.get("camera_mount") != "eye_in_hand" or meta.get("board_fixed") is not True
                or meta["square_size_mm"] != 25 or tuple(meta["pattern_inner_corners"]) != PATTERN
                or tuple(meta["image_size"]) != (1280, 720) or meta["camera_name"] != "GENERAL WEBCAM"):
            raise ValueError("旧会话的安装方式、棋盘或相机规格不匹配")
        old_K, old_dist = load_intrinsics(meta["intrinsics_file"])
        if not np.array_equal(old_K, self.K) or not np.array_equal(old_dist, self.dist):
            raise ValueError("内参与旧会话不一致，不能混合采集")
        self.samples = [json.loads(p.read_text(encoding="utf-8"))
                        for p in sorted((self.folder / "samples").glob("sample_*.json"))]

    def save(self, packet):
        raw = np.array(packet["raw"], dtype=int)
        if any(np.max(np.abs(raw[:5]-np.array(s["raw_positions"])[:5])) < 6 for s in self.samples):
            raise ValueError("与已有姿态太接近；请调整机械臂姿态，夹爪开合不算新姿态")
        q = self.mapping.to_q(raw)
        T_BE = self.mapping.arm.fk(q)
        corners = detect(packet["frame"])
        if corners is None:
            raise ValueError("照片未检测到完整 48 个角点，请让棋盘完整入镜后重拍")
        T_CM = solve_pnp(corners, self.K, self.dist)
        # An even-by-even corner grid has a 180-degree ambiguity. Relative rotation
        # angles are invariant under the unknown hand-eye transform. Compare both
        # orders with recent samples, retaining historical data unchanged.
        flipped = False
        mismatch = None
        if self.samples:
            candidates = [T_CM, solve_pnp(corners[::-1].copy(), self.K, self.dist)]
            scores = []
            for candidate in candidates:
                errors = []
                for prior in self.samples[-3:]:
                    A = np.array(prior["T_base_end_effector"])
                    B = np.array(prior["T_camera_board"])
                    errors.append(abs(rotation_angle(A[:3,:3].T @ T_BE[:3,:3])
                                      - rotation_angle(B[:3,:3].T @ candidate[:3,:3])))
                scores.append(float(np.median(errors)))
            flipped = bool(np.argmin(scores))
            mismatch = min(scores)
            if mismatch > 4:
                raise ValueError(f"图像与 FK 的旋转变化不一致（{mismatch:.1f}°）；检查棋盘/相机固定和运动是否停稳")
            if flipped:
                corners = corners[::-1].copy()
                T_CM = candidates[1]
        rvec, _ = cv2.Rodrigues(T_CM[:3,:3])
        projected, _ = cv2.projectPoints(board_points(25), rvec, T_CM[:3,3], self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((projected-corners)**2, axis=2))))
        if not np.isfinite(rms) or rms > 2.0 or T_CM[2,3] <= 0:
            raise ValueError(f"棋盘位姿质量不足，重投影 RMS={rms:.2f}px，请重拍")
        index = max([s["index"] for s in self.samples], default=0) + 1
        while any((self.folder / sub / f"sample_{index:02d}{ext}").exists()
                  for sub, ext in (("samples", ".json"), ("images", ".png"), ("corners", ".jpg"))):
            index += 1
        name = f"sample_{index:02d}"
        sample = dict(index=index, captured_at=packet["captured_at"],
                      raw_positions=raw.tolist(), raw_positions_before=packet["raw_before"],
                      joint_angles_rad=q.tolist(), T_base_end_effector=T_BE.tolist(),
                      T_camera_board=T_CM.tolist(), camera_frame_age_s=packet["frame_age"],
                      capture_bracket_s=packet["bracket_seconds"], source="six_sliders_fk_gui",
                      pnp_rms_px=rms, corner_order_reversed=flipped,
                      relative_rotation_mismatch_deg=mismatch,
                      synchronization="stationary >=1s; raw/frame-receipt/raw; no exposure timestamp",
                      base_frame="base_link", end_effector_frame="gripper_frame_link",
                      translation_unit="m", camera_matrix=self.K.tolist(), dist_coeffs=self.dist.tolist())
        annotated = packet["frame"].copy()
        cv2.drawChessboardCorners(annotated, PATTERN, corners, True)
        cv2.drawFrameAxes(annotated, self.K, self.dist, rvec, T_CM[:3,3], .05)
        write_image(self.folder / "images" / (name + ".png"), packet["frame"])
        write_image(self.folder / "corners" / (name + ".jpg"), annotated)
        write_json(self.folder / "samples" / (name + ".json"), sample)
        self.samples.append(sample)
        return sample
