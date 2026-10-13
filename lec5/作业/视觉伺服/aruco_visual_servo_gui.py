"""Week 6 simple ArUco visual servo with observe, single-step and automatic modes."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from aruco_visual_servo_core import (ArucoTracker, MARKER_ID, MARKER_SIZE_M,
    camera_delta_to_base, choose_step, draw_overlay)
from calibrate_intrinsics import Camera
from hand_eye_calibration import load_intrinsics

import sys
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "robot_control" / "IK_SCS215"
sys.path.insert(0, str(MODEL_DIR))
from robot_adapter import Calibration, Robot, load_config
from trajectory import Controller, MotionStopped


DEFAULT_INTRINSICS = HERE / "intrinsics.json"
DEFAULT_HAND_EYE = HERE / "hand_eye_candidate.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class MotionWorker(threading.Thread):
    def __init__(self, mapping, T_EC, commands, output, closing, cancel, port=None):
        super().__init__(daemon=True, name="aruco-servo-motion")
        self.mapping, self.T_EC = mapping, np.asarray(T_EC, dtype=float)
        self.commands, self.output = commands, output
        self.closing, self.cancel, self.port = closing, cancel, port

    def emit(self, kind, value):
        self.output.put((kind, value))

    def run(self):
        robot = None
        try:
            cfg = load_config(MODEL_DIR / "config.json")
            if self.port:
                self.mapping.cfg["port"] = self.port
            robot = Robot(self.mapping)
            controller = Controller(robot, realtime=True)
            self.emit("connected", self.port or self.mapping.cfg.get("port") or "自动识别串口")
            while not self.closing.is_set():
                try:
                    kind, value = self.commands.get(timeout=.1)
                except queue.Empty:
                    continue
                if kind == "close":
                    break
                if kind == "hold":
                    try:
                        robot.hold()
                        self.emit("held", None)
                    except Exception as exc:
                        self.emit("motion_error", f"保持失败：{exc}")
                    continue
                if kind == "off":
                    try:
                        robot.off()
                        self.emit("off", None)
                    except Exception as exc:
                        self.emit("motion_error", f"卸力失败：{exc}")
                    continue
                if kind != "move":
                    continue
                self.cancel.clear()
                camera_delta = np.asarray(value, dtype=float)
                try:
                    raw = robot.read_positions()
                    T_BE = self.mapping.arm.fk(self.mapping.to_q(raw))
                    base_delta = camera_delta_to_base(T_BE, self.T_EC, camera_delta)
                    self.emit("move_started", {
                        "camera_delta_mm": (camera_delta * 1000).tolist(),
                        "base_delta_mm": (base_delta * 1000).tolist(),
                        "start_raw": raw.tolist(),
                    })
                    report = controller.move(
                        base_delta, cancel=self.cancel.is_set,
                        emit=lambda message: self.emit("motion_log", message),
                    )
                    self.emit("move_finished", report)
                except MotionStopped as exc:
                    self.emit("move_stopped", str(exc))
                except Exception as exc:
                    self.emit("motion_error", str(exc))
        except Exception as exc:
            self.emit("robot_error", str(exc))
        finally:
            if robot is not None:
                robot.close()
            self.emit("worker_closed", None)


class App:
    def __init__(self, root, intrinsics, hand_eye, port=None, observe_only=False,
                 no_camera=False, smoke_seconds=None):
        self.root = root
        self.K, self.dist = load_intrinsics(intrinsics)
        hand_eye_data = read_json(hand_eye)
        self.T_EC = np.asarray(hand_eye_data["T_end_effector_camera"], dtype=float)
        if self.T_EC.shape != (4, 4) or not np.isfinite(self.T_EC).all():
            raise ValueError("手眼矩阵无效")
        self.hand_eye_status = hand_eye_data.get("status", "unknown")
        self.tracker = ArucoTracker(self.K, self.dist, MARKER_ID, MARKER_SIZE_M)
        self.session = HERE / "data" / ("visual_servo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.session.mkdir(parents=True)
        self.event_file = self.session / "events.jsonl"
        self.observe_only, self.no_camera = observe_only, no_camera
        self.camera = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aruco")
        self.detect_future = None
        self.detect_frame = None
        self.observation = None
        self.observation_stamp = 0.
        self.history = deque(maxlen=8)
        self.last_submit = self.last_frame_stamp = self.last_auto_request = 0.
        self.last_state = 0.
        self.preview_size = None
        self.busy = self.closing_ui = self.robot_connected = False
        self.auto_enabled = False
        self.auto_steps = 0
        self.commands, self.output = queue.Queue(), queue.Queue()
        self.closing_event, self.cancel_event = threading.Event(), threading.Event()
        self.worker = None

        cfg = load_config(MODEL_DIR / "config.json")
        self.mapping = Calibration(cfg)

        root.title("Week 6 · ArUco 视觉伺服")
        root.geometry("1180x820")
        root.minsize(900, 680)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=7)
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="ArUco ID 0 · 简单视觉伺服",
                  font=("Microsoft YaHei UI", 19, "bold")).pack(anchor="w")
        ttk.Label(outer, text="DICT_4X4_50 · 黑色方块 5 cm · 画面按 40% / 20% / 40% 分区 · 每步 1 cm").pack(anchor="w", pady=(3, 8))

        self.preview = tk.Canvas(outer, width=960, height=540, bg="#162232", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview_image = self.preview.create_image(480, 270, anchor="center")
        self.preview_message = self.preview.create_text(480, 270, text="正在打开相机……", fill="white",
                                                        font=("Microsoft YaHei UI", 12))

        self.vision_text = tk.StringVar(value="等待 ArUco ID 0")
        self.decision_text = tk.StringVar(value="当前动作：等待")
        self.robot_text = tk.StringVar(value="仅观察模式" if observe_only else "机器人正在只读连接")
        ttk.Label(outer, textvariable=self.vision_text, foreground="#166b5d").pack(anchor="w", pady=(8, 2))
        ttk.Label(outer, textvariable=self.decision_text).pack(anchor="w")
        ttk.Label(outer, textvariable=self.robot_text).pack(anchor="w", pady=(2, 6))

        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.verified = tk.BooleanVar(value=False)
        self.verify_box = ttk.Checkbutton(
            controls, variable=self.verified,
            text="已实测确认手眼标定各轴误差 < 5 cm",
            command=self.refresh_buttons,
        )
        self.verify_box.pack(side="left")
        ttk.Label(controls, text="目标距离(cm)").pack(side="left", padx=(20, 4))
        self.target_depth = tk.StringVar(value="20")
        ttk.Spinbox(controls, from_=15, to=60, increment=1, width=6,
                    textvariable=self.target_depth).pack(side="left")
        ttk.Label(controls, text="自动最多步数").pack(side="left", padx=(16, 4))
        self.max_steps = tk.StringVar(value="20")
        ttk.Spinbox(controls, from_=1, to=50, width=5, textvariable=self.max_steps).pack(side="left")

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=8)
        self.step_button = ttk.Button(buttons, text="单步移动 1 cm", command=self.request_step, state="disabled")
        self.step_button.pack(side="left")
        self.auto_button = ttk.Button(buttons, text="开始连续伺服", command=self.toggle_auto, state="disabled")
        self.auto_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="停止并保持 · Esc", command=self.stop_and_hold).pack(side="left", padx=8)
        ttk.Button(buttons, text="打开日志目录", command=lambda: os.startfile(str(self.session))).pack(side="right")

        self.notice = tk.StringVar(value="默认只观察。实体误差验证完成并勾选后，运动按钮才会启用。")
        ttk.Label(outer, textvariable=self.notice, wraplength=1120, foreground="#5b6470").pack(anchor="w")
        ttk.Label(outer, text=f"手眼结果：{Path(hand_eye).name} · 状态 {self.hand_eye_status}",
                  foreground="#777777").pack(anchor="w", pady=(4, 0))

        root.bind("<Escape>", lambda _event: self.stop_and_hold())
        root.protocol("WM_DELETE_WINDOW", self.close)
        if not no_camera:
            self.camera = Camera(self.session)
        if not observe_only:
            self.worker = MotionWorker(self.mapping, self.T_EC, self.commands, self.output,
                                       self.closing_event, self.cancel_event, port)
            self.worker.start()
        if smoke_seconds:
            root.after(int(smoke_seconds * 1000), self.close)
        root.after(40, self.tick)

    def write_event(self, kind, value):
        row = {"time": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, "value": value}
        with self.event_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=lambda x: np.asarray(x).tolist()) + "\n")

    def dump_state(self, closed=False):
        observation = self.observation
        value = {
            "pid": os.getpid(), "closed": closed, "updated_at": time.time(),
            "observe_only": self.observe_only, "camera_frames": self.camera.frames if self.camera else 0,
            "aruco_found": observation is not None,
            "marker_id": observation.marker_id if observation is not None else None,
            "center_px": observation.center_px.tolist() if observation is not None else None,
            "depth_cm": observation.depth_m * 100 if observation is not None else None,
            "pnp_rms_px": observation.reprojection_rms_px if observation is not None else None,
            "stable": self.detection_stable(), "decision": self.current_decision().action,
            "robot_connected": self.robot_connected, "motion_unlocked": self.verified.get(),
            "busy": self.busy, "auto_enabled": self.auto_enabled,
        }
        (self.session / "visual_servo_state.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def target_depth_m(self):
        value = float(self.target_depth.get()) / 100
        if not .15 <= value <= .60:
            raise ValueError("目标距离必须是15～60 cm")
        return value

    def detection_stable(self):
        if len(self.history) < 4 or time.monotonic() - self.observation_stamp > .5:
            return False
        recent = np.asarray([[*item.center_px, item.depth_m] for item in list(self.history)[-4:]])
        return bool(np.max(np.ptp(recent[:, :2], axis=0)) <= 6 and np.ptp(recent[:, 2]) <= .012)

    def current_decision(self):
        try:
            return choose_step(self.observation, (1280, 720), target_depth_m=self.target_depth_m())
        except ValueError as exc:
            self.notice.set(str(exc))
            return choose_step(None, (1280, 720))

    def request_step(self):
        if self.busy or not self.robot_connected or not self.verified.get():
            return
        if not self.detection_stable():
            self.notice.set("ArUco 尚未稳定；保持卡片和机械臂静止后再试。")
            return
        decision = self.current_decision()
        if decision.done or not np.linalg.norm(decision.camera_delta_m):
            self.auto_enabled = False
            self.notice.set(decision.explanation)
            return
        if self.observation.reprojection_rms_px > 3:
            self.notice.set(f"ArUco 位姿重投影误差 {self.observation.reprojection_rms_px:.2f}px，未运动。")
            return
        self.busy = True
        self.commands.put(("move", decision.camera_delta_m.copy()))
        self.notice.set("正在按当前图像决策执行单步；运动结束后重新观察。")
        self.write_event("step_requested", {"action": decision.action,
                                             "camera_delta_m": decision.camera_delta_m.tolist()})
        self.refresh_buttons()

    def toggle_auto(self):
        if self.auto_enabled:
            self.stop_and_hold()
            return
        if not self.verified.get() or not self.robot_connected:
            return
        self.auto_enabled = True
        self.auto_steps = 0
        self.auto_button.configure(text="停止连续伺服")
        self.notice.set("连续模式已启动：每次只走1 cm，随后重新检测；Esc 随时停止并保持。")
        self.write_event("auto_started", None)

    def stop_and_hold(self):
        self.auto_enabled = False
        self.cancel_event.set()
        if self.worker is not None:
            self.commands.put(("hold", None))
        self.auto_button.configure(text="开始连续伺服")
        self.notice.set("已请求停止；机器人将保持当前位置。")
        self.write_event("stop_requested", None)

    def refresh_buttons(self):
        enabled = (not self.observe_only and self.robot_connected and self.verified.get()
                   and self.detection_stable() and not self.busy and not self.closing_ui)
        state = "normal" if enabled else "disabled"
        self.step_button.configure(state=state)
        self.auto_button.configure(state=state if not self.auto_enabled else "normal")

    def pump_worker(self):
        try:
            while True:
                kind, value = self.output.get_nowait()
                self.write_event(kind, value)
                if kind == "connected":
                    self.robot_connected = True
                    self.robot_text.set(f"机器人已连接（当前只读）：{value}")
                elif kind == "move_started":
                    self.robot_text.set(f"运动中 · base 位移 {np.round(value['base_delta_mm'], 1).tolist()} mm")
                elif kind == "motion_log":
                    self.robot_text.set(value)
                elif kind == "move_finished":
                    self.busy = False
                    self.auto_steps += 1
                    self.robot_text.set(f"上一步结束：{value.get('reason')} · 端点误差 {value.get('endpoint_error_mm', 0):.1f} mm")
                    self.notice.set("运动结束，正在等待新的稳定图像。")
                    self.history.clear()
                elif kind in ("move_stopped", "held"):
                    self.busy = False
                    self.robot_text.set("已停止并保持")
                elif kind in ("motion_error", "robot_error"):
                    self.busy = False
                    self.auto_enabled = False
                    self.robot_text.set(f"机器人错误：{value}")
                    self.notice.set(value)
                elif kind == "off":
                    self.busy = False
                    self.robot_text.set("六关节已卸力")
                elif kind == "worker_closed":
                    self.robot_connected = False
        except queue.Empty:
            pass

    def update_detection(self):
        if self.detect_future is not None and self.detect_future.done():
            try:
                self.observation = self.detect_future.result()
                self.observation_stamp = time.monotonic()
                if self.observation is None:
                    self.history.clear()
                else:
                    self.history.append(self.observation)
            except Exception as exc:
                self.observation = None
                self.history.clear()
                self.notice.set(f"ArUco 检测错误：{exc}")
            self.detect_future = None
        latest = self.camera.sample() if self.camera else None
        if latest is None or time.monotonic() - latest[0] > .5:
            self.vision_text.set("相机暂无新画面")
            return
        stamp, frame = latest
        if self.detect_future is None and time.monotonic() - self.last_submit >= .08:
            self.last_submit = time.monotonic()
            self.detect_frame = frame.copy()
            self.detect_future = self.pool.submit(self.tracker.detect, self.detect_frame)
        decision = self.current_decision()
        if self.observation is None:
            self.vision_text.set("未检测到 DICT_4X4_50 · ID 0")
        else:
            x, y = self.observation.center_px
            stable = "稳定" if self.detection_stable() else "等待稳定"
            self.vision_text.set(
                f"ID 0 · 中心 ({x:.1f}, {y:.1f}) px · 距离 {self.observation.depth_m*100:.1f} cm · "
                f"PnP RMS {self.observation.reprojection_rms_px:.2f}px · {stable}")
        self.decision_text.set(f"当前动作：{decision.explanation}")
        display = draw_overlay(frame, self.observation, decision)
        width, height = max(2, self.preview.winfo_width()), max(2, self.preview.winfo_height())
        if stamp != self.last_frame_stamp or self.preview_size != (width, height):
            self.last_frame_stamp, self.preview_size = stamp, (width, height)
            rgb = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            rgb.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(rgb)
            self.preview.coords(self.preview_image, width // 2, height // 2)
            self.preview.itemconfigure(self.preview_image, image=self.photo)
            self.preview.itemconfigure(self.preview_message, state="hidden")

    def tick(self):
        self.pump_worker()
        if not self.closing_ui:
            self.update_detection()
            if self.auto_enabled and not self.busy:
                try:
                    limit = int(self.max_steps.get())
                except ValueError:
                    limit = 0
                decision = self.current_decision()
                if decision.done:
                    self.auto_enabled = False
                    self.auto_button.configure(text="开始连续伺服")
                    self.notice.set(decision.explanation)
                    self.write_event("target_reached", decision.explanation)
                elif self.auto_steps >= limit:
                    self.auto_enabled = False
                    self.auto_button.configure(text="开始连续伺服")
                    self.notice.set(f"已达到自动步数上限 {limit}，停止并保持。")
                elif self.detection_stable() and time.monotonic() - self.last_auto_request > .6:
                    self.last_auto_request = time.monotonic()
                    self.request_step()
            self.refresh_buttons()
            if time.monotonic() - self.last_state >= 1:
                self.dump_state()
                self.last_state = time.monotonic()
            self.root.after(40, self.tick)
            return
        if self.worker is None or not self.worker.is_alive():
            self.pool.shutdown(wait=False, cancel_futures=True)
            if self.camera is not None:
                self.camera.close()
            self.dump_state(closed=True)
            self.root.destroy()
        else:
            self.root.after(50, self.tick)

    def close(self):
        if self.closing_ui:
            return
        self.closing_ui = True
        self.auto_enabled = False
        self.cancel_event.set()
        self.closing_event.set()
        self.commands.put(("close", None))
        self.notice.set("正在停止运动并关闭相机；已控制的关节保持当前位置。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--port")
    parser.add_argument("--observe-only", action="store_true", help="只开相机，不连接或控制机器人")
    parser.add_argument("--no-camera", action="store_true", help="仅用于界面冒烟测试")
    parser.add_argument("--smoke-seconds", type=float)
    args = parser.parse_args()
    if args.no_camera and not args.smoke_seconds:
        parser.error("--no-camera 仅用于 --smoke-seconds 界面测试")
    root = tk.Tk()
    App(root, args.intrinsics.resolve(), args.hand_eye.resolve(), args.port,
        args.observe_only or args.no_camera, args.no_camera, args.smoke_seconds)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("ArUco 视觉伺服启动失败", str(exc))
        except tk.TclError:
            pass
        raise SystemExit(1)
