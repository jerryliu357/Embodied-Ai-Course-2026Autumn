"""Six 0..1 servo sliders + live USB preview + FK hand-eye Capture."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from fk_capture_core import (HERE, FK_DIR, JOINTS, Mailbox, JointWorker, Camera,
                             SampleStore, load_mapping, new_session, detect, write_json)
from control_fk import FKWindow


class App(FKWindow):
    """Reuse lec4's six slider rows; provide a camera/capture workspace."""
    def __init__(self, root, data, mapping, store, port=None, demo=False, camera=True):
        self.root, self.data, self.mapping, self.samples = root, data, mapping, store
        self.port, self.demo = port, demo
        self.worker = self.camera = None
        self.mailbox, self.output = Mailbox(), queue.Queue()
        self.rows, self.snapshot = {}, {}
        self.playing = self.closing = self.updating = self.busy = self.ready = False
        self.pressed = set()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")
        self.vision_future = self.save_future = None
        self.last_detect = self.last_state = self.last_frame = 0
        self.preview_frame = None
        self.preview_size = None
        self.corners_found = False
        self.camera_enabled = camera
        root.title("Week 6 · 六关节 FK + 相机采集" + (" · 离线模拟" if demo else ""))
        root.geometry("1360x850")
        root.minsize(1060, 720)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=5)
        style.configure("Joint.TButton", font=("Microsoft YaHei UI", 9), padding=2)
        style.configure("Actual.Horizontal.TProgressbar", background="#149b88", thickness=6)
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="六关节控制 · 相机预览 · FK 采集",
                  font=("Microsoft YaHei UI", 19, "bold")).pack(anchor="w")
        ttk.Label(outer, text="调整关节 → 等停稳 → Capture；照片与实测关节 FK 位姿成对保存。").pack(anchor="w", pady=5)
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=8)
        self.connection = tk.StringVar(value="尚未连接")
        ttk.Label(bar, textvariable=self.connection).pack(side="left")
        self.speed = tk.StringVar(value="60")
        ttk.Label(bar, text="速度上限（计数/秒）").pack(side="left", padx=(25, 5))
        speed = ttk.Spinbox(bar, from_=1, to=120, textvariable=self.speed, width=6, command=self.change_speed)
        speed.pack(side="left")
        speed.bind("<Return>", self.change_speed)
        speed.bind("<FocusOut>", self.change_speed)
        self.connect_button = ttk.Button(bar, text="重连机器人", command=self.connect)
        self.connect_button.pack(side="right")
        self.status = tk.StringVar(value="启动只读取位置，拖动后才控制相应关节。")
        self.notice = tk.StringVar(value="保持机器人底座、棋盘和相机安装固定；每次小幅调整，注意实体间距。")
        ttk.Label(outer, textvariable=self.notice, wraplength=1250).pack(side="bottom", anchor="w", pady=5)
        ttk.Label(outer, textvariable=self.status, foreground="#146e61").pack(side="bottom", anchor="w")
        bottom = ttk.Frame(outer)
        bottom.pack(side="bottom", fill="x", pady=10)
        ttk.Button(bottom, text="暂停并保持 · 空格", command=self.hold).pack(side="left")
        ttk.Button(bottom, text="全部卸力 · Esc（先托稳）", command=self.emergency).pack(side="left", padx=10)
        ttk.Button(bottom, text="退出并保持", command=self.close).pack(side="right")
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(body, text="六个滑条 · 0～1", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        canvas = tk.Canvas(left, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        rows = ttk.Frame(canvas)
        item = canvas.create_window(0, 0, anchor="nw", window=rows)
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
        for sid, name in JOINTS.items():
            self.make_row(rows, sid, name)
        right = ttk.LabelFrame(body, text="USB 相机 · 1280 × 720", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        # A Label requests the natural size of each PhotoImage. Resizing the next
        # frame from that requested size creates a positive feedback loop and the
        # whole window grows after the camera opens. A Canvas owns fixed layout
        # geometry; changing its image item never changes the requested size.
        self.preview = tk.Canvas(right, bg="#172638", highlightthickness=0,
                                 width=640, height=360)
        self.preview.pack(fill="both", expand=True)
        self.preview_image = self.preview.create_image(320, 180, anchor="center")
        self.preview_message = self.preview.create_text(
            320, 180, text="正在打开相机……", fill="white",
            font=("Microsoft YaHei UI", 11))
        self.vision_status = tk.StringVar(value="等待画面")
        ttk.Label(right, textvariable=self.vision_status).pack(anchor="w", pady=7)
        self.fk_status = tk.StringVar(value="FK 末端位置：等待实测反馈")
        ttk.Label(right, textvariable=self.fk_status, wraplength=600).pack(anchor="w", pady=4)
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=8)
        self.capture_button = ttk.Button(buttons, text="Capture · C", command=self.capture, state="disabled")
        self.capture_button.pack(side="left")
        ttk.Button(buttons, text="重开相机", command=self.restart_camera).pack(side="left", padx=8)
        ttk.Button(buttons, text="打开数据目录", command=lambda: os.startfile(str(store.folder))).pack(side="right")
        self.count = tk.StringVar(value=f"已有 {len(store.samples)} 组；将继续追加保存")
        ttk.Label(right, textvariable=self.count).pack(anchor="w")
        ttk.Label(right, text=f"会话：{store.folder.name}\n完整棋盘：6 × 8 内角点；单格：25 mm\n采集不同朝向的姿态；只改变夹爪不产生新的 FK 位姿。",
                  wraplength=600, foreground="#536579").pack(anchor="w", pady=6)
        root.bind("<KeyPress>", self.keydown)
        root.bind("<KeyRelease>", lambda e: self.pressed.discard(e.keysym))
        root.bind("<FocusOut>", lambda e: self.pressed.clear())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self.pump)
        root.after(100, self.restart_camera)
        root.after(200, self.connect)

    def change_speed(self, event=None):
        try:
            value = int(self.speed.get())
            if not 1 <= value <= 120:
                raise ValueError()
            self.mailbox.speed = value
            return value
        except ValueError:
            self.notice.set("请输入 1～120 的整数；当前仍使用上次有效速度。")
            return None

    def connect(self):
        if self.closing or (self.worker is not None and self.worker.is_alive()):
            return
        self.mailbox = Mailbox()
        self.mailbox.speed = 60
        self.change_speed()
        self.connect_button.configure(state="disabled")
        self.connection.set("正在读取六关节……")
        self.worker = JointWorker(self.data, self.port, self.mailbox, self.output, self.camera, self.demo)
        self.worker.start()

    def restart_camera(self):
        if self.closing or self.busy:
            return
        if not self.camera_enabled:
            self.vision_status.set("离线测试：相机关闭")
            return
        try:
            if self.camera is not None:
                self.camera.close()
            self.camera = Camera(self.samples.folder)
            if self.worker is not None:
                self.worker.camera = self.camera
        except Exception as exc:
            self.notice.set(f"相机启动失败：{exc}")

    def keydown(self, event):
        if event.keysym in self.pressed:
            return
        if event.keysym == "Escape":
            self.pressed.add(event.keysym)
            self.emergency()
            return "break"
        if isinstance(self.root.focus_get(), (ttk.Entry, ttk.Spinbox, tk.Entry)):
            return
        action = {"space": self.hold, "c": self.capture, "C": self.capture}.get(event.keysym)
        if action:
            self.pressed.add(event.keysym)
            action()
            return "break"

    def capture(self):
        if self.closing or self.busy:
            return
        if any(row["dragging"] for row in self.rows.values()):
            self.notice.set("请先松开滑条，再等关节停稳。")
            return
        if not self.worker or not self.worker.is_alive() or not self.ready:
            self.notice.set("请先连接机器人，松开滑条并等六个关节停稳 1 秒。")
            return
        if not self.camera or self.camera.sample() is None or self.mailbox.cancelled():
            self.notice.set("请先等待相机画面和机器人就绪。")
            return
        self.busy = self.playing = True
        self.capture_button.configure(state="disabled")
        self.notice.set("正在读取同一静止姿态的关节和图像……")
        self.mailbox.submit("capture")

    def pump(self):
        try:
            while True:
                kind, value = self.output.get_nowait()
                if kind == "connected":
                    self.connection.set(value)
                elif kind == "snapshot":
                    snapshot, self.ready = value
                    self.show_snapshot(snapshot, self.busy, "关节已稳定，可采集" if self.ready else "等待六关节停稳 ≥ 1 秒")
                    if all(s["online"] for s in snapshot.values()):
                        raw = [snapshot[sid]["raw"] for sid in JOINTS]
                        T = self.mapping.arm.fk(self.mapping.to_q(raw))
                        self.fk_status.set("实测 FK · base → gripper_frame\nX %.1f   Y %.1f   Z %.1f mm" % tuple(T[:3,3]*1000))
                elif kind == "captured":
                    self.save_future = self.pool.submit(self.samples.save, value)
                    self.notice.set("正在检查棋盘并保存配对数据……")
                elif kind == "capture_error":
                    self.busy = self.playing = False
                    self.notice.set(value)
                elif kind == "capture_cancelled":
                    if self.save_future is None:
                        self.busy = self.playing = False
                elif kind in ("notice", "error"):
                    self.notice.set(value)
                elif kind == "closed":
                    self.ready = False
                    self.busy = self.playing = False
                    self.connection.set("连接已关闭")
                    self.connect_button.configure(state="normal")
                    for row in self.rows.values():
                        row["initialized"] = False
                        for key in ("slider", "entry", "button"):
                            row[key].configure(state="disabled")
        except queue.Empty:
            pass
        if self.save_future is not None and self.save_future.done():
            try:
                sample = self.save_future.result()
                self.count.set(f"已保存 {len(self.samples.samples)} 组；最新 sample_{sample['index']:02d}")
                self.notice.set(f"第 {sample['index']} 组保存成功，RMS {sample['pnp_rms_px']:.2f}px。请换一个姿态后继续。")
            except Exception as exc:
                self.notice.set(f"本次未保存：{exc}")
            self.save_future = None
            self.busy = self.playing = False
        if self.closing:
            if (not self.worker or not self.worker.is_alive()) and self.save_future is None:
                self.pool.shutdown(wait=False, cancel_futures=True)
                if self.camera:
                    self.camera.close()
                self.dump_state(closed=True)
                self.root.destroy()
                return
        else:
            self.update_preview()
        latest = self.camera.sample() if self.camera else None
        fresh = latest is not None and time.monotonic()-latest[0] < .5
        self.capture_button.configure(state="normal" if self.ready and fresh and not self.busy and not self.closing and not self.demo else "disabled")
        if time.monotonic()-self.last_state > 1:
            self.dump_state()
            self.last_state = time.monotonic()
        self.root.after(40, self.pump)

    def update_preview(self):
        latest = self.camera.sample() if self.camera else None
        if latest is None or time.monotonic()-latest[0] > .5:
            self.vision_status.set("相机暂无新画面；关闭其他相机程序后点击「重开相机」")
            return
        stamp, frame = latest
        width = max(2, self.preview.winfo_width())
        height = max(2, self.preview.winfo_height())
        size = (width, height)
        if stamp != self.last_frame or size != self.preview_size:
            self.last_frame = stamp
            self.preview_size = size
            self.preview_frame = frame
            rgb = Image.fromarray(cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB))
            rgb.thumbnail(size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(rgb)
            self.preview.coords(self.preview_image, width // 2, height // 2)
            self.preview.itemconfigure(self.preview_image, image=self.photo)
            self.preview.itemconfigure(self.preview_message, state="hidden")
        if self.vision_future is not None and self.vision_future.done():
            try:
                self.corners_found = self.vision_future.result() is not None
                self.vision_status.set("棋盘检测：48 / 48 角点 ✓" if self.corners_found else "棋盘不完整：调整到 48 个角点全部可见")
            except Exception as exc:
                self.vision_status.set(str(exc))
            self.vision_future = None
        if self.vision_future is None and time.monotonic()-self.last_detect > .5 and not self.busy:
            self.last_detect = time.monotonic()
            self.vision_future = self.pool.submit(detect, frame.copy())

    def dump_state(self, closed=False):
        try:
            write_json(self.samples.folder / "fk_gui_state.json", dict(pid=os.getpid(), closed=closed,
                       updated_at=time.time(), connection=self.connection.get(), ready=self.ready,
                       camera_frames=self.camera.frames if self.camera else 0,
                       camera_frame_age_s=time.monotonic()-self.last_frame if self.last_frame else None,
                       corners_found=self.corners_found, sample_count=len(self.samples.samples),
                       notice=self.notice.get(), joints=self.snapshot))
        except OSError:
            pass

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.mailbox.closing.set()
        self.notice.set("正在保持已控制关节并关闭相机；已取出的样本会完成保存。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="继续指定目录；默认继续最近的手眼采集")
    parser.add_argument("--new-session", action="store_true")
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--port")
    parser.add_argument("--demo", action="store_true", help="模拟舵机；不会保存模拟标定数据")
    parser.add_argument("--no-camera", action="store_true", help="仅限 --demo")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--smoke-seconds", type=float, help="仅限 --demo：自动退出测试界面")
    args = parser.parse_args()
    if (args.no_camera or args.smoke_seconds) and not args.demo:
        parser.error("--no-camera / --smoke-seconds 仅用于 --demo")
    if args.session and args.new_session:
        parser.error("--session 与 --new-session 不能同时使用")
    data = json.loads((FK_DIR / "calibration.json").read_text(encoding="utf-8-sig"))
    mapping = load_mapping(data)
    session = args.session
    if session is None and not args.new_session:
        paths = sorted((HERE / "data").glob("hand_eye_*/session.json"))
        session = paths[-1].parent if paths else None
    intrinsics = args.intrinsics
    if intrinsics is None and session is not None:
        intrinsics = Path(json.loads((session / "session.json").read_text(encoding="utf-8-sig"))["intrinsics_file"])
    if intrinsics is None:
        paths = sorted((HERE / "data").glob("*/intrinsics.json"))
        if not paths:
            raise ValueError("找不到内参，请用 --intrinsics 指定")
        intrinsics = paths[-1]
    if args.check_files:
        from hand_eye_calibration import load_intrinsics
        load_intrinsics(intrinsics)
        if session:
            store = SampleStore(session, intrinsics, mapping)
            print(f"OK: six-servo mapping, FK, intrinsics; existing samples={len(store.samples)}")
        return
    if session is None:
        session = new_session(intrinsics)
        print(f"NEW_HAND_EYE_SESSION={session.resolve()}", flush=True)
    store = SampleStore(session, intrinsics, mapping)
    root = tk.Tk()
    app = App(root, data, mapping, store, args.port, args.demo, not args.no_camera)
    if args.smoke_seconds:
        root.after(int(args.smoke_seconds*1000), app.close)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("FK 相机采集启动失败", str(exc))
        except tk.TclError:
            pass
        raise SystemExit(1)
