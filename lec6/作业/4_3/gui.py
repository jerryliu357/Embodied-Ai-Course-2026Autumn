"""4.3 dual-model GUI. Default: simulated camera/robot and real model APIs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

import cv2
from PIL import Image, ImageTk
from audio_io import Recorder, Speaker
from session import Session


def fit_size(source, available):
    sw, sh = source
    width, height = max(1, available[0]), max(1, available[1])
    scale = min(width / sw, height / sh)
    return max(1, int(sw * scale)), max(1, int(sh * scale))


class App:
    def __init__(self, root, real=False, offline=False, mute=False, port=None):
        self.root, self.port = root, port
        self.session = None
        self.events = queue.Queue()
        self.recorder, self.speaker = Recorder(), Speaker()
        self.closing, self.busy, self.recording = False, False, False
        self.close_done = threading.Event()
        self.close_error = None
        self.photo = None
        self.frames_rendered = 0
        self.mode = tk.StringVar(value="离线演示" if offline else "真机" if real else "模拟（真实API）")
        self.with_image = tk.BooleanVar(value=True)
        self.read_aloud = tk.BooleanVar(value=not mute)
        self.status = tk.StringVar(value="准备就绪")
        self.camera_status = tk.StringVar(value="等待画面")
        self.robot_status = tk.StringVar(value="尚未读取机器人状态")
        root.title("4.3 视觉语音机械臂 Agent")
        root.geometry(f"{min(1240, root.winfo_screenwidth()-80)}x{min(850, root.winfo_screenheight()-100)}")
        root.minsize(900, 620)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.stop())
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        top = ttk.Frame(root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="运行模式").pack(side="left")
        self.mode_box = ttk.Combobox(top, textvariable=self.mode, state="readonly", width=20,
                                    values=("模拟（真实API）", "真机", "离线演示"))
        self.mode_box.pack(side="left", padx=6)
        self.mode_button = ttk.Button(top, text="应用模式 / 重连", command=self.change_mode)
        self.mode_button.pack(side="left", padx=5)
        ttk.Checkbutton(top, text="附带当前画面", variable=self.with_image).pack(side="left", padx=8)
        ttk.Checkbutton(top, text="语音播报", variable=self.read_aloud, command=self.toggle_speech).pack(side="left")
        ttk.Button(top, text="退出并释放设备", command=self.close).pack(side="right")
        ttk.Button(top, text="停止动作 / Esc", command=self.stop).pack(side="right", padx=5)
        audio = ttk.Frame(root, padding=(8, 0, 8, 8))
        audio.grid(row=1, column=0, sticky="ew")
        ttk.Label(audio, text="麦克风").pack(side="left")
        self.mic = ttk.Combobox(audio, state="readonly", width=40)
        self.mic.pack(side="left", padx=6)
        self.mic_ids = [None]
        try:
            devices = self.recorder.devices()
            self.mic_ids += [i for i, _ in devices]
            self.mic["values"] = ["系统默认输入"] + [f"{i}: {name}" for i, name in devices]
        except Exception as exc:
            self.mic["values"] = ["麦克风不可用：" + str(exc)]
        self.mic.current(0)
        self.record_button = ttk.Button(audio, text="开始录音", command=self.toggle_record)
        self.record_button.pack(side="left", padx=5)
        self.load_audio = ttk.Button(audio, text="载入WAV测试", command=self.open_audio)
        self.load_audio.pack(side="left", padx=5)
        self.record_label = ttk.Label(audio, text="说完后点击“结束录音并发送”")
        self.record_label.pack(side="left", padx=6)
        body = ttk.Frame(root, padding=(8, 0))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1, uniform="body")
        body.columnconfigure(1, weight=1, uniform="body")
        body.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(body, text="腕部相机 / 模拟画面", padding=5)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        # Canvas requested size never follows the rendered image size.
        self.canvas = tk.Canvas(left, width=1, height=1, bg="#17212b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.image_item = self.canvas.create_image(0, 0, anchor="center")
        ttk.Label(left, textvariable=self.camera_status, wraplength=450).grid(row=1, column=0, sticky="w")
        ttk.Label(left, textvariable=self.robot_status, wraplength=450).grid(row=2, column=0, sticky="w")
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(5, weight=1)
        ttk.Label(right, text="识别出的语音 / 本轮文字").grid(row=0, column=0, sticky="w")
        self.transcript = ScrolledText(right, height=4, width=30, wrap="word", state="disabled")
        self.transcript.grid(row=1, column=0, sticky="ew")
        ttk.Label(right, text="Omni环境理解").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.scene = ScrolledText(right, height=4, width=30, wrap="word", state="disabled")
        self.scene.grid(row=3, column=0, sticky="ew")
        ttk.Label(right, text="模型回答 / 工具执行").grid(row=4, column=0, sticky="w", pady=(6, 0))
        tabs = ttk.Notebook(right)
        tabs.grid(row=5, column=0, sticky="nsew")
        self.answers = ScrolledText(tabs, width=30, height=8, wrap="word", state="disabled")
        self.logs = ScrolledText(tabs, width=30, height=8, wrap="word", state="disabled")
        tabs.add(self.answers, text="对话回答")
        tabs.add(self.logs, text="调用记录")
        bottom = ttk.Frame(root, padding=8)
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(bottom)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", lambda event: self.send_text())
        self.send_button = ttk.Button(bottom, text="发送文字", command=self.send_text)
        self.send_button.grid(row=0, column=1, padx=6)
        shortcuts = ttk.Frame(bottom)
        shortcuts.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        self.shortcuts = []
        for label, command in (("状态", "查看机器人当前状态"), ("上移1cm", "末端向上移动1厘米"),
                               ("回middle", "回到middle姿态"), ("检测ArUco", "检测ArUco 0"),
                               ("开始追踪", "如果检测到ArUco 0，开始追踪到20厘米，启动后回复")):
            button = ttk.Button(shortcuts, text=label, command=lambda text=command: self.submit(text=text))
            button.pack(side="left", padx=(0, 5))
            self.shortcuts.append(button)
        ttk.Label(bottom, textvariable=self.status, wraplength=1100).grid(row=2, column=0, columnspan=2, sticky="w")
        self.start_session()
        root.after(80, self.tick)

    def emit(self, kind, value):
        self.events.put((kind, value))

    def start_session(self):
        try:
            self.session = Session(dry_run=self.mode.get() != "真机", offline=self.mode.get() == "离线演示",
                                   emit=self.emit, port=self.port)
            self.status.set(f"{self.mode.get()} · 准备就绪 · 日志：{self.session.folder}")
            self.active_mode = self.mode.get()
        except Exception as exc:
            self.session = None
            self.status.set("启动失败：" + str(exc))

    @staticmethod
    def write(widget, text, replace=False):
        widget.configure(state="normal")
        if replace:
            widget.delete("1.0", "end")
        widget.insert("end", str(text) + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def set_busy(self, value):
        self.busy = value
        state = "disabled" if value or self.recording or self.closing else "normal"
        for button in [self.send_button, self.load_audio, self.mode_button, *self.shortcuts]:
            button.configure(state=state)
        self.mode_box.configure(state="disabled" if state == "disabled" else "readonly")
        self.record_button.configure(state="disabled" if value or self.closing else "normal")

    def send_text(self):
        text = self.entry.get().strip()
        if text and not self.busy and not self.recording:
            self.entry.delete(0, "end")
            self.submit(text=text)

    def submit(self, text=None, audio_path=None):
        if self.closing or self.busy or self.recording:
            return
        if self.session is None:
            self.status.set("请先应用模式 / 重连")
            return
        use_image, session = self.with_image.get(), self.session
        self.speaker.stop()
        session.prepare_turn()
        self.set_busy(True)
        if text:
            self.write(self.transcript, text, replace=True)
            self.write(self.answers, "你：" + text)
        else:
            self.write(self.transcript, "正在识别语音……", replace=True)
        def work():
            try:
                session.turn(text=text, audio_path=audio_path, use_image=use_image)
            except KeyboardInterrupt:
                self.emit("stage", "本轮已停止；未执行的后续动作已取消")
            except Exception as exc:
                self.emit("error", session.redact(exc))
            finally:
                self.emit("idle", None)
        threading.Thread(target=work, name="agent-turn-43", daemon=True).start()

    def toggle_record(self):
        if self.busy or self.closing or self.session is None:
            return
        if not self.recording:
            try:
                self.speaker.stop()
                self.recorder.start(self.mic_ids[max(0, self.mic.current())])
                self.recording, self.record_start = True, time.monotonic()
                self.record_button.configure(text="结束录音并发送")
                self.mic.configure(state="disabled")
                self.set_busy(False)
            except Exception as exc:
                self.status.set("录音失败：" + str(exc))
        else:
            self.recording = False
            self.record_button.configure(text="开始录音")
            self.mic.configure(state="readonly")
            self.set_busy(False)
            try:
                path = self.session.folder / f"microphone_{time.time_ns()}.wav"
                self.recorder.finish(path)
                self.record_label.configure(text="录音已保存")
                self.submit(audio_path=path)
            except Exception as exc:
                self.status.set("录音失败：" + str(exc))

    def open_audio(self):
        path = filedialog.askopenfilename(title="选择测试录音", filetypes=[("PCM WAV", "*.wav")])
        if path:
            self.submit(audio_path=Path(path))

    def toggle_speech(self):
        if not self.read_aloud.get():
            self.speaker.stop()

    def stop(self):
        self.speaker.stop()
        if self.recording:
            self.recorder.close()
            self.recording = False
            self.record_button.configure(text="开始录音")
            self.mic.configure(state="readonly")
            self.set_busy(self.busy)
        if self.session:
            session = self.session
            session.stop()
            def hold():
                try:
                    self.emit("log", "停止结果：" + json.dumps(session.hold_after_stop(), ensure_ascii=False))
                except Exception as exc:
                    self.emit("error", session.redact(exc))
            threading.Thread(target=hold, name="stop-43", daemon=True).start()
        self.status.set("已请求停止动作与本轮调用；正在等待的网络请求结束后即可继续输入")

    def change_mode(self):
        if self.busy or self.recording or self.closing:
            return
        self.set_busy(True)
        old = self.session
        self.session = None
        self.speaker.stop()
        def work():
            try:
                if old:
                    old.close()
                self.emit("reconnect", None)
            except Exception as exc:
                self.emit("error", str(exc))
                self.emit("idle", None)
        threading.Thread(target=work, daemon=True).start()

    def tick(self):
        if self.closing:
            return
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "perception":
                self.write(self.transcript, value["transcript"], replace=True)
                self.write(self.scene, value["scene_summary"] + "\n" + value["uncertainty"], replace=True)
            elif kind == "answer":
                self.write(self.answers, "Agent：" + value + "\n")
                if (self.read_aloud.get() and not self.recording and self.session
                        and not self.session.cancelled.is_set()):
                    token = self.speaker.token()
                    threading.Thread(target=self.speaker.speak, args=(value, token), name="speech-43", daemon=True).start()
                self.status.set("本轮完成，可以继续输入")
            elif kind == "error":
                self.write(self.answers, "本轮未完成：" + value)
                self.status.set("本轮未完成：" + value)
            elif kind in ("log", "tracking"):
                self.write(self.logs, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
                if kind == "tracking":
                    self.status.set("追踪：" + str(value.get("reason", value.get("status"))))
            elif kind in ("stage", "progress"):
                self.status.set(value)
            elif kind == "camera":
                self.camera_status.set(value)
            elif kind == "idle":
                self.set_busy(False)
            elif kind == "reconnect":
                self.start_session()
                self.set_busy(False)
        if self.session:
            sample = self.session.preview()
            if sample:
                frame, stamp = sample
                width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
                if width > 1 and height > 1:
                    size = fit_size((frame.shape[1], frame.shape[0]), (width, height))
                    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(size, Image.Resampling.BILINEAR)
                    self.photo = ImageTk.PhotoImage(rgb)
                    self.canvas.itemconfigure(self.image_item, image=self.photo)
                    self.canvas.coords(self.image_item, width / 2, height / 2)
                    self.frames_rendered += 1
                    self.camera_status.set(f"{self.active_mode} · {frame.shape[1]}×{frame.shape[0]} · 画面年龄 {time.monotonic()-stamp:.1f}s")
            state = self.session.runtime._cached_state()  # Cached only; GUI never reads serial.
            raw = state.get("joints_raw", {})
            self.robot_status.set(f"{'已连接' if state['connected'] else '等待工具首次连接'} · 追踪 {state['tracking']['status']}\n"
                                  + ("ID1～6：" + ", ".join(str(v) for v in raw.values()) if raw else ""))
        if self.recording:
            self.record_label.configure(text=f"正在录音 {time.monotonic()-self.record_start:.1f} 秒")
        self.root.after(100, self.tick)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.session:
            self.session.stop()
        self.set_busy(True)
        self.status.set("正在关闭录音、相机、播报和机器人连接……")
        def cleanup():
            errors = []
            for closer in (self.recorder.close, self.speaker.close,
                           self.session.close if self.session else lambda: None):
                try:
                    closer()
                except Exception as exc:
                    errors.append(str(exc))
            self.close_error = "; ".join(errors) or None
            if self.close_error:
                print("关闭异常：" + self.close_error, flush=True)
            self.close_done.set()
        threading.Thread(target=cleanup, name="cleanup-43", daemon=True).start()
        self.root.after(100, self._wait_closed)

    def _wait_closed(self):
        if self.close_done.is_set():
            self.root.destroy()
        else:
            self.root.after(100, self._wait_closed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--real", action="store_true", help="真实相机和机械臂")
    modes.add_argument("--dry-run", action="store_true", help="模拟硬件，真实API（默认）")
    modes.add_argument("--offline", action="store_true", help="完全离线GUI演示，不连接任何硬件或模型API")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--smoke-seconds", type=float, help="GUI测试后自动退出；仅模拟模式可用")
    args = parser.parse_args()
    if args.real and args.smoke_seconds:
        parser.error("自动GUI测试仅允许模拟模式")
    root = tk.Tk()
    app = App(root, real=args.real, offline=args.offline, mute=args.mute, port=args.port)
    if args.smoke_seconds:
        root.after(int(args.smoke_seconds * 1000), app.close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
        root.mainloop()
    return 1 if app.close_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
