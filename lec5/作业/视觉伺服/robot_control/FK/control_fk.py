"""Week 4 / 1.1：六滑条示教、保存姿态、键盘播放有时序的动作。

默认连接实机，但启动只读；--demo 完全离线，录制仅保存在内存。
--check-files 只检查 JSON 和所需姿态，不创建窗口、不打开串口。
"""
import argparse
import copy
import json
import queue
import tkinter as tk
from tkinter import ttk, messagebox

from arm_backend import JOINTS, POSITION_TOLERANCE, target_position
from action_store import ActionStore, CALIBRATION_PATH, finite_number
from motion import Mailbox, SerialWorker


class FKWindow:
    def __init__(self, root, data, store, port=None, demo=False, autostart=True):
        self.root, self.data, self.store, self.port, self.demo = root, data, store, port, demo
        self.worker = None
        self.mailbox, self.output = Mailbox(), queue.Queue()
        self.rows, self.time_rows, self.snapshot = {}, [], {}
        self.playing = self.closing = self.updating = False
        self.pressed = set()
        root.title("Week 4 · FK 预设姿态与动作" + (" · 离线模拟" if demo else ""))
        root.geometry("1260x900")
        root.minsize(1040, 720)
        style = ttk.Style(root)
        style.theme_use("clam")
        root.configure(bg="#f3f6fb")
        style.configure("TFrame", background="#f3f6fb")
        style.configure("TLabelframe", background="#f3f6fb")
        style.configure("TLabelframe.Label", background="#f3f6fb")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10), background="#f3f6fb")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=5)
        style.configure("Joint.TButton", font=("Microsoft YaHei UI", 9), padding=2)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 19, "bold"), foreground="#163052")
        style.configure("Actual.Horizontal.TProgressbar", background="#149b88", thickness=6)
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="预设姿态与动作", style="Title.TLabel").pack(anchor="w")
        description = "离线模拟：不打开串口，数值仅为示例，保存操作只影响内存。" if demo else "沿用上次六关节校准 · 启动读取当前位置 · 录制实测姿态 · 按键播放"
        ttk.Label(outer, text=description, foreground="#9a5700" if demo else "#536579").pack(anchor="w", pady=(3, 10))
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 12))
        self.connection = tk.StringVar(value="尚未连接")
        ttk.Label(bar, textvariable=self.connection).pack(side="left")
        self.connect_button = ttk.Button(bar, text="连接", command=self.connect)
        self.connect_button.pack(side="right")
        settings = store.document["settings"]
        self.speed = tk.StringVar(value=str(settings["speed_limit"]))
        self.duration = tk.StringVar(value=str(settings["pose_seconds"]))
        ttk.Label(bar, text="速度上限（计数/秒）").pack(side="left", padx=(20, 4))
        speed = ttk.Spinbox(bar, from_=1, to=1000, textvariable=self.speed, width=6, command=self.change_speed)
        speed.pack(side="left")
        speed.bind("<Return>", self.change_speed)
        speed.bind("<FocusOut>", self.change_speed)
        ttk.Label(bar, text="单姿态切换（秒）").pack(side="left", padx=(15, 4))
        ttk.Spinbox(bar, from_=0.1, to=120, increment=0.1, textvariable=self.duration, width=6).pack(side="left")
        body = ttk.Frame(outer)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(body, text="1 · 调整六个关节", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        canvas = tk.Canvas(left, highlightthickness=0, background="#f3f6fb")
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        row_frame = ttk.Frame(canvas)
        item = canvas.create_window(0, 0, anchor="nw", window=row_frame)
        row_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
        for sid, name in JOINTS.items():
            self.make_row(row_frame, sid, name)
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        poses = ttk.LabelFrame(right, text="2 · 保存姿态 / 播放姿态", padding=10)
        poses.pack(fill="x", pady=(0, 12))
        self.pose = tk.StringVar(value="stand")
        self.pose_choice = ttk.Combobox(poses, textvariable=self.pose, state="readonly",
                                       values=list(store.document["poses"]))
        self.pose_choice.pack(fill="x")
        self.pose_choice.bind("<<ComboboxSelected>>", self.refresh_pose)
        self.pose_info = tk.StringVar()
        ttk.Label(poses, textvariable=self.pose_info, wraplength=405, foreground="#536579").pack(anchor="w", pady=7)
        buttons = ttk.Frame(poses)
        buttons.pack(fill="x")
        self.capture_button = ttk.Button(buttons, text="记录当前实测姿态", command=self.capture)
        self.capture_button.pack(side="left")
        ttk.Button(buttons, text="播放所选姿态", command=self.play_pose).pack(side="right")
        ttk.Label(poses, text="先录制 stand 立正，再录制 lift 抬起、left 向左、right 向右。\n录制只保存位置，不会重新校准或主动移动。",
                  wraplength=405).pack(anchor="w", pady=(10, 0))
        actions = ttk.LabelFrame(right, text="3 · 动作顺序与时间", padding=10)
        actions.pack(fill="both", expand=True)
        self.action = tk.StringVar(value="wave")
        choice = ttk.Combobox(actions, textvariable=self.action, state="readonly", values=list(store.document["actions"]))
        choice.pack(fill="x")
        choice.bind("<<ComboboxSelected>>", self.refresh_steps)
        self.step_frame = ttk.Frame(actions)
        self.step_frame.pack(fill="x", pady=10)
        self.refresh_steps()
        ttk.Button(actions, text="保存时间 / 顺序设置", command=self.save_settings).pack(fill="x")
        ttk.Button(actions, text="播放所选动作", command=self.play_action).pack(fill="x", pady=5)
        ttk.Label(actions, text="时间短于速度上限允许值时会延长，并显示实际规划时长。\n每段稳定到位后再进入下一段；滑条调整不参与动作排队。",
                  wraplength=405, foreground="#536579").pack(anchor="w", pady=5)
        bottom = ttk.Frame(outer)
        ttk.Button(bottom, text="1 · 立正", command=lambda: self.play_pose("stand")).pack(side="left")
        ttk.Button(bottom, text="2 · 挥手", command=lambda: self.play_action("wave")).pack(side="left", padx=6)
        ttk.Button(bottom, text="暂停保持 · 空格", command=self.hold).pack(side="left", padx=6)
        tk.Button(bottom, text="全部卸力 · Esc", command=self.emergency, bg="#ba3340", fg="white",
                  font=("Microsoft YaHei UI", 10, "bold"), relief="flat", padx=10, pady=5).pack(side="left", padx=6)
        ttk.Button(bottom, text="退出并保持", command=self.close).pack(side="right")
        self.status = tk.StringVar(value="尚未开始动作。")
        self.notice = tk.StringVar(value="托稳后再卸力。键盘 1 / 2 / 空格在非输入框区域生效，Esc 始终有效。")
        # 先为底部状态和停止按钮留位置，小窗口下关节区滚动，不遮住控制反馈。
        ttk.Label(outer, textvariable=self.notice, wraplength=1170).pack(side="bottom", anchor="w", pady=(5, 0))
        ttk.Label(outer, textvariable=self.status, foreground="#146e61", wraplength=1170).pack(side="bottom", anchor="w")
        bottom.pack(side="bottom", fill="x", pady=(12, 7))
        body.pack(fill="both", expand=True)
        root.bind("<KeyPress>", self.keydown)
        root.bind("<KeyRelease>", lambda event: self.pressed.discard(event.keysym))
        root.bind("<FocusOut>", lambda event: self.pressed.clear())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_pose()
        root.after(50, self.pump)
        if autostart:
            root.after(150, self.connect)

    def make_row(self, parent, sid, name):
        box = ttk.Frame(parent, padding=(0, 3, 0, 5))
        box.pack(fill="x")
        box.columnconfigure(0, weight=1)
        ttk.Label(box, text=f"ID{sid} · {name}", font=("Microsoft YaHei UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        value, entry_value = tk.DoubleVar(), tk.StringVar(value="—")
        entry = ttk.Entry(box, textvariable=entry_value, width=7, state="disabled")
        entry.grid(row=0, column=1, padx=4)
        button = ttk.Button(box, text="设置", state="disabled", style="Joint.TButton", command=lambda: self.apply(sid))
        button.grid(row=0, column=2)
        ttk.Button(box, text="卸力", style="Joint.TButton", command=lambda: self.mailbox.submit("off_joint", sid)).grid(row=0, column=3, padx=4)
        slider = ttk.Scale(box, from_=0, to=1, variable=value, state="disabled", command=lambda x: self.drag(sid, x))
        slider.grid(row=1, column=0, columnspan=4, sticky="ew", pady=2)
        progress = ttk.Progressbar(box, maximum=1, style="Actual.Horizontal.TProgressbar")
        progress.grid(row=2, column=0, columnspan=4, sticky="ew")
        text = tk.StringVar(value="等待连接")
        ttk.Label(box, textvariable=text, wraplength=620, font=("Microsoft YaHei UI", 9), foreground="#536579").grid(row=3, column=0, columnspan=4, sticky="w", pady=(3, 0))
        self.rows[sid] = dict(value=value, entry_value=entry_value, slider=slider, entry=entry,
                              button=button, progress=progress, text=text, initialized=False, dragging=False)
        slider.bind("<ButtonPress-1>", lambda e: self.begin_drag(sid))
        slider.bind("<ButtonRelease-1>", lambda e: self.end_drag(sid))
        entry.bind("<Return>", lambda e: self.apply(sid))

    def begin_drag(self, sid):
        self.rows[sid]["dragging"] = self.rows[sid]["initialized"] and not self.playing

    def drag(self, sid, value):
        if self.updating or not self.rows[sid]["dragging"]:
            return
        self.rows[sid]["entry_value"].set(f"{float(value):.4f}")
        self.mailbox.submit("target", (sid, float(value)))

    def end_drag(self, sid):
        self.drag(sid, self.rows[sid]["value"].get())
        self.rows[sid]["dragging"] = False

    def apply(self, sid):
        if self.playing or not self.rows[sid]["initialized"]:
            return
        try:
            value = float(self.rows[sid]["entry_value"].get())
            target_position(value, self.data["joints"][str(sid)])
            self.mailbox.submit("target", (sid, value))
        except ValueError as exc:
            self.notice.set(str(exc))

    def change_speed(self, event=None):
        try:
            speed = int(self.speed.get())
            finite_number(speed, 1, 1000, "速度上限")
            self.mailbox.speed = speed
            return speed
        except ValueError:
            self.notice.set("速度上限请输入 1～1000 的整数。")
            return None

    def refresh_pose(self, event=None):
        name = self.pose.get()
        pose = self.store.document["poses"][name]
        values = pose["values"]
        detail = "待录制：调整好六关节后点击记录。" if values is None else "已保存：" + ", ".join(f"{x:.3f}" for x in values)
        self.pose_info.set(f"{pose['label']} / {name}\n{detail}")

    def refresh_steps(self, event=None):
        for widget in self.step_frame.winfo_children():
            widget.destroy()
        self.time_rows.clear()
        for col, text in enumerate(("顺序", "姿态", "运动秒", "停留秒")):
            ttk.Label(self.step_frame, text=text).grid(row=0, column=col, sticky="w", padx=3)
        for i, step in enumerate(self.store.document["actions"][self.action.get()]["steps"], 1):
            pose, seconds, hold = tk.StringVar(value=step["pose"]), tk.StringVar(value=str(step["seconds"])), tk.StringVar(value=str(step.get("hold", 0)))
            ttk.Label(self.step_frame, text=str(i)).grid(row=i, column=0, padx=3, pady=3)
            ttk.Combobox(self.step_frame, textvariable=pose, values=list(self.store.document["poses"]), state="readonly", width=10).grid(row=i, column=1, padx=3)
            ttk.Entry(self.step_frame, textvariable=seconds, width=7).grid(row=i, column=2, padx=3)
            ttk.Entry(self.step_frame, textvariable=hold, width=7).grid(row=i, column=3, padx=3)
            self.time_rows.append((pose, seconds, hold))

    def save_settings(self):
        try:
            speed = self.change_speed()
            if speed is None:
                return False
            duration = finite_number(float(self.duration.get()), 0.1, 120, "姿态切换时间")
            updated = copy.deepcopy(self.store.document)
            updated["settings"] = {"speed_limit": speed, "pose_seconds": duration}
            updated["actions"][self.action.get()]["steps"] = [
                {"pose": pose.get(), "seconds": float(seconds.get()), "hold": float(hold.get())}
                for pose, seconds, hold in self.time_rows]
            self.store.persist(updated)
            self.notice.set("设置已保存" + ("（离线模拟，仅内存）。" if self.demo else "到 actions.json。"))
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.notice.set(str(exc))
            return False

    def capture(self):
        if self.playing:
            self.notice.set("请先暂停动作并等关节停稳。")
            return
        name = self.pose.get()
        if self.store.document["poses"][name]["values"] is not None:
            if not messagebox.askyesno("覆盖姿态", f"用当前实测位置覆盖 {name}？", parent=self.root):
                return
        self.mailbox.submit("capture", name)

    def play_pose(self, name=None):
        try:
            if self.change_speed() is None:
                return
            duration = finite_number(float(self.duration.get()), 0.1, 120, "切换时间")
            self.mailbox.submit("play", self.store.resolve(name or self.pose.get(), seconds=duration))
        except (ValueError, KeyError) as exc:
            self.notice.set(str(exc))

    def play_action(self, name=None):
        try:
            if not self.save_settings():
                return
            self.mailbox.submit("play", self.store.resolve(name or self.action.get(), is_action=True))
        except (ValueError, KeyError) as exc:
            self.notice.set(str(exc))

    def hold(self):
        self.mailbox.interrupt()
        self.notice.set("正在取消后续运动并保持当前位置……")

    def emergency(self):
        self.mailbox.interrupt(off=True)
        self.notice.set("正在关闭六只扭矩，请托稳机械臂。")

    def keydown(self, event):
        if event.keysym == "Escape":
            self.emergency()
            return "break"
        if event.widget.winfo_class() in ("TEntry", "Entry", "TCombobox", "TSpinbox", "Spinbox", "Text"):
            return
        if event.keysym in self.pressed:
            return "break"
        actions = {"1": lambda: self.play_pose("stand"), "2": lambda: self.play_action("wave"), "space": self.hold}
        if event.keysym in actions:
            self.pressed.add(event.keysym)
            actions[event.keysym]()
            return "break"

    def connect(self):
        if self.worker is not None and self.worker.is_alive():
            return
        self.mailbox = Mailbox()
        self.change_speed()
        for row in self.rows.values():
            row["initialized"] = False
        self.connect_button.configure(state="disabled")
        self.connection.set("正在读取六关节……")
        self.worker = SerialWorker(self.data, self.port, self.mailbox, self.output, demo=self.demo)
        self.worker.start()

    def show_snapshot(self, snapshot, playing, status):
        self.snapshot, self.playing = snapshot, playing
        self.status.set(status)
        self.updating = True
        try:
            for sid, state in snapshot.items():
                row = self.rows[sid]
                ratio = state["ratio"]
                margin = POSITION_TOLERANCE / self.data["joints"][str(sid)]["span_raw"]
                valid = state["online"] and ratio is not None and -margin <= ratio <= 1 + margin
                for widget in (row["slider"], row["entry"], row["button"]):
                    widget.configure(state="normal" if valid and not playing and not self.closing else "disabled")
                if valid:
                    row["progress"]["value"] = max(0, min(1, ratio))
                    target = state["target"] if state["target"] is not None else ratio
                    if not row["dragging"] and self.root.focus_get() != row["entry"]:
                        row["value"].set(max(0, min(1, target)))
                        row["entry_value"].set(f"{target:.4f}")
                    row["initialized"] = True
                    row["text"].set(f"实测 {ratio:.4f} / {state['raw']}  ·  {state['status']}")
                else:
                    row["initialized"] = False
                    row["text"].set(f"原始 {state['raw']} · {state['status']}")
        finally:
            self.updating = False
        self.capture_button.configure(state="disabled" if playing else "normal")

    def pump(self):
        try:
            while True:
                kind, value = self.output.get_nowait()
                if kind == "connected":
                    self.connection.set(value)
                elif kind == "snapshot":
                    self.show_snapshot(*value)
                elif kind in ("notice", "error"):
                    self.notice.set(value)
                elif kind == "capture":
                    name, raw = value
                    try:
                        self.store.record(name, raw)
                        self.refresh_pose()
                        self.notice.set(f"已记录 {name} 的六个实测值。" + ("离线模拟，仅内存。" if self.demo else "已写入 actions.json。"))
                    except (ValueError, OSError, RuntimeError) as exc:
                        self.notice.set(str(exc))
                elif kind == "closed":
                    if self.closing:
                        self.root.destroy()
                        return
                    self.connection.set("连接已关闭")
                    self.connect_button.configure(state="normal")
                    for row in self.rows.values():
                        row["initialized"] = False
                        for key in ("slider", "entry", "button"):
                            row[key].configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(50, self.pump)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.worker is None or not self.worker.is_alive():
            self.root.destroy()
            return
        self.notice.set("正在停止后续运动并保持已控制关节……")
        self.mailbox.closing.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="默认自动查找上次转接板，例如 COM9")
    parser.add_argument("--demo", action="store_true", help="离线模拟；不读写真实舵机或保存模拟姿态")
    parser.add_argument("--check-files", action="store_true", help="只检查配置，不连接硬件")
    args = parser.parse_args()
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8-sig"))
    store = ActionStore(data, memory_only=args.demo)
    if args.check_files:
        print("校准及动作文件匹配；ID1～ID6 的端点有效。")
        for name, pose in store.document["poses"].items():
            print(f"{name}: {'待实机录制' if pose['values'] is None else '已保存'}")
        return
    root = tk.Tk()
    FKWindow(root, data, store, port=args.port, demo=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
