"""六滑条控制 SCS215，使用 lerobot 环境运行。

启动只读六只舵机；滑条从实测比例开始，鼠标拖动才控制对应舵机。
直接使用 LeRobot FeetechMotorsBus 的 read / sync_write：位置环在舵机内。
不直接使用 STS3215 的配置，也不将 SCS 未确认的地址 23 当作积分参数写入。
沿用 calibration.json 的六组端点，不修改该文件。

普通目标连续限速更新，不设置六关节一起到位的等待关卡。
数值跨界仍按本次校准展开；固件反向或实物停滞时只停止有问题的关节。
关闭窗口停止后续目标，并让本程序控制过的舵机保持实测位置。
Esc / 全部卸力会关闭六只扭矩；请托住机械臂。
"""
import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from record_poses import raw_to_ratio, validate_calibration
from set_id import JOINTS, LOCK_REGISTER, TORQUE_REGISTER, find_port, inspect

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")
POSITION_COUNT = 1024
DEFAULT_SPEED = 100
UPDATE_SECONDS = 0.05
MAX_TARGET_LEAD = 64       # 目标最多领先实测位置 64 个计数，不等所有关节到位
POSITION_TOLERANCE = 6
REVERSE_TOLERANCE = 12
STALL_SECONDS = 5.0
MAX_SAMPLE_DELTA = 192
COMM_ERRORS = (OSError, RuntimeError)  # SerialException 继承 OSError


class LeRobotSCSBus:
    """官方 LeRobot SCS 通信层；只适配型号验证、原始计数和连续寄存器读取。

    不调用 SOFollower.configure()：其 STS3215 默认寄存器不适用于 SCS215。
    不套用官方角度归一化：用户的正向跨界端点由本文件单独换算。
    """
    def __init__(self, port):
        # 在串口线程中加载，避免 LeRobot 首次导入阻塞窗口。
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        self.names = {sid: f"id{sid}" for sid in JOINTS}
        motors = {name: Motor(sid, "scs_series", MotorNormMode.RANGE_0_100)
                  for sid, name in self.names.items()}
        self.bus = FeetechMotorsBus(port, motors, protocol_version=1)
        self.registers = {spec: name for name, spec in
                          self.bus.model_ctrl_table["scs_series"].items()}
        self.speeds = {}
        try:
            # 通用 SCS 表没有 SCS215 型号条目，因此由 initialize 逐只验证 05 23。
            # connect/set_baudrate 只打开并配置电脑串口，不改变电机配置。
            self.bus.connect(handshake=False)
            self.bus.set_baudrate(1_000_000)
        except Exception:
            # 串口被另一个窗口占用时，SDK 的 ser 可能尚未创建。
            if self.bus.port_handler.ser is not None:
                self.bus.port_handler.closePort()
            raise

    def read(self, sid, address, length=1):
        name = self.registers.get((address, length))
        if name is not None:
            value = self.bus.read(name, self.names[sid], normalize=False)
            return int(value).to_bytes(length, "big")
        # SCS 不支持 sync_read；连续状态块通过同一官方 SDK 连接读取。
        values, comm, error = self.bus.packet_handler.readTxRx(
            self.bus.port_handler, sid, address, length)
        if comm != 0 or error or len(values) != length:
            raise RuntimeError(f"ID{sid} 读取 {address:#x} 失败：通信={comm}，状态={error:#x}")
        return bytes(values)

    def _write_register(self, sid, address, length, value):
        name = self.registers[(address, length)]
        self.bus.sync_write(name, {self.names[sid]: value}, normalize=False)
        if self.read(sid, address, length) != value.to_bytes(length, "big"):
            raise RuntimeError(f"ID{sid} {name} 写入后读回不一致")

    def write_checked(self, sid, address, value):
        self._write_register(sid, address, 1, value)

    def write_limits(self, sid):
        self._write_register(sid, 9, 2, 0)
        self._write_register(sid, 11, 2, 1023)

    def write_position(self, sid, raw, speed):
        # 与官方 send_action 相同，用 Goal_Position 发目标。只包含当前选中的 ID。
        if self.speeds.get(sid) != speed:
            self._write_register(sid, 44, 2, 0)
            self._write_register(sid, 46, 2, speed)
            self.speeds[sid] = speed
        self._write_register(sid, 42, 2, raw)

    def off_all(self):
        self.bus.sync_write("Torque_Enable", {name: 0 for name in self.names.values()},
                            normalize=False)

    def close(self):
        # 保持/卸力由控制器明确执行，关闭连接本身不碰其余关节。
        self.bus.disconnect(disable_torque=False)


def circular_delta(current, previous):
    return (current - previous + 512) % POSITION_COUNT - 512


def raw_to_unwrapped(raw, joint):
    ratio, adjustment = raw_to_ratio(raw, joint)
    value = round(joint["zero_position"] + ratio * joint["span_raw"])
    if adjustment:
        value += circular_delta(raw, adjustment["target_raw"])
    return ratio, value


def target_position(value, joint):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("请输入 0～1 的数值。")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("目标比例必须在 0～1 之间。")
    return round(joint["zero_position"] + value * joint["span_raw"])


def write_position(bus, sid, raw, speed):
    if type(raw) is not int or not 0 <= raw < POSITION_COUNT:
        raise ValueError("目标原始位置必须为 0～1023。")
    if type(speed) is not int or not 1 <= speed <= 1000:
        raise ValueError("速度必须为 1～1000 的整数。")
    bus.write_position(sid, raw, speed)


@dataclass
class JointState:
    sid: int
    calibration: dict
    online: bool = False
    raw: int = 0
    position: int = 0
    ratio: float | None = None
    desired: int | None = None
    command: float = 0.0
    torque: int = 0
    voltage: float = 0.0
    load_percent: float = 0.0
    p: int = 0
    d: int = 0
    limits: tuple = (0, 1023)
    active: bool = False
    owned: bool = False
    status: str = "等待连接"
    last_sent: int | None = None
    anchor: int = 0
    direction: int = 0
    progress_position: int = 0
    progress_time: float = 0.0
    tick_time: float = 0.0
    samples: deque = field(default_factory=deque)

    def observe(self, raw, now):
        if not 0 <= raw < POSITION_COUNT:
            raise RuntimeError(f"ID{self.sid} 返回了异常位置 {raw}")
        if self.active or self.owned:
            delta = circular_delta(raw, self.raw)
            if abs(delta) > MAX_SAMPLE_DELTA:
                raise RuntimeError(f"ID{self.sid} 反馈跳变过大：{delta}")
            self.position += delta
            self.ratio = (self.position - self.calibration["zero_position"]) / self.calibration["span_raw"]
        else:
            try:
                self.ratio, self.position = raw_to_unwrapped(raw, self.calibration)
            except ValueError:
                self.ratio = None
                self.position = raw
        self.raw = raw
        self.samples.append((now, self.position))
        while len(self.samples) > 1 and self.samples[1][0] <= now - 0.25:
            self.samples.popleft()

    def settled(self, now):
        values = [value for _, value in self.samples]
        return (len(values) >= 3 and now - self.samples[0][0] >= 0.25
                and max(values) - min(values) <= 3)


class ArmController:
    """单线程串口后端；UI 线程只交换消息，绝不直接访问串口。"""
    def __init__(self, bus, data, cancelled=lambda: False):
        validate_calibration(data)
        self.bus = bus
        self.cancelled = cancelled
        self.joints = {sid: JointState(sid, data["joints"][str(sid)]) for sid in JOINTS}

    def initialize(self, now):
        # 没有任何 WRITE；不会因界面滑条初始化而使能扭矩或发送目标。
        for sid, state in self.joints.items():
            if self.cancelled():
                break
            try:
                info = inspect(self.bus, sid)
                if info["model_bytes"] != "05 23":
                    raise RuntimeError("型号不匹配，当前程序用于 SCS215 / 05 23")
                limits = self.bus.read(sid, 9, 4)
                gains = self.bus.read(sid, 21, 2)
                state.limits = (int.from_bytes(limits[:2], "big"), int.from_bytes(limits[2:], "big"))
                state.p, state.d = gains
                state.torque, state.voltage = info["torque"], info["voltage_v"]
                state.observe(info["position"], now)
                state.online = True
                state.command = state.position
                state.status = "只读 · 拖动后控制" if state.ratio is not None else "当前位置超出校准范围"
            except COMM_ERRORS as exc:
                state.status = str(exc)

    def _full_range(self, state):
        if state.limits == (0, 1023):
            return
        # 仅解除用户当前操作的这只舵机的旧限位，不卸力其他关节。
        self.bus.write_checked(state.sid, TORQUE_REGISTER, 0)
        state.torque = 0
        try:
            self.bus.write_checked(state.sid, LOCK_REGISTER, 0)
            self.bus.write_limits(state.sid)
            time.sleep(0.05)
            if self.bus.read(state.sid, 9, 4) != b"\0\0\x03\xff":
                raise RuntimeError("解除旧角度限位后的读回结果不一致")
        finally:
            self.bus.write_checked(state.sid, LOCK_REGISTER, 1)
        state.limits = (0, 1023)

    def set_target(self, sid, value, speed, now):
        state = self.joints[sid]
        target = target_position(value, state.calibration)
        if not state.online:
            raise ValueError(f"ID{sid} 未连接，请重新连接后再试。")
        if state.ratio is None or not (state.calibration["zero_position"] - POSITION_TOLERANCE
                                      <= state.position <=
                                      state.calibration["one_unwrapped_position"] + POSITION_TOLERANCE):
            raise ValueError(f"ID{sid} 当前位置不在记录范围内。")
        if self.cancelled():
            return
        if state.torque not in (0, 1):
            raise ValueError(f"ID{sid} 当前扭矩状态 {state.torque} 不是普通位置控制状态。")
        # 新的鼠标事件才会激活。目标替换而非排队，松开后不会回放旧的拖动路径。
        if not state.owned:
            self._full_range(state)
            motion = self.bus.read(sid, 56, 8)
            state.observe(int.from_bytes(motion[:2], "big"), now)
            if state.ratio is None:
                raise ValueError(f"ID{sid} 准备期间位置移出了校准范围。")
            if self.cancelled():
                return
            # 先覆盖旧目标再开启该关节。其他五只没有任何写指令。
            state.owned = True  # 目标或使能应答失败时也必须尝试关闭该关节
            write_position(self.bus, sid, state.raw, speed)
            state.last_sent = state.position
            if self.cancelled():
                return
            self.bus.write_checked(sid, TORQUE_REGISTER, 1)
            state.torque = 1
        direction = (target > state.position) - (target < state.position)
        if not state.active or direction != state.direction:
            state.progress_time = now
            state.progress_position = state.position
            state.anchor = state.position
        state.direction = direction
        state.command = float(state.position)
        state.desired = target
        state.tick_time = now
        state.active = True
        state.status = "正在跟随滑条"

    def poll(self, now):
        for sid, state in self.joints.items():
            if self.cancelled():
                return
            if not state.online:
                continue
            try:
                motion = self.bus.read(sid, 56, 8)
                torque = self.bus.read(sid, TORQUE_REGISTER)[0]
                state.observe(int.from_bytes(motion[:2], "big"), now)
                state.voltage = motion[6] / 10
                state.load_percent = (int.from_bytes(motion[4:6], "big") & 0x3FF) / 10
                state.torque = torque
                if state.owned and torque != 1:
                    state.active = state.owned = False
                    state.status = "舵机扭矩已关闭；再次拖动可重新控制"
                if state.ratio is None and not state.owned:
                    state.status = "当前位置超出校准范围"
            except COMM_ERRORS as exc:
                self.fault(sid, str(exc))
                state.online = False

    def advance(self, speed, now):
        for sid, state in self.joints.items():
            if self.cancelled():
                return
            if not state.active:
                continue
            try:
                low = state.calibration["zero_position"]
                high = state.calibration["one_unwrapped_position"]
                if not low - POSITION_TOLERANCE <= state.position <= high + POSITION_TOLERANCE:
                    raise RuntimeError("实际位置超出本次校准范围，已停止该关节")
                if state.direction * (state.position - state.progress_position) < -REVERSE_TOLERANCE:
                    raise RuntimeError("实际运动与目标方向相反，已停止该关节（可能是固件跨界方向不同）")
                remaining = state.desired - state.position
                same_turn = state.position // 1024 == state.desired // 1024
                reached = abs(remaining) <= POSITION_TOLERANCE and same_turn
                if state.direction * (state.position - state.progress_position) >= 3:
                    state.progress_position, state.progress_time = state.position, now
                elif not reached and now - state.progress_time >= STALL_SECONDS:
                    raise RuntimeError(f"停在原始 {state.raw}，目标 {state.desired % 1024}，"
                                       f"误差 {abs(remaining)}；该关节 5 秒无进展，已卸力")
                elapsed = max(0.0, min(now - state.tick_time, 0.1))
                state.tick_time = now
                step = speed * elapsed
                difference = state.desired - state.command
                state.command += max(-step, min(step, difference))
                # 独立限速 + 限制目标超前量，不等待某个中间点精确到位。
                state.command = max(state.position - MAX_TARGET_LEAD,
                                    min(state.position + MAX_TARGET_LEAD, state.command))
                state.command = max(low, min(high, state.command))
                command = round(state.command)
                if command != state.last_sent:
                    write_position(self.bus, sid, command % POSITION_COUNT, speed)
                    state.last_sent = command
                if command == state.desired and reached and state.settled(now):
                    state.active = False
                    state.status = "已到位 · 保持"
                else:
                    state.status = f"跟随中 · 误差 {abs(remaining)} 个计数"
            except COMM_ERRORS as exc:
                self.fault(sid, str(exc))

    def fault(self, sid, reason):
        state = self.joints[sid]
        was_owned = state.owned
        state.active = state.owned = False
        state.last_sent = None
        if was_owned:
            try:
                self.bus.write_checked(sid, TORQUE_REGISTER, 0)
                state.torque = 0
            except COMM_ERRORS as exc:
                reason += f"；卸力未确认，请托住并断电：{exc}"
        state.status = reason

    def off(self, ids):
        # 先取消所有待运动状态，再逐只关闭；单只失败也继续处理其余关节。
        ids = list(ids)
        for sid in ids:
            self.joints[sid].active = self.joints[sid].owned = False
            self.joints[sid].last_sent = None
        if set(ids) == set(JOINTS):
            try:
                self.bus.off_all()
            except COMM_ERRORS:
                pass
        for sid in ids:
            state = self.joints[sid]
            try:
                self.bus.write_checked(sid, TORQUE_REGISTER, 0)
                state.torque = 0
                state.desired = None
                state.status = "已卸力 · 托稳后可重新拖动"
            except COMM_ERRORS as exc:
                state.status = f"卸力未确认，请托住并断电：{exc}"

    def hold(self, ids, speed):
        for sid in ids:
            state = self.joints[sid]
            if not state.owned:
                continue
            state.active = False
            try:
                raw = int.from_bytes(self.bus.read(sid, 56, 2), "big")
                write_position(self.bus, sid, raw, speed)
                state.desired = state.position + circular_delta(raw, state.raw)
                state.command = float(state.desired)
                state.last_sent = state.desired
                state.status = "已暂停 · 保持当前位置"
            except COMM_ERRORS as exc:
                self.fault(sid, str(exc))

    def snapshot(self):
        return {sid: {"raw": s.raw, "ratio": s.ratio, "online": s.online,
                      "target": None if s.desired is None else
                      (s.desired - s.calibration["zero_position"]) / s.calibration["span_raw"],
                      "torque": s.torque, "voltage": s.voltage, "load": s.load_percent,
                      "p": s.p, "d": s.d, "status": s.status,
                      "active": s.active, "owned": s.owned}
                for sid, s in self.joints.items()}


class Mailbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.targets = {}
        self.actions = deque()
        self.speed = DEFAULT_SPEED
        self.emergency = threading.Event()
        self.closing = threading.Event()

    def target(self, sid, value):
        with self.lock:
            if not self.emergency.is_set() and not self.closing.is_set():
                self.targets[sid] = value

    def stop_all(self):
        with self.lock:
            self.targets.clear()
            self.actions.clear()
            self.emergency.set()

    def action(self, name, sid):
        with self.lock:
            if self.emergency.is_set() or self.closing.is_set():
                return
            self.targets.pop(sid, None)
            self.actions.append((name, sid))

    def take(self):
        with self.lock:
            result = (dict(self.targets), list(self.actions), self.speed)
            self.targets.clear()
            self.actions.clear()
            return result


class SerialWorker(threading.Thread):
    def __init__(self, data, port, mailbox, output, bus_factory=LeRobotSCSBus):
        super().__init__(daemon=True, name="SCS215-serial")
        self.data, self.port = data, port
        self.mailbox, self.output, self.bus_factory = mailbox, output, bus_factory

    def emit(self, kind, value):
        self.output.put((kind, value))

    def run(self):
        bus = controller = None
        try:
            port = self.port or find_port()
            bus = self.bus_factory(port)
            controller = ArmController(bus, self.data, lambda:
                                       self.mailbox.emergency.is_set() or self.mailbox.closing.is_set())
            controller.initialize(time.monotonic())
            self.emit("connected", port)
            self.emit("snapshot", controller.snapshot())
            while not self.mailbox.closing.is_set():
                if self.mailbox.emergency.is_set():
                    self.mailbox.take()  # 丢弃卸力前尚未执行的鼠标目标
                    controller.off(JOINTS)
                    self.mailbox.emergency.clear()
                    self.emit("snapshot", controller.snapshot())
                    self.emit("notice", "已执行全部卸力，请托稳机械臂。")
                    continue
                began = time.monotonic()
                controller.poll(began)
                if self.mailbox.emergency.is_set() or self.mailbox.closing.is_set():
                    continue
                targets, actions, speed = self.mailbox.take()
                for action, sid in actions:
                    if self.mailbox.emergency.is_set() or self.mailbox.closing.is_set():
                        break
                    if action == "off":
                        controller.off([sid])
                    elif action == "hold":
                        controller.hold([sid], speed)
                for sid, value in targets.items():
                    try:
                        controller.set_target(sid, value, speed, time.monotonic())
                    except ValueError as exc:
                        controller.joints[sid].status = str(exc)
                    except COMM_ERRORS as exc:
                        controller.fault(sid, str(exc))
                controller.advance(speed, time.monotonic())
                self.emit("snapshot", controller.snapshot())
                self.mailbox.closing.wait(max(0.0, UPDATE_SECONDS - (time.monotonic() - began)))
            if self.mailbox.emergency.is_set():
                controller.off(JOINTS)
            else:
                controller.hold(JOINTS, self.mailbox.speed)
        except Exception as exc:
            if controller is not None:
                controller.off([sid for sid, state in controller.joints.items() if state.owned])
            self.emit("error", f"连接/控制失败：{exc}")
        finally:
            try:
                if bus is not None:
                    bus.close()
            finally:
                self.emit("closed", None)


class ArmWindow:
    def __init__(self, root, data, port=None, autostart=True, worker_factory=SerialWorker):
        self.root, self.data, self.port = root, data, port
        self.worker_factory = worker_factory
        self.worker = None
        self.mailbox = Mailbox()
        self.output = queue.Queue()
        self.rows = {}
        self.closing = False
        self.reconnecting = False
        self.updating = False
        root.title("SCS215 · 六关节滑条控制")
        root.geometry("1120x850")
        root.minsize(920, 640)
        root.configure(bg="#f3f6fb")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fb")
        style.configure("TLabel", background="#f3f6fb", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 19, "bold"), foreground="#172b4d")
        style.configure("Joint.TLabel", font=("Microsoft YaHei UI", 11, "bold"), foreground="#172b4d")
        style.configure("Actual.Horizontal.TProgressbar", background="#16a085", troughcolor="#e0e9ef")
        style.configure("Target.Horizontal.TScale", background="#3478f6", troughcolor="#d8e4f3")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(9, 5))
        outer = ttk.Frame(root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="六关节滑条控制", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="蓝色滑条设目标，绿色条显示实测。拖动哪只，只控制哪只；启动时不会复位。"
                  ).pack(anchor="w", pady=(5, 12))
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 12))
        self.connection = tk.StringVar(value="等待连接")
        ttk.Label(bar, textvariable=self.connection).pack(side="left")
        self.connect_button = ttk.Button(bar, text="重新连接", command=self.connect)
        self.connect_button.pack(side="right")
        ttk.Label(bar, text="目标变化速度（计数/秒）").pack(side="left", padx=(25, 5))
        self.speed = tk.StringVar(value=str(DEFAULT_SPEED))
        speed_box = ttk.Spinbox(bar, from_=1, to=1000, textvariable=self.speed, width=6,
                               command=self.change_speed)
        speed_box.pack(side="left")
        speed_box.bind("<Return>", self.change_speed)
        speed_box.bind("<FocusOut>", self.change_speed)
        # 关节区可滚动，小屏幕/高 DPI 下底部卸力按钮仍始终可见。
        joints_area = ttk.Frame(outer)
        canvas = tk.Canvas(joints_area, bg="#f3f6fb", highlightthickness=0, height=600)
        scrollbar = ttk.Scrollbar(joints_area, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        joint_rows = ttk.Frame(canvas)
        inner = canvas.create_window(0, 0, anchor="nw", window=joint_rows)
        joint_rows.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(inner, width=event.width))
        for sid, name in JOINTS.items():
            self.make_row(joint_rows, sid, name)
        bottom = ttk.Frame(outer)
        bottom.pack(side="bottom", fill="x", pady=(12, 5))
        tk.Button(bottom, text="全部卸力  ·  Esc", command=self.emergency,
                  bg="#c83945", fg="white", activebackground="#a92430", activeforeground="white",
                  font=("Microsoft YaHei UI", 11, "bold"), relief="flat", padx=18, pady=8
                  ).pack(side="left")
        ttk.Button(bottom, text="退出并保持当前位置", command=self.close).pack(side="right")
        self.notice = tk.StringVar(value="正在准备。首次拖动会开启对应舵机扭矩；卸力时请托住机械臂。")
        ttk.Label(outer, textvariable=self.notice, wraplength=840).pack(side="bottom", anchor="w", pady=5)
        ttk.Label(outer, text="使用 LeRobot 飞特驱动。位置控制在舵机内部执行；P / D 仅显示当前值。",
                  foreground="#607089").pack(side="bottom", anchor="w", pady=(2, 0))
        joints_area.pack(fill="both", expand=True)
        root.bind("<Escape>", self.emergency)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self.pump)
        if autostart:
            root.after(150, self.connect)

    def make_row(self, parent, sid, name):
        box = ttk.Frame(parent, padding=(0, 6, 0, 7))
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text=f"ID {sid}  {name}", style="Joint.TLabel", width=19).grid(row=0, column=0, sticky="w")
        value = tk.DoubleVar(value=0)
        slider = ttk.Scale(box, from_=0, to=1, variable=value, state="disabled", style="Target.Horizontal.TScale",
                           command=lambda val, i=sid: self.drag(i, val))
        slider.grid(row=0, column=1, sticky="ew", padx=(8, 16))
        entry_value = tk.StringVar(value="—")
        entry = ttk.Entry(box, textvariable=entry_value, width=8, justify="center", state="disabled")
        entry.grid(row=0, column=2, padx=(0, 5))
        apply_button = ttk.Button(box, text="设置", command=lambda i=sid: self.apply_entry(i), state="disabled")
        apply_button.grid(row=0, column=3)
        ttk.Button(box, text="暂停保持", command=lambda i=sid: self.mailbox.action("hold", i)).grid(row=0, column=4, padx=5)
        ttk.Button(box, text="卸力", command=lambda i=sid: self.mailbox.action("off", i)).grid(row=0, column=5)
        actual = tk.StringVar(value="实测 —")
        ttk.Label(box, textvariable=actual, foreground="#087f70").grid(row=1, column=0, sticky="w", pady=(4, 0))
        progress = ttk.Progressbar(box, style="Actual.Horizontal.TProgressbar", maximum=1)
        progress.grid(row=1, column=1, sticky="ew", padx=(8, 16), pady=(4, 0))
        status = tk.StringVar(value="等待连接")
        ttk.Label(box, textvariable=status).grid(row=2, column=0, columnspan=6, sticky="w", pady=(3, 0))
        self.rows[sid] = {"value": value, "slider": slider, "entry": entry,
                          "entry_value": entry_value, "apply": apply_button,
                          "actual": actual, "progress": progress, "status": status,
                          "initialized": False, "dragging": False}
        slider.bind("<ButtonPress-1>", lambda event, i=sid: self.begin_drag(i))
        slider.bind("<ButtonRelease-1>", lambda event, i=sid: self.end_drag(i))
        slider.bind("<Left>", lambda event, i=sid: self.nudge(i, -0.01))
        slider.bind("<Right>", lambda event, i=sid: self.nudge(i, 0.01))
        entry.bind("<Return>", lambda event, i=sid: self.apply_entry(i))

    def connect(self):
        if self.worker is not None and self.worker.is_alive():
            self.reconnecting = True
            self.connect_button.configure(state="disabled")
            self.notice.set("正在暂停当前运动并重新连接……")
            self.mailbox.closing.set()
            return
        self.reconnecting = False
        self.mailbox = Mailbox()
        self.change_speed()
        for row in self.rows.values():
            row["initialized"] = False
            for widget in (row["slider"], row["entry"], row["apply"]):
                widget.configure(state="disabled")
        self.connection.set("正在读取六个当前位置……")
        self.connect_button.configure(state="disabled")
        self.worker = self.worker_factory(self.data, self.port, self.mailbox, self.output)
        self.worker.start()

    def change_speed(self, event=None):
        try:
            value = int(self.speed.get())
            if not 1 <= value <= 1000:
                raise ValueError
        except ValueError:
            self.speed.set(str(self.mailbox.speed))
            return
        with self.mailbox.lock:
            self.mailbox.speed = value

    def begin_drag(self, sid):
        self.rows[sid]["dragging"] = self.rows[sid]["initialized"]

    def drag(self, sid, value):
        row = self.rows[sid]
        if self.updating or not row["dragging"]:
            return  # 程序设置初值绝不能触发运动
        value = float(value)
        row["entry_value"].set(f"{value:.4f}")
        self.mailbox.target(sid, value)

    def end_drag(self, sid):
        row = self.rows[sid]
        if row["dragging"]:
            self.drag(sid, row["value"].get())
        row["dragging"] = False

    def apply_entry(self, sid):
        row = self.rows[sid]
        if not row["initialized"]:
            return
        try:
            value = float(row["entry_value"].get())
            target_position(value, self.data["joints"][str(sid)])
        except ValueError:
            self.notice.set(f"ID{sid}：请输入 0～1 的有限数字。")
            return
        self.updating = True
        row["value"].set(value)
        self.updating = False
        self.mailbox.target(sid, value)

    def nudge(self, sid, step):
        row = self.rows[sid]
        if row["initialized"]:
            row["entry_value"].set(f"{max(0.0, min(1.0, row['value'].get() + step)):.4f}")
            self.apply_entry(sid)
        return "break"

    def show_snapshot(self, snapshot):
        self.updating = True
        try:
            for sid, state in snapshot.items():
                row = self.rows[sid]
                ratio = state["ratio"]
                margin = POSITION_TOLERANCE / self.data["joints"][str(sid)]["span_raw"]
                available = state["online"] and ratio is not None and -margin <= ratio <= 1 + margin
                enabled = available and not self.closing
                for widget in (row["slider"], row["entry"], row["apply"]):
                    widget.configure(state="normal" if enabled else "disabled")
                if available and not row["initialized"]:
                    initial = max(0.0, min(1.0, ratio))
                    row["value"].set(initial)
                    row["entry_value"].set(f"{initial:.4f}")
                    row["initialized"] = True
                if not state["online"]:
                    row["actual"].set("未连接")
                elif ratio is None:
                    row["actual"].set(f"范围外 · 原始 {state['raw']}")
                else:
                    row["actual"].set(f"实测 {ratio:.4f}  ·  {state['raw']}")
                    row["progress"]["value"] = max(0.0, min(1.0, ratio))
                target = "" if state["target"] is None else f"目标 {state['target']:.4f}  |  "
                row["status"].set(f"{target}{state['status']}   ·   {state['voltage']:.1f} V"
                                  f"   ·   输出 {state['load']:.1f}%   ·   P {state['p']} / D {state['d']}")
        finally:
            self.updating = False

    def emergency(self, event=None):
        self.mailbox.stop_all()
        self.notice.set("正在关闭六只扭矩，请托稳机械臂。")

    def pump(self):
        try:
            while True:
                kind, value = self.output.get_nowait()
                if kind == "snapshot":
                    self.show_snapshot(value)
                elif kind == "connected":
                    self.connection.set(f"LeRobot · {value} · 1 Mbps")
                    self.connect_button.configure(state="normal")
                    self.notice.set("滑条已从实测位置初始化。拖动哪只，只控制哪只；其他关节扭矩状态保持原样。")
                elif kind in ("notice", "error"):
                    self.notice.set(value)
                elif kind == "closed":
                    self.connect_button.configure(state="normal")
                    self.connection.set("串口已关闭")
                    if self.closing:
                        self.root.destroy()
                        return
                    for row in self.rows.values():
                        row["slider"].configure(state="disabled")
                        row["entry"].configure(state="disabled")
                        row["apply"].configure(state="disabled")
                    if self.reconnecting:
                        # closed 消息在工作线程 finally 的末尾；短暂等待线程退出。
                        self.root.after(100, self.connect)
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
        self.notice.set("正在取消后续运动，让已控制的关节保持当前位置……")
        self.mailbox.closing.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="默认自动找到原串口板，例如 --port COM9")
    args = parser.parse_args()
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    validate_calibration(data)
    root = tk.Tk()
    window = ArmWindow(root, data, args.port)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        window.mailbox.stop_all()
    finally:
        window.mailbox.closing.set()
        if window.worker is not None:
            window.worker.join(timeout=10)


if __name__ == "__main__":
    main()
