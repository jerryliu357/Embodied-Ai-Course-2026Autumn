"""SCS215 backend extracted from the working lec3 controller.
No hardware is opened at import. Original joint mapping and raw-count conventions are preserved.
"""
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from serial.tools import list_ports

BAUDRATE = 1_000_000
ADAPTER_SERIAL = "5B90151486"
TORQUE_REGISTER = 40
LOCK_REGISTER = 48
POSITION_COUNT = 1024
DEFAULT_SPEED = 100
UPDATE_SECONDS = 0.05
ENDPOINT_TOLERANCE = 3
MAX_TARGET_LEAD = 64
POSITION_TOLERANCE = 6
REVERSE_TOLERANCE = 12
STALL_SECONDS = 5.0
MAX_SAMPLE_DELTA = 192
COMM_ERRORS = (OSError, RuntimeError)
JOINTS = {1: "底座旋转", 2: "肩部抬升", 3: "肘部弯曲",
          4: "手腕弯曲", 5: "手腕旋转", 6: "夹爪"}

def find_port():
    ports = list(list_ports.comports())
    matches = [p.device for p in ports if p.serial_number == ADAPTER_SERIAL]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(f"{p.device} ({p.description})" for p in ports)
    raise RuntimeError(f"未找到原 USB 转接板，可用 --port COM号码 指定。当前串口：{available or '无'}")



def inspect(bus, servo_id):
    info = bus.read(servo_id, 0, 9)
    # SCS 电位器版的字节序字段为 1；STS 磁编码版不能用本脚本写配置。
    if info[2] != 1:
        raise RuntimeError(f"字节序标识为 {info[2]}，与当前 SCS215 配置不符，停止写入。")
    if info[5] != servo_id or info[6] != 0:
        raise RuntimeError("编号或波特率寄存器与通信参数不一致")
    motion = bus.read(servo_id, 0x38, 8)
    return {
        "id": servo_id, "model_bytes": info[3:5].hex(" "),
        "position": int.from_bytes(motion[:2], "big"),
        "speed_raw": int.from_bytes(motion[2:4], "big"),
        "voltage_v": motion[6] / 10, "temperature_c": motion[7],
        "torque": bus.read(servo_id, TORQUE_REGISTER)[0],
        "lock": bus.read(servo_id, LOCK_REGISTER)[0],
    }



def validate_calibration(data):
    if (not isinstance(data, dict) or data.get("model") != "SCS215"
            or data.get("byte_order") != "big"
            or data.get("position_count") != POSITION_COUNT
            or data.get("range_capture_completed") is not True):
        raise ValueError("请先用 calibrate.py 完成当前 SCS215 的端点采集。")
    joints = data.get("joints")
    if not isinstance(joints, dict) or set(joints) != {str(sid) for sid in JOINTS}:
        raise ValueError("校准文件必须包含 ID 1～6 的端点。")
    for sid in JOINTS:
        item = joints[str(sid)]
        if not isinstance(item, dict):
            raise ValueError(f"ID={sid} 校准记录格式错误。")
        values = [item.get(key) for key in
                  ("zero_position", "one_position", "span_raw", "one_unwrapped_position")]
        if not all(type(value) is int for value in values):
            raise ValueError(f"ID={sid} 的端点、跨度必须是整数。")
        zero, one, span, end = values
        if (item.get("id") != sid or item.get("raw_direction") != 1
                or not 0 <= zero < POSITION_COUNT or not 0 <= one < POSITION_COUNT
                or not 0 < span < POSITION_COUNT or end != zero + span
                or end % POSITION_COUNT != one):
            raise ValueError(f"ID={sid} 的端点、正方向与跨度不一致。")



def calibration_fingerprint(data):
    """预设只对录制时使用的端点有效，避免范围变化后沿用旧比例。"""
    validate_calibration(data)
    basis = {
        "model": data["model"], "byte_order": data["byte_order"],
        "position_count": data["position_count"],
        "joints": [[sid, data["joints"][str(sid)]["zero_position"],
                    data["joints"][str(sid)]["one_position"],
                    data["joints"][str(sid)]["span_raw"]] for sid in JOINTS],
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()



def raw_to_ratio(raw, joint):
    """在已记录的正向区间内反算比例；不把区间外的姿态随意截断。"""
    if type(raw) is not int or not 0 <= raw < POSITION_COUNT:
        raise ValueError("原始位置必须是 0～1023 的整数。")
    zero, span = joint["zero_position"], joint["span_raw"]
    delta = (raw - zero) % POSITION_COUNT
    if delta <= span:
        return delta / span, None
    distance_before_zero = POSITION_COUNT - delta
    distance_after_one = delta - span
    if min(distance_before_zero, distance_after_one) <= ENDPOINT_TOLERANCE:
        value = 0.0 if distance_before_zero < distance_after_one else 1.0
        return value, {
            "measured_raw": raw, "target_raw": ratio_to_raw(value, joint),
            "adjustment_counts": min(distance_before_zero, distance_after_one),
            "reason": "endpoint_reading_tolerance",
        }
    raise ValueError(
        f"ID={joint['id']} 当前位置 {raw} 超出你记录的 0→1 区间 "
        f"({zero}→{joint['one_position']})。请调整该关节，已有端点不会改变。"
    )



def ratio_to_raw(value, joint):
    return target_position(value, joint) % POSITION_COUNT


class LeRobotSCSBus:
    """官方 LeRobot SCS 通信层；只适配型号验证、原始计数和连续寄存器读取。

    不调用 SOFollower.configure()：其 STS3215 默认寄存器不适用于 SCS215。
    不套用官方角度归一化：用户的正向跨界端点由本文件单独换算。
    """
    def __init__(self, port):
        # 在串口线程中加载，避免 LeRobot 首次导入阻塞窗口。
        from driver_loader import load_driver
        load_driver()
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

