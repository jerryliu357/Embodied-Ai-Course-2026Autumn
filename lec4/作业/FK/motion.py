"""多关节时间插值、动作队列与串口线程。使用 lec3 控制器的实测反馈。"""
from collections import deque
import copy
import queue
import threading
import time

from arm_backend import (ArmController, LeRobotSCSBus, JOINTS, find_port, target_position,
                         POSITION_TOLERANCE, MAX_TARGET_LEAD, REVERSE_TOLERANCE, STALL_SECONDS,
                         UPDATE_SECONDS, circular_delta, write_position)
from action_store import finite_number, pose_values


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class MotionEngine:
    def __init__(self, controller, notify=lambda text: None):
        self.c, self.notify = controller, notify
        self.steps = []
        self.index = 0
        self.segment = None
        self.speed = 100
        self.status = "启动只读取位置；选择姿态或拖动滑条后才运动。"

    @property
    def playing(self):
        return bool(self.steps)

    def ready(self):
        for sid, s in self.c.joints.items():
            if not s.online or s.ratio is None or not -POSITION_TOLERANCE <= s.position - s.calibration["zero_position"] <= s.calibration["span_raw"] + POSITION_TOLERANCE:
                raise ValueError(f"ID{sid} 未连接或当前位置不在校准范围内。")
            if s.torque not in (0, 1):
                raise ValueError(f"ID{sid} 当前不是普通位置控制模式。")

    def play(self, steps, speed, now):
        if self.playing:
            raise ValueError("动作正在执行；先按空格暂停，再选择新动作。")
        if not steps:
            raise ValueError("动作没有步骤。")
        finite_number(speed, 1, 1000, "速度上限")
        if type(speed) is not int:
            raise ValueError("速度上限必须为整数。")
        # 整个动作一次性检查，不能走到一半才发现后面的姿态尚未录制。
        for step in steps:
            pose_values(step["values"])
            finite_number(step["seconds"], 0.1, 120, "运动时间")
            finite_number(step.get("hold", 0), 0, 60, "停留时间")
        self.ready()
        self.c.hold(JOINTS, speed)
        self.speed, self.steps, self.index = speed, copy.deepcopy(steps), 0
        try:
            self._begin(now)
        except Exception:
            self.stop()
            raise

    def _begin(self, now):
        self.ready()
        step = self.steps[self.index]
        # set_target 先将旧目标覆盖为实测位置，再使能；六个关节都准备完才开始计时。
        for sid, value in zip(JOINTS, step["values"]):
            if self.c.cancelled():
                return
            self.c.set_target(sid, value, self.speed, now)
        starts = {sid: s.position for sid, s in self.c.joints.items()}
        ends = {sid: s.desired for sid, s in self.c.joints.items()}
        # 三次 smoothstep 的峰值斜率为 1.5，保证命令变化速度不超过上限。
        minimum = 1.5 * max(abs(ends[sid] - starts[sid]) for sid in JOINTS) / self.speed
        duration = max(step["seconds"], minimum)
        self.segment = {"starts": starts, "ends": ends, "duration": duration,
                        "elapsed": 0.0, "last": None, "arrived": None,
                        "settle_deadline": None}
        for s in self.c.joints.values():
            s.progress_time, s.progress_position = now, s.position
        extended = f"（请求 {step['seconds']:g}s，按速度上限延长）" if duration > step["seconds"] + 0.01 else ""
        self.status = f"第 {self.index + 1}/{len(self.steps)} 段：{step['pose']}，规划 {duration:.2f}s{extended}"
        self.notify(self.status)

    def stop(self, off=False):
        self.steps, self.segment = [], None
        if off:
            self.c.off(JOINTS)
            self.status = "已全部卸力；动作队列已清空。"
        else:
            self.c.hold(JOINTS, self.speed)
            self.status = "已暂停并保持当前位置；再次播放从当前位置重新开始。"

    def _check(self, now):
        for sid, s in self.c.joints.items():
            if not s.online or not s.owned or s.torque != 1:
                raise RuntimeError(f"ID{sid} 控制中断：{s.status}")
            low, high = s.calibration["zero_position"], s.calibration["one_unwrapped_position"]
            if not low - POSITION_TOLERANCE <= s.position <= high + POSITION_TOLERANCE:
                self.c.fault(sid, "实际位置超出校准范围")
                raise RuntimeError(f"ID{sid} 实际位置超出校准范围。")
            if s.direction * (s.position - s.progress_position) < -REVERSE_TOLERANCE:
                self.c.fault(sid, "实际运动方向与目标相反")
                raise RuntimeError(f"ID{sid} 实际运动方向与目标相反。")
            if abs(s.command - s.position) <= POSITION_TOLERANCE:
                s.progress_time, s.progress_position = now, s.position
            elif s.direction * (s.position - s.progress_position) >= 3:
                s.progress_time, s.progress_position = now, s.position
            elif now - s.progress_time >= STALL_SECONDS:
                self.c.fault(sid, "持续停滞，已卸力该关节")
                raise RuntimeError(f"ID{sid} 持续 {STALL_SECONDS:g} 秒未跟上目标。")

    def tick(self, now, manual_speed):
        if not self.playing:
            self.c.advance(manual_speed, now)
            return
        if self.c.cancelled():
            return
        try:
            self._check(now)
            segment = self.segment
            dt = 0 if segment["last"] is None else max(0, min(0.1, now - segment["last"]))
            segment["last"] = now
            # 串口或实机跟踪滞后时暂停整条轨迹的时钟，不跳过中间路径。
            candidate = min(segment["duration"], segment["elapsed"] + dt)
            alpha = smoothstep(candidate / segment["duration"])
            commands = {sid: round(start + alpha * (segment["ends"][sid] - start))
                        for sid, start in segment["starts"].items()}
            if all(abs(commands[sid] - self.c.joints[sid].position) <= MAX_TARGET_LEAD for sid in JOINTS):
                segment["elapsed"] = candidate
                for sid, command in commands.items():
                    if self.c.cancelled():
                        return
                    s = self.c.joints[sid]
                    s.command = command
                    if command != s.last_sent:
                        write_position(self.c.bus, sid, command % 1024, self.speed)
                        s.last_sent = command
                    s.status = f"动作跟随 · {self.steps[self.index]['pose']}"
            else:
                self.status = f"第 {self.index + 1} 段实测跟踪滞后，延长时间等待反馈。"
            if segment["elapsed"] < segment["duration"]:
                return
            if segment["settle_deadline"] is None:
                segment["settle_deadline"] = now + 8.0
            arrived = all(abs(s.position - segment["ends"][sid]) <= POSITION_TOLERANCE
                          and s.position // 1024 == segment["ends"][sid] // 1024
                          and s.settled(now) for sid, s in self.c.joints.items())
            if not arrived:
                segment["arrived"] = None
                if now >= segment["settle_deadline"]:
                    raise RuntimeError("段末未稳定到位；动作已暂停，请检查位置反馈。")
                return
            if segment["arrived"] is None:
                segment["arrived"] = now
            if now - segment["arrived"] < self.steps[self.index].get("hold", 0):
                return
            self.index += 1
            if self.index < len(self.steps):
                self._begin(now)
            else:
                self.steps, self.segment = [], None
                for s in self.c.joints.values():
                    s.active = False
                    s.status = "已到位 · 保持"
                self.status = "播放完成，保持最后姿态。"
                self.notify(self.status)
        except (OSError, RuntimeError, ValueError) as exc:
            self.stop()
            self.status = f"动作已停止：{exc}"
            self.notify(self.status)


class Mailbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.commands = deque()
        self.targets = {}
        self.emergency = threading.Event()
        self.holding = threading.Event()
        self.closing = threading.Event()
        self.speed = 100

    def cancelled(self):
        return self.emergency.is_set() or self.holding.is_set() or self.closing.is_set()

    def submit(self, name, payload=None):
        with self.lock:
            if self.cancelled():
                return
            if name == "target":
                self.targets[payload[0]] = payload[1]
            else:
                self.commands.append((name, payload))

    def interrupt(self, off=False):
        with self.lock:
            self.commands.clear()
            self.targets.clear()
            (self.emergency if off else self.holding).set()

    def take(self):
        with self.lock:
            result = list(self.commands), dict(self.targets), self.speed
            self.commands.clear()
            self.targets.clear()
            return result


class SerialWorker(threading.Thread):
    def __init__(self, data, port, mailbox, output, demo=False):
        super().__init__(daemon=True, name="FK-serial")
        self.data, self.port, self.mailbox, self.output, self.demo = data, port, mailbox, output, demo

    def emit(self, kind, value):
        self.output.put((kind, value))

    def run(self):
        bus = controller = engine = None
        try:
            port = "离线模拟（不连接串口）" if self.demo else (self.port or find_port())
            bus = DemoBus(self.data) if self.demo else LeRobotSCSBus(port)
            controller = ArmController(bus, self.data, self.mailbox.cancelled)
            controller.initialize(time.monotonic())
            engine = MotionEngine(controller, lambda message: self.emit("notice", message))
            self.emit("connected", port)
            while not self.mailbox.closing.is_set():
                began = time.monotonic()
                if self.mailbox.emergency.is_set():
                    self.mailbox.take()
                    engine.stop(off=True)
                    self.mailbox.emergency.clear()
                    self.mailbox.holding.clear()
                    self.emit("notice", engine.status)
                elif self.mailbox.holding.is_set():
                    self.mailbox.take()
                    engine.stop()
                    self.mailbox.holding.clear()
                    self.emit("notice", engine.status)
                if self.demo:
                    bus.step(UPDATE_SECONDS)
                controller.poll(began)
                commands, targets, speed = self.mailbox.take()
                for name, payload in commands:
                    if self.mailbox.cancelled():
                        break
                    try:
                        if name == "play":
                            engine.play(payload, speed, time.monotonic())
                        elif name == "capture":
                            engine.ready()
                            if engine.playing or any(s.active or not s.settled(began) for s in controller.joints.values()):
                                raise ValueError("请等六个关节停稳后再记录；保存的是实测值。")
                            self.emit("capture", (payload, {sid: s.raw for sid, s in controller.joints.items()}))
                        elif name == "off_joint":
                            if engine.playing:
                                engine.stop()
                            controller.off([payload])
                    except ValueError as exc:
                        self.emit("notice", str(exc))
                for sid, value in targets.items():
                    if self.mailbox.cancelled():
                        break
                    if engine.playing:
                        self.emit("notice", "动作播放中；按空格暂停后可拖动滑条。")
                        break
                    try:
                        controller.set_target(sid, value, speed, time.monotonic())
                    except (OSError, RuntimeError, ValueError) as exc:
                        if not isinstance(exc, ValueError):
                            controller.fault(sid, str(exc))
                        self.emit("notice", str(exc))
                engine.tick(time.monotonic(), speed)
                self.emit("snapshot", (controller.snapshot(), engine.playing, engine.status))
                self.mailbox.closing.wait(max(0, UPDATE_SECONDS - (time.monotonic() - began)))
            engine.stop(off=self.mailbox.emergency.is_set())
        except Exception as exc:
            if controller is not None:
                controller.off([sid for sid, s in controller.joints.items() if s.owned])
            self.emit("error", str(exc))
        finally:
            try:
                if bus is not None:
                    bus.close()
            finally:
                self.emit("closed", None)


class DemoBus:
    """仅用于界面预览/离线测试，模拟数值路径；不证明实机能跨反馈边界。"""
    def __init__(self, data):
        self.memory, self.writes, self.stuck = {}, [], set()
        self.closed = False
        values = data.get("poses", {}).get("middle", [0.5] * 6)
        for sid, value in zip(JOINTS, values):
            mem = bytearray(80)
            mem[:9] = bytes([1, 0, 1, 5, 35, sid, 0, 0, 1])
            mem[9:13], mem[21:23], mem[48] = b"\0\0\x03\xff", bytes([15, 15]), 1
            mem[56:58] = (target_position(value, data["joints"][str(sid)]) % 1024).to_bytes(2, "big")
            mem[62:64] = bytes([51, 20])
            self.memory[sid] = mem

    def read(self, sid, address, length=1):
        return bytes(self.memory[sid][address:address + length])

    def write_checked(self, sid, address, value):
        self.memory[sid][address] = value
        self.writes.append((sid, address, value))

    def write_limits(self, sid):
        self.memory[sid][9:13] = b"\0\0\x03\xff"
        self.writes.append((sid, 9, (0, 1023)))

    def write_position(self, sid, raw, speed):
        self.memory[sid][42:48] = raw.to_bytes(2, "big") + b"\0\0" + speed.to_bytes(2, "big")
        self.writes.append((sid, 42, raw))

    def off_all(self):
        for sid in JOINTS:
            self.write_checked(sid, 40, 0)

    def step(self, dt):
        for sid, mem in self.memory.items():
            if not mem[40] or sid in self.stuck:
                continue
            raw, goal = int.from_bytes(mem[56:58], "big"), int.from_bytes(mem[42:44], "big")
            cap = max(1, int(int.from_bytes(mem[46:48], "big") * dt))
            delta = max(-cap, min(cap, circular_delta(goal, raw)))
            mem[56:58] = ((raw + delta) % 1024).to_bytes(2, "big")

    def close(self):
        self.closed = True
