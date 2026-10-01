"""User calibration + the TA's unmodified SCS215 bus. No ports opened on import.

TA motor sources and the matching utility modules are bundled locally.
No site-packages files or old controllers are modified.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

from ik_solver import URDFArm

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "lerobot_EAI_SCS215"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def load_config(path=HERE / "config.json"):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("calibration_path", "urdf_path"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    bounds = {"speed_cm_s": (.1, 3), "jog_step_cm": (.1, 5),
              "control_hz": (5, 50), "planning_step_mm": (1, 10),
              "max_joint_speed_deg_s": (1, 45), "servo_speed": (1, 300),
              "final_count_tolerance": (1, 10), "settle_seconds": (.1, 10),
              "max_tracking_error_counts": (20, 150),
              "endpoint_tolerance_mm": (1, 20), "line_tolerance_mm": (1, 25),
              "tracking_soft_counts": (1, 15), "tracking_pause_counts": (5, 30),
              "tracking_soft_mm": (1, 10), "tracking_pause_mm": (3, 25),
              "tracking_ki_per_s": (0, 2), "integral_limit_counts": (0, 40),
              "integral_rate_counts_s": (1, 15), "integral_deadband_counts": (0, 3),
              "no_progress_timeout_s": (3, 30), "arrival_dwell_s": (.1, 2)}
    for key, (lo, hi) in bounds.items():
        val = cfg[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or not lo <= val <= hi:
            raise ValueError(f"config.json: {key} 必须在 {lo}～{hi}。")
    if (cfg["tracking_soft_counts"] >= cfg["tracking_pause_counts"]
            or cfg["tracking_soft_mm"] >= cfg["tracking_pause_mm"]
            or cfg["arrival_dwell_s"] >= cfg["settle_seconds"]):
        raise ValueError("跟随减速阈值必须小于暂停阈值；到位稳定时间必须小于终点调整时间。")
    return cfg


class Calibration:
    def __init__(self, cfg):
        self.cfg = cfg
        self.source = Path(cfg["calibration_path"])
        data = json.loads(self.source.read_text(encoding="utf-8-sig"))
        if data.get("model") != "SCS215":
            raise ValueError("标定型号必须是 SCS215。")
        joints = data["joints"]
        if len(joints) != 6 or [j["id"] for j in joints] != list(range(1, 7)):
            raise ValueError("标定必须按顺序包含 ID1～6。")
        self.minimum = np.array([j["range_0"] for j in joints], dtype=float)
        self.maximum = np.array([j["range_1"] for j in joints], dtype=float)
        self.zero = np.array([j["angle_zero"] for j in joints], dtype=float)
        for array in (self.minimum, self.maximum, self.zero):
            if (not np.isfinite(array).all() or np.any(array != np.round(array))
                    or np.any(array < 0) or np.any(array > 1023)):
                raise ValueError("标定原始读数必须是 0～1023 的整数。")
        if np.any(self.maximum <= self.minimum) or np.any(self.zero < self.minimum) or np.any(self.zero > self.maximum):
            raise ValueError("此程序使用重装后的连续反馈区间，要求 0端点 < 1端点，零位在区间内。")
        gain = cfg.get("degrees_per_count_override")
        self.degrees_per_count = float(data["degrees_per_count"] if gain is None else gain)
        if not np.isfinite(self.degrees_per_count) or not 0 < self.degrees_per_count < 1:
            raise ValueError("每读数对应角度无效。")
        signs = np.asarray(cfg["direction_signs"], dtype=float)
        if signs.shape != (5,) or not np.isin(signs, [-1, 1]).all():
            raise ValueError("direction_signs 必须是五个 +1 或 -1。")
        self.gain = np.deg2rad(self.degrees_per_count) * signs
        limits = {NAMES[i]: tuple(sorted(self.gain[i] * (np.array([self.minimum[i], self.maximum[i]]) - self.zero[i])))
                  for i in range(5)}
        self.arm = URDFArm(cfg["urdf_path"], calibrated_limits=limits)
        if self.arm.names != list(NAMES[:5]):
            raise ValueError(f"URDF 关节顺序不符：{self.arm.names}")
        edge = np.vstack([self.zero[:5] + self.arm.lower / self.gain,
                          self.zero[:5] + self.arm.upper / self.gain])
        self.raw_lower = np.ceil(edge.min(axis=0) - 1e-9).astype(int)
        self.raw_upper = np.floor(edge.max(axis=0) + 1e-9).astype(int)
        if np.any(self.raw_lower >= self.raw_upper):
            raise ValueError("标定范围与 URDF 没有可用交集。")
        angle_edges = np.vstack([self.to_q(self.raw_lower), self.to_q(self.raw_upper)])
        self.arm.lower = angle_edges.min(axis=0)
        self.arm.upper = angle_edges.max(axis=0)

    def to_q(self, raw):
        raw = np.asarray(raw, dtype=float)
        if raw.shape not in ((5,), (6,)) or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1023):
            raise ValueError("需要五个或六个有效原始位置。")
        return (raw[:5] - self.zero[:5]) * self.gain

    def check_start(self, raw):
        raw = np.asarray(raw)
        if raw.shape != (6,):
            raise ValueError("需要读取六个关节。")
        for i in range(5):
            if not self.raw_lower[i] <= raw[i] <= self.raw_upper[i]:
                raise ValueError(f"ID{i+1} 当前 {raw[i]} 超出标定与URDF交集 [{self.raw_lower[i]}, {self.raw_upper[i]}]；未发运动。")
        if not self.minimum[5] <= raw[5] <= self.maximum[5]:
            raise ValueError("夹爪当前位置超出记录范围；请先调整夹爪。")

    def to_raw(self, q):
        q = self.arm.configuration(q)
        if np.any(q < self.arm.lower - 1e-9) or np.any(q > self.arm.upper + 1e-9):
            raise ValueError("IK目标超过关节范围。")
        raw = np.rint(self.zero[:5] + q / self.gain).astype(int)
        return np.clip(raw, self.raw_lower, self.raw_upper)

    def quantize(self, q, target):
        """Select neighbouring integer counts jointly to reduce TCP rounding error."""
        self.to_raw(q)
        ideal = self.zero[:5] + q / self.gain
        choices = [sorted({int(np.clip(v, lo, hi)) for v in (np.floor(x), np.ceil(x))})
                   for x, lo, hi in zip(ideal, self.raw_lower, self.raw_upper)]
        candidates = [np.asarray(c, dtype=int) for c in itertools.product(*choices)]
        return min(candidates, key=lambda r: float(np.sum((self.arm.fk(self.to_q(r))[:3, 3] - target)**2)
                                                       + 1e-12*np.sum((r-ideal)**2)))

    def ratios(self, raw):
        # Do not silently clamp measured positions outside calibration.
        return (np.asarray(raw) - self.minimum) / (self.maximum - self.minimum)


def load_course_driver():
    source = json.loads((VENDOR / "SOURCE.json").read_text(encoding="utf-8-sig"))
    for relative, expected in source["files"].items():
        if hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"助教驱动源码校验失败：{relative}")
    runtime = json.loads((VENDOR / "RUNTIME_SOURCE.json").read_text(encoding="utf-8"))
    for relative, expected in runtime["files"].items():
        if hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"提交包支持模块校验失败：{relative}")
    # Use the bundled package, including its utility modules. This also works
    # when the original editable LeRobot checkout is absent on another computer.
    vendor_package = VENDOR / "src" / "lerobot"
    loaded_package = sys.modules.get("lerobot")
    if loaded_package is not None and not Path(loaded_package.__file__).resolve().is_relative_to(vendor_package):
        raise RuntimeError("本进程已载入其他 LeRobot，请直接运行本目录的 control_ik.py。")
    source_dir = str(VENDOR / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    loaded = sys.modules.get("lerobot.motors")
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(vendor_package):
        raise RuntimeError("本进程已载入另一版本舵机驱动，请直接运行新目录的 control_ik.py。")
    motors = importlib.import_module("lerobot.motors")
    feetech = importlib.import_module("lerobot.motors.feetech")
    return motors, feetech.FeetechMotorsBus, source


def make_bus(mapping, port):
    motors_module, bus_class, source = load_course_driver()
    Motor, Mode, MotorCalibration = motors_module.Motor, motors_module.MotorNormMode, motors_module.MotorCalibration
    motors = {name: Motor(i+1, "scs215", Mode.RANGE_M100_100 if i < 5 else Mode.RANGE_0_100)
              for i, name in enumerate(NAMES)}
    calibration = {name: MotorCalibration(id=i+1, drive_mode=0, homing_offset=0,
                                        range_min=int(mapping.minimum[i]), range_max=int(mapping.maximum[i]))
                   for i, name in enumerate(NAMES)}
    return bus_class(port, motors, calibration=calibration, protocol_version=1), source


def find_port(cfg):
    if cfg.get("port"):
        return cfg["port"]
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    matches = [p.device for p in ports if p.serial_number == cfg["adapter_serial"]]
    if len(matches) != 1:
        raise RuntimeError("未找到原转接板，请用 --port COM号码 指定。可用串口：" +
                           ", ".join(f"{p.device} {p.description}" for p in ports))
    return matches[0]


class Robot:
    simulated = False

    def __init__(self, mapping, bus=None):
        self.mapping, self.cfg = mapping, mapping.cfg
        self.owned = False
        self.last_target = None
        self.bus = bus
        if self.bus is None:
            self.bus, _ = make_bus(mapping, find_port(self.cfg))
        try:
            # TA handshake validates model number and firmware; only reads.
            self.bus.connect()
            self.raw = self.read_positions()
            self.hardware_limits = np.array([[self.read_reg("Min_Position_Limit", name),
                                              self.read_reg("Max_Position_Limit", name)] for name in NAMES])
        except BaseException:
            self.close()
            raise

    def read_reg(self, register, name):
        return int(self.bus.read(register, name, normalize=False, num_retry=2))

    def read_positions(self):
        self.raw = np.array([self.read_reg("Present_Position", name) for name in NAMES], dtype=int)
        self.mapping.to_q(self.raw)
        return self.raw.copy()

    def adopt_stable_hold(self, current):
        """Read-only adoption of an already enabled, stationary holding target.

        Avoid removing a load-compensated goal on program restart. Never adopt
        a disabled joint, large/invalid goal, ongoing motion or another speed mode.
        """
        if any(self.read_reg("Torque_Enable", n) != 1 for n in NAMES):
            return None
        goals = np.array([self.read_reg("Goal_Position", n) for n in NAMES])
        if (np.max(np.abs(goals-current)) > self.cfg["integral_limit_counts"]
                or goals[5] != current[5]):
            return None
        try:
            self.mapping.check_start(goals)
        except ValueError:
            return None
        if np.any(goals < self.hardware_limits[:, 0]) or np.any(goals > self.hardware_limits[:, 1]):
            return None
        if any(self.read_reg("Running_Time", n) != 0 or
               self.read_reg("Goal_Velocity", n) != int(self.cfg["servo_speed"]) for n in NAMES):
            return None
        time.sleep(.2)
        if np.max(np.abs(self.read_positions()-current)) > 2:
            return None
        self.last_target, self.owned = goals.copy(), True
        return goals.copy()

    def snapshot(self):
        registers = ("Goal_Position", "Present_Position", "Torque_Enable", "Present_Voltage",
                     "Present_Temperature", "Status", "P_Coefficient", "D_Coefficient", "Present_Load")
        return {str(i+1): {reg: self.read_reg(reg, name) for reg in registers}
                for i, name in enumerate(NAMES)}

    def preflight(self, raw_path):
        # Refresh in case another tool changed hardware limits since connection.
        self.hardware_limits = np.array([[self.read_reg("Min_Position_Limit", name),
                                          self.read_reg("Max_Position_Limit", name)] for name in NAMES])
        for i in range(6):
            lo, hi = self.hardware_limits[i]
            if lo >= hi or np.any(raw_path[:, i] < lo) or np.any(raw_path[:, i] > hi):
                raise RuntimeError(f"ID{i+1} 路径与硬件限位 [{lo}, {hi}] 不符；未发运动。diagnose 可查看，未自动改写EEPROM。")
        self.check_health()

    def check_health(self):
        for name in NAMES:
            status = self.read_reg("Status", name)
            if status:
                raise RuntimeError(f"{name} 报告硬件状态异常 0x{status:02x}。")
            if self.owned and self.read_reg("Torque_Enable", name) != 1:
                raise RuntimeError(f"{name} 扭矩已关闭；不自动反复使能。")

    def write_raw(self, raw):
        raw = np.asarray(raw)
        if (raw.shape != (6,) or not np.isfinite(raw).all() or np.any(raw != np.rint(raw))
                or np.any(raw < self.mapping.minimum) or np.any(raw > self.mapping.maximum)):
            raise ValueError("待发送原始目标超出六关节标定范围。")
        if (np.any(raw[:5] < self.mapping.raw_lower) or np.any(raw[:5] > self.mapping.raw_upper)
                or np.any(raw < self.hardware_limits[:, 0]) or np.any(raw > self.hardware_limits[:, 1])):
            raise ValueError("待发送目标（含反馈补偿）超出URDF/标定/硬件限位交集。")
        # Official bus API. Raw mode avoids the driver's truncating float
        # normalization creating an extra one-count error. No custom packets.
        self.bus.sync_write("Goal_Position", dict(zip(NAMES, map(int, raw))), normalize=False, num_retry=1)
        self.last_target = raw.astype(int).copy()

    def verify_target(self):
        stored = np.array([self.read_reg("Goal_Position", name) for name in NAMES])
        if self.last_target is not None and not np.array_equal(stored, self.last_target):
            raise RuntimeError(f"目标寄存器不一致：期望 {self.last_target.tolist()}，读回 {stored.tolist()}")
        return stored

    def arm(self, current):
        if self.owned:
            # Consecutive moves must not discard the previous load-compensated
            # holding goal. A newly connected session still stages/enables all six.
            self.verify_target()
            self.check_health()
            return
        # Stage current-position goals before enabling torque, avoiding a jump
        # toward goals left in RAM by an earlier program. No calibration/PID writes.
        self.write_raw(current)
        self.verify_target()
        for register, value in (("Running_Time", 0), ("Goal_Velocity", int(self.cfg["servo_speed"]))):
            self.bus.sync_write(register, dict.fromkeys(NAMES, value), normalize=False, num_retry=1)
            if any(self.read_reg(register, name) != value for name in NAMES):
                raise RuntimeError(f"{register} 写入读回不符。")
        self.owned = True  # Include a partial enable in subsequent stop handling.
        self.bus.sync_write("Torque_Enable", dict.fromkeys(NAMES, 1), normalize=False, num_retry=1)
        self.check_health()

    def hold(self):
        if self.owned:
            # Holding is distinct from unloading. This method is only called on
            # explicit stop or a real execution fault, not for normal final error.
            self.write_raw(self.read_positions())
            self.verify_target()

    def off(self):
        self.bus.sync_write("Torque_Enable", dict.fromkeys(NAMES, 0), normalize=False, num_retry=2)
        self.owned = False
        states = [self.read_reg("Torque_Enable", name) for name in NAMES]
        if any(states):
            raise RuntimeError(f"卸力未全部确认：{states}")

    def close(self):
        if self.bus is not None and self.bus.is_connected:
            self.bus.disconnect(disable_torque=False)


class Simulation:
    """Integer encoder targets and a configurable five-count motor deadband."""
    simulated = True

    def __init__(self, mapping, initial=None, deadband=0, stuck=False):
        self.mapping, self.cfg = mapping, mapping.cfg
        self.raw = np.rint(mapping.zero if initial is None else initial).astype(int)
        self.owned = False
        self.last_target = self.raw.copy()
        self.deadband, self.stuck = deadband, stuck
        self.sent = []

    def read_positions(self):
        return self.raw.copy()

    def preflight(self, raw_path):
        pass

    def check_health(self):
        pass

    def arm(self, current):
        self.owned = True

    def write_raw(self, raw):
        self.last_target = np.asarray(raw, dtype=int).copy()
        self.sent.append(self.last_target.copy())
        if not self.stuck:
            delta = self.last_target - self.raw
            mask = np.abs(delta) > self.deadband
            self.raw[mask] = self.last_target[mask]

    def verify_target(self):
        return self.last_target.copy()

    def hold(self):
        if self.owned:
            self.last_target = self.raw.copy()

    def off(self):
        self.owned = False

    def close(self):
        pass

    def snapshot(self):
        return {str(i+1): {"Present_Position": int(r), "Goal_Position": int(self.last_target[i]),
                           "Torque_Enable": int(self.owned)} for i, r in enumerate(self.raw)}
