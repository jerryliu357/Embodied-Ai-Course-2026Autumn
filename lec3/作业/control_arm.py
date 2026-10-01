"""从当前位置控制六只 SCS215，不预先复位。

直接运行：交互输入 middle / zero / rotated / rest，或六个 0～1 数值。
也可运行 control_arm.py middle 或 control_arm.py 0.5 0.5 0.5 0.5 0.5 0.5。
--dry-run 仅计算目标，不连接串口；--check 仅查询硬件和检查本次路线。
六只使用同一个速度寄存器数值，同步下发；行程不同，不保证同时到达。
展开为连续计数后分段下发，支持 1023/0 跨界计数；每段核对实际方向。
成功后保持位置；q 保持退出；托住后输入 off 卸力；Ctrl+C 急停卸力。
运动范围只使用 calibration.json 的六组 0/1 端点。
首次运动前自动将旧硬件角度限位放宽到 0～1023，读回并锁定 EEPROM。
首次使用请托稳机械臂，配置期间会关闭扭矩；不修改编号或校准文件。
"""
import argparse
import json
import math
import re
import sys
import time
from collections import deque

import serial
from calibrate import OUTPUT, POSITION_COUNT, ratio_to_raw
from record_poses import POSES, get_pose_values, raw_to_ratio, validate_calibration
from set_id import JOINTS, LOCK_REGISTER, TORQUE_REGISTER, ServoBus, find_port, inspect, packet

SPEED = 100                  # 六只统一速度参数；不是 0～1 比例/秒，也不是角度/秒。
MOVE_TIMEOUT_SECONDS = 30.0
POSITION_TOLERANCE = 6       # 原始计数单位
SETTLE_SECONDS = 0.25        # 用连续位置样本确认稳定，不要求瞬时速度寄存器恒为 0
SETTLE_SPREAD = 3            # 稳定窗口内位置最大值与最小值之差
POLL_INTERVAL = 0.05
STALL_SECONDS = 5.0
DIAGNOSTIC_AFTER_SECONDS = 2.0  # 未继续接近目标时，只读一次运行状态辅助定位
WAYPOINT_STEP = 64           # 普通段每次最多推进的原始计数
MAX_FEEDBACK_STEP = 192      # 单次反馈过大时无法可靠判断跨界，不猜方向
REVERSE_TOLERANCE = 6        # 一段内明显反向则停机
MIN_LIMIT_REGISTER = 0x09
FULL_POSITION_RANGE = (0, POSITION_COUNT - 1)
GOAL_POSITION_REGISTER = 0x2A
PRESENT_POSITION_REGISTER = 0x38


class MoveRejected(ValueError):
    """在下发指令之前发现输入、起点或路线不适合执行。"""


def parse_target(text, data):
    text = text.strip().lower()
    if text in POSES:
        return get_pose_values(data, text)
    if text.startswith("["):
        values = json.loads(text)
    else:
        try:
            values = [float(item) for item in re.split(r"[\s,，]+", text) if item]
        except ValueError as exc:
            raise ValueError("请输入 middle / zero / rotated / rest，或六个 0～1 数字。") from exc
    if (not isinstance(values, list) or len(values) != 6
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values)):
        raise ValueError("必须输入六个 0～1 的有限数值，顺序为 ID1～ID6。")
    return values


def targets_from_values(data, values):
    # 再次验证供其他脚本直接调用时传入的值。
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError("目标需要六个数值。")
    return {sid: ratio_to_raw(value, data["joints"][str(sid)])
            for sid, value in zip(JOINTS, values)}


def sync_packet(address, payloads):
    if set(payloads) != set(JOINTS):
        raise ValueError("同步包必须明确指定 ID1～ID6。")
    width = len(payloads[1])
    if width < 1 or any(len(value) != width for value in payloads.values()):
        raise ValueError("同步写入长度不一致。")
    parameters = [address, width]
    for sid in JOINTS:
        parameters.extend([sid, *payloads[sid]])
    return packet(0xFE, 0x83, parameters)


def sync_write(bus, address, payloads):
    outgoing = sync_packet(address, payloads)
    bus.serial.reset_input_buffer()
    if bus.serial.write(outgoing) != len(outgoing):
        raise RuntimeError("串口未完整发送同步指令。")
    bus.serial.flush()
    # SYNC WRITE 无应答，逐个读回目标寄存器以确认收到。
    for sid, expected in payloads.items():
        actual = bus.read(sid, address, len(expected))
        if actual != expected:
            raise RuntimeError(f"ID={sid} 地址 {address:#x} 写入后读回不一致。")


def write_positions(bus, positions, speed):
    if type(speed) is not int or not 1 <= speed <= 1000:
        raise ValueError("速度参数必须是 1～1000 的整数，0 不用于本程序。")
    payloads = {}
    for sid, position in positions.items():
        if type(position) is not int or not 0 <= position < POSITION_COUNT:
            raise ValueError(f"ID={sid} 原始目标不合法：{position}")
        # SCS215：高字节在前，连续写位置、时间(0)、相同速度。
        payloads[sid] = position.to_bytes(2, "big") + b"\0\0" + speed.to_bytes(2, "big")
    sync_write(bus, GOAL_POSITION_REGISTER, payloads)


def read_states(bus):
    states = {}
    for sid in JOINTS:
        state = inspect(bus, sid)
        if state["model_bytes"] != "05 23":
            raise MoveRejected(f"ID={sid} 型号字节不是这批 SCS215 的 05 23。")
        limits = bus.read(sid, MIN_LIMIT_REGISTER, 4)
        state["limits"] = (int.from_bytes(limits[:2], "big"), int.from_bytes(limits[2:], "big"))
        states[sid] = state
    return states


def read_stable_states(bus):
    states = read_states(bus)
    tracker = PositionTracker({sid: state["position"] for sid, state in states.items()})
    settling = PositionSettling()
    began = time.monotonic()
    settling.update(tracker.positions, began)
    while True:
        time.sleep(POLL_INTERVAL)
        latest = read_feedback(bus)
        tracker.update(latest)
        now = time.monotonic()
        settled = settling.update(tracker.positions, now)
        if all(settled.values()):
            for sid, (position, measured_speed) in latest.items():
                states[sid]["position"], states[sid]["speed_raw"] = position, measured_speed
            return states
        if now - began >= SETTLE_SECONDS + 0.25:
            pending = [sid for sid, ready in settled.items() if not ready]
            raise MoveRejected(f"ID={pending} 起点还在变化，请托稳后重新输入。")


def range_updates(states):
    return {sid: state["limits"] for sid, state in states.items()
            if tuple(state["limits"]) != FULL_POSITION_RANGE}


def ensure_position_range(bus, states):
    """清除历史窄限位，保留位置模式；仅修改需要修改的 EEPROM。

    上下限连续四字节一次写入 00 00 03 FF（大端）。不能写成全零：
    SCSCL 的上下限全零表示 PWM 模式，不是取消位置模式的角度限制。
    """
    changes = range_updates(states)
    if not changes:
        return False
    print("正在解除旧硬件角度限位，请托稳机械臂；配置期间关闭六只扭矩。", flush=True)
    release_torque(bus)
    payload = b"\x00\x00" + (POSITION_COUNT - 1).to_bytes(2, "big")
    for sid, old_limits in changes.items():
        try:
            bus.write_checked(sid, LOCK_REGISTER, 0)
            bus.exchange(sid, 3, [MIN_LIMIT_REGISTER, *payload], 0, optional=True)
            time.sleep(0.05)
            if bus.read(sid, MIN_LIMIT_REGISTER, len(payload)) != payload:
                raise RuntimeError(f"ID={sid} 设置 0～1023 后读回不一致，未开始运动。")
        finally:
            # 即使配置中途失败或被中断，也尝试重新锁定；不恢复旧限位。
            bus.write_checked(sid, LOCK_REGISTER, 1)
        print(f"ID={sid} 旧限位 {list(old_limits)} → [0, 1023]，读回通过，EEPROM 已锁定。",
              flush=True)
    # 在任何目标/扭矩开启指令之前确认全部设置；不凭写入请求判断成功。
    for sid in JOINTS:
        if (bus.read(sid, MIN_LIMIT_REGISTER, len(payload)) != payload
                or bus.read(sid, TORQUE_REGISTER) != b"\0"):
            raise RuntimeError(f"ID={sid} 配置后的范围或关闭扭矩状态不符，未开始运动。")
    return True


def read_feedback(bus):
    feedback = {}
    for sid in JOINTS:
        raw = bus.read(sid, PRESENT_POSITION_REGISTER, 4)
        position = int.from_bytes(raw[:2], "big")
        speed = int.from_bytes(raw[2:], "big") & 0x7FFF
        if not 0 <= position < POSITION_COUNT:
            raise RuntimeError(f"ID={sid} 反馈位置异常：{position}")
        feedback[sid] = (position, speed)
    return feedback


def circular_delta(current, previous):
    """相邻慢速反馈的有符号增量；1023→0 是 +1，0→1023 是 -1。"""
    return (current - previous + POSITION_COUNT // 2) % POSITION_COUNT - POSITION_COUNT // 2


def current_unwrapped(raw, joint):
    value, adjustment = raw_to_ratio(raw, joint)
    position = round(joint["zero_position"] + value * joint["span_raw"])
    if adjustment:
        position += circular_delta(raw, adjustment["target_raw"])
    return value, position


def joint_waypoints(start, target):
    """在展开坐标中分段；跨界时从本圈上端接到下一圈下端，反向同理。"""
    low, high = FULL_POSITION_RANGE
    direction = 1 if target >= start else -1
    points, position = [], start
    while position != target:
        turn = position // POSITION_COUNT
        if target // POSITION_COUNT == turn:
            stop = target
        else:
            stop = turn * POSITION_COUNT + (high if direction > 0 else low)
        if stop == position:
            following_turn = turn + direction
            following = following_turn * POSITION_COUNT + (low if direction > 0 else high)
            position = following
        else:
            position += direction * min(WAYPOINT_STEP, abs(stop - position))
        if direction * (target - position) < 0:
            raise MoveRejected("中间目标超出本次起终点区间。")
        points.append(position)
    return points or [target]


def build_waypoints(starts, ends):
    routes = {sid: joint_waypoints(starts[sid], ends[sid])
              for sid in JOINTS}
    return [{sid: route[min(index, len(route) - 1)] for sid, route in routes.items()}
            for index in range(max(map(len, routes.values())))]


def check_route(data, states, values):
    """仅用本次校准端点判断运动范围，旧硬件限位由执行前的配置步骤解除。"""
    targets = targets_from_values(data, values)
    current_values, errors = [], []
    for sid, value in zip(JOINTS, values):
        item, state = data["joints"][str(sid)], states[sid]
        current = state["position"]
        if state["torque"] not in (0, 1):
            errors.append(f"ID={sid} 扭矩状态 {state['torque']} 不适用于本程序。")
        # 起点是否稳定由 read_stable_states 的连续位置窗口判断。
        # 本批舵机位置仅抖动 1 个计数时，瞬时速度也可能在 0/±50 间跳动。
        try:
            current_value, _ = current_unwrapped(current, item)
            current_values.append(current_value)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise MoveRejected("本次未下发运动指令：\n" + "\n".join(errors))
    return current_values, targets


def release_torque(bus):
    """先快速同步卸力；即使某只读回失败，也继续尝试其他五只。"""
    failures = []
    try:
        outgoing = sync_packet(TORQUE_REGISTER, {sid: b"\0" for sid in JOINTS})
        bus.serial.write(outgoing)
        bus.serial.flush()
    except (OSError, serial.SerialException):
        pass
    for sid in JOINTS:
        try:
            bus.write_checked(sid, TORQUE_REGISTER, 0)
        except (RuntimeError, OSError, serial.SerialException) as exc:
            failures.append(f"ID={sid}: {exc}")
    if failures:
        raise RuntimeError("未能确认全部卸力，请托住机械臂并断开舵机电源：\n" + "\n".join(failures))
    print("六只舵机扭矩已关闭，请托稳机械臂。", flush=True)


class PositionTracker:
    """跨多个小段连续保留圈数，不在原始数字跳回 0 时丢失进度。"""
    def __init__(self, unwrapped):
        self.positions = dict(unwrapped)
        self.raw = {sid: value % POSITION_COUNT for sid, value in unwrapped.items()}

    def update(self, feedback):
        next_positions = {}
        for sid, (raw, _) in feedback.items():
            change = circular_delta(raw, self.raw[sid])
            if abs(change) > MAX_FEEDBACK_STEP:
                raise RuntimeError(f"ID={sid} 相邻反馈变化 {change} 太大，不能可靠追踪本次路线。")
            next_positions[sid] = self.positions[sid] + change
        self.positions = next_positions
        self.raw = {sid: pair[0] for sid, pair in feedback.items()}


class PositionSettling:
    """至少三个样本覆盖 0.25 秒、位置波动不超过 3 个计数才算稳定。"""
    def __init__(self):
        self.samples = {sid: deque() for sid in JOINTS}

    def update(self, positions, now):
        settled = {}
        for sid, position in positions.items():
            samples = self.samples[sid]
            samples.append((now, position))
            while len(samples) > 1 and samples[1][0] <= now - SETTLE_SECONDS:
                samples.popleft()
            values = [value for _, value in samples]
            settled[sid] = (len(samples) >= 3
                            and now - samples[0][0] >= SETTLE_SECONDS
                            and max(values) - min(values) <= SETTLE_SPREAD)
        return settled


def waiting_description(feedback, positions, targets, settled=None):
    waiting = []
    for sid, (raw, speed) in feedback.items():
        target = targets[sid]
        distance = abs(positions[sid] - target)
        same_turn = positions[sid] // POSITION_COUNT == target // POSITION_COUNT
        position_stable = settled is None or settled[sid]
        if distance > POSITION_TOLERANCE or not same_turn or not position_stable:
            boundary = "，尚未通过跨界点" if not same_turn else ""
            stability = "，位置尚未稳定" if not position_stable else ""
            waiting.append(f"ID{sid}: 当前 {raw} / 本段目标 {target % POSITION_COUNT}，"
                           f"误差 {distance}，速度反馈 {speed}{boundary}{stability}")
    return "；".join(waiting) or "六只已进入到位范围，等待稳定"


def report_stall_snapshot(bus, sid):
    """只读卡住时的目标、负载与供电；诊断失败也不能阻止后续停机。"""
    try:
        goal = bus.read(sid, GOAL_POSITION_REGISTER, 6)
        motion = bus.read(sid, PRESENT_POSITION_REGISTER, 8)
        torque = bus.read(sid, TORQUE_REGISTER)[0]
        details = {
            "目标寄存器": int.from_bytes(goal[:2], "big"),
            "位置": int.from_bytes(motion[:2], "big"),
            "速度反馈原码": int.from_bytes(motion[2:4], "big"),
            "负载反馈原码": int.from_bytes(motion[4:6], "big"),
            "驱动输出占空比%": (int.from_bytes(motion[4:6], "big") & 0x3FF) / 10,
            "电压V": motion[6] / 10,
            "温度C": motion[7],
            "扭矩开关": torque,
        }
        print(f"ID{sid} 停滞诊断（负载原码不是力矩单位）：{details}", flush=True)
    except (RuntimeError, OSError, serial.SerialException) as exc:
        print(f"ID{sid} 停滞诊断读取失败：{exc}", flush=True)


def wait_until_reached(bus, tracker, targets_unwrapped, timeout, final_segment=False):
    began = time.monotonic()
    start = dict(tracker.positions)
    progress = {sid: (abs(start[sid] - targets_unwrapped[sid]), began) for sid in JOINTS}
    stable, last_display = 0, began - 1
    diagnosed = set()
    settling = PositionSettling()
    settling.update(tracker.positions, began)
    waiting = "尚未取得反馈"
    while time.monotonic() - began < timeout:
        feedback = read_feedback(bus)
        tracker.update(feedback)
        now = time.monotonic()
        settled = settling.update(tracker.positions, now)
        reached = True
        for sid, (_, speed) in feedback.items():
            position, target = tracker.positions[sid], targets_unwrapped[sid]
            distance = abs(position - target)
            # 目标 1024(原始0) 不能被上一圈的 1023 冒充到位。
            same_turn = position // POSITION_COUNT == target // POSITION_COUNT
            joint_reached = distance <= POSITION_TOLERANCE and same_turn and settled[sid]
            reached &= joint_reached
            direction = (target > start[sid]) - (target < start[sid])
            if direction * (position - start[sid]) < -REVERSE_TOLERANCE:
                raise RuntimeError(f"ID={sid} 实际运动与本段方向相反，停止驱动；普通位置模式可能没有按期望方向跨界。")
            low, high = sorted((start[sid], target))
            if not low - 15 <= position <= high + 15:
                raise RuntimeError(f"ID={sid} 展开反馈 {position} 偏离本段 [{low}, {high}]。")
            prior_distance, changed_at = progress[sid]
            if prior_distance - distance >= 3:
                progress[sid] = (distance, now)
            elif not joint_reached:
                if now - changed_at >= STALL_SECONDS:
                    if sid not in diagnosed:
                        report_stall_snapshot(bus, sid)
                    raise RuntimeError(
                        f"ID={sid} 连续 {STALL_SECONDS:g} 秒未满足本段到位条件："
                        f"当前 {feedback[sid][0]}，本段目标 {target % POSITION_COUNT}，"
                        f"误差 {distance}，位置稳定={settled[sid]}，速度反馈 {speed}；停止驱动。"
                    )
                if now - changed_at >= DIAGNOSTIC_AFTER_SECONDS and sid not in diagnosed:
                    print("仍在等待：" + waiting_description(feedback, tracker.positions, targets_unwrapped, settled),
                          flush=True)
                    report_stall_snapshot(bus, sid)
                    diagnosed.add(sid)
        stable = stable + 1 if reached else 0
        if stable >= (3 if final_segment else 1):
            return {sid: pair[0] for sid, pair in feedback.items()}
        if now - last_display >= 1:
            print("当前位置：" + "  ".join(f"{sid}={pair[0]}" for sid, pair in feedback.items()), flush=True)
            waiting = waiting_description(feedback, tracker.positions, targets_unwrapped, settled)
            print("等待到位：" + waiting, flush=True)
            last_display = now
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"等待到位超时：{waiting}；停止驱动。")


def move_arm(bus, data, values, speed, timeout, check_only=False):
    validate_calibration(data)
    if type(speed) is not int or not 1 <= speed <= 1000:
        raise ValueError("速度参数必须是 1～1000 的整数。")
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("超时应大于 0 且不超过 300 秒。")
    states = read_stable_states(bus)
    current_values, targets = check_route(data, states, values)
    try:
        if not check_only and ensure_position_range(bus, states):
            # 配置时曾卸力，必须重新实测起点，不能使用卸力前的位置。
            states = read_stable_states(bus)
            current_values, targets = check_route(data, states, values)
        current = {sid: state["position"] for sid, state in states.items()}
        starts = {sid: current_unwrapped(current[sid], data["joints"][str(sid)])[1] for sid in JOINTS}
        ends = {sid: round(data["joints"][str(sid)]["zero_position"]
                           + value * data["joints"][str(sid)]["span_raw"])
                for sid, value in zip(JOINTS, values)}
        waypoints = build_waypoints(starts, ends)
        print("起点比例（ID1～ID6）：", [round(value, 6) for value in current_values])
        print("目标比例（ID1～ID6）：", list(values))
        print("原始目标：", targets)
        print(f"按连续计数规划 {len(waypoints)} 个同步小段，0/1023 均按有效计数处理。")
        if check_only:
            changes = range_updates(states)
            if changes:
                print(f"正式运行时会将这些舵机的旧限位设为 0～1023：{changes}")
            print("校准范围内的路线检查通过；本次只读，未修改硬件或发送运动指令。")
            return current
        # 在使能前覆盖历史目标为当前读数，避免先追逐旧目标。
        write_positions(bus, current, speed)
        sync_write(bus, TORQUE_REGISTER, {sid: b"\x01" for sid in JOINTS})
        tracker = PositionTracker(starts)
        deadline = time.monotonic() + timeout
        for index, waypoint in enumerate(waypoints):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("整段运动超时，停止驱动。")
            step_raw = {sid: value % POSITION_COUNT for sid, value in waypoint.items()}
            print(f"第 {index + 1}/{len(waypoints)} 段目标：{step_raw}", flush=True)
            write_positions(bus, step_raw, speed)
            actual = wait_until_reached(bus, tracker, waypoint, remaining,
                                        final_segment=index == len(waypoints) - 1)
        print("已到位并保持：", actual, flush=True)
        return actual
    except (KeyboardInterrupt, ValueError, RuntimeError, OSError, serial.SerialException) as exc:
        if check_only:
            raise
        print("运动中止，正在卸力，请托住机械臂。", flush=True)
        try:
            release_torque(bus)
            if isinstance(exc, KeyboardInterrupt):
                exc.torque_released = True
        except (RuntimeError, OSError, serial.SerialException) as cleanup_error:
            print(str(cleanup_error), file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="*", help="姿态名称或六个比例；省略则进入交互输入")
    parser.add_argument("--port", help="默认自动识别原 USB 转接板")
    parser.add_argument("--speed", type=int, default=SPEED, help="六只共用的速度参数，默认 100")
    parser.add_argument("--timeout", type=float, default=MOVE_TIMEOUT_SECONDS, help="到位超时秒数，默认 30")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="仅计算目标，不打开串口")
    mode.add_argument("--check", action="store_true", help="只读检查路线，列出正式运行时要解除的旧硬件限位")
    args = parser.parse_args()
    if not 1 <= args.speed <= 1000 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 300:
        raise ValueError("速度应为 1～1000，超时应大于 0 且不超过 300 秒。")
    data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    validate_calibration(data)
    bus = None
    single = bool(args.target)
    print("输入 middle / zero / rotated / rest，或 ID1～ID6 的六个 0～1 数值。")
    print("范围以 calibration.json 的六组端点为准；首次运行请托稳，程序会卸力并解除旧硬件限位。")
    print("每次从实测当前位置开始；q 保持退出，托住后输入 off 卸力，Ctrl+C 急停卸力。")
    try:
        while True:
            command = " ".join(args.target) if single else input("目标> ").strip()
            if not command and not single:
                continue
            if command.lower() in ("q", "quit", "exit"):
                break
            if command.lower() == "off":
                if not args.dry_run and not args.check:
                    if bus is None:
                        bus = ServoBus(args.port or find_port())
                    release_torque(bus)
                else:
                    print("预览/检查模式不发送卸力指令。")
                if single:
                    break
                continue
            try:
                values = parse_target(command, data)
                targets = targets_from_values(data, values)
                if args.dry_run:
                    print("目标比例：", values)
                    print("原始目标：", targets)
                    print("仅换算预览，未连接舵机；当前位置和实际路线尚未检查。")
                else:
                    if bus is None:
                        bus = ServoBus(args.port or find_port())
                    move_arm(bus, data, values, args.speed, args.timeout, check_only=args.check)
            except ValueError as exc:
                if single:
                    raise
                print(str(exc))
            if single:
                break
    except KeyboardInterrupt as exc:
        if bus is not None and not args.check and not getattr(exc, "torque_released", False):
            release_torque(bus)
        print("\n已中止。")
    finally:
        if bus is not None:
            bus.close()
    print("程序已退出；正常退出不会自动复位或卸力。")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print("输入结束；保留当前扭矩状态，不自动复位。")
    except (OSError, ValueError, RuntimeError, serial.SerialException) as exc:
        print(f"控制停止：{exc}", file=sys.stderr)
        sys.exit(1)
