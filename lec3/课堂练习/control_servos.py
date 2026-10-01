"""任务3：用两个 0～1 的数字同步控制两只 SCS215。

修改下方 SERVO_1_VALUE、SERVO_2_VALUE 后直接运行即可。
依赖同目录的 set_range.py、servo_config.json，仅使用 pyserial 和标准库。
--dry-run 只显示目标位置，不打开串口；正常运行结束后关闭两只舵机的扭矩。
"""

# ===== 只需修改这两个数字 =====
SERVO_1_VALUE = 0.5  # ID=1：0 对应起点，1 对应相对 90°
SERVO_2_VALUE = 0.5  # ID=2：0 对应起点，1 对应相对 45°

SPEED = 100          # SCS215 速度寄存器数值，不代表角度/秒；0 表示不限速，故不使用
TIMEOUT_SECONDS = 10
POSITION_TOLERANCE = 6  # 允许的位置误差（原始位置单位）

import argparse
import json
import math
import time

from set_range import CONFIG_PATH, PRESENT_POSITION, TORQUE, ServoBus, find_port, verify

GOAL_POSITION = 0x2A


def calculate_targets(config, values):
    """将比例映射到保存的起点/终点，自动支持第二只舵机的反方向。"""
    if config.get("model") != "SCS215" or not config.get("verified_at"):
        raise ValueError("请先运行 set_range.py --apply，生成已验证的 SCS215 配置。")
    targets = {}
    for servo_id, value in values.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"ID={servo_id} 的输入必须是 0～1 的有限数字，收到 {value!r}")
        item = config["servos"][str(servo_id)]
        start, end = item["start_position"], item["end_position"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= min(start, end) < max(start, end) <= 1023
                or item["min_position"] != min(start, end)
                or item["max_position"] != max(start, end)):
            raise ValueError(f"ID={servo_id} 的起点、终点或限位配置不合法")
        targets[servo_id] = round(start + value * (end - start))
    return targets


def sync_packet(targets):
    """SYNC WRITE：广播 ID=FE，指令=83，每只舵机写入位置/时间/速度六字节。"""
    parameters = [GOAL_POSITION, 6]
    for servo_id, position in targets.items():
        parameters += [servo_id, *position.to_bytes(2, "big"),
                       0, 0, *SPEED.to_bytes(2, "big")]
    body = bytes([0xFE, len(parameters) + 2, 0x83, *parameters])
    return b"\xff\xff" + body + bytes([(~sum(body)) & 255])


def sync_positions(bus, targets):
    bus.serial.reset_input_buffer()
    bus.serial.write(sync_packet(targets))
    bus.serial.flush()
    time.sleep(0.05)
    # 广播同步写无应答，逐个读回目标寄存器，确认两只都收到了。
    for servo_id, expected in targets.items():
        actual = int.from_bytes(bus.read(servo_id, GOAL_POSITION, 2), "big")
        if actual != expected:
            raise RuntimeError(f"ID={servo_id} 目标写入失败：期望 {expected}，读回 {actual}")


def wait_until_reached(bus, targets):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    stable_samples = 0
    last_positions = {}
    while time.monotonic() < deadline:
        reached = True
        for servo_id, target in targets.items():
            data = bus.read(servo_id, PRESENT_POSITION, 4)
            position = int.from_bytes(data[:2], "big")
            speed = int.from_bytes(data[2:], "big") & 0x7FFF
            last_positions[servo_id] = position
            reached &= abs(position - target) <= POSITION_TOLERANCE and speed == 0
        stable_samples = stable_samples + 1 if reached else 0
        if stable_samples >= 3:
            return last_positions
        time.sleep(0.1)
    raise TimeoutError(f"等待到位超时，目标：{targets}，最后读到：{last_positions}")


def disable_torque(port, servo_ids):
    """独立重开串口关闭扭矩；一只失败仍尝试另一只，最多尝试三次。"""
    failures = []
    for servo_id in servo_ids:
        for attempt in range(3):
            cleanup_bus = None
            try:
                cleanup_bus = ServoBus(port)
                cleanup_bus.write(servo_id, TORQUE, [0])
                if cleanup_bus.read(servo_id, TORQUE, 1) != b"\x00":
                    raise RuntimeError("扭矩关闭读回失败")
                print(f"ID={servo_id} 扭矩已关闭。", flush=True)
                break
            except Exception as exc:
                if attempt == 2:
                    failures.append(f"ID={servo_id}: {exc}")
            finally:
                if cleanup_bus is not None:
                    cleanup_bus.close()
            time.sleep(0.2)
    if failures:
        raise RuntimeError("无法确认扭矩关闭，请手动断电：" + "; ".join(failures))


def move_servos(port, config, targets):
    bus = ServoBus(port)
    torque_may_be_on = False
    try:
        current = {}
        # 先检查两只舵机均在线、硬件限位与配置一致、扭矩关闭。
        for servo_id in targets:
            item = config["servos"][str(servo_id)]
            state = verify(bus, servo_id, item)
            if state["speed_raw"] != 0:
                raise RuntimeError(f"ID={servo_id} 仍在运动，请静止后运行")
            current[servo_id] = state["position"]
            if not item["min_position"] <= current[servo_id] <= item["max_position"]:
                raise RuntimeError(f"ID={servo_id} 当前位置在设置范围之外，请关闭扭矩后调整")
        # 先在扭矩关闭时将旧目标覆盖为当前位置，避免开启扭矩时先跑向旧目标。
        sync_positions(bus, current)
        torque_may_be_on = True
        for servo_id in targets:
            bus.write(servo_id, TORQUE, [1])
        # 同一个数据包下发两个不同目标；同步下发不保证同时到达。
        sync_positions(bus, targets)
        print("两个目标已同步下发并读回确认，正在等待到位……", flush=True)
        actual = wait_until_reached(bus, targets)
        for servo_id in targets:
            print(f"ID={servo_id}：目标 {targets[servo_id]}，实测 {actual[servo_id]}")
    finally:
        try:
            bus.close()
        finally:
            if torque_may_be_on:
                disable_torque(port, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只计算，不连接舵机")
    parser.add_argument("--port", help="默认自动查找原 USB 转接板")
    args = parser.parse_args()
    if type(SPEED) is not int or not 1 <= SPEED <= 1000:
        raise ValueError("SPEED 必须是 1～1000 的整数")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    values = {1: SERVO_1_VALUE, 2: SERVO_2_VALUE}
    targets = calculate_targets(config, values)
    for servo_id, target in targets.items():
        degrees = values[servo_id] * config["servos"][str(servo_id)]["requested_range_deg"]
        print(f"ID={servo_id}：输入 {values[servo_id]}，相对角度约 {degrees:g}°，目标位置 {target}")
    if args.dry_run:
        print("仅预览：未连接串口，也未发送运动指令。")
        return
    move_servos(args.port or find_port(), config, targets)


if __name__ == "__main__":
    main()
