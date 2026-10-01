"""任务2：以当前位置为起点，设置 SCS215 的 90°/45° 运动范围。

只使用 pyserial 和标准库，不依赖 lerobot 或厂商 SDK。
直接运行只预览；加 --apply 写入；加 --check 只验证已保存的设置。
首次设置后再次运行会复用配置；--recapture 可重新采集当前位置。
全程保持扭矩关闭，不发送目标位置指令。
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

BAUDRATE = 1_000_000
ADAPTER_SERIAL = "5B90151486"
CONFIG_PATH = Path(__file__).with_name("servo_config.json")
RANGES_DEG = {1: 90, 2: 45}
MAX_POSITION = 1023
FULL_RANGE_DEG = 300.0
# 飞特 SCSCL 控制表；双字节数据按高字节在前编码。
MIN_LIMIT = 0x09
TORQUE = 0x28
EEPROM_LOCK = 0x30
PRESENT_POSITION = 0x38


def find_port():
    ports = [p.device for p in list_ports.comports()
             if p.serial_number == ADAPTER_SERIAL]
    if len(ports) != 1:
        raise RuntimeError("未找到原 USB 转接板，请检查连接或通过 --port 指定串口。")
    return ports[0]


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ServoBus:
    def __init__(self, port):
        self.serial = serial.Serial(port, BAUDRATE, timeout=0.5, write_timeout=1)

    def close(self):
        self.serial.close()

    def request(self, servo_id, instruction, parameters, data_length):
        body = bytes([servo_id, len(parameters) + 2, instruction, *parameters])
        packet = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
        self.serial.reset_input_buffer()
        self.serial.write(packet)
        reply = self.serial.read(data_length + 6)
        if (len(reply) != data_length + 6
                or reply[:4] != bytes([255, 255, servo_id, data_length + 2])
                or (sum(reply[2:]) & 255) != 255):
            raise RuntimeError(f"ID={servo_id} 无完整有效应答：{reply.hex(' ')}")
        if reply[4] != 0:
            raise RuntimeError(f"ID={servo_id} 报告状态错误：0x{reply[4]:02X}")
        return reply[5:-1]

    def read(self, servo_id, address, length):
        # 只读查询允许有限重试；写入失败则中止并明确报告。
        for attempt in range(3):
            try:
                return self.request(servo_id, 0x02, [address, length], length)
            except (RuntimeError, serial.SerialException):
                if attempt == 2:
                    raise
                time.sleep(0.1)

    def write(self, servo_id, address, values):
        self.request(servo_id, 0x03, [address, *values], 0)
        time.sleep(0.05)

    def snapshot(self, servo_id):
        stored_id = self.read(servo_id, 0x05, 1)[0]
        limits = self.read(servo_id, MIN_LIMIT, 4)
        motion = self.read(servo_id, PRESENT_POSITION, 4)
        if stored_id != servo_id:
            raise RuntimeError("舵机编号读回不一致")
        return {
            "position": int.from_bytes(motion[:2], "big"),
            "speed_raw": int.from_bytes(motion[2:], "big"),
            "min_position": int.from_bytes(limits[:2], "big"),
            "max_position": int.from_bytes(limits[2:], "big"),
            "torque": self.read(servo_id, TORQUE, 1)[0],
            "lock": self.read(servo_id, EEPROM_LOCK, 1)[0],
        }


def make_range(snapshot, degrees):
    start = snapshot["position"]
    low, high = snapshot["min_position"], snapshot["max_position"]
    if not 0 <= low <= start <= high <= MAX_POSITION or low == high:
        raise RuntimeError("当前位置或原有限位异常，不能自动设置。")
    span = round(degrees * MAX_POSITION / FULL_RANGE_DEG)
    # 优先沿数值增加方向；空间不足时反向，且不扩大原有限位。
    if start + span <= high:
        end = start + span
    elif start - span >= low:
        end = start - span
    else:
        raise RuntimeError("从当前位置向任一方向均无足够行程，请调整起点。")
    return {
        "start_position": start,
        "end_position": end,
        "min_position": min(start, end),
        "max_position": max(start, end),
        "direction": 1 if end > start else -1,
        "requested_range_deg": degrees,
        "nominal_range_deg": span * FULL_RANGE_DEG / MAX_POSITION,
    }


def verify(bus, servo_id, item):
    result = bus.snapshot(servo_id)
    if (result["min_position"] != item["min_position"]
            or result["max_position"] != item["max_position"]
            or result["lock"] != 1 or result["torque"] != 0):
        raise RuntimeError(f"ID={servo_id} 限位/锁/扭矩读回验证失败：{result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="不指定时按 USB 序列号查找串口")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="写入并验证角度限位")
    mode.add_argument("--check", action="store_true", help="只读取并验证已有配置")
    parser.add_argument("--recapture", action="store_true", help="重新采集起点")
    args = parser.parse_args()
    port = args.port or find_port()
    bus = ServoBus(port)
    try:
        before = {str(sid): bus.snapshot(sid) for sid in RANGES_DEG}
        if CONFIG_PATH.exists() and not args.recapture:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            if args.check:
                raise RuntimeError("没有可验证的已有配置。")
            config = {
                "model": "SCS215", "baudrate": BAUDRATE,
                "adapter_serial": ADAPTER_SERIAL, "port": port,
                "position_max": MAX_POSITION, "nominal_full_range_deg": FULL_RANGE_DEG,
                "mapping": "round(start_position + x * (end_position - start_position))",
                "servos": {str(sid): make_range(before[str(sid)], deg)
                           for sid, deg in RANGES_DEG.items()},
            }
        for sid in RANGES_DEG:
            item = config["servos"][str(sid)]
            a, b = item["start_position"], item["end_position"]
            if not (0 <= min(a, b) < max(a, b) <= MAX_POSITION
                    and item["min_position"] == min(a, b)
                    and item["max_position"] == max(a, b)
                    and abs(a-b) == round(RANGES_DEG[sid]*MAX_POSITION/FULL_RANGE_DEG)):
                raise RuntimeError(f"ID={sid} 配置范围不合法")
            print(f"ID={sid}: 起点 {a} -> 终点 {b}，"
                  f"范围约 {item['nominal_range_deg']:.3f}°", flush=True)
        if args.check:
            for sid in RANGES_DEG:
                print(f"ID={sid} 验证通过：{verify(bus, sid, config['servos'][str(sid)])}")
            return
        if not args.apply:
            print("以上为预览。使用 --apply 写入设置；不会发送转动指令。")
            return
        for state in before.values():
            if state["torque"] != 0 or state["speed_raw"] != 0:
                raise RuntimeError("请先让两只舵机停止并关闭扭矩，再采集和设置限位。")
        for sid in RANGES_DEG:
            item = config["servos"][str(sid)]
            if not item["min_position"] <= before[str(sid)]["position"] <= item["max_position"]:
                raise RuntimeError(f"ID={sid} 当前位置不在拟设置范围内。")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        record_path = CONFIG_PATH.with_name(f"range_setup_{stamp}.json")
        record = {"time": datetime.now().isoformat(), "port": port,
                  "before": before, "planned": config, "status": "in_progress", "after": {}}
        save_json(record_path, record)  # 改动前保存原限位，便于恢复和写报告。
        try:
            for sid in RANGES_DEG:
                item = config["servos"][str(sid)]
                try:
                    bus.write(sid, EEPROM_LOCK, [0])
                    limits = (item["min_position"].to_bytes(2, "big")
                              + item["max_position"].to_bytes(2, "big"))
                    bus.write(sid, MIN_LIMIT, limits)
                finally:
                    # 无论限位写入是否成功，都尝试重新锁定 EEPROM。
                    bus.write(sid, EEPROM_LOCK, [1])
                record["after"][str(sid)] = verify(bus, sid, item)
                print(f"ID={sid}: 限位读回正确，EEPROM 已锁定，扭矩关闭。", flush=True)
            config["verified_at"] = datetime.now().isoformat()
            config["port"] = port
            save_json(CONFIG_PATH, config)
            record["status"] = "verified"
        except Exception as exc:
            record["status"] = "failed_or_partial"
            record["error"] = str(exc)
            raise
        finally:
            save_json(record_path, record)
        print(f"配置已保存：{CONFIG_PATH}\n实验记录：{record_path}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
