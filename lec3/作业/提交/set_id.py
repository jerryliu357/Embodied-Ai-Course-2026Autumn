"""给单独连接的 SCS215 编号。只使用 pyserial 和标准库。

直接运行后输入目标 ID（1～6）；也可使用 --id 3。
每次必须只连接一只舵机：扫描无法识别多个具有相同 ID 的舵机。
--check 只扫描、读取状态，不修改编号。不会发送位置或开启扭矩指令。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

BAUDRATE = 1_000_000
ADAPTER_SERIAL = "5B90151486"
ID_REGISTER = 0x05
TORQUE_REGISTER = 0x28
LOCK_REGISTER = 0x30  # SCS215/SCSCL；不能用于 STS3215（锁地址不同）
JOINTS = {1: "底座旋转", 2: "肩部抬升", 3: "肘部弯曲",
          4: "手腕弯曲", 5: "手腕旋转", 6: "夹爪"}


def find_port():
    ports = list(list_ports.comports())
    matches = [p.device for p in ports if p.serial_number == ADAPTER_SERIAL]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(f"{p.device} ({p.description})" for p in ports)
    raise RuntimeError(f"未找到原 USB 转接板，可用 --port COM号码 指定。当前串口：{available or '无'}")


def packet(servo_id, instruction, parameters):
    body = bytes([servo_id, len(parameters) + 2, instruction, *parameters])
    return b"\xff\xff" + body + bytes([(~sum(body)) & 255])


class ServoBus:
    def __init__(self, port):
        self.serial = serial.Serial(port, BAUDRATE, timeout=0.3, write_timeout=1)

    def close(self):
        self.serial.close()

    def exchange(self, servo_id, instruction, parameters, data_length, optional=False,
                 allowed_ids=None):
        self.serial.reset_input_buffer()
        self.serial.write(packet(servo_id, instruction, parameters))
        reply = self.serial.read(data_length + 6)
        if not reply and optional:
            return None
        allowed_ids = allowed_ids or {servo_id}
        if (len(reply) != data_length + 6 or reply[:2] != b"\xff\xff"
                or reply[2] not in allowed_ids or reply[3] != data_length + 2
                or (sum(reply[2:]) & 255) != 255):
            raise RuntimeError(f"ID={servo_id} 应答超时或损坏：{reply.hex(' ')}")
        if reply[4]:
            raise RuntimeError(f"ID={servo_id} 状态报错：0x{reply[4]:02X}")
        return reply[5:-1]

    def read(self, servo_id, address, length=1):
        return self.exchange(servo_id, 2, [address, length], length)

    def write_checked(self, servo_id, address, value):
        # 某些返回级别不回复 WRITE，因此以 READ 读回结果为准。
        self.exchange(servo_id, 3, [address, value], 0, optional=True)
        time.sleep(0.05)
        if self.read(servo_id, address) != bytes([value]):
            raise RuntimeError(f"ID={servo_id} 地址 {address:#04x} 写入后读回不一致")

    def scan(self):
        found = []
        previous_timeout = self.serial.timeout
        self.serial.timeout = 0.03
        try:
            for servo_id in range(254):  # 逐一查询，不使用广播改号。
                if self.exchange(servo_id, 1, [], 0, optional=True) is not None:
                    found.append(servo_id)
        finally:
            self.serial.timeout = previous_timeout
        return found


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


def relock(bus, candidates):
    """改号中断时，尝试用新/旧编号找到舵机并重新锁定。"""
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            if bus.read(candidate, ID_REGISTER) != bytes([candidate]):
                continue
            bus.write_checked(candidate, LOCK_REGISTER, 1)
            return candidate
        except (RuntimeError, serial.SerialException) as exc:
            errors.append(str(exc))
    raise RuntimeError("未能确认 EEPROM 重新锁定：" + "; ".join(errors))


def change_id(bus, old_id, new_id):
    # 关闭扭矩，避免改号时保持驱动力；不改变目标位置、角度限位或波特率。
    bus.write_checked(old_id, TORQUE_REGISTER, 0)
    if old_id == new_id:
        bus.write_checked(old_id, LOCK_REGISTER, 1)
        return
    try:
        bus.write_checked(old_id, LOCK_REGISTER, 0)
        # 固件可能用旧 ID 或新 ID 回复改号；也可能不回复写指令。
        bus.exchange(old_id, 3, [ID_REGISTER, new_id], 0, optional=True,
                     allowed_ids={old_id, new_id})
        time.sleep(0.1)
        if bus.read(new_id, ID_REGISTER) != bytes([new_id]):
            raise RuntimeError("新 ID 读回不一致")
    finally:
        relock(bus, [new_id, old_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, choices=range(1, 7), help="目标编号 1～6")
    parser.add_argument("--port", help="默认通过 USB 序列号自动查找")
    parser.add_argument("--check", action="store_true", help="只查询，不写入")
    args = parser.parse_args()
    print("每次只连接一只 SCS215；不要将它继续串接到其他舵机。", flush=True)
    new_id = args.id
    if not args.check and new_id is None:
        for sid, joint in JOINTS.items():
            print(f"  {sid}: {joint}")
        new_id = int(input("给当前这只舵机设置编号（1～6）：").strip())
        if new_id not in JOINTS:
            raise ValueError("编号必须为 1～6")
    port = args.port or find_port()
    bus = ServoBus(port)
    try:
        print(f"串口 {port}，波特率 {BAUDRATE}。正在扫描，约需 10 秒……", flush=True)
        found = bus.scan()
        if len(found) != 1:
            raise RuntimeError(f"检测到编号 {found}；需要且只能有一只舵机，未执行改号。")
        old_id = found[0]
        before = inspect(bus, old_id)
        print("当前状态：", json.dumps(before, ensure_ascii=False), flush=True)
        if args.check:
            print("只读检查完成，没有修改配置。")
            return
        if before["speed_raw"] != 0:
            raise RuntimeError("舵机仍在运动，请静止后再编号。")
        print(f"设置 ID {old_id} -> {new_id}（{JOINTS[new_id]}）", flush=True)
        change_id(bus, old_id, new_id)
        after = inspect(bus, new_id)
        if after["lock"] != 1 or after["torque"] != 0:
            raise RuntimeError("编号后的锁或扭矩状态验证失败")
        if old_id != new_id and bus.exchange(old_id, 1, [], 0, optional=True) is not None:
            raise RuntimeError("旧编号仍有响应，可能连接了多只舵机，请检查连接")
        print(f"成功：ID={new_id}，读回验证通过，EEPROM 已锁定，扭矩关闭。", flush=True)
        record = {"time": datetime.now().isoformat(), "port": port,
                  "joint": JOINTS[new_id], "before": before, "after": after}
        log_path = Path(__file__).with_name("id_setup_log.jsonl")
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"操作记录：{log_path}")
        except OSError as exc:
            print(f"硬件改号已成功，但记录文件写入失败：{exc}", file=sys.stderr)
    finally:
        bus.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, serial.SerialException, EOFError) as exc:
        print(f"操作未完成：{exc}", file=sys.stderr)
        print("请保持单只连接，使用 --check 查询当前编号后再操作。", file=sys.stderr)
        sys.exit(1)
