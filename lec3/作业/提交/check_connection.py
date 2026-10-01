"""只读检查 ID=1～6 的舵机连接，不改号、不控制转动。"""

import sys

import serial
from serial.tools import list_ports

PORT = None  # 默认自动识别；更换转接板时可手动填写，例如 "COM9"
BAUDRATE = 1_000_000
ADAPTER_SERIAL = "5B90151486"
JOINTS = {1: "底座旋转", 2: "肩部抬升", 3: "肘部弯曲",
          4: "手腕弯曲", 5: "手腕旋转", 6: "夹爪"}


def check_servo(ser, servo_id):
    # READ：读取 0x05 地址保存的 ID，共 1 字节。
    body = [servo_id, 4, 2, 0x05, 1]
    command = bytes([255, 255, *body, (~sum(body)) & 255])
    ser.reset_input_buffer()
    ser.write(command)
    reply = ser.read(7)
    if not reply:
        return "未响应"
    if (len(reply) != 7 or reply[:4] != bytes([255, 255, servo_id, 3])
            or (sum(reply[2:]) & 255) != 255):
        return f"应答数据异常：{reply.hex(' ')}"
    if reply[4] != 0:
        return f"有响应，但舵机报告错误：0x{reply[4]:02X}"
    if reply[5] != servo_id:
        return f"读回编号不一致：{reply[5]}"
    return None


def main():
    port = PORT
    if port is None:
        matches = [p.device for p in list_ports.comports()
                   if p.serial_number == ADAPTER_SERIAL]
        if len(matches) != 1:
            raise RuntimeError("未找到 USB 转接板，请检查连接或填写代码顶部的 PORT。")
        port = matches[0]
    print(f"串口：{port}，波特率：{BAUDRATE}，开始只读检查。", flush=True)
    failed = []
    with serial.Serial(port, BAUDRATE, timeout=0.3, write_timeout=1) as ser:
        for servo_id, joint in JOINTS.items():
            try:
                error = check_servo(ser, servo_id)
            except serial.SerialException as exc:
                error = str(exc)
            if error:
                failed.append(servo_id)
            print(f"ID={servo_id}（{joint}）：{error or '连接正常'}", flush=True)
    if failed:
        print(f"检查未通过的编号：{failed}")
        return 1
    print("检查通过：ID=1～6 全部正常响应，编号与校验和正确。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, serial.SerialException) as exc:
        print(f"连接检查失败：{exc}", file=sys.stderr)
        sys.exit(1)
