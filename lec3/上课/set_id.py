import time
import serial

PORT = "COM9"
OLD_ID = 1
NEW_ID = 2

def write_command(servo_id, address, value):
    body = [servo_id, 4, 3, address, value]
    checksum = (~sum(body)) & 0xFF
    return bytes([0xFF, 0xFF, *body, checksum])


with serial.Serial(PORT, 1_000_000, timeout=0.5, write_timeout=1) as ser:
    commands = [
        write_command(OLD_ID, 0x28, 0),  # 关闭扭矩
        write_command(OLD_ID, 0x30, 0),  # 解锁 EEPROM
        write_command(OLD_ID, 0x05, NEW_ID),  # 写入新 ID
        write_command(NEW_ID, 0x30, 1),  # 使用新 ID 锁定 EEPROM
    ]

    for command in commands:
        ser.reset_input_buffer()
        ser.write(command)
        ser.read(6)
        time.sleep(0.1)

    # 通过新 ID 读回编号和 EEPROM 锁，确认修改确实成功。
    for address, expected in ((0x05, NEW_ID), (0x30, 1)):
        body = [NEW_ID, 4, 2, address, 1]
        ser.reset_input_buffer()
        ser.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
        reply = ser.read(7)
        if not (
            len(reply) == 7
            and reply[:5] == bytes([0xFF, 0xFF, NEW_ID, 3, 0])
            and reply[5] == expected
            and (sum(reply[2:]) & 0xFF) == 0xFF
        ):
            raise RuntimeError(f"读回验证失败，地址 {address:#04x}，返回：{reply.hex(' ')}")

print(f"已读回确认：舵机 ID 为 {NEW_ID}，EEPROM 已锁定。")
