import time
import serial

with serial.Serial("COM9", 1_000_000, timeout=0.2) as ser:
    ser.write(bytes.fromhex("FF FF 01 04 03 28 01 CE"))  # 打开扭矩
    ser.read(6)
    ser.write(bytes.fromhex("FF FF 01 05 03 2A 01 90 3B"))  # 转到位置 400
    # ser.write(bytes.fromhex("FF FF 01 05 03 2A 03 20 A9"))  # 转到位置 800
    ser.read(6)
    time.sleep(2)
    ser.write(bytes.fromhex("FF FF 01 04 03 28 00 CF"))  # 关闭扭矩
