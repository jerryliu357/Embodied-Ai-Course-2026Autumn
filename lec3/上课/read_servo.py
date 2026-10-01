import serial                    # 这是一个专门处理串口通信的 python 库

# 根据你的串口地址填写
with serial.Serial("COM9", 1_000_000, timeout=0.2) as ser:
    ser.write(bytes.fromhex("FF FF 01 04 02 38 02 BE"))
    reply = ser.read(8)

position = (reply[5] << 8) | reply[6] # 大端格式，高八位左移 8 位（相当于乘 256）再拼接低八位
print("当前位置：", position)
