import time
import serial

PORT = "COM9"
ROUNDS = 3          # 往返次数
INTERVAL = 2       # 每次到达目标位置前留出的时间（秒）


def write_register(address, values):
    body = [1, len(values) + 3, 3, address, *values]
    command = bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
    # 每条指令单独打开和关闭串口。
    with serial.Serial(PORT, 1_000_000, timeout=0.5, write_timeout=1) as ser:
        ser.reset_input_buffer()
        ser.write(command)
        reply = ser.read(6)
    if reply != bytes.fromhex("FF FF 01 02 00 FC"):
        raise RuntimeError(f"舵机未正常确认指令，返回：{reply.hex(' ')}")


def stop_servo():
    for attempt in range(3):
        try:
            write_register(0x28, [0])
            print("扭矩已关闭，舵机已确认。", flush=True)
            return
        except (serial.SerialException, RuntimeError):
            if attempt == 2:
                print("无法确认扭矩关闭，请手动切断舵机电源。", flush=True)
                raise
            time.sleep(1)


def main():
    try:
        write_register(0x28, [1])  # 打开扭矩
        for round_number in range(1, ROUNDS + 1):
            for position in (800, 400):
                write_register(0x2A, [position >> 8, position & 0xFF])
                print(f"第 {round_number}/{ROUNDS} 轮，目标位置：{position}", flush=True)
                time.sleep(INTERVAL)
    finally:
        # 正常结束或中途报错时，都尝试关闭扭矩。
        stop_servo()


if __name__ == "__main__":
    main()
