"""只读观察 SCS215 位置是否连续；默认检查 ID=5，不发送任何寄存器写入。

先关闭其他占用串口的程序。校准退出后扭矩应保持关闭。
运行后输入回车开始，托住机械臂，缓慢、小幅活动待检查关节；Ctrl+C 结束。
不要求摆到四个姿态或走完整个行程；看到明显跳变可立即停止。
只在终端显示，不保存 CSV、JSON 或其他测量文件；Ctrl+C 结束。
"""

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
# 允许从任意工作目录直接运行 settings 中的脚本。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lec3" / "作业"))
import serial

from set_id import JOINTS, TORQUE_REGISTER, ServoBus, find_port, inspect

POSITION_REGISTER = 0x38


def classify(previous, current):
    """仅标记可疑读数，不用取模推算角度，也不自动认定是传感器死区。"""
    if not 0 <= current <= 1023:
        raise ValueError(f"位置不在 0～1023 内：{current}")
    delta = 0 if previous is None else current - previous
    notes = []
    if previous is not None and abs(delta) > 512:
        notes.append("大跳变：请停止活动并观察；仅凭跳变不能认定死区")
    elif previous is not None and abs(delta) > 100:
        notes.append("相邻变化较大，需结合活动速度判断")
    if current <= 20 or current >= 1003:
        notes.append("接近反馈量程端点")
    return delta, "；".join(notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, choices=JOINTS, default=5)
    parser.add_argument("--port", help="默认自动寻找原转接板")
    args = parser.parse_args()
    print("只读观察位置：不改编号、限位或扭矩，不发送运动指令。")
    print("仅终端显示，不保存测量文件。")
    print("请先托稳机械臂。只需缓慢小幅活动待检查关节，不要强行通过卡点。")
    bus = ServoBus(args.port or find_port())
    try:
        # 不擅自释放已开启的扭矩，避免开程序就使机械臂失去支撑。
        if bus.read(args.id, TORQUE_REGISTER) != b"\x00":
            raise RuntimeError(f"ID {args.id} 扭矩未关闭；本程序不会自动卸力，请先托稳并卸力。")
        info = inspect(bus, args.id)
        print(f"ID={args.id} {JOINTS[args.id]}，当前位置 {info['position']}。")
        answer = input("托稳后按回车开始观察，输入其他内容退出：").strip()
        if answer:
            return
        print("现在缓慢活动该关节；若出现跳变或阻力，停止活动。按 Ctrl+C 结束。")
        start = time.monotonic()
        previous = None
        while True:
            if bus.read(args.id, TORQUE_REGISTER) != b"\x00":
                raise RuntimeError("检测到扭矩开启，停止读取，请勿继续手动活动。")
            raw = int.from_bytes(bus.read(args.id, POSITION_REGISTER, 2), "big")
            delta, note = classify(previous, raw)
            print(f"{time.monotonic()-start:6.1f}s  ID={args.id}  位置={raw:4d}  变化={delta:+5d}  {note}",
                  flush=True)
            previous = raw
            time.sleep(0.1)
    finally:
        bus.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已结束，未改变舵机状态。")
    except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
        print(f"观察停止：{exc}", file=sys.stderr)
        sys.exit(1)
