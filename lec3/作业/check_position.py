"""只读观察 SCS215 位置是否连续；默认检查 ID=2，不发送任何寄存器写入。

先关闭其他占用串口的程序。校准退出后扭矩应保持关闭。
运行后输入回车开始，托住机械臂，缓慢、小幅活动待检查关节；Ctrl+C 结束。
不要求摆到四个姿态或走完整个行程；看到明显跳变可立即停止。
读数自动保存为 diagnostics/position_id*_*.csv，不能作为校准结果。
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

from set_id import JOINTS, TORQUE_REGISTER, ServoBus, find_port, inspect

POSITION_REGISTER = 0x38
LOG_DIR = Path(__file__).with_name("diagnostics")


def classify(previous, current):
    """仅标记可疑读数，不用取模推算角度，也不自动认定是传感器死区。"""
    if not 0 <= current <= 1023:
        raise ValueError(f"位置不在 0～1023 内：{current}")
    delta = 0 if previous is None else current - previous
    notes = []
    if previous is not None and abs(delta) > 512:
        notes.append("大跳变：停止活动，检查记录")
    elif previous is not None and abs(delta) > 100:
        notes.append("相邻变化较大，需结合活动速度判断")
    if current <= 20 or current >= 1003:
        notes.append("接近反馈量程端点")
    return delta, "；".join(notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, choices=JOINTS, default=1)
    parser.add_argument("--port", help="默认自动寻找原转接板")
    args = parser.parse_args()
    print("只读观察位置：不改编号、限位或扭矩，不发送运动指令。")
    print("请先托稳机械臂。只需缓慢小幅活动待检查关节，不要强行通过卡点。")
    bus = ServoBus(args.port or find_port())
    try:
        # 不擅自释放已开启的扭矩，避免开程序就使机械臂失去支撑。
        enabled = [sid for sid in JOINTS if bus.read(sid, TORQUE_REGISTER) != b"\x00"]
        if enabled:
            raise RuntimeError(f"ID {enabled} 扭矩未关闭，本程序未修改状态。请先处理后再检查。")
        info = inspect(bus, args.id)
        print(f"ID={args.id} {JOINTS[args.id]}，当前位置 {info['position']}。")
        answer = input("托稳后按回车开始观察，输入其他内容退出：").strip()
        if answer:
            return
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"position_id{args.id}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        print("现在缓慢活动该关节；若出现跳变或阻力，停止活动。按 Ctrl+C 结束。")
        print(f"记录：{path}")
        try:
            with path.open("x", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["elapsed_s", "id", "position_raw", "delta_raw", "note"])
                start = time.monotonic()
                previous = None
                while True:
                    if bus.read(args.id, TORQUE_REGISTER) != b"\x00":
                        raise RuntimeError("检测到扭矩开启，停止读取，请勿继续手动活动。")
                    raw = int.from_bytes(bus.read(args.id, POSITION_REGISTER, 2), "big")
                    delta, note = classify(previous, raw)
                    elapsed = time.monotonic() - start
                    writer.writerow([f"{elapsed:.3f}", args.id, raw, delta, note])
                    stream.flush()
                    print(f"{elapsed:6.1f}s  ID={args.id}  位置={raw:4d}  变化={delta:+5d}  {note}",
                          flush=True)
                    previous = raw
                    if abs(delta) > 512:
                        print("已记录大跳变，观察结束。这不能单独证明死区，需结合实际活动过程判断。")
                        break
                    time.sleep(0.1)
        finally:
            print(f"本次读数保存在：{path}")
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
