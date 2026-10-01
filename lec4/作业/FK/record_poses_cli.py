"""终端逐个记录姿态，只读舵机位置，不发送运动或扭矩指令。"""
import argparse
import json
import time

from action_store import ActionStore, CALIBRATION_PATH, POSE_LABELS
from arm_backend import JOINTS, LeRobotSCSBus, circular_delta, find_port, inspect, raw_to_ratio

ORDER = ("stand", "middle", "zero", "rotated", "rest", "left", "right", "lift")


def capture(bus, calibration):
    samples = []
    for _ in range(5):
        samples.append({sid: int.from_bytes(bus.read(sid, 56, 2), "big") for sid in JOINTS})
        time.sleep(0.1)
    positions = {}
    for sid in JOINTS:
        readings = [sample[sid] for sample in samples]
        first = readings[0]
        continuous = [first + circular_delta(raw, first) for raw in readings]
        if max(continuous) - min(continuous) > 3:
            raise ValueError(f"ID{sid} 尚未停稳：{readings}。请摆稳后重试。")
        positions[sid] = sorted(continuous)[2] % 1024
        raw_to_ratio(positions[sid], calibration["joints"][str(sid)])
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="默认自动识别转接板，例如 COM9")
    parser.add_argument("--start", choices=ORDER, default="stand", help="从指定姿态开始，方便中断后继续")
    args = parser.parse_args()
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8-sig"))
    store = ActionStore(calibration)
    port = args.port or find_port()
    bus = LeRobotSCSBus(port)
    try:
        enabled = []
        for sid in JOINTS:
            info = inspect(bus, sid)
            if info["model_bytes"] != "05 23":
                raise RuntimeError(f"ID{sid} 型号不匹配：{info['model_bytes']}")
            if info["torque"] != 0:
                enabled.append(sid)
        if enabled:
            raise RuntimeError(f"ID {enabled} 的扭矩尚未关闭。请先托稳机械臂，在控制界面卸力并退出窗口，再运行此脚本手摆录制。")
        print(f"已连接 {port}，六只舵机均在线且扭矩关闭。", flush=True)
        print("顺序：" + " → ".join(ORDER))
        print("摆好整条机械臂后按回车：读取六个实测位置并保存，已有同名姿态会被替换。")
        print("输入 s 跳过并保留原记录；q 或 Ctrl+C 退出。每个姿态立即保存，中途退出不会丢失已录内容。")
        print("本程序不发送运动指令，也不改变扭矩；请托稳机械臂。\n")
        saved = []
        for name in ORDER[ORDER.index(args.start):]:
            while True:
                answer = input(f"[{ORDER.index(name) + 1}/8] 摆好 {name}（{POSE_LABELS[name]}）后按回车：").strip().lower()
                if answer == "q":
                    print("已退出，之前保存的姿态保留。")
                    return
                if answer == "s":
                    print(f"已跳过 {name}，保留原记录。\n")
                    break
                if answer:
                    print("请直接按回车，或输入 s / q。")
                    continue
                if any(bus.read(sid, 40)[0] != 0 for sid in JOINTS):
                    raise RuntimeError("检测到扭矩开启，停止手摆录制。")
                try:
                    positions = capture(bus, calibration)
                except ValueError as exc:
                    print(f"未保存：{exc}")
                    continue
                if json.loads(CALIBRATION_PATH.read_text(encoding="utf-8-sig")) != calibration:
                    raise RuntimeError("录制期间校准文件发生变化，请重新运行。")
                store.record(name, positions)
                saved.append(name)
                print("原始位置：", [positions[sid] for sid in JOINTS])
                print("0～1 比例：", [round(x, 4) for x in store.document["poses"][name]["values"]])
                print(f"已保存 {name}。\n", flush=True)
                break
        print("录制结束，本次保存：" + ("、".join(saved) or "无"))
        print(f"文件：{store.path}")
        print("重新打开 control_fk.py 即可加载新姿态。")
    finally:
        bus.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n录制已取消，之前保存的姿态保留。")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"录制停止：{exc}")
        raise SystemExit(1)
