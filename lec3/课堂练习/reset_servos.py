"""回到第二项练习保存的起点（相对角度 0°）。

直接运行：同步回到起点，检查到位，最后关闭扭矩。
加 --dry-run：只显示起点，不连接串口。
依赖同目录的 control_servos.py、set_range.py 和 servo_config.json。
此处“复位”仅指回到起点，不修改舵机 ID、角度限位或出厂参数。
"""

import argparse
import json

import control_servos
from set_range import CONFIG_PATH, find_port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只预览，不驱动舵机")
    parser.add_argument("--port", help="默认自动查找原 USB 转接板")
    args = parser.parse_args()

    if type(control_servos.SPEED) is not int or not 1 <= control_servos.SPEED <= 1000:
        raise ValueError("control_servos.py 中的 SPEED 必须是 1～1000 的整数")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    # 两个比例都固定为 0，不受控制脚本顶部用户输入值的影响。
    targets = control_servos.calculate_targets(config, {1: 0.0, 2: 0.0})
    for servo_id, position in targets.items():
        print(f"ID={servo_id}：回到记录的起点 {position}（相对 0°）", flush=True)

    if args.dry_run:
        print("仅预览：未连接串口，也未发送运动指令。")
        return

    control_servos.move_servos(args.port or find_port(), config, targets)
    print("两只舵机已回到起点容差范围内，扭矩已关闭。")


if __name__ == "__main__":
    main()
