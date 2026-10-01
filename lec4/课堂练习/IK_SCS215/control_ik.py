"""SCS215 Cartesian control. Windows lerobot environment; no Placo required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from robot_adapter import HERE, NAMES, Calibration, Robot, Simulation, load_config, make_bus
from trajectory import Controller, MotionStopped

DIRECTIONS = {"forward": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0),
              "right": (0, -1, 0), "up": (0, 0, 1), "down": (0, 0, -1)}
KEYS = dict(zip("wsadrf", ("forward", "back", "left", "right", "up", "down")))
HELP = """固定底座坐标：前 +X，左 +Y，上 +Z；距离单位 cm。
  up 5 / down 5 / forward 5 / back 5 / left 5 / right 5
  move 0 0 5   三个分量依次为前、左、上
  speed 1      末端推进速度上限，0.1～3 cm/s；跟不上会自动放慢（单独输入一行）
  step 1       键盘每次移动距离，0.1～5 cm；不改变轨迹执行频率
  jog          键盘 W/S 前后，A/D 左右，R/F 上下，Q 返回
  status       六个读数、0～1比例、模型末端坐标
  diagnose     只读目标、位置、扭矩、电压、P/D等
  stop         停止并保持当前读数；不卸力
  off          六个关节全部卸力（先托稳机械臂）
  quit         退出，保留当前目标和扭矩状态
运动中：空格 / Esc / Q / Ctrl+C 停止并保持；不会积压运动按键。
"""


def cancel_key():
    if sys.platform != "win32" or not sys.stdin.isatty():
        return False
    import msvcrt
    stop = False
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
        elif key in (" ", "\x1b", "\x03", "q", "Q"):
            stop = True
    return stop


def displacement(tokens):
    if len(tokens) == 2 and tokens[0] in DIRECTIONS:
        number = float(tokens[1])
        if not np.isfinite(number) or number <= 0:
            raise ValueError("距离用有限正数，单位cm。")
        return np.array(DIRECTIONS[tokens[0]]) * number / 100
    if len(tokens) == 4 and tokens[0] == "move":
        delta = np.array([float(s) for s in tokens[1:]]) / 100
        if np.isfinite(delta).all():
            return delta
    raise ValueError("例如 up 5 或 move 0 0 5；speed 1 需要单独输入。")


def status(session):
    raw = session.read_positions()
    q = session.mapping.to_q(raw)
    print("ID1～6 原始读数：", raw.tolist())
    print("ID1～6 0～1比例：", np.round(session.mapping.ratios(raw), 4).tolist())
    print("ID1～5 模型角度（°）：", np.round(np.rad2deg(q), 2).tolist())
    print("模型估计夹爪坐标（mm）：", np.round(session.mapping.arm.fk(q)[:3, 3]*1000, 2).tolist())


def save_report(controller):
    if controller.last_report is not None:
        name = "last_simulation.json" if controller.session.simulated else "last_motion.json"
        (HERE / name).write_text(json.dumps(controller.last_report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def do_move(controller, delta, quiet=False):
    try:
        report = controller.move(delta, cancel_key, emit=(lambda _: None) if quiet else print)
    finally:
        save_report(controller)
    print("完成（模型反馈估计）" if report["success_model_estimate"] else "未完整达到原直线目标")
    print("请求位移mm：", np.round(report["requested_mm"], 2).tolist(),
          "反馈估计mm：", np.round(report["model_feedback_delta_mm"], 2).tolist())
    print(f"终点误差 {report['endpoint_error_mm']:.1f} mm；横向偏差 {report['max_lateral_feedback_mm']:.1f} mm；"
          f"对最终IK参考的计数差 {report['final_count_error']}；{report['reason']}")
    print(f"用时 {report['elapsed_s']:.1f} 秒；参考轨迹推进 {report['reference_fraction']*100:.0f}% "
          "（参考推进不是实际到位，以上实测位移才是反馈）。")
    if report["reason"] == "target_not_reached":
        print("最终补偿目标仍然保留，未自动卸力。可用 diagnose 查看；stop 停止保持，off 卸力。")
    if report.get("stopped_and_held"):
        print("已取消剩余轨迹并保持当时读数；未卸力。diagnose 可查看各关节状态。")
    return report


def jog(controller, step_cm):
    if sys.platform != "win32" or not sys.stdin.isatty():
        raise ValueError("jog 需要 Windows 的交互式 CMD / PowerShell。")
    import msvcrt
    print(f"键盘：W/S前后，A/D左右，R/F上下，每次 {step_cm:g}cm；Q返回。")
    while True:
        key = msvcrt.getwch().lower()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        if key in ("q", "\x03"):
            return
        if key in (" ", "\x1b"):
            controller.session.hold()
            print("已停止并保持。")
        if key in KEYS:
            try:
                # Keep all IK/no-solution messages visible in keyboard mode.
                do_move(controller, np.array(DIRECTIONS[KEYS[key]]) * step_cm / 100)
            except MotionStopped:
                print("已停止并保持，Q返回命令行。")


def check(mapping):
    # Instantiate only: checks SDK and TA source loading without opening COM.
    bus, source = make_bus(mapping, "OFFLINE")
    print("离线检查通过；未连接串口。NumPy IK，不使用 Placo。")
    print("助教驱动版本：", source["commit"])
    print("实际驱动文件：", sys.modules[type(bus).__module__].__file__)
    print("标定来源：", mapping.source)
    print(f"每读数 {mapping.degrees_per_count:.9f}°，1024对应 {mapping.degrees_per_count*1024:g}°。")
    for i in range(5):
        print(f"ID{i+1} {NAMES[i]}：0/1端点 {int(mapping.minimum[i])}/{int(mapping.maximum[i])}，"
              f"零位 {int(mapping.zero[i])}，与URDF交集 {mapping.raw_lower[i]}～{mapping.raw_upper[i]}")
    print("角度比例沿用当前标定；300/320/330的物理比例尚待准确测量。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="检查依赖和标定；不连硬件")
    mode.add_argument("--read-only", action="store_true", help="连接并只读检查；不改扭矩")
    mode.add_argument("--simulate", action="store_true", help="离线整数读数模拟；不连硬件")
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--port", help="可选，例如COM9；默认按转接板序列号寻找")
    parser.add_argument("--command", help='执行一条命令，例如 "up 5"')
    parser.add_argument("--start-raw", nargs=6, type=int, help="仅模拟使用的六个起点读数")
    parser.add_argument("--deadband", type=int, default=0, help="模拟舵机死带计数，默认0")
    args = parser.parse_args()
    if args.command is not None and (not args.command.strip() or args.check or args.read_only):
        parser.error("--command 不能是空命令，也不能与 --check/--read-only 同时使用。")
    if (args.start_raw is not None or args.deadband != 0) and not args.simulate:
        parser.error("--start-raw/--deadband 仅用于 --simulate。")
    if not 0 <= args.deadband <= 20:
        parser.error("--deadband 范围是0～20。")
    cfg = load_config(args.config)
    if args.port:
        cfg["port"] = args.port
    mapping = Calibration(cfg)
    if args.check:
        check(mapping)
        return 0
    session = Simulation(mapping, args.start_raw, args.deadband) if args.simulate else Robot(mapping)
    controller = Controller(session, realtime=not args.simulate)
    try:
        print("离线模拟，未连接机械臂。" if args.simulate else "已连接；当前仅读取。输入运动命令后才使能，从实测当前位置开始。")
        print(f"角度比例：{mapping.degrees_per_count:g}°/读数；末端模型：gripper_frame_link；夹爪保持当前开合。")
        print("执行器 v3：反馈调速 + 保持关节解分支 + 有限积分补偿；不会按时间把剩余目标强行发完。")
        status(session)
        if args.read_only:
            print(json.dumps(session.snapshot(), ensure_ascii=False, indent=2))
            return 0
        print(HELP)
        step = cfg["jog_step_cm"]
        while True:
            try:
                line = args.command if args.command is not None else input("IK-SCS215> ")
                tokens = shlex.split(line.strip().lower())
                if not tokens:
                    continue
                if tokens in (["quit"], ["exit"]):
                    break
                if tokens == ["status"]:
                    status(session)
                elif tokens == ["help"]:
                    print(HELP)
                elif tokens == ["diagnose"]:
                    print(json.dumps(session.snapshot(), ensure_ascii=False, indent=2))
                    print("Present_Voltage单位0.1V；Present_Load为原始驱动反馈，不是力矩测量。")
                elif tokens == ["off"]:
                    session.off()
                    print("六关节扭矩均已关闭。")
                elif tokens == ["stop"]:
                    session.hold()
                    print("已停止并保持。")
                elif tokens == ["jog"]:
                    jog(controller, step)
                elif len(tokens) == 2 and tokens[0] in ("speed", "step"):
                    value = float(tokens[1])
                    upper = 3 if tokens[0] == "speed" else 5
                    if not np.isfinite(value) or not .1 <= value <= upper:
                        raise ValueError(f"允许范围0.1～{upper}。")
                    if tokens[0] == "speed":
                        controller.speed_cm_s = value
                        print(f"末端推进速度上限：{value:g} cm/s；实际跟随后自动调速。")
                    else:
                        step = value
                        print(f"键盘移动距离：{value:g} cm")
                else:
                    report = do_move(controller, displacement(tokens))
                    if args.command is not None and not report["success_model_estimate"]:
                        return 2
            except KeyboardInterrupt:
                session.hold()
                print("\n已停止并保持；off 可卸力，quit 可退出。")
                if args.command is not None:
                    return 130
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"本次停止：{exc}")
                if args.command is not None:
                    return 1
            if args.command is not None:
                break
    finally:
        session.close()
        print("串口已关闭；未自动复位或卸力。" if not args.simulate else "模拟结束。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"无法启动：{exc}")
        raise SystemExit(1)
