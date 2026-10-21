"""实时显示六个舵机读数，只读；Ctrl+C退出，不改变扭矩状态。"""

import argparse
import os
from pathlib import Path
import sys
import time

from exercise_paths import ROBOT_DIR

sys.path.insert(0, str(ROBOT_DIR))
from robot_adapter import NAMES, find_port, load_config, make_bus
from task3 import Calibration
from task3_hardware import read_diagnostic

LABELS = ("底座旋转", "肩部抬升", "肘部弯曲", "手腕弯曲", "手腕旋转", "夹爪开合")


def enable_terminal():
    if os.name != "nt":
        return sys.stdout.isatty()
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.windll.kernel32
    kernel.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel.GetStdHandle.restype = wintypes.HANDLE
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = kernel.GetStdHandle(-11)
    mode = wintypes.DWORD()
    return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                and kernel.SetConsoleMode(handle, mode.value | 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="如COM9；默认自动查找原转接板")
    parser.add_argument("--once", action="store_true", help="只读一次后退出")
    args = parser.parse_args()
    bus = None
    ansi = False
    try:
        cfg = load_config()
        if args.port:
            cfg["port"] = args.port
        mapping = Calibration(cfg)
        port = find_port(cfg)
        bus, _ = make_bus(mapping, port)
        bus.connect(handshake=False)
        bus.set_baudrate(1_000_000)
        ansi = not args.once and enable_terminal()
        if ansi:
            print("\033[?1049h\033[?25l", end="", flush=True)
        while True:
            lines = [f"六舵机实时读数  {port}  {time.strftime('%H:%M:%S')}  Ctrl+C退出",
                     "只读：不运动、不使能、不卸力。手动摆动前请确认已经卸力。",
                     "ID  关节       原始读数   零位   偏移计数   电压    扭矩   报警",
                     "-" * 76]
            for i, name in enumerate(NAMES):
                values = {reg: read_diagnostic(bus, name, reg) for reg in
                          ("Present_Position", "Present_Voltage", "Torque_Enable")}
                raw = values["Present_Position"]["value"]
                voltage = values["Present_Voltage"]["value"]
                torque = values["Torque_Enable"]["value"]
                zero = int(mapping.zero[i])
                position_text = "失败" if raw is None else str(raw)
                delta = "--" if raw is None else f"{raw-zero:+d}"
                volt = "--" if voltage is None else f"{voltage/10:.1f}V"
                torque_text = {0: "卸力", 1: "使能"}.get(torque, str(torque))
                alarms = sorted({v["servo_alarm"] for v in values.values()} - {0})
                alarm_text = ",".join(f"0x{a:02X}" for a in alarms) or "无"
                if any(v["value"] is None for v in values.values()):
                    alarm_text += " 通信失败"
                lines.append(f"{i+1:<3} {LABELS[i]}   {position_text:>6}   {zero:>4}   {delta:>7}   {volt:>5}   {torque_text}   {alarm_text}")
            lines.extend(["", "零位偏移是计数差，不是角度；ID6零位为夹爪闭合参考。"])
            block = "\n".join(lines)
            print(("\033[H\033[J" if ansi else "") + block, flush=True)
            if args.once:
                return 0
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        if ansi:
            print("\033[?25h\033[?1049l", end="", flush=True)
            ansi = False
        print(f"读取失败：{exc}")
        return 1
    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)
        if ansi:
            print("\033[?25h\033[?1049l", end="", flush=True)
        print("已退出读数程序，未改变舵机扭矩状态。")


if __name__ == "__main__":
    sys.exit(main())
