"""Direct ID3 test: one fixed relative target, no IK or trajectory interpolation.

Default: current - 30 counts, speed register 100, observe up to 6 seconds.
Only ID3 is written. End/interrupt holds its measured position, never auto-returns
or unloads other joints. Uses the TA SCS215 driver in the lerobot environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

from robot_adapter import HERE, find_port, load_config, load_course_driver

NAME = "elbow_flex"
REGISTERS = ("Goal_Position", "Present_Position", "Torque_Enable", "Present_Voltage",
             "Present_Temperature", "Status", "Present_Load", "P_Coefficient", "D_Coefficient",
             "Min_Position_Limit", "Max_Position_Limit", "Max_Torque_Limit",
             "Minimum_Startup_Force", "CW_Dead_Zone", "CCW_Dead_Zone",
             "Protective_Torque", "Protection_Time", "Overload_Torque",
             "Running_Time", "Goal_Velocity")
REPORT = HERE / "last_id3_test.json"


def read(bus, register):
    return int(bus.read(register, NAME, normalize=False, num_retry=2))


def write_checked(bus, register, value):
    bus.sync_write(register, {NAME: int(value)}, normalize=False, num_retry=1)
    stored = read(bus, register)
    if stored != value:
        raise RuntimeError(f"ID3 {register} 写入 {value}，读回 {stored}，不一致。")


def snapshot(bus):
    return {register: read(bus, register) for register in REGISTERS}


def trial(bus, low, high, delta=-30, seconds=6., speed=100, emit=print,
          clock=time.monotonic, sleep=time.sleep):
    if type(delta) is not int or delta == 0 or abs(delta) > 40:
        raise ValueError("单次位移请选择 -40～40 的非零整数读数。")
    if not math.isfinite(seconds) or not 1 <= seconds <= 10 or not 1 <= speed <= 150:
        raise ValueError("观察时间1～10秒；速度寄存器1～150。")
    before = snapshot(bus)
    start = before["Present_Position"]
    goal = start + delta
    hard_low, hard_high = before["Min_Position_Limit"], before["Max_Position_Limit"]
    if not 0 <= hard_low < hard_high <= 1023:
        raise RuntimeError(f"ID3硬件位置模式范围无效：{hard_low}～{hard_high}。")
    low, high = max(low, hard_low), min(high, hard_high)
    if not low <= start <= high or not low <= goal <= high:
        raise ValueError(f"起点 {start} / 目标 {goal} 超出允许范围 {low}～{high}；未发运动。")
    if before["Torque_Enable"] not in (0, 1) or before["Status"]:
        raise RuntimeError("ID3不在正常位置控制状态；未发运动。")
    result = {"recorded_at": datetime.now().astimezone().isoformat(), "id": 3,
              "kind": "single_fixed_position_no_ik", "requested_delta_counts": delta,
              "start_raw": start, "target_raw": goal, "before": before,
              "samples": [], "success": False, "tolerance_counts": 5,
              "other_joints_written": False}
    touched = False
    changed_parameters = []
    try:
        emit(f"ID3 起点 {start} → 固定目标 {goal}（{delta:+d}计数）。只发送一次测试目标，观察最长 {seconds:g} 秒。")
        # Replace stale goals before enabling. Start holding at the measured
        # position rather than unexpectedly moving toward an old RAM target.
        touched = True
        write_checked(bus, "Goal_Position", start)
        for register, value in (("Running_Time", 0), ("Goal_Velocity", speed)):
            if before[register] != value:
                changed_parameters.append(register)
                write_checked(bus, register, value)
        write_checked(bus, "Torque_Enable", 1)
        write_checked(bus, "Goal_Position", goal)
        began = clock()
        stable = 0
        while True:
            sample = {"seconds": round(clock()-began, 3),
                      "position": read(bus, "Present_Position"),
                      "goal_register": read(bus, "Goal_Position"),
                      "torque": read(bus, "Torque_Enable"),
                      "load_feedback": read(bus, "Present_Load"),
                      "voltage_v": read(bus, "Present_Voltage")/10,
                      "temperature_c": read(bus, "Present_Temperature"),
                      "status": read(bus, "Status")}
            sample["error_counts"] = goal - sample["position"]
            sample["moved_counts"] = sample["position"] - start
            result["samples"].append(sample)
            emit(f"{sample['seconds']:4.1f}s  目标读回={sample['goal_register']}  实际={sample['position']}  "
                 f"已移动={sample['moved_counts']:+d}  剩余误差={sample['error_counts']:+d}  "
                 f"驱动反馈={sample['load_feedback']}  电压={sample['voltage_v']:.1f}V")
            if sample["goal_register"] != goal or sample["torque"] != 1 or sample["status"]:
                raise RuntimeError("目标、扭矩或硬件状态异常，停止本次测试。")
            if not low <= sample["position"] <= high or abs(sample["moved_counts"]) > abs(delta)+15:
                raise RuntimeError("反馈超出本次小位移的预期范围，停止测试。")
            stable = stable+1 if abs(sample["error_counts"]) <= 5 else 0
            if stable >= 3:
                result["success"] = True
                break
            if clock()-began >= seconds:
                break
            sleep(.15)
        result["before_hold"] = snapshot(bus)
        result["final_raw"] = result["before_hold"]["Present_Position"]
        result["moved_counts"] = result["final_raw"]-start
        result["remaining_error_counts"] = goal-result["final_raw"]
        result["success"] = result["success"] and abs(result["remaining_error_counts"]) <= 5
        result["reason"] = "reached" if result["success"] else "not_reached"
    except (Exception, KeyboardInterrupt) as exc:
        result["reason"] = "interrupted_or_error"
        result["message"] = str(exc) or type(exc).__name__
        raise
    finally:
        if touched:
            try:
                actual = read(bus, "Present_Position")
                write_checked(bus, "Goal_Position", actual)
                result["hold_goal"] = actual
                emit(f"测试结束：ID3目标已改为实际位置 {actual}，保持当前扭矩；不自动返回或卸力。")
            except Exception as exc:
                result["hold_error"] = str(exc)
                emit(f"ID3停止保持未确认：{exc}。请托稳机械臂并处理供电。")
            for register in reversed(changed_parameters):
                try:
                    write_checked(bus, register, before[register])
                except Exception as exc:
                    result.setdefault("restore_errors", {})[register] = str(exc)
        # Overwrite one report, including failures; no accumulating CSV files.
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=int, default=-30, help="相对当前位置的原始计数差，默认-30")
    parser.add_argument("--seconds", type=float, default=6.)
    parser.add_argument("--speed", type=int, default=100, help="ID3速度寄存器参数，不是末端cm/s")
    parser.add_argument("--port")
    parser.add_argument("--read-only", action="store_true", help="只读ID3，不改变任何寄存器")
    args = parser.parse_args()
    cfg = load_config()
    if args.port:
        cfg["port"] = args.port
    data = json.loads(Path(cfg["calibration_path"]).read_text(encoding="utf-8-sig"))
    joint = next(j for j in data["joints"] if j["id"] == 3)
    motor_module, bus_class, _ = load_course_driver()
    bus = bus_class(find_port(cfg), {NAME: motor_module.Motor(3, "scs215", motor_module.MotorNormMode.RANGE_0_100)},
                    protocol_version=1)
    try:
        bus.connect()
        print(f"助教SCS215驱动，{bus.port}；仅操作ID3，其他关节状态保留。")
        if args.read_only:
            print(json.dumps(snapshot(bus), ensure_ascii=False, indent=2))
            return 0
        print(f"将移动 {args.delta:+d} 计数（按现有标定约 {args.delta*data['degrees_per_count']:+.1f}°）。Ctrl+C可停止。")
        result = trial(bus, joint["range_0"], joint["range_1"], args.delta, args.seconds, args.speed)
        print("ID3单独测试：到达目标范围。" if result["success"] else "ID3单独测试：未达到目标。")
        print(f"实际移动 {result['moved_counts']:+d} 计数，剩余误差 {result['remaining_error_counts']:+d} 计数。")
        print(f"记录：{REPORT}")
        return 0 if result["success"] else 2
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已中止。")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"测试停止：{exc}")
        raise SystemExit(1)
