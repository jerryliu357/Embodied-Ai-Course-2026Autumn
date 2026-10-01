"""SCS215 两端点校准：统一记录六个 0 -> 按 ID 逐个记录 1 -> 保存。
Windows 终端运行，使用 lerobot 环境；依赖同目录 set_id.py。
仅在确认支撑后关闭六个舵机扭矩，不写目标位置、ID、EEPROM 或硬件限位。
持续读取移动过程，统一以原始读数增大为正方向；不做 90 度检查。
数值跨界用 1024 展开记录，但跨界不等于可靠角度反馈，跨界区间标为待验证。
"""
import argparse
import json
import math
import msvcrt
import statistics
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import serial
from set_id import ADAPTER_SERIAL, BAUDRATE, JOINTS, TORQUE_REGISTER, ServoBus, find_port, inspect

OUTPUT = Path(__file__).with_name("calibration.json")
JOINT_NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
POSITION_REGISTER = 0x38
SAMPLE_COUNT = 5
MAX_SAMPLE_SPREAD = 3
MIN_SPAN = 10
EDGE_MARGIN = 20
POSITION_COUNT = 1024  # 数值计数周期，不代表 SCS215 有完整 360° 有效位置反馈。
MAX_STEP = 120        # 相邻慢速采样的变化阈值；不是角度或机械限位。
MAX_SAMPLE_GAP = 0.6  # 秒；采样间隔过大时要求重采。
DIRECTION_CONVENTION = "所有关节均按原始读数增大为正方向；统一记录 0 端，逐个关节记录 1 端"


class Cancelled(Exception):
    pass


def prompt(message):
    answer = input(message).strip().lower()
    if answer in ("q", "quit", "exit"):
        raise Cancelled()
    return answer


def poll_key():
    """非阻塞读按键，等待回车时仍持续读取舵机位置。"""
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
        elif key in ("q", "Q", "\x03"):
            raise Cancelled()
        elif key in ("\r", "\n"):
            return "enter"
    return None


def read_positions(bus):
    positions = {}
    for sid in JOINTS:
        if bus.read(sid, TORQUE_REGISTER) != b"\x00":
            raise RuntimeError(f"ID={sid} 扭矩未关闭，请停止手动移动。")
        raw = int.from_bytes(bus.read(sid, POSITION_REGISTER, 2), "big")
        if not 0 <= raw <= 1023:
            raise RuntimeError(f"ID={sid} 位置异常：{raw}")
        positions[sid] = raw
    return positions


class MotionTrace:
    """记录正向/反向数值跨界；展开计数仅用于描述路线，不代表跨界可控。"""
    def __init__(self, zero, timestamp):
        self.previous = dict(zero)
        self.timestamp = timestamp
        self.minimum, self.maximum = dict(zero), dict(zero)
        self.unwrapped = dict(zero)
        self.forward_wraps = {sid: 0 for sid in zero}
        self.backward_wraps = {sid: 0 for sid in zero}
        self.near_edge = set()
        self.max_step = {sid: 0 for sid in zero}
        self.samples, self.max_gap = 0, 0.0
        self.issues = set()

    def observe(self, positions, timestamp):
        old_issues = set(self.issues)
        gap = timestamp - self.timestamp
        self.max_gap = max(self.max_gap, gap)
        if gap > MAX_SAMPLE_GAP:
            self.issues.add("采样间隔过大，不能确认移动过程，请重新采集。")
        for sid, raw in positions.items():
            difference = raw - self.previous[sid]
            if difference < -POSITION_COUNT / 2:
                difference += POSITION_COUNT
                self.forward_wraps[sid] += 1
            elif difference > POSITION_COUNT / 2:
                difference -= POSITION_COUNT
                self.backward_wraps[sid] += 1
            self.unwrapped[sid] += difference
            step = abs(difference)
            self.max_step[sid] = max(self.max_step[sid], step)
            self.minimum[sid] = min(self.minimum[sid], raw)
            self.maximum[sid] = max(self.maximum[sid], raw)
            if step > MAX_STEP:
                self.issues.add(f"ID={sid} 相邻采样变化过大，请放慢后重采，不能可靠判断这段路线。")
            if not EDGE_MARGIN <= raw <= 1023 - EDGE_MARGIN:
                self.near_edge.add(sid)
        self.previous, self.timestamp = dict(positions), timestamp
        self.samples += 1
        return sorted(self.issues - old_issues)

    def summary(self):
        return {"samples": self.samples, "max_sample_gap_seconds": self.max_gap,
                "min_raw": self.minimum, "max_raw": self.maximum,
                "max_step_raw": self.max_step, "issues": sorted(self.issues),
                "unwrapped_end_raw": self.unwrapped,
                "forward_wraps": self.forward_wraps, "backward_wraps": self.backward_wraps,
                "near_feedback_edge_ids": sorted(self.near_edge),
                "meaning": "展开计数只描述数值路线，不能补足传感器盲区或保证位置闭环跨界运动"}


def capture_pose(bus, title, traces=None, active_ids=None):
    """只确认 active_ids 的端点；持续跟踪尚未记录 1 端的关节。"""
    active_ids = tuple(JOINTS if active_ids is None else active_ids)
    print(f"\n{title}")
    if len(active_ids) == 1:
        print(f"本次只记录 ID={active_ids[0]}；其他关节可以调整，不必保持之前的姿态。", flush=True)
        print("当前关节摆稳后按回车；q 或 Ctrl+C 退出。", flush=True)
    else:
        print("全部摆稳后按回车统一记录；q 或 Ctrl+C 退出。下方是实时原始位置。", flush=True)
    samples = deque(maxlen=SAMPLE_COUNT)
    last_display = -1.0
    while True:
        positions = read_positions(bus)
        now = time.monotonic()
        samples.append((now, positions))
        for sid, trace in (traces or {}).items():
            old_wraps = trace.forward_wraps[sid] + trace.backward_wraps[sid]
            old_edges = set(trace.near_edge)
            new_issues = trace.observe({sid: positions[sid]}, now)
            if sid not in active_ids:
                continue
            for issue in new_issues:
                print(f"\n采样提醒：{issue} 输入 q 可退出。", flush=True)
            if old_wraps != trace.forward_wraps[sid] + trace.backward_wraps[sid]:
                print("检测到数值跨界：已记录，不按反向误判；此关节的跨界控制仍待验证。", flush=True)
            if trace.near_edge - old_edges:
                print(f"ID {sorted(trace.near_edge - old_edges)} 接近反馈边缘，请停止向边缘推进，q 可退出。", flush=True)
        if now - last_display >= 0.75:
            print("  ".join(f"ID{sid}={positions[sid]:4d}" for sid in active_ids), flush=True)
            last_display = now
        if poll_key() == "enter":
            if len(samples) < SAMPLE_COUNT:
                print("请保持片刻，再按回车。")
            elif any(samples[i][0] - samples[i-1][0] > MAX_SAMPLE_GAP for i in range(1, len(samples))):
                print("最近采样有间断，请摆稳后重新按回车。")
            else:
                unstable = [sid for sid in active_ids
                            if max(p[sid] for _, p in samples) - min(p[sid] for _, p in samples)
                            > MAX_SAMPLE_SPREAD]
                pose = {sid: int(statistics.median(p[sid] for _, p in samples)) for sid in active_ids}
                edge = [sid for sid in active_ids if any(
                    not EDGE_MARGIN <= p[sid] <= 1023 - EDGE_MARGIN for _, p in samples)]
                if unstable:
                    print(f"ID {unstable} 还在晃动，请托稳后再按回车。")
                elif edge:
                    print(f"ID {edge} 太接近 0/1023，请向有效范围内留出余量。")
                else:
                    print("已记录：" + "，".join(f"ID={sid}: {raw}" for sid, raw in pose.items()))
                    return pose, now
        time.sleep(0.05)


def build_joints(zero, one, traces):
    errors, joints = [], {}
    for sid in zero:
        trace = traces[sid]
        a, b = zero[sid], one[sid]
        if any(type(v) is not int or not EDGE_MARGIN <= v <= 1023 - EDGE_MARGIN for v in (a, b)):
            errors.append(f"ID={sid} 的端点不在留有余量的反馈范围内。")
            continue
        # 用全过程确定走了哪条数值路线；只看两端不能区分反向和正向跨界。
        unwrapped_end = trace.unwrapped[sid] + b - trace.previous[sid]
        span = unwrapped_end - a
        if span < 0:
            errors.append(f"ID={sid} 本次终点位于读数减小方向（{a}→{b}），请按读数增大方向重采。")
            continue
        if span < MIN_SPAN:
            errors.append(f"ID={sid} 两端太接近（{a}→{b}），需要分别设置 0 和 1。")
            continue
        if span >= POSITION_COUNT:
            errors.append(f"ID={sid} 记录超过一个数值周期；本程序只记录一段小于一周期的范围。")
            continue
        low, high = sorted((a, b))
        wraps = unwrapped_end >= POSITION_COUNT
        boundary_unverified = (wraps or sid in trace.near_edge
                               or trace.forward_wraps[sid] > 0 or trace.backward_wraps[sid] > 0)
        joints[str(sid)] = {
            "id": sid, "name": JOINT_NAMES[sid], "label": JOINTS[sid],
            "zero_position": a, "one_position": b,
            "one_unwrapped_position": unwrapped_end, "span_raw": span,
            "min_position": None if wraps else low, "max_position": None if wraps else high,
            "raw_direction": 1, "wraps_numeric_boundary": wraps,
            "requires_boundary_resolution": boundary_unverified,
        }
    if errors:
        raise ValueError("\n".join(errors))
    return joints


def capture_one_positions(bus, zero, timestamp):
    traces = {sid: MotionTrace({sid: zero[sid]}, timestamp) for sid in JOINTS}
    pending = dict(traces)
    one, joints, recorded_at = {}, {}, {}
    for index, sid in enumerate(JOINTS, 1):
        pose, captured_at = capture_pose(
            bus,
            f"第二步 {index}/6：只把 ID={sid}（{JOINTS[sid]}）摆到希望定义为 1 的终点。"
            f"已记录的 0 点为 {zero[sid]}，方向按原始读数增大。",
            traces=pending, active_ids=(sid,),
        )
        # 冻结这个关节的终点及轨迹；之后移动它不会改变校准记录。
        pending.pop(sid)
        one[sid] = pose[sid]
        recorded_at[str(sid)] = captured_at - timestamp
        issues = sorted(traces[sid].issues)
        try:
            item = build_joints({sid: zero[sid]}, {sid: one[sid]}, traces)
        except ValueError as exc:
            issues.extend(str(exc).splitlines())
        if issues:
            raise ValueError(f"ID={sid} 的采集需要检查：\n" + "\n".join(issues))
        joints.update(item)
        print(f"ID={sid} 的 1 点已保留为 {one[sid]}，接下来不必保持此位置。", flush=True)
    return one, joints, traces, recorded_at


def ratio_to_raw(value, joint):
    """纯数值插值预览；跨界记录不代表该目标或经过的路径可用于实际位置控制。"""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("输入必须是 0～1 的有限数字")
    return round(joint["zero_position"] + value * joint["span_raw"]) % POSITION_COUNT


def disable_all_torque(bus):
    failures = []
    for sid in JOINTS:
        try:
            bus.write_checked(sid, TORQUE_REGISTER, 0)
        except (RuntimeError, OSError, serial.SerialException) as exc:
            failures.append(f"ID={sid}: {exc}")
    if failures:
        raise RuntimeError("无法确认全部扭矩关闭，请保持支撑并停止手动移动。\n" + "\n".join(failures))
    print("六只舵机扭矩均已关闭，现在可以轻轻手动摆放。", flush=True)


def save_calibration(data):
    if OUTPUT.exists():
        if prompt("calibration.json 已存在，输入 overwrite 用本次记录替换；其他输入取消：") != "overwrite":
            raise Cancelled()
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="默认自动识别原 USB 转接板")
    args = parser.parse_args()
    print("六关节两端点校准：统一记录六个 0 -> 按 ID 1～6 逐个记录 1。")
    print(f"方向约定：{DIRECTION_CONVENTION}。")
    print("这里的正方向指数字增大，不要求从外面看一定是顺时针。")
    print("选择无碰撞且留有余量的端点；不需要六个关节同时到达 1。")
    port = args.port or find_port()
    bus = ServoBus(port)
    saved, torque_disabled = False, False
    try:
        for sid in JOINTS:
            info = inspect(bus, sid)
            print(f"ID={sid} {JOINTS[sid]}：位置 {info['position']}")
        answer = prompt("请固定底座并托住机械臂；回车关闭六只舵机扭矩，q 退出：")
        while answer:
            answer = prompt("准备好后直接按回车，q 退出：")
        disable_all_torque(bus)
        torque_disabled = True
        while True:
            zero, timestamp = capture_pose(bus, "第一步：把六个关节分别摆到你希望定义为 0 的起点。")
            try:
                one, joints, traces, recorded_at = capture_one_positions(bus, zero, timestamp)
            except ValueError as exc:
                print(f"\n{exc}")
                print("这不是 90° 姿态检查；可选择更小的连续范围或放慢移动后重试。")
                if prompt("输入 r 从六个 0 端重新采集，其他输入退出：") != "r":
                    raise Cancelled()
            else:
                break
        print("\n=== 两端点记录与数值插值预览（非运动指令） ===")
        for sid, item in joints.items():
            print(f"ID={sid} {item['label']}：0 -> {item['zero_position']}，"
                  f"0.5 -> {ratio_to_raw(0.5, item)}，1 -> {item['one_position']}")
        boundary_ids = [sid for sid, item in joints.items() if item["requires_boundary_resolution"]]
        if boundary_ids:
            print(f"ID {boundary_ids} 的路线涉及跨界或反馈边缘：仅保存记录，不能直接当成已验证的控制范围。")
        print("仅保存本次端点及采样记录，没有驱动验证或自动复位，也不会改写硬件限位。")
        answer = prompt("确认各关节的 0/1 方向符合你的选择后，回车保存，q 取消：")
        while answer:
            answer = prompt("直接按回车保存，q 取消：")
        save_calibration({
            "schema_version": 5, "method": "manual_common_zero_individual_one_positive_raw",
            "model": "SCS215", "byte_order": "big", "created_at": datetime.now().isoformat(),
            "port": port, "baudrate": BAUDRATE, "adapter_serial": ADAPTER_SERIAL,
            "mapping": "round(zero_position + x * span_raw) % 1024",
            "position_count": POSITION_COUNT,
            "wrap_interpretation": "numeric_record_only_not_full_circle_angle_feedback",
            "direction_convention": DIRECTION_CONVENTION,
            "requires_boundary_resolution": bool(boundary_ids),
            "calibration_status": "recorded_requires_boundary_resolution" if boundary_ids else "endpoints_recorded",
            "range_capture_completed": True, "motion_tested": False,
            "collision_free_for_all_combinations": False,
            "zero_pose_raw": zero, "one_positions_raw": one,
            "one_positions_simultaneous": False, "capture_order": list(JOINTS),
            "one_capture_elapsed_seconds": recorded_at,
            "trajectory_check": {str(sid): trace.summary() for sid, trace in traces.items()},
            "joints": joints,
        })
        saved = True
        print(f"已保存：{OUTPUT}")
    finally:
        bus.close()
        if torque_disabled:
            print("扭矩保持关闭，请托稳机械臂并手动放好。")
        if not saved:
            print("本次未保存新校准，已有 calibration.json 保持不变。")


if __name__ == "__main__":
    try:
        main()
    except (Cancelled, KeyboardInterrupt, EOFError):
        print("\n校准已取消。")
    except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
        print(f"校准停止：{exc}", file=sys.stderr)
        sys.exit(1)
