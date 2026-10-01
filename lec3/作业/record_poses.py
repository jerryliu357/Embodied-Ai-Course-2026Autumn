"""按官方图依次手摆 Middle / Zero / Rotated / Rest，保存四组六关节比例。

默认进入手动采集；--list 或 --show middle 仅查看文件，不打开串口。
沿用 calibration.json 的端点，追加 poses；不重新校准范围、不发送运动目标。
运行环境：lerobot。依赖同目录 calibrate.py、set_id.py。
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import serial
from calibrate import (OUTPUT, POSITION_COUNT, MAX_SAMPLE_SPREAD, Cancelled,
                       capture_pose, disable_all_torque, prompt, ratio_to_raw)
from set_id import JOINTS, ServoBus, find_port, inspect

SOURCE_URL = "https://huggingface.co/docs/lerobot/main/en/assemble_so101#calibrate"
POSES = {
    "middle": "Middle：对照官网第 1 张图摆好中间姿态。",
    "zero": "Zero：对照官网第 2 张图摆好零位姿态。",
    "rotated": "Rotated：对照官网第 3 张图，位于机械臂右侧，夹爪张开。",
    "rest": "Rest：对照官网第 4 张图摆好休息姿态。",
}
ENDPOINT_TOLERANCE = MAX_SAMPLE_SPREAD  # 最多 3 个计数的端点抖动，会明确记录修正。


def validate_calibration(data):
    if (not isinstance(data, dict) or data.get("model") != "SCS215"
            or data.get("byte_order") != "big"
            or data.get("position_count") != POSITION_COUNT
            or data.get("range_capture_completed") is not True):
        raise ValueError("请先用 calibrate.py 完成当前 SCS215 的端点采集。")
    joints = data.get("joints")
    if not isinstance(joints, dict) or set(joints) != {str(sid) for sid in JOINTS}:
        raise ValueError("校准文件必须包含 ID 1～6 的端点。")
    for sid in JOINTS:
        item = joints[str(sid)]
        if not isinstance(item, dict):
            raise ValueError(f"ID={sid} 校准记录格式错误。")
        values = [item.get(key) for key in
                  ("zero_position", "one_position", "span_raw", "one_unwrapped_position")]
        if not all(type(value) is int for value in values):
            raise ValueError(f"ID={sid} 的端点、跨度必须是整数。")
        zero, one, span, end = values
        if (item.get("id") != sid or item.get("raw_direction") != 1
                or not 0 <= zero < POSITION_COUNT or not 0 <= one < POSITION_COUNT
                or not 0 < span < POSITION_COUNT or end != zero + span
                or end % POSITION_COUNT != one):
            raise ValueError(f"ID={sid} 的端点、正方向与跨度不一致。")


def calibration_fingerprint(data):
    """预设只对录制时使用的端点有效，避免范围变化后沿用旧比例。"""
    validate_calibration(data)
    basis = {
        "model": data["model"], "byte_order": data["byte_order"],
        "position_count": data["position_count"],
        "joints": [[sid, data["joints"][str(sid)]["zero_position"],
                    data["joints"][str(sid)]["one_position"],
                    data["joints"][str(sid)]["span_raw"]] for sid in JOINTS],
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()


def raw_to_ratio(raw, joint):
    """在已记录的正向区间内反算比例；不把区间外的姿态随意截断。"""
    if type(raw) is not int or not 0 <= raw < POSITION_COUNT:
        raise ValueError("原始位置必须是 0～1023 的整数。")
    zero, span = joint["zero_position"], joint["span_raw"]
    delta = (raw - zero) % POSITION_COUNT
    if delta <= span:
        return delta / span, None
    distance_before_zero = POSITION_COUNT - delta
    distance_after_one = delta - span
    if min(distance_before_zero, distance_after_one) <= ENDPOINT_TOLERANCE:
        value = 0.0 if distance_before_zero < distance_after_one else 1.0
        return value, {
            "measured_raw": raw, "target_raw": ratio_to_raw(value, joint),
            "adjustment_counts": min(distance_before_zero, distance_after_one),
            "reason": "endpoint_reading_tolerance",
        }
    raise ValueError(
        f"ID={joint['id']} 当前位置 {raw} 超出你记录的 0→1 区间 "
        f"({zero}→{joint['one_position']})。请调整该关节，已有端点不会改变。"
    )


def encode_pose(raw_positions, data):
    values, adjustments, errors = [], {}, []
    for sid in JOINTS:
        try:
            value, adjustment = raw_to_ratio(raw_positions[sid], data["joints"][str(sid)])
            values.append(value)
            if adjustment:
                adjustments[str(sid)] = adjustment
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("\n".join(errors))
    return values, adjustments


def get_pose_values(data, name):
    """后续控制程序可调用；只读数据，不连接舵机。"""
    validate_calibration(data)
    if name not in POSES:
        raise ValueError(f"未知姿态：{name}")
    metadata = data.get("pose_recording", {})
    if (not isinstance(metadata, dict)
            or metadata.get("calibration_fingerprint") != calibration_fingerprint(data)
            or metadata.get("joint_order") != list(JOINTS)):
        raise ValueError("没有当前范围对应的姿态记录，或端点已改变，请重新运行 record_poses.py。")
    poses = data.get("poses", {})
    values = poses.get(name) if isinstance(poses, dict) else None
    if (not isinstance(values, list) or len(values) != 6
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values)):
        raise ValueError(f"{name} 必须是六个 0～1 的有限数值。")
    return list(values)


def load_pose_targets(name, path=OUTPUT):
    """返回 {舵机 ID: 目标原始位置}，仅换算，不发送运动指令。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    values = get_pose_values(data, name)
    return {sid: ratio_to_raw(value, data["joints"][str(sid)])
            for sid, value in zip(JOINTS, values)}


def save_poses(path, original_bytes, data, poses, raw_poses, adjustments, captured_at):
    if set(poses) != set(POSES) or set(raw_poses) != set(POSES) or set(captured_at) != set(POSES):
        raise ValueError("需要完整采集四个姿态后才能保存。")
    updated = copy.deepcopy(data)
    updated["poses"] = poses
    updated["pose_recording"] = {
        "joint_order": list(JOINTS),
        "calibration_fingerprint": calibration_fingerprint(data),
        "source_url": SOURCE_URL, "recorded_at": datetime.now().isoformat(),
        "captured_at": captured_at, "raw_positions": raw_poses,
        "endpoint_adjustments": adjustments, "motion_tested": False,
    }
    for name in POSES:
        get_pose_values(updated, name)
    path = Path(path)
    if path.read_bytes() != original_bytes:
        raise RuntimeError("采集期间 calibration.json 被其他程序修改，未覆盖，请重新采集。")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".poses_", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(updated, file, ensure_ascii=False, indent=2)
            file.write("\n")
        if path.read_bytes() != original_bytes:
            raise RuntimeError("保存前校准文件发生变化，未覆盖。")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def confirm(message):
    while prompt(message):
        print("直接按回车继续，输入 q 取消。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="默认自动识别原 USB 转接板")
    viewing = parser.add_mutually_exclusive_group()
    viewing.add_argument("--list", action="store_true", help="只查看已保存的四组比例")
    viewing.add_argument("--show", choices=POSES, help="只查看某个姿态的比例和目标读数")
    args = parser.parse_args()
    original_bytes = OUTPUT.read_bytes()
    data = json.loads(original_bytes.decode("utf-8"))
    validate_calibration(data)
    if args.list or args.show:
        if not data.get("poses"):
            print("尚未保存姿态预设。直接运行本程序可手动采集四个姿态。")
            return
        for name in ([args.show] if args.show else POSES):
            if data["poses"].get(name) is None:
                print(f"{name}：尚未记录，请重新采集。")
                continue
            values = get_pose_values(data, name)
            print(f"{name}（ID1～ID6）：{values}")
            if args.show:
                print("对应原始目标：", load_pose_targets(name))
        print("只读查看完成，没有连接舵机。")
        return
    print("四姿态预设采集：Middle -> Zero -> Rotated -> Rest。")
    print(f"请打开官方 follower 四张姿态图：{SOURCE_URL}")
    print("每个姿态摆好整条机械臂，再回车记录六个关节；不做 90° 检查。")
    print("沿用现有 0/1 端点。Middle 不一定是六个 0.5，Zero 也不一定是六个 0。")
    print("本程序只采集，没有自动运动或复位。q / Ctrl+C 取消时文件保持不变。")
    if data.get("poses"):
        print("已有姿态预设；完成四次采集并确认保存后，会替换这四组预设。")
    bus = ServoBus(args.port or find_port())
    torque_disabled, saved = False, False
    try:
        for sid in JOINTS:
            state = inspect(bus, sid)
            print(f"ID={sid} {JOINTS[sid]} 在线，位置 {state['position']}")
        confirm("请固定底座并托住机械臂，回车关闭六只舵机扭矩，q 退出：")
        disable_all_torque(bus)
        torque_disabled = True
        poses, raw_poses, adjustments, captured_at = {}, {}, {}, {}
        for index, (name, instructions) in enumerate(POSES.items(), 1):
            while True:
                raw, _ = capture_pose(bus, f"{index}/4 {instructions}")
                try:
                    values, adjusted = encode_pose(raw, data)
                except ValueError as exc:
                    print(f"本姿态尚未记录：\n{exc}\n调整后再回车，或按 q 退出。")
                    continue
                break
            poses[name] = values
            raw_poses[name] = [raw[sid] for sid in JOINTS]
            adjustments[name] = adjusted
            captured_at[name] = datetime.now().isoformat()
            print(f"已记录 {name}（ID1～ID6）：" + ", ".join(f"{x:.6f}" for x in values))
            for sid, adjustment in adjusted.items():
                print(f"ID={sid} 比端点多偏出 {adjustment['adjustment_counts']} 个计数，"
                      f"按端点 {adjustment['target_raw']} 记录，原始读数仍保留。")
        print("\n四组姿态采集完成；joints 和原有端点采样记录保持不变。")
        confirm("回车将这四组预设保存到 calibration.json，q 取消：")
        save_poses(OUTPUT, original_bytes, data, poses, raw_poses, adjustments, captured_at)
        saved = True
        print(f"已保存四组姿态：{OUTPUT}")
    finally:
        bus.close()
        if torque_disabled:
            print("扭矩保持关闭，请托稳机械臂并手动放好。")
        if not saved:
            print("本次未保存姿态，calibration.json 保持不变。")


if __name__ == "__main__":
    try:
        main()
    except (Cancelled, KeyboardInterrupt, EOFError):
        print("\n姿态采集已取消。")
    except (OSError, RuntimeError, ValueError, serial.SerialException) as exc:
        print(f"操作停止：{exc}", file=sys.stderr)
        sys.exit(1)
