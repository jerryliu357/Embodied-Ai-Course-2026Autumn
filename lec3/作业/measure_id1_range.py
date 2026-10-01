"""手动测量 ID=1 的连续位置区间，并与现有硬件限位比较。

使用 lerobot 环境运行；依赖同目录 set_id.py。
只操作 ID=1：确认后写一次扭矩关闭，不写位置、EEPROM 或其他舵机。
慢慢手转舵盘，Ctrl+C 结束并显示结果。CSV/JSON 保存在 diagnostics。
测量的是原始位置单位，不凭型号或外观推算实际角度，也不宣称运动验证完成。
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

from set_id import TORQUE_REGISTER, ServoBus, find_port, inspect

SERVO_ID = 1
PORT = None  # 自动识别原转接板；需要时可填写 "COM9"。
POSITION_REGISTER = 0x38
MIN_LIMIT_REGISTER = 0x09
LOG_DIR = Path(__file__).with_name("diagnostics")
EDGE = 20              # 排除反馈量程最边缘，不能把端点停滞当作有效活动。
MARGIN = 20            # 在本次观测区间内，再各留 20 个原始位置单位。
MAX_STEP = 80          # 相邻采样变化过大时分段，既可能是跳变，也可能是手转过快。
MIN_SAMPLES = 10


class RangeTracker:
    """只保留各连续段统计；绝不将跳变前后两段取模拼接。"""

    def __init__(self):
        self.previous = None
        self.current = None
        self.segments = []
        self.jump_count = 0
        self.sample_count = 0

    def end_segment(self):
        if self.current is not None:
            self.segments.append(self.current)
            self.current = None

    def add(self, raw):
        if type(raw) is not int or not 0 <= raw <= 1023:
            raise ValueError(f"位置数值异常：{raw}")
        delta = 0 if self.previous is None else raw - self.previous
        self.previous = raw
        self.sample_count += 1
        notes = []
        if abs(delta) > MAX_STEP:
            self.jump_count += 1
            self.end_segment()
            notes.append("变化过大，已另起一段；暂停转动，之后慢一点")
        if raw < EDGE or raw > 1023 - EDGE:
            self.end_segment()
            notes.append("接近反馈端点；停止向这一侧转动，可轻轻反向返回")
        elif self.current is None:
            self.current = {"min": raw, "max": raw, "samples": 1}
        else:
            self.current["min"] = min(self.current["min"], raw)
            self.current["max"] = max(self.current["max"], raw)
            self.current["samples"] += 1
        return delta, "；".join(notes)

    def report(self, limits):
        segments = list(self.segments)
        if self.current is not None:
            segments.append(dict(self.current))
        usable = [s for s in segments if s["samples"] >= MIN_SAMPLES
                  and s["max"] - s["min"] > 2 * MARGIN]
        best = max(usable, key=lambda s: s["max"] - s["min"], default=None)
        observed = [best["min"], best["max"]] if best else None
        candidate = [observed[0] + MARGIN, observed[1] - MARGIN] if best else None
        limits_valid = 0 <= limits[0] < limits[1] <= 1023
        overlap = None
        if candidate and limits_valid:
            low, high = max(candidate[0], limits[0]), min(candidate[1], limits[1])
            if low < high:
                overlap = [low, high]
        return {
            "id": SERVO_ID, "units": "raw_position_0_to_1023",
            "sample_count": self.sample_count, "large_step_count": self.jump_count,
            "continuous_segments": segments,
            "observed_continuous_range_raw": observed,
            "margin_raw": MARGIN, "excluded_edge_raw": EDGE,
            "split_threshold_raw": MAX_STEP,
            "candidate_range_raw": candidate,
            "candidate_midpoint_raw": round(sum(candidate) / 2) if candidate else None,
            "hardware_limits_raw": limits, "hardware_limits_valid": limits_valid,
            "candidate_within_existing_limits_raw": overlap,
            "angle_range_measured": False, "full_travel_measured": False,
            "motion_tested": False,
        }


def print_report(result):
    print("\n=== ID=1 本次范围测量 ===")
    print(f"采样 {result['sample_count']} 次，因相邻变化过大分段 {result['large_step_count']} 次。")
    print(f"舵机现有硬件限位：{result['hardware_limits_raw']}（没有修改）")
    if result["candidate_range_raw"] is None:
        print("没有测到足够宽的连续区间，本次不能给出候选控制范围。")
        print("仅静止采样、活动太小或转动过快都可能导致这个结果。")
        return
    print(f"本次最宽的连续读数区间：{result['observed_continuous_range_raw']}")
    print(f"两端各留 {MARGIN} 单位后的候选范围：{result['candidate_range_raw']}")
    print(f"该候选范围的中点：{result['candidate_midpoint_raw']}（供以后对齐安装参考）")
    allowed = result["candidate_within_existing_limits_raw"]
    if allowed:
        print(f"同时满足现有硬件限位的候选目标范围：{allowed}")
        if allowed != result["candidate_range_raw"]:
            print("现有硬件限位限制了本次观测范围；需另行确认后才能修改限位。")
    elif not result["hardware_limits_valid"]:
        print("现有限位不是有效的位置区间（两端均为 0 时可能是电机模式）。")
    else:
        print("候选范围与现有硬件限位没有可用交集，暂不能按这个范围下位置指令。")
    print("结果仅说明手动采样的反馈与寄存器限位，没有验证电机驱动或机械安全行程。")
    print("数字是原始位置，不是角度；仅靠位置反馈不能测出真实的 180° 或最大行程。")


def measure(bus, limits, before):
    tracker = RangeTracker()
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = LOG_DIR / f"id1_range_{stamp}.csv"
    json_path = csv_path.with_suffix(".json")
    status, error = "interrupted", None
    print("\n慢慢手转已经脱离负载的舵盘，往两侧探索你希望使用的范围，再缓慢返回。")
    print("遇到卡住或明显阻力就停；接近 0/1023 或发生跳变时也先停，再反向返回。")
    print("不需要强行转满一圈。屏幕会实时显示；完成后按 Ctrl+C 输出结果。")
    print(f"记录：{csv_path}", flush=True)
    try:
        with csv_path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", "id", "raw", "delta", "segment_min", "segment_max", "note"])
            start = time.monotonic()
            last_display = -1.0
            while True:
                if bus.read(SERVO_ID, TORQUE_REGISTER) != b"\x00":
                    raise RuntimeError("ID=1 扭矩不再为 0，请停止手动活动。")
                raw = int.from_bytes(bus.read(SERVO_ID, POSITION_REGISTER, 2), "big")
                delta, note = tracker.add(raw)
                elapsed = time.monotonic() - start
                segment = tracker.current
                low, high = (segment["min"], segment["max"]) if segment else ("", "")
                writer.writerow([f"{elapsed:.3f}", SERVO_ID, raw, delta, low, high, note])
                stream.flush()
                if note or elapsed - last_display >= 0.3:
                    print(f"位置={raw:4d}  变化={delta:+5d}  当前连续段=[{low}, {high}]  {note}",
                          flush=True)
                    last_display = elapsed
                time.sleep(0.1)
    except KeyboardInterrupt:
        status = "stopped_by_user"
    except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
        status, error = "error", str(exc)
        print(f"\n采样中止：{error}")
    result = tracker.report(limits)
    result.update(created_at=datetime.now().isoformat(), before=before,
                  status=status, error=error, csv_file=csv_path.name,
                  use_for_motion=False)
    # 通信错误前的片段可供排查，但不能作为本次完整测量建议。
    if error:
        print("本次发生错误，以下区间仅供排查，需重新完成测量。")
    print_report(result)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存：{json_path}")
    return 1 if error else 0


def main():
    print("只检查 ID=1。程序不会主动转动舵机；确认后只关闭 ID=1 扭矩。")
    bus = ServoBus(PORT or find_port())
    torque_disabled = False
    try:
        before = inspect(bus, SERVO_ID)
        raw_limits = bus.read(SERVO_ID, MIN_LIMIT_REGISTER, 4)
        limits = [int.from_bytes(raw_limits[:2], "big"), int.from_bytes(raw_limits[2:], "big")]
        print(f"当前位置={before['position']}，扭矩={before['torque']}，硬件限位={limits}")
        answer = input("确认 ID=1 已脱离上部负载并固定好，回车关闭扭矩开始测量；q 退出：").strip()
        if answer:
            return 0
        bus.write_checked(SERVO_ID, TORQUE_REGISTER, 0)
        torque_disabled = True
        return measure(bus, limits, before)
    finally:
        bus.close()
        if torque_disabled:
            print("串口已关闭，ID=1 扭矩保持关闭。")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
    except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
        print(f"检查停止：{exc}", file=sys.stderr)
        sys.exit(1)
