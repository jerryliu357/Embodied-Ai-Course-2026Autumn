"""姿态和动作文件。所有比例按 ID1～ID6 排列；不连接串口。"""
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import tempfile

from arm_backend import JOINTS, calibration_fingerprint, raw_to_ratio

HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = HERE / "calibration.json"
ACTIONS_PATH = HERE / "actions.json"
POSE_LABELS = {"stand": "立正", "lift": "抬起", "left": "向左", "right": "向右",
               "middle": "上次 Middle", "zero": "上次 Zero",
               "rotated": "上次 Rotated", "rest": "上次 Rest"}


def finite_number(value, low, high, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{label} 必须在 {low}～{high} 之间。")
    return float(value)


def pose_values(values):
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("每个姿态必须有 ID1～ID6 的六个比例。")
    return [finite_number(x, 0, 1, "关节比例") for x in values]


def default_actions(calibration):
    fingerprint = calibration_fingerprint(calibration)
    poses = {name: {"label": label, "values": None, "source": "待实机录制"}
             for name, label in POSE_LABELS.items() if name in ("stand", "lift", "left", "right")}
    metadata = calibration.get("pose_recording", {})
    if metadata.get("calibration_fingerprint") == fingerprint:
        for name, values in calibration.get("poses", {}).items():
            poses[name] = {"label": POSE_LABELS.get(name, name), "values": pose_values(values),
                           "source": "lec3 最终校准记录"}
    return {"schema_version": 1, "joint_order": list(JOINTS),
            "calibration_fingerprint": fingerprint,
            "settings": {"speed_limit": 100, "pose_seconds": 3.0},
            "poses": poses,
            "actions": {"wave": {"label": "挥手", "steps": [
                {"pose": "lift", "seconds": 2.0, "hold": 0.3},
                {"pose": "left", "seconds": 1.0, "hold": 0.2},
                {"pose": "right", "seconds": 1.0, "hold": 0.2},
                {"pose": "left", "seconds": 1.0, "hold": 0.2},
                {"pose": "lift", "seconds": 1.0, "hold": 0.0}]}}}


def validate_document(document, calibration):
    if (document.get("schema_version") != 1 or document.get("joint_order") != list(JOINTS)
            or document.get("calibration_fingerprint") != calibration_fingerprint(calibration)):
        raise ValueError("actions.json 与当前六关节校准不匹配，请核对文件。")
    settings = document["settings"]
    finite_number(settings["speed_limit"], 1, 1000, "速度上限")
    if type(settings["speed_limit"]) is not int:
        raise ValueError("速度上限必须是整数。")
    finite_number(settings["pose_seconds"], 0.1, 120, "姿态切换时间")
    for pose in document["poses"].values():
        if pose["values"] is not None:
            pose_values(pose["values"])
    for action in document["actions"].values():
        if not isinstance(action["steps"], list) or not 1 <= len(action["steps"]) <= 100:
            raise ValueError("每个动作需要 1～100 个步骤。")
        for step in action["steps"]:
            if step["pose"] not in document["poses"]:
                raise ValueError(f"动作引用了不存在的姿态：{step['pose']}")
            finite_number(step["seconds"], 0.1, 120, "每段时间")
            finite_number(step.get("hold", 0), 0, 60, "停留时间")


class ActionStore:
    def __init__(self, calibration, path=ACTIONS_PATH, memory_only=False):
        self.calibration, self.path, self.memory_only = calibration, Path(path), memory_only
        self.original = self.path.read_bytes() if self.path.exists() else None
        self.document = (json.loads(self.original.decode("utf-8-sig")) if self.original
                         else default_actions(calibration))
        validate_document(self.document, calibration)

    def persist(self, updated):
        validate_document(updated, self.calibration)
        if not self.memory_only:
            current = self.path.read_bytes() if self.path.exists() else None
            if current != self.original:
                raise RuntimeError("actions.json 被其他程序修改，请重新打开窗口后保存。")
            encoded = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                f.write(encoded)
            try:
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            self.original = encoded
        self.document = updated

    def record(self, name, raw_positions):
        if name not in self.document["poses"]:
            raise ValueError("请选择一个姿态名称。")
        values, adjustments = [], {}
        for sid in JOINTS:
            ratio, adjustment = raw_to_ratio(raw_positions[sid], self.calibration["joints"][str(sid)])
            values.append(ratio)
            if adjustment:
                adjustments[str(sid)] = adjustment
        updated = copy.deepcopy(self.document)
        updated["poses"][name].update(values=values, raw_positions=raw_positions,
            endpoint_adjustments=adjustments, recorded_at=datetime.now().isoformat(timespec="seconds"),
            source="离线模拟（仅内存）" if self.memory_only else "六关节稳定实测位置")
        self.persist(updated)

    def resolve(self, name, is_action=False, seconds=None):
        if is_action:
            steps = self.document["actions"][name]["steps"]
        else:
            steps = [{"pose": name, "seconds": seconds or self.document["settings"]["pose_seconds"], "hold": 0}]
        result = []
        missing = [step["pose"] for step in steps if self.document["poses"][step["pose"]]["values"] is None]
        if missing:
            labels = [POSE_LABELS.get(key, key) for key in dict.fromkeys(missing)]
            raise ValueError("请先记录这些姿态：" + "、".join(labels))
        for step in steps:
            finite_number(step["seconds"], 0.1, 120, "切换时间")
            result.append({**step, "values": pose_values(self.document["poses"][step["pose"]]["values"])})
        return copy.deepcopy(result)
