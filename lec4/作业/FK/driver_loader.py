"""Load the exact motor runtime bundled with this submission; no serial I/O."""
from pathlib import Path
import hashlib
import importlib
import json
import sys

VENDOR = Path(__file__).resolve().parent / "vendor" / "lerobot_runtime"


def load_driver():
    source = json.loads((VENDOR / "SOURCE.json").read_text(encoding="utf-8"))
    for relative, expected in source["files"].items():
        if hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"提交包驱动校验失败：{relative}")
    package = VENDOR / "src" / "lerobot"
    loaded = sys.modules.get("lerobot")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(package):
        raise RuntimeError("本进程已载入其他 LeRobot，请直接运行本目录的 control_fk.py。")
    source_dir = str(VENDOR / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    return importlib.import_module("lerobot.motors.feetech")
