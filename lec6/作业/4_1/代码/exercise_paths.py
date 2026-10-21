"""作业4.1的资源路径，与启动工作目录无关。"""
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent
PRACTICE_DIR = CODE_DIR.parent
WORKSPACE_ROOT = PRACTICE_DIR.parent
PHOTOS_DIR = PRACTICE_DIR / "photos"
GUIDE_PATH = CODE_DIR / "notes.txt"
RUNS_DIR = PRACTICE_DIR / "task3_runs"
ROBOT_DIR = CODE_DIR / "dependencies" / "IK_SCS215"
