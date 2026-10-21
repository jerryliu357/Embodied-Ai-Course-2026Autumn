"""作业4.2的路径；使用随作业附带的课程代码与标定。"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXERCISE41 = ROOT / "4_1" / "代码"
LESSON4 = HERE / "dependencies"
LESSON5 = HERE / "dependencies"
for folder in (LESSON5, EXERCISE41 / "dependencies" / "IK_SCS215", EXERCISE41):
    sys.path.insert(0, str(folder))
INTRINSICS = LESSON5 / "intrinsics.json"
HAND_EYE = LESSON5 / "hand_eye_candidate.json"
POSES = LESSON4 / "FK" / "actions.json"
