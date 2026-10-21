"""打开电脑摄像头，倒计时后拍照，保存到课堂练习/photos。

默认摄像头编号为 2（当前电脑的 HP 5MP Camera），倒计时 5 秒。
不弹出实时预览窗口。示例：python task1.py --camera 2 --delay 5
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


from exercise_paths import PHOTOS_DIR
# 当前 Windows DirectShow 设备顺序：
# 0 = IdeaCamera(App)，1 = GENERAL WEBCAM，2 = HP 5MP Camera（内置摄像头）。
# 设备增减后编号可能改变，可通过 --camera 指定。
DEFAULT_CAMERA_INDEX = 2


def read_frame(camera):
    success, frame = camera.read()
    if not success or frame is None or frame.size == 0:
        raise RuntimeError("摄像头已打开，但读取画面失败。请关闭其他占用摄像头的应用后重试。")
    return frame


def capture_photo(camera_index: int, delay: int) -> Path:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在打开摄像头 {camera_index}……", flush=True)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_index, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 {camera_index}。请检查 Windows 摄像头权限、"
                "物理遮挡开关，关闭相机/视频会议应用；设备增减后请重新确认摄像头编号。"
            )

        # 请求此分辨率；实际分辨率由摄像头支持情况决定。
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 连续读取，等待自动曝光稳定，并避免拍到摄像头刚启动的旧画面。
        warmup_end = time.monotonic() + 1.5
        while time.monotonic() < warmup_end:
            read_frame(camera)
            time.sleep(0.01)

        print("摄像头已就绪，请摆好动作，让肩膀、肘部和手部进入画面。", flush=True)
        deadline = time.monotonic() + delay
        last_remaining = None
        while time.monotonic() < deadline:
            remaining = max(1, math.ceil(deadline - time.monotonic()))
            if remaining != last_remaining:
                print(f"{remaining} 秒后拍照……", flush=True)
                last_remaining = remaining
            read_frame(camera)
            time.sleep(0.01)

        frame = read_frame(camera)
        # 丢弃可能缓存在采集接口中的少量帧，获取倒计时结束后的画面。
        for _ in range(3):
            frame = read_frame(camera)
    finally:
        camera.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 5 and gray.std() < 5:
        raise RuntimeError(
            "摄像头返回了近乎全黑的画面，未保存照片。请确认选中了内置摄像头、"
            "镜头遮挡开关已打开，且环境光线充足。"
        )

    # imencode + Python 文件写入支持中文路径；不使用 cv2.imwrite。
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("照片 JPEG 编码失败。")

    filename = f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    photo_path = PHOTOS_DIR / filename
    with photo_path.open("xb") as photo_file:
        photo_file.write(encoded.tobytes())

    height, width = frame.shape[:2]
    print(f"\n拍照成功！分辨率：{width} × {height}")
    print(f"照片已保存到：{photo_path}")
    return photo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="电脑摄像头倒计时拍照，保存到课堂练习/photos。")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help="摄像头编号，默认 2（当前电脑内置的 HP 5MP Camera）。")
    parser.add_argument("--delay", type=int, default=5, help="拍照倒计时秒数，默认 5。")
    args = parser.parse_args()
    if args.camera < 0 or args.delay < 0:
        parser.error("摄像头编号和倒计时秒数不能为负数。")

    try:
        capture_photo(args.camera, args.delay)
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"拍照失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消拍照，摄像头已释放。")
        sys.exit(130)
