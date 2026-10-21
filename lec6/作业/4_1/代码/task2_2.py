"""任务二（硅基流动）：识别本地照片中的手臂动作，在终端打印结果。

仅使用 Python 标准库，无需安装额外依赖。
"""

import base64
import json
import os
import sys
from pathlib import Path
from exercise_paths import PHOTOS_DIR
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# 在这里填写硅基流动的 API Key；填入后请勿公开分享此文件。
# 也可通过环境变量 SILICONFLOW_API_KEY 设置，环境变量优先。
API_KEY = ""  # 使用环境变量，提交文件不包含密钥。

# 任务书方案二中的模型名称。
MODEL = "Qwen/Qwen3.5-4B"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
IMAGE_PATH = (
    PHOTOS_DIR
    / "WIN_20261019_19_40_22_Pro.jpg"
)

PROMPT = """请用中文描述照片中人物的手臂姿态，用于机械臂动作模仿的视觉分析。
请分别说明：
1. 左臂和右臂的位置：上臂朝向、是否抬起、肘部是否弯曲、前臂朝向。
2. 能看清的手部状态：手掌是否张开、是否握拳、手掌朝向。
3. 用一句话概括主要动作。
左右以人物自身的左右为准；如果照片的镜像关系或遮挡导致无法确定，请明确说明，
必要时用“画面左侧/右侧”描述。只描述图像中能够确认的信息，不要猜测看不清的部分。
这是一张静态照片，不要凭空推断挥手等连续运动。不要生成机械臂关节参数或控制代码。
"""


def main() -> int:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip() or API_KEY.strip()
    if not api_key or api_key == "在这里填写你的硅基流动APIKey":
        print("尚未设置 API Key。请填写 API_KEY，或设置 SILICONFLOW_API_KEY 环境变量。")
        return 1

    try:
        image_base64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    except OSError as exc:
        print(f"读取图片失败：{IMAGE_PATH}\n{exc}")
        return 1

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }},
                {"type": "text", "text": PROMPT},
            ],
        }],
        # 硅基流动的关闭思考参数，与智谱的 thinking 参数不同。
        # 这里直接发送 HTTP JSON，所以不需要 extra_body 包装。
        "enable_thinking": False,
        "max_tokens": 1024,
        "stream": False,
    }
    request = Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"图片：{IMAGE_PATH}")
    print(f"平台：硅基流动\n模型：{MODEL}\n深度思考：关闭")
    print("正在发送图片并等待模型分析……", flush=True)
    try:
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
    except HTTPError as exc:
        hints = {
            400: "请检查请求参数、图片格式以及模型是否支持图片输入。",
            401: "请检查是否填写了正确的硅基流动 API Key。",
            403: "请检查账号认证状态和模型调用权限。",
            404: "请在硅基流动模型广场核对 MODEL 名称及模型是否仍可用。",
            429: "请求受到限流或额度限制，请查看控制台并稍后重试。",
        }
        print(f"API 调用失败（HTTP {exc.code}）：{hints.get(exc.code, '请查看平台状态或稍后重试。')}")
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            error = detail.get("error", detail)
            message = error.get("message") if isinstance(error, dict) else str(error)
            if message:
                print("平台提示：" + str(message).replace(api_key, "[已隐藏密钥]"))
        except (ValueError, UnicodeError, AttributeError):
            pass
        return 1
    except TimeoutError:
        print("请求超时，请检查网络或稍后重试。")
        return 1
    except URLError as exc:
        print(f"无法连接硅基流动 API，请检查网络或代理设置：{exc.reason}")
        return 1
    except (ValueError, UnicodeError):
        print("API 返回的内容不是有效的 JSON，暂时无法解析。")
        return 1

    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型未返回正文")
    except (KeyError, IndexError, TypeError, ValueError):
        print("API 未返回有效的动作描述，请检查模型设置或稍后重试。")
        return 1

    print("\n========== 模型动作分析结果 ==========\n")
    print(content.strip())
    if choice.get("finish_reason") == "length":
        print("\n提示：输出达到长度上限，可增大 max_tokens 后重试。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消。")
        sys.exit(130)
