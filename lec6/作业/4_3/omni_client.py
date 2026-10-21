"""Omni produces perception JSON only; Qwen3.5 owns native tool calls."""
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
from bridge import HERE43, resolve_api_key, strict_json

OMNI_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def parse_perception(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = strict_json(text)
    fields = ("transcript", "user_intent", "scene_summary", "uncertainty")
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) for k in fields):
        raise ValueError("Omni未返回有效的感知JSON，本轮没有执行动作。")
    return {key: value[key] for key in fields}


class OmniClient:
    def __init__(self, key=None, timeout=60):
        self.key = resolve_api_key() if key is None else key
        self.timeout = timeout

    def redact(self, value):
        return str(value).replace(self.key, "[隐藏密钥]") if self.key else str(value)

    def perceive(self, frame, text=None, audio_path=None):
        content = []
        if audio_path is not None:
            content.append({"type": "audio_url", "audio_url": {
                "url": "data:audio/wav;base64," + base64.b64encode(Path(audio_path).read_bytes()).decode()}})
        if frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("图像编码失败")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}})
        content.append({"type": "text", "text":
                        "请按系统要求返回感知JSON。" if audio_path is not None else
                        "用户原始文字：" + (text or "")})
        payload = {"model": OMNI_MODEL, "messages": [
            {"role": "system", "content": (HERE43 / "omni_prompt.txt").read_text(encoding="utf-8")},
            {"role": "user", "content": content}], "temperature": 0, "max_tokens": 1500, "stream": False}
        # This endpoint rejects enable_thinking; no tools are sent to Omni.
        req = Request("https://api.siliconflow.cn/v1/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            if exc.code == 402:
                raise RuntimeError("硅基流动账户余额不足（HTTP 402），请充值后重新发送。本轮未执行动作。") from None
            raise RuntimeError(f"Omni HTTP {exc.code}: " + self.redact(exc.read().decode(errors="replace"))[:500]) from None
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Omni响应未完整结束，本轮没有执行动作。")
        message = choice["message"]
        # SiliconFlow sometimes puts all Instruct output in reasoning_content.
        # Accept only the declared perception JSON; never parse/execute tool-call text.
        perception = parse_perception(message.get("content") or message.get("reasoning_content") or "")
        if text is not None and audio_path is None:
            perception["transcript"] = text
        return perception

