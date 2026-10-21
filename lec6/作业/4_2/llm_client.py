"""硅基流动真实API：4.1同款模型与Key，不把密钥写入日志。"""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import runtime_paths  # 设置课程模块路径
from task3 import MODEL, API_URL, resolve_api_key


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复JSON字段：{key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"非有限JSON数值：{value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


class SiliconFlowClient:
    def __init__(self, api_key=None, timeout=90, retries=1):
        self.api_key = resolve_api_key() if api_key is None else api_key
        self.model = MODEL
        self.timeout = timeout
        self.retries = retries

    def redact(self, text):
        return str(text).replace(self.api_key, "[隐藏密钥]") if self.api_key else str(text)

    def chat(self, messages, tools=None, json_mode=False):
        payload = dict(model=self.model, messages=messages, enable_thinking=False,
                       stream=False, temperature=0.1, max_tokens=2500)
        if tools is not None:
            payload.update(tools=tools, tool_choice="auto")
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = Request(API_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        # 只重发当前HTTP请求；工具执行不在此函数中，不会因网络重试重复运动。
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                detail = self.redact(exc.read().decode("utf-8", errors="replace"))
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RuntimeError(f"硅基流动 HTTP {exc.code}：{detail[:800]}") from None
                print(f"[API] 服务返回{exc.code}，正在重试当前请求……", flush=True)
            except (URLError, TimeoutError) as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"硅基流动请求失败：{self.redact(exc)}") from None
                print("[API] 网络请求失败，正在重试当前请求……", flush=True)
            time.sleep(1)
        try:
            choice = result["choices"][0]
            if choice.get("finish_reason") not in ("stop", "tool_calls"):
                raise ValueError(f"模型输出未完整结束：{choice.get('finish_reason')}")
            original = choice["message"]
            message = {"role": "assistant", "content": original.get("content")}
            if original.get("tool_calls"):
                message["tool_calls"] = original["tool_calls"]
            return message
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("硅基流动响应格式不正确。") from None

