"""Dual-model session, shared camera/robot, cancellation across API requests."""
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import uuid

import cv2
from bridge import HERE43, Agent, RunLog, RobotTools, SiliconFlowClient
from omni_client import OmniClient


class CancelClient:
    def __init__(self, client, cancelled):
        self.client, self.cancelled = client, cancelled
        self.model = client.model
        self.redact = client.redact

    def chat(self, *args, **kwargs):
        if self.cancelled.is_set():
            raise KeyboardInterrupt()
        result = self.client.chat(*args, **kwargs)
        if self.cancelled.is_set():
            raise KeyboardInterrupt()
        return result


class CancelRuntime:
    def __init__(self, runtime, cancelled):
        self.runtime, self.cancelled = runtime, cancelled

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def invoke(self, name, arguments):
        if self.cancelled.is_set():
            raise KeyboardInterrupt()
        return self.runtime.invoke(name, arguments)


class OfflinePlanner:
    """Explicit offline GUI demo only. Never selected as an API-error fallback."""
    model = "离线演示（非真实LLM）"
    redact = staticmethod(str)

    def chat(self, messages, tools=None, **kwargs):
        if messages[-1]["role"] == "tool":
            result = json.loads(messages[-1]["content"])
            return {"role": "assistant", "content": "离线演示工具结果：" + json.dumps(result, ensure_ascii=False)}
        envelope = json.loads(messages[-1]["content"])
        text = envelope["user_transcript"]
        if "停止" in text:
            name, args = "stop_tracking", {}
        elif "追踪" in text:
            name, args = "start_tracking", {"object_id": 0}
        elif "检测" in text:
            name, args = "detect_object", {"description": "ArUco 0"}
        elif "middle" in text or "中位" in text:
            name, args = "fk_move", {"pose_name": "middle"}
        elif "上" in text and ("移动" in text or "移" in text):
            name, args = "ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 1}
        elif "状态" in text:
            name, args = "get_robot_state", {}
        else:
            return {"role": "assistant", "content": "离线演示：合成场景有ArUco 0。可测试：查看状态、向上移动1厘米、回middle、检测ArUco 0、开始追踪、停止追踪。"}
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": uuid.uuid4().hex, "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}}]}


class Session:
    def __init__(self, dry_run=True, offline=False, emit=lambda *args: None, run_dir=None,
                 planner=None, omni=None, port=None):
        if offline and not dry_run:
            raise ValueError("离线演示只允许模拟硬件")
        self.offline, self.emit = offline, emit
        self.closed = threading.Event()
        self.cancelled = threading.Event()
        self.turn_lock = threading.Lock()
        self.folder = Path(run_dir) if run_dir else HERE43 / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        planner = planner or (OfflinePlanner() if offline else SiliconFlowClient(timeout=60, retries=1))
        self.omni = omni if omni is not None else (None if offline else OmniClient())
        self.redact = planner.redact
        self.log = RunLog(self.folder, self.redact)
        self.runtime = RobotTools(self.folder, planner, dry_run=dry_run, port=port, emit=self._runtime_event)
        self.agent = Agent(CancelClient(planner, self.cancelled), CancelRuntime(self.runtime, self.cancelled),
                           self.log, output=lambda text: self.emit("log", self.redact(text)))
        extra = (HERE43 / "planner_prompt.txt").read_text(encoding="utf-8")
        self.agent.messages[0]["content"] += "\n" + extra
        if offline:
            self.agent.messages[0]["content"] += "\n当前使用离线演示规则，不是真实LLM。"
        self.log.save("system_prompt.json", self.agent.messages[0])
        self.log.save("session.json", {"dry_run": dry_run, "offline": offline,
                                      "planner": planner.model, "omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct"})
        self.latest = None
        self.frame_lock = threading.Lock()
        self.preview_thread = threading.Thread(target=self._preview, name="preview-43", daemon=True)
        self.preview_thread.start()

    def _runtime_event(self, kind, value):
        self.log.event(kind, value)
        if kind == "tracking_status":
            self.emit("tracking", value)
        elif kind == "motion_progress":
            self.emit("progress", value.get("message", ""))

    def _preview(self):
        last_error = None
        while not self.closed.is_set():
            try:
                frame, stamp = self.runtime.vision.frame()
                with self.frame_lock:
                    self.latest = (frame, stamp)
                if last_error:
                    self.emit("camera", "摄像头已恢复")
                last_error = None
            except Exception as exc:
                error = self.redact(exc)
                if error != last_error and not self.closed.is_set():
                    self.emit("camera", error)
                last_error = error
                self.closed.wait(1)
            self.closed.wait(.1)

    def preview(self):
        with self.frame_lock:
            return self.latest

    def turn(self, text=None, audio_path=None, use_image=True):
        if self.closed.is_set():
            raise RuntimeError("会话已经关闭")
        if not self.turn_lock.acquire(blocking=False):
            raise RuntimeError("上一轮尚未结束")
        try:
            # The GUI clears cancellation synchronously BEFORE scheduling this turn.
            if self.cancelled.is_set():
                raise KeyboardInterrupt()
            frame = None
            if use_image:
                frame, stamp = self.runtime.vision.frame()
                path = self.folder / f"input_{time.time_ns()}.jpg"
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    encoded.tofile(str(path))
                self.log.event("input_image", {"path": str(path), "age_s": time.monotonic() - stamp})
            if audio_path:
                self.log.event("input_audio", {"path": str(audio_path)})
            if self.offline:
                if audio_path:
                    raise ValueError("离线演示不识别语音；请选择模拟模式（真实API）测试语音。")
                perception = {"transcript": text or "", "user_intent": text or "",
                              "scene_summary": "合成场景中的ArUco 0（模拟画面）" if use_image else "未提供图像",
                              "uncertainty": "离线演示，未调用Omni"}
            elif audio_path or use_image:
                self.emit("stage", "Omni正在理解语音和画面……")
                perception = self.omni.perceive(frame, text=text, audio_path=audio_path)
            else:
                perception = {"transcript": text or "", "user_intent": text or "",
                              "scene_summary": "本轮未附图像；环境问题请调用detect_object", "uncertainty": ""}
            if self.cancelled.is_set() or self.closed.is_set():
                raise KeyboardInterrupt()
            self.log.event("perception", perception)
            self.emit("perception", perception)
            transcript = (text if audio_path is None and text is not None else perception["transcript"]).strip()
            if not transcript:
                answer = "没有听清你的话，请重新录音。"
                self.emit("answer", answer)
                return answer
            self.emit("stage", "Qwen3.5正在选择工具或回答……")
            envelope = {"user_transcript": transcript, "perception_context": perception,
                        "image_is_simulated": self.runtime.dry_run if use_image else None}
            answer = self.agent.turn(json.dumps(envelope, ensure_ascii=False))
            if not self.cancelled.is_set():
                self.emit("answer", answer)
            return answer
        except Exception as exc:
            self.log.event("turn_error", {"error": self.redact(exc)})
            raise
        finally:
            self.turn_lock.release()

    def prepare_turn(self):
        if self.closed.is_set() or self.turn_lock.locked():
            raise RuntimeError("会话忙或已关闭")
        self.cancelled.clear()

    def stop(self):
        self.cancelled.set()
        self.runtime.interrupt()

    def hold_after_stop(self):
        if self.runtime.robot is not None and not self.closed.is_set():
            return self.runtime.stop_tracking()
        return {"ok": True, "status": "stopped", "connected": False}

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.stop()
        try:
            self.runtime.close()
        finally:
            self.preview_thread.join(timeout=8)
            self.log.event("closed", {"preview_stopped": not self.preview_thread.is_alive(),
                                      "robot_worker_stopped": not self.runtime.worker.is_alive()})
