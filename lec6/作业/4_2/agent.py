"""4.2：终端输入自然语言 → SiliconFlow tool_calls → 六个本地机器人工具。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import threading

from runtime_paths import HERE, INTRINSICS, HAND_EYE
from llm_client import SiliconFlowClient, strict_json, MODEL
from robot_tools import RobotTools
from tool_specs import TOOLS


class RunLog:
    def __init__(self, folder, redact=str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.redact = redact
        self.lock = threading.Lock()

    def event(self, kind, value):
        record = {"time": datetime.now().isoformat(), "event": kind, "data": value}
        text = self.redact(json.dumps(record, ensure_ascii=False))
        with self.lock, (self.folder / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(text + "\n")

    def save(self, name, value):
        text = self.redact(json.dumps(value, ensure_ascii=False, indent=2))
        with self.lock:
            (self.folder / name).write_text(text, encoding="utf-8")


class Agent:
    def __init__(self, client, runtime, log, output=print):
        self.client, self.runtime, self.log, self.output = client, runtime, log, output
        prompt = (HERE / "prompt.txt").read_text(encoding="utf-8")
        mode = "当前是dry-run：工具结果都是模拟状态，不代表实物运动。" if runtime.dry_run else "当前是真机模式：工具会执行用户要求的实物动作。"
        self.messages = [{"role": "system", "content": prompt + "\n" + mode}]
        self.executed_ids = {}
        self.turn_calls = []
        (log.folder / "prompt.txt").write_text(prompt + "\n" + mode, encoding="utf-8")
        log.save("tools.json", TOOLS)

    def _tool_result(self, call):
        call_id = call["id"]
        name = call["function"]["name"]
        raw_args = call["function"]["arguments"]
        fingerprint = json.dumps([name, raw_args], ensure_ascii=False)
        if call_id in self.executed_ids:
            original, result = self.executed_ids[call_id]
            if original != fingerprint:
                return {"ok": False, "error": "模型重复使用工具调用ID且参数不同，未执行。"}
            return result
        try:
            arguments = strict_json(raw_args)
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": f"工具参数不是有效JSON：{exc}"}
        self.output(f"[工具] {name}({json.dumps(arguments, ensure_ascii=False)})")
        self.log.event("tool_call", {"tool_call_id": call_id, "name": name, "arguments": arguments})
        result = self.runtime.invoke(name, arguments)
        self.executed_ids[call_id] = (fingerprint, result)
        self.turn_calls.append({"name": name, "arguments": arguments, "result": result})
        self.log.event("tool_result", {"tool_call_id": call_id, "result": result})
        self.output("[结果] " + self.client.redact(json.dumps(result, ensure_ascii=False)))
        return result

    def turn(self, user_text, max_rounds=16):
        self.turn_calls = []
        self.executed_ids = {}  # 调用ID去重只限本轮；新一轮查询必须读取新状态。
        self.messages.append({"role": "user", "content": user_text})
        self.log.event("user", user_text)
        try:
            for _ in range(max_rounds):
                self.output("[模型] 正在处理……")
                message = self.client.chat(self.messages, tools=TOOLS)
                calls = message.get("tool_calls", [])
                ids = set()
                for call in calls:
                    if (not isinstance(call, dict) or call.get("type") != "function"
                            or not isinstance(call.get("id"), str) or not call["id"]
                            or call["id"] in ids or not isinstance(call.get("function"), dict)
                            or not isinstance(call["function"].get("name"), str)
                            or not isinstance(call["function"].get("arguments"), str)):
                        raise ValueError("模型tool_calls格式不正确或调用ID重复，未执行这一批。")
                    ids.add(call["id"])
                self.messages.append(message)
                self.log.event("assistant", message)
                if not calls:
                    answer = message.get("content")
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("模型未返回正文或原生tool_calls。")
                    self.output("Agent：" + self.client.redact(answer))
                    return answer
                interrupted = False
                prior_failure = False
                for call in calls:
                    name = call["function"]["name"]
                    if interrupted:
                        result = {"ok": False, "status": "cancelled", "error": "用户已停止，未执行后续工具。"}
                    elif prior_failure and name in ("ik_move", "fk_move", "start_tracking"):
                        result = {"ok": False, "status": "not_executed", "error": "本批先前工具未成功，请依据返回结果重新决定后续动作。"}
                    else:
                        try:
                            result = self._tool_result(call)
                        except KeyboardInterrupt:
                            self.runtime.interrupt()
                            interrupted = True
                            result = {"ok": False, "status": "cancelled", "error": "用户Ctrl+C停止。"}
                    prior_failure = prior_failure or result.get("ok") is False
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "content": json.dumps(result, ensure_ascii=False)})
                if interrupted:
                    self.output("已停止本轮执行，可以继续输入指令。")
                    return "用户停止"
            raise RuntimeError("本轮工具调用轮数过多，已停止继续请求模型；请查看日志后重新下达指令。")
        except (Exception, KeyboardInterrupt) as exc:
            # 结束失败轮次，避免下一条指令意外续做先前尚未执行的动作。
            if isinstance(exc, KeyboardInterrupt):
                self.runtime.interrupt()
            last_assistant = next((i for i in range(len(self.messages) - 1, -1, -1)
                                   if self.messages[i]["role"] == "assistant"), None)
            if last_assistant is not None:
                answered = {m.get("tool_call_id") for m in self.messages[last_assistant + 1:] if m["role"] == "tool"}
                for call in self.messages[last_assistant].get("tool_calls", []):
                    if call["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                              "content": json.dumps({"ok": False, "status": "cancelled", "error": "本轮已中断，结果未确认。"}, ensure_ascii=False)})
            detail = self.client.redact(exc) or "用户中断"
            self.messages.append({"role": "assistant", "content": f"本轮未完成：{detail}。已执行部分以工具结果为准；未执行动作取消，不自动续做。"})
            self.log.event("turn_error", detail)
            raise
        finally:
            self.log.save("conversation.json", self.messages)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="真实调用LLM，但机器人和相机使用模拟，不连接硬件")
    parser.add_argument("--check", action="store_true", help="纯离线检查路径、模型、标定、工具定义，不联网不连接硬件")
    parser.add_argument("--test-api", action="store_true", help="用模拟状态验证真实LLM原生工具调用，自动使用dry-run")
    parser.add_argument("--commands", type=Path, help="逐行运行保存的自然语言测试指令，#开头为注释")
    parser.add_argument("--port", help="串口，默认按原转接板序列号自动查找")
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS)
    parser.add_argument("--hand-eye", type=Path, default=HAND_EYE)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.commands is not None and not args.commands.is_file():
        raise SystemExit(f"指令文件不存在：{args.commands}")
    run_dir = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runtime = None
    try:
        client = None if args.check else SiliconFlowClient()
        log = RunLog(run_dir, client.redact if client else str)

        def emit(kind, value):
            log.event(kind, value)
            if kind == "tracking_status":
                print("\n[追踪] " + json.dumps(value, ensure_ascii=False), flush=True)

        dry_run = args.dry_run or args.test_api or args.check
        runtime = RobotTools(run_dir, client, dry_run=dry_run, port=args.port, emit=emit,
                             intrinsics=args.intrinsics, hand_eye=args.hand_eye)
        log.save("session.json", {"model": MODEL, "dry_run": dry_run, "intrinsics": str(args.intrinsics),
                                  "hand_eye": str(args.hand_eye), "tool_names": [x["function"]["name"] for x in TOOLS]})
        if args.check:
            state = runtime.get_robot_state()
            if not state["ok"]:
                raise RuntimeError(state)
            print(f"离线检查通过：{MODEL}；六个工具；有效预设 {list(runtime.poses)}；未连接硬件或调用API。")
            return 0
        agent = Agent(client, runtime, log)
        print(f"模型：{MODEL}；深度思考关闭；模式：{'dry-run（LLM真实调用，硬件模拟）' if dry_run else '真机'}")
        print("输入自然语言指令；/quit退出并卸力；Ctrl+C停止当前运动；/help显示示例。")
        print(f"日志：{run_dir}")
        if args.test_api:
            agent.turn("请实际调用get_robot_state查询机器人当前状态和可用预设，再用一句话报告结果。")
            if not any(call["name"] == "get_robot_state" and call["result"].get("ok") for call in agent.turn_calls):
                raise RuntimeError("模型没有成功调用get_robot_state，原生工具调用验证未通过。")
            print("真实API工具调用验证通过。")
            return 0
        if args.commands:
            commands = [line.strip() for line in args.commands.read_text(encoding="utf-8-sig").splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
            for command in commands:
                print("\n你：" + command)
                agent.turn(command)
            return 0
        while True:
            try:
                text = input("\n你：").strip()
                if text in ("/quit", "/exit", "exit", "quit", "退出"):
                    break
                if text == "/help":
                    print((HERE / "test_commands.txt").read_text(encoding="utf-8"))
                    continue
                if text:
                    agent.turn(text)
            except KeyboardInterrupt:
                runtime.interrupt()
                print("\n已请求停止当前运动；可继续输入，或/quit退出。")
            except EOFError:
                break
            except Exception as exc:
                log.event("error", client.redact(exc))
                print("本轮未完成：" + client.redact(exc))
        return 0
    except KeyboardInterrupt:
        if runtime:
            runtime.interrupt()
        return 130
    except Exception as exc:
        redact = client.redact if 'client' in locals() and client else str
        print("Agent未完成：" + redact(exc))
        return 1
    finally:
        if runtime:
            try:
                runtime.close()
            except Exception as exc:
                print(f"退出未完全确认：{exc}")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())

