"""离线验证：不连接相机/串口，不发送网络请求。"""
from concurrent.futures import Future
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from agent import Agent, RunLog
from llm_client import SiliconFlowClient, strict_json
from robot_tools import RobotTools
from tool_specs import TOOLS, TOOL_NAMES


def call(name, arguments, identifier="call-1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class FakeClient:
    model = "Qwen/Qwen3.5-4B"
    redact = staticmethod(str)

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, tools=None):
        self.requests.append(json.loads(json.dumps(messages)))
        return next(self.responses)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = RobotTools(self.temp.name, None, dry_run=True)

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def test_initial_state_and_complete_calibrated_ranges(self):
        result = self.runtime.invoke("get_robot_state", {})
        self.assertTrue(result["ok"])
        self.assertTrue(result["simulated"])
        self.assertEqual(result["available_poses"]["middle"], [383, 683, 641, 649, 688, 112])
        self.assertIn("stand", result["unavailable_poses"])
        self.assertEqual(result["calibrated_ranges"]["shoulder_lift"], (317, 995))
        self.assertEqual(result["calibrated_ranges"]["elbow_flex"], (319, 910))
        self.assertEqual(result["calibrated_ranges"]["wrist_roll"], (191, 1010))
        self.assertEqual(result["calibrated_ranges"]["gripper"], (96, 494))

    def test_no_hardware_constructed_in_dry_run(self):
        with patch("robot_tools.task41.Robot", side_effect=AssertionError("hardware must not open")), \
             patch("vision.Camera", side_effect=AssertionError("camera must not open")):
            self.assertTrue(self.runtime.invoke("get_robot_state", {})["ok"])
            result = self.runtime.invoke("detect_object", {"description": "ArUco 0"})
            self.assertTrue(result["found"])
            self.assertTrue(result["trackable"])
            self.assertAlmostEqual(result["distance_cm"], 25, delta=1)
            self.assertTrue(Path(result["image_path"]).is_file())

    def test_unknown_marker_and_ordinary_object_not_trackable(self):
        for description in ("ArUco 7", "红色杯子"):
            result = self.runtime.invoke("detect_object", {"description": description})
            self.assertFalse(result["found"])
            self.assertFalse(result["trackable"])
        missing = self.runtime.invoke("start_tracking", {"object_id": 7})
        self.assertEqual(missing["error_code"], "target_not_found")

    def test_visual_model_invalid_box_is_discarded_without_inventing_coordinates(self):
        from vision import Vision
        from unittest.mock import Mock
        client = Mock()
        client.chat.return_value = {"content": json.dumps({"found": True, "label": "桌面",
                                   "bbox_normalized": [0, 650, 1000, 1000], "description": "木质桌面"})}
        vision = Vision(Path(self.temp.name) / "vision", client)
        try:
            with patch.object(vision, "frame", return_value=(np.zeros((720, 1280, 3), dtype=np.uint8), time.monotonic())):
                result = vision.detect("桌面")
            self.assertTrue(result["found"])
            self.assertIsNone(result["bbox_normalized"])
            self.assertIn("warning", result)
            self.assertFalse(result["trackable"])
            self.assertTrue(Path(result["image_path"]).is_file())
        finally:
            vision.close()

    def test_rejects_invalid_arguments_and_unregistered_tools(self):
        cases = [("start_tracking", {"object_id": "cup"}), ("start_tracking", {"object_id": True}),
                 ("start_tracking", {"object_id": 50}), ("start_tracking", {"object_id": 0, "target_distance_cm": -1}),
                 ("fk_move", {"joints_raw": [0] * 6}), ("fk_move", {"joints_raw": [True] * 6}),
                 ("fk_move", {"pose_name": "middle", "joints_raw": [383, 683, 641, 649, 688, 112]}),
                 ("fk_move", {"pose_name": "stand"}), ("fk_move", {}),
                 ("ik_move", {"dx_cm": float("nan"), "dy_cm": 0, "dz_cm": 0}),
                 ("get_robot_state", {"unexpected": 1}), ("__getattribute__", {"name": "close"})]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                self.assertFalse(self.runtime.invoke(name, args)["ok"])

    def test_ik_uses_existing_solver_and_preserves_wrist_roll_and_gripper(self):
        result = self.runtime.invoke("ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 1})
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["actual_delta_cm"][2], 1, delta=.1)
        self.assertEqual(result["final_raw"][4:], [688, 112])
        self.assertFalse(result["physical_distance_verified"])

    def test_zero_ik_no_motion(self):
        result = self.runtime.invoke("ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 0})
        self.assertTrue(result["ok"])
        self.assertEqual(self.runtime.robot.sent, [])

    def test_out_of_range_connection_then_corrected_position_can_move(self):
        from robot_adapter import Simulation
        robot = Simulation(self.runtime.mapping)
        robot.raw[2] = 915
        with patch("robot_tools.Simulation", return_value=robot):
            state = self.runtime.get_robot_state()
        self.assertTrue(state["connected"])
        self.assertEqual(state["out_of_range_joints"],
                         [{"name": "elbow_flex", "raw": 915, "min": 319, "max": 910}])
        self.assertIsNotNone(self.runtime.controller)
        blocked = self.runtime.ik_move(0, 0, 1)
        self.assertFalse(blocked["ok"])
        self.assertEqual(robot.sent, [])
        # Reproduce manual repositioning after the first failed request.
        robot.raw = np.rint(self.runtime.mapping.zero).astype(int)
        self.assertTrue(self.runtime.ik_move(0, 0, 1)["ok"])

    def test_failed_controller_initialization_closes_session_and_retries(self):
        from robot_adapter import Simulation
        robot = Simulation(self.runtime.mapping)
        with patch("robot_tools.Simulation", return_value=robot), \
             patch.object(robot, "close", wraps=robot.close) as close, \
             patch("robot_tools.Controller", side_effect=RuntimeError("initialization failed")):
            failed = self.runtime.get_robot_state()
            self.assertFalse(failed["ok"])
            close.assert_called_once()
        self.assertIsNone(self.runtime.robot)
        self.assertIsNone(self.runtime.controller)
        self.assertFalse(self.runtime.state["connected"])
        self.assertTrue(self.runtime.ik_move(0, 0, 1)["ok"])

    def test_fk_accepts_full_range_and_large_absolute_displacement(self):
        target = self.runtime.upper.tolist()
        result = self.runtime.invoke("fk_move", {"joints_raw": target})
        self.assertTrue(result["ok"], result)
        np.testing.assert_array_equal(result["final_raw"], target)
        lower = self.runtime.fk_move(joints_raw=self.runtime.lower.tolist())
        self.assertTrue(lower["ok"], lower)
        self.assertLessEqual(np.max(np.abs(np.array(lower["final_raw"]) - self.runtime.lower)), 5)
        middle = self.runtime.invoke("fk_move", {"pose_name": "middle"})
        self.assertTrue(middle["ok"], middle)
        self.assertEqual(middle["target_raw"], [383, 683, 641, 649, 688, 112])
        self.assertLessEqual(np.max(np.abs(np.array(middle["final_raw"]) - middle["target_raw"])), 5)

    def test_start_stop_tracking_and_motion_exclusion(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.runtime._tracking_step

        def pause_step():
            entered.set()
            release.wait(3)
            original()

        with patch.object(self.runtime, "_tracking_step", side_effect=pause_step):
            started = self.runtime.invoke("start_tracking", {"object_id": 0})
            self.assertTrue(started["ok"])
            self.assertEqual(started["status"], "running")
            self.assertTrue(entered.wait(2))
            snapshot = self.runtime.get_robot_state()
            self.assertEqual(snapshot["tracking"]["status"], "running")
            # 排队的IK在追踪未停止时必须返回冲突，不向同一机械臂同时派发第二套轨迹。
            future = Future()
            self.runtime.queue.put(("ik", {"delta": [0, 0, 1], "speed": 1}, future, self.runtime.cancel_generation))
            release.set()
            self.assertEqual(future.result(timeout=5)["error_code"], "tracking_active")
            result = self.runtime.invoke("stop_tracking", {})
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "stopped")
        self.assertTrue(self.runtime.invoke("stop_tracking", {})["ok"])

    def test_only_serial_worker_reads_positions(self):
        self.runtime.get_robot_state()
        original = self.runtime.robot.read_positions
        threads = []

        def checked_read():
            threads.append(threading.current_thread().name)
            return original()

        with patch.object(self.runtime.robot, "read_positions", side_effect=checked_read):
            self.runtime.get_robot_state()
            self.runtime.fk_move(pose_name="middle")
            self.runtime.ik_move(0, 0, 1)
        self.assertTrue(threads)
        self.assertEqual(set(threads), {"robot-serial-owner"})

    def test_cancelled_queued_move_is_not_executed(self):
        self.runtime.get_robot_state()
        old_generation = self.runtime.cancel_generation
        self.runtime.interrupt()
        future = Future()
        self.runtime.queue.put(("fk", {"target": self.runtime.upper.tolist(), "seconds": 5}, future, old_generation))
        result = future.result(timeout=3)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.runtime.robot.sent, [])

    def test_execution_error_is_reported_not_success(self):
        self.runtime.get_robot_state()
        with patch.object(self.runtime.controller, "move", side_effect=RuntimeError("Input voltage error!")):
            result = self.runtime.ik_move(0, 0, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("Input voltage", result["error"])

    def test_partial_ik_is_not_reported_as_completed(self):
        self.runtime.get_robot_state()
        report = {"success_model_estimate": False, "model_feedback_delta_mm": [0, 0, 2],
                  "reason": "reachable_part_finished", "endpoint_error_mm": 8,
                  "final_raw": self.runtime.mapping.zero.astype(int).tolist()}
        with patch.object(self.runtime.controller, "move", return_value=report):
            result = self.runtime.ik_move(0, 0, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "partial")

    def test_tracking_continues_after_path_deviation_and_reaches_visual_target(self):
        self.runtime.get_robot_state()
        original = self.runtime.controller.move
        reports = []
        def move(*args, **kwargs):
            report = original(*args, **kwargs)
            if not reports:
                # Same completed-step/path-deviation verdict as the user's real log.
                report.update(success_model_estimate=False, reason="path_tolerance_exceeded",
                              endpoint_error_mm=4.013118190470511,
                              max_lateral_feedback_mm=10.77318055504852, final_count_error=2)
            reports.append(report)
            return report
        with patch.object(self.runtime.controller, "move", side_effect=move):
            self.assertTrue(self.runtime.start_tracking(0)["ok"])
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = self.runtime.get_robot_state()
                if state["tracking"]["status"] != "running":
                    break
                time.sleep(.05)
        self.assertEqual(state["tracking"]["status"], "completed", state)
        self.assertGreater(len(reports), 1)
        self.assertEqual(reports[0]["reason"], "path_tolerance_exceeded")

    def test_tracking_partial_reports_and_missing_frames_do_not_end_auto_mode(self):
        self.runtime.get_robot_state()
        original_step = self.runtime._tracking_step
        def step():
            time.sleep(.02)  # Distinct camera timestamps on Windows clocks.
            original_step()
        partial = {"ok": False, "status": "partial", "reason": "ik_blocked",
                   "actual_delta_cm": [0., 0., 0.], "endpoint_error_mm": 10.}
        # Drive the simulated steps deterministically; background loop is idle.
        with patch.object(self.runtime, "_tracking_step"), \
             patch.object(self.runtime, "_ik", return_value=partial) as ik:
            self.runtime.start_tracking(0)
            with patch.object(self.runtime.vision, "observe",
                              side_effect=lambda _: (None, np.zeros((720, 1280, 3)), time.monotonic())):
                for _ in range(5):
                    step()
            self.assertEqual(self.runtime.state["tracking"]["status"], "running")
            self.assertFalse(ik.called)
            for _ in range(16):
                step()
            self.assertEqual(ik.call_count, 4)
            self.assertEqual(self.runtime.state["tracking"]["status"], "running")
            self.assertEqual(self.runtime.state["tracking"]["steps"], 4)
            self.assertTrue(self.runtime.stop_tracking()["ok"])

    def test_tracking_requires_new_stable_frames_after_each_move(self):
        self.runtime.get_robot_state()
        original_step = self.runtime._tracking_step
        def step():
            time.sleep(.02)
            original_step()
        result = {"ok": True, "reason": "finished", "actual_delta_cm": [0., 0., 1.]}
        with patch.object(self.runtime, "_tracking_step"), \
             patch.object(self.runtime, "_ik", return_value=result) as ik:
            self.runtime.start_tracking(0)
            for _ in range(4):
                step()
            self.assertEqual(ik.call_count, 1)
            observation, frame, _ = self.runtime.vision.observe(0)
            stale = self.runtime.tracking_frame_after - .01
            with patch.object(self.runtime.vision, "observe", return_value=(observation, frame, stale)):
                for _ in range(4):
                    step()
            self.assertEqual(ik.call_count, 1)
            for _ in range(3):
                step()
            self.assertEqual(ik.call_count, 1)
            step()
            self.assertEqual(ik.call_count, 2)
            self.runtime.stop_tracking()


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = RobotTools(self.temp.name, None, dry_run=True)
        self.log = RunLog(Path(self.temp.name) / "log")

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def test_native_tool_loop_returns_matching_tool_call_ids(self):
        client = FakeClient([call("get_robot_state", {}), {"role": "assistant", "content": "当前是模拟状态。"}])
        agent = Agent(client, self.runtime, self.log, output=lambda x: None)
        self.assertEqual(agent.turn("读取状态"), "当前是模拟状态。")
        message = client.requests[1][-1]
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call-1")
        self.assertTrue(json.loads(message["content"])["simulated"])

    def test_dependent_stop_ik_state_tools_are_executed_in_order(self):
        client = FakeClient([call("stop_tracking", {}, "c1"), call("ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 1}, "c2"),
                             call("get_robot_state", {}, "c3"), {"role": "assistant", "content": "完成模拟。"}])
        agent = Agent(client, self.runtime, self.log, output=lambda x: None)
        agent.turn("停止追踪，上移1厘米，再查状态")
        self.assertEqual([item["name"] for item in agent.turn_calls], ["stop_tracking", "ik_move", "get_robot_state"])
        self.assertTrue(all(item["result"]["ok"] for item in agent.turn_calls))

    def test_replayed_call_id_does_not_repeat_motion(self):
        repeated = call("ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 1})
        client = FakeClient([repeated, repeated, {"role": "assistant", "content": "完成模拟。"}])
        agent = Agent(client, self.runtime, self.log, output=lambda x: None)
        agent.turn("上移1厘米")
        self.assertEqual(len(agent.turn_calls), 1)

    def test_new_turn_reusing_id_refreshes_state(self):
        query = call("get_robot_state", {})
        answer = {"role": "assistant", "content": "模拟状态已读取。"}
        agent = Agent(FakeClient([query, answer, query, answer]), self.runtime, self.log, output=lambda x: None)
        with patch.object(self.runtime, "invoke", wraps=self.runtime.invoke) as invoke:
            agent.turn("查状态")
            agent.turn("再查状态")
        self.assertEqual(invoke.call_count, 2)

    def test_failed_tool_prevents_dependent_motion_in_same_batch(self):
        response = call("ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 1})
        response["tool_calls"] += call("fk_move", {"pose_name": "middle"}, "call-2")["tool_calls"]
        agent = Agent(FakeClient([response, {"role": "assistant", "content": "运动失败，未执行后续动作。"}]),
                      self.runtime, self.log, output=lambda x: None)
        with patch.object(self.runtime, "invoke", return_value={"ok": False, "status": "failed"}) as invoke:
            agent.turn("上移后回中位")
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(json.loads(agent.messages[-2]["content"])["status"], "not_executed")

    def test_duplicate_ids_in_one_response_are_rejected_before_execution(self):
        response = call("get_robot_state", {})
        response["tool_calls"] *= 2
        agent = Agent(FakeClient([response]), self.runtime, self.log, output=lambda x: None)
        with self.assertRaises(ValueError):
            agent.turn("读取状态")
        self.assertIsNone(self.runtime.robot)

    def test_malformed_arguments_are_returned_to_model(self):
        response = call("fk_move", {})
        response["tool_calls"][0]["function"]["arguments"] = '{"pose_name":"middle","pose_name":"stand"}'
        client = FakeClient([response, {"role": "assistant", "content": "参数错误。"}])
        agent = Agent(client, self.runtime, self.log, output=lambda x: None)
        agent.turn("回中位")
        self.assertFalse(json.loads(client.requests[1][-1]["content"])["ok"])
        self.assertIsNone(self.runtime.robot)

    def test_network_failure_closes_turn_without_repeating_executed_tool(self):
        client = FakeClient([call("get_robot_state", {})])
        agent = Agent(client, self.runtime, self.log, output=lambda x: None)
        with self.assertRaises(StopIteration):
            agent.turn("查询状态")
        self.assertEqual(len(agent.turn_calls), 1)
        self.assertEqual(agent.messages[-1]["role"], "assistant")
        self.assertIn("未执行动作取消", agent.messages[-1]["content"])

    def test_model_request_interrupt_marks_turn_cancelled(self):
        client = FakeClient([])
        agent = Agent(client, self.runtime, self.log, output=lambda x: None)
        with patch.object(client, "chat", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            agent.turn("上移1厘米")
        self.assertTrue(self.runtime.cancel.is_set())
        self.assertIn("未执行动作取消", agent.messages[-1]["content"])

    def test_interrupt_does_not_leave_unmatched_tool_calls(self):
        response = call("fk_move", {"pose_name": "middle"})
        response["tool_calls"] += call("ik_move", {"dx_cm": 0, "dy_cm": 0, "dz_cm": 1}, "call-2")["tool_calls"]
        agent = Agent(FakeClient([response]), self.runtime, self.log, output=lambda x: None)
        with patch.object(self.runtime, "invoke", side_effect=KeyboardInterrupt):
            agent.turn("回中位再上移")
        self.assertEqual([message["tool_call_id"] for message in agent.messages if message["role"] == "tool"], ["call-1", "call-2"])
        self.assertTrue(all(json.loads(message["content"])["status"] == "cancelled"
                            for message in agent.messages if message["role"] == "tool"))

    def test_schema_contains_exactly_six_tools(self):
        self.assertEqual(set(TOOL_NAMES), {"get_robot_state", "detect_object", "start_tracking", "stop_tracking", "ik_move", "fk_move"})
        self.assertEqual(len(TOOLS), 6)


class ClientTests(unittest.TestCase):
    def test_payload_uses_requested_model_and_thinking_off(self):
        response = {"choices": [{"finish_reason": "tool_calls", "message": call("get_robot_state", {})}]}
        captured = []

        def fake_open(request, timeout):
            captured.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())

        with patch("llm_client.urlopen", side_effect=fake_open):
            message = SiliconFlowClient("unit-secret").chat([{"role": "user", "content": "查询状态"}], tools=TOOLS)
        self.assertEqual(captured[0]["model"], "Qwen/Qwen3.5-4B")
        self.assertFalse(captured[0]["enable_thinking"])
        self.assertEqual(captured[0]["tools"], TOOLS)
        self.assertEqual(message["tool_calls"][0]["id"], "call-1")

    def test_http_errors_redact_key_and_truncated_output_is_not_executed(self):
        error = HTTPError("https://example.invalid", 400, "bad", {}, io.BytesIO(b"unit-secret wrong"))
        with patch("llm_client.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                SiliconFlowClient("unit-secret").chat([])
        self.assertNotIn("unit-secret", str(caught.exception))
        response = {"choices": [{"finish_reason": "length", "message": call("fk_move", {"pose_name": "middle"})}]}
        with patch("llm_client.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(ValueError):
                SiliconFlowClient("unit-secret").chat([])

    def test_log_redaction_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as folder:
            log = RunLog(folder, SiliconFlowClient("unit-secret").redact)
            log.event("test", {"value": "unit-secret"})
            self.assertNotIn("unit-secret", (Path(folder) / "events.jsonl").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            strict_json('{"dx_cm": NaN}')

    def test_transient_http_retry_returns_one_message(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}
        with patch("llm_client.urlopen", side_effect=[TimeoutError("timeout"), io.BytesIO(json.dumps(response).encode())]) as request, \
             patch("llm_client.time.sleep"):
            message = SiliconFlowClient("unit-secret").chat([])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(message["content"], "ok")


if __name__ == "__main__":
    unittest.main()
