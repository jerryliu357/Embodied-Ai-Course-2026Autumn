"""六个工具的执行层：一个串口工作线程，复用lec4/lec5控制器。"""
from concurrent.futures import Future
import copy
import inspect
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

from runtime_paths import POSES, LESSON4
import task3 as task41
from robot_adapter import NAMES, Simulation, load_config
from trajectory import Controller, MotionStopped
from aruco_visual_servo_core import camera_delta_to_base, choose_step
from tool_specs import TOOL_NAMES
from vision import Vision


def finite(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label}必须为{'正' if positive else '有限'}数值")
    return float(value)


class SimulationClock:
    def __init__(self):
        self.value = 0.

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class RobotTools:
    def __init__(self, run_dir, client, dry_run=False, port=None, emit=lambda *args: None, **vision_options):
        self.run_dir, self.dry_run, self.emit = Path(run_dir), dry_run, emit
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg = load_config()
        if port:
            cfg["port"] = port
        # 沿用4.1已确认的完整标定区间，不额外收窄为旧URDF限位。
        self.mapping = task41.Calibration(cfg)
        self.lower, self.upper = task41.physical_bounds(self.mapping)
        self.poses, self.missing_poses = self._load_poses()
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.cancel_generation = 0
        self.closing = threading.Event()
        self.close_error = None
        self.queue = queue.Queue()
        self.robot = self.controller = None
        self.tracking_observations = []
        self.tracking_frame_after = 0.
        self.tracking_last_stamp = 0.
        self.state = {"connected": False, "simulated": dry_run, "sampled_at": None,
                      "active_motion": None, "last_result": None, "error": None,
                      "tracking": {"status": "idle", "object_id": None}}
        self.T_BE = self.mapping.arm.fk(self.mapping.to_q(self.mapping.zero))
        self.vision = Vision(self.run_dir / "photos", client, dry_run=dry_run, **vision_options)
        self.vision.pose_provider = self.pose_matrix
        self.worker = threading.Thread(target=self._worker, name="robot-serial-owner", daemon=True)
        self.worker.start()

    def _load_poses(self):
        document = json.loads(POSES.read_text(encoding="utf-8-sig"))
        calibration = json.loads((LESSON4 / "FK" / "calibration.json").read_text(encoding="utf-8-sig"))
        # Stored ratios belong to the calibration used when recording the pose.
        # Widening current bounds must not rescale an existing physical posture.
        source_joints = [calibration["joints"][str(i + 1)] for i in range(6)]
        compatible = (
            calibration["degrees_per_count"] == self.mapping.degrees_per_count
            and all(joint["angle_zero_position"] == self.mapping.zero[i]
                    and joint["raw_direction"] == 1 for i, joint in enumerate(source_joints)))
        source_minimum = np.array([joint["min_position"] for joint in source_joints])
        source_maximum = np.array([joint["max_position"] for joint in source_joints])
        poses, missing = {}, []
        for name, entry in document["poses"].items():
            values = entry.get("values")
            if not compatible or values is None:
                missing.append(name)
                continue
            ratios = np.asarray(values, dtype=float)
            if ratios.shape != (6,) or not np.isfinite(ratios).all() or np.any((ratios < 0) | (ratios > 1)):
                raise ValueError(f"已保存姿态{name}无效")
            raw = np.rint(source_minimum + ratios * (source_maximum - source_minimum)).astype(int)
            try:
                task41.validate_positions(raw, self.lower, self.upper)
            except ValueError:
                missing.append(name)
                continue
            poses[name] = raw.tolist()
        return poses, missing

    def pose_matrix(self):
        with self.lock:
            return self.T_BE.copy()

    def _publish_positions(self, raw):
        q = self.mapping.to_q(raw)
        matrix = self.mapping.arm.fk(q)
        with self.lock:
            self.T_BE = matrix
            self.state.update(joints_raw=dict(zip(NAMES, map(int, raw))),
                              joint_angles_deg=dict(zip(NAMES[:5], map(float, np.rad2deg(q)))),
                              gripper_raw=int(raw[5]), tcp_position_cm=(matrix[:3, 3] * 100).tolist(),
                              tcp_frame="base", tcp_source="encoder_feedback_and_fk_model",
                              sampled_at=time.time())

    def _connect(self):
        if self.robot is not None and self.controller is not None:
            return
        # Publish a connected session only after every initialization step succeeds.
        robot = self.robot
        try:
            if robot is None:
                robot = Simulation(self.mapping) if self.dry_run else task41.Robot(self.mapping, unload_on_close=True)
            if self.dry_run:
                robot.hardware_limits = np.column_stack((self.lower, self.upper))
            lower, upper = task41.physical_bounds(self.mapping, robot.hardware_limits)
            initial_raw = robot.read_positions()
            controller = Controller(robot, realtime=not self.dry_run)
        except BaseException:
            self.robot = self.controller = None
            with self.lock:
                self.state["connected"] = False
            if robot is not None:
                robot.close()
            raise
        original_read = robot.read_positions

        def observed_read():
            raw = original_read()
            self._publish_positions(raw)
            return raw

        robot.read_positions = observed_read
        self.robot, self.controller = robot, controller
        self.lower, self.upper = lower, upper
        self._publish_positions(initial_raw)
        with self.lock:
            self.state.update(connected=True, error=None)
        self.emit("connected", {"simulated": self.dry_run})

    def _cached_state(self):
        with self.lock:
            result = copy.deepcopy(self.state)
        result.update(ok=True, available_poses=self.poses, unavailable_poses=self.missing_poses,
                      calibrated_ranges=dict(zip(NAMES, zip(self.lower.tolist(), self.upper.tolist()))))
        result["sample_age_s"] = None if result["sampled_at"] is None else round(time.time() - result["sampled_at"], 3)
        measured = result.get("joints_raw", {})
        result["out_of_range_joints"] = [
            {"name": name, "raw": measured[name], "min": int(lo), "max": int(hi)}
            for name, lo, hi in zip(NAMES, self.lower, self.upper)
            if name in measured and not lo <= measured[name] <= hi]
        return result

    def _submit(self, name, **arguments):
        if self.closing.is_set():
            return {"ok": False, "error": "程序正在退出"}
        future = Future()
        with self.lock:
            generation = self.cancel_generation
        self.queue.put((name, arguments, future, generation))
        try:
            return future.result()
        except KeyboardInterrupt:
            self.interrupt()
            try:
                future.result(timeout=20)
            except Exception:
                pass
            raise

    def invoke(self, name, arguments):
        if name not in TOOL_NAMES:
            return {"ok": False, "error_code": "unknown_tool", "error": f"未注册工具：{name}"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error_code": "invalid_arguments", "error": "参数必须为JSON对象"}
        try:
            function = getattr(self, name)
            inspect.signature(function).bind(**arguments)
            result = function(**arguments)
            result.setdefault("simulated", self.dry_run)
            return result
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error_code": "invalid_arguments", "error": str(exc), "simulated": self.dry_run}
        except Exception as exc:
            return {"ok": False, "error_code": "tool_failed", "error": str(exc), "simulated": self.dry_run}

    def get_robot_state(self):
        with self.lock:
            active = self.state["active_motion"] or self.state["tracking"]["status"] == "running"
        # 运动中返回由同一串口线程持续发布的反馈，不从第二个线程访问串口。
        return self._cached_state() if active else self._submit("state")

    def detect_object(self, description):
        return self.vision.detect(description)

    def start_tracking(self, object_id, target_distance_cm=20):
        if type(object_id) is not int or not 0 <= object_id < 50:
            raise ValueError("只支持DICT_4X4_50的ArUco整数ID 0~49")
        distance = finite(target_distance_cm, "target_distance_cm", positive=True)
        return self._submit("start", object_id=object_id, distance=distance)

    def stop_tracking(self):
        with self.lock:
            if self.state["tracking"]["status"] == "running":
                self.cancel.set()
        return self._submit("stop")

    def ik_move(self, dx_cm, dy_cm, dz_cm, speed_cm_s=1.):
        delta = [finite(value, label) for value, label in zip((dx_cm, dy_cm, dz_cm), ("dx_cm", "dy_cm", "dz_cm"))]
        speed = finite(speed_cm_s, "speed_cm_s", positive=True)
        return self._submit("ik", delta=delta, speed=speed)

    def fk_move(self, pose_name=None, joints_raw=None, duration_s=5.):
        if (pose_name is None) == (joints_raw is None):
            raise ValueError("pose_name与joints_raw必须二选一")
        if pose_name is not None:
            if not isinstance(pose_name, str) or pose_name not in self.poses:
                raise ValueError(f"姿态{pose_name!r}未录制或标定不匹配；当前可用：{list(self.poses)}")
            joints_raw = self.poses[pose_name]
        if not isinstance(joints_raw, list) or len(joints_raw) != 6 or any(type(v) is not int for v in joints_raw):
            raise ValueError("joints_raw必须是按ID1~6排序的六个整数原始计数")
        task41.validate_positions(joints_raw, self.lower, self.upper)
        duration = finite(duration_s, "duration_s", positive=True)
        return self._submit("fk", target=joints_raw, seconds=duration)

    def _save_motion(self, kind, report):
        path = self.run_dir / f"{kind}_{time.time_ns()}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _ik(self, delta_cm, speed):
        delta_m = np.asarray(delta_cm, dtype=float) / 100
        if not np.any(delta_m):
            return {"ok": True, "status": "completed", "actual_delta_cm": [0., 0., 0.], "reason": "零位移，无需运动"}
        self.controller.speed_cm_s = speed
        report = self.controller.move(delta_m, cancel=self.cancel.is_set,
                                      emit=lambda message: self.emit("motion_progress", {"message": message}))
        log = self._save_motion("ik", report)
        complete = report["success_model_estimate"]
        return {"ok": bool(complete), "status": "completed" if complete else "partial",
                "requested_delta_cm": list(delta_cm),
                "actual_delta_cm": (np.asarray(report["model_feedback_delta_mm"]) / 10).tolist(),
                "reason": report["reason"], "endpoint_error_mm": report["endpoint_error_mm"],
                "max_lateral_feedback_mm": report.get("max_lateral_feedback_mm"),
                "final_count_error": report.get("final_count_error"),
                "final_raw": report["final_raw"], "physical_distance_verified": False, "motion_log": log}

    def _handle(self, name, arguments):
        self._connect()
        if name == "state":
            self.robot.read_positions()
            self.robot.check_health()
            return self._cached_state()
        if name == "stop":
            was_running = self.state["tracking"]["status"] == "running"
            with self.lock:
                self.state["tracking"]["status"] = "stopped"
            if self.robot.owned:
                self.robot.hold()
            self.controller = Controller(self.robot, realtime=not self.dry_run)
            self.robot.read_positions()
            return {"ok": True, "status": "stopped", "was_running": was_running, "holding": self.robot.owned}
        if self.state["tracking"]["status"] == "running":
            return {"ok": False, "error_code": "tracking_active", "error": "正在追踪，请先调用stop_tracking再切换动作。"}
        task41.validate_positions(self.robot.read_positions(), self.lower, self.upper)
        self.cancel.clear()
        if name == "start":
            observation, _, _ = self.vision.observe(arguments["object_id"])
            if observation is None:
                return {"ok": False, "error_code": "target_not_found", "error": "新画面中未发现指定ArUco，未启动。"}
            task_id = uuid.uuid4().hex[:12]
            self.tracking_observations.clear()
            self.tracking_frame_after = time.monotonic()
            self.tracking_last_stamp = 0.
            with self.lock:
                self.state["tracking"] = {"status": "running", "task_id": task_id, "object_id": arguments["object_id"],
                                          "target_distance_cm": arguments["distance"], "steps": 0,
                                          "phase": "observing"}
            return {"ok": True, **self.state["tracking"]}
        with self.lock:
            self.state["active_motion"] = name
        try:
            if name == "ik":
                result = self._ik(arguments["delta"], arguments["speed"])
            elif name == "fk":
                start = self.robot.read_positions()
                target = np.asarray(arguments["target"], dtype=int)
                kwargs = {}
                if self.dry_run:
                    clock = SimulationClock()
                    kwargs = {"clock": clock.now, "sleep": clock.sleep}
                report = task41.execute_motion(self.robot, start, target, self.lower, self.upper,
                                               arguments["seconds"], cancel=self.cancel.is_set, **kwargs)
                log = self._save_motion("fk", report)
                self.controller = Controller(self.robot, realtime=not self.dry_run)
                result = {"ok": True, "status": "completed", "final_raw": report["final_raw"],
                          "target_raw": report["target_raw"],
                          "max_error_counts": report["max_error_counts"], "motion_log": log}
            else:
                raise ValueError(f"未知内部命令：{name}")
            self.robot.read_positions()
            result["tcp_position_cm"] = self.state["tcp_position_cm"]
            with self.lock:
                self.state.update(last_result=result, error=None)
            return result
        finally:
            with self.lock:
                self.state["active_motion"] = None

    def _tracking_step(self):
        with self.lock:
            tracking = self.state["tracking"].copy()

        def publish():
            with self.lock:
                self.state["tracking"] = tracking.copy()

        if self.cancel.is_set() or self.closing.is_set():
            tracking["status"] = "stopped"
            publish()
            self.robot.hold()
            return
        observation, frame, stamp = self.vision.observe(tracking["object_id"])
        # As in lec5: only fresh, stable post-motion observations decide the next step.
        if stamp <= max(self.tracking_frame_after, self.tracking_last_stamp):
            return
        self.tracking_last_stamp = stamp
        if observation is None:
            self.tracking_observations.clear()
            tracking.update(phase="observing", reason="暂未检测到目标，等待新画面")
            publish()
            return
        self.tracking_observations.append([*observation.center_px, observation.depth_m])
        self.tracking_observations = self.tracking_observations[-4:]
        tracking.update(distance_cm=observation.depth_m * 100, center_px=observation.center_px.tolist(),
                        observed_at=time.time() - (time.monotonic() - stamp), phase="observing")
        recent = np.asarray(self.tracking_observations)
        if (len(recent) < 4 or np.max(np.ptp(recent[:, :2], axis=0)) > 6
                or np.ptp(recent[:, 2]) > .012 or observation.reprojection_rms_px > 3):
            tracking["reason"] = "等待新的稳定图像，再决定下一步"
            publish()
            return
        decision = choose_step(observation, (frame.shape[1], frame.shape[0]), target_depth_m=tracking["target_distance_cm"] / 100)
        tracking.update(action=decision.action, reason=decision.explanation)
        publish()
        if decision.done:
            tracking.update(status="completed" if decision.action == "done" else "too_close", reason=decision.explanation)
            publish()
            self.emit("tracking_status", tracking.copy())
            return
        raw = self.robot.read_positions()
        matrix = self.mapping.arm.fk(self.mapping.to_q(raw))
        delta = camera_delta_to_base(matrix, self.vision.T_EC, decision.camera_delta_m)
        tracking["phase"] = "moving"
        publish()
        result = self._ik((delta * 100).tolist(), self.mapping.cfg["speed_cm_s"])
        tracking["steps"] += 1
        self.tracking_observations.clear()
        self.tracking_frame_after = time.monotonic()
        tracking.update(phase="observing", last_step_reason=result["reason"],
                        last_step_ok=result["ok"], endpoint_error_mm=result.get("endpoint_error_mm"),
                        max_lateral_feedback_mm=result.get("max_lateral_feedback_mm"),
                        reason="本步结束，等待新的稳定画面后继续纠偏")
        with self.lock:
            self.state["last_result"] = result
        # A partial straight-line result is not a failed visual-servo task.
        # Reobserve/replan, rather than replaying the old delta or claiming arrival.
        publish()
        self.emit("tracking_status", tracking.copy())

    def _failure(self, exc):
        cancelled = isinstance(exc, (KeyboardInterrupt, MotionStopped))
        result = {"ok": False, "status": "cancelled" if cancelled else "failed", "error": str(exc) or "用户停止",
                  "error_code": "cancelled" if cancelled else "execution_error"}
        if self.robot is not None:
            try:
                self.robot.hold()
            except Exception as hold_error:
                result["hold_error"] = str(hold_error)
        with self.lock:
            self.state.update(active_motion=None, last_result=result, error=None if cancelled else result["error"])
            if self.state["tracking"]["status"] == "running":
                self.state["tracking"].update(status="stopped" if cancelled else "failed", reason=result["error"])
        return result

    def _worker(self):
        try:
            while True:
                try:
                    name, arguments, future, generation = self.queue.get(timeout=.1)
                except queue.Empty:
                    name = None
                if name == "close":
                    break
                if name is not None:
                    try:
                        if generation != self.cancel_generation and name not in ("state", "stop"):
                            result = {"ok": False, "status": "cancelled", "error": "排队动作已被用户取消，未执行。"}
                        else:
                            result = self._handle(name, arguments)
                    except (Exception, KeyboardInterrupt) as exc:
                        result = self._failure(exc)
                    future.set_result(result)
                if self.state["tracking"]["status"] == "running" and self.queue.empty() and not self.closing.is_set():
                    try:
                        self._tracking_step()
                    except (Exception, KeyboardInterrupt) as exc:
                        self.emit("tracking_status", self._failure(exc))
        finally:
            try:
                if self.robot is not None:
                    self.robot.close()  # 真实机器人由4.1关闭逻辑卸力并确认。
                    release = getattr(self.robot, "release_result", None)
                    if release is not None and not release.get("confirmed"):
                        self.close_error = "六个舵机卸力未全部确认"
            except Exception as exc:
                self.close_error = str(exc)
            finally:
                with self.lock:
                    self.state["connected"] = False

    def interrupt(self):
        with self.lock:
            self.cancel_generation += 1
            self.cancel.set()

    def close(self):
        if self.closing.is_set():
            return
        self.closing.set()
        self.cancel.set()
        self.queue.put(("close", {}, Future(), self.cancel_generation))
        self.worker.join(timeout=20)
        self.vision.close()
        if self.worker.is_alive():
            raise RuntimeError("机械臂线程尚未结束，退出卸力未确认。")
        if self.close_error:
            raise RuntimeError(self.close_error)

