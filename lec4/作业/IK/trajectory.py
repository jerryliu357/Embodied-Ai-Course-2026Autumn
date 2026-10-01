"""Cartesian planning with feedback-paced, continuously corrected execution."""
from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import time

import numpy as np

from ik_solver import PositionIK, cartesian_step
from tracking import JointTracker, pace


class MotionStopped(RuntimeError):
    pass


@dataclass
class Plan:
    start_raw: np.ndarray
    requested_delta: np.ndarray
    points: np.ndarray
    angles: np.ndarray
    times: np.ndarray
    modes: list[str]
    blocked: bool


class Planner:
    def __init__(self, mapping):
        self.mapping, self.arm, self.cfg = mapping, mapping.arm, mapping.cfg
        self.ik = PositionIK(self.arm, tolerance_m=.00005, max_iterations=40)

    def _local(self, target, q, center=False):
        local_arm = copy(self.ik.arm)
        local_arm.lower = np.maximum(self.ik.arm.lower, q-np.deg2rad(6))
        local_arm.upper = np.minimum(self.ik.arm.upper, q+np.deg2rad(6))
        seed = q.copy()
        if center:
            # Redundant joint motion in the position Jacobian's nullspace:
            # move away from end stops while keeping the TCP position fixed.
            jac = self.arm.fk_jacobian(q)[1][:, :4]
            null = np.eye(4)-np.linalg.pinv(jac)@jac
            desired = (self.arm.lower[:4]+self.arm.upper[:4])/2-q[:4]
            shift = null @ desired
            shift *= min(1., np.deg2rad(3)/max(1e-12, np.max(np.abs(shift))))
            seed[:4] += shift
            seed = np.clip(seed, local_arm.lower, local_arm.upper)
        return PositionIK(local_arm, tolerance_m=.00005, max_iterations=60).solve(target, seed)

    def build(self, start_raw, delta, speed_cm_s, cancel=lambda: False):
        self.mapping.check_start(start_raw)
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("需要三个有限位移值。")
        length = float(np.linalg.norm(delta))
        if not .0001 <= length <= .5:
            raise ValueError("位移总长度必须在 0.01～50 cm。")
        if not np.isfinite(speed_cm_s) or not .1 <= speed_cm_s <= 3:
            raise ValueError("速度必须在 0.1～3 cm/s。")
        q = self.mapping.to_q(start_raw)
        # Wrist roll is redundant for XYZ translation. Preserve it instead of
        # spending hundreds of counts near a singularity for a sub-mm TCP gain.
        local_arm = copy(self.arm)
        local_arm.lower, local_arm.upper = self.arm.lower.copy(), self.arm.upper.copy()
        local_arm.lower[4] = local_arm.upper[4] = q[4]
        self.ik = PositionIK(local_arm, tolerance_m=.00005, max_iterations=60)
        start = self.arm.fk(q)[:3, 3]
        direction = delta / length
        qs, points, modes = [q.copy()], [start.copy()], ["start"]
        blocked = False
        nominal = self.cfg["planning_step_mm"] / 1000
        posture_steps = 0
        for _ in range(1000):
            if cancel():
                raise MotionStopped("已取消规划，未发送新的运动目标。")
            progress = float((points[-1] - start) @ direction)
            if progress >= length - self.ik.tolerance:
                break
            step = min(nominal, length - progress)
            chosen = None
            # All primary targets lie on the ORIGINAL fixed base-frame line.
            for fraction in (1., .5, .25, .125):
                target = start + (progress + fraction * step) * direction
                result = self._local(target, q, center=True)
                if not result.success:
                    result = self._local(target, q)
                if result.success:
                    chosen = (result.q, result.position, "line" if fraction == 1 else "shortened")
                    break
            if chosen is None:
                same_point = self._local(points[-1], q, center=True)
                center_q = (self.arm.lower[:4]+self.arm.upper[:4])/2
                improved = np.linalg.norm(same_point.q[:4]-center_q) < np.linalg.norm(q[:4]-center_q)-1e-4
                if same_point.success and improved and posture_steps < 60:
                    q = same_point.q
                    qs.append(q.copy()); points.append(points[-1].copy()); modes.append("posture")
                    posture_steps += 1
                    continue
                # Keep the desired forward component while respecting limits.
                result = cartesian_step(self.ik, q, direction * step, max_joint_delta=np.deg2rad(6))
                if result.mode == "blocked":
                    blocked = True
                    break
                chosen = (result.q, result.position, "projected")
            q_next, p_next, mode = chosen
            advance = float((p_next - points[-1]) @ direction)
            if advance <= 5e-5 or advance > length - progress + self.ik.tolerance:
                blocked = True
                break
            qs.append(q_next.copy()); points.append(p_next.copy()); modes.append(mode)
            q = q_next
        else:
            blocked = True
        times = [0.]
        for i in range(1, len(qs)):
            cart_time = np.linalg.norm(points[i] - points[i-1]) / (speed_cm_s / 100)
            joint_time = np.max(np.abs(qs[i] - qs[i-1])) / np.deg2rad(self.cfg["max_joint_speed_deg_s"])
            times.append(times[-1] + max(float(cart_time), float(joint_time), .02))
        return Plan(np.asarray(start_raw).copy(), delta, np.array(points), np.array(qs),
                    np.array(times), modes, blocked)

    def frames(self, plan):
        """Timed joint interpolation, quantized through the actual encoder mapping.

        IK nodes are spaced 5 mm by default. Frames are streamed at 20 Hz, without
        waiting for any intermediate target to be reached. Joint interpolation
        approximates each short Cartesian segment; its error is measured below.
        """
        period = 1 / self.cfg["control_hz"]
        times = list(np.arange(0., plan.times[-1], period)) + [float(plan.times[-1])]
        frames = []
        for t in times:
            idx = min(max(int(np.searchsorted(plan.times, t, side="right")), 1), len(plan.times)-1)
            if idx == 0:
                q, p = plan.angles[0], plan.points[0]
            else:
                fraction = (t-plan.times[idx-1]) / (plan.times[idx]-plan.times[idx-1])
                q = plan.angles[idx-1] + fraction * (plan.angles[idx]-plan.angles[idx-1])
                p = plan.points[idx-1] + fraction * (plan.points[idx]-plan.points[idx-1])
            raw = np.r_[self.mapping.quantize(q, p), plan.start_raw[5]].astype(int)
            if t == 0.:
                raw = plan.start_raw.copy()
            frames.append((t, raw))
        return frames


class Controller:
    def __init__(self, session, clock=time.monotonic, sleep=time.sleep, realtime=True):
        self.session, self.mapping, self.cfg = session, session.mapping, session.cfg
        self.planner = Planner(self.mapping)
        self.speed_cm_s = self.cfg["speed_cm_s"]
        self.clock, self.sleep, self.realtime = clock, sleep, realtime
        self.last_report = None
        self.feedback, self.trace, self.sent = [], [], []
        self.holding_bias = None
        self.holding_command = None

    def _wait_until(self, deadline, cancel):
        if cancel():
            raise MotionStopped("用户停止运动。")
        if not self.realtime:
            return
        while self.clock() < deadline:
            if cancel():
                raise MotionStopped("用户停止运动。")
            self.sleep(min(.01, max(0, deadline-self.clock())))

    @staticmethod
    def _sample(plan, phase):
        index = min(max(int(np.searchsorted(plan.times, phase, side="right")), 1), len(plan.times)-1)
        fraction = np.clip((phase-plan.times[index-1]) / (plan.times[index]-plan.times[index-1]), 0, 1)
        point = plan.points[index-1] + fraction * (plan.points[index]-plan.points[index-1])
        q = plan.angles[index-1] + fraction * (plan.angles[index]-plan.angles[index-1])
        return point, q, plan.modes[index]

    def _reference(self, target, actual, nominal_q):
        # Correct interpolation to the exact Cartesian line while preserving the
        # chosen joint branch. Re-seeding from noisy measured angles each cycle
        # made the redundant posture drift into an elbow limit on reverse moves.
        # Actual feedback still controls phase, Cartesian lag and joint PI.
        result = self.planner.ik.solve(target, nominal_q, max_joint_delta=np.deg2rad(3))
        if not result.success:
            return None
        return np.r_[self.mapping.quantize(result.q, target), actual[5]].astype(int)

    def move(self, delta, cancel=lambda: False, emit=print):
        self.last_report = None
        initial = self.session.read_positions()
        existing = None
        if not self.session.owned and hasattr(self.session, "adopt_stable_hold"):
            existing = self.session.adopt_stable_hold(initial)
        plan = self.planner.build(initial, delta, self.speed_cm_s, cancel)
        frames = self.planner.frames(plan)  # offline preflight only, NOT a send queue
        self.feedback, self.trace, self.sent = [initial.copy()], [], [initial.copy()]
        if len(plan.points) < 2:
            emit("IK 无解：当前未找到可行前进方向；未发运动。")
            return self._report(plan, initial, initial, initial, "ik_blocked", False)
        if plan.blocked or "projected" in plan.modes:
            emit("IK 无解（部分原直线目标，当前局部求解未找到解）：执行可达部分，尽量保留原运动方向。")
        elif "shortened" in plan.modes:
            emit("部分规划段已缩短以避开关节限位，仍沿原直线到达目标。")
        if "projected" in plan.modes:
            emit("包含可行方向投影：此部分会偏离原直线，将单独报告偏差。")
        if "posture" in plan.modes:
            emit("为避开关节限位，路径包含保持末端位置的关节姿态调整。")
        emit(f"反馈跟随：设定速度上限 {self.speed_cm_s:g} cm/s，理想用时约 {plan.times[-1]:.1f} 秒；"
             f"落后时自动减速/暂停推进并纠偏，中间点不停车验收到位。终点容差 {self.cfg['final_count_tolerance']} 计数。")
        self.session.preflight(np.array([raw for _, raw in frames]))
        actual = self.session.read_positions()
        if np.max(np.abs(actual-initial)) > self.cfg["final_count_tolerance"]:
            raise RuntimeError("规划期间关节位置改变，请托稳后重新输入命令；未执行旧轨迹。")
        tracker = JointTracker(self.mapping, actual, getattr(self.session, "hardware_limits", None))
        reference, command = initial.copy(), initial.copy()
        if existing is not None:
            tracker.bias = (existing[:5]-actual[:5]).astype(float)
            tracker.output = existing[:5].astype(float)
            command = existing.copy()
        if (self.session.owned and self.holding_bias is not None
                and np.array_equal(self.session.last_target, self.holding_command)):
            tracker.bias = self.holding_bias.copy()
            tracker.output = self.holding_command[:5].astype(float).copy()
            command = self.holding_command.copy()
        self.holding_bias = self.holding_command = None
        point = plan.points[0].copy()
        period = 1 / self.cfg["control_hz"]
        phase = elapsed = next_diagnostic = 0.
        end_began = stable_began = None
        best_distance = float(np.linalg.norm(plan.points[-1]-point))
        last_improvement = 0.
        reason, settled = "tracking_timeout", False
        # Allows a slower arm to take longer; limits a forever-running command.
        budget = max(60., plan.times[-1]*8 + self.cfg["settle_seconds"])
        direction = plan.requested_delta / np.linalg.norm(plan.requested_delta)
        final_reference = frames[-1][1].copy()
        stable_anchor = actual.copy()
        frozen_final = False
        try:
            if cancel():
                raise MotionStopped("已停止，未使能。")
            self.session.arm(actual)  # always all six, goals staged before enable
            previous_tick = self.clock()
            while elapsed < budget:
                self._wait_until(previous_tick + period, cancel)
                now = self.clock()
                dt = max(period, now-previous_tick) if self.realtime else period
                previous_tick = now
                elapsed += dt
                # A delayed serial cycle never causes a burst of catch-up goals
                # or a large integral jump. Physical elapsed time still counts.
                control_dt = min(dt, 2*period)
                actual = self.session.read_positions()
                self.feedback.append(actual.copy())
                self.mapping.check_start(actual)
                measured_point = self.mapping.arm.fk(self.mapping.to_q(actual))[:3, 3]
                motor_error = np.abs(command-actual)
                if np.max(motor_error) > self.cfg["max_tracking_error_counts"]:
                    i = int(np.argmax(motor_error))
                    raise RuntimeError(f"ID{i+1} 严重落后：下发 {command[i]} / 反馈 {actual[i]}；停止推进并保持。")

                scale = pace(self.cfg, reference, actual, np.linalg.norm(point-measured_point)*1000)
                candidate_phase = min(float(plan.times[-1]), phase + control_dt * scale)
                candidate_point, nominal_q, mode = self._sample(plan, candidate_phase)
                if not frozen_final:
                    candidate = self._reference(candidate_point, actual, nominal_q)
                    if candidate is None:
                        # Do not advance progress or drop accumulated drive error
                        # just because a local online IK solve failed this cycle.
                        scale = 0.
                    else:
                        candidate[4:] = initial[4:]
                        phase, point, reference = candidate_phase, candidate_point, candidate
                        if phase >= plan.times[-1]-1e-9:
                            frozen_final = True
                            final_reference = reference.copy()
                            end_began = elapsed
                # Feedback PI drives a nominal joint reference, not the next
                # nominal waypoint. The bias cannot become unbounded or escape
                # calibrated/URDF/EEPROM limits; no firmware gain writes occur.
                raw = np.r_[tracker.command(reference, actual, control_dt), initial[5]].astype(int)
                if not np.array_equal(raw, command):
                    if cancel():
                        raise MotionStopped("用户停止运动。")
                    self.session.write_raw(raw)
                    command = raw.copy()
                    self.sent.append(command.copy())

                relative = measured_point - plan.points[0]
                along = float(relative @ direction)
                lateral = float(np.linalg.norm(relative - along*direction))
                distance = float(np.linalg.norm(plan.points[-1]-measured_point))
                if distance < best_distance - .0015:
                    best_distance, last_improvement = distance, elapsed
                if mode == "posture" and scale > .2 and np.max(np.abs(reference-actual)) <= 5:
                    last_improvement = elapsed  # planned nullspace motion, not a stuck TCP
                reference_error = int(np.max(np.abs(reference-actual)))
                self.trace.append({"elapsed_s": elapsed, "phase_s": phase,
                                   "reference_fraction": float(phase/plan.times[-1]),
                                   "actual_along_mm": along*1000, "lateral_mm": lateral*1000,
                                   "pace_scale": scale, "reference_error_counts": reference_error,
                                   "reference_raw": reference.tolist(), "command_raw": command.tolist(),
                                   "bias_counts": tracker.bias.tolist()})
                if elapsed >= next_diagnostic:
                    self.session.verify_target()
                    self.session.check_health()
                    next_diagnostic = elapsed + .5
                    progress = np.clip(100*along/np.linalg.norm(plan.requested_delta), 0, 100)
                    emit(f"实测沿向 {along*1000:.1f}/{np.linalg.norm(plan.requested_delta)*1000:.1f} mm "
                         f"（{progress:.0f}%，模型估计） 横偏 {lateral*1000:.1f} mm | "
                         f"推进倍率 {scale:.2f}，参考差 {reference_error} 计数，"
                         f"补偿 {np.round(tracker.bias, 1).tolist()}")

                arrived = (frozen_final and reference_error <= self.cfg["final_count_tolerance"]
                           and distance*1000 <= self.cfg["endpoint_tolerance_mm"])
                if not arrived:
                    stable_began = None
                elif stable_began is None or np.max(np.abs(actual-stable_anchor)) > 2:
                    stable_began, stable_anchor = elapsed, actual.copy()
                if stable_began is not None and elapsed-stable_began >= self.cfg["arrival_dwell_s"]:
                    reason, settled = "finished", True
                    break
                if end_began is not None and elapsed-end_began >= self.cfg["settle_seconds"]:
                    reason = "target_not_reached"
                    break
                if (not arrived and elapsed-last_improvement >= self.cfg["no_progress_timeout_s"]):
                    reason = "tracking_stalled"
                    emit(f"连续 {self.cfg['no_progress_timeout_s']:g} 秒没有可确认的末端前进；"
                         "停止后续运动并保持当前读数，不会把剩余轨迹继续发完。")
                    break
            self.session.verify_target()
            self.session.check_health()
            report = self._report(plan, actual, final_reference, command, reason, settled)
            if reason in ("tracking_stalled", "tracking_timeout"):
                self.session.hold()
                report["stopped_and_held"] = True
                report["hold_raw"] = np.asarray(self.session.last_target).tolist()
            else:
                self.holding_bias, self.holding_command = tracker.bias.copy(), command.copy()
            return report
        except (Exception, KeyboardInterrupt) as exc:
            try:
                self.session.hold()
            except Exception as hold_error:
                emit(f"停止保持未确认：{hold_error}。请托稳机械臂并处理供电。")
            try:
                report = self._report(plan, self.session.read_positions(), final_reference, command, "stopped", False)
                report["message"] = str(exc)
            except Exception:
                pass
            raise

    def _report(self, plan, feedback, final_reference, command, reason, settled):
        arm = self.mapping.arm
        start = plan.points[0]
        direction = plan.requested_delta / np.linalg.norm(plan.requested_delta)
        positions = np.array([arm.fk(self.mapping.to_q(raw))[:3, 3] for raw in self.feedback])
        actual_delta = arm.fk(self.mapping.to_q(feedback))[:3, 3] - start
        relative = positions-start
        lateral = np.linalg.norm(relative - np.outer(relative @ direction, direction), axis=1)
        sent_positions = np.array([arm.fk(self.mapping.to_q(raw))[:3, 3] for raw in self.sent])
        sent_relative = sent_positions-start
        sent_lateral = np.linalg.norm(sent_relative - np.outer(sent_relative @ direction, direction), axis=1)
        endpoint_error = float(np.linalg.norm(actual_delta - plan.requested_delta)*1000)
        success = bool(settled and not plan.blocked and "projected" not in plan.modes
                       and endpoint_error <= self.cfg["endpoint_tolerance_mm"]
                       and np.max(lateral)*1000 <= self.cfg["line_tolerance_mm"])
        if reason == "finished" and not success:
            reason = "reachable_part_finished" if plan.blocked or "projected" in plan.modes else "path_tolerance_exceeded"
        report = {"simulated": self.session.simulated, "solver": "numpy_bounded_damped_least_squares",
                  "execution": "feedback_paced_branch_preserving_ik_v3",
                  "driver": "TA scs215 FeetechMotorsBus", "success_model_estimate": success,
                  "physical_distance_verified": False, "reason": reason,
                  "requested_mm": (plan.requested_delta*1000).tolist(),
                  "model_feedback_delta_mm": (actual_delta*1000).tolist(),
                  "endpoint_error_mm": endpoint_error, "max_lateral_feedback_mm": float(np.max(lateral)*1000),
                  "max_lateral_command_counts_mm": float(np.max(sent_lateral)*1000),
                  "final_count_error": int(np.max(np.abs(feedback-final_reference))),
                  "final_motor_command_error": int(np.max(np.abs(feedback-command))),
                  "initial_raw": plan.start_raw.tolist(), "target_raw": final_reference.tolist(),
                  "last_command_raw": command.tolist(),
                  "final_raw": np.asarray(feedback).tolist(), "planned_duration_s": float(plan.times[-1]),
                  "elapsed_s": self.trace[-1]["elapsed_s"] if self.trace else 0.,
                  "reference_fraction": self.trace[-1]["reference_fraction"] if self.trace else 0.,
                  "contains_projection": "projected" in plan.modes, "ik_blocked": plan.blocked,
                  "feedback_raw": [r.tolist() for r in self.feedback], "tracking_trace": self.trace,
                  "sent_raw": [r.tolist() for r in self.sent],
                  "degrees_per_count": self.mapping.degrees_per_count}
        self.last_report = report
        return report


