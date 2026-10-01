"""Offline regression checks. Never opens a real serial port."""
import unittest

import numpy as np

from control_ik import displacement
from robot_adapter import Calibration, NAMES, Robot, Simulation, load_config, make_bus
from trajectory import Controller, MotionStopped, Planner
from tracking import JointTracker


START = np.array([381, 455, 764, 837, 675, 104])


class FakeBus:
    def __init__(self, raw=START):
        self.is_connected = False
        self.writes = []
        self.reg = {name: {"Present_Position": int(raw[i]), "Goal_Position": int(raw[i]),
                           "Min_Position_Limit": 0, "Max_Position_Limit": 1023,
                           "Torque_Enable": 0, "Status": 0,
                           "Running_Time": 0, "Goal_Velocity": 100}
                    for i, name in enumerate(NAMES)}

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        if disable_torque:
            raise AssertionError("Unexpected automatic unloading")
        self.is_connected = False

    def read(self, register, name, **_):
        return self.reg[name][register]

    def sync_write(self, register, values, **kwargs):
        self.writes.append((register, values.copy(), kwargs.copy()))
        for name, value in values.items():
            self.reg[name][register] = value


class LagPlant(Simulation):
    """Independent plant: speed-limited first-order motion + gravity residual.

    The plant moves on reads, not on writes. Repeated/unchanged goals still drive
    it, just like a real position servo. load[] is a steady position offset from
    a constant motor goal; it is not known to the controller.
    """
    def __init__(self, mapping, initial=START, rates=None, load=None, stuck_id=None):
        super().__init__(mapping, initial)
        self.position = self.raw.astype(float)
        self.rates = np.asarray(rates if rates is not None else [50, 25, 15, 30, 50, 50])
        self.load = np.asarray(load if load is not None else np.zeros(6))
        self.stuck_id = stuck_id

    def read_positions(self):
        if self.owned:
            error = self.last_target + self.load - self.position
            delta = np.clip(error * .25, -self.rates/self.cfg["control_hz"], self.rates/self.cfg["control_hz"])
            if self.stuck_id is not None:
                delta[self.stuck_id-1] = 0
            self.position += delta
            self.raw = np.rint(self.position).astype(int)
        return self.raw.copy()

    def write_raw(self, raw):
        self.last_target = np.asarray(raw, dtype=int).copy()
        self.sent.append(self.last_target.copy())


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = Calibration(load_config())

    def test_latest_calibration_and_scale(self):
        m = self.mapping
        np.testing.assert_array_equal(m.zero, [383, 683, 641, 649, 688, 112])
        np.testing.assert_array_equal(m.minimum, [57, 317, 319, 321, 400, 97])
        np.testing.assert_array_equal(m.maximum, [773, 995, 903, 957, 976, 486])
        self.assertEqual(m.degrees_per_count, 330/1024)
        np.testing.assert_array_equal(m.to_q(m.zero), np.zeros(5))

    def test_scale_override_does_not_rewrite_source(self):
        cfg = load_config()
        before = self.mapping.source.read_bytes()
        cfg["degrees_per_count_override"] = 320/1024
        m = Calibration(cfg)
        self.assertEqual(m.degrees_per_count, 320/1024)
        self.assertEqual(before, self.mapping.source.read_bytes())

    def test_fk_jacobian_matches_finite_differences(self):
        m = self.mapping
        q = m.to_q(START)
        _, jacobian = m.arm.fk_jacobian(q)
        for i in range(5):
            dq = np.eye(5)[i] * 1e-6
            numerical = (m.arm.fk(q+dq)[:3, 3] - m.arm.fk(q-dq)[:3, 3]) / 2e-6
            np.testing.assert_allclose(jacobian[:, i], numerical, atol=1e-8)

    def test_raw_roundtrip_and_limits(self):
        m = self.mapping
        for raw in (m.raw_lower, m.raw_upper, START[:5]):
            np.testing.assert_array_equal(m.to_raw(m.to_q(raw)), raw)
        bad = START.copy(); bad[4] = 200
        with self.assertRaises(ValueError):
            m.check_start(bad)

    def test_ta_driver_loaded_and_scs215_serialization(self):
        bus, source = make_bus(self.mapping, "OFFLINE")
        self.assertFalse(bus.is_connected)
        self.assertEqual(source["commit"], "bd2ee0bd961904090873d208fe79dc0ab3a4f863")
        self.assertEqual(bus.model_number_table["scs215"], 1315)
        self.assertFalse(bus.supports_homing_offset)
        self.assertFalse(bus.supports_sync_read)
        self.assertEqual(bus._serialize_data(0x0123, 2), [1, 35])
        for key in ("I_Coefficient", "Return_Delay_Time", "Acceleration"):
            self.assertNotIn(key, bus.model_ctrl_table["scs215"])

    def test_ta_raw_sync_write_preserves_integer_targets(self):
        bus, _ = make_bus(self.mapping, "OFFLINE")
        # Emulate an open port flag; the packet writer is replaced below, so no
        # serial port is opened and no bytes can reach hardware.
        bus.port_handler.is_open = True
        captured = []
        bus._sync_write = lambda *args, **kw: captured.append((args, kw))
        bus.sync_write("Goal_Position", dict(zip(NAMES, map(int, START))), normalize=False)
        args, _ = captured[0]
        self.assertEqual(args[:2], (42, 2))
        self.assertEqual(args[2], dict(enumerate(map(int, START), 1)))

    def test_connect_read_close_never_write(self):
        bus = FakeBus()
        robot = Robot(self.mapping, bus)
        np.testing.assert_array_equal(robot.read_positions(), START)
        robot.close()
        self.assertEqual(bus.writes, [])

    def test_current_goal_written_before_torque_and_no_reserved_writes(self):
        bus = FakeBus()
        robot = Robot(self.mapping, bus)
        robot.arm(START)
        regs = [r for r, _, _ in bus.writes]
        self.assertLess(regs.index("Goal_Position"), regs.index("Torque_Enable"))
        self.assertEqual(regs, ["Goal_Position", "Running_Time", "Goal_Velocity", "Torque_Enable"])
        robot.close()
        self.assertTrue(all(bus.reg[n]["Torque_Enable"] == 1 for n in NAMES))

    def test_goal_readback_mismatch_detected(self):
        bus = FakeBus()
        robot = Robot(self.mapping, bus)
        robot.write_raw(START)
        bus.reg[NAMES[2]]["Goal_Position"] += 1
        with self.assertRaisesRegex(RuntimeError, "目标寄存器"):
            robot.verify_target()

    def test_restart_adopts_stable_enabled_goal_without_dropping_load_bias(self):
        bus = FakeBus()
        for name in NAMES:
            bus.reg[name]["Torque_Enable"] = 1
        bus.reg[NAMES[2]]["Goal_Position"] -= 18
        robot = Robot(self.mapping, bus)
        target = robot.adopt_stable_hold(START)
        self.assertEqual(target[2], START[2]-18)
        robot.arm(START)
        self.assertEqual(bus.writes, [])
        self.assertTrue(robot.owned)
        robot.close()

    def test_disabled_or_distant_old_goals_are_not_adopted(self):
        bus = FakeBus()
        robot = Robot(self.mapping, bus)
        self.assertIsNone(robot.adopt_stable_hold(START))
        for name in NAMES:
            bus.reg[name]["Torque_Enable"] = 1
        bus.reg[NAMES[2]]["Goal_Position"] -= 90
        self.assertIsNone(robot.adopt_stable_hold(START))
        self.assertFalse(robot.owned)
        self.assertEqual(bus.writes, [])

    def test_near_elbow_limit_can_retreat_on_line_without_false_projection(self):
        raw = [391, 531, 899, 459, 678, 104]
        p = Planner(self.mapping).build(raw, [-.05, 0, 0], 1)
        self.assertFalse(p.blocked)
        self.assertNotIn("projected", p.modes)
        np.testing.assert_allclose(p.points[-1]-p.points[0], [-.05, 0, 0], atol=.00005)
        self.assertLess(np.max(np.abs(np.diff(p.angles, axis=0))), np.deg2rad(6.01))
        np.testing.assert_allclose(p.angles[:, 4], self.mapping.to_q(raw)[4], atol=1e-12)

    def test_old_hardware_limits_detected_without_changes(self):
        bus = FakeBus()
        bus.reg[NAMES[0]]["Min_Position_Limit"] = 836
        robot = Robot(self.mapping, bus)
        with self.assertRaisesRegex(RuntimeError, "硬件限位"):
            robot.preflight(np.array([START]))
        self.assertEqual(bus.writes, [])

    def test_explicit_off_applies_to_six(self):
        bus = FakeBus()
        robot = Robot(self.mapping, bus)
        robot.arm(START)
        robot.off()
        self.assertFalse(robot.owned)
        self.assertTrue(all(bus.reg[n]["Torque_Enable"] == 0 for n in NAMES))

    def test_all_six_directions_five_cm(self):
        for direction in np.r_[np.eye(3), -np.eye(3)]:
            with self.subTest(direction=direction):
                sim = Simulation(self.mapping, START)
                report = Controller(sim, realtime=False).move(direction*.05, emit=lambda _: None)
                self.assertTrue(report["success_model_estimate"], report)
                self.assertLess(report["endpoint_error_mm"], 2)
                self.assertLess(report["max_lateral_feedback_mm"], 2)
                self.assertTrue(all(raw[5] == START[5] for raw in sim.sent))

    def test_five_count_deadband_still_gets_entire_trajectory(self):
        sim = Simulation(self.mapping, START, deadband=5)
        report = Controller(sim, realtime=False).move([0, 0, .05], emit=lambda _: None)
        self.assertGreater(len(sim.sent), 50)
        self.assertLessEqual(report["final_count_error"], 5)
        self.assertGreater(report["model_feedback_delta_mm"][2], 40)
        self.assertEqual(report["reason"], "finished")

    def test_stuck_motor_is_not_reported_as_success_or_unloaded(self):
        sim = Simulation(self.mapping, START, stuck=True)
        controller = Controller(sim, realtime=False)
        report = controller.move([0, 0, .05], emit=lambda _: None)
        self.assertFalse(report["success_model_estimate"])
        self.assertEqual(report["reason"], "tracking_stalled")
        self.assertLess(report["reference_fraction"], .5)
        self.assertEqual(max(t["actual_along_mm"] for t in report["tracking_trace"]), 0.)
        self.assertTrue(sim.owned)
        np.testing.assert_array_equal(sim.last_target, START)
        self.assertTrue(report["stopped_and_held"])

    def test_slow_plant_slows_reference_and_finishes_without_waypoint_gates(self):
        sim = LagPlant(self.mapping, rates=[3, 3, 3, 3, 3, 3])
        controller = Controller(sim, realtime=False)
        controller.speed_cm_s = 3
        report = controller.move([0, 0, .05], emit=lambda _: None)
        self.assertTrue(report["success_model_estimate"], str({k:v for k,v in report.items() if not isinstance(v, list)}))
        self.assertGreater(report["elapsed_s"], 2 * report["planned_duration_s"])
        self.assertTrue(any(t["pace_scale"] < .5 for t in report["tracking_trace"]))
        self.assertLess(report["max_lateral_feedback_mm"], 10)

    def test_gravity_residual_gets_bounded_correction_at_user_logged_start(self):
        sim = LagPlant(self.mapping, initial=[385, 402, 818, 718, 678, 104], load=[0, 1, 18, 7, 0, 0])
        report = Controller(sim, realtime=False).move([0, 0, .05], emit=lambda _: None)
        self.assertTrue(report["success_model_estimate"], str({k:v for k,v in report.items() if not isinstance(v, list)}))
        self.assertLessEqual(report["final_count_error"], 5)
        self.assertGreater(report["final_motor_command_error"], 10)  # command includes load compensation
        self.assertLessEqual(max(np.max(np.abs(t["bias_counts"])) for t in report["tracking_trace"]), 30)
        self.assertTrue(all(raw[5] == 104 for raw in sim.sent))
        for raw in sim.sent:
            self.mapping.check_start(raw)

    def test_single_stuck_joint_cannot_silently_send_all_goals(self):
        sim = LagPlant(self.mapping, stuck_id=3)
        report = Controller(sim, realtime=False).move([0, 0, .05], emit=lambda _: None)
        # Position-only IK may redistribute among healthy joints. If it can do
        # that accurately it is valid; otherwise it must stop, not declare success.
        if report["success_model_estimate"]:
            self.assertLessEqual(report["endpoint_error_mm"], 8)
            self.assertLessEqual(report["max_lateral_feedback_mm"], 10)
        else:
            self.assertIn(report["reason"], ("tracking_stalled", "tracking_timeout", "target_not_reached"))
        self.assertTrue(sim.owned)

    def test_compensation_saturates_at_effective_limits_and_unwinds(self):
        tracker = JointTracker(self.mapping, START)
        reference = self.mapping.raw_upper.copy()
        tracker.output = reference.astype(float)
        for _ in range(100):
            command = tracker.command(reference, reference-20, .05)
        np.testing.assert_array_equal(command, reference)
        np.testing.assert_array_equal(tracker.bias, np.zeros(5))
        tracker.bias[:] = 15
        for _ in range(100):
            tracker.command(reference-50, reference-20, .05)
        self.assertTrue(np.all(tracker.bias < 0))

    def test_firmware_parameters_never_written_by_feedback_loop(self):
        bus = FakeBus()
        robot = Robot(self.mapping, bus)
        robot.arm(START)
        tracker = JointTracker(self.mapping, START, robot.hardware_limits)
        for _ in range(30):
            raw = np.r_[tracker.command(START[:5]+4, START, .05), START[5]]
            robot.write_raw(raw)
        self.assertEqual(set(r for r,_,_ in bus.writes),
                         {"Goal_Position", "Running_Time", "Goal_Velocity", "Torque_Enable"})
        self.assertTrue(all(bus.reg[n]["Torque_Enable"] == 1 for n in NAMES))

    def test_unreachable_command_reports_and_makes_positive_progress(self):
        sim = Simulation(self.mapping, START)
        messages = []
        report = Controller(sim, realtime=False).move([.4, 0, 0], emit=messages.append)
        self.assertFalse(report["success_model_estimate"])
        self.assertTrue(any("IK 无解" in s for s in messages))
        self.assertGreater(report["model_feedback_delta_mm"][0], 0)
        self.assertTrue(report["contains_projection"] or report["ik_blocked"])
        for raw in sim.sent:
            self.mapping.check_start(raw)

    def test_user_stop_cancels_remaining_targets_and_keeps_torque(self):
        sim = Simulation(self.mapping, START)
        controller = Controller(sim, realtime=False)
        def cancel():
            return len(sim.sent) >= 8
        with self.assertRaises(MotionStopped):
            controller.move([0, 0, .05], cancel, emit=lambda _: None)
        self.assertEqual(len(sim.sent), 8)
        np.testing.assert_array_equal(sim.last_target, sim.raw)
        self.assertTrue(sim.owned)

    def test_projection_never_counts_as_original_line_success(self):
        plan = Planner(self.mapping).build(START, [0, 0, .05], 1)
        plan.modes[1] = "projected"
        sim = Simulation(self.mapping, START)
        controller = Controller(sim, realtime=False)
        controller.planner.build = lambda *a: plan
        report = controller.move([0, 0, .05], emit=lambda _: None)
        self.assertFalse(report["success_model_estimate"])

    def test_invalid_commands_never_make_motion(self):
        for tokens in (["up", "nan"], ["up", "-5"], ["move", "inf", "0", "0"], ["speed", "1", "up", "5"]):
            with self.assertRaises(ValueError):
                displacement(tokens)
        np.testing.assert_array_equal(displacement(["up", "5"]), [0, 0, .05])

    def test_planning_quantization_preserves_current_start(self):
        planner = Planner(self.mapping)
        plan = planner.build(START, [0, 0, .05], 1)
        frames = planner.frames(plan)
        np.testing.assert_array_equal(frames[0][1], START)
        self.assertAlmostEqual(frames[-1][0], .05/.01, delta=.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
