"""离线测试：不连接真实串口。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_store import ActionStore, CALIBRATION_PATH, default_actions
from arm_backend import ArmController, JOINTS, target_position, LeRobotSCSBus, calibration_fingerprint
from motion import DemoBus, Mailbox, MotionEngine, smoothstep


def data():
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    # 模拟姿态只用于测试，不能依赖用户是否已经完成实机姿态重录。
    calibration["poses"] = {"middle": [0.5] * 6}
    calibration["pose_recording"] = {"calibration_fingerprint": calibration_fingerprint(calibration)}
    return calibration


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.data = data()
        self.bus = DemoBus(self.data)
        self.c = ArmController(self.bus, self.data)
        self.c.initialize(0)
        self.events = []
        self.engine = MotionEngine(self.c, self.events.append)
        self.now = 0.0

    def tick(self, count=1):
        for _ in range(count):
            self.now += 0.05
            self.bus.step(0.05)
            self.c.poll(self.now)
            self.engine.tick(self.now, 100)

    def values(self):
        return [max(0, min(1, s.ratio)) for s in self.c.joints.values()]

    def step(self, values, seconds=2, hold=0, name="test"):
        return {"pose": name, "values": values, "seconds": seconds, "hold": hold}

    def test_start_is_read_only(self):
        self.tick(10)
        self.assertEqual(self.bus.writes, [])
        self.assertTrue(all(s.online for s in self.c.joints.values()))

    def test_single_slider_only_controls_selected_joint(self):
        self.c.set_target(2, 0.60, 100, self.now)
        self.tick(60)
        self.assertEqual({write[0] for write in self.bus.writes}, {2})
        self.assertAlmostEqual(self.c.joints[2].ratio, 0.60, delta=0.003)

    def test_invalid_late_step_does_not_move_any_joint(self):
        valid = self.step(self.values())
        invalid = self.step([0.5, 0.5, float("nan"), 0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            self.engine.play([valid, invalid], 100, 0)
        self.assertEqual(self.bus.writes, [])

    def test_offline_joint_rejects_entire_pose_before_writes(self):
        self.c.joints[5].online = False
        with self.assertRaises(ValueError):
            self.engine.play([self.step(self.values())], 100, 0)
        self.assertEqual(self.bus.writes, [])

    def test_time_interpolation_shared_progress_and_no_initial_reset(self):
        values = self.values()
        values[0] += 0.10
        values[1] += 0.05
        self.engine.play([self.step(values)], 100, 0)
        seg = copy.deepcopy(self.engine.segment)
        for sid, start in seg["starts"].items():
            first = next(w[2] for w in self.bus.writes if w[0] == sid and w[1] == 42)
            self.assertEqual(first, start % 1024)
        self.tick(21)
        alpha = smoothstep(self.engine.segment["elapsed"] / 2)
        for sid in (1, 2):
            expected = round(seg["starts"][sid] + alpha * (seg["ends"][sid] - seg["starts"][sid]))
            self.assertEqual(self.c.joints[sid].last_sent, expected)
        self.tick(40)
        self.assertFalse(self.engine.playing)
        for sid, value in zip(JOINTS, values):
            self.assertAlmostEqual(self.c.joints[sid].position,
                                   target_position(value, self.data["joints"][str(sid)]), delta=6)

    def test_short_duration_is_extended_by_speed_limit(self):
        values = self.values()
        values[0] += 0.2
        self.engine.play([self.step(values, seconds=0.1)], 100, 0)
        distance = abs(self.engine.segment["ends"][1] - self.engine.segment["starts"][1])
        self.assertAlmostEqual(self.engine.segment["duration"], 1.5 * distance / 100)
        self.assertIn("延长", self.events[-1])

    def test_sequence_waits_for_hold_and_ends_at_last_pose(self):
        a, b = self.values(), self.values()
        a[0] += 0.03
        b[0] -= 0.03
        self.engine.play([self.step(a, 0.6, hold=0.8, name="left"), self.step(b, 0.7, name="right")], 100, 0)
        self.tick(25)
        self.assertEqual(self.engine.index, 0)
        self.tick(55)
        self.assertFalse(self.engine.playing)
        self.assertEqual(len(self.events), 3)
        self.assertAlmostEqual(self.c.joints[1].ratio, b[0], delta=0.002)

    def test_pause_discards_remaining_steps_and_holds_current(self):
        values = self.values()
        values[1] += 0.2
        self.engine.play([self.step(values), self.step(self.values())], 100, 0)
        self.tick(12)
        positions = {sid: s.raw for sid, s in self.c.joints.items()}
        self.engine.stop()
        self.tick(100)
        self.assertFalse(self.engine.playing)
        self.assertEqual(positions, {sid: s.raw for sid, s in self.c.joints.items()})

    def test_stall_cancels_whole_sequence_and_unloads_failed_joint(self):
        self.bus.stuck.add(2)
        values = self.values()
        values[1] += 0.3
        self.engine.play([self.step(values), self.step(self.values())], 100, 0)
        self.tick(180)
        self.assertFalse(self.engine.playing)
        self.assertEqual(self.bus.memory[2][40], 0)
        self.assertIn("ID2", self.engine.status)
        self.assertTrue(all(not s.active for s in self.c.joints.values()))

    def test_reverse_feedback_stops_playback(self):
        values = self.values()
        values[0] += 0.1
        self.engine.play([self.step(values)], 100, 0)
        raw = self.c.joints[1].raw - 15
        self.bus.memory[1][56:58] = raw.to_bytes(2, "big")
        self.c.poll(0.1)
        self.engine.tick(0.1, 100)
        self.assertFalse(self.engine.playing)
        self.assertEqual(self.bus.memory[1][40], 0)

    def test_numeric_boundary_uses_recorded_unwrapped_path(self):
        joint = self.data["joints"]["3"]
        # 专门的跨界测试区间，不使用已改为不跨界的实机 ID3 范围。
        joint.update(zero_position=900, one_position=176, one_unwrapped_position=1200,
                     span_raw=300, min_position=None, max_position=None,
                     wraps_numeric_boundary=True, requires_boundary_resolution=True)
        self.bus.memory[3][56:58] = (1010).to_bytes(2, "big")
        self.c.initialize(0)
        values = self.values()
        values[2] = (1040 - joint["zero_position"]) / joint["span_raw"]
        self.engine.play([self.step(values, seconds=1)], 100, 0)
        self.tick(50)
        self.assertEqual(self.c.joints[3].position, 1040)
        raw_commands = [w[2] for w in self.bus.writes if w[0] == 3 and w[1] == 42]
        self.assertTrue(any(x >= 1020 for x in raw_commands))
        self.assertTrue(any(x < 20 for x in raw_commands))

    def test_emergency_clears_pending_commands(self):
        mailbox = Mailbox()
        mailbox.submit("target", (1, 0.7))
        mailbox.submit("play", [self.step(self.values())])
        mailbox.interrupt(off=True)
        mailbox.submit("target", (2, 0.8))
        self.assertEqual(mailbox.take()[:2], ([], {}))
        self.engine.play([self.step(self.values())], 100, 0)
        self.engine.stop(off=True)
        writes = len(self.bus.writes)
        self.tick(10)
        self.assertEqual(len(self.bus.writes), writes)
        self.assertTrue(all(self.bus.memory[sid][40] == 0 for sid in JOINTS))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.data = data()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "actions.json"
        self.store = ActionStore(self.data, self.path)

    def test_imported_poses_are_kept_and_new_poses_not_fabricated(self):
        self.assertEqual(self.store.document["poses"]["middle"]["values"], self.data["poses"]["middle"])
        self.assertIsNone(self.store.document["poses"]["stand"]["values"])
        with self.assertRaisesRegex(ValueError, "请先记录"):
            self.store.resolve("wave", is_action=True)

    def test_record_saves_actual_counts_and_roundtrips(self):
        raw = {sid: target_position(0.4, self.data["joints"][str(sid)]) % 1024 for sid in JOINTS}
        self.store.record("stand", raw)
        loaded = ActionStore(self.data, self.path)
        values = loaded.resolve("stand")[0]["values"]
        for sid, value in zip(JOINTS, values):
            self.assertEqual(target_position(value, self.data["joints"][str(sid)]) % 1024, raw[sid])
        self.assertEqual(loaded.document["poses"]["stand"]["raw_positions"], {str(k): v for k, v in raw.items()})

    def test_demo_does_not_write_files(self):
        store = ActionStore(self.data, self.path, memory_only=True)
        store.persist(default_actions(self.data))
        self.assertFalse(self.path.exists())

    def test_calibration_mismatch_and_external_changes_are_rejected(self):
        self.store.persist(self.store.document)
        changed = copy.deepcopy(self.data)
        changed["joints"]["1"]["zero_position"] += 1
        changed["joints"]["1"]["one_position"] += 1
        changed["joints"]["1"]["one_unwrapped_position"] += 1
        with self.assertRaisesRegex(ValueError, "不匹配"):
            ActionStore(changed, self.path)
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "其他程序"):
            self.store.persist(self.store.document)


class OfficialDriverTests(unittest.TestCase):
    def test_actual_lerobot_sdk_uses_big_endian_single_joint_packets(self):
        from lerobot.motors.feetech import FeetechMotorsBus
        with patch.object(FeetechMotorsBus, "connect"), patch.object(FeetechMotorsBus, "set_baudrate"):
            bus = LeRobotSCSBus("OFFLINE")
        bus.bus.port_handler.is_open = True
        writes, memory = [], {}
        def transmit(port, address, length, params, param_length):
            writes.append((address, length, list(params)))
            for offset in range(0, len(params), length + 1):
                memory[(params[offset], address)] = bytes(params[offset + 1:offset + 1 + length])
            return 0
        bus.bus.packet_handler.syncWriteTxOnly = transmit
        bus.bus.packet_handler.readTxRx = lambda port, sid, address, length: (list(memory[(sid, address)]), 0, 0)
        bus.write_position(3, 889, 100)
        self.assertEqual(writes[-1], (42, 2, [3, 3, 121]))


if __name__ == "__main__":
    unittest.main()
