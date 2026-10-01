"""Offline regression checks. Never opens a real serial port."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control_arm_gui as gui


def calibration():
    return json.loads(gui.CALIBRATION_PATH.read_text(encoding="utf-8"))


class FakeBus:
    def __init__(self, data=None):
        data = data or calibration()
        self.memory = {}
        self.writes = []
        self.stuck = set()
        self.closed = False
        for sid in gui.JOINTS:
            mem = bytearray(80)
            mem[:9] = bytes([1, 0, 1, 5, 35, sid, 0, 0, 1])
            mem[9:13] = bytes([0, 0, 3, 255])
            mem[21:23] = bytes([15, 15])
            mem[48] = 1
            raw = gui.target_position(0.5, data["joints"][str(sid)]) % 1024
            mem[56:58] = raw.to_bytes(2, "big")
            mem[62:64] = bytes([51, 20])
            self.memory[sid] = mem

    def read(self, sid, address, length=1):
        return bytes(self.memory[sid][address:address + length])

    def write_checked(self, sid, address, value):
        self.writes.append((sid, address, value))
        self.memory[sid][address] = value

    def write_limits(self, sid):
        self.writes.append((sid, 9, (0, 1023)))
        self.memory[sid][9:13] = bytes([0, 0, 3, 255])

    def write_position(self, sid, raw, speed):
        self.writes.append((sid, 42, raw))
        self.memory[sid][42:48] = raw.to_bytes(2, "big") + b"\0\0" + speed.to_bytes(2, "big")

    def off_all(self):
        for sid in self.memory:
            self.write_checked(sid, 40, 0)

    def close(self):
        self.closed = True

    def position(self, sid, raw):
        self.memory[sid][56:58] = raw.to_bytes(2, "big")

    def step(self):
        # Simulate a servo supporting this application's unwrapped path.
        for sid, mem in self.memory.items():
            if mem[40] != 1 or sid in self.stuck:
                continue
            raw = int.from_bytes(mem[56:58], "big")
            goal = int.from_bytes(mem[42:44], "big")
            change = max(-5, min(5, gui.circular_delta(goal, raw)))
            self.position(sid, (raw + change) % 1024)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.data = calibration()
        self.bus = FakeBus(self.data)
        self.controller = gui.ArmController(self.bus, self.data)
        self.now = 0.0
        self.controller.initialize(self.now)

    def tick(self, count):
        for _ in range(count):
            self.now += 0.05
            self.bus.step()
            self.controller.poll(self.now)
            self.controller.advance(100, self.now)

    def test_initialization_and_close_are_read_only(self):
        self.assertTrue(all(s.online for s in self.controller.joints.values()))
        self.assertEqual(self.bus.writes, [])
        self.controller.hold(gui.JOINTS, 100)
        self.assertEqual(self.bus.writes, [])

    def test_only_selected_joint_receives_writes(self):
        self.controller.set_target(2, 0.65, 100, 0)
        self.tick(40)
        self.assertEqual({x[0] for x in self.bus.writes}, {2})
        state = self.controller.joints[2]
        self.assertFalse(state.active)
        self.assertLessEqual(abs(state.ratio - 0.65), 0.002)
        self.assertTrue(state.owned)

    def test_stall_of_one_joint_does_not_block_another(self):
        self.bus.stuck.add(3)
        self.controller.set_target(3, 0.8, 100, 0)
        self.controller.set_target(2, 0.65, 100, 0)
        self.tick(130)
        self.assertFalse(self.controller.joints[3].owned)
        self.assertEqual(self.bus.memory[3][40], 0)
        self.assertTrue(self.controller.joints[2].owned)
        self.assertLessEqual(abs(self.controller.joints[2].ratio - 0.65), 0.002)

    def test_bidirectional_numeric_wrap(self):
        for sid in (3, 5, 6):
            # 使用固定的跨界测试区间，不依赖用户是否已重新装配某个关节。
            data = calibration()
            joint = data["joints"][str(sid)]
            joint.update(zero_position=900, one_position=176,
                         one_unwrapped_position=1200, span_raw=300,
                         min_position=None, max_position=None,
                         wraps_numeric_boundary=True, requires_boundary_resolution=True)
            for start, end in ((1010, 1040), (1040, 1010)):
                bus = FakeBus(data)
                bus.position(sid, start % 1024)
                controller = gui.ArmController(bus, data)
                controller.initialize(0)
                controller.set_target(sid, (end - joint["zero_position"]) / joint["span_raw"], 100, 0)
                for n in range(1, 40):
                    bus.step()
                    controller.poll(n * 0.05)
                    controller.advance(100, n * 0.05)
                self.assertAlmostEqual(controller.joints[sid].position, end, delta=1)
                self.assertFalse(controller.joints[sid].active)

    def test_numeric_endpoints_are_not_excluded(self):
        joint = {"zero_position": 900, "span_raw": 300}
        for count in (1023, 1024):
            ratio = (count - joint["zero_position"]) / joint["span_raw"]
            self.assertEqual(gui.target_position(ratio, joint), count)

    def test_wrong_physical_direction_stops_only_that_joint(self):
        self.controller.set_target(3, 0.9, 100, 0)
        state = self.controller.joints[3]
        self.bus.position(3, (state.raw - 14) % 1024)
        self.controller.poll(0.1)
        self.controller.advance(100, 0.1)
        self.assertFalse(state.owned)
        self.assertEqual(self.bus.memory[3][40], 0)
        self.assertEqual({x[0] for x in self.bus.writes}, {3})

    def test_invalid_targets_do_not_write(self):
        for value in (-0.01, 1.01, float("nan"), float("inf"), True, "0.5"):
            with self.assertRaises(ValueError):
                self.controller.set_target(1, value, 100, 0)
        self.assertEqual(self.bus.writes, [])

    def test_out_of_range_start_does_not_enable(self):
        self.bus.position(1, 0)
        self.controller.poll(0.1)
        with self.assertRaises(ValueError):
            self.controller.set_target(1, 0.5, 100, 0.1)
        self.assertEqual(self.bus.writes, [])

    def test_old_limits_only_change_selected_joint(self):
        self.bus.memory[1][9:13] = bytes([3, 68, 3, 221])
        self.controller.initialize(0)
        self.controller.set_target(1, 0.6, 100, 0)
        self.assertEqual(self.bus.read(1, 9, 4), b"\0\0\x03\xff")
        self.assertEqual(self.bus.memory[1][48], 1)
        self.assertEqual({x[0] for x in self.bus.writes}, {1})

    def test_hold_cancels_pending_motion_without_returning_to_start(self):
        self.controller.set_target(2, 0.9, 100, 0)
        self.tick(5)
        raw = self.controller.joints[2].raw
        self.controller.hold([2], 100)
        self.tick(20)
        self.assertEqual(self.controller.joints[2].raw, raw)
        self.assertFalse(self.controller.joints[2].active)
        self.assertEqual(self.bus.memory[2][40], 1)

    def test_latest_target_and_emergency_discard_queue(self):
        mailbox = gui.Mailbox()
        for value in (0.1, 0.4, 0.8):
            mailbox.target(1, value)
        self.assertEqual(mailbox.take()[0], {1: 0.8})
        mailbox.target(1, 0.8)
        mailbox.stop_all()
        mailbox.target(2, 0.4)
        mailbox.action("hold", 3)
        self.assertEqual(mailbox.take()[:2], ({}, []))
        self.controller.set_target(1, 0.9, 100, 0)
        self.controller.off(gui.JOINTS)
        count = len(self.bus.writes)
        self.tick(20)
        self.assertEqual(len(self.bus.writes), count)


class OfficialDriverTests(unittest.TestCase):
    def test_busy_port_reports_original_connection_error(self):
        from lerobot.motors.feetech import FeetechMotorsBus
        with patch.object(FeetechMotorsBus, "connect", side_effect=ConnectionError("port busy")):
            with self.assertRaisesRegex(ConnectionError, "port busy"):
                gui.LeRobotSCSBus("OFFLINE")

    def test_lerobot_scs_big_endian_and_single_id_packets(self):
        from lerobot.motors.feetech import FeetechMotorsBus
        # Keep the actual LeRobot + SDK serializers, replace only serial I/O.
        with patch.object(FeetechMotorsBus, "connect"), patch.object(FeetechMotorsBus, "set_baudrate"):
            adapter = gui.LeRobotSCSBus("OFFLINE")
        adapter.bus.port_handler.is_open = True
        writes = []
        memory = {}

        def transmit(port, address, length, params, param_length):
            writes.append((address, length, list(params)))
            self.assertEqual(len(params), param_length)
            for offset in range(0, len(params), length + 1):
                sid = params[offset]
                memory[(sid, address)] = bytes(params[offset + 1:offset + 1 + length])
            return 0

        def receive(port, sid, address, length):
            return list(memory[(sid, address)]), 0, 0

        adapter.bus.packet_handler.syncWriteTxOnly = transmit
        adapter.bus.packet_handler.readTxRx = receive
        adapter.write_position(3, 889, 100)
        self.assertEqual(writes, [(44, 2, [3, 0, 0]), (46, 2, [3, 0, 100]),
                                  (42, 2, [3, 3, 121])])
        adapter.write_position(3, 0, 100)
        adapter.write_position(3, 1023, 100)
        self.assertEqual(writes[-2:], [(42, 2, [3, 0, 0]), (42, 2, [3, 3, 255])])
        adapter.write_checked(3, 40, 1)
        self.assertEqual(writes[-1], (40, 1, [3, 1]))
        adapter.off_all()
        self.assertEqual(writes[-1], (40, 1, [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]))


if __name__ == "__main__":
    unittest.main()
