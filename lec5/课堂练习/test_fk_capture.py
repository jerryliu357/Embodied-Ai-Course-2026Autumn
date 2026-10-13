"""Offline checks: no real camera or serial port is opened."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import fk_capture_core as core
from calibrate_intrinsics import read_image

DATA = json.loads((core.FK_DIR / "calibration.json").read_text(encoding="utf-8-sig"))
SESSION = core.HERE / "data" / "hand_eye_20261012_202343"
META = json.loads((SESSION / "session.json").read_text(encoding="utf-8-sig"))


class CaptureTests(unittest.TestCase):
    def controller(self):
        bus = core.DemoBus(DATA)
        c = core.CaptureController(bus, DATA)
        c.initialize(time.monotonic())
        return bus, c

    def test_read_only_start_and_one_joint_only(self):
        bus, c = self.controller()
        self.assertEqual(bus.writes, [])
        old_limits = [bus.read(i, 9, 4) for i in core.JOINTS]
        now = time.monotonic()
        ratio = c.snapshot()[1]["ratio"] + .02
        c.set_target(1, ratio, 60, now)
        for step in range(1, 50):
            stamp = now + step*.05
            bus.step(.05)
            c.poll(stamp)
            c.advance(60, stamp)
        self.assertTrue(bus.writes)
        self.assertTrue(all(w[0] == 1 and w[1] in (40, 42) for w in bus.writes))
        self.assertEqual(old_limits, [bus.read(i, 9, 4) for i in core.JOINTS])
        self.assertLessEqual(abs(c.joints[1].raw-core.target_position(ratio, DATA['joints']['1'])), 6)

    def test_hardware_limits_not_rewritten(self):
        bus, c = self.controller()
        c.joints[1].limits = (100, 700)
        with self.assertRaisesRegex(ValueError, "限位"):
            c.set_target(1, 1.0, 60, time.monotonic())
        self.assertEqual(bus.writes, [])

    def test_fk_matches_previous_samples_and_ignores_gripper(self):
        mapping = core.load_mapping(DATA)
        for path in (SESSION / "samples").glob("*.json"):
            sample = json.loads(path.read_text(encoding="utf-8"))
            raw = sample['raw_positions']
            T = mapping.arm.fk(mapping.to_q(raw))
            np.testing.assert_allclose(T, sample['T_base_end_effector'], atol=1e-12)
            raw[-1] = 400
            np.testing.assert_allclose(T, mapping.arm.fk(mapping.to_q(raw)))

    def test_stability_requires_full_second_and_rejects_drift(self):
        _, c = self.controller()
        snapshot = c.snapshot()
        gate = core.Stability()
        for t in (0, .2, .4, .6, .8):
            self.assertFalse(gate.add(t, snapshot))
        self.assertTrue(gate.add(1.01, snapshot))
        snapshot[3]['raw'] += 4
        self.assertFalse(gate.add(1.1, snapshot))
        snapshot[3]['active'] = True
        self.assertFalse(gate.add(1.2, snapshot))
        self.assertEqual(len(gate.history), 0)

    def test_bracket_freshness_motion_and_cancel(self):
        bus, c = self.controller()
        class FakeCamera:
            def sample(self):
                return time.monotonic(), np.zeros((720, 1280, 3), np.uint8)
        packet = core.bracket_frame(c, FakeCamera(), lambda: False)
        self.assertEqual(packet['raw'], packet['raw_before'])
        with self.assertRaisesRegex(ValueError, "取消"):
            core.bracket_frame(c, FakeCamera(), lambda: True)
        class StaleCamera:
            def sample(self):
                return 0, np.zeros((1, 1, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "新画面"):
            core.bracket_frame(c, StaleCamera(), lambda: False, timeout=.03)
        class MovingCamera(FakeCamera):
            def sample(self):
                value = int.from_bytes(bus.read(1, 56, 2), 'big') + 10
                bus.memory[1][56:58] = value.to_bytes(2, 'big')
                return super().sample()
        with self.assertRaisesRegex(ValueError, "移动"):
            core.bracket_frame(c, MovingCamera(), lambda: False)

    def test_saved_pair_duplicate_and_corner_reversal(self):
        mapping = core.load_mapping(DATA)
        with tempfile.TemporaryDirectory(prefix="fk_capture_test_") as temp:
            folder = Path(temp)
            for part in ('samples', 'images', 'corners'):
                (folder/part).mkdir()
            core.write_json(folder/'session.json', META)
            store = core.SampleStore(folder, META['intrinsics_file'], mapping)
            def packet(index):
                source = json.loads((SESSION/'samples'/f'sample_{index:02d}.json').read_text(encoding='utf-8'))
                return dict(raw=source['raw_positions'], raw_before=source['raw_positions'],
                            frame=read_image(SESSION/'images'/f'sample_{index:02d}.png'),
                            captured_at=source['captured_at'], frame_age=.01, bracket_seconds=.02)
            first = packet(4)
            saved = store.save(first)
            np.testing.assert_allclose(saved['T_base_end_effector'], mapping.arm.fk(mapping.to_q(first['raw'])))
            self.assertTrue((folder/'images'/'sample_01.png').exists())
            with self.assertRaisesRegex(ValueError, "太接近"):
                store.save(first)
            second = packet(5)
            normal = core.detect(second['frame'])
            with patch.object(core, 'detect', return_value=normal[::-1].copy()):
                saved = store.save(second)
            self.assertTrue(saved['corner_order_reversed'])
            self.assertLess(saved['relative_rotation_mismatch_deg'], 1)
            self.assertEqual(len(list((folder/'samples').glob('*.json'))), 2)
            with self.assertRaisesRegex(ValueError, '48'):
                blank = packet(3)
                blank['frame'][:] = 255
                store.save(blank)


if __name__ == '__main__':
    unittest.main(verbosity=2)
