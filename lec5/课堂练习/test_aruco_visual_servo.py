import unittest

import cv2
import numpy as np

from aruco_visual_servo_core import (ArucoTracker, MarkerObservation,
    camera_delta_to_base, choose_step)


class VisualServoCoreTests(unittest.TestCase):
    def observation(self, x, y, z=.4):
        return MarkerObservation(0, np.zeros((4, 2)), np.array([x, y]),
                                 np.zeros(3), np.array([0., 0., z]), 0.)

    def test_zone_decisions(self):
        self.assertEqual(choose_step(self.observation(100, 360), (1280, 720)).action, "left")
        self.assertEqual(choose_step(self.observation(1180, 360), (1280, 720)).action, "right")
        self.assertEqual(choose_step(self.observation(640, 50), (1280, 720)).action, "up")
        self.assertEqual(choose_step(self.observation(640, 680), (1280, 720)).action, "down")
        self.assertEqual(choose_step(self.observation(640, 360, .4), (1280, 720)).action, "forward")
        self.assertTrue(choose_step(self.observation(640, 360, .2), (1280, 720)).done)

    def test_camera_delta_rotation(self):
        T_BE = np.eye(4); T_EC = np.eye(4)
        T_BE[:3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(camera_delta_to_base(T_BE, T_EC, [1., 0., 0.]), [0., 1., 0.])

    def test_generated_marker_detects_as_id_zero(self):
        K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
        tracker = ArucoTracker(K, np.zeros(5))
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 240)
        frame = np.full((480, 640, 3), 255, np.uint8)
        frame[120:360, 200:440] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        found = tracker.detect(frame)
        self.assertIsNotNone(found)
        self.assertEqual(found.marker_id, 0)
        np.testing.assert_allclose(found.center_px, [319.5, 239.5], atol=1.)


if __name__ == "__main__":
    unittest.main()
