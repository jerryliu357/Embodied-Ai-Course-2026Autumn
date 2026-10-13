"""Synthetic ground truth verifies AX=XB conventions and reported units."""
import unittest
import cv2
import numpy as np
from evaluate_hand_eye import evaluate, make_T, fit


class HandEyeTests(unittest.TestCase):
    def test_known_camera_transform_and_fixed_board(self):
        rng = np.random.default_rng(42)
        X = make_T(cv2.Rodrigues(np.array([.3, -.2, .4]))[0], [.04, -.03, .08])
        Y = make_T(cv2.Rodrigues(np.array([-.2, .1, .3]))[0], [.3, .1, .2])
        samples = []
        for i in range(15):
            A = make_T(cv2.Rodrigues(rng.normal(0, .35, 3))[0], rng.uniform(-.2, .2, 3))
            B = np.linalg.inv(X) @ np.linalg.inv(A) @ Y
            samples.append(dict(index=i+1, T_base_end_effector=A.tolist(), T_camera_board=B.tolist()))
        for method in ('PARK', 'HORAUD', 'DANIILIDIS'):
            np.testing.assert_allclose(fit(samples, method), X, atol=1e-9)
        result = evaluate(samples, 'PARK')
        np.testing.assert_allclose(result['T_base_board_mean'], Y, atol=1e-9)
        self.assertLess(result['leave_one_out']['position_mm']['max'], 1e-6)
        self.assertLess(result['leave_one_out']['orientation_deg']['max'], 1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
