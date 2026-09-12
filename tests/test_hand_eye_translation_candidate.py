import math
import unittest

import numpy as np

from scripts.vio_calibration_candidates import (
    HandEyeMotionPair,
    estimate_camera_imu_translation_candidate,
)


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


class HandEyeTranslationCandidateTests(unittest.TestCase):
    def test_known_lever_arm_is_recovered_candidate_only(self):
        R_x = rz(30.0)
        t_x = np.array([0.12, -0.04, 0.08])
        pairs = []
        motions = [
            (rz(20.0), np.array([0.3, 0.1, 0.0])),
            (rx(25.0), np.array([0.0, 0.2, 0.1])),
            (rz(-15.0) @ rx(10.0), np.array([0.1, -0.1, 0.2])),
            (rx(-20.0) @ rz(10.0), np.array([-0.2, 0.1, 0.1])),
        ]
        for R_b, t_b in motions:
            # AX = XB, with X = camera_from_imu.
            R_a = R_x @ R_b @ R_x.T
            t_a = R_x @ t_b + t_x - R_a @ t_x
            pairs.append(
                HandEyeMotionPair(
                    rotation_camera_motion=tuple(R_a.reshape(-1)),
                    translation_camera_motion_m=tuple(t_a),
                    rotation_imu_motion=tuple(R_b.reshape(-1)),
                    translation_imu_motion_m=tuple(t_b),
                )
            )
        candidate = estimate_camera_imu_translation_candidate(
            pairs,
            rotation_camera_from_imu=R_x,
            max_rms_translation_residual_m=1e-8,
        )
        self.assertTrue(np.allclose(candidate.translation_camera_from_imu_m, t_x, atol=1e-7))
        self.assertLess(candidate.rms_translation_residual_m, 1e-8)
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.lever_arm_paid)

    def test_rotation_poor_excitation_fails_closed(self):
        R_x = np.eye(3)
        pairs = []
        for distance in (0.1, 0.2, 0.3):
            R = np.eye(3)
            t = np.array([distance, 0.0, 0.0])
            pairs.append(
                HandEyeMotionPair(
                    rotation_camera_motion=tuple(R.reshape(-1)),
                    translation_camera_motion_m=tuple(t),
                    rotation_imu_motion=tuple(R.reshape(-1)),
                    translation_imu_motion_m=tuple(t),
                )
            )
        with self.assertRaises(ValueError):
            estimate_camera_imu_translation_candidate(
                pairs,
                rotation_camera_from_imu=R_x,
                max_rms_translation_residual_m=0.1,
            )


if __name__ == "__main__":
    unittest.main()
