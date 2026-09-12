import math
import unittest

import numpy as np

from scripts.visual_inertial_pose import (
    IMUSample,
    preintegrate_imu_prior,
    rotation_disagreement_deg,
    visual_inertial_rotation_consistent,
)


class VisualInertialPoseTests(unittest.TestCase):
    def test_rest_specific_force_cancels_gravity(self):
        samples = [
            IMUSample(0.0, (0, 0, 0), (0, 0, 9.80665)),
            IMUSample(1.0, (0, 0, 0), (0, 0, 9.80665)),
        ]
        prior = preintegrate_imu_prior(samples)
        self.assertTrue(np.allclose(prior.position_delta_m, (0, 0, 0), atol=1e-6))
        self.assertTrue(np.allclose(prior.velocity_delta_m_s, (0, 0, 0), atol=1e-6))
        self.assertEqual(prior.status, "candidate")

    def test_constant_yaw_integrates_orientation(self):
        rate = math.pi / 2
        samples = [
            IMUSample(0.0, (0, 0, rate), (0, 0, 9.80665)),
            IMUSample(1.0, (0, 0, rate), (0, 0, 9.80665)),
        ]
        prior = preintegrate_imu_prior(samples)
        R = np.asarray(prior.rotation_delta).reshape(3, 3)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        self.assertTrue(np.allclose(R, expected, atol=1e-6))

    def test_constant_forward_specific_force_integrates_metric_delta(self):
        samples = [
            IMUSample(0.0, (0, 0, 0), (1, 0, 9.80665)),
            IMUSample(1.0, (0, 0, 0), (1, 0, 9.80665)),
        ]
        prior = preintegrate_imu_prior(samples)
        self.assertTrue(np.allclose(prior.velocity_delta_m_s, (1, 0, 0), atol=1e-6))
        self.assertTrue(np.allclose(prior.position_delta_m, (0.5, 0, 0), atol=1e-6))

    def test_visual_rotation_agreement_is_gate_not_promotion(self):
        imu = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        visual = imu.copy()
        self.assertAlmostEqual(rotation_disagreement_deg(imu, visual), 0.0, places=6)
        self.assertTrue(
            visual_inertial_rotation_consistent(imu, visual, max_error_deg=2.0)
        )
        self.assertFalse(
            visual_inertial_rotation_consistent(imu, np.eye(3), max_error_deg=10.0)
        )


if __name__ == "__main__":
    unittest.main()
