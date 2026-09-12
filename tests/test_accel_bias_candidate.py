import unittest

import numpy as np

from scripts.accel_bias_candidate import estimate_accel_bias_candidate
from scripts.vio_calibration_candidates import TimedVectorSample


class AccelerometerBiasCandidateTests(unittest.TestCase):
    def test_stationary_accel_bias_uses_explicit_gravity_and_orientation(self):
        gravity = np.array([0.0, 0.0, -9.80665])
        R_world_from_imu = np.eye(3)
        bias = np.array([0.10, -0.05, 0.20])
        expected_specific = -(R_world_from_imu.T @ gravity)
        measured = expected_specific + bias
        samples = [
            TimedVectorSample(0.0, tuple(measured + np.array([0.001, 0.0, 0.0]))),
            TimedVectorSample(0.1, tuple(measured + np.array([-0.001, 0.0, 0.0]))),
            TimedVectorSample(0.2, tuple(measured)),
            TimedVectorSample(0.3, tuple(measured)),
        ]
        candidate = estimate_accel_bias_candidate(
            samples,
            rotation_world_from_imu=R_world_from_imu,
            gravity_world_m_s2=gravity,
            min_samples=4,
            max_axis_std_m_s2=0.01,
        )
        self.assertTrue(np.allclose(candidate.bias_m_s2, bias, atol=1e-6))
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.online_bias_paid)
        self.assertTrue(candidate.gravity_reference_required)

    def test_noisy_stationary_accel_abstains(self):
        gravity = (0.0, 0.0, -9.80665)
        samples = [
            TimedVectorSample(0.0, (0.0, 0.0, 9.80665)),
            TimedVectorSample(0.1, (1.0, 0.0, 9.80665)),
            TimedVectorSample(0.2, (-1.0, 0.0, 9.80665)),
            TimedVectorSample(0.3, (0.5, 0.0, 9.80665)),
        ]
        candidate = estimate_accel_bias_candidate(
            samples,
            rotation_world_from_imu=np.eye(3),
            gravity_world_m_s2=gravity,
            min_samples=4,
            max_axis_std_m_s2=0.2,
        )
        self.assertEqual(candidate.status, "abstain")
        self.assertTrue(candidate.noisy)
        self.assertFalse(candidate.online_bias_paid)


if __name__ == "__main__":
    unittest.main()
