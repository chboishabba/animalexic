import math
import unittest

import numpy as np

from scripts.vio_calibration_candidates import (
    TimedVectorSample,
    estimate_camera_imu_rotation_candidate,
    estimate_gyro_bias_candidate,
)


class VIOCalibrationCandidateTests(unittest.TestCase):
    def test_stationary_gyro_bias_is_candidate_only(self):
        samples = [
            TimedVectorSample(0.0, (0.01, -0.02, 0.005)),
            TimedVectorSample(0.1, (0.011, -0.019, 0.004)),
            TimedVectorSample(0.2, (0.009, -0.021, 0.006)),
            TimedVectorSample(0.3, (0.010, -0.020, 0.005)),
        ]
        candidate = estimate_gyro_bias_candidate(
            samples,
            min_samples=4,
            max_axis_std_rad_s=0.002,
        )
        self.assertTrue(np.allclose(candidate.bias_rad_s, (0.01, -0.02, 0.005), atol=1e-6))
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.online_bias_paid)

    def test_noisy_stationary_bias_abstains(self):
        samples = [
            TimedVectorSample(0.0, (0.0, 0.0, 0.0)),
            TimedVectorSample(0.1, (0.2, 0.0, 0.0)),
            TimedVectorSample(0.2, (-0.2, 0.0, 0.0)),
            TimedVectorSample(0.3, (0.1, 0.0, 0.0)),
        ]
        candidate = estimate_gyro_bias_candidate(
            samples,
            min_samples=4,
            max_axis_std_rad_s=0.05,
        )
        self.assertEqual(candidate.status, "abstain")
        self.assertTrue(candidate.noisy)
        self.assertFalse(candidate.online_bias_paid)

    def test_camera_imu_rotation_recovers_known_rotation(self):
        angle = math.pi / 2
        expected = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0],
             [math.sin(angle), math.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
        imu = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=float,
        )
        camera = (expected @ imu.T).T
        candidate = estimate_camera_imu_rotation_candidate(
            imu,
            camera,
            max_rms_vector_residual=1e-6,
        )
        recovered = np.asarray(candidate.rotation_camera_from_imu).reshape(3, 3)
        self.assertTrue(np.allclose(recovered, expected, atol=1e-6))
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.extrinsic_calibration_paid)

    def test_collinear_rotation_calibration_fails_closed(self):
        imu = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        camera = imu.copy()
        with self.assertRaises(ValueError):
            estimate_camera_imu_rotation_candidate(
                imu,
                camera,
                max_rms_vector_residual=0.1,
            )


if __name__ == "__main__":
    unittest.main()
