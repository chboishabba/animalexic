import unittest

import numpy as np

from scripts.rolling_shutter_pose_transport import (
    RollingShutterReadoutCandidate,
    row_capture_time_offset_s,
    interpolate_row_pose,
)


class RollingShutterPoseTransportTests(unittest.TestCase):
    def test_row_offsets_are_centered_on_frame_time(self):
        model = RollingShutterReadoutCandidate(0.020, "top_to_bottom", "metadata", "candidate")
        self.assertAlmostEqual(row_capture_time_offset_s(0, 101, model), -0.010, places=9)
        self.assertAlmostEqual(row_capture_time_offset_s(50, 101, model), 0.0, places=9)
        self.assertAlmostEqual(row_capture_time_offset_s(100, 101, model), 0.010, places=9)

    def test_bottom_to_top_reverses_row_offsets(self):
        model = RollingShutterReadoutCandidate(0.020, "bottom_to_top", "metadata", "candidate")
        self.assertAlmostEqual(row_capture_time_offset_s(0, 101, model), 0.010, places=9)
        self.assertAlmostEqual(row_capture_time_offset_s(100, 101, model), -0.010, places=9)

    def test_row_pose_interpolates_translation_within_bracketing_keyframes(self):
        model = RollingShutterReadoutCandidate(0.020, "top_to_bottom", "metadata", "candidate")
        before = type("KF", (), {
            "time_s": 0.99,
            "position_world_m": (0.0, 0.0, 0.0),
            "rotation_world_from_camera": tuple(np.eye(3).reshape(-1)),
            "status": "candidate",
        })()
        after = type("KF", (), {
            "time_s": 1.01,
            "position_world_m": (2.0, 0.0, 0.0),
            "rotation_world_from_camera": tuple(np.eye(3).reshape(-1)),
            "status": "candidate",
        })()
        pose = interpolate_row_pose(
            frame_time_s=1.0,
            row=50,
            image_height=101,
            readout=model,
            before=before,
            after=after,
        )
        self.assertTrue(np.allclose(pose.position_world_m, (1.0, 0.0, 0.0), atol=1e-9))
        self.assertEqual(pose.status, "candidate")
        self.assertFalse(pose.readout_calibration_paid)

    def test_unpaid_or_abstained_readout_does_not_promote(self):
        model = RollingShutterReadoutCandidate(0.020, "top_to_bottom", "guess", "abstain")
        with self.assertRaises(ValueError):
            row_capture_time_offset_s(10, 100, model)


if __name__ == "__main__":
    unittest.main()
