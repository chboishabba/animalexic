import unittest

from scripts.rolling_shutter_pose_transport import (
    RowTimingObservation,
    estimate_readout_candidate,
)


class RollingShutterReadoutCandidateTests(unittest.TestCase):
    def test_top_to_bottom_readout_is_recovered_candidate_only(self):
        observations = [
            RowTimingObservation(0, 101, -0.0100, "track0"),
            RowTimingObservation(25, 101, -0.0050, "track1"),
            RowTimingObservation(50, 101, 0.0000, "track2"),
            RowTimingObservation(75, 101, 0.0050, "track3"),
            RowTimingObservation(100, 101, 0.0100, "track4"),
        ]
        candidate = estimate_readout_candidate(
            observations,
            min_observations=5,
            min_normalized_row_span=0.8,
            max_rms_timing_residual_s=1e-8,
        )
        self.assertAlmostEqual(candidate.readout_time_s, 0.020, places=9)
        self.assertEqual(candidate.direction, "top_to_bottom")
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.readout_calibration_paid)

    def test_bottom_to_top_direction_is_recovered(self):
        observations = [
            RowTimingObservation(0, 101, 0.0100, "a"),
            RowTimingObservation(50, 101, 0.0000, "b"),
            RowTimingObservation(100, 101, -0.0100, "c"),
        ]
        candidate = estimate_readout_candidate(
            observations,
            min_observations=3,
            min_normalized_row_span=0.9,
            max_rms_timing_residual_s=1e-8,
        )
        self.assertAlmostEqual(candidate.readout_time_s, 0.020, places=9)
        self.assertEqual(candidate.direction, "bottom_to_top")

    def test_insufficient_row_span_fails_closed(self):
        observations = [
            RowTimingObservation(49, 101, -0.0002, "a"),
            RowTimingObservation(50, 101, 0.0, "b"),
            RowTimingObservation(51, 101, 0.0002, "c"),
        ]
        with self.assertRaises(ValueError):
            estimate_readout_candidate(
                observations,
                min_observations=3,
                min_normalized_row_span=0.5,
                max_rms_timing_residual_s=0.001,
            )


if __name__ == "__main__":
    unittest.main()
