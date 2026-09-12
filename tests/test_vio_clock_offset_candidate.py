import unittest

from scripts.vio_calibration_candidates import (
    TimedVectorSample,
    estimate_clock_offset_candidate,
)


class VIOClockOffsetCandidateTests(unittest.TestCase):
    def samples(self, times, values):
        return [TimedVectorSample(t, (v, v * v, 0.5 * v)) for t, v in zip(times, values)]

    def test_known_clock_shift_is_recovered_as_candidate_only(self):
        reference = self.samples([0, 1, 2, 3, 4], [0, 1, 4, 2, 5])
        sensor = self.samples([0.2, 1.2, 2.2, 3.2, 4.2], [0, 1, 4, 2, 5])
        candidate = estimate_clock_offset_candidate(
            reference,
            sensor,
            search_offsets_s=[-0.3, -0.2, -0.1, 0.0],
            min_overlap=4,
            min_residual_margin=0.05,
        )
        self.assertAlmostEqual(candidate.offset_s, -0.2, places=9)
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.clock_alignment_paid)
        self.assertLess(candidate.rms_residual, 1e-9)

    def test_ambiguous_constant_motion_abstains(self):
        reference = self.samples([0, 1, 2, 3], [1, 1, 1, 1])
        sensor = self.samples([0.2, 1.2, 2.2, 3.2], [1, 1, 1, 1])
        candidate = estimate_clock_offset_candidate(
            reference,
            sensor,
            search_offsets_s=[-0.2, 0.0],
            min_overlap=3,
            min_residual_margin=0.01,
        )
        self.assertEqual(candidate.status, "abstain")
        self.assertTrue(candidate.ambiguous)
        self.assertFalse(candidate.clock_alignment_paid)

    def test_insufficient_overlap_fails_closed(self):
        reference = self.samples([0, 1], [0, 1])
        sensor = self.samples([10, 11], [0, 1])
        with self.assertRaises(ValueError):
            estimate_clock_offset_candidate(
                reference,
                sensor,
                search_offsets_s=[0.0],
                min_overlap=2,
                min_residual_margin=0.0,
            )


if __name__ == "__main__":
    unittest.main()
