import unittest
import numpy as np

from scripts.shared_world_guard_transport import (
    WorldRayObservation,
    adapt_world_rays_to_guard_frames,
    compare_guard_transport,
)


class KF:
    def __init__(self, t, p, status="candidate"):
        self.time_s = t
        self.position_world_m = p
        self.status = status


class SharedWorldGuardTransportTests(unittest.TestCase):
    def test_adapter_uses_world_camera_origin_per_observation(self):
        keyframes = {
            0: [KF(0.0, (1.0, 2.0, 3.0))],
            1: [KF(0.0, (-1.0, 0.0, 2.0))],
        }
        obs = [
            WorldRayObservation(0, 0.0, (2.0, 2.0, 5.0), 1.0, 0.2),
            WorldRayObservation(1, 0.0, (0.0, 0.0, 5.0), 0.5, 0.4),
        ]
        frames = adapt_world_rays_to_guard_frames(obs, keyframes)
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertTrue(np.allclose(frame.camera_origins, [[1, 2, 3], [-1, 0, 2]]))
        self.assertTrue(np.allclose(frame.points, [[2, 2, 5], [0, 0, 5]]))

    def test_missing_or_unpaid_keyframe_fails_closed(self):
        with self.assertRaises(ValueError):
            adapt_world_rays_to_guard_frames(
                [WorldRayObservation(0, 1.0, (0, 0, 1), 1.0, 0.1)],
                {0: [KF(1.0, (0, 0, 0), status="promoted")]},
            )

    def test_transport_comparison_emits_signed_ternary_frontier(self):
        reference_states = np.array([0, 1, 2, 2], dtype=np.uint8)
        candidate_states = np.array([0, 0, 2, 1], dtype=np.uint8)
        reference_score = np.array([0.0, 2.0, 5.0, 4.0])
        candidate_score = np.array([0.0, 1.0, 6.0, 5.0])
        comparison = compare_guard_transport(
            reference_states,
            candidate_states,
            reference_score,
            candidate_score,
        )
        self.assertAlmostEqual(comparison.state_agreement, 0.5)
        self.assertAlmostEqual(comparison.ascended_iou, 0.5)
        self.assertEqual(comparison.frontier.tolist(), [0, -1, 0, -1])
        self.assertTrue(np.allclose(comparison.score_residual, [0, -1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
