import unittest

import numpy as np

from scripts.translation_pose_graph import (
    TranslationConstraint,
    optimize_translation_pose_graph,
)


class TranslationPoseGraphTests(unittest.TestCase):
    def test_chain_plus_loop_recovers_keyframe_positions(self):
        constraints = [
            TranslationConstraint(0, 1, (1.0, 0.0, 0.0), 1.0, "visual"),
            TranslationConstraint(1, 2, (1.0, 1.0, 0.0), 1.0, "visual"),
            TranslationConstraint(0, 2, (2.0, 1.0, 0.0), 2.0, "loop"),
        ]
        candidate = optimize_translation_pose_graph(
            keyframe_count=3,
            constraints=constraints,
            anchor_position_world_m=(0.0, 0.0, 0.0),
            max_rms_constraint_residual_m=1e-8,
        )
        positions = np.asarray(candidate.positions_world_m)
        self.assertTrue(np.allclose(positions[0], (0, 0, 0), atol=1e-8))
        self.assertTrue(np.allclose(positions[1], (1, 0, 0), atol=1e-8))
        self.assertTrue(np.allclose(positions[2], (2, 1, 0), atol=1e-8))
        self.assertLess(candidate.rms_constraint_residual_m, 1e-8)
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.full_vio_optimization_paid)

    def test_disconnected_pose_graph_fails_closed(self):
        constraints = [
            TranslationConstraint(0, 1, (1.0, 0.0, 0.0), 1.0, "visual"),
        ]
        with self.assertRaises(ValueError):
            optimize_translation_pose_graph(
                keyframe_count=3,
                constraints=constraints,
                anchor_position_world_m=(0.0, 0.0, 0.0),
                max_rms_constraint_residual_m=0.1,
            )

    def test_inconsistent_loop_abstains_on_residual_gate(self):
        constraints = [
            TranslationConstraint(0, 1, (1.0, 0.0, 0.0), 1.0, "visual"),
            TranslationConstraint(1, 2, (1.0, 0.0, 0.0), 1.0, "visual"),
            TranslationConstraint(0, 2, (5.0, 0.0, 0.0), 1.0, "loop"),
        ]
        candidate = optimize_translation_pose_graph(
            keyframe_count=3,
            constraints=constraints,
            anchor_position_world_m=(0.0, 0.0, 0.0),
            max_rms_constraint_residual_m=0.2,
        )
        self.assertEqual(candidate.status, "abstain")
        self.assertGreater(candidate.rms_constraint_residual_m, 0.2)
        self.assertFalse(candidate.full_vio_optimization_paid)


if __name__ == "__main__":
    unittest.main()
