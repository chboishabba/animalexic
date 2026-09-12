import math
import unittest

import numpy as np

from scripts.rotation_pose_graph import (
    RotationConstraint,
    optimize_rotation_pose_graph,
)


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def relative_target_from_source(R_world_source, R_world_target):
    return R_world_target.T @ R_world_source


class RotationPoseGraphTests(unittest.TestCase):
    def test_chain_plus_loop_recovers_orientations(self):
        truth = [np.eye(3), rz(20), rz(45)]
        constraints = [
            RotationConstraint(0, 1, tuple(relative_target_from_source(truth[0], truth[1]).reshape(-1)), 1.0, "visual"),
            RotationConstraint(1, 2, tuple(relative_target_from_source(truth[1], truth[2]).reshape(-1)), 1.0, "visual"),
            RotationConstraint(0, 2, tuple(relative_target_from_source(truth[0], truth[2]).reshape(-1)), 2.0, "loop"),
        ]
        candidate = optimize_rotation_pose_graph(
            frame_count=3,
            constraints=constraints,
            anchor_rotation_world_from_frame=np.eye(3),
            max_rms_rotation_residual_deg=1e-6,
        )
        recovered = [np.asarray(r).reshape(3, 3) for r in candidate.rotations_world_from_frame]
        for got, expected in zip(recovered, truth):
            self.assertTrue(np.allclose(got, expected, atol=1e-6))
        self.assertLess(candidate.rms_rotation_residual_deg, 1e-6)
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.full_vio_optimization_paid)

    def test_disconnected_rotation_graph_fails_closed(self):
        constraints = [
            RotationConstraint(0, 1, tuple(np.eye(3).reshape(-1)), 1.0, "visual")
        ]
        with self.assertRaises(ValueError):
            optimize_rotation_pose_graph(
                frame_count=3,
                constraints=constraints,
                anchor_rotation_world_from_frame=np.eye(3),
                max_rms_rotation_residual_deg=1.0,
            )

    def test_inconsistent_rotation_loop_abstains(self):
        truth = [np.eye(3), rz(10), rz(20)]
        constraints = [
            RotationConstraint(0, 1, tuple(relative_target_from_source(truth[0], truth[1]).reshape(-1)), 1.0, "visual"),
            RotationConstraint(1, 2, tuple(relative_target_from_source(truth[1], truth[2]).reshape(-1)), 1.0, "visual"),
            RotationConstraint(0, 2, tuple(relative_target_from_source(truth[0], rz(80)).reshape(-1)), 1.0, "bad-loop"),
        ]
        candidate = optimize_rotation_pose_graph(
            frame_count=3,
            constraints=constraints,
            anchor_rotation_world_from_frame=np.eye(3),
            max_rms_rotation_residual_deg=2.0,
        )
        self.assertEqual(candidate.status, "abstain")
        self.assertGreater(candidate.rms_rotation_residual_deg, 2.0)


if __name__ == "__main__":
    unittest.main()
