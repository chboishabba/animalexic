import unittest

import numpy as np

from scripts.pose_graph_trajectory_adapter import compose_pose_graph_trajectory
from scripts.rotation_pose_graph import RotationPoseGraphCandidate
from scripts.translation_pose_graph import TranslationPoseGraphCandidate


class PoseGraphTrajectoryAdapterTests(unittest.TestCase):
    def test_translation_and_rotation_candidates_reenter_existing_trajectory_carrier(self):
        translation = TranslationPoseGraphCandidate(
            positions_world_m=((0, 0, 0), (1, 0, 0), (2, 1, 0)),
            rms_constraint_residual_m=0.0,
            constraint_count=3,
            rank=6,
            status="candidate",
        )
        rotations = RotationPoseGraphCandidate(
            rotations_world_from_frame=(
                tuple(np.eye(3).reshape(-1)),
                tuple(np.eye(3).reshape(-1)),
                tuple(np.eye(3).reshape(-1)),
            ),
            rms_rotation_residual_deg=0.0,
            constraint_count=3,
            iteration_count=1,
            status="candidate",
        )
        trajectory = compose_pose_graph_trajectory(
            translation,
            rotations,
            times_s=(0.0, 1.0, 2.0),
            velocities_world_m_s=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        )
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(trajectory[2].position_world_m, (2.0, 1.0, 0.0))
        self.assertEqual(trajectory[2].velocity_world_m_s, (1.0, 1.0, 0.0))
        self.assertTrue(all(k.status == "candidate" for k in trajectory))

    def test_abstained_component_fails_closed(self):
        translation = TranslationPoseGraphCandidate(
            positions_world_m=((0, 0, 0), (1, 0, 0)),
            rms_constraint_residual_m=1.0,
            constraint_count=1,
            rank=3,
            status="abstain",
        )
        rotations = RotationPoseGraphCandidate(
            rotations_world_from_frame=(tuple(np.eye(3).reshape(-1)), tuple(np.eye(3).reshape(-1))),
            rms_rotation_residual_deg=0.0,
            constraint_count=1,
            iteration_count=1,
            status="candidate",
        )
        with self.assertRaises(ValueError):
            compose_pose_graph_trajectory(translation, rotations, times_s=(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
