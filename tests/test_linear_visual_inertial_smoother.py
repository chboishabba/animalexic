import unittest

import numpy as np

from scripts.linear_visual_inertial_smoother import (
    InertialPVConstraint,
    VisualTranslationConstraint,
    optimize_fixed_rotation_position_velocity,
)


class LinearVisualInertialSmootherTests(unittest.TestCase):
    def test_constant_velocity_chain_with_visual_loop(self):
        rotations = [np.eye(3), np.eye(3), np.eye(3)]
        inertial = [
            InertialPVConstraint(0, 1, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, "imu01"),
            InertialPVConstraint(1, 2, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, "imu12"),
        ]
        visual = [
            VisualTranslationConstraint(0, 2, (2.0, 0.0, 0.0), 2.0, "visual-loop")
        ]
        candidate = optimize_fixed_rotation_position_velocity(
            rotations_world_from_camera=rotations,
            inertial_constraints=inertial,
            visual_constraints=visual,
            anchor_position_world_m=(0.0, 0.0, 0.0),
            anchor_velocity_world_m_s=(1.0, 0.0, 0.0),
            max_rms_constraint_residual=1e-8,
        )
        positions = np.asarray(candidate.positions_world_m)
        velocities = np.asarray(candidate.velocities_world_m_s)
        self.assertTrue(np.allclose(positions, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], atol=1e-7))
        self.assertTrue(np.allclose(velocities, [[1, 0, 0], [1, 0, 0], [1, 0, 0]], atol=1e-7))
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.full_nonlinear_vio_paid)

    def test_inconsistent_visual_constraint_abstains(self):
        rotations = [np.eye(3), np.eye(3)]
        inertial = [
            InertialPVConstraint(0, 1, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, "imu")
        ]
        visual = [
            VisualTranslationConstraint(0, 1, (4.0, 0.0, 0.0), 1.0, "bad-visual")
        ]
        candidate = optimize_fixed_rotation_position_velocity(
            rotations_world_from_camera=rotations,
            inertial_constraints=inertial,
            visual_constraints=visual,
            anchor_position_world_m=(0.0, 0.0, 0.0),
            anchor_velocity_world_m_s=(1.0, 0.0, 0.0),
            max_rms_constraint_residual=0.2,
        )
        self.assertEqual(candidate.status, "abstain")
        self.assertGreater(candidate.rms_constraint_residual, 0.2)

    def test_missing_inertial_connectivity_fails_closed(self):
        with self.assertRaises(ValueError):
            optimize_fixed_rotation_position_velocity(
                rotations_world_from_camera=[np.eye(3), np.eye(3), np.eye(3)],
                inertial_constraints=[
                    InertialPVConstraint(0, 1, 1.0, (0, 0, 0), (0, 0, 0), 1.0, "imu")
                ],
                visual_constraints=[],
                anchor_position_world_m=(0, 0, 0),
                anchor_velocity_world_m_s=(0, 0, 0),
                max_rms_constraint_residual=1.0,
            )


if __name__ == "__main__":
    unittest.main()
