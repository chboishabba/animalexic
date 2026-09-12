import unittest

import numpy as np

from scripts.pose_guard_sensitivity import (
    ConsumerQualityPolicy,
    GeometryFibrePerturbation,
    perturb_guard_frames,
    run_guard_sensitivity_portfolio,
)
from scripts.shared_world_guard_transport import GuardFrameInputs
from scripts.voxel_guard import VoxelGridSpec, VoxelGuardParams


class PoseGuardPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridSpec(
            origin=np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            voxel_size=0.25,
            dims=(10, 10, 10),
        )
        self.params = VoxelGuardParams(
            alpha=1.0,
            alpha_h=1.0,
            beta=0.0,
            tau_p=0.02,
            tau_a=0.04,
            h_a=1.0,
            epsilon_rho=10.0,
            ray_decay=0.0,
            sigma_rho=10.0,
            gamma_neighbor=0.0,
        )
        self.frames = [
            GuardFrameInputs(
                time_s=0.0,
                points=np.array([[1.5, 0.0, 0.5], [1.5, 0.5, 0.5]], dtype=np.float32),
                camera_origins=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
                weights=np.ones(2, dtype=np.float32),
                residuals=np.zeros(2, dtype=np.float32),
                origin_factors=np.ones(2, dtype=np.float32),
            )
        ]
        self.policy = ConsumerQualityPolicy(0.99, 0.99, 0.02)

    def test_zero_perturbation_preserves_guard_exactly(self):
        runs = run_guard_sensitivity_portfolio(
            self.frames,
            [GeometryFibrePerturbation("zero")],
            self.grid,
            self.params,
            self.policy,
        )
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0].case.within_policy)
        self.assertEqual(runs[0].case.state_change_count, 0)
        self.assertAlmostEqual(runs[0].case.max_abs_score_residual, 0.0)

    def test_camera_origin_perturbation_changes_consumer_surface(self):
        runs = run_guard_sensitivity_portfolio(
            self.frames,
            [GeometryFibrePerturbation("origin", origin_delta_m=(0.5, 0.5, 0.0))],
            self.grid,
            self.params,
            self.policy,
        )
        self.assertFalse(runs[0].case.within_policy)
        self.assertGreater(runs[0].case.state_change_count, 0)
        self.assertGreater(runs[0].case.frontier_nonzero_count, 0)

    def test_orientation_perturbation_rotates_endpoint_about_camera(self):
        perturbed = perturb_guard_frames(
            self.frames,
            GeometryFibrePerturbation(
                "yaw", rotation_delta_deg_xyz=(0.0, 0.0, 90.0)
            ),
        )
        original_ray = self.frames[0].points[0] - self.frames[0].camera_origins[0]
        rotated_ray = perturbed[0].points[0] - perturbed[0].camera_origins[0]
        self.assertAlmostEqual(np.linalg.norm(original_ray), np.linalg.norm(rotated_ray), places=6)
        self.assertFalse(np.allclose(original_ray, rotated_ray))


if __name__ == "__main__":
    unittest.main()
