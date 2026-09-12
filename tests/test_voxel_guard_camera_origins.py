import unittest

import numpy as np

from scripts.voxel_guard import (
    VoxelGridSpec,
    VoxelGuardParams,
    accumulate_candidate_voxels,
)


class VoxelGuardCameraOriginTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridSpec(
            origin=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
            voxel_size=1.0,
            dims=(8, 8, 8),
        )
        self.params = VoxelGuardParams(alpha=0.0, alpha_h=0.0, ray_decay=0.0)
        self.points = [np.array([[2.5, 0.5, 0.5], [2.5, 1.5, 0.5]], dtype=np.float32)]
        self.weights = [np.array([1.0, 1.0], dtype=np.float32)]
        self.residuals = [np.array([0.0, 0.0], dtype=np.float32)]

    def test_omitted_origins_match_explicit_zero_origins(self):
        legacy = accumulate_candidate_voxels(
            self.grid, self.points, self.weights, self.residuals, self.params
        )
        explicit = accumulate_candidate_voxels(
            self.grid,
            self.points,
            self.weights,
            self.residuals,
            self.params,
            frame_camera_origins=[np.zeros((2, 3), dtype=np.float32)],
        )
        for old, new in zip(legacy, explicit):
            self.assertTrue(np.array_equal(old, new))

    def test_per_point_origins_change_ray_support(self):
        zero = accumulate_candidate_voxels(
            self.grid,
            self.points,
            self.weights,
            self.residuals,
            self.params,
            frame_camera_origins=[np.zeros((2, 3), dtype=np.float32)],
        )[0]
        shifted = accumulate_candidate_voxels(
            self.grid,
            self.points,
            self.weights,
            self.residuals,
            self.params,
            frame_camera_origins=[
                np.array([[0.0, 2.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
            ],
        )[0]
        self.assertFalse(np.array_equal(zero, shifted))
        # Endpoints remain supported even though the traversed camera rays differ.
        for point in self.points[0]:
            idx = tuple(np.floor((point - self.grid.origin) / self.grid.voxel_size).astype(int))
            self.assertGreater(float(zero[idx]), 0.0)
            self.assertGreater(float(shifted[idx]), 0.0)

    def test_origin_shape_must_match_points(self):
        with self.assertRaises(ValueError):
            accumulate_candidate_voxels(
                self.grid,
                self.points,
                self.weights,
                self.residuals,
                self.params,
                frame_camera_origins=[np.zeros((1, 3), dtype=np.float32)],
            )

    def test_origin_factor_still_applies_with_nonzero_origins(self):
        evidence = accumulate_candidate_voxels(
            self.grid,
            [self.points[0][:1]],
            [self.weights[0][:1]],
            [self.residuals[0][:1]],
            self.params,
            frame_origin_factors=[np.array([0.25], dtype=np.float32)],
            frame_camera_origins=[np.array([[0.0, 2.0, 0.0]], dtype=np.float32)],
        )[0]
        self.assertAlmostEqual(float(np.sum(evidence)), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
