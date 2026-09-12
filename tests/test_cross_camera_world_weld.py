import math
import unittest

import numpy as np

import scripts.cross_camera_world_weld as weld
from scripts.visual_inertial_pose import VisualInertialTrajectoryKeyframe


def rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class CrossCameraWorldWeldTests(unittest.TestCase):
    def points(self):
        return np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 0.5]],
            float,
        )

    def test_rigid_weld_recovers_shared_world_transform(self):
        R = rz(25)
        t = np.array([2.0, -1.0, 0.4])
        src = self.points()
        dst = (R @ src.T).T + t
        result = weld.estimate_world_weld(src, dst, allow_scale=False)
        self.assertEqual(result.status, "candidate")
        self.assertAlmostEqual(result.scale, 1.0, places=8)
        self.assertLess(result.rms_residual_m, 1e-8)
        self.assertTrue(
            np.allclose(
                np.asarray(result.rotation_target_from_source).reshape(3, 3),
                R,
                atol=1e-8,
            )
        )
        self.assertTrue(
            np.allclose(result.translation_target_from_source_m, t, atol=1e-8)
        )
        self.assertTrue(result.metric_scale_preserved)

    def test_similarity_weld_exposes_scale_coordinate(self):
        R = rz(-15)
        scale = 1.7
        t = np.array([-0.3, 0.8, 1.1])
        src = self.points()
        dst = (scale * (R @ src.T)).T + t
        result = weld.estimate_world_weld(src, dst, allow_scale=True)
        self.assertAlmostEqual(result.scale, scale, places=8)
        self.assertLess(result.rms_residual_m, 1e-8)
        self.assertFalse(result.metric_scale_preserved)

    def test_collinear_anchors_fail_closed(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
        with self.assertRaises(ValueError):
            weld.estimate_world_weld(src, src, allow_scale=False)

    def test_candidate_trajectory_transforms_to_shared_world(self):
        R = rz(90)
        t = np.array([10.0, 0.0, 0.0])
        candidate = weld.WorldWeldCandidate(
            tuple(R.reshape(-1)),
            tuple(t),
            1.0,
            0.0,
            True,
            4,
            "candidate",
        )
        local = [
            VisualInertialTrajectoryKeyframe(
                0.0,
                tuple(np.eye(3).reshape(-1)),
                (0, 0, 0),
                (0, 0, 0),
            ),
            VisualInertialTrajectoryKeyframe(
                1.0,
                tuple(np.eye(3).reshape(-1)),
                (1, 0, 0),
                (1, 0, 0),
            ),
        ]
        world = weld.apply_world_weld_to_trajectory(local, candidate)
        self.assertTrue(np.allclose(world[0].position_world_m, (10, 0, 0), atol=1e-8))
        self.assertTrue(np.allclose(world[1].position_world_m, (10, 1, 0), atol=1e-8))
        self.assertTrue(np.allclose(world[1].velocity_world_m_s, (0, 1, 0), atol=1e-8))
        self.assertEqual(world[1].status, "candidate")


if __name__ == "__main__":
    unittest.main()
