import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scripts.issue20_pose_recovery import (
    derive_camera_matrix_from_fov,
    known_relative_pose_from_observations,
    match_static_features,
    recover_and_score_issue20_pair,
    recover_pose_from_image_pair,
    score_recovered_pose_against_known,
)


def render_scene(path1, path2):
    h, w = 480, 640
    f = 700.0
    K = np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
    rng = np.random.default_rng(4)
    points_world = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, 80),
            rng.uniform(-1.0, 1.0, 80),
            rng.uniform(5.0, 9.0, 80),
        ]
    )
    yaw = math.radians(6.0)
    R = np.array(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]],
        dtype=np.float64,
    )
    C2 = np.array([1.2, 0.08, 0.02], dtype=np.float64)
    t2 = -R @ C2

    def project(Rcw, tcw):
        camera = (Rcw @ points_world.T).T + tcw
        pixels = (K @ camera.T).T
        return pixels[:, :2] / pixels[:, 2:3]

    p1 = project(np.eye(3), np.zeros(3))
    p2 = project(R, t2)
    img1 = np.zeros((h, w), np.uint8)
    img2 = np.zeros((h, w), np.uint8)
    for i, (a, b) in enumerate(zip(p1, p2)):
        x1, y1 = map(int, np.round(a))
        x2, y2 = map(int, np.round(b))
        if 8 <= x1 < w - 8 and 8 <= y1 < h - 8 and 8 <= x2 < w - 8 and 8 <= y2 < h - 8:
            prng = np.random.default_rng(1000 + i)
            patch = (prng.integers(0, 2, size=(11, 11)) * 255).astype(np.uint8)
            img1[y1 - 5 : y1 + 6, x1 - 5 : x1 + 6] = np.maximum(
                img1[y1 - 5 : y1 + 6, x1 - 5 : x1 + 6], patch
            )
            img2[y2 - 5 : y2 + 6, x2 - 5 : x2 + 6] = np.maximum(
                img2[y2 - 5 : y2 + 6, x2 - 5 : x2 + 6], patch
            )
    cv2.imwrite(str(path1), img1)
    cv2.imwrite(str(path2), img2)
    return K, R, C2


class RecoveryTests(unittest.TestCase):
    def test_fov_matrix_matches_known_intrinsics(self):
        fov = 2 * math.degrees(math.atan(320 / 700))
        K = derive_camera_matrix_from_fov(640, 480, fov)
        self.assertAlmostEqual(K[0, 0], 700, places=6)
        self.assertAlmostEqual(K[1, 1], 700, places=6)

    def test_static_feature_matcher_respects_dynamic_masks(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.png"
            p2 = Path(td) / "b.png"
            render_scene(p1, p2)
            img1 = cv2.imread(str(p1), 0)
            img2 = cv2.imread(str(p2), 0)
            dynamic = np.zeros_like(img1)
            dynamic[:, :320] = 255
            a, b = match_static_features(
                img1,
                img2,
                dynamic_mask1=dynamic,
                dynamic_mask2=dynamic,
                min_matches=8,
            )
            self.assertEqual(a.shape, b.shape)
            self.assertGreaterEqual(len(a), 8)
            self.assertTrue(np.all(a[:, 0] >= 320))
            self.assertTrue(np.all(b[:, 0] >= 320))

    def test_image_pair_recovery_and_known_pose_score(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.png"
            p2 = Path(td) / "b.png"
            K, R, C2 = render_scene(p1, p2)
            estimate = recover_pose_from_image_pair(p1, p2, K, min_matches=12)
            score = score_recovered_pose_against_known(estimate, R, C2)
            self.assertLess(score.rotation_error_deg, 3.0)
            self.assertLess(score.translation_direction_error_deg, 8.0)
            self.assertGreaterEqual(score.inlier_count, 8)
            self.assertEqual(estimate.status, "candidate")
            self.assertFalse(estimate.metric_scale_paid)

    def test_known_observations_convert_to_relative_validation_geometry(self):
        a = SimpleNamespace(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0
        )
        b = SimpleNamespace(
            position=(1.0, 0.0, 0.0), yaw_deg=90.0, pitch_deg=0.0, roll_deg=0.0
        )
        R21, C2 = known_relative_pose_from_observations(a, b)
        self.assertTrue(np.allclose(C2, np.array([1.0, 0.0, 0.0]), atol=1e-9))
        expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.allclose(R21, expected, atol=1e-9))

    def test_issue20_pair_wrapper_recovers_and_scores_against_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "cam0.png"
            p2 = Path(td) / "cam1.png"
            _, _, C2 = render_scene(p1, p2)
            fov = 2 * math.degrees(math.atan(320 / 700))
            a = SimpleNamespace(
                camera_id=0,
                frame_index=0,
                position=(0.0, 0.0, 0.0),
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=0.0,
                fov_deg=fov,
                image_file="cam0.png",
            )
            b = SimpleNamespace(
                camera_id=1,
                frame_index=0,
                position=tuple(C2),
                yaw_deg=0.0,
                pitch_deg=0.0,
                roll_deg=-6.0,
                fov_deg=fov,
                image_file="cam1.png",
            )
            estimate, score = recover_and_score_issue20_pair(a, b, td, min_matches=12)
            self.assertLess(score.rotation_error_deg, 3.0)
            self.assertLess(score.translation_direction_error_deg, 8.0)
            self.assertFalse(estimate.metric_scale_paid)


if __name__ == "__main__":
    unittest.main()
