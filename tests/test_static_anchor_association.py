import math
import unittest

import numpy as np

from scripts.static_anchor_association import (
    StaticAnchorObservation,
    associate_static_anchors,
    robust_estimate_world_weld,
)


class StaticAnchorAssociationTests(unittest.TestCase):
    def test_association_requires_same_identity_static_and_time_window(self):
        source = [
            StaticAnchorObservation("wall:a", "phoneA", 1.00, (0.0, 0.0, 0.0), True, 0.9, "track:a0"),
            StaticAnchorObservation("wall:b", "phoneA", 1.00, (1.0, 0.0, 0.0), False, 0.9, "track:b0"),
            StaticAnchorObservation("wall:c", "phoneA", 4.00, (0.0, 1.0, 0.0), True, 0.9, "track:c0"),
        ]
        target = [
            StaticAnchorObservation("wall:a", "phoneB", 1.03, (2.0, 1.0, 0.0), True, 0.8, "track:a1"),
            StaticAnchorObservation("wall:b", "phoneB", 1.02, (3.0, 1.0, 0.0), True, 0.9, "track:b1"),
            StaticAnchorObservation("wall:c", "phoneB", 1.00, (2.0, 2.0, 0.0), True, 0.9, "track:c1"),
        ]
        pairs = associate_static_anchors(source, target, max_time_delta_s=0.10, min_confidence=0.5)
        self.assertEqual([p.anchor_id for p in pairs], ["wall:a"])
        self.assertEqual(pairs[0].source_provenance_ref, "track:a0")
        self.assertEqual(pairs[0].target_provenance_ref, "track:a1")
        self.assertEqual(pairs[0].status, "candidate")

    def test_association_selects_closest_time_when_identity_repeats(self):
        source = [
            StaticAnchorObservation("lamp", "phoneA", 2.0, (0, 0, 0), True, 0.9, "a"),
        ]
        target = [
            StaticAnchorObservation("lamp", "phoneB", 1.7, (1, 0, 0), True, 0.9, "far"),
            StaticAnchorObservation("lamp", "phoneB", 2.02, (1, 0, 0), True, 0.7, "near"),
        ]
        pairs = associate_static_anchors(source, target, max_time_delta_s=0.5)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].target_provenance_ref, "near")

    @staticmethod
    def make_outlier_problem():
        source = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.2], [0.3, 0.2, 1.0], [1.2, 0.4, 0.8],
        ])
        angle = math.radians(25.0)
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t = np.array([2.0, -0.5, 0.3])
        target = (R @ source.T).T + t
        target[-1] = np.array([8.0, -7.0, 5.0])
        pairs = []
        for i, (a, b) in enumerate(zip(source, target)):
            pairs.extend(associate_static_anchors(
                [StaticAnchorObservation(f"p{i}", "A", 1.0, tuple(a), True, 1.0, f"A:{i}")],
                [StaticAnchorObservation(f"p{i}", "B", 1.0, tuple(b), True, 1.0, f"B:{i}")],
                max_time_delta_s=0.1,
            ))
        return pairs, R, t

    def test_robust_weld_rejects_geometric_outlier(self):
        pairs, expected_R, expected_t = self.make_outlier_problem()
        receipt = robust_estimate_world_weld(
            pairs,
            max_residual_m=0.03,
            min_inliers=4,
            allow_scale=False,
        )
        self.assertEqual(receipt.status, "candidate")
        self.assertEqual(receipt.inlier_count, 5)
        self.assertEqual(receipt.outlier_count, 1)
        self.assertEqual(receipt.inlier_anchor_ids, ("p0", "p1", "p2", "p3", "p4"))
        recovered_R = np.asarray(receipt.weld.rotation_target_from_source).reshape(3, 3)
        recovered_t = np.asarray(receipt.weld.translation_target_from_source_m)
        self.assertTrue(np.allclose(recovered_R, expected_R, atol=1e-6))
        self.assertTrue(np.allclose(recovered_t, expected_t, atol=1e-6))

    def test_robust_weld_fails_closed_without_consensus(self):
        pairs, _, _ = self.make_outlier_problem()
        with self.assertRaises(ValueError):
            robust_estimate_world_weld(
                pairs[:3],
                max_residual_m=1e-6,
                min_inliers=4,
                allow_scale=False,
            )

    def test_similarity_weld_retains_explicit_scale_coordinate(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.4, 1.0]], float)
        scale = 1.7
        target = scale * source + np.array([3.0, -1.0, 0.5])
        pairs = []
        for i, (a, b) in enumerate(zip(source, target)):
            pairs.extend(associate_static_anchors(
                [StaticAnchorObservation(f"s{i}", "A", 0.0, tuple(a), True, 1.0, f"sa{i}")],
                [StaticAnchorObservation(f"s{i}", "B", 0.0, tuple(b), True, 1.0, f"sb{i}")],
            ))
        receipt = robust_estimate_world_weld(pairs, max_residual_m=1e-6, min_inliers=4, allow_scale=True)
        self.assertAlmostEqual(receipt.weld.scale, scale, places=6)
        self.assertFalse(receipt.weld.metric_scale_preserved)
        self.assertTrue(receipt.similarity_scale_exposed)


if __name__ == "__main__":
    unittest.main()
