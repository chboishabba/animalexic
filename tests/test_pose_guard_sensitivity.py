import unittest

import numpy as np

from scripts.pose_guard_sensitivity import (
    ConsumerQualityPolicy,
    GeometryFibrePerturbation,
    sensitivity_case_from_comparison,
    select_refinement_frontier,
)
from scripts.shared_world_guard_transport import GuardTransportComparison


class PoseGuardSensitivityTests(unittest.TestCase):
    def test_quality_policy_accepts_exact_transport(self):
        comparison = GuardTransportComparison(
            state_agreement=1.0,
            ascended_iou=1.0,
            score_residual=np.zeros((2, 2), dtype=float),
            state_change_mask=np.zeros((2, 2), dtype=bool),
            frontier=np.zeros((2, 2), dtype=np.int8),
        )
        policy = ConsumerQualityPolicy(0.99, 0.95, 0.1)
        case = sensitivity_case_from_comparison(
            GeometryFibrePerturbation("oracle"), comparison, policy
        )
        self.assertTrue(case.within_policy)
        self.assertEqual(case.state_change_count, 0)
        self.assertEqual(case.frontier_nonzero_count, 0)

    def test_policy_rejects_consumer_visible_pose_defect(self):
        comparison = GuardTransportComparison(
            state_agreement=0.75,
            ascended_iou=0.5,
            score_residual=np.array([0.0, 0.4, -0.2, 0.0]),
            state_change_mask=np.array([False, True, True, False]),
            frontier=np.array([0, -1, 1, 0], dtype=np.int8),
        )
        policy = ConsumerQualityPolicy(0.95, 0.9, 0.25)
        case = sensitivity_case_from_comparison(
            GeometryFibrePerturbation("camera_origin", origin_delta_m=(0.1, 0, 0)),
            comparison,
            policy,
        )
        self.assertFalse(case.within_policy)
        self.assertEqual(case.state_change_count, 2)
        self.assertEqual(case.frontier_nonzero_count, 2)
        self.assertAlmostEqual(case.max_abs_score_residual, 0.4)

    def test_refinement_frontier_is_pareto_not_weighted_scalar(self):
        policy = ConsumerQualityPolicy(0.99, 0.99, 0.05)
        cases = [
            sensitivity_case_from_comparison(
                GeometryFibrePerturbation("origin"),
                GuardTransportComparison(
                    0.70,
                    0.60,
                    np.array([0.8]),
                    np.array([True]),
                    np.array([-1], dtype=np.int8),
                ),
                policy,
            ),
            sensitivity_case_from_comparison(
                GeometryFibrePerturbation("residual"),
                GuardTransportComparison(
                    0.95,
                    0.90,
                    np.array([0.2]),
                    np.array([True]),
                    np.array([-1], dtype=np.int8),
                ),
                policy,
            ),
            sensitivity_case_from_comparison(
                GeometryFibrePerturbation("orientation"),
                GuardTransportComparison(
                    0.80,
                    0.95,
                    np.array([1.0]),
                    np.array([True]),
                    np.array([1], dtype=np.int8),
                ),
                policy,
            ),
        ]
        frontier = select_refinement_frontier(cases)
        names = {case.perturbation.name for case in frontier}
        self.assertEqual(names, {"origin", "orientation"})


if __name__ == "__main__":
    unittest.main()
