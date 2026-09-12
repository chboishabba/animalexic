import unittest

from scripts.pose_guard_sensitivity import (
    ConsumerQualityPolicy,
    GeometryFibrePerturbation,
    active_perturbation_fibres,
    quality_targeted_refinement,
)
from scripts.pose_guard_sensitivity_experiment import _grid, _params, _reference_frames


class QualityTargetedPoseRefinementTests(unittest.TestCase):
    def test_active_fibre_attribution_is_coordinate_preserving(self):
        perturbation = GeometryFibrePerturbation(
            "mixed",
            origin_delta_m=(0.1, 0.0, 0.0),
            rotation_delta_deg_xyz=(0.0, 0.0, 5.0),
            scale_factor=1.02,
            residual_delta=1.0,
        )
        self.assertEqual(
            active_perturbation_fibres(perturbation),
            ("camera_origin", "orientation", "metric_scale", "observation_residual"),
        )

    def test_quality_targeted_refinement_reports_consumer_plateau(self):
        policy = ConsumerQualityPolicy(0.995, 0.98, 0.025)
        result = quality_targeted_refinement(
            _reference_frames(),
            GeometryFibrePerturbation(
                "yaw_10deg", rotation_delta_deg_xyz=(0.0, 0.0, 10.0)
            ),
            _grid(),
            _params(),
            policy,
            max_steps=10,
        )
        self.assertGreater(len(result.history), 1)
        self.assertFalse(result.history[0].case.within_policy)
        self.assertFalse(result.history[-1].case.within_policy)
        self.assertEqual(result.termination, "consumer_plateau")
        self.assertEqual(
            active_perturbation_fibres(result.history[0].case.perturbation),
            ("orientation",),
        )


if __name__ == "__main__":
    unittest.main()
