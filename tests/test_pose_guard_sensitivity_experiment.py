import unittest

from scripts.pose_guard_sensitivity_experiment import run_controlled_probe


class PoseGuardSensitivityExperimentTests(unittest.TestCase):
    def test_controlled_probe_contains_oracle_and_is_deterministic(self):
        result = run_controlled_probe()
        self.assertEqual(
            [row["name"] for row in result["cases"]],
            ["oracle", "origin_25cm", "yaw_10deg", "scale_plus_10pct", "residual_plus_5"],
        )
        self.assertTrue(result["cases"][0]["within_policy"])
        self.assertEqual(result["cases"][0]["state_change_count"], 0)
        self.assertEqual(result["cases"][0]["frontier_nonzero_count"], 0)
        self.assertTrue(any(not row["within_policy"] for row in result["cases"][1:]))
        self.assertEqual(result, run_controlled_probe())


if __name__ == "__main__":
    unittest.main()
