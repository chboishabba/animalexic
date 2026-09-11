import unittest

from scripts.camera_pose_fibre import (
    PoseCandidate,
    PoseEvidence,
    PoseSufficiencyPolicy,
    static_pose_evidence,
    dynamic_pose_evidence,
    fuse_pose_candidates,
    pose_sufficient_for_consumer,
)


class CameraPoseFibreTests(unittest.TestCase):
    def test_dynamic_target_is_excluded_from_pose_evidence(self):
        samples = [
            {"id": "wall", "static": True, "reprojection_error": 0.4},
            {"id": "dog", "static": False, "reprojection_error": 0.1},
        ]
        kept = static_pose_evidence(samples)
        self.assertEqual([s["id"] for s in kept], ["wall"])

    def test_imu_is_prior_not_authoritative_pose(self):
        ev = dynamic_pose_evidence(
            camera_id=2,
            frame_index=10,
            visual_track_count=42,
            visual_reprojection_rms=0.8,
            imu_rotation_sigma_deg=1.2,
            metric_scale_source=None,
            clock_offset_sigma_ms=3.0,
            rolling_shutter_sigma_ms=1.0,
        )
        self.assertTrue(ev.has_visual_support)
        self.assertTrue(ev.has_imu_prior)
        self.assertFalse(ev.metric_scale_paid)
        self.assertEqual(ev.status, "candidate")

    def test_candidate_fusion_preserves_scale_and_clock_debt(self):
        a = PoseCandidate("cam0", 5, 0.9, True, True, False, 4.0, 1.0)
        b = PoseCandidate("cam0", 5, 0.7, True, True, True, 2.0, 0.5)
        fused = fuse_pose_candidates([a, b])
        self.assertAlmostEqual(fused.confidence, 0.8)
        self.assertTrue(fused.metric_scale_paid)
        self.assertEqual(fused.clock_offset_sigma_ms, 2.0)
        self.assertEqual(fused.rolling_shutter_sigma_ms, 0.5)
        self.assertEqual(fused.status, "candidate")

    def test_pose_sufficiency_is_consumer_indexed(self):
        pose = PoseCandidate("cam0", 7, 0.85, True, True, True, 1.0, 0.4)
        loose = PoseSufficiencyPolicy(
            min_confidence=0.8,
            require_metric_scale=True,
            max_clock_offset_sigma_ms=2.0,
            max_rolling_shutter_sigma_ms=1.0,
        )
        strict = PoseSufficiencyPolicy(
            min_confidence=0.95,
            require_metric_scale=True,
            max_clock_offset_sigma_ms=0.5,
            max_rolling_shutter_sigma_ms=0.2,
        )
        self.assertTrue(pose_sufficient_for_consumer(pose, loose))
        self.assertFalse(pose_sufficient_for_consumer(pose, strict))


if __name__ == "__main__":
    unittest.main()
