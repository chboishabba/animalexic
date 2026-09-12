import math
import unittest

import numpy as np

import scripts.camera_pose_fibre as pose_module
from scripts.camera_pose_fibre import (
    PoseCandidate,
    PoseEvidence,
    PoseSufficiencyPolicy,
    static_pose_evidence,
    dynamic_pose_evidence,
    fuse_pose_candidates,
    pose_sufficient_for_consumer,
)
from scripts.issue20_multiview import CameraObservation, pixel_ray_world


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

    @staticmethod
    def synthetic_correspondences():
        f = 700.0
        cx, cy = 320.0, 240.0
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        points_world = np.array(
            [
                [-1.2, -0.8, 6.0], [-0.3, -1.0, 7.0], [0.8, -0.7, 6.5],
                [1.4, -0.2, 8.0], [-1.0, 0.3, 7.5], [0.0, 0.1, 5.5],
                [0.9, 0.4, 7.2], [1.5, 0.9, 8.5], [-0.8, 1.0, 6.8],
                [0.3, 1.2, 9.0], [-1.5, 0.7, 9.5], [1.1, 1.3, 6.2],
            ],
            dtype=np.float64,
        )
        yaw = math.radians(7.0)
        R = np.array(
            [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]],
            dtype=np.float64,
        )
        camera2_center = np.array([1.4, 0.15, 0.05], dtype=np.float64)
        t = -R @ camera2_center

        def project(Rcw, tcw):
            camera = (Rcw @ points_world.T).T + tcw
            pixels = (K @ camera.T).T
            return pixels[:, :2] / pixels[:, 2:3]

        p1 = project(np.eye(3), np.zeros(3))
        p2 = project(R, t)
        return K, p1, p2, R, camera2_center

    def test_controlled_pose_perturbation_keeps_ground_truth_identity(self):
        self.assertTrue(
            hasattr(pose_module, "perturb_camera_observation"),
            "controlled pose-perturbation producer is not implemented yet",
        )
        camera = CameraObservation(
            camera_id=3,
            frame_index=11,
            position=(1.0, 2.0, 3.0),
            yaw_deg=10.0,
            pitch_deg=2.0,
            roll_deg=-3.0,
            fov_deg=60.0,
            image_file="cam3_0011.png",
        )
        perturbed = pose_module.perturb_camera_observation(
            camera,
            translation_delta=(0.25, -0.5, 0.75),
            yaw_delta_deg=4.0,
            pitch_delta_deg=-1.0,
            roll_delta_deg=2.5,
        )
        self.assertEqual(perturbed.camera_id, camera.camera_id)
        self.assertEqual(perturbed.frame_index, camera.frame_index)
        self.assertEqual(perturbed.position, (1.25, 1.5, 3.75))
        self.assertAlmostEqual(perturbed.yaw_deg, 14.0)
        self.assertAlmostEqual(perturbed.pitch_deg, 1.0)
        self.assertAlmostEqual(perturbed.roll_deg, -0.5)
        self.assertEqual(perturbed.pose_source, "synthetic_perturbed_known_pose")

    def test_static_correspondences_recover_relative_pose_and_keep_scale_debt(self):
        self.assertTrue(
            hasattr(pose_module, "recover_relative_pose_from_correspondences"),
            "relative-pose producer is not implemented yet",
        )
        K, p1, p2, expected_R, _ = self.synthetic_correspondences()
        static_mask = np.ones(len(p1), dtype=bool)
        static_mask[-1] = False
        p2[-1] = np.array([50.0, 50.0])  # dynamic/outlier track must be excluded
        estimate = pose_module.recover_relative_pose_from_correspondences(
            p1,
            p2,
            K,
            static_mask=static_mask,
        )
        recovered_R = np.asarray(estimate.rotation_cam2_from_cam1).reshape(3, 3)
        delta_R = recovered_R @ expected_R.T
        angle_error = math.degrees(math.acos(np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertLess(angle_error, 1.0)
        self.assertGreaterEqual(estimate.inlier_count, 8)
        self.assertFalse(estimate.metric_scale_paid)
        self.assertEqual(estimate.static_correspondence_count, len(p1) - 1)
        self.assertEqual(estimate.status, "candidate")

    def test_metric_scale_payment_materialises_same_camera_ray_contract(self):
        self.assertTrue(
            hasattr(pose_module, "recover_relative_pose_from_correspondences"),
            "relative-pose producer is not implemented yet",
        )
        self.assertTrue(
            hasattr(pose_module, "materialise_recovered_camera_observation"),
            "recovered-pose adapter is not implemented yet",
        )
        K, p1, p2, _, expected_center = self.synthetic_correspondences()
        baseline = float(np.linalg.norm(expected_center))
        estimate = pose_module.recover_relative_pose_from_correspondences(
            p1,
            p2,
            K,
            metric_baseline=baseline,
            metric_scale_source="synthetic_known_baseline",
        )
        self.assertTrue(estimate.metric_scale_paid)
        self.assertAlmostEqual(np.linalg.norm(estimate.translation_cam2_from_cam1), baseline, places=5)
        recovered = pose_module.materialise_recovered_camera_observation(
            estimate,
            camera_id=1,
            frame_index=0,
            fov_deg=2.0 * math.degrees(math.atan(320.0 / 700.0)),
            image_file="cam1_0000.png",
        )
        self.assertEqual(recovered.pose_source, "image_recovered_relative_pose")
        self.assertLess(np.linalg.norm(np.asarray(recovered.position) - expected_center), 0.08)
        ray = pixel_ray_world(recovered, u=320.0, v=240.0, width=640, height=480)
        self.assertAlmostEqual(np.linalg.norm(ray), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
