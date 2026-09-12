import math
import unittest

import numpy as np

import scripts.visual_inertial_pose as vio


def rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class VisualInertialCorrectionTests(unittest.TestCase):
    def prior(self, yaw_deg=10.0, p=(1.0, 0.0, 0.0)):
        return vio.InertialPosePrior(
            tuple(rz(yaw_deg).reshape(-1)),
            (1.0, 0.0, 0.0),
            tuple(p),
            1.0,
        )

    def test_visual_correction_requires_extrinsic_and_clock_receipts(self):
        self.assertTrue(
            hasattr(vio, "correct_visual_inertial_segment"),
            "visual correction producer missing",
        )
        with self.assertRaises(ValueError):
            vio.correct_visual_inertial_segment(
                self.prior(),
                rz(-10),
                (-1, 0, 0),
                visual_metric_scale_paid=True,
            )

    def test_visual_rotation_corrects_inertial_prediction_in_camera_convention(self):
        result = vio.correct_visual_inertial_segment(
            self.prior(yaw_deg=12),
            rz(-10),
            None,
            visual_metric_scale_paid=False,
            rotation_camera_from_imu=np.eye(3),
            translation_camera_from_imu_m=(0, 0, 0),
            camera_imu_extrinsic_source="synthetic_rig",
            clock_offset_s=0.0,
            clock_alignment_source="synthetic_sync",
            max_rotation_residual_deg=3.0,
        )
        self.assertEqual(result.status, "candidate")
        self.assertAlmostEqual(result.rotation_residual_deg, 2.0, places=5)
        self.assertTrue(result.visual_rotation_applied)
        self.assertFalse(result.visual_metric_position_applied)
        self.assertTrue(
            np.allclose(
                np.asarray(result.rotation_cam_next_from_cam_prev).reshape(3, 3),
                rz(-10),
                atol=1e-8,
            )
        )
        self.assertTrue(
            np.allclose(result.camera_center_delta_in_prev_m, (1, 0, 0), atol=1e-8)
        )

    def test_metric_visual_translation_corrects_position(self):
        R = rz(-10)
        expected_center = np.array([1.2, 0.1, 0.0])
        t = -(R @ expected_center)
        result = vio.correct_visual_inertial_segment(
            self.prior(yaw_deg=10, p=(1.0, 0, 0)),
            R,
            t,
            visual_metric_scale_paid=True,
            rotation_camera_from_imu=np.eye(3),
            translation_camera_from_imu_m=(0, 0, 0),
            camera_imu_extrinsic_source="synthetic_rig",
            clock_offset_s=0.002,
            clock_alignment_source="audio_sync",
            max_rotation_residual_deg=1.0,
            max_position_residual_m=0.3,
        )
        self.assertEqual(result.status, "candidate")
        self.assertTrue(result.visual_metric_position_applied)
        self.assertLess(result.position_residual_m, 0.25)
        self.assertTrue(
            np.allclose(result.camera_center_delta_in_prev_m, expected_center, atol=1e-8)
        )
        self.assertTrue(result.camera_imu_extrinsic_paid)
        self.assertTrue(result.clock_alignment_paid)

    def test_large_rotation_disagreement_abstains(self):
        result = vio.correct_visual_inertial_segment(
            self.prior(yaw_deg=20),
            rz(0),
            None,
            visual_metric_scale_paid=False,
            rotation_camera_from_imu=np.eye(3),
            translation_camera_from_imu_m=(0, 0, 0),
            camera_imu_extrinsic_source="synthetic_rig",
            clock_offset_s=0.0,
            clock_alignment_source="synthetic_sync",
            max_rotation_residual_deg=5.0,
        )
        self.assertEqual(result.status, "abstain")
        self.assertFalse(result.visual_rotation_applied)

    def test_candidate_segments_compose_into_time_indexed_local_trajectory(self):
        self.assertTrue(
            hasattr(vio, "compose_candidate_trajectory"),
            "trajectory composer missing",
        )
        seg1 = vio.VisualInertialSegmentCandidate(
            tuple(np.eye(3).reshape(-1)),
            (1, 0, 0),
            (1, 0, 0),
            0.1,
            None,
            True,
            False,
            True,
            True,
            1.0,
            "candidate",
        )
        seg2 = vio.VisualInertialSegmentCandidate(
            tuple(rz(-90).reshape(-1)),
            (1, 0, 0),
            (1, 0, 0),
            0.2,
            None,
            True,
            False,
            True,
            True,
            1.0,
            "candidate",
        )
        traj = vio.compose_candidate_trajectory([seg1, seg2])
        self.assertEqual(len(traj), 3)
        self.assertAlmostEqual(traj[-1].time_s, 2.0)
        self.assertTrue(np.allclose(traj[-1].position_world_m, (2, 0, 0), atol=1e-8))
        self.assertEqual(traj[-1].status, "candidate")


if __name__ == "__main__":
    unittest.main()
