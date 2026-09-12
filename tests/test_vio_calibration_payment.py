import unittest

import numpy as np

from scripts.vio_calibration_candidates import (
    CalibrationAcceptanceReceipt,
    CameraIMURotationCandidate,
    ClockOffsetCandidate,
    GyroBiasCandidate,
    accept_camera_imu_rotation_candidate,
    accept_clock_offset_candidate,
    accept_gyro_bias_candidate,
)
from scripts.visual_inertial_pose import (
    IMUSample,
    correct_visual_inertial_segment,
    preintegrate_imu_prior,
)


class VIOCalibrationPaymentTests(unittest.TestCase):
    def test_candidate_requires_exact_acceptance_receipt(self):
        candidate = ClockOffsetCandidate(-0.2, 0.0, 1.0, 5, False, "candidate")
        receipt = CalibrationAcceptanceReceipt(
            coordinate="clock_offset",
            candidate_reference="clock-probe-17",
            accepted_by="operator",
            receipt_ref="clock-payment-17",
        )
        paid = accept_clock_offset_candidate(
            candidate, receipt, candidate_reference="clock-probe-17"
        )
        self.assertAlmostEqual(paid.offset_s, -0.2)
        self.assertEqual(paid.source_reference, "clock-payment-17")

        bad = CalibrationAcceptanceReceipt(
            coordinate="clock_offset",
            candidate_reference="other",
            accepted_by="operator",
            receipt_ref="bad",
        )
        with self.assertRaises(ValueError):
            accept_clock_offset_candidate(
                candidate, bad, candidate_reference="clock-probe-17"
            )

    def test_abstained_candidate_cannot_be_paid(self):
        candidate = ClockOffsetCandidate(0.0, 1.0, 0.0, 4, True, "abstain")
        receipt = CalibrationAcceptanceReceipt(
            "clock_offset", "ambiguous", "operator", "receipt"
        )
        with self.assertRaises(ValueError):
            accept_clock_offset_candidate(
                candidate, receipt, candidate_reference="ambiguous"
            )

    def test_paid_extrinsic_and_clock_feed_existing_correction_contract(self):
        extrinsic_candidate = CameraIMURotationCandidate(
            tuple(float(x) for x in np.eye(3).reshape(-1)),
            0.0,
            4,
            "candidate",
        )
        extrinsic = accept_camera_imu_rotation_candidate(
            extrinsic_candidate,
            translation_camera_from_imu_m=(0.0, 0.0, 0.0),
            receipt=CalibrationAcceptanceReceipt(
                "camera_imu_rotation", "extrinsic-1", "operator", "extrinsic-payment-1"
            ),
            candidate_reference="extrinsic-1",
        )
        clock = accept_clock_offset_candidate(
            ClockOffsetCandidate(0.0, 0.0, 1.0, 4, False, "candidate"),
            CalibrationAcceptanceReceipt(
                "clock_offset", "clock-1", "operator", "clock-payment-1"
            ),
            candidate_reference="clock-1",
        )
        prior = preintegrate_imu_prior(
            [
                IMUSample(0.0, (0, 0, 0), (0, 0, 9.80665)),
                IMUSample(1.0, (0, 0, 0), (0, 0, 9.80665)),
            ]
        )
        segment = correct_visual_inertial_segment(
            prior,
            np.eye(3),
            visual_metric_scale_paid=False,
            rotation_camera_from_imu=extrinsic.rotation_camera_from_imu,
            translation_camera_from_imu_m=extrinsic.translation_camera_from_imu_m,
            camera_imu_extrinsic_source=extrinsic.source_reference,
            clock_offset_s=clock.offset_s,
            clock_alignment_source=clock.source_reference,
        )
        self.assertEqual(segment.status, "candidate")
        self.assertTrue(segment.camera_imu_extrinsic_paid)
        self.assertTrue(segment.clock_alignment_paid)

    def test_paid_gyro_bias_is_still_not_online_bias_optimization(self):
        candidate = GyroBiasCandidate(
            (0.01, -0.02, 0.005),
            (0.001, 0.001, 0.001),
            10,
            False,
            "candidate",
        )
        paid = accept_gyro_bias_candidate(
            candidate,
            CalibrationAcceptanceReceipt(
                "gyro_bias", "bias-1", "operator", "bias-payment-1"
            ),
            candidate_reference="bias-1",
        )
        self.assertEqual(paid.bias_rad_s, (0.01, -0.02, 0.005))
        self.assertFalse(paid.online_bias_optimization_paid)


if __name__ == "__main__":
    unittest.main()
