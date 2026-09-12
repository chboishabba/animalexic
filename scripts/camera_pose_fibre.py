"""Governed camera-pose fibre for fixed multicam and handheld Regime B.

Pose is treated as a time-indexed candidate with explicit evidence and debt.
Dynamic target pixels are not admissible pose evidence by default; IMU is a
prior, not an authoritative pose; metric scale, clock alignment, and rolling
shutter remain explicit coordinates until paid.

The relative-pose producer in this module is deliberately narrower than a full
VIO/SLAM backend.  It recovers a calibrated two-view pose from static-scene
correspondences, keeps monocular scale debt explicit, and can materialise a
metric-paid estimate into the same CameraObservation contract used by the
known-pose Issue-20 path.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PoseEvidence:
    camera_id: int
    frame_index: int
    has_visual_support: bool
    has_imu_prior: bool
    metric_scale_paid: bool
    visual_track_count: int
    visual_reprojection_rms: float
    imu_rotation_sigma_deg: float
    clock_offset_sigma_ms: float
    rolling_shutter_sigma_ms: float
    metric_scale_source: str | None
    status: str = "candidate"


@dataclass(frozen=True)
class PoseCandidate:
    camera_id: str
    frame_index: int
    confidence: float
    has_visual_support: bool
    has_imu_prior: bool
    metric_scale_paid: bool
    clock_offset_sigma_ms: float
    rolling_shutter_sigma_ms: float
    status: str = "candidate"


@dataclass(frozen=True)
class PoseSufficiencyPolicy:
    min_confidence: float
    require_metric_scale: bool
    max_clock_offset_sigma_ms: float
    max_rolling_shutter_sigma_ms: float


@dataclass(frozen=True)
class RelativePoseEstimate:
    """Candidate relative pose from calibrated static-scene correspondences.

    ``rotation_cam2_from_cam1`` and ``translation_cam2_from_cam1`` follow the
    standard calibrated two-view convention::

        X_cam2 = R * X_cam1 + t

    Without an external metric scale receipt, ``translation_cam2_from_cam1``
    is unit-norm and therefore carries direction only.
    """

    rotation_cam2_from_cam1: tuple[float, ...]
    translation_cam2_from_cam1: tuple[float, float, float]
    translation_direction_cam2_from_cam1: tuple[float, float, float]
    static_correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    metric_scale_paid: bool
    metric_scale_source: str | None
    status: str = "candidate"


def static_pose_evidence(samples: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Keep only scene evidence marked static for camera-pose estimation."""
    return [sample for sample in samples if bool(sample.get("static", False))]


def dynamic_pose_evidence(
    *,
    camera_id: int,
    frame_index: int,
    visual_track_count: int,
    visual_reprojection_rms: float,
    imu_rotation_sigma_deg: float,
    metric_scale_source: str | None,
    clock_offset_sigma_ms: float,
    rolling_shutter_sigma_ms: float,
) -> PoseEvidence:
    """Construct one time-indexed pose-evidence receipt.

    A real VIO/SfM backend may populate this record later.  This layer only
    carries the governed evidence boundary and does not implement SLAM itself.
    """
    if visual_track_count < 0:
        raise ValueError("visual_track_count must be non-negative")
    for name, value in (
        ("visual_reprojection_rms", visual_reprojection_rms),
        ("imu_rotation_sigma_deg", imu_rotation_sigma_deg),
        ("clock_offset_sigma_ms", clock_offset_sigma_ms),
        ("rolling_shutter_sigma_ms", rolling_shutter_sigma_ms),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    return PoseEvidence(
        camera_id=camera_id,
        frame_index=frame_index,
        has_visual_support=visual_track_count > 0,
        has_imu_prior=imu_rotation_sigma_deg >= 0.0,
        metric_scale_paid=bool(metric_scale_source),
        visual_track_count=visual_track_count,
        visual_reprojection_rms=visual_reprojection_rms,
        imu_rotation_sigma_deg=imu_rotation_sigma_deg,
        clock_offset_sigma_ms=clock_offset_sigma_ms,
        rolling_shutter_sigma_ms=rolling_shutter_sigma_ms,
        metric_scale_source=metric_scale_source,
    )


def fuse_pose_candidates(candidates: Iterable[PoseCandidate]) -> PoseCandidate:
    """Fuse compatible candidate receipts without promoting them.

    This is an evidence aggregation seam, not bundle adjustment.  Backends may
    replace the numerical rule while preserving the candidate-only contract.
    """
    values = list(candidates)
    if not values:
        raise ValueError("at least one pose candidate is required")
    first = values[0]
    if any(v.camera_id != first.camera_id or v.frame_index != first.frame_index for v in values):
        raise ValueError("pose candidates must share camera_id and frame_index")
    return PoseCandidate(
        camera_id=first.camera_id,
        frame_index=first.frame_index,
        confidence=mean(v.confidence for v in values),
        has_visual_support=any(v.has_visual_support for v in values),
        has_imu_prior=any(v.has_imu_prior for v in values),
        metric_scale_paid=any(v.metric_scale_paid for v in values),
        clock_offset_sigma_ms=min(v.clock_offset_sigma_ms for v in values),
        rolling_shutter_sigma_ms=min(v.rolling_shutter_sigma_ms for v in values),
        status="candidate",
    )


def pose_sufficient_for_consumer(
    pose: PoseCandidate,
    policy: PoseSufficiencyPolicy,
) -> bool:
    """Consumer-indexed adequacy gate; sufficiency is not global pose truth."""
    if pose.status != "candidate":
        return False
    if pose.confidence < policy.min_confidence:
        return False
    if not pose.has_visual_support:
        return False
    if policy.require_metric_scale and not pose.metric_scale_paid:
        return False
    if pose.clock_offset_sigma_ms > policy.max_clock_offset_sigma_ms:
        return False
    if pose.rolling_shutter_sigma_ms > policy.max_rolling_shutter_sigma_ms:
        return False
    return True


def recover_relative_pose_from_correspondences(
    points_cam1: Sequence[Sequence[float]],
    points_cam2: Sequence[Sequence[float]],
    camera_matrix: Sequence[Sequence[float]],
    *,
    static_mask: Sequence[bool] | None = None,
    metric_baseline: float | None = None,
    metric_scale_source: str | None = None,
    ransac_threshold_px: float = 1.0,
    confidence: float = 0.999,
) -> RelativePoseEstimate:
    """Recover a calibrated relative pose from static-scene point matches.

    Dynamic/object tracks must be removed through ``static_mask`` before the
    essential-matrix fit.  Translation remains scale-free unless both a
    positive ``metric_baseline`` and an explicit ``metric_scale_source`` are
    supplied.  The function returns a candidate receipt; it does not promote
    camera geometry.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("relative pose recovery requires numpy and OpenCV (cv2)") from exc

    p1 = np.asarray(points_cam1, dtype=np.float64)
    p2 = np.asarray(points_cam2, dtype=np.float64)
    K = np.asarray(camera_matrix, dtype=np.float64)
    if p1.ndim != 2 or p1.shape[1:] != (2,) or p2.shape != p1.shape:
        raise ValueError("point correspondences must be matching Nx2 arrays")
    if K.shape != (3, 3):
        raise ValueError("camera_matrix must be 3x3")
    if ransac_threshold_px <= 0:
        raise ValueError("ransac_threshold_px must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between zero and one")

    if static_mask is None:
        keep = np.ones(len(p1), dtype=bool)
    else:
        keep = np.asarray(static_mask, dtype=bool)
        if keep.shape != (len(p1),):
            raise ValueError("static_mask must contain one value per correspondence")
    p1_static = p1[keep]
    p2_static = p2[keep]
    if len(p1_static) < 5:
        raise ValueError("at least five static correspondences are required")

    if (metric_baseline is None) != (metric_scale_source is None):
        raise ValueError("metric_baseline and metric_scale_source must be supplied together")
    if metric_baseline is not None and metric_baseline <= 0:
        raise ValueError("metric_baseline must be positive")
    if metric_scale_source is not None and not metric_scale_source.strip():
        raise ValueError("metric_scale_source must be non-empty")

    essential, fit_mask = cv2.findEssentialMat(
        p1_static,
        p2_static,
        K,
        method=cv2.RANSAC,
        prob=float(confidence),
        threshold=float(ransac_threshold_px),
    )
    if essential is None or fit_mask is None:
        raise ValueError("essential-matrix recovery failed")
    essential = np.asarray(essential, dtype=np.float64)
    if essential.shape[0] > 3:
        essential = essential[:3, :3]
    if essential.shape != (3, 3):
        raise ValueError("essential-matrix recovery returned an invalid shape")

    inlier_count, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        p1_static,
        p2_static,
        K,
        mask=fit_mask,
    )
    if int(inlier_count) < 1:
        raise ValueError("relative-pose recovery produced no inliers")

    direction = np.asarray(translation, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("relative-pose recovery produced zero translation")
    direction /= norm
    scale = float(metric_baseline) if metric_baseline is not None else 1.0
    scaled_translation = direction * scale
    pose_inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else int(inlier_count)
    pose_inliers = min(pose_inliers, len(p1_static))

    return RelativePoseEstimate(
        rotation_cam2_from_cam1=tuple(float(x) for x in np.asarray(rotation).reshape(-1)),
        translation_cam2_from_cam1=tuple(float(x) for x in scaled_translation),
        translation_direction_cam2_from_cam1=tuple(float(x) for x in direction),
        static_correspondence_count=int(len(p1_static)),
        inlier_count=pose_inliers,
        inlier_ratio=float(pose_inliers) / float(len(p1_static)),
        metric_scale_paid=metric_baseline is not None,
        metric_scale_source=metric_scale_source,
        status="candidate",
    )


def _euler_zyx_from_world_rotation(rotation_world_from_camera):
    """Return source-convention yaw(Z), pitch(X), roll(Y) in degrees."""
    import math
    import numpy as np

    R = np.asarray(rotation_world_from_camera, dtype=np.float64).reshape(3, 3)
    roll_y = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    cos_roll = math.cos(roll_y)
    if abs(cos_roll) > 1e-9:
        pitch_x = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw_z = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        pitch_x = 0.0
        yaw_z = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return math.degrees(yaw_z), math.degrees(pitch_x), math.degrees(roll_y)


def materialise_recovered_camera_observation(
    estimate: RelativePoseEstimate,
    *,
    camera_id: int,
    frame_index: int,
    fov_deg: float,
    image_file: str,
    anchor_position: Sequence[float] = (0.0, 0.0, 0.0),
    anchor_rotation_world_from_camera: Sequence[Sequence[float]] | None = None,
):
    """Adapt a metric-paid relative pose into Issue-20's camera contract.

    This is the parity seam between learned pose and the known-pose ray/voxel
    producer.  Scale-free estimates are intentionally rejected because their
    translation cannot yet share the metric world grid.
    """
    if estimate.status != "candidate":
        raise ValueError("only candidate relative poses can be materialised")
    if not estimate.metric_scale_paid:
        raise ValueError("metric scale must be paid before world-camera materialisation")
    if not (0.0 < fov_deg < 180.0):
        raise ValueError("fov_deg must lie between zero and 180 degrees")
    if not image_file:
        raise ValueError("image_file must be non-empty")

    import numpy as np
    from scripts.issue20_multiview import CameraObservation

    R21 = np.asarray(estimate.rotation_cam2_from_cam1, dtype=np.float64).reshape(3, 3)
    t21 = np.asarray(estimate.translation_cam2_from_cam1, dtype=np.float64).reshape(3)
    anchor_pos = np.asarray(anchor_position, dtype=np.float64).reshape(3)
    if anchor_rotation_world_from_camera is None:
        A = np.eye(3, dtype=np.float64)
    else:
        A = np.asarray(anchor_rotation_world_from_camera, dtype=np.float64).reshape(3, 3)

    camera2_center_in_cam1 = -R21.T @ t21
    position_world = anchor_pos + A @ camera2_center_in_cam1
    rotation_world_from_cam2 = A @ R21.T
    yaw_deg, pitch_deg, roll_deg = _euler_zyx_from_world_rotation(rotation_world_from_cam2)

    return CameraObservation(
        camera_id=int(camera_id),
        frame_index=int(frame_index),
        position=tuple(float(x) for x in position_world),
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
        fov_deg=float(fov_deg),
        image_file=str(image_file),
        pose_source="image_recovered_relative_pose",
    )
