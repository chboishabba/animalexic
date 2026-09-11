"""Governed camera-pose fibre for fixed multicam and handheld Regime B.

Pose is treated as a time-indexed candidate with explicit evidence and debt.
Dynamic target pixels are not admissible pose evidence by default; IMU is a
prior, not an authoritative pose; metric scale, clock alignment, and rolling
shutter remain explicit coordinates until paid.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping


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
