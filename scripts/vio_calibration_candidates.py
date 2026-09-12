from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TimedVectorSample:
    time_s: float
    value: tuple[float, ...]


@dataclass(frozen=True)
class ClockOffsetCandidate:
    offset_s: float
    rms_residual: float
    residual_margin: float
    overlap_count: int
    ambiguous: bool
    status: str
    clock_alignment_paid: bool = False


@dataclass(frozen=True)
class GyroBiasCandidate:
    bias_rad_s: tuple[float, float, float]
    axis_std_rad_s: tuple[float, float, float]
    sample_count: int
    noisy: bool
    status: str
    online_bias_paid: bool = False


@dataclass(frozen=True)
class CameraIMURotationCandidate:
    rotation_camera_from_imu: tuple[float, ...]
    rms_vector_residual: float
    pair_count: int
    status: str
    extrinsic_calibration_paid: bool = False


@dataclass(frozen=True)
class HandEyeMotionPair:
    rotation_camera_motion: tuple[float, ...]
    translation_camera_motion_m: tuple[float, float, float]
    rotation_imu_motion: tuple[float, ...]
    translation_imu_motion_m: tuple[float, float, float]


@dataclass(frozen=True)
class CameraIMUTranslationCandidate:
    translation_camera_from_imu_m: tuple[float, float, float]
    rms_translation_residual_m: float
    motion_pair_count: int
    linear_rank: int
    status: str
    lever_arm_paid: bool = False


@dataclass(frozen=True)
class CalibrationAcceptanceReceipt:
    coordinate: str
    candidate_reference: str
    accepted_by: str
    receipt_ref: str


@dataclass(frozen=True)
class PaidClockAlignment:
    offset_s: float
    source_reference: str


@dataclass(frozen=True)
class PaidCameraIMUExtrinsic:
    rotation_camera_from_imu: tuple[float, ...]
    translation_camera_from_imu_m: tuple[float, float, float]
    source_reference: str


@dataclass(frozen=True)
class PaidGyroBias:
    bias_rad_s: tuple[float, float, float]
    source_reference: str
    online_bias_optimization_paid: bool = False


def _validated_series(samples):
    values = list(samples)
    if len(values) < 2:
        raise ValueError("at least two timed vector samples are required")
    times = np.asarray([float(sample.time_s) for sample in values], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("sample timestamps must be finite and strictly increasing")
    vectors = [np.asarray(sample.value, dtype=np.float64) for sample in values]
    if any(vector.ndim != 1 or vector.size == 0 for vector in vectors):
        raise ValueError("sample vectors must be non-empty one-dimensional values")
    width = vectors[0].shape
    if any(vector.shape != width for vector in vectors):
        raise ValueError("sample vectors must have equal dimension")
    matrix = np.vstack(vectors)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sample vectors must be finite")
    return times, matrix


def _interpolate_matrix(source_times, source_values, query_times):
    columns = [
        np.interp(query_times, source_times, source_values[:, column])
        for column in range(source_values.shape[1])
    ]
    return np.stack(columns, axis=1)


def estimate_clock_offset_candidate(
    reference_samples,
    sensor_samples,
    *,
    search_offsets_s,
    min_overlap: int,
    min_residual_margin: float,
    max_rms_residual: float | None = None,
) -> ClockOffsetCandidate:
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least two")
    if min_residual_margin < 0 or not math.isfinite(min_residual_margin):
        raise ValueError("min_residual_margin must be finite and non-negative")
    if max_rms_residual is not None and (
        max_rms_residual < 0 or not math.isfinite(max_rms_residual)
    ):
        raise ValueError("max_rms_residual must be finite and non-negative")
    reference_times, reference_values = _validated_series(reference_samples)
    sensor_times, sensor_values = _validated_series(sensor_samples)
    offsets = [float(offset) for offset in search_offsets_s]
    if not offsets or not all(math.isfinite(offset) for offset in offsets):
        raise ValueError("search_offsets_s must contain finite values")
    scored = []
    for offset in offsets:
        aligned_sensor_times = sensor_times + offset
        start = max(reference_times[0], aligned_sensor_times[0])
        stop = min(reference_times[-1], aligned_sensor_times[-1])
        mask = (reference_times >= start) & (reference_times <= stop)
        query_times = reference_times[mask]
        if len(query_times) < min_overlap:
            continue
        interpolated = _interpolate_matrix(aligned_sensor_times, sensor_values, query_times)
        residual = reference_values[mask] - interpolated
        rms = float(np.sqrt(np.mean(np.square(residual))))
        scored.append((rms, offset, int(len(query_times))))
    if not scored:
        raise ValueError("no candidate clock offset has sufficient overlap")
    scored.sort(key=lambda row: (row[0], abs(row[1]), row[1]))
    best_rms, best_offset, best_overlap = scored[0]
    second_rms = scored[1][0] if len(scored) > 1 else math.inf
    margin = float(second_rms - best_rms)
    ambiguous = margin < min_residual_margin
    if max_rms_residual is not None and best_rms > max_rms_residual:
        ambiguous = True
    return ClockOffsetCandidate(
        offset_s=float(best_offset),
        rms_residual=float(best_rms),
        residual_margin=margin,
        overlap_count=best_overlap,
        ambiguous=bool(ambiguous),
        status="abstain" if ambiguous else "candidate",
        clock_alignment_paid=False,
    )


def estimate_gyro_bias_candidate(
    stationary_gyro_samples,
    *,
    min_samples: int,
    max_axis_std_rad_s: float,
) -> GyroBiasCandidate:
    if min_samples < 2:
        raise ValueError("min_samples must be at least two")
    if max_axis_std_rad_s < 0 or not math.isfinite(max_axis_std_rad_s):
        raise ValueError("max_axis_std_rad_s must be finite and non-negative")
    _, values = _validated_series(stationary_gyro_samples)
    if values.shape[1] != 3:
        raise ValueError("gyro samples must be 3-vectors")
    if len(values) < min_samples:
        raise ValueError("not enough stationary gyro samples")
    bias = np.mean(values, axis=0)
    axis_std = np.std(values, axis=0)
    noisy = bool(np.any(axis_std > max_axis_std_rad_s))
    return GyroBiasCandidate(
        bias_rad_s=tuple(float(x) for x in bias),
        axis_std_rad_s=tuple(float(x) for x in axis_std),
        sample_count=int(len(values)),
        noisy=noisy,
        status="abstain" if noisy else "candidate",
        online_bias_paid=False,
    )


def estimate_camera_imu_rotation_candidate(
    imu_vectors,
    camera_vectors,
    *,
    max_rms_vector_residual: float,
) -> CameraIMURotationCandidate:
    if max_rms_vector_residual < 0 or not math.isfinite(max_rms_vector_residual):
        raise ValueError("max_rms_vector_residual must be finite and non-negative")
    imu = np.asarray(imu_vectors, dtype=np.float64)
    camera = np.asarray(camera_vectors, dtype=np.float64)
    if imu.ndim != 2 or imu.shape[1:] != (3,) or camera.shape != imu.shape:
        raise ValueError("paired IMU/camera vectors must be matching Nx3 arrays")
    if len(imu) < 3 or not np.all(np.isfinite(imu)) or not np.all(np.isfinite(camera)):
        raise ValueError("at least three finite vector pairs are required")
    if np.linalg.matrix_rank(imu, tol=1e-10) < 2:
        raise ValueError("camera/IMU rotation calibration lacks non-collinear excitation")
    covariance = camera.T @ imu
    U, _, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0
    rotation = U @ correction @ Vt
    predicted = (rotation @ imu.T).T
    residual = predicted - camera
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    abstain = rms > max_rms_vector_residual
    return CameraIMURotationCandidate(
        rotation_camera_from_imu=tuple(float(x) for x in rotation.reshape(-1)),
        rms_vector_residual=rms,
        pair_count=int(len(imu)),
        status="abstain" if abstain else "candidate",
        extrinsic_calibration_paid=False,
    )


def estimate_camera_imu_translation_candidate(
    motion_pairs,
    *,
    rotation_camera_from_imu,
    max_rms_translation_residual_m: float,
) -> CameraIMUTranslationCandidate:
    """Estimate hand-eye lever arm t_X from A X = X B with known R_X.

    For each paired relative motion, translation obeys
    (R_A - I) t_X = R_X t_B - t_A.  The stacked system must have full rank;
    pure translations or single-axis-poor excitation therefore fail closed.
    """
    if max_rms_translation_residual_m < 0 or not math.isfinite(max_rms_translation_residual_m):
        raise ValueError("max_rms_translation_residual_m must be finite and non-negative")
    pairs = list(motion_pairs)
    if len(pairs) < 2:
        raise ValueError("at least two paired hand-eye motions are required")
    R_x = np.asarray(rotation_camera_from_imu, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(R_x)):
        raise ValueError("camera/IMU rotation must be finite")
    lhs = []
    rhs = []
    for pair in pairs:
        R_a = np.asarray(pair.rotation_camera_motion, dtype=np.float64).reshape(3, 3)
        R_b = np.asarray(pair.rotation_imu_motion, dtype=np.float64).reshape(3, 3)
        t_a = np.asarray(pair.translation_camera_motion_m, dtype=np.float64).reshape(3)
        t_b = np.asarray(pair.translation_imu_motion_m, dtype=np.float64).reshape(3)
        if not all(np.all(np.isfinite(x)) for x in (R_a, R_b, t_a, t_b)):
            raise ValueError("hand-eye motion coordinates must be finite")
        # Rotation consistency is a prerequisite for using the translation equation.
        if not np.allclose(R_a @ R_x, R_x @ R_b, atol=1e-5, rtol=1e-5):
            raise ValueError("hand-eye motion pair violates supplied rotation extrinsic")
        lhs.append(R_a - np.eye(3))
        rhs.append(R_x @ t_b - t_a)
    A = np.vstack(lhs)
    b = np.concatenate(rhs)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    if rank < 3:
        raise ValueError("hand-eye lever-arm system lacks full rotational excitation")
    t_x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual = A @ t_x - b
    rms = float(np.sqrt(np.mean(np.square(residual))))
    abstain = rms > max_rms_translation_residual_m
    return CameraIMUTranslationCandidate(
        translation_camera_from_imu_m=tuple(float(x) for x in t_x),
        rms_translation_residual_m=rms,
        motion_pair_count=len(pairs),
        linear_rank=rank,
        status="abstain" if abstain else "candidate",
        lever_arm_paid=False,
    )


def _validate_acceptance(receipt, *, coordinate: str, candidate_reference: str) -> None:
    if receipt.coordinate != coordinate:
        raise ValueError("acceptance receipt coordinate does not match candidate")
    if receipt.candidate_reference != candidate_reference:
        raise ValueError("acceptance receipt candidate reference does not match")
    if not receipt.accepted_by or not receipt.receipt_ref or not candidate_reference:
        raise ValueError("acceptance receipt requires actor, receipt ref, and candidate ref")


def accept_clock_offset_candidate(candidate, receipt, *, candidate_reference: str) -> PaidClockAlignment:
    if candidate.status != "candidate" or candidate.ambiguous:
        raise ValueError("only unambiguous clock candidates may be paid")
    _validate_acceptance(receipt, coordinate="clock_offset", candidate_reference=candidate_reference)
    return PaidClockAlignment(float(candidate.offset_s), receipt.receipt_ref)


def accept_camera_imu_rotation_candidate(
    candidate,
    *,
    translation_camera_from_imu_m,
    receipt,
    candidate_reference: str,
) -> PaidCameraIMUExtrinsic:
    if candidate.status != "candidate":
        raise ValueError("only candidate camera/IMU rotations may be paid")
    _validate_acceptance(receipt, coordinate="camera_imu_rotation", candidate_reference=candidate_reference)
    translation = np.asarray(translation_camera_from_imu_m, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("camera/IMU lever-arm translation must be finite xyz")
    return PaidCameraIMUExtrinsic(
        rotation_camera_from_imu=tuple(candidate.rotation_camera_from_imu),
        translation_camera_from_imu_m=tuple(float(x) for x in translation),
        source_reference=receipt.receipt_ref,
    )


def accept_gyro_bias_candidate(candidate, receipt, *, candidate_reference: str) -> PaidGyroBias:
    if candidate.status != "candidate" or candidate.noisy:
        raise ValueError("only non-noisy gyro bias candidates may be paid")
    _validate_acceptance(receipt, coordinate="gyro_bias", candidate_reference=candidate_reference)
    return PaidGyroBias(
        bias_rad_s=tuple(candidate.bias_rad_s),
        source_reference=receipt.receipt_ref,
        online_bias_optimization_paid=False,
    )
