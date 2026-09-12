from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from scripts.vio_calibration_candidates import TimedVectorSample


@dataclass(frozen=True)
class AccelBiasCandidate:
    bias_m_s2: tuple[float, float, float]
    axis_std_m_s2: tuple[float, float, float]
    sample_count: int
    noisy: bool
    gravity_reference_required: bool
    status: str
    online_bias_paid: bool = False


def estimate_accel_bias_candidate(
    stationary_accel_samples,
    *,
    rotation_world_from_imu,
    gravity_world_m_s2,
    min_samples: int,
    max_axis_std_m_s2: float,
) -> AccelBiasCandidate:
    """Estimate a stationary accelerometer-bias candidate.

    At rest the unbiased specific force in IMU coordinates is ``-R^T g``.
    Gravity and orientation are therefore explicit inputs; the estimator does
    not infer them or silently fold gravity error into accelerometer bias.
    """
    values = list(stationary_accel_samples)
    if min_samples < 2 or len(values) < min_samples:
        raise ValueError("not enough stationary accelerometer samples")
    if max_axis_std_m_s2 < 0 or not math.isfinite(max_axis_std_m_s2):
        raise ValueError("max_axis_std_m_s2 must be finite and non-negative")

    times = np.asarray([float(sample.time_s) for sample in values], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("accelerometer timestamps must be finite and strictly increasing")
    measured = np.asarray([sample.value for sample in values], dtype=np.float64)
    if measured.shape != (len(values), 3) or not np.all(np.isfinite(measured)):
        raise ValueError("stationary accelerometer samples must be finite 3-vectors")

    R = np.asarray(rotation_world_from_imu, dtype=np.float64)
    gravity = np.asarray(gravity_world_m_s2, dtype=np.float64)
    if R.shape != (3, 3) or gravity.shape != (3,):
        raise ValueError("orientation must be 3x3 and gravity must be xyz")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(gravity)):
        raise ValueError("orientation and gravity must be finite")

    expected_specific_force = -(R.T @ gravity)
    bias_samples = measured - expected_specific_force[None, :]
    bias = np.mean(bias_samples, axis=0)
    axis_std = np.std(bias_samples, axis=0)
    noisy = bool(np.any(axis_std > max_axis_std_m_s2))
    return AccelBiasCandidate(
        bias_m_s2=tuple(float(x) for x in bias),
        axis_std_m_s2=tuple(float(x) for x in axis_std),
        sample_count=len(values),
        noisy=noisy,
        gravity_reference_required=True,
        status="abstain" if noisy else "candidate",
        online_bias_paid=False,
    )
