from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IMUSample:
    timestamp_s: float
    gyro_rad_s: tuple[float, float, float]
    specific_force_m_s2: tuple[float, float, float]


@dataclass(frozen=True)
class InertialPosePrior:
    rotation_delta: tuple[float, ...]
    velocity_delta_m_s: tuple[float, float, float]
    position_delta_m: tuple[float, float, float]
    duration_s: float
    status: str = "candidate"


def _so3_exp(omega):
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        K = np.array(
            [[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3) + K
    axis = omega / theta
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def preintegrate_imu_prior(
    samples,
    *,
    gravity_world=(0.0, 0.0, -9.80665),
    gyro_bias_rad_s=(0.0, 0.0, 0.0),
    accel_bias_m_s2=(0.0, 0.0, 0.0),
):
    """Preintegrate IMU data as a candidate pose prior.

    This is intentionally not a full VIO optimizer.  The returned metric deltas
    remain an inertial prior and require visual consistency / bias / timing
    checks before they can contribute to a promoted camera trajectory.
    """
    values = list(samples)
    if len(values) < 2:
        raise ValueError("at least two IMU samples are required")
    if any(values[i + 1].timestamp_s <= values[i].timestamp_s for i in range(len(values) - 1)):
        raise ValueError("IMU timestamps must be strictly increasing")

    R = np.eye(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    p = np.zeros(3, dtype=np.float64)
    g = np.asarray(gravity_world, dtype=np.float64)
    gb = np.asarray(gyro_bias_rad_s, dtype=np.float64)
    ab = np.asarray(accel_bias_m_s2, dtype=np.float64)

    for a, b in zip(values, values[1:]):
        dt = float(b.timestamp_s - a.timestamp_s)
        omega = np.asarray(a.gyro_rad_s, dtype=np.float64) - gb
        specific_force = np.asarray(a.specific_force_m_s2, dtype=np.float64) - ab
        acceleration_world = R @ specific_force + g
        p = p + v * dt + 0.5 * acceleration_world * dt * dt
        v = v + acceleration_world * dt
        R = R @ _so3_exp(omega * dt)

    return InertialPosePrior(
        rotation_delta=tuple(float(x) for x in R.reshape(-1)),
        velocity_delta_m_s=tuple(float(x) for x in v),
        position_delta_m=tuple(float(x) for x in p),
        duration_s=float(values[-1].timestamp_s - values[0].timestamp_s),
    )


def rotation_disagreement_deg(rotation_a, rotation_b):
    A = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)
    B = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
    delta = A @ B.T
    return math.degrees(
        math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
    )


def visual_inertial_rotation_consistent(
    imu_rotation,
    visual_rotation,
    *,
    max_error_deg: float,
):
    """Return a consistency gate only; this function performs no promotion."""
    if max_error_deg < 0:
        raise ValueError("max_error_deg must be non-negative")
    return rotation_disagreement_deg(imu_rotation, visual_rotation) <= max_error_deg
