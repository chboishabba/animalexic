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


@dataclass(frozen=True)
class VisualInertialSegmentCandidate:
    rotation_cam_next_from_cam_prev: tuple[float, ...]
    camera_center_delta_in_prev_m: tuple[float, float, float]
    velocity_delta_in_prev_m_s: tuple[float, float, float]
    rotation_residual_deg: float
    position_residual_m: float | None
    visual_rotation_applied: bool
    visual_metric_position_applied: bool
    camera_imu_extrinsic_paid: bool
    clock_alignment_paid: bool
    duration_s: float
    status: str = "candidate"


@dataclass(frozen=True)
class VisualInertialTrajectoryKeyframe:
    time_s: float
    rotation_world_from_camera: tuple[float, ...]
    position_world_m: tuple[float, float, float]
    velocity_world_m_s: tuple[float, float, float]
    status: str = "candidate"


def _so3_exp(omega):
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        K = np.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + K
    axis = omega / theta
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
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
    """Preintegrate IMU data as a candidate pose prior."""
    values = list(samples)
    if len(values) < 2:
        raise ValueError("at least two IMU samples are required")
    if any(
        values[i + 1].timestamp_s <= values[i].timestamp_s
        for i in range(len(values) - 1)
    ):
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


def correct_visual_inertial_segment(
    prior: InertialPosePrior,
    visual_rotation_cam_next_from_cam_prev,
    visual_translation_cam_next_from_cam_prev=None,
    *,
    visual_metric_scale_paid: bool,
    rotation_camera_from_imu=None,
    translation_camera_from_imu_m=None,
    camera_imu_extrinsic_source: str | None = None,
    clock_offset_s: float | None = None,
    clock_alignment_source: str | None = None,
    max_rotation_residual_deg: float = 5.0,
    max_position_residual_m: float | None = None,
):
    """Correct one inertial interval with a static-scene visual relative pose.

    The IMU prior carries a world-from-IMU delta while calibrated two-view
    recovery carries camera-next-from-camera-prev.  The camera<-IMU rigid
    transform is therefore mandatory before the two observers can be compared.

    Visual rotation replaces the inertial prediction only when the residual
    gate passes.  Visual translation replaces inertial metric position only
    when its metric scale has been paid.  A failed residual gate returns an
    ``abstain`` candidate rather than silently choosing one observer.
    """
    if prior.status != "candidate":
        raise ValueError("only candidate inertial priors can be corrected")
    if (
        not camera_imu_extrinsic_source
        or rotation_camera_from_imu is None
        or translation_camera_from_imu_m is None
    ):
        raise ValueError("camera/IMU extrinsic transform and source are required")
    if clock_offset_s is None or not clock_alignment_source:
        raise ValueError("clock offset and alignment source are required")
    if max_rotation_residual_deg < 0:
        raise ValueError("max_rotation_residual_deg must be non-negative")
    if max_position_residual_m is not None and max_position_residual_m < 0:
        raise ValueError("max_position_residual_m must be non-negative")

    R_imu = np.asarray(prior.rotation_delta, dtype=np.float64).reshape(3, 3)
    R_ci = np.asarray(rotation_camera_from_imu, dtype=np.float64).reshape(3, 3)
    t_ci = np.asarray(translation_camera_from_imu_m, dtype=np.float64).reshape(3)
    R_visual = np.asarray(
        visual_rotation_cam_next_from_cam_prev, dtype=np.float64
    ).reshape(3, 3)

    # Convert the world-from-IMU increment to the calibrated two-view camera
    # convention: camera-next-from-camera-prev.
    R_pred = R_ci @ R_imu.T @ R_ci.T
    rotation_residual = rotation_disagreement_deg(R_pred, R_visual)

    # Include the camera/IMU lever arm so non-coincident sensors do not silently
    # collapse to the same origin.
    camera_center_in_imu = -(R_ci.T @ t_ci)
    p_imu_prev = np.asarray(prior.position_delta_m, dtype=np.float64).reshape(3)
    camera_delta_prev = R_ci @ (
        p_imu_prev + R_imu @ camera_center_in_imu - camera_center_in_imu
    )
    velocity_prev = R_ci @ np.asarray(
        prior.velocity_delta_m_s, dtype=np.float64
    ).reshape(3)

    status = "candidate"
    apply_visual_rotation = rotation_residual <= max_rotation_residual_deg
    if not apply_visual_rotation:
        status = "abstain"

    corrected_delta = camera_delta_prev
    position_residual = None
    apply_visual_position = False
    if visual_translation_cam_next_from_cam_prev is not None:
        t_visual = np.asarray(
            visual_translation_cam_next_from_cam_prev, dtype=np.float64
        ).reshape(3)
        if visual_metric_scale_paid:
            visual_camera_delta = -(R_visual.T @ t_visual)
            position_residual = float(
                np.linalg.norm(camera_delta_prev - visual_camera_delta)
            )
            position_ok = (
                max_position_residual_m is None
                or position_residual <= max_position_residual_m
            )
            if position_ok and status == "candidate":
                corrected_delta = visual_camera_delta
                apply_visual_position = True
            elif not position_ok:
                status = "abstain"
        elif max_position_residual_m is not None:
            raise ValueError(
                "metric position residual requires metric-paid visual translation"
            )

    rotation_out = (
        R_visual if apply_visual_rotation and status == "candidate" else R_pred
    )
    return VisualInertialSegmentCandidate(
        rotation_cam_next_from_cam_prev=tuple(float(x) for x in rotation_out.reshape(-1)),
        camera_center_delta_in_prev_m=tuple(float(x) for x in corrected_delta),
        velocity_delta_in_prev_m_s=tuple(float(x) for x in velocity_prev),
        rotation_residual_deg=float(rotation_residual),
        position_residual_m=position_residual,
        visual_rotation_applied=bool(apply_visual_rotation and status == "candidate"),
        visual_metric_position_applied=bool(
            apply_visual_position and status == "candidate"
        ),
        camera_imu_extrinsic_paid=True,
        clock_alignment_paid=True,
        duration_s=float(prior.duration_s),
        status=status,
    )


def compose_candidate_trajectory(segments):
    """Compose accepted local visual-inertial segments into a candidate path.

    This is deterministic SE(3) accumulation only.  It does not perform bundle
    adjustment, smoothing, loop closure, online bias estimation, or promotion.
    """
    R_wc = np.eye(3, dtype=np.float64)
    p_w = np.zeros(3, dtype=np.float64)
    v_w = np.zeros(3, dtype=np.float64)
    time_s = 0.0
    out = [
        VisualInertialTrajectoryKeyframe(
            time_s=0.0,
            rotation_world_from_camera=tuple(float(x) for x in R_wc.reshape(-1)),
            position_world_m=(0.0, 0.0, 0.0),
            velocity_world_m_s=(0.0, 0.0, 0.0),
        )
    ]

    for segment in segments:
        if segment.status != "candidate":
            raise ValueError("trajectory composition requires candidate segments only")
        if segment.duration_s <= 0:
            raise ValueError("segment duration must be positive")

        R_next_prev = np.asarray(
            segment.rotation_cam_next_from_cam_prev, dtype=np.float64
        ).reshape(3, 3)
        delta_prev = np.asarray(
            segment.camera_center_delta_in_prev_m, dtype=np.float64
        ).reshape(3)
        vel_prev = np.asarray(
            segment.velocity_delta_in_prev_m_s, dtype=np.float64
        ).reshape(3)

        p_w = p_w + R_wc @ delta_prev
        v_w = R_wc @ vel_prev
        R_wc = R_wc @ R_next_prev.T
        time_s += float(segment.duration_s)
        out.append(
            VisualInertialTrajectoryKeyframe(
                time_s=time_s,
                rotation_world_from_camera=tuple(float(x) for x in R_wc.reshape(-1)),
                position_world_m=tuple(float(x) for x in p_w),
                velocity_world_m_s=tuple(float(x) for x in v_w),
                status="candidate",
            )
        )

    return out
