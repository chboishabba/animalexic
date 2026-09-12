from __future__ import annotations

import math

import numpy as np

from scripts.visual_inertial_pose import VisualInertialTrajectoryKeyframe


def compose_pose_graph_trajectory(
    translation_candidate,
    rotation_candidate,
    *,
    times_s,
    velocities_world_m_s=None,
):
    """Re-enter separate pose-graph candidates into the existing trajectory carrier.

    This is a representation adapter only.  Translation and rotation were not
    jointly optimized, and zero/default velocities do not imply inertial state
    optimization or promotion.
    """
    if translation_candidate.status != "candidate":
        raise ValueError("translation pose graph must be candidate")
    if rotation_candidate.status != "candidate":
        raise ValueError("rotation pose graph must be candidate")

    positions = list(translation_candidate.positions_world_m)
    rotations = list(rotation_candidate.rotations_world_from_frame)
    times = [float(value) for value in times_s]
    if len(positions) != len(rotations) or len(times) != len(positions):
        raise ValueError("translation, rotation, and time carriers must have equal length")
    if not times or not all(math.isfinite(value) for value in times):
        raise ValueError("trajectory times must be finite")
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ValueError("trajectory times must be strictly increasing")

    if velocities_world_m_s is None:
        velocities = [(0.0, 0.0, 0.0) for _ in positions]
    else:
        velocities = list(velocities_world_m_s)
        if len(velocities) != len(positions):
            raise ValueError("velocity carrier length must match trajectory")

    out = []
    for time_s, position, rotation, velocity in zip(times, positions, rotations, velocities):
        p = np.asarray(position, dtype=np.float64)
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        v = np.asarray(velocity, dtype=np.float64)
        if p.shape != (3,) or v.shape != (3,):
            raise ValueError("position and velocity must be xyz")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(v)):
            raise ValueError("trajectory coordinates must be finite")
        out.append(
            VisualInertialTrajectoryKeyframe(
                time_s=time_s,
                rotation_world_from_camera=tuple(float(x) for x in R.reshape(-1)),
                position_world_m=tuple(float(x) for x in p),
                velocity_world_m_s=tuple(float(x) for x in v),
                status="candidate",
            )
        )
    return out
