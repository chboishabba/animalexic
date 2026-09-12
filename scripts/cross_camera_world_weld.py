from dataclasses import dataclass

import numpy as np

from scripts.visual_inertial_pose import VisualInertialTrajectoryKeyframe


@dataclass(frozen=True)
class WorldWeldCandidate:
    rotation_target_from_source: tuple[float, ...]
    translation_target_from_source_m: tuple[float, float, float]
    scale: float
    rms_residual_m: float
    metric_scale_preserved: bool
    anchor_count: int
    status: str = "candidate"


def estimate_world_weld(
    source_points_m,
    target_points_m,
    *,
    allow_scale: bool = False,
):
    """Estimate an SE(3) or Sim(3) candidate weld from shared static anchors.

    ``allow_scale=False`` preserves a metric local map and estimates only a
    rigid transform.  ``allow_scale=True`` exposes a similarity scale coordinate
    rather than silently forcing a scale-uncertain map into the metric world.
    """
    src = np.asarray(source_points_m, dtype=np.float64)
    dst = np.asarray(target_points_m, dtype=np.float64)
    if src.ndim != 2 or src.shape[1:] != (3,) or dst.shape != src.shape:
        raise ValueError("source and target anchors must be matching Nx3 arrays")
    if len(src) < 3:
        raise ValueError("at least three shared anchors are required")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("anchor coordinates must be finite")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    if np.linalg.matrix_rank(X, tol=1e-10) < 2:
        raise ValueError("shared anchors are geometrically degenerate")

    covariance = (Y.T @ X) / float(len(src))
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0
    R = U @ correction @ Vt

    scale = 1.0
    if allow_scale:
        variance = float(np.sum(X * X) / float(len(src)))
        if variance <= 1e-15:
            raise ValueError("source anchor variance is too small")
        scale = float(
            np.sum(singular_values * np.diag(correction)) / variance
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("similarity weld produced invalid scale")

    t = dst_mean - scale * (R @ src_mean)
    predicted = (scale * (R @ src.T)).T + t
    residuals = np.linalg.norm(predicted - dst, axis=1)
    rms = float(np.sqrt(np.mean(residuals * residuals)))

    return WorldWeldCandidate(
        rotation_target_from_source=tuple(float(x) for x in R.reshape(-1)),
        translation_target_from_source_m=tuple(float(x) for x in t),
        scale=float(scale),
        rms_residual_m=rms,
        metric_scale_preserved=not allow_scale,
        anchor_count=int(len(src)),
        status="candidate",
    )


def apply_world_weld_to_trajectory(
    trajectory,
    weld: WorldWeldCandidate,
):
    """Transport candidate local keyframes into the target shared world."""
    if weld.status != "candidate":
        raise ValueError("only candidate world welds may transform a trajectory")
    R = np.asarray(
        weld.rotation_target_from_source, dtype=np.float64
    ).reshape(3, 3)
    t = np.asarray(
        weld.translation_target_from_source_m, dtype=np.float64
    ).reshape(3)
    scale = float(weld.scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("weld scale must be positive and finite")

    out = []
    for keyframe in trajectory:
        if keyframe.status != "candidate":
            raise ValueError("world weld requires candidate trajectory keyframes")
        R_source_camera = np.asarray(
            keyframe.rotation_world_from_camera, dtype=np.float64
        ).reshape(3, 3)
        p_source = np.asarray(
            keyframe.position_world_m, dtype=np.float64
        ).reshape(3)
        v_source = np.asarray(
            keyframe.velocity_world_m_s, dtype=np.float64
        ).reshape(3)
        out.append(
            VisualInertialTrajectoryKeyframe(
                time_s=float(keyframe.time_s),
                rotation_world_from_camera=tuple(
                    float(x) for x in (R @ R_source_camera).reshape(-1)
                ),
                position_world_m=tuple(
                    float(x) for x in (scale * (R @ p_source) + t)
                ),
                velocity_world_m_s=tuple(
                    float(x) for x in (scale * (R @ v_source))
                ),
                status="candidate",
            )
        )
    return out
