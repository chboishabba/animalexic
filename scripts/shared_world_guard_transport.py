from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WorldRayObservation:
    camera_id: int
    time_s: float
    point_world_m: tuple[float, float, float]
    weight: float
    residual: float
    origin_factor: float = 1.0


@dataclass(frozen=True)
class GuardFrameInputs:
    time_s: float
    points: np.ndarray
    camera_origins: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    origin_factors: np.ndarray


@dataclass(frozen=True)
class GuardTransportComparison:
    state_agreement: float
    ascended_iou: float
    score_residual: np.ndarray
    state_change_mask: np.ndarray
    frontier: np.ndarray


def _index_candidate_keyframes(
    keyframes_by_camera: Mapping[int, Sequence[object]],
) -> dict[tuple[int, float], np.ndarray]:
    index: dict[tuple[int, float], np.ndarray] = {}
    for camera_id, frames in keyframes_by_camera.items():
        for keyframe in frames:
            if getattr(keyframe, "status", None) != "candidate":
                continue
            key = (int(camera_id), float(keyframe.time_s))
            if key in index:
                raise ValueError("duplicate candidate keyframe for camera/time")
            position = np.asarray(keyframe.position_world_m, dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("keyframe position must be finite xyz")
            index[key] = position
    return index


def adapt_world_rays_to_guard_frames(
    observations,
    keyframes_by_camera: Mapping[int, Sequence[object]],
) -> list[GuardFrameInputs]:
    """Map world observations onto the existing guarded voxel input contract.

    The adapter owns no promotion policy. It only transports candidate
    world-welded camera origins and observation evidence into the existing
    frame_points/frame_camera_origins/weights/residuals representation.
    """
    index = _index_candidate_keyframes(keyframes_by_camera)
    grouped: dict[float, list[tuple[np.ndarray, np.ndarray, float, float, float]]] = {}

    for observation in observations:
        key = (int(observation.camera_id), float(observation.time_s))
        if key not in index:
            raise ValueError("missing candidate keyframe for observation camera/time")
        point = np.asarray(observation.point_world_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("observation point must be finite xyz")
        if not np.isfinite(observation.weight) or observation.weight <= 0:
            raise ValueError("observation weight must be positive and finite")
        if not np.isfinite(observation.residual) or observation.residual < 0:
            raise ValueError("observation residual must be non-negative and finite")
        if not np.isfinite(observation.origin_factor) or observation.origin_factor <= 0:
            raise ValueError("origin_factor must be positive and finite")

        grouped.setdefault(float(observation.time_s), []).append(
            (
                point,
                index[key],
                float(observation.weight),
                float(observation.residual),
                float(observation.origin_factor),
            )
        )

    frames: list[GuardFrameInputs] = []
    for time_s in sorted(grouped):
        rows = grouped[time_s]
        frames.append(
            GuardFrameInputs(
                time_s=time_s,
                points=np.asarray([row[0] for row in rows], dtype=np.float32),
                camera_origins=np.asarray([row[1] for row in rows], dtype=np.float32),
                weights=np.asarray([row[2] for row in rows], dtype=np.float32),
                residuals=np.asarray([row[3] for row in rows], dtype=np.float32),
                origin_factors=np.asarray([row[4] for row in rows], dtype=np.float32),
            )
        )
    return frames


def compare_guard_transport(
    reference_states,
    candidate_states,
    reference_score,
    candidate_score,
    *,
    ascended_state: int = 2,
) -> GuardTransportComparison:
    """Compare a candidate geometry transport with an oracle guard run.

    The signed ternary frontier is deliberately analogous to dashiRTX's signed
    transport frontier: -1 means candidate guard state fell below the oracle,
    0 means the state agrees, and +1 means it rose above the oracle. It is a
    diagnostic/refinement coordinate only and carries no promotion authority.
    """
    reference_states = np.asarray(reference_states)
    candidate_states = np.asarray(candidate_states)
    reference_score = np.asarray(reference_score, dtype=np.float64)
    candidate_score = np.asarray(candidate_score, dtype=np.float64)

    if (
        reference_states.shape != candidate_states.shape
        or reference_states.shape != reference_score.shape
        or reference_states.shape != candidate_score.shape
    ):
        raise ValueError("guard comparison arrays must share shape")

    state_agreement = (
        float(np.mean(reference_states == candidate_states))
        if reference_states.size
        else 1.0
    )
    reference_ascended = reference_states == ascended_state
    candidate_ascended = candidate_states == ascended_state
    union = int(np.count_nonzero(reference_ascended | candidate_ascended))
    intersection = int(np.count_nonzero(reference_ascended & candidate_ascended))
    ascended_iou = 1.0 if union == 0 else float(intersection) / float(union)

    state_delta = candidate_states.astype(np.int16) - reference_states.astype(np.int16)
    frontier = np.sign(state_delta).astype(np.int8)

    return GuardTransportComparison(
        state_agreement=state_agreement,
        ascended_iou=ascended_iou,
        score_residual=candidate_score - reference_score,
        state_change_mask=reference_states != candidate_states,
        frontier=frontier,
    )
