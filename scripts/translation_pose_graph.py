from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TranslationConstraint:
    source_index: int
    target_index: int
    delta_world_m: tuple[float, float, float]
    weight: float
    provenance: str


@dataclass(frozen=True)
class TranslationPoseGraphCandidate:
    positions_world_m: tuple[tuple[float, float, float], ...]
    rms_constraint_residual_m: float
    constraint_count: int
    rank: int
    status: str
    full_vio_optimization_paid: bool = False


def _connected(keyframe_count: int, constraints) -> bool:
    adjacency = {i: set() for i in range(keyframe_count)}
    for c in constraints:
        adjacency[c.source_index].add(c.target_index)
        adjacency[c.target_index].add(c.source_index)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == keyframe_count


def optimize_translation_pose_graph(
    *,
    keyframe_count: int,
    constraints,
    anchor_position_world_m,
    max_rms_constraint_residual_m: float,
) -> TranslationPoseGraphCandidate:
    """Optimize translation-only keyframe positions from relative constraints.

    Keyframe 0 is anchored.  Every constraint encodes
    ``p_target - p_source = delta_world_m`` and is weighted before one global
    least-squares solve.  This is deliberately not full VIO: rotations,
    velocities, biases and inertial covariance are not optimized here.
    """
    if keyframe_count < 2:
        raise ValueError("keyframe_count must be at least two")
    if max_rms_constraint_residual_m < 0 or not math.isfinite(max_rms_constraint_residual_m):
        raise ValueError("max_rms_constraint_residual_m must be finite and non-negative")
    values = list(constraints)
    if len(values) < keyframe_count - 1:
        raise ValueError("not enough constraints to connect pose graph")
    anchor = np.asarray(anchor_position_world_m, dtype=np.float64)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor position must be finite xyz")

    for c in values:
        if not (0 <= c.source_index < keyframe_count and 0 <= c.target_index < keyframe_count):
            raise ValueError("constraint keyframe index out of range")
        if c.source_index == c.target_index:
            raise ValueError("constraint endpoints must differ")
        delta = np.asarray(c.delta_world_m, dtype=np.float64)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("constraint delta must be finite xyz")
        if not math.isfinite(c.weight) or c.weight <= 0:
            raise ValueError("constraint weight must be positive and finite")
        if not c.provenance:
            raise ValueError("constraint provenance is required")
    if not _connected(keyframe_count, values):
        raise ValueError("translation pose graph is disconnected")

    # Unknown vector contains positions for keyframes 1..N-1, three coordinates each.
    rows = []
    rhs = []
    for c in values:
        root_weight = math.sqrt(float(c.weight))
        delta = np.asarray(c.delta_world_m, dtype=np.float64)
        for axis in range(3):
            row = np.zeros(3 * (keyframe_count - 1), dtype=np.float64)
            target_rhs = float(delta[axis])
            if c.target_index == 0:
                target_rhs += float(anchor[axis])
            else:
                row[3 * (c.target_index - 1) + axis] += 1.0
            if c.source_index == 0:
                target_rhs -= float(anchor[axis])
            else:
                row[3 * (c.source_index - 1) + axis] -= 1.0
            rows.append(root_weight * row)
            rhs.append(root_weight * target_rhs)

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    expected_rank = 3 * (keyframe_count - 1)
    if rank < expected_rank:
        raise ValueError("translation pose graph lacks full anchored rank")
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    positions = [anchor]
    for i in range(1, keyframe_count):
        positions.append(solution[3 * (i - 1):3 * i])

    residuals = []
    for c in values:
        predicted = positions[c.target_index] - positions[c.source_index]
        residuals.append(np.linalg.norm(predicted - np.asarray(c.delta_world_m, dtype=np.float64)))
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return TranslationPoseGraphCandidate(
        positions_world_m=tuple(tuple(float(x) for x in p) for p in positions),
        rms_constraint_residual_m=rms,
        constraint_count=len(values),
        rank=rank,
        status="abstain" if rms > max_rms_constraint_residual_m else "candidate",
        full_vio_optimization_paid=False,
    )
