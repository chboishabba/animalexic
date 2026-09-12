from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class InertialPVConstraint:
    source_index: int
    target_index: int
    dt_s: float
    delta_position_source_m: tuple[float, float, float]
    delta_velocity_source_m_s: tuple[float, float, float]
    weight: float
    provenance: str


@dataclass(frozen=True)
class VisualTranslationConstraint:
    source_index: int
    target_index: int
    delta_world_m: tuple[float, float, float]
    weight: float
    provenance: str


@dataclass(frozen=True)
class FixedRotationVISmootherCandidate:
    positions_world_m: tuple[tuple[float, float, float], ...]
    velocities_world_m_s: tuple[tuple[float, float, float], ...]
    rms_constraint_residual: float
    linear_rank: int
    inertial_constraint_count: int
    visual_constraint_count: int
    status: str
    full_nonlinear_vio_paid: bool = False


def _inertial_connected(frame_count, constraints) -> bool:
    adjacency = {i: set() for i in range(frame_count)}
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
    return len(seen) == frame_count


def optimize_fixed_rotation_position_velocity(
    *,
    rotations_world_from_camera,
    inertial_constraints,
    visual_constraints,
    anchor_position_world_m,
    anchor_velocity_world_m_s,
    max_rms_constraint_residual: float,
) -> FixedRotationVISmootherCandidate:
    """Solve position+velocity with rotations fixed and explicit VIO constraints.

    For each inertial interval:
      p_j = p_i + v_i dt + R_i Δp
      v_j = v_i + R_i Δv
    where Δp/Δv are the existing preintegrated zero-initial-velocity deltas.
    Visual constraints add world-frame relative translation equations.
    """
    rotations = [np.asarray(r, dtype=np.float64).reshape(3, 3) for r in rotations_world_from_camera]
    frame_count = len(rotations)
    if frame_count < 2 or any(not np.all(np.isfinite(r)) for r in rotations):
        raise ValueError("at least two finite fixed rotations are required")
    inertial = list(inertial_constraints)
    visual = list(visual_constraints)
    if not inertial or not _inertial_connected(frame_count, inertial):
        raise ValueError("inertial constraints must connect all keyframes to the anchor")
    if max_rms_constraint_residual < 0 or not math.isfinite(max_rms_constraint_residual):
        raise ValueError("max_rms_constraint_residual must be finite and non-negative")
    p0 = np.asarray(anchor_position_world_m, dtype=np.float64)
    v0 = np.asarray(anchor_velocity_world_m_s, dtype=np.float64)
    if p0.shape != (3,) or v0.shape != (3,) or not np.all(np.isfinite(p0)) or not np.all(np.isfinite(v0)):
        raise ValueError("anchor position and velocity must be finite xyz")

    # Unknowns: [p_1..p_N-1, v_1..v_N-1], each xyz.
    width = 6 * (frame_count - 1)
    rows = []
    rhs = []

    def p_slot(index, axis):
        return 3 * (index - 1) + axis

    def v_slot(index, axis):
        return 3 * (frame_count - 1) + 3 * (index - 1) + axis

    for c in inertial:
        if not (0 <= c.source_index < frame_count and 0 <= c.target_index < frame_count) or c.source_index == c.target_index:
            raise ValueError("inertial constraint indices are invalid")
        if c.dt_s <= 0 or not math.isfinite(c.dt_s) or c.weight <= 0 or not math.isfinite(c.weight) or not c.provenance:
            raise ValueError("inertial constraint requires positive dt/weight and provenance")
        dp = np.asarray(c.delta_position_source_m, dtype=np.float64)
        dv = np.asarray(c.delta_velocity_source_m_s, dtype=np.float64)
        if dp.shape != (3,) or dv.shape != (3,) or not np.all(np.isfinite(dp)) or not np.all(np.isfinite(dv)):
            raise ValueError("inertial deltas must be finite xyz")
        world_dp = rotations[c.source_index] @ dp
        world_dv = rotations[c.source_index] @ dv
        rw = math.sqrt(float(c.weight))
        for axis in range(3):
            # p_j - p_i - dt v_i = R_i Δp
            row = np.zeros(width)
            target = float(world_dp[axis])
            if c.target_index == 0:
                target -= p0[axis]
            else:
                row[p_slot(c.target_index, axis)] += 1.0
            if c.source_index == 0:
                target += p0[axis] + c.dt_s * v0[axis]
            else:
                row[p_slot(c.source_index, axis)] -= 1.0
                row[v_slot(c.source_index, axis)] -= c.dt_s
            rows.append(rw * row)
            rhs.append(rw * target)

            # v_j - v_i = R_i Δv
            row = np.zeros(width)
            target = float(world_dv[axis])
            if c.target_index == 0:
                target -= v0[axis]
            else:
                row[v_slot(c.target_index, axis)] += 1.0
            if c.source_index == 0:
                target += v0[axis]
            else:
                row[v_slot(c.source_index, axis)] -= 1.0
            rows.append(rw * row)
            rhs.append(rw * target)

    for c in visual:
        if not (0 <= c.source_index < frame_count and 0 <= c.target_index < frame_count) or c.source_index == c.target_index:
            raise ValueError("visual constraint indices are invalid")
        if c.weight <= 0 or not math.isfinite(c.weight) or not c.provenance:
            raise ValueError("visual constraint requires positive weight and provenance")
        delta = np.asarray(c.delta_world_m, dtype=np.float64)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("visual delta must be finite xyz")
        rw = math.sqrt(float(c.weight))
        for axis in range(3):
            row = np.zeros(width)
            target = float(delta[axis])
            if c.target_index == 0:
                target -= p0[axis]
            else:
                row[p_slot(c.target_index, axis)] += 1.0
            if c.source_index == 0:
                target += p0[axis]
            else:
                row[p_slot(c.source_index, axis)] -= 1.0
            rows.append(rw * row)
            rhs.append(rw * target)

    A = np.vstack(rows)
    b = np.asarray(rhs)
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    if rank < width:
        raise ValueError("fixed-rotation visual-inertial system lacks full anchored rank")
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    positions = [p0]
    velocities = [v0]
    for i in range(1, frame_count):
        positions.append(solution[p_slot(i, 0):p_slot(i, 0) + 3])
        velocities.append(solution[v_slot(i, 0):v_slot(i, 0) + 3])

    residual_norms = []
    for c in inertial:
        predicted_dp = positions[c.target_index] - positions[c.source_index] - velocities[c.source_index] * c.dt_s
        predicted_dv = velocities[c.target_index] - velocities[c.source_index]
        residual_norms.append(np.linalg.norm(predicted_dp - rotations[c.source_index] @ np.asarray(c.delta_position_source_m)))
        residual_norms.append(np.linalg.norm(predicted_dv - rotations[c.source_index] @ np.asarray(c.delta_velocity_source_m_s)))
    for c in visual:
        predicted = positions[c.target_index] - positions[c.source_index]
        residual_norms.append(np.linalg.norm(predicted - np.asarray(c.delta_world_m)))
    rms = float(np.sqrt(np.mean(np.square(residual_norms))))

    return FixedRotationVISmootherCandidate(
        positions_world_m=tuple(tuple(float(x) for x in p) for p in positions),
        velocities_world_m_s=tuple(tuple(float(x) for x in v) for v in velocities),
        rms_constraint_residual=rms,
        linear_rank=rank,
        inertial_constraint_count=len(inertial),
        visual_constraint_count=len(visual),
        status="abstain" if rms > max_rms_constraint_residual else "candidate",
        full_nonlinear_vio_paid=False,
    )
