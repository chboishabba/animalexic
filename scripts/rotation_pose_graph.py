from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RotationConstraint:
    source_index: int
    target_index: int
    rotation_target_from_source: tuple[float, ...]
    weight: float
    provenance: str


@dataclass(frozen=True)
class RotationPoseGraphCandidate:
    rotations_world_from_frame: tuple[tuple[float, ...], ...]
    rms_rotation_residual_deg: float
    constraint_count: int
    iteration_count: int
    status: str
    full_vio_optimization_paid: bool = False


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0
    return U @ correction @ Vt


def _angle_deg(rotation: np.ndarray) -> float:
    value = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(value))


def _connected(frame_count: int, constraints) -> bool:
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


def optimize_rotation_pose_graph(
    *,
    frame_count: int,
    constraints,
    anchor_rotation_world_from_frame,
    max_rms_rotation_residual_deg: float,
    max_iterations: int = 64,
    convergence_deg: float = 1e-9,
) -> RotationPoseGraphCandidate:
    """Bounded chordal SO(3) averaging over relative-rotation constraints.

    A constraint stores ``R_target_from_source``. With node orientations
    ``R_world_from_frame`` the relation is
    ``R_w_target = R_w_source @ R_target_from_source.T``.
    Frame 0 is anchored and never updated.
    """
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if max_rms_rotation_residual_deg < 0 or not math.isfinite(max_rms_rotation_residual_deg):
        raise ValueError("max_rms_rotation_residual_deg must be finite and non-negative")
    values = list(constraints)
    if len(values) < frame_count - 1:
        raise ValueError("not enough constraints to connect rotation graph")
    anchor = np.asarray(anchor_rotation_world_from_frame, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(anchor)):
        raise ValueError("anchor rotation must be finite")
    anchor = _project_so3(anchor)

    parsed = []
    for c in values:
        if not (0 <= c.source_index < frame_count and 0 <= c.target_index < frame_count):
            raise ValueError("constraint frame index out of range")
        if c.source_index == c.target_index:
            raise ValueError("constraint endpoints must differ")
        R_ts = np.asarray(c.rotation_target_from_source, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(R_ts)):
            raise ValueError("constraint rotation must be finite")
        if not math.isfinite(c.weight) or c.weight <= 0:
            raise ValueError("constraint weight must be positive and finite")
        if not c.provenance:
            raise ValueError("constraint provenance is required")
        parsed.append((c, _project_so3(R_ts)))
    if not _connected(frame_count, values):
        raise ValueError("rotation pose graph is disconnected")

    # Deterministic BFS initialization from the anchor.
    rotations: list[np.ndarray | None] = [None] * frame_count
    rotations[0] = anchor
    progress = True
    while progress and any(rotation is None for rotation in rotations):
        progress = False
        for c, R_ts in parsed:
            if rotations[c.source_index] is not None and rotations[c.target_index] is None:
                rotations[c.target_index] = rotations[c.source_index] @ R_ts.T
                progress = True
            elif rotations[c.target_index] is not None and rotations[c.source_index] is None:
                rotations[c.source_index] = rotations[c.target_index] @ R_ts
                progress = True
    if any(rotation is None for rotation in rotations):
        raise ValueError("could not initialize connected rotation graph")
    rotations = [np.asarray(rotation, dtype=np.float64) for rotation in rotations]

    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        new_rotations = [anchor]
        max_change = 0.0
        for node in range(1, frame_count):
            proposals = []
            weights = []
            for c, R_ts in parsed:
                if c.target_index == node:
                    proposals.append(rotations[c.source_index] @ R_ts.T)
                    weights.append(float(c.weight))
                elif c.source_index == node:
                    proposals.append(rotations[c.target_index] @ R_ts)
                    weights.append(float(c.weight))
            if not proposals:
                raise ValueError("rotation node has no constraints")
            average = sum(w * proposal for w, proposal in zip(weights, proposals)) / sum(weights)
            projected = _project_so3(average)
            change = _angle_deg(rotations[node].T @ projected)
            max_change = max(max_change, change)
            new_rotations.append(projected)
        rotations = new_rotations
        if max_change <= convergence_deg:
            break

    residuals = []
    for c, R_ts in parsed:
        predicted = rotations[c.target_index].T @ rotations[c.source_index]
        residuals.append(_angle_deg(predicted @ R_ts.T))
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return RotationPoseGraphCandidate(
        rotations_world_from_frame=tuple(
            tuple(float(x) for x in rotation.reshape(-1)) for rotation in rotations
        ),
        rms_rotation_residual_deg=rms,
        constraint_count=len(values),
        iteration_count=iterations,
        status="abstain" if rms > max_rms_rotation_residual_deg else "candidate",
        full_vio_optimization_paid=False,
    )
