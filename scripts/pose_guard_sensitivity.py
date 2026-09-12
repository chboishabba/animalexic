from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from scripts.shared_world_guard_transport import (
    GuardFrameInputs,
    GuardTransportComparison,
    compare_guard_transport,
)
from scripts.voxel_guard import accumulate_candidate_voxels, guard_voxels


@dataclass(frozen=True)
class GeometryFibrePerturbation:
    name: str
    origin_delta_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_delta_deg_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_factor: float = 1.0
    residual_delta: float = 0.0


@dataclass(frozen=True)
class ConsumerQualityPolicy:
    min_state_agreement: float
    min_ascended_iou: float
    max_abs_score_residual: float

    def __post_init__(self):
        if not 0.0 <= self.min_state_agreement <= 1.0:
            raise ValueError("min_state_agreement must be in [0,1]")
        if not 0.0 <= self.min_ascended_iou <= 1.0:
            raise ValueError("min_ascended_iou must be in [0,1]")
        if self.max_abs_score_residual < 0:
            raise ValueError("max_abs_score_residual must be non-negative")


@dataclass(frozen=True)
class SensitivityCase:
    perturbation: GeometryFibrePerturbation
    state_agreement: float
    ascended_iou: float
    max_abs_score_residual: float
    state_change_count: int
    frontier_nonzero_count: int
    within_policy: bool

    @property
    def defect_coordinates(self) -> tuple[float, float, float]:
        return (
            1.0 - self.state_agreement,
            1.0 - self.ascended_iou,
            self.max_abs_score_residual,
        )


@dataclass(frozen=True)
class GuardRun:
    evidence: np.ndarray
    temporal_hits: np.ndarray
    score: np.ndarray
    residual: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class SensitivityRun:
    case: SensitivityCase
    comparison: GuardTransportComparison
    guard_run: GuardRun


def sensitivity_case_from_comparison(
    perturbation: GeometryFibrePerturbation,
    comparison: GuardTransportComparison,
    policy: ConsumerQualityPolicy,
) -> SensitivityCase:
    residual = np.asarray(comparison.score_residual, dtype=np.float64)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    state_change_count = int(np.count_nonzero(comparison.state_change_mask))
    frontier_nonzero_count = int(np.count_nonzero(comparison.frontier))
    within_policy = (
        comparison.state_agreement >= policy.min_state_agreement
        and comparison.ascended_iou >= policy.min_ascended_iou
        and max_abs <= policy.max_abs_score_residual
    )
    return SensitivityCase(
        perturbation=perturbation,
        state_agreement=float(comparison.state_agreement),
        ascended_iou=float(comparison.ascended_iou),
        max_abs_score_residual=max_abs,
        state_change_count=state_change_count,
        frontier_nonzero_count=frontier_nonzero_count,
        within_policy=bool(within_policy),
    )


def _dominates(a: SensitivityCase, b: SensitivityCase) -> bool:
    """Return True when a is at least as consumer-visible as b on all axes."""
    ax = a.defect_coordinates
    bx = b.defect_coordinates
    return all(x >= y for x, y in zip(ax, bx)) and any(
        x > y for x, y in zip(ax, bx)
    )


def select_refinement_frontier(cases) -> list[SensitivityCase]:
    """Return Pareto-maximal consumer-visible defects without scalarization."""
    unpaid = [case for case in cases if not case.within_policy]
    frontier = []
    for case in unpaid:
        if any(_dominates(other, case) for other in unpaid if other is not case):
            continue
        frontier.append(case)
    return frontier


def _rotation_xyz_degrees(delta_deg_xyz) -> np.ndarray:
    rx, ry, rz = [math.radians(float(x)) for x in delta_deg_xyz]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def perturb_guard_frames(
    frames,
    perturbation: GeometryFibrePerturbation,
) -> list[GuardFrameInputs]:
    """Inject one controlled geometry-fibre defect into guard-frame inputs.

    Endpoint rotation is performed about each original camera origin, then the
    ray length is scaled explicitly. Camera-origin translation is applied as a
    separate coordinate so downstream attribution can distinguish the fibres.
    """
    origin_delta = np.asarray(perturbation.origin_delta_m, dtype=np.float64)
    if origin_delta.shape != (3,) or not np.all(np.isfinite(origin_delta)):
        raise ValueError("origin_delta_m must be finite xyz")
    scale = float(perturbation.scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale_factor must be positive and finite")
    residual_delta = float(perturbation.residual_delta)
    if not np.isfinite(residual_delta):
        raise ValueError("residual_delta must be finite")
    R = _rotation_xyz_degrees(perturbation.rotation_delta_deg_xyz)

    out = []
    for frame in frames:
        points = np.asarray(frame.points, dtype=np.float64)
        origins = np.asarray(frame.camera_origins, dtype=np.float64)
        if points.shape != origins.shape or points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("frame points and camera_origins must be matching Nx3 arrays")
        ray = points - origins
        rotated_scaled_ray = scale * (R @ ray.T).T
        candidate_origins = origins + origin_delta[None, :]
        candidate_points = candidate_origins + rotated_scaled_ray
        candidate_residuals = np.maximum(
            0.0,
            np.asarray(frame.residuals, dtype=np.float64) + residual_delta,
        )
        out.append(
            GuardFrameInputs(
                time_s=float(frame.time_s),
                points=candidate_points.astype(np.float32),
                camera_origins=candidate_origins.astype(np.float32),
                weights=np.asarray(frame.weights, dtype=np.float32).copy(),
                residuals=candidate_residuals.astype(np.float32),
                origin_factors=np.asarray(frame.origin_factors, dtype=np.float32).copy(),
            )
        )
    return out


def execute_guard_frames(frames, grid_spec, params) -> GuardRun:
    frames = list(frames)
    evidence, temporal_hits, score, residual = accumulate_candidate_voxels(
        grid_spec,
        [frame.points for frame in frames],
        [frame.weights for frame in frames],
        [frame.residuals for frame in frames],
        params,
        frame_origin_factors=[frame.origin_factors for frame in frames],
        frame_camera_origins=[frame.camera_origins for frame in frames],
    )
    states = guard_voxels(score, temporal_hits, residual, params)
    return GuardRun(evidence, temporal_hits, score, residual, states)


def run_guard_sensitivity_portfolio(
    reference_frames,
    perturbations,
    grid_spec,
    params,
    policy: ConsumerQualityPolicy,
) -> list[SensitivityRun]:
    """Run controlled geometry defects through one unchanged voxel consumer."""
    reference_frames = list(reference_frames)
    oracle = execute_guard_frames(reference_frames, grid_spec, params)
    runs = []
    for perturbation in perturbations:
        candidate_frames = perturb_guard_frames(reference_frames, perturbation)
        candidate = execute_guard_frames(candidate_frames, grid_spec, params)
        comparison = compare_guard_transport(
            oracle.states,
            candidate.states,
            oracle.score,
            candidate.score,
        )
        case = sensitivity_case_from_comparison(perturbation, comparison, policy)
        runs.append(SensitivityRun(case, comparison, candidate))
    return runs


def active_perturbation_fibres(
    perturbation: GeometryFibrePerturbation,
    *,
    atol: float = 1e-12,
) -> tuple[str, ...]:
    """Name explicit defect coordinates without collapsing them into one score."""
    active = []
    if any(abs(float(x)) > atol for x in perturbation.origin_delta_m):
        active.append("camera_origin")
    if any(abs(float(x)) > atol for x in perturbation.rotation_delta_deg_xyz):
        active.append("orientation")
    if abs(float(perturbation.scale_factor) - 1.0) > atol:
        active.append("metric_scale")
    if abs(float(perturbation.residual_delta)) > atol:
        active.append("observation_residual")
    return tuple(active)


def _shrink_perturbation(
    perturbation: GeometryFibrePerturbation,
    factor: float,
    step: int,
) -> GeometryFibrePerturbation:
    if not 0.0 < factor < 1.0:
        raise ValueError("refinement factor must be in (0,1)")
    return GeometryFibrePerturbation(
        name=f"{perturbation.name}@refine{step}",
        origin_delta_m=tuple(float(x) * factor for x in perturbation.origin_delta_m),
        rotation_delta_deg_xyz=tuple(
            float(x) * factor for x in perturbation.rotation_delta_deg_xyz
        ),
        scale_factor=1.0 + (float(perturbation.scale_factor) - 1.0) * factor,
        residual_delta=float(perturbation.residual_delta) * factor,
    )


def quality_targeted_refinement(
    reference_frames,
    initial_perturbation: GeometryFibrePerturbation,
    grid_spec,
    params,
    policy: ConsumerQualityPolicy,
    *,
    max_steps: int = 8,
    shrink_factor: float = 0.5,
) -> list[SensitivityRun]:
    """Shrink only the named defect coordinates until the consumer is adequate.

    This is a bounded diagnostic/refinement experiment, not a pose optimizer.
    It preserves the observation carrier and asks how much coordinate error the
    downstream consumer tolerates. No scalar loss is used and no promotion is
    performed.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    history = []
    perturbation = initial_perturbation
    for step in range(max_steps + 1):
        run = run_guard_sensitivity_portfolio(
            reference_frames,
            [perturbation],
            grid_spec,
            params,
            policy,
        )[0]
        history.append(run)
        if run.case.within_policy:
            break
        perturbation = _shrink_perturbation(
            perturbation,
            shrink_factor,
            step + 1,
        )
    return history
