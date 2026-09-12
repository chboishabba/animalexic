from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.shared_world_guard_transport import GuardTransportComparison


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
