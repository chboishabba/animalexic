from __future__ import annotations

import json

import numpy as np

from scripts.pose_guard_sensitivity import (
    ConsumerQualityPolicy,
    GeometryFibrePerturbation,
    run_guard_sensitivity_portfolio,
    select_refinement_frontier,
)
from scripts.shared_world_guard_transport import GuardFrameInputs
from scripts.voxel_guard import VoxelGridSpec, VoxelGuardParams


def _reference_frames() -> list[GuardFrameInputs]:
    return [
        GuardFrameInputs(
            time_s=0.0,
            points=np.array(
                [
                    [1.50, -0.25, 0.50],
                    [1.50, 0.25, 0.50],
                    [1.25, 0.00, 0.75],
                    [1.75, 0.00, 0.25],
                ],
                dtype=np.float32,
            ),
            camera_origins=np.array(
                [
                    [0.00, -0.50, 0.00],
                    [0.00, -0.50, 0.00],
                    [0.00, 0.50, 0.00],
                    [0.00, 0.50, 0.00],
                ],
                dtype=np.float32,
            ),
            weights=np.array([1.0, 0.9, 1.0, 0.8], dtype=np.float32),
            residuals=np.array([0.0, 0.2, 0.0, 0.1], dtype=np.float32),
            origin_factors=np.ones(4, dtype=np.float32),
        ),
        GuardFrameInputs(
            time_s=1.0,
            points=np.array(
                [
                    [1.55, -0.20, 0.50],
                    [1.55, 0.30, 0.50],
                    [1.30, 0.05, 0.75],
                    [1.80, 0.05, 0.25],
                ],
                dtype=np.float32,
            ),
            camera_origins=np.array(
                [
                    [0.05, -0.50, 0.00],
                    [0.05, -0.50, 0.00],
                    [0.05, 0.50, 0.00],
                    [0.05, 0.50, 0.00],
                ],
                dtype=np.float32,
            ),
            weights=np.array([1.0, 0.9, 1.0, 0.8], dtype=np.float32),
            residuals=np.array([0.0, 0.2, 0.0, 0.1], dtype=np.float32),
            origin_factors=np.ones(4, dtype=np.float32),
        ),
    ]


def _grid() -> VoxelGridSpec:
    return VoxelGridSpec(
        origin=np.array([-0.5, -1.0, -0.5], dtype=np.float32),
        voxel_size=0.20,
        dims=(14, 12, 9),
    )


def _params() -> VoxelGuardParams:
    return VoxelGuardParams(
        alpha=1.0,
        alpha_h=1.0,
        beta=0.25,
        h_max=4.0,
        tau_p=0.015,
        tau_a=0.035,
        h_a=1.0,
        epsilon_rho=10.0,
        ray_decay=0.10,
        sigma_rho=8.0,
        gamma_neighbor=0.15,
    )


def run_controlled_probe() -> dict:
    policy = ConsumerQualityPolicy(
        min_state_agreement=0.995,
        min_ascended_iou=0.98,
        max_abs_score_residual=0.025,
    )
    perturbations = [
        GeometryFibrePerturbation("oracle"),
        GeometryFibrePerturbation("origin_25cm", origin_delta_m=(0.25, 0.0, 0.0)),
        GeometryFibrePerturbation(
            "yaw_10deg", rotation_delta_deg_xyz=(0.0, 0.0, 10.0)
        ),
        GeometryFibrePerturbation("scale_plus_10pct", scale_factor=1.10),
        GeometryFibrePerturbation("residual_plus_5", residual_delta=5.0),
    ]
    runs = run_guard_sensitivity_portfolio(
        _reference_frames(), perturbations, _grid(), _params(), policy
    )
    frontier = select_refinement_frontier([run.case for run in runs])
    rows = []
    for run in runs:
        rows.append(
            {
                "name": run.case.perturbation.name,
                "state_agreement": round(run.case.state_agreement, 9),
                "ascended_iou": round(run.case.ascended_iou, 9),
                "max_abs_score_residual": round(
                    run.case.max_abs_score_residual, 9
                ),
                "state_change_count": int(run.case.state_change_count),
                "frontier_nonzero_count": int(run.case.frontier_nonzero_count),
                "frontier_negative_count": int(
                    np.count_nonzero(run.comparison.frontier < 0)
                ),
                "frontier_positive_count": int(
                    np.count_nonzero(run.comparison.frontier > 0)
                ),
                "within_policy": bool(run.case.within_policy),
            }
        )
    return {
        "schema": "animalexic_pose_guard_sensitivity_v1",
        "consumer_policy": {
            "min_state_agreement": policy.min_state_agreement,
            "min_ascended_iou": policy.min_ascended_iou,
            "max_abs_score_residual": policy.max_abs_score_residual,
        },
        "cases": rows,
        "pareto_refinement_frontier": [case.perturbation.name for case in frontier],
    }


def main() -> None:
    print(json.dumps(run_controlled_probe(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
