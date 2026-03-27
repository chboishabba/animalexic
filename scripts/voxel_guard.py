#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VOXEL_GROUNDED = 0
VOXEL_PLATEAU = 1
VOXEL_ASCENDED = 2


@dataclass
class VoxelGuardParams:
    alpha: float = 0.85
    alpha_h: float = 0.90
    beta: float = 0.10
    h_max: float = 8.0
    tau_p: float = 1.5
    tau_a: float = 3.0
    h_a: float = 2.0
    epsilon_rho: float = 64.0
    ray_decay: float = 0.35


@dataclass
class VoxelGridSpec:
    origin: np.ndarray
    voxel_size: float
    dims: tuple[int, int, int]


def dda_voxel_line(
    origin: np.ndarray,
    endpoint: np.ndarray,
    grid_origin: np.ndarray,
    voxel_size: float,
    dims: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    origin = origin.astype(np.float32)
    endpoint = endpoint.astype(np.float32)
    direction = endpoint - origin
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return []
    direction /= length

    grid_min = grid_origin.astype(np.float32)
    grid_max = grid_min + np.asarray(dims, dtype=np.float32) * float(voxel_size)

    t_min = 0.0
    t_max = length
    for axis in range(3):
        d = float(direction[axis])
        if abs(d) < 1e-12:
            if origin[axis] < grid_min[axis] or origin[axis] > grid_max[axis]:
                return []
            continue
        t1 = float((grid_min[axis] - origin[axis]) / d)
        t2 = float((grid_max[axis] - origin[axis]) / d)
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return []

    start_world = origin + direction * max(0.0, t_min)

    def world_to_voxel(point: np.ndarray) -> np.ndarray:
        return np.floor((point - grid_origin) / voxel_size).astype(np.int32)

    start = world_to_voxel(start_world)
    end = world_to_voxel(endpoint)
    if np.any(end < 0) or end[0] >= dims[0] or end[1] >= dims[1] or end[2] >= dims[2]:
        return []

    ix, iy, iz = int(start[0]), int(start[1]), int(start[2])
    ex, ey, ez = int(end[0]), int(end[1]), int(end[2])

    step_x = 1 if direction[0] >= 0 else -1
    step_y = 1 if direction[1] >= 0 else -1
    step_z = 1 if direction[2] >= 0 else -1

    def axis_t_max(i: int, d: float, axis_origin: float, grid_axis_origin: float, step: int) -> float:
        if abs(d) < 1e-12:
            return np.inf
        next_boundary = grid_axis_origin + (i + (1 if step > 0 else 0)) * voxel_size
        return (next_boundary - axis_origin) / d

    def axis_t_delta(d: float) -> float:
        if abs(d) < 1e-12:
            return np.inf
        return voxel_size / abs(d)

    t_max_x = axis_t_max(ix, float(direction[0]), float(origin[0]), float(grid_origin[0]), step_x)
    t_max_y = axis_t_max(iy, float(direction[1]), float(origin[1]), float(grid_origin[1]), step_y)
    t_max_z = axis_t_max(iz, float(direction[2]), float(origin[2]), float(grid_origin[2]), step_z)
    t_delta_x = axis_t_delta(float(direction[0]))
    t_delta_y = axis_t_delta(float(direction[1]))
    t_delta_z = axis_t_delta(float(direction[2]))

    hits: list[tuple[int, int, int]] = []
    max_steps = int(sum(dims))
    for _ in range(max_steps):
        if 0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]:
            hits.append((ix, iy, iz))
        if ix == ex and iy == ey and iz == ez:
            break
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            ix += step_x
            t_max_x += t_delta_x
        elif t_max_y <= t_max_z:
            iy += step_y
            t_max_y += t_delta_y
        else:
            iz += step_z
            t_max_z += t_delta_z
        if ix < -1 or iy < -1 or iz < -1 or ix > dims[0] or iy > dims[1] or iz > dims[2]:
            break
    return hits


def build_grid_spec(
    points_xyz: np.ndarray,
    *,
    voxel_size: float,
    margin_voxels: int = 2,
) -> VoxelGridSpec:
    mins = np.min(points_xyz, axis=0)
    maxs = np.max(points_xyz, axis=0)
    margin = float(margin_voxels) * voxel_size
    origin = mins - margin
    extents = (maxs - mins) + 2.0 * margin
    dims = tuple(int(max(1, np.ceil(extents[i] / voxel_size))) for i in range(3))
    return VoxelGridSpec(origin=origin.astype(np.float32), voxel_size=float(voxel_size), dims=dims)


def accumulate_candidate_voxels(
    grid_spec: VoxelGridSpec,
    frame_points: Iterable[np.ndarray],
    frame_weights: Iterable[np.ndarray],
    frame_residuals: Iterable[np.ndarray],
    params: VoxelGuardParams,
    frame_origin_factors: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evidence = np.zeros(grid_spec.dims, dtype=np.float32)
    temporal_hits = np.zeros(grid_spec.dims, dtype=np.float32)
    score = np.zeros(grid_spec.dims, dtype=np.float32)
    residual_num = np.zeros(grid_spec.dims, dtype=np.float32)
    residual_den = np.zeros(grid_spec.dims, dtype=np.float32)
    camera_origin = np.zeros(3, dtype=np.float32)

    if frame_origin_factors is None:
        frame_origin_factors = (None for _ in frame_points)

    for points_xyz, weights, residuals, origin_factors in zip(
        frame_points, frame_weights, frame_residuals, frame_origin_factors
    ):
        frame_evidence = np.zeros(grid_spec.dims, dtype=np.float32)
        frame_residual_num = np.zeros(grid_spec.dims, dtype=np.float32)
        frame_residual_den = np.zeros(grid_spec.dims, dtype=np.float32)
        touched_this_frame: set[tuple[int, int, int]] = set()
        if origin_factors is None:
            origin_factors = np.ones((len(points_xyz),), dtype=np.float32)
        for point_xyz, w, residual, origin_factor in zip(points_xyz, weights, residuals, origin_factors):
            voxels = dda_voxel_line(
                camera_origin,
                point_xyz,
                grid_spec.origin,
                grid_spec.voxel_size,
                grid_spec.dims,
            )
            if not voxels:
                continue
            voxel_count = len(voxels)
            ray_weights = np.exp(
                -float(params.ray_decay) * np.arange(voxel_count - 1, -1, -1, dtype=np.float32)
            )
            ray_sum = float(np.sum(ray_weights))
            if ray_sum <= 1e-8:
                continue
            ray_weights /= ray_sum
            base_weight = float(w) * float(origin_factor)
            for voxel, ray_weight in zip(voxels, ray_weights):
                contrib = base_weight * float(ray_weight)
                frame_evidence[voxel] += contrib
                frame_residual_num[voxel] += contrib * float(residual)
                frame_residual_den[voxel] += contrib
                touched_this_frame.add(voxel)
        evidence = float(params.alpha) * evidence + frame_evidence
        temporal_hits = float(params.alpha_h) * temporal_hits
        for idx in touched_this_frame:
            temporal_hits[idx] += 1.0
        residual_num = float(params.alpha) * residual_num + frame_residual_num
        residual_den = float(params.alpha) * residual_den + frame_residual_den

    score = evidence * (1.0 + float(params.beta) * np.minimum(temporal_hits, float(params.h_max)))
    residual = np.zeros_like(score, dtype=np.float32)
    valid_residual = residual_den > 1e-6
    residual[valid_residual] = residual_num[valid_residual] / residual_den[valid_residual]
    return evidence, temporal_hits, score, residual


def guard_voxels(
    score: np.ndarray,
    temporal_hits: np.ndarray,
    residual: np.ndarray,
    params: VoxelGuardParams,
) -> np.ndarray:
    states = np.full(score.shape, VOXEL_GROUNDED, dtype=np.uint8)
    plateau = score >= float(params.tau_p)
    ascended = plateau & (score >= float(params.tau_a)) & (
        temporal_hits >= float(params.h_a)
    ) & (
        residual <= float(params.epsilon_rho)
    )
    states[plateau] = VOXEL_PLATEAU
    states[ascended] = VOXEL_ASCENDED
    return states


def voxel_centers_from_mask(mask: np.ndarray, grid_spec: VoxelGridSpec) -> np.ndarray:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers = grid_spec.origin[None, :] + (ijk.astype(np.float32) + 0.5) * float(grid_spec.voxel_size)
    return centers.astype(np.float32)


def write_points_ply_ascii(path: Path, points_xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points_xyz:
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
