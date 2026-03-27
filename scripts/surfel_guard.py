#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

SURFEL_GROUNDED = 0
SURFEL_PLATEAU = 1
SURFEL_ASCENDED = 2


@dataclass
class SurfelGuardParams:
    alpha: float = 0.85
    alpha_h: float = 0.90
    beta: float = 0.35
    h_max: float = 8.0
    tau_p: float = 0.5
    tau_a: float = 1.0
    h_a: float = 2.0
    epsilon_rho: float = 64.0
    sigma_rho: float = 24.0
    gamma_neighbor: float = 0.20
    cell_size: float = 0.25
    pos_eps: float = 0.20
    normal_eps: float = 0.52  # radians (~30 deg)
    spread_sigma: float = 0.10
    spread_max: float = 0.08
    drift_sigma: float = 0.05
    drift_max: float = 0.05


def _cell_key(pos: np.ndarray, cell_size: float) -> Tuple[int, int, int]:
    return (
        int(np.floor(pos[0] / cell_size)),
        int(np.floor(pos[1] / cell_size)),
        int(np.floor(pos[2] / cell_size)),
    )


def _angle_between(n1: np.ndarray, n2: np.ndarray) -> float:
    d = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(np.arccos(d)) if -1.0 <= d <= 1.0 else np.pi


def init_surfel_store() -> tuple[list[dict], dict[Tuple[int, int, int], list[int]]]:
    return [], {}


def accumulate_frame_into_surfels(
    surfels: list[dict],
    grid: dict[Tuple[int, int, int], list[int]],
    frame_idx: int,
    points_xyz: np.ndarray,
    weights: np.ndarray,
    residuals: np.ndarray,
    params: SurfelGuardParams,
) -> dict[str, float]:
    if len(points_xyz) == 0:
        return {
            "points": 0.0,
            "new_surfels": 0.0,
            "merged_surfels": 0.0,
            "merge_accept_rate": 0.0,
            "mean_merge_dist": 0.0,
            "mean_input_residual": 0.0,
        }
    stats = {
        "points": float(len(points_xyz)),
        "new_surfels": 0.0,
        "merged_surfels": 0.0,
        "merge_accept_rate": 0.0,
        "mean_merge_dist": 0.0,
        "mean_input_residual": float(np.mean(residuals)) if len(residuals) else 0.0,
    }
    merge_dists: list[float] = []
    for pos, w, residual in zip(points_xyz, weights, residuals):
        if w <= 0:
            continue
        n = pos.astype(np.float32)
        norm = float(np.linalg.norm(n))
        if norm > 1e-6:
            n /= norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        key = _cell_key(pos, params.cell_size)
        best_idx = None
        best_dist = float(params.pos_eps)
        for dk in ((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)):
            nk = (key[0] + dk[0], key[1] + dk[1], key[2] + dk[2])
            for idx in grid.get(nk, []):
                s = surfels[idx]
                if int(s.get("last_frame", -1)) == int(frame_idx):
                    continue
                dist = float(np.linalg.norm(pos - s["pos"]))
                if dist < best_dist:
                    ang = _angle_between(n, s["normal"])
                    if ang <= float(params.normal_eps):
                        best_dist = dist
                        best_idx = idx

        residual_weight = float(np.exp(-float(residual) / max(1e-6, float(params.sigma_rho))))
        contrib = float(w) * residual_weight
        if best_idx is None:
            idx = len(surfels)
            surfels.append(
                {
                    "pos": pos.astype(np.float32),
                    "centroid": pos.astype(np.float32),
                    "normal": n.astype(np.float32),
                    "weight": contrib,
                    "hits": 1.0,
                    "frame_hits": 1.0,
                    "frame_count": 1,
                    "obs_count": 1.0,
                    "last_frame": int(frame_idx),
                    "residual": float(residual),
                    "residual_ema": float(residual),
                    "support_spread": 0.0,
                }
            )
            grid.setdefault(key, []).append(idx)
            stats["new_surfels"] += 1.0
        else:
            s = surfels[best_idx]
            total_w = s["weight"] + contrib
            if total_w <= 1e-6:
                continue
            merge_dist = float(np.linalg.norm(pos - s["pos"]))
            merge_dists.append(merge_dist)
            s["centroid"] = (s["centroid"] * s["weight"] + pos * contrib) / total_w
            s["normal"] = s["normal"] * s["weight"] + n * contrib
            n_norm = float(np.linalg.norm(s["normal"]))
            if n_norm > 1e-6:
                s["normal"] /= n_norm
            s["weight"] = float(params.alpha) * s["weight"] + contrib
            s["obs_count"] = float(params.alpha_h) * s["obs_count"] + 1.0
            if int(s.get("last_frame", -1)) != int(frame_idx):
                s["frame_hits"] = float(params.alpha_h) * s["frame_hits"] + 1.0
                s["frame_count"] = int(s.get("frame_count", 1)) + 1
                s["last_frame"] = int(frame_idx)
            if float(residual) <= float(s["residual"]):
                # Keep the surfel anchored to the best-supported observed surface point
                # instead of drifting the output position toward an off-surface centroid.
                s["pos"] = pos.astype(np.float32)
            s["residual"] = min(s["residual"], float(residual))
            s["residual_ema"] = (
                float(params.alpha) * s["residual_ema"] + (1.0 - float(params.alpha)) * float(residual)
            )
            s["support_spread"] = max(float(s["support_spread"]), merge_dist)
            stats["merged_surfels"] += 1.0
    accepted = stats["new_surfels"] + stats["merged_surfels"]
    if accepted > 0.0:
        stats["merge_accept_rate"] = stats["merged_surfels"] / accepted
    if merge_dists:
        stats["mean_merge_dist"] = float(np.mean(merge_dists))
    return stats

def accumulate_candidate_surfels(
    frame_points: Iterable[np.ndarray],
    frame_weights: Iterable[np.ndarray],
    frame_residuals: Iterable[np.ndarray],
    params: SurfelGuardParams,
) -> list[dict]:
    surfels, grid = init_surfel_store()
    for frame_idx, (points_xyz, weights, residuals) in enumerate(zip(frame_points, frame_weights, frame_residuals)):
        accumulate_frame_into_surfels(surfels, grid, frame_idx, points_xyz, weights, residuals, params)
    return surfels


def guard_surfels(surfels: List[dict], params: SurfelGuardParams) -> list[int]:
    states: List[int] = []
    if not surfels:
        return states
    neighbor_norm = np.ones(len(surfels), dtype=np.float32)

    for s in surfels:
        frame_hits = float(s.get("frame_hits", s["hits"]))
        frame_count = int(s.get("frame_count", max(1, int(round(frame_hits)))))
        mean_e = s["weight"] / max(frame_hits, 1.0)
        score = mean_e * (1.0 + float(params.beta) * min(frame_hits, float(params.h_max)))
        score *= (1.0 + float(params.gamma_neighbor) * neighbor_norm[len(states)])
        spread_penalty = float(np.exp(-float(s.get("support_spread", 0.0)) / max(1e-6, float(params.spread_sigma))))
        centroid = s.get("centroid", s["pos"])
        centroid_drift = float(np.linalg.norm(centroid - s["pos"]))
        drift_penalty = float(np.exp(-centroid_drift / max(1e-6, float(params.drift_sigma))))
        residual_gate = float(np.exp(-float(s.get("residual_ema", s["residual"])) / max(1e-6, float(params.sigma_rho))))
        score *= spread_penalty * drift_penalty * residual_gate
        state = SURFEL_GROUNDED
        if score >= float(params.tau_p):
            state = SURFEL_PLATEAU
        if (
            score >= float(params.tau_a)
            and frame_count >= int(np.ceil(float(params.h_a)))
            and s["residual"] <= float(params.epsilon_rho)
            and centroid_drift <= float(params.drift_max)
        ):
            state = SURFEL_ASCENDED
        states.append(state)
    return states


def save_surfel_state(path: Path, surfels: list[dict], states: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pos=np.stack([s["pos"] for s in surfels]) if surfels else np.zeros((0, 3), dtype=np.float32),
        centroid=np.stack([s.get("centroid", s["pos"]) for s in surfels]) if surfels else np.zeros((0, 3), dtype=np.float32),
        normal=np.stack([s["normal"] for s in surfels]) if surfels else np.zeros((0, 3), dtype=np.float32),
        weight=np.array([s["weight"] for s in surfels], dtype=np.float32),
        hits=np.array([s["hits"] for s in surfels], dtype=np.float32),
        frame_hits=np.array([s.get("frame_hits", s["hits"]) for s in surfels], dtype=np.float32),
        frame_count=np.array([s.get("frame_count", 1) for s in surfels], dtype=np.int32),
        obs_count=np.array([s.get("obs_count", s["hits"]) for s in surfels], dtype=np.float32),
        residual=np.array([s["residual"] for s in surfels], dtype=np.float32),
        residual_ema=np.array([s.get("residual_ema", s["residual"]) for s in surfels], dtype=np.float32),
        support_spread=np.array([s.get("support_spread", 0.0) for s in surfels], dtype=np.float32),
        states=np.array(states, dtype=np.uint8),
    )


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


def write_colored_ply_ascii(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points_xyz, colors_rgb):
            f.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
