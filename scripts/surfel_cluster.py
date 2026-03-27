#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from surfel_guard import SURFEL_ASCENDED, SURFEL_PLATEAU, write_colored_ply_ascii


def _stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _fit_local_geometry(points: np.ndarray, neighbors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros_like(points, dtype=np.float32)
    curvature = np.zeros((len(points),), dtype=np.float32)
    default_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for idx, nbr in enumerate(neighbors):
        if nbr.size < 3:
            normals[idx] = default_normal
            curvature[idx] = 1.0
            continue
        local = points[nbr]
        center = np.mean(local, axis=0, dtype=np.float32)
        centered = local - center[None, :]
        cov = (centered.T @ centered) / max(len(local) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float32))
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        normal = eigvecs[:, order[0]].astype(np.float32)
        if normal[2] < 0.0:
            normal *= -1.0
        norm = float(np.linalg.norm(normal))
        normals[idx] = normal / norm if norm > 1e-6 else default_normal
        denom = float(np.sum(eigvals))
        curvature[idx] = float(eigvals[0] / denom) if denom > 1e-8 else 1.0
    return normals, curvature


def _neighbor_lists(points: np.ndarray, radius: float, max_neighbors: int) -> list[np.ndarray]:
    if points.size == 0:
        return []
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    neighbors: list[np.ndarray] = []
    for idx in range(len(points)):
        mask = (dist[idx] <= float(radius)) & (dist[idx] > 0.0)
        nbr = np.nonzero(mask)[0]
        if nbr.size > int(max_neighbors):
            order = np.argsort(dist[idx, nbr])
            nbr = nbr[order[: int(max_neighbors)]]
        nbr = np.concatenate([[idx], nbr]).astype(np.int32)
        neighbors.append(nbr)
    return neighbors


def _cluster_graph(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    *,
    edge_radius: float,
    normal_angle_deg: float,
    curvature_max_diff: float,
) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dot = np.clip(normals @ normals.T, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    curv_diff = np.abs(curvature[:, None] - curvature[None, :])

    visited = np.zeros((n,), dtype=bool)
    labels = np.full((n,), -1, dtype=np.int32)
    next_label = 0
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        labels[root] = next_label
        q: deque[int] = deque([root])
        while q:
            idx = q.popleft()
            mask = (
                (dist[idx] <= float(edge_radius))
                & (dist[idx] > 0.0)
                & (angle[idx] <= float(normal_angle_deg))
                & (curv_diff[idx] <= float(curvature_max_diff))
            )
            nbrs = np.nonzero(mask)[0]
            for nbr in nbrs:
                if visited[nbr]:
                    continue
                visited[nbr] = True
                labels[nbr] = next_label
                q.append(int(nbr))
        next_label += 1
    return labels


def _relabel_min_size(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if labels.size == 0:
        return labels
    counts: dict[int, int] = {}
    for label in labels.tolist():
        counts[label] = counts.get(label, 0) + 1
    out = labels.copy()
    for label, count in counts.items():
        if count < int(min_cluster_size):
            out[out == label] = -1
    valid = sorted({int(x) for x in out.tolist() if int(x) >= 0})
    remap = {label: idx for idx, label in enumerate(valid)}
    for old, new in remap.items():
        out[out == old] = new
    return out


def _cluster_palette(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    base = np.array(
        [
            [60, 220, 80],
            [255, 180, 60],
            [80, 180, 255],
            [210, 120, 255],
            [255, 110, 110],
            [120, 240, 220],
            [220, 220, 90],
            [255, 140, 220],
        ],
        dtype=np.uint8,
    )
    for idx, label in enumerate(labels.tolist()):
        if int(label) < 0:
            colors[idx] = np.array([90, 90, 90], dtype=np.uint8)
        else:
            colors[idx] = base[int(label) % len(base)]
    return colors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfel-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--include-plateau", action="store_true", help="Cluster ascended + plateau instead of ascended only")
    ap.add_argument("--neighbor-radius", type=float, default=0.25)
    ap.add_argument("--max-neighbors", type=int, default=16)
    ap.add_argument("--edge-radius", type=float, default=0.18)
    ap.add_argument("--normal-angle-deg", type=float, default=35.0)
    ap.add_argument("--curvature-max-diff", type=float, default=0.08)
    ap.add_argument("--min-cluster-size", type=int, default=6)
    args = ap.parse_args()

    blob_path = args.surfel_dir / "surfels_state.npz"
    if not blob_path.exists():
        raise SystemExit(f"missing surfel state: {blob_path}")
    blob = np.load(blob_path)
    pos = blob["pos"].astype(np.float32)
    states = blob["states"].astype(np.uint8)
    support_spread = blob["support_spread"].astype(np.float32) if "support_spread" in blob else np.zeros((len(pos),), dtype=np.float32)
    residual = blob["residual_ema"].astype(np.float32) if "residual_ema" in blob else blob["residual"].astype(np.float32)
    frame_count = blob["frame_count"].astype(np.int32) if "frame_count" in blob else np.ones((len(pos),), dtype=np.int32)

    keep_mask = states == SURFEL_ASCENDED
    if args.include_plateau:
        keep_mask |= states == SURFEL_PLATEAU
    points = pos[keep_mask]
    kept_states = states[keep_mask]
    kept_spread = support_spread[keep_mask]
    kept_residual = residual[keep_mask]
    kept_frame_count = frame_count[keep_mask]

    if points.size == 0:
        raise SystemExit("no surfels available for clustering under the chosen filter")

    neighbors = _neighbor_lists(points, float(args.neighbor_radius), int(args.max_neighbors))
    normals, curvature = _fit_local_geometry(points, neighbors)
    labels = _cluster_graph(
        points,
        normals,
        curvature,
        edge_radius=float(args.edge_radius),
        normal_angle_deg=float(args.normal_angle_deg),
        curvature_max_diff=float(args.curvature_max_diff),
    )
    labels = _relabel_min_size(labels, int(args.min_cluster_size))

    summary_clusters = []
    for label in sorted({int(x) for x in labels.tolist() if int(x) >= 0}):
        mask = labels == label
        pts = points[mask]
        summary_clusters.append(
            {
                "label": int(label),
                "count": int(np.count_nonzero(mask)),
                "centroid": np.mean(pts, axis=0, dtype=np.float32).tolist(),
                "bbox_min": np.min(pts, axis=0).tolist(),
                "bbox_max": np.max(pts, axis=0).tolist(),
                "residual": _stats(kept_residual[mask]),
                "support_spread": _stats(kept_spread[mask]),
                "frame_count": _stats(kept_frame_count[mask].astype(np.float32)),
                "state_counts": {
                    "ascended": int(np.count_nonzero(kept_states[mask] == SURFEL_ASCENDED)),
                    "plateau": int(np.count_nonzero(kept_states[mask] == SURFEL_PLATEAU)),
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "surfel_clusters.npz",
        pos=points,
        states=kept_states,
        labels=labels,
        normals=normals,
        curvature=curvature,
        residual=kept_residual,
        support_spread=kept_spread,
        frame_count=kept_frame_count,
    )
    summary = {
        "surfel_dir": str(args.surfel_dir),
        "filter": "ascended+plateau" if args.include_plateau else "ascended",
        "params": {
            "neighbor_radius": float(args.neighbor_radius),
            "max_neighbors": int(args.max_neighbors),
            "edge_radius": float(args.edge_radius),
            "normal_angle_deg": float(args.normal_angle_deg),
            "curvature_max_diff": float(args.curvature_max_diff),
            "min_cluster_size": int(args.min_cluster_size),
        },
        "counts": {
            "input_surfels": int(len(points)),
            "clusters": int(len(summary_clusters)),
            "noise": int(np.count_nonzero(labels < 0)),
        },
        "clusters": summary_clusters,
    }
    (args.output_dir / "surfel_clusters_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_colored_ply_ascii(args.output_dir / "surfel_clusters.ply", points, _cluster_palette(labels))

    print(f"[surfel-cluster] wrote state: {args.output_dir / 'surfel_clusters.npz'}")
    print(f"[surfel-cluster] wrote summary: {args.output_dir / 'surfel_clusters_summary.json'}")
    print(f"[surfel-cluster] wrote ply: {args.output_dir / 'surfel_clusters.ply'}")
    print(
        f"[surfel-cluster] counts: input={summary['counts']['input_surfels']} "
        f"clusters={summary['counts']['clusters']} noise={summary['counts']['noise']}"
    )


if __name__ == "__main__":
    main()
