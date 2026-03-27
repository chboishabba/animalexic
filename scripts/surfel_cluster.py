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


def _neighbor_lists(points: np.ndarray, radius: float, max_neighbors: int) -> list[np.ndarray]:
    if points.size == 0:
        return []
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    neighbors: list[np.ndarray] = []
    for idx in range(len(points)):
        nbr = np.nonzero((dist[idx] <= float(radius)) & (dist[idx] > 0.0))[0]
        if nbr.size > int(max_neighbors):
            order = np.argsort(dist[idx, nbr])
            nbr = nbr[order[: int(max_neighbors)]]
        neighbors.append(np.concatenate([[idx], nbr]).astype(np.int32))
    return neighbors


def _fit_local_geometry(points: np.ndarray, neighbors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros_like(points, dtype=np.float32)
    curvature = np.zeros((len(points),), dtype=np.float32)
    fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for idx, nbr in enumerate(neighbors):
        if nbr.size < 3:
            normals[idx] = fallback
            curvature[idx] = 1.0
            continue
        local = points[nbr]
        centered = local - np.mean(local, axis=0, dtype=np.float32)[None, :]
        cov = (centered.T @ centered) / max(len(local) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float32))
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        normal = eigvecs[:, order[0]].astype(np.float32)
        if normal[2] < 0:
            normal *= -1.0
        norm = float(np.linalg.norm(normal))
        normals[idx] = normal / norm if norm > 1e-6 else fallback
        denom = float(np.sum(eigvals))
        curvature[idx] = float(eigvals[0] / denom) if denom > 1e-8 else 1.0
    return normals, curvature


def _component_labels(
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

    labels = np.full((n,), -1, dtype=np.int32)
    visited = np.zeros((n,), dtype=bool)
    next_label = 0
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        labels[root] = next_label
        q: deque[int] = deque([root])
        while q:
            idx = q.popleft()
            nbrs = np.nonzero(
                (dist[idx] <= float(edge_radius))
                & (dist[idx] > 0.0)
                & (angle[idx] <= float(normal_angle_deg))
                & (curv_diff[idx] <= float(curvature_max_diff))
            )[0]
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
    out = labels.copy()
    counts: dict[int, int] = {}
    for label in out.tolist():
        counts[label] = counts.get(label, 0) + 1
    for label, count in counts.items():
        if count < int(min_cluster_size):
            out[out == label] = -1
    valid = sorted({int(v) for v in out.tolist() if int(v) >= 0})
    mapping = {old: new for new, old in enumerate(valid)}
    for old, new in mapping.items():
        out[out == old] = new
    return out


def _cluster_palette(labels: np.ndarray, selected_label: int | None = None) -> np.ndarray:
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
        elif selected_label is not None and int(label) == int(selected_label):
            colors[idx] = np.array([40, 255, 120], dtype=np.uint8)
        else:
            colors[idx] = base[int(label) % len(base)]
    return colors


def _cluster_score(core_count: int, plateau_count: int, frame_mean: float, residual_mean: float, spread_mean: float) -> float:
    return (
        (2.0 * float(core_count))
        + (0.5 * float(plateau_count))
        + (0.5 * float(frame_mean))
        - (0.05 * float(residual_mean))
        - (8.0 * float(spread_mean))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfel-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expand-plateau", action="store_true", help="Attach nearby plateau surfels to ascended core clusters")
    ap.add_argument("--neighbor-radius", type=float, default=0.25)
    ap.add_argument("--max-neighbors", type=int, default=16)
    ap.add_argument("--edge-radius", type=float, default=0.18)
    ap.add_argument("--normal-angle-deg", type=float, default=35.0)
    ap.add_argument("--curvature-max-diff", type=float, default=0.08)
    ap.add_argument("--min-cluster-size", type=int, default=6)
    ap.add_argument("--expand-radius", type=float, default=0.22)
    ap.add_argument("--expand-normal-angle-deg", type=float, default=40.0)
    args = ap.parse_args()

    blob_path = args.surfel_dir / "surfels_state.npz"
    if not blob_path.exists():
        raise SystemExit(f"missing surfel state: {blob_path}")
    blob = np.load(blob_path)
    pos = blob["pos"].astype(np.float32)
    states = blob["states"].astype(np.uint8)
    surfel_normals = blob["normal"].astype(np.float32)
    support_spread = blob["support_spread"].astype(np.float32) if "support_spread" in blob else np.zeros((len(pos),), dtype=np.float32)
    residual = blob["residual_ema"].astype(np.float32) if "residual_ema" in blob else blob["residual"].astype(np.float32)
    frame_count = blob["frame_count"].astype(np.int32) if "frame_count" in blob else np.ones((len(pos),), dtype=np.int32)
    weights = blob["weight"].astype(np.float32) if "weight" in blob else np.ones((len(pos),), dtype=np.float32)

    asc_mask = states == SURFEL_ASCENDED
    plateau_mask = states == SURFEL_PLATEAU
    core_points = pos[asc_mask]
    if core_points.size == 0:
        raise SystemExit("no ascended surfels available for clustering")
    core_neighbors = _neighbor_lists(core_points, float(args.neighbor_radius), int(args.max_neighbors))
    core_normals, core_curvature = _fit_local_geometry(core_points, core_neighbors)
    core_labels = _component_labels(
        core_points,
        core_normals,
        core_curvature,
        edge_radius=float(args.edge_radius),
        normal_angle_deg=float(args.normal_angle_deg),
        curvature_max_diff=float(args.curvature_max_diff),
    )
    core_labels = _relabel_min_size(core_labels, int(args.min_cluster_size))

    expanded_labels = np.full((len(pos),), -1, dtype=np.int32)
    expanded_labels[asc_mask] = core_labels

    selected_clusters = sorted({int(x) for x in core_labels.tolist() if int(x) >= 0})
    cluster_rows: list[dict[str, object]] = []
    if args.expand_plateau and selected_clusters:
        expand_idx = np.nonzero(plateau_mask)[0]
        core_idx = np.nonzero(asc_mask)[0]
        for p_idx in expand_idx.tolist():
            p = pos[p_idx]
            p_n = surfel_normals[p_idx].astype(np.float32)
            best_label = -1
            best_dist = float(args.expand_radius)
            for c_local, c_global in enumerate(core_idx.tolist()):
                c_label = int(core_labels[c_local])
                if c_label < 0:
                    continue
                dist = float(np.linalg.norm(p - pos[c_global]))
                if dist >= best_dist:
                    continue
                dot = float(np.clip(np.dot(p_n, core_normals[c_local]), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(dot)))
                if ang <= float(args.expand_normal_angle_deg):
                    best_dist = dist
                    best_label = c_label
            if best_label >= 0:
                expanded_labels[p_idx] = best_label

    for label in selected_clusters:
        mask = expanded_labels == label
        pts = pos[mask]
        cluster_states = states[mask]
        core_count = int(np.count_nonzero(cluster_states == SURFEL_ASCENDED))
        plateau_count = int(np.count_nonzero(cluster_states == SURFEL_PLATEAU))
        residual_mean = float(np.mean(residual[mask])) if np.any(mask) else 0.0
        spread_mean = float(np.mean(support_spread[mask])) if np.any(mask) else 0.0
        frame_mean = float(np.mean(frame_count[mask])) if np.any(mask) else 0.0
        score = _cluster_score(core_count, plateau_count, frame_mean, residual_mean, spread_mean)
        cluster_rows.append(
            {
                "label": int(label),
                "count": int(np.count_nonzero(mask)),
                "core_count": core_count,
                "plateau_count": plateau_count,
                "centroid": np.mean(pts, axis=0, dtype=np.float32).tolist(),
                "bbox_min": np.min(pts, axis=0).tolist(),
                "bbox_max": np.max(pts, axis=0).tolist(),
                "score": float(score),
                "residual": _stats(residual[mask]),
                "support_spread": _stats(support_spread[mask]),
                "frame_count": _stats(frame_count[mask].astype(np.float32)),
                "weight": _stats(weights[mask]),
            }
        )

    selected_label = None
    if cluster_rows:
        selected_label = int(max(cluster_rows, key=lambda row: float(row["score"]))["label"])
    selected_mask = expanded_labels == selected_label if selected_label is not None else np.zeros((len(pos),), dtype=bool)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "surfel_clusters.npz",
        pos=pos,
        states=states,
        normals=surfel_normals,
        weights=weights,
        residual=residual,
        support_spread=support_spread,
        frame_count=frame_count,
        labels=expanded_labels,
        selected_mask=selected_mask.astype(np.uint8),
    )
    summary = {
        "surfel_dir": str(args.surfel_dir),
        "mode": "ascended_core_expand_plateau" if args.expand_plateau else "ascended_core_only",
        "params": {
            "neighbor_radius": float(args.neighbor_radius),
            "max_neighbors": int(args.max_neighbors),
            "edge_radius": float(args.edge_radius),
            "normal_angle_deg": float(args.normal_angle_deg),
            "curvature_max_diff": float(args.curvature_max_diff),
            "min_cluster_size": int(args.min_cluster_size),
            "expand_radius": float(args.expand_radius),
            "expand_normal_angle_deg": float(args.expand_normal_angle_deg),
        },
        "counts": {
            "ascended_input": int(np.count_nonzero(asc_mask)),
            "plateau_input": int(np.count_nonzero(plateau_mask)),
            "clusters": int(len(cluster_rows)),
            "noise": int(np.count_nonzero(expanded_labels < 0)),
            "selected_object": int(np.count_nonzero(selected_mask)),
        },
        "selected_label": selected_label,
        "clusters": cluster_rows,
    }
    (args.output_dir / "surfel_clusters_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_colored_ply_ascii(args.output_dir / "surfel_clusters.ply", pos, _cluster_palette(expanded_labels, selected_label))
    if np.any(selected_mask):
        write_colored_ply_ascii(
            args.output_dir / "surfel_selected_object.ply",
            pos[selected_mask],
            np.tile(np.array([[40, 255, 120]], dtype=np.uint8), (int(np.count_nonzero(selected_mask)), 1)),
        )

    print(f"[surfel-cluster] wrote state: {args.output_dir / 'surfel_clusters.npz'}")
    print(f"[surfel-cluster] wrote summary: {args.output_dir / 'surfel_clusters_summary.json'}")
    print(f"[surfel-cluster] wrote ply: {args.output_dir / 'surfel_clusters.ply'}")
    print(
        f"[surfel-cluster] counts: ascended={summary['counts']['ascended_input']} "
        f"plateau={summary['counts']['plateau_input']} clusters={summary['counts']['clusters']} "
        f"selected={summary['counts']['selected_object']} noise={summary['counts']['noise']}"
    )


if __name__ == "__main__":
    main()
