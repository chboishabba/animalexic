#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fixed_rig_runtime import load_calibration_artifact
from voxel_guard import VoxelGridSpec, voxel_centers_from_mask


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return (arr > 0).astype(np.uint8)


def _load_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return arr


def _load_promoted_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    blob = np.load(path)
    return (
        blob["canonical_disp_q8"].astype(np.uint16),
        blob["promoted_mask"].astype(np.uint8),
        blob["depth"].astype(np.float32),
    )


def _frame_indices(runtime_dir: Path) -> list[int]:
    out = []
    for path in runtime_dir.glob("promoted_mask_f*.png"):
        stem = path.stem
        try:
            out.append(int(stem.split("_f", 1)[1]))
        except Exception:
            continue
    return sorted(set(out))


def _reproject_points_from_disp(
    disp_map: np.ndarray,
    promoted_mask: np.ndarray,
    q_matrix: np.ndarray,
    *,
    stride: int,
    max_depth: float,
    min_disp: float,
) -> np.ndarray:
    ys, xs = np.nonzero(promoted_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ys = ys[:: max(1, stride)]
    xs = xs[:: max(1, stride)]

    disp = disp_map[ys, xs].astype(np.float32)
    valid_disp = disp >= float(min_disp)
    if not np.any(valid_disp):
        return np.zeros((0, 3), dtype=np.float32)

    ys = ys[valid_disp]
    xs = xs[valid_disp]
    disp = disp[valid_disp]

    q = q_matrix.astype(np.float32)
    vec = np.stack(
        [
            xs.astype(np.float32),
            ys.astype(np.float32),
            disp,
            np.ones_like(disp, dtype=np.float32),
        ],
        axis=0,
    )
    homog = q @ vec
    w = homog[3]
    good = np.abs(w) > 1e-6
    if not np.any(good):
        return np.zeros((0, 3), dtype=np.float32)

    xyz = (homog[:3, good] / w[good]).T.astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if np.mean(xyz[:, 2]) < 0.0:
        xyz[:, 2] *= -1.0
    depth_ok = xyz[:, 2] > 0.0
    if max_depth > 0:
        depth_ok &= xyz[:, 2] <= float(max_depth)
    xyz = xyz[depth_ok]
    return xyz.astype(np.float32)


def _write_points_ply_ascii(path: Path, points_xyz: np.ndarray) -> None:
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


def _write_colored_ply_ascii(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
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


def _nearest_neighbor_distances(query: np.ndarray, reference: np.ndarray, *, batch: int = 1024) -> np.ndarray:
    if query.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if reference.size == 0:
        return np.full((len(query),), np.inf, dtype=np.float32)
    dists = np.empty((len(query),), dtype=np.float32)
    ref = reference.astype(np.float32)
    for start in range(0, len(query), batch):
        end = min(start + batch, len(query))
        chunk = query[start:end].astype(np.float32)
        diff = chunk[:, None, :] - ref[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        dists[start:end] = np.sqrt(np.min(sq, axis=1))
    return dists


def _stats(distances: np.ndarray) -> dict:
    if distances.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {
            "count": int(distances.size),
            "mean": float("inf"),
            "median": float("inf"),
            "p90": float("inf"),
            "max": float("inf"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--voxel-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame-limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-depth", type=float, default=0.0)
    ap.add_argument("--min-disp", type=float, default=1.0)
    ap.add_argument("--grounded-sample", type=int, default=5000)
    args = ap.parse_args()

    artifact = load_calibration_artifact(args.calibration)
    voxel_state_path = args.voxel_dir / "voxel_state.npz"
    if not voxel_state_path.exists():
        raise SystemExit(f"missing voxel state: {voxel_state_path}")

    voxel_blob = np.load(voxel_state_path)
    states = voxel_blob["states"].astype(np.uint8)
    origin = voxel_blob["origin"].astype(np.float32)
    voxel_size = float(voxel_blob["voxel_size"][0])
    dims = tuple(int(x) for x in voxel_blob["dims"])
    grid_spec = VoxelGridSpec(origin=origin, voxel_size=voxel_size, dims=dims)

    ascended_mask = states == 2
    plateau_mask = states == 1
    grounded_mask = states == 0
    ascended_centers = voxel_centers_from_mask(ascended_mask, grid_spec)
    plateau_centers = voxel_centers_from_mask(plateau_mask, grid_spec)
    grounded_centers = voxel_centers_from_mask(grounded_mask, grid_spec)

    frame_indices = _frame_indices(args.runtime_dir)
    if args.frame_limit > 0:
        frame_indices = frame_indices[: args.frame_limit]

    promoted_points = []
    for frame_idx in frame_indices:
        promoted_pack = _load_promoted_npz(args.runtime_dir / f"promoted_depth_f{frame_idx:04d}.npz")
        promoted_mask = None
        disp_map = None
        if promoted_pack is not None:
            canonical_disp_q8, promoted_mask, _depth = promoted_pack
            disp_map = canonical_disp_q8.astype(np.float32) / 256.0
        else:
            promoted_mask = _load_mask(args.runtime_dir / f"promoted_mask_f{frame_idx:04d}.png")
            disp_map = _load_gray(args.runtime_dir / f"canonical_disp_f{frame_idx:04d}.png")
        if promoted_mask is None or disp_map is None:
            continue
        pts = _reproject_points_from_disp(
            disp_map,
            promoted_mask,
            artifact.q_matrix,
            stride=max(1, args.stride),
            max_depth=float(args.max_depth),
            min_disp=float(args.min_disp),
        )
        if pts.size:
            promoted_points.append(pts)

    promoted_cloud = np.concatenate(promoted_points, axis=0) if promoted_points else np.zeros((0, 3), dtype=np.float32)

    grounded_eval = grounded_centers
    if grounded_eval.size > 0 and len(grounded_eval) > int(args.grounded_sample):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(grounded_eval), size=int(args.grounded_sample), replace=False)
        grounded_eval = grounded_eval[idx]

    ascended_dist = _nearest_neighbor_distances(ascended_centers, promoted_cloud)
    plateau_dist = _nearest_neighbor_distances(plateau_centers, promoted_cloud)
    grounded_dist = _nearest_neighbor_distances(grounded_eval, promoted_cloud)

    asc_stats = _stats(ascended_dist)
    plateau_stats = _stats(plateau_dist)
    grounded_stats = _stats(grounded_dist)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_points_ply_ascii(args.output_dir / "promoted_points.ply", promoted_cloud)
    _write_points_ply_ascii(args.output_dir / "ascended_voxels.ply", ascended_centers)
    _write_points_ply_ascii(args.output_dir / "plateau_voxels.ply", plateau_centers)

    overlay_points = []
    overlay_colors = []
    if promoted_cloud.size:
        overlay_points.append(promoted_cloud)
        overlay_colors.append(np.tile(np.array([[80, 80, 255]], dtype=np.uint8), (len(promoted_cloud), 1)))
    if plateau_centers.size:
        overlay_points.append(plateau_centers)
        overlay_colors.append(np.tile(np.array([[255, 180, 60]], dtype=np.uint8), (len(plateau_centers), 1)))
    if ascended_centers.size:
        overlay_points.append(ascended_centers)
        overlay_colors.append(np.tile(np.array([[60, 220, 80]], dtype=np.uint8), (len(ascended_centers), 1)))
    if overlay_points:
        overlay_pts = np.concatenate(overlay_points, axis=0)
        overlay_cols = np.concatenate(overlay_colors, axis=0)
        _write_colored_ply_ascii(args.output_dir / "voxel_overlay.ply", overlay_pts, overlay_cols)

    summary = {
        "runtime_dir": str(args.runtime_dir),
        "voxel_dir": str(args.voxel_dir),
        "calibration_id": artifact.calibration_id,
        "grid": {
            "origin": [float(x) for x in origin],
            "voxel_size": float(voxel_size),
            "dims": [int(x) for x in dims],
        },
        "counts": {
            "grounded": int(np.count_nonzero(grounded_mask)),
            "plateau": int(np.count_nonzero(plateau_mask)),
            "ascended": int(np.count_nonzero(ascended_mask)),
            "promoted_points": int(len(promoted_cloud)),
        },
        "residuals_to_promoted_cloud": {
            "ascended": asc_stats,
            "plateau": plateau_stats,
            "grounded_sample": grounded_stats,
        },
        "recommendation": (
            "ascended voxels look coherent; consider lowering tau_a slightly"
            if asc_stats["mean"] < plateau_stats["mean"] or not np.isfinite(plateau_stats["mean"])
            else "ascended voxels are not yet tighter than plateau; increase h_a before lowering tau_a"
        ),
    }
    (args.output_dir / "voxel_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[voxel-quality] wrote summary: {args.output_dir / 'voxel_quality.json'}")
    print(
        "[voxel-quality] residuals to promoted cloud: "
        f"ascended_mean={asc_stats['mean']:.4f} "
        f"plateau_mean={plateau_stats['mean']:.4f} "
        f"grounded_sample_mean={grounded_stats['mean']:.4f}"
    )
    print(
        "[voxel-quality] counts: "
        f"ascended={summary['counts']['ascended']} "
        f"plateau={summary['counts']['plateau']} "
        f"promoted_points={summary['counts']['promoted_points']}"
    )


if __name__ == "__main__":
    main()
