#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fixed_rig_runtime import load_calibration_artifact
from surfel_guard import SURFEL_ASCENDED, SURFEL_GROUNDED, SURFEL_PLATEAU, write_colored_ply_ascii, write_points_ply_ascii


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
    return xyz[depth_ok].astype(np.float32)


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
    ap.add_argument("--surfel-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame-limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-depth", type=float, default=0.0)
    ap.add_argument("--min-disp", type=float, default=1.0)
    ap.add_argument("--grounded-sample", type=int, default=5000)
    args = ap.parse_args()

    artifact = load_calibration_artifact(args.calibration)
    surfel_state_path = args.surfel_dir / "surfels_state.npz"
    if not surfel_state_path.exists():
        raise SystemExit(f"missing surfel state: {surfel_state_path}")

    surfel_blob = np.load(surfel_state_path)
    states = surfel_blob["states"].astype(np.uint8)
    pos = surfel_blob["pos"].astype(np.float32)
    centroid = surfel_blob["centroid"].astype(np.float32) if "centroid" in surfel_blob else pos
    support_spread = surfel_blob["support_spread"].astype(np.float32) if "support_spread" in surfel_blob else np.zeros((len(pos),), dtype=np.float32)
    frame_count = surfel_blob["frame_count"].astype(np.int32) if "frame_count" in surfel_blob else np.ones((len(pos),), dtype=np.int32)

    asc_mask = states == SURFEL_ASCENDED
    plat_mask = states == SURFEL_PLATEAU
    ground_mask = states == SURFEL_GROUNDED
    plat_single_mask = plat_mask & (frame_count <= 1)
    plat_multi_mask = plat_mask & (frame_count >= 2)
    ground_multi_mask = ground_mask & (frame_count >= 2)
    nonpromoted_multi_mask = plat_multi_mask | ground_multi_mask
    ascended = pos[asc_mask]
    plateau = pos[plat_mask]
    plateau_single = pos[plat_single_mask]
    plateau_multi = pos[plat_multi_mask]
    grounded = pos[ground_mask]
    grounded_multi = pos[ground_multi_mask]
    nonpromoted_multi = pos[nonpromoted_multi_mask]
    ascended_centroid = centroid[asc_mask]
    plateau_centroid = centroid[plat_mask]
    plateau_single_centroid = centroid[plat_single_mask]
    plateau_multi_centroid = centroid[plat_multi_mask]
    grounded_centroid = centroid[ground_mask]
    grounded_multi_centroid = centroid[ground_multi_mask]
    nonpromoted_multi_centroid = centroid[nonpromoted_multi_mask]

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

    grounded_eval = grounded
    grounded_centroid_eval = grounded_centroid
    grounded_spread_eval = support_spread[ground_mask]
    if grounded_eval.size > 0 and len(grounded_eval) > int(args.grounded_sample):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(grounded_eval), size=int(args.grounded_sample), replace=False)
        grounded_eval = grounded_eval[idx]
        grounded_centroid_eval = grounded_centroid_eval[idx]
        grounded_spread_eval = grounded_spread_eval[idx]

    ascended_anchor_dist = _nearest_neighbor_distances(ascended, promoted_cloud)
    plateau_anchor_dist = _nearest_neighbor_distances(plateau, promoted_cloud)
    plateau_single_anchor_dist = _nearest_neighbor_distances(plateau_single, promoted_cloud)
    plateau_multi_anchor_dist = _nearest_neighbor_distances(plateau_multi, promoted_cloud)
    grounded_anchor_dist = _nearest_neighbor_distances(grounded_eval, promoted_cloud)
    grounded_multi_anchor_dist = _nearest_neighbor_distances(grounded_multi, promoted_cloud)
    nonpromoted_multi_anchor_dist = _nearest_neighbor_distances(nonpromoted_multi, promoted_cloud)

    ascended_centroid_dist = _nearest_neighbor_distances(ascended_centroid, promoted_cloud)
    plateau_centroid_dist = _nearest_neighbor_distances(plateau_centroid, promoted_cloud)
    plateau_single_centroid_dist = _nearest_neighbor_distances(plateau_single_centroid, promoted_cloud)
    plateau_multi_centroid_dist = _nearest_neighbor_distances(plateau_multi_centroid, promoted_cloud)
    grounded_centroid_dist = _nearest_neighbor_distances(grounded_centroid_eval, promoted_cloud)
    grounded_multi_centroid_dist = _nearest_neighbor_distances(grounded_multi_centroid, promoted_cloud)
    nonpromoted_multi_centroid_dist = _nearest_neighbor_distances(nonpromoted_multi_centroid, promoted_cloud)

    asc_anchor_stats = _stats(ascended_anchor_dist)
    plateau_anchor_stats = _stats(plateau_anchor_dist)
    plateau_single_anchor_stats = _stats(plateau_single_anchor_dist)
    plateau_multi_anchor_stats = _stats(plateau_multi_anchor_dist)
    grounded_anchor_stats = _stats(grounded_anchor_dist)
    grounded_multi_anchor_stats = _stats(grounded_multi_anchor_dist)
    nonpromoted_multi_anchor_stats = _stats(nonpromoted_multi_anchor_dist)
    asc_centroid_stats = _stats(ascended_centroid_dist)
    plateau_centroid_stats = _stats(plateau_centroid_dist)
    plateau_single_centroid_stats = _stats(plateau_single_centroid_dist)
    plateau_multi_centroid_stats = _stats(plateau_multi_centroid_dist)
    grounded_centroid_stats = _stats(grounded_centroid_dist)
    grounded_multi_centroid_stats = _stats(grounded_multi_centroid_dist)
    nonpromoted_multi_centroid_stats = _stats(nonpromoted_multi_centroid_dist)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_points_ply_ascii(args.output_dir / "promoted_points.ply", promoted_cloud)
    write_points_ply_ascii(args.output_dir / "ascended_surfels.ply", ascended)
    write_points_ply_ascii(args.output_dir / "plateau_surfels.ply", plateau)

    overlay_points = []
    overlay_colors = []
    if promoted_cloud.size:
        overlay_points.append(promoted_cloud)
        overlay_colors.append(np.tile(np.array([[80, 80, 255]], dtype=np.uint8), (len(promoted_cloud), 1)))
    if plateau.size:
        overlay_points.append(plateau)
        overlay_colors.append(np.tile(np.array([[255, 180, 60]], dtype=np.uint8), (len(plateau), 1)))
    if ascended.size:
        overlay_points.append(ascended)
        overlay_colors.append(np.tile(np.array([[60, 220, 80]], dtype=np.uint8), (len(ascended), 1)))
    if overlay_points:
        overlay_pts = np.concatenate(overlay_points, axis=0)
        overlay_cols = np.concatenate(overlay_colors, axis=0)
        write_colored_ply_ascii(args.output_dir / "surfel_overlay.ply", overlay_pts, overlay_cols)

    if nonpromoted_multi_centroid_stats["count"] > 0:
        asc_better = asc_centroid_stats["mean"] < nonpromoted_multi_centroid_stats["mean"]
        centroid_margin = nonpromoted_multi_centroid_stats["mean"] - asc_centroid_stats["mean"]
        recommendation = (
            "ascended surfels are tighter than the non-promoted multi-frame support set; proceed with controlled densification"
            if asc_better
            else "ascended surfels are not yet tighter than the non-promoted multi-frame support set; hold densification and tighten guard/support first"
        )
    else:
        asc_better = False
        centroid_margin = None
        recommendation = "no non-promoted multi-frame comparison set exists yet; keep current guard and grow only after another candidate class appears"
    summary = {
        "runtime_dir": str(args.runtime_dir),
        "surfel_dir": str(args.surfel_dir),
        "calibration_id": artifact.calibration_id,
        "counts": {
            "grounded": int(np.count_nonzero(ground_mask)),
            "plateau": int(np.count_nonzero(plat_mask)),
            "plateau_singleton": int(np.count_nonzero(plat_single_mask)),
            "plateau_multiframe": int(np.count_nonzero(plat_multi_mask)),
            "grounded_multiframe": int(np.count_nonzero(ground_multi_mask)),
            "nonpromoted_multiframe": int(np.count_nonzero(nonpromoted_multi_mask)),
            "ascended": int(np.count_nonzero(asc_mask)),
            "promoted_points": int(len(promoted_cloud)),
        },
        "residuals_to_promoted_cloud": {
            "anchor": {
                "ascended": asc_anchor_stats,
                "plateau": plateau_anchor_stats,
                "plateau_singleton": plateau_single_anchor_stats,
                "plateau_multiframe": plateau_multi_anchor_stats,
                "grounded_multiframe": grounded_multi_anchor_stats,
                "nonpromoted_multiframe": nonpromoted_multi_anchor_stats,
                "grounded_sample": grounded_anchor_stats,
            },
            "centroid": {
                "ascended": asc_centroid_stats,
                "plateau": plateau_centroid_stats,
                "plateau_singleton": plateau_single_centroid_stats,
                "plateau_multiframe": plateau_multi_centroid_stats,
                "grounded_multiframe": grounded_multi_centroid_stats,
                "nonpromoted_multiframe": nonpromoted_multi_centroid_stats,
                "grounded_sample": grounded_centroid_stats,
            },
        },
        "support_spread": {
            "ascended": _stats(support_spread[asc_mask]),
            "plateau": _stats(support_spread[plat_mask]),
            "plateau_singleton": _stats(support_spread[plat_single_mask]),
            "plateau_multiframe": _stats(support_spread[plat_multi_mask]),
            "grounded_multiframe": _stats(support_spread[ground_multi_mask]),
            "nonpromoted_multiframe": _stats(support_spread[nonpromoted_multi_mask]),
            "grounded_sample": _stats(grounded_spread_eval),
        },
        "governance": {
            "ascended_beats_nonpromoted_multiframe": bool(asc_better),
            "centroid_margin_vs_nonpromoted_multiframe": centroid_margin,
        },
        "recommendation": recommendation,
    }
    (args.output_dir / "surfel_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[surfel-quality] wrote summary: {args.output_dir / 'surfel_quality.json'}")
    print(
        "[surfel-quality] centroid residuals: "
        f"ascended_mean={asc_centroid_stats['mean']:.4f} "
        f"nonpromoted_multiframe_mean={nonpromoted_multi_centroid_stats['mean']:.4f} "
        f"plateau_singleton_mean={plateau_single_centroid_stats['mean']:.4f}"
    )
    print(
        "[surfel-quality] counts: "
        f"ascended={summary['counts']['ascended']} "
        f"plateau={summary['counts']['plateau']} "
        f"promoted_points={summary['counts']['promoted_points']}"
    )


if __name__ == "__main__":
    main()
