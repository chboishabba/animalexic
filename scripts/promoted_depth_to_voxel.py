#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fixed_rig_runtime import load_calibration_artifact
from voxel_guard import (
    VoxelGuardParams,
    VOXEL_ASCENDED,
    VOXEL_GROUNDED,
    VOXEL_PLATEAU,
    accumulate_candidate_voxels,
    build_grid_spec,
    guard_voxels,
    voxel_centers_from_mask,
    write_points_ply_ascii,
)


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


def _load_weight_map(path: Path) -> np.ndarray | None:
    arr = _load_gray(path)
    if arr is None:
        return None
    return arr.astype(np.float32) / 255.0


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(promoted_mask)
    if ys.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    ys = ys[:: max(1, stride)]
    xs = xs[:: max(1, stride)]

    disp = disp_map[ys, xs].astype(np.float32)
    valid_disp = disp >= float(min_disp)
    if not np.any(valid_disp):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

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
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz = (homog[:3, good] / w[good]).T.astype(np.float32)
    ys = ys[good]
    xs = xs[good]
    disp = disp[good]
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    ys = ys[finite]
    xs = xs[finite]
    disp = disp[finite]
    if xyz.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    if np.mean(xyz[:, 2]) < 0.0:
        xyz[:, 2] *= -1.0
    depth_ok = xyz[:, 2] > 0.0
    if max_depth > 0:
        depth_ok &= xyz[:, 2] <= float(max_depth)
    xyz = xyz[depth_ok]
    ys = ys[depth_ok]
    xs = xs[depth_ok]
    disp = disp[depth_ok]
    if xyz.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    return xyz, ys.astype(np.int32), xs.astype(np.int32), disp.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame-limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4, help="Pixel stride over promoted pixels")
    ap.add_argument("--voxel-size", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=0.0, help="Ignore reprojected points deeper than this; 0 disables")
    ap.add_argument("--min-disp", type=float, default=1.0, help="Ignore promoted disparity values below this high-byte approximation")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--alpha-h", dest="alpha_h", type=float, default=0.90)
    ap.add_argument("--persistence-beta", type=float, default=0.10)
    ap.add_argument("--persistence-cap", type=float, default=8.0)
    ap.add_argument("--tau-p", type=float, default=0.5)
    ap.add_argument("--tau-a", type=float, default=1.0)
    ap.add_argument("--h-a", type=float, default=2.0)
    ap.add_argument("--epsilon-rho", type=float, default=64.0)
    ap.add_argument("--ray-decay", type=float, default=0.35)
    ap.add_argument("--save-ply", action="store_true")
    args = ap.parse_args()

    artifact = load_calibration_artifact(args.calibration)
    frame_indices = _frame_indices(args.runtime_dir)
    if args.frame_limit > 0:
        frame_indices = frame_indices[: args.frame_limit]

    frame_points: list[np.ndarray] = []
    frame_weights: list[np.ndarray] = []
    frame_origin_factors: list[np.ndarray] = []
    frame_residuals: list[np.ndarray] = []
    frame_rows: list[dict] = []
    for frame_idx in frame_indices:
        promoted_pack = _load_promoted_npz(args.runtime_dir / f"promoted_depth_f{frame_idx:04d}.npz")
        conf_map = _load_gray(args.runtime_dir / f"candidate_conf_f{frame_idx:04d}.png")
        grad_map = _load_gray(args.runtime_dir / f"candidate_disp_gradient_f{frame_idx:04d}.png")
        cost_map = _load_gray(args.runtime_dir / f"candidate_cost_f{frame_idx:04d}.png")
        origin_map = _load_weight_map(args.runtime_dir / f"candidate_origin_factor_f{frame_idx:04d}.png")
        if promoted_pack is not None:
            canonical_disp_q8, promoted_mask, depth_map = promoted_pack
            points_xyz, ys, xs, disp = _reproject_points_from_disp(
                (canonical_disp_q8.astype(np.float32) / 256.0),
                promoted_mask,
                artifact.q_matrix,
                stride=max(1, args.stride),
                max_depth=float(args.max_depth),
                min_disp=float(args.min_disp),
            )
        else:
            promoted_mask = _load_mask(args.runtime_dir / f"promoted_mask_f{frame_idx:04d}.png")
            canonical_disp = _load_gray(args.runtime_dir / f"canonical_disp_f{frame_idx:04d}.png")
            if promoted_mask is None or canonical_disp is None:
                continue
            depth_map = None
            points_xyz, ys, xs, disp = _reproject_points_from_disp(
                canonical_disp,
                promoted_mask,
                artifact.q_matrix,
                stride=max(1, args.stride),
                max_depth=float(args.max_depth),
                min_disp=float(args.min_disp),
            )

        if len(points_xyz) == 0:
            frame_points.append(points_xyz)
            frame_weights.append(np.zeros((0,), dtype=np.float32))
            frame_origin_factors.append(np.zeros((0,), dtype=np.float32))
            frame_residuals.append(np.zeros((0,), dtype=np.float32))
            frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "promoted_pixels": int(np.count_nonzero(promoted_mask)),
                    "projected_points": 0,
                    "source": "npz" if promoted_pack is not None else "png",
                    "origin_mean": float(np.mean(origin_map)) if origin_map is not None and origin_map.size else 1.0,
                }
            )
            continue

        conf_vals = (
            conf_map[ys, xs].astype(np.float32) / 255.0
            if conf_map is not None
            else np.ones((len(points_xyz),), dtype=np.float32)
        )
        grad_vals = (
            grad_map[ys, xs].astype(np.float32) / 255.0
            if grad_map is not None
            else np.ones((len(points_xyz),), dtype=np.float32)
        )
        support_map = cv2.blur(promoted_mask.astype(np.float32), (3, 3))
        support_vals = support_map[ys, xs].astype(np.float32) if support_map.size else np.ones((len(points_xyz),), dtype=np.float32)
        origin_vals = (
            origin_map[ys, xs].astype(np.float32)
            if origin_map is not None
            else np.ones((len(points_xyz),), dtype=np.float32)
        )
        g_vals = (0.5 + 0.5 * grad_vals) * (0.5 + 0.5 * np.clip(support_vals, 0.0, 1.0))
        base_weights = np.clip(conf_vals * g_vals, 0.0, 1.0).astype(np.float32)
        residuals = cost_map[ys, xs].astype(np.float32) if cost_map is not None else np.zeros((len(points_xyz),), dtype=np.float32)

        frame_points.append(points_xyz)
        frame_weights.append(base_weights)
        frame_origin_factors.append(origin_vals)
        frame_residuals.append(residuals)
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "promoted_pixels": int(np.count_nonzero(promoted_mask)),
                "projected_points": int(len(points_xyz)),
                "source": "npz" if promoted_pack is not None else "png",
                "origin_mean": float(np.mean(origin_vals)) if origin_vals.size else 1.0,
                "support_mean": float(np.mean(support_vals)) if support_vals.size else 1.0,
                "base_weight_mean": float(np.mean(base_weights)) if base_weights.size else 0.0,
            }
        )

    if not frame_points or sum(len(p) for p in frame_points) == 0:
        raise SystemExit("no promoted depth points available for voxel projection")

    all_points = np.concatenate(frame_points, axis=0)
    grid_spec = build_grid_spec(all_points, voxel_size=float(args.voxel_size))
    params = VoxelGuardParams(
        alpha=float(args.alpha),
        alpha_h=float(args.alpha_h),
        beta=float(args.persistence_beta),
        h_max=float(args.persistence_cap),
        tau_p=float(args.tau_p),
        tau_a=float(args.tau_a),
        h_a=float(args.h_a),
        epsilon_rho=float(args.epsilon_rho),
        ray_decay=float(args.ray_decay),
    )
    evidence, frame_hits, score, residual = accumulate_candidate_voxels(
        grid_spec,
        frame_points,
        frame_weights,
        frame_residuals,
        params,
        frame_origin_factors=frame_origin_factors,
    )
    states = guard_voxels(
        score,
        frame_hits,
        residual,
        params,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "voxel_state.npz",
        evidence=evidence,
        support=evidence,
        frame_hits=frame_hits,
        score=score,
        residual=residual,
        states=states,
        origin=grid_spec.origin,
        voxel_size=np.asarray([grid_spec.voxel_size], dtype=np.float32),
        dims=np.asarray(grid_spec.dims, dtype=np.int32),
    )

    ascended_mask = states == VOXEL_ASCENDED
    plateau_mask = states == VOXEL_PLATEAU
    summary = {
        "runtime_dir": str(args.runtime_dir),
        "calibration_id": artifact.calibration_id,
        "frames": frame_rows,
        "grid": {
            "origin": [float(x) for x in grid_spec.origin],
            "voxel_size": float(grid_spec.voxel_size),
            "dims": [int(x) for x in grid_spec.dims],
        },
        "counts": {
            "grounded": int(np.count_nonzero(states == VOXEL_GROUNDED)),
            "plateau": int(np.count_nonzero(plateau_mask)),
            "ascended": int(np.count_nonzero(ascended_mask)),
        },
        "evidence": {
            "mean": float(np.mean(evidence[evidence > 0])) if np.any(evidence > 0) else 0.0,
            "max": float(np.max(evidence)) if evidence.size else 0.0,
        },
        "score": {
            "mean": float(np.mean(score[score > 0])) if np.any(score > 0) else 0.0,
            "max": float(np.max(score)) if score.size else 0.0,
        },
        "residual": {
            "mean": float(np.mean(residual[score > 0])) if np.any(score > 0) else 0.0,
            "max": float(np.max(residual[score > 0])) if np.any(score > 0) else 0.0,
        },
    }
    (args.output_dir / "voxel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_ply:
        write_points_ply_ascii(args.output_dir / "voxels_ascended.ply", voxel_centers_from_mask(ascended_mask, grid_spec))
        write_points_ply_ascii(args.output_dir / "voxels_plateau_or_higher.ply", voxel_centers_from_mask(plateau_mask | ascended_mask, grid_spec))

    print(f"[voxel] wrote state: {args.output_dir / 'voxel_state.npz'}")
    print(f"[voxel] wrote summary: {args.output_dir / 'voxel_summary.json'}")
    print(
        "[voxel] counts: "
        f"grounded={summary['counts']['grounded']} "
        f"plateau={summary['counts']['plateau']} "
        f"ascended={summary['counts']['ascended']}"
    )


if __name__ == "__main__":
    main()
