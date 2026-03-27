#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fixed_rig_runtime import load_calibration_artifact
from surfel_guard import (
    SURFEL_ASCENDED,
    SURFEL_GROUNDED,
    SURFEL_PLATEAU,
    SurfelGuardParams,
    accumulate_candidate_surfels,
    guard_surfels,
    write_colored_ply_ascii,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame-limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-depth", type=float, default=0.0)
    ap.add_argument("--min-disp", type=float, default=1.0)
    ap.add_argument("--cell-size", type=float, default=0.25)
    ap.add_argument("--pos-eps", type=float, default=0.20)
    ap.add_argument("--normal-eps", type=float, default=0.52)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--alpha-h", dest="alpha_h", type=float, default=0.90)
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--h-max", dest="h_max", type=float, default=8.0)
    ap.add_argument("--tau-p", type=float, default=0.5)
    ap.add_argument("--tau-a", type=float, default=1.0)
    ap.add_argument("--h-a", type=float, default=2.0)
    ap.add_argument("--epsilon-rho", type=float, default=64.0)
    ap.add_argument("--sigma-rho", type=float, default=24.0)
    ap.add_argument("--gamma-neighbor", dest="gamma_neighbor", type=float, default=0.20)
    ap.add_argument("--save-ply", action="store_true")
    args = ap.parse_args()

    artifact = load_calibration_artifact(args.calibration)
    frame_indices = _frame_indices(args.runtime_dir)
    if args.frame_limit > 0:
        frame_indices = frame_indices[: args.frame_limit]

    frame_points: list[np.ndarray] = []
    frame_weights: list[np.ndarray] = []
    frame_residuals: list[np.ndarray] = []
    for frame_idx in frame_indices:
        promoted_pack = _load_promoted_npz(args.runtime_dir / f"promoted_depth_f{frame_idx:04d}.npz")
        conf_map = _load_gray(args.runtime_dir / f"candidate_conf_f{frame_idx:04d}.png")
        grad_map = _load_gray(args.runtime_dir / f"candidate_disp_gradient_f{frame_idx:04d}.png")
        cost_map = _load_gray(args.runtime_dir / f"candidate_cost_f{frame_idx:04d}.png")
        if promoted_pack is not None:
            canonical_disp_q8, promoted_mask, _ = promoted_pack
            points_xyz = _reproject_points_from_disp(
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
            points_xyz = _reproject_points_from_disp(
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
            frame_residuals.append(np.zeros((0,), dtype=np.float32))
            continue

        conf_vals = conf_map[:, :][points_xyz.shape[0] * 0 : points_xyz.shape[0]] if conf_map is not None else None
        grad_vals = grad_map[:, :][points_xyz.shape[0] * 0 : points_xyz.shape[0]] if grad_map is not None else None

        # fallback: if maps exist use nearest-neighbor sampling via pixel rounding
        if conf_map is not None:
            # project back to pixel grid approximation
            conf_vals = np.ones((len(points_xyz),), dtype=np.float32) * np.mean(conf_map) / 255.0
        else:
            conf_vals = np.ones((len(points_xyz),), dtype=np.float32)
        if grad_map is not None:
            grad_vals = np.ones((len(points_xyz),), dtype=np.float32) * (0.5 + 0.5 * np.mean(grad_map) / 255.0)
        else:
            grad_vals = np.ones((len(points_xyz),), dtype=np.float32)

        weights = np.clip(conf_vals * grad_vals, 0.0, 1.0).astype(np.float32)
        residuals = cost_map.flatten().astype(np.float32)[: len(points_xyz)] if cost_map is not None else np.zeros(
            (len(points_xyz),), dtype=np.float32
        )

        frame_points.append(points_xyz)
        frame_weights.append(weights)
        frame_residuals.append(residuals)

    if not frame_points or sum(len(p) for p in frame_points) == 0:
        raise SystemExit("no promoted depth points available for surfel projection")

    params = SurfelGuardParams(
        alpha=float(args.alpha),
        alpha_h=float(args.alpha_h),
        beta=float(args.beta),
        h_max=float(args.h_max),
        tau_p=float(args.tau_p),
        tau_a=float(args.tau_a),
        h_a=float(args.h_a),
        epsilon_rho=float(args.epsilon_rho),
        sigma_rho=float(args.sigma_rho),
        gamma_neighbor=float(args.gamma_neighbor),
        cell_size=float(args.cell_size),
        pos_eps=float(args.pos_eps),
        normal_eps=float(args.normal_eps),
    )
    surfels = accumulate_candidate_surfels(frame_points, frame_weights, frame_residuals, params)
    states = guard_surfels(surfels, params)

    asc_mask = np.array([s == SURFEL_ASCENDED for s in states], dtype=np.uint8)
    plat_mask = np.array([s == SURFEL_PLATEAU for s in states], dtype=np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "surfels_state.npz",
        pos=np.stack([s["pos"] for s in surfels]) if surfels else np.zeros((0, 3), dtype=np.float32),
        normal=np.stack([s["normal"] for s in surfels]) if surfels else np.zeros((0, 3), dtype=np.float32),
        weight=np.array([s["weight"] for s in surfels], dtype=np.float32),
        hits=np.array([s["hits"] for s in surfels], dtype=np.float32),
        residual=np.array([s["residual"] for s in surfels], dtype=np.float32),
        states=np.array(states, dtype=np.uint8),
    )

    summary = {
        "runtime_dir": str(args.runtime_dir),
        "calibration_id": artifact.calibration_id,
        "frames": len(frame_indices),
        "counts": {
            "grounded": int(np.count_nonzero(np.array(states) == SURFEL_GROUNDED)),
            "plateau": int(np.count_nonzero(plat_mask)),
            "ascended": int(np.count_nonzero(asc_mask)),
        },
    }
    (args.output_dir / "surfels_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_ply and surfels:
        write_points_ply_ascii(args.output_dir / "surfels_all.ply", np.stack([s["pos"] for s in surfels]))
        colors = []
        pts = []
        for s, st in zip(surfels, states):
            pts.append(s["pos"])
            if st == SURFEL_ASCENDED:
                colors.append([60, 220, 80])
            elif st == SURFEL_PLATEAU:
                colors.append([255, 180, 60])
            else:
                colors.append([80, 80, 255])
        write_colored_ply_ascii(args.output_dir / "surfels_overlay.ply", np.stack(pts), np.array(colors, dtype=np.uint8))

    print(f"[surfel] wrote state: {args.output_dir / 'surfels_state.npz'}")
    print(f"[surfel] wrote summary: {args.output_dir / 'surfels_summary.json'}")
    print(
        f"[surfel] counts: grounded={summary['counts']['grounded']} plateau={summary['counts']['plateau']} ascended={summary['counts']['ascended']}"
    )


if __name__ == "__main__":
    main()
