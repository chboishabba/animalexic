#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from fixed_rig_runtime import load_calibration_artifact
from surfel_guard import (
    SURFEL_ASCENDED,
    SURFEL_GROUNDED,
    SURFEL_PLATEAU,
    SurfelGuardParams,
    accumulate_frame_into_surfels,
    guard_surfels,
    init_surfel_store,
    save_surfel_state,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(promoted_mask)
    if ys.size == 0:
        return (np.zeros((0, 3), dtype=np.float32), ys, xs)
    ys = ys[:: max(1, stride)]
    xs = xs[:: max(1, stride)]

    disp = disp_map[ys, xs].astype(np.float32)
    valid_disp = disp >= float(min_disp)
    if not np.any(valid_disp):
        return (np.zeros((0, 3), dtype=np.float32), ys, xs)

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
        return (np.zeros((0, 3), dtype=np.float32), ys, xs)

    xyz = (homog[:3, good] / w[good]).T.astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return (np.zeros((0, 3), dtype=np.float32), ys, xs)

    if np.mean(xyz[:, 2]) < 0.0:
        xyz[:, 2] *= -1.0
    depth_ok = xyz[:, 2] > 0.0
    if max_depth > 0:
        depth_ok &= xyz[:, 2] <= float(max_depth)
    xyz = xyz[depth_ok]
    # keep pixel coordinates aligned with filtered points
    ys = ys[depth_ok]
    xs = xs[depth_ok]
    return (xyz.astype(np.float32), ys, xs)


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
    ap.add_argument("--tau-p", type=float, default=0.5)
    ap.add_argument("--tau-a", type=float, default=1.0)
    ap.add_argument("--h-a", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--alpha-h", dest="alpha_h", type=float, default=0.90)
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--h-max", dest="h_max", type=float, default=8.0)
    ap.add_argument("--epsilon-rho", type=float, default=64.0)
    ap.add_argument("--sigma-rho", type=float, default=24.0)
    ap.add_argument("--gamma-neighbor", dest="gamma_neighbor", type=float, default=0.20)
    ap.add_argument("--spread-sigma", type=float, default=0.10)
    ap.add_argument("--spread-max", type=float, default=0.08)
    ap.add_argument("--drift-sigma", type=float, default=0.05)
    ap.add_argument("--drift-max", type=float, default=0.05)
    ap.add_argument("--save-snapshots", action="store_true")
    ap.add_argument("--snapshots-dir", type=Path)
    ap.add_argument("--save-ply", action="store_true")
    ap.add_argument("--early-stop-window", type=int, default=0, help="Stop after this many low-gain frames; 0 disables")
    ap.add_argument("--min-frames-before-stop", type=int, default=12, help="Do not apply early-stop logic before this many frames")
    ap.add_argument("--min-ascended-before-stop", type=int, default=20, help="Do not apply early-stop logic until this many ascended surfels exist")
    ap.add_argument("--min-new-ascended", type=int, default=1, help="Minimum new ascended surfels expected in the stop window")
    ap.add_argument("--min-residual-improvement", type=float, default=0.0005, help="Minimum residual improvement expected in the stop window")
    ap.add_argument("--max-scene-change", type=float, default=0.85, help="Stop if average scene-change score exceeds this in the stop window")
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
            points_xyz, ys_used, xs_used = _reproject_points_from_disp(
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
            points_xyz, ys_used, xs_used = _reproject_points_from_disp(
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

        # sample evidence maps at the same pixels used for reprojection
        if conf_map is not None and ys_used.size:
            conf_vals = conf_map[ys_used, xs_used].astype(np.float32) / 255.0
        else:
            conf_vals = np.ones((len(points_xyz),), dtype=np.float32)

        if grad_map is not None and ys_used.size:
            grad_vals = 0.5 + 0.5 * (grad_map[ys_used, xs_used].astype(np.float32) / 255.0)
        else:
            grad_vals = np.ones((len(points_xyz),), dtype=np.float32)

        weights = np.clip(conf_vals * grad_vals, 0.0, 1.0).astype(np.float32)

        if cost_map is not None and ys_used.size:
            residuals = cost_map[ys_used, xs_used].astype(np.float32)
        else:
            residuals = np.zeros((len(points_xyz),), dtype=np.float32)

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
        spread_sigma=float(args.spread_sigma),
        spread_max=float(args.spread_max),
        drift_sigma=float(args.drift_sigma),
        drift_max=float(args.drift_max),
    )
    snapshots_dir = args.snapshots_dir or (args.output_dir / "surfels_snapshots")
    if args.save_snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    surfels, grid = init_surfel_store()
    frame_stats: list[dict[str, float | int | bool]] = []
    prev_ascended = 0
    prev_residual_margin: float | None = None
    stopped_early = False
    stop_reason = ""
    processed_frame_indices: list[int] = []

    for frame_idx, points_xyz, weights, residuals in zip(frame_indices, frame_points, frame_weights, frame_residuals):
        acc_stats = accumulate_frame_into_surfels(surfels, grid, frame_idx, points_xyz, weights, residuals, params)
        states = guard_surfels(surfels, params)
        state_arr = np.array(states, dtype=np.uint8)
        asc_mask = state_arr == SURFEL_ASCENDED
        plat_mask = state_arr == SURFEL_PLATEAU
        ascended_count = int(np.count_nonzero(asc_mask))
        plateau_count = int(np.count_nonzero(plat_mask))
        grounded_count = int(np.count_nonzero(state_arr == SURFEL_GROUNDED))
        new_ascended = ascended_count - prev_ascended
        ascended_residuals = [float(surfels[idx]["residual_ema"]) for idx, st in enumerate(states) if st == SURFEL_ASCENDED]
        nonpromoted_multiframe_residuals = [
            float(surfels[idx]["residual_ema"])
            for idx, st in enumerate(states)
            if st != SURFEL_ASCENDED and int(surfels[idx].get("frame_count", 1)) >= 2
        ]
        ascended_mean_residual = float(mean(ascended_residuals)) if ascended_residuals else None
        nonpromoted_multiframe_mean_residual = (
            float(mean(nonpromoted_multiframe_residuals)) if nonpromoted_multiframe_residuals else None
        )
        residual_margin = None
        if ascended_mean_residual is not None and nonpromoted_multiframe_mean_residual is not None:
            residual_margin = nonpromoted_multiframe_mean_residual - ascended_mean_residual
        residual_margin_improvement = 0.0
        if prev_residual_margin is not None and residual_margin is not None:
            residual_margin_improvement = residual_margin - prev_residual_margin
        scene_change_score = 1.0 - float(acc_stats["merge_accept_rate"])
        if float(acc_stats["mean_merge_dist"]) > 0.0 and float(params.pos_eps) > 1e-6:
            scene_change_score = min(
                1.0,
                max(
                    scene_change_score,
                    float(acc_stats["mean_merge_dist"]) / float(params.pos_eps),
                ),
            )

        frame_row = {
            "frame_idx": int(frame_idx),
            "points": int(acc_stats["points"]),
            "new_surfels": int(acc_stats["new_surfels"]),
            "merged_surfels": int(acc_stats["merged_surfels"]),
            "merge_accept_rate": float(acc_stats["merge_accept_rate"]),
            "mean_merge_dist": float(acc_stats["mean_merge_dist"]),
            "mean_input_residual": float(acc_stats["mean_input_residual"]),
            "grounded": grounded_count,
            "plateau": plateau_count,
            "ascended": ascended_count,
            "new_ascended": int(new_ascended),
            "ascended_mean_residual": ascended_mean_residual,
            "nonpromoted_multiframe_mean_residual": nonpromoted_multiframe_mean_residual,
            "residual_margin": residual_margin,
            "residual_margin_improvement": float(residual_margin_improvement),
            "scene_change_score": float(scene_change_score),
            "stop_candidate": False,
        }
        frame_stats.append(frame_row)
        processed_frame_indices.append(int(frame_idx))

        if args.save_snapshots:
            save_surfel_state(snapshots_dir / f"surfels_state_f{frame_idx:04d}.npz", surfels, states)

        if residual_margin is not None:
            prev_residual_margin = residual_margin
        prev_ascended = ascended_count

        if (
            args.early_stop_window > 0
            and len(frame_stats) >= max(int(args.early_stop_window), int(args.min_frames_before_stop))
            and ascended_count >= int(args.min_ascended_before_stop)
        ):
            window = frame_stats[-int(args.early_stop_window):]
            avg_new_asc = float(mean(float(r["new_ascended"]) for r in window))
            avg_improvement = float(mean(float(r["residual_margin_improvement"]) for r in window))
            avg_scene_change = float(mean(float(r["scene_change_score"]) for r in window))
            low_gain = avg_new_asc < float(args.min_new_ascended) and avg_improvement < float(args.min_residual_improvement)
            scene_break = avg_scene_change > float(args.max_scene_change)
            if low_gain:
                frame_row["stop_candidate"] = True
                stopped_early = True
                stop_reason = (
                    f"low_gain(avg_new_asc={avg_new_asc:.3f}, avg_improvement={avg_improvement:.5f}, "
                    f"avg_scene_change={avg_scene_change:.3f}, scene_break={scene_break})"
                )
                break

    states = guard_surfels(surfels, params)

    asc_mask = np.array([s == SURFEL_ASCENDED for s in states], dtype=np.uint8)
    plat_mask = np.array([s == SURFEL_PLATEAU for s in states], dtype=np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_surfel_state(args.output_dir / "surfels_state.npz", surfels, states)

    summary = {
        "runtime_dir": str(args.runtime_dir),
        "calibration_id": artifact.calibration_id,
        "frames": len(processed_frame_indices),
        "frame_indices": processed_frame_indices,
        "early_stop": {
            "enabled": bool(args.early_stop_window > 0),
            "stopped": stopped_early,
            "reason": stop_reason,
            "window": int(args.early_stop_window),
            "min_frames_before_stop": int(args.min_frames_before_stop),
            "min_ascended_before_stop": int(args.min_ascended_before_stop),
            "min_new_ascended": int(args.min_new_ascended),
            "min_residual_improvement": float(args.min_residual_improvement),
            "max_scene_change": float(args.max_scene_change),
        },
        "counts": {
            "grounded": int(np.count_nonzero(np.array(states) == SURFEL_GROUNDED)),
            "plateau": int(np.count_nonzero(plat_mask)),
            "ascended": int(np.count_nonzero(asc_mask)),
        },
    }
    (args.output_dir / "surfels_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "surfels_frame_stats.json").write_text(json.dumps(frame_stats, indent=2), encoding="utf-8")
    if args.save_snapshots:
        summary["snapshots_dir"] = str(snapshots_dir)
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
    if frame_stats:
        last = frame_stats[-1]
        print(
            "[surfel] last-frame stats: "
            f"frame={last['frame_idx']} new_ascended={last['new_ascended']} "
            f"merge_accept={last['merge_accept_rate']:.3f} scene_change={last['scene_change_score']:.3f}"
        )
    if stopped_early:
        print(f"[surfel] early stop: {stop_reason}")
    if args.save_snapshots:
        print(f"[surfel] wrote snapshots: {snapshots_dir}")


if __name__ == "__main__":
    main()
