#!/usr/bin/env python3
"""
OpenCV SBS oracle for stereo sanity checks.

This is a reference baseline, not promoted canonical state:
- splits SBS into left/right
- optionally bootstraps self-calibration from the first pair
- rectifies with provided or bootstrapped geometry
- computes StereoSGBM disparity
- optionally reprojects to 3D and writes a downsampled PLY preview
- writes per-frame candidate artifacts and a JSONL receipt log
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fixed_rig_runtime import depth_from_disparity, load_calibration_artifact, rectify_pair
from run_stereo_dispatch import FrameStreamer, _seed_and_self_calibrate, _wait_for_frame, save_png


def _split_sbs(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def _suggest_calibration_path(path: Path) -> Path | None:
    if path.exists():
        return path
    candidates = sorted(Path("outputs").glob("**/selfcal_bootstrap.npz"))
    if candidates:
        return candidates[0]
    return None


def _make_stereo(num_disparities: int, block_size: int, min_disparity: int, uniqueness_ratio: int):
    import cv2

    P1 = 8 * block_size * block_size
    P2 = 32 * block_size * block_size
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )


def _write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if len(colors) != len(points):
            raise ValueError("points/colors length mismatch")
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _save_color_map(path: Path, disp: np.ndarray, valid: np.ndarray, num_disparities: int):
    import cv2

    disp_vis = np.clip(disp, 0.0, float(num_disparities))
    disp_vis = (255.0 * disp_vis / max(1.0, float(num_disparities))).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    disp_vis[valid == 0] = 0
    cv2.imwrite(str(path), disp_vis)


def _reproject_and_write(
    *,
    outdir: Path,
    frame_index: int,
    disp: np.ndarray,
    valid: np.ndarray,
    q_matrix: np.ndarray,
    left_rect: np.ndarray,
    ply_stride: int,
):
    import cv2

    points_3d = cv2.reprojectImageTo3D(disp, q_matrix)
    finite = np.isfinite(points_3d).all(axis=2)
    mask = (valid != 0) & finite
    if not np.any(mask):
        return 0

    ys, xs = np.where(mask)
    ys = ys[::ply_stride]
    xs = xs[::ply_stride]
    points = points_3d[ys, xs]
    colors = left_rect[ys, xs] if left_rect.ndim == 3 else np.stack([left_rect[ys, xs]] * 3, axis=1)
    ply_path = outdir / f"cloud_f{frame_index:04d}.ply"
    _write_ply_ascii(ply_path, points, colors)
    return int(len(points))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--youtube", help="YouTube URL (SBS source)")
    src.add_argument("--file", help="Local video file path (SBS source)")
    ap.add_argument("--sbs", action="store_true", help="Input is side-by-side stereo and will be split into left/right")
    ap.add_argument("--output-dir", default="outputs/opencv_oracle", help="Where to write oracle artifacts")
    ap.add_argument("--width", type=int, default=640, help="Fallback scale width")
    ap.add_argument("--height", type=int, default=360, help="Fallback scale height")
    ap.add_argument("--auto-res", action="store_true", help="Probe the source and scale to its native decoded resolution")
    ap.add_argument("--every-n", type=int, default=2, help="Subsample frames")
    ap.add_argument("--start-seconds", type=float, default=0.0, help="Seek this many seconds into the source")
    ap.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for first frames")
    ap.add_argument("--min-frame-mean", type=float, default=5.0, help="Wait until frame mean exceeds this before locking input")
    ap.add_argument("--max-frames", type=int, default=8, help="How many frames to process")
    ap.add_argument("--calibration", type=str, help="Path to fixed-rig calibration artifact (.npz)")
    ap.add_argument("--selfcal-rig-id", type=str, default="opencv_oracle_selfcal", help="Rig id for generated self-calibration")
    ap.add_argument("--selfcal-max-features", type=int, default=6000, help="ORB feature budget for bootstrap self-calibration")
    ap.add_argument("--force-unrectified", action="store_true", help="Skip calibration bootstrap and run unrectified")
    ap.add_argument("--save-debug", action="store_true", help="Write rectified/disparity/valid debug images")
    ap.add_argument("--save-ply", action="store_true", help="Write a downsampled point cloud preview per frame")
    ap.add_argument("--ply-stride", type=int, default=8, help="Subsample stride for PLY export")
    ap.add_argument("--num-disparities", type=int, default=128, help="StereoSGBM disparity search range")
    ap.add_argument("--block-size", type=int, default=7, help="StereoSGBM block size")
    ap.add_argument("--min-disparity", type=int, default=0, help="StereoSGBM minimum disparity")
    ap.add_argument("--uniqueness-ratio", type=int, default=10, help="StereoSGBM uniqueness ratio")
    args = ap.parse_args()

    if not (args.youtube or args.file):
        ap.error("provide --youtube or --file")
    if not args.sbs:
        ap.error("this oracle currently expects SBS input; pass --sbs")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    receipt_path = outdir / "oracle_receipts.jsonl"
    selfcal_path = outdir / "selfcal_bootstrap.npz"

    calibration = None
    bootstrap_seed = None
    bootstrap_mode = "none"
    if args.calibration:
        calibration_path = Path(args.calibration)
        try:
            calibration = load_calibration_artifact(calibration_path)
        except Exception as exc:
            print(f"[oracle] failed to load calibration artifact {calibration_path}: {exc}")
            suggestion = _suggest_calibration_path(calibration_path)
            if suggestion is not None and suggestion != calibration_path:
                print(f"[oracle] try --calibration {suggestion}")
            raise SystemExit(1)
        bootstrap_mode = "provided"
        print(f"[oracle] loaded calibration artifact {calibration.calibration_id}")

    streamer = FrameStreamer(
        args.youtube or args.file,
        args.width,
        args.height,
        args.every_n,
        gray=True,
        youtube=bool(args.youtube),
        start_seconds=args.start_seconds,
        auto_res=args.auto_res,
        status_prefix="[oracle]",
    )

    if calibration is None and not args.force_unrectified:
        bootstrap_seed, bootstrap_result = _seed_and_self_calibrate(
            dual_file_mode=False,
            left_streamer=None,
            right_streamer=None,
            streamer=streamer,
            args=args,
            timeout=args.timeout,
            selfcal_output=selfcal_path,
            outdir=outdir,
            rig_id=args.selfcal_rig_id,
            max_features=args.selfcal_max_features,
            save_debug=args.save_debug,
            status_prefix="[oracle]",
        )
        if isinstance(bootstrap_result, Exception):
            print(f"[oracle] bootstrap self-calibration failed: {bootstrap_result}")
            print("[oracle] continuing unrectified")
            bootstrap_mode = "unrectified_fallback"
        else:
            calibration = bootstrap_result
            bootstrap_mode = "selfcal_bootstrap"
            print(f"[oracle] bootstrap calibration ready: {calibration.calibration_id}")
    elif calibration is None:
        print("[oracle] --force-unrectified set; skipping bootstrap calibration")

    stereo = _make_stereo(
        num_disparities=_round_up_to_multiple(args.num_disparities, 16),
        block_size=max(5, args.block_size | 1),
        min_disparity=args.min_disparity,
        uniqueness_ratio=args.uniqueness_ratio,
    )

    last_seen = None
    frame_counter = 0
    summary = {
        "bootstrap_mode": bootstrap_mode,
        "calibration_id": calibration.calibration_id if calibration else None,
        "output_dir": str(outdir),
        "frames": [],
    }
    try:
        with receipt_path.open("a", encoding="utf-8") as receipt_file:
            for frame_index in range(args.max_frames):
                if bootstrap_seed is not None:
                    source_frame, left, right = bootstrap_seed
                    bootstrap_seed = None
                else:
                    seed_frame = _wait_for_frame(streamer, args, last_seen, status_prefix="[oracle]")
                    if seed_frame is None:
                        print(f"[oracle] no more frames available after frame {frame_index}; stopping")
                        break
                    source_frame, left, right = seed_frame

                source_selected_index = None if source_frame is None else source_frame.get("selected_index")
                source_pts_time = None if source_frame is None else source_frame.get("pts_time")
                last_seen = (source_selected_index, round(float(source_pts_time), 6)) if source_pts_time is not None else (source_selected_index, None)
                if args.sbs:
                    left, right = _split_sbs(left)

                t_rect = time.perf_counter()
                if calibration is not None:
                    left_rect, right_rect = rectify_pair(left, right, calibration)
                else:
                    left_rect, right_rect = left, right
                rect_ms = (time.perf_counter() - t_rect) * 1000.0

                t_disp = time.perf_counter()
                disp_q8 = stereo.compute(left_rect, right_rect)
                disp = disp_q8.astype(np.float32) / 16.0
                valid = (disp > 0.0).astype(np.uint8)
                disp_ms = (time.perf_counter() - t_disp) * 1000.0

                valid_count = int(valid.sum())
                total = int(valid.size)
                valid_pct = (100.0 * valid_count / total) if total else 0.0
                if valid_count:
                    disp_valid = disp[valid != 0]
                    disp_min = float(disp_valid.min())
                    disp_max = float(disp_valid.max())
                    disp_mean = float(disp_valid.mean())
                else:
                    disp_min = disp_max = disp_mean = 0.0

                ply_points = 0
                if calibration is not None and args.save_ply:
                    ply_points = _reproject_and_write(
                        outdir=outdir,
                        frame_index=frame_counter,
                        disp=disp,
                        valid=valid,
                        q_matrix=calibration.q_matrix,
                        left_rect=left_rect,
                        ply_stride=max(1, args.ply_stride),
                    )

                if args.save_debug:
                    save_png(left_rect.astype(np.uint8), outdir / f"rect_left_f{frame_counter:04d}.png")
                    save_png(right_rect.astype(np.uint8), outdir / f"rect_right_f{frame_counter:04d}.png")
                    _save_color_map(outdir / f"disp_f{frame_counter:04d}.png", disp, valid, stereo.getNumDisparities())
                    save_png(valid * 255, outdir / f"valid_f{frame_counter:04d}.png")

                record = {
                    "frame_index": frame_counter,
                    "source_frame_index": source_selected_index,
                    "source_pts_time": source_pts_time,
                    "bootstrap_mode": bootstrap_mode,
                    "rectify_ms": rect_ms,
                    "disparity_ms": disp_ms,
                    "valid_pixels": valid_count,
                    "total_pixels": total,
                    "valid_pct": valid_pct,
                    "disp_min": disp_min,
                    "disp_max": disp_max,
                    "disp_mean": disp_mean,
                    "ply_points": ply_points,
                }
                receipt_file.write(json.dumps(record, sort_keys=True) + "\n")
                receipt_file.flush()
                summary["frames"].append(record)

                print(
                    "[oracle] frame"
                    f" {frame_counter}: valid={valid_count}/{total} ({valid_pct:.2f}%)"
                    f" disp_mean={disp_mean:.2f} rectify_ms={rect_ms:.1f} disparity_ms={disp_ms:.1f}"
                )
                frame_counter += 1
    except KeyboardInterrupt:
        print("[oracle] interrupted; shutting down")
    finally:
        streamer.close()

    (outdir / "oracle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[oracle] wrote summary: {outdir / 'oracle_summary.json'}")
    print(f"[oracle] wrote receipts: {receipt_path}")


if __name__ == "__main__":
    main()
