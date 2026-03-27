#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from fixed_rig_runtime import load_calibration_artifact

SURFEL_GROUNDED = 0
SURFEL_PLATEAU = 1
SURFEL_ASCENDED = 2


def _project_points(points: np.ndarray, yaw: float, pitch: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    yaw_rot = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    pitch_rot = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rotated = points @ yaw_rot.T @ pitch_rot.T
    return rotated[:, :2] * scale, rotated[:, 2].copy()


def _draw_panel(
    canvas: np.ndarray,
    points: np.ndarray,
    states: np.ndarray,
    *,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    scale: float,
    center: np.ndarray,
    title: str,
    show_all: bool,
) -> None:
    canvas[:] = (10, 10, 14)
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    if points.size == 0:
        cv2.putText(canvas, "no points", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        return

    mask = np.ones(len(points), dtype=bool) if show_all else (states == SURFEL_ASCENDED)
    pts = points[mask]
    sts = states[mask]
    if pts.size == 0:
        cv2.putText(canvas, "no visible points", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        return

    centered = pts - center[None, :]
    xy, depth = _project_points(centered, yaw, pitch, scale)
    order = np.argsort(depth)
    xy = xy[order]
    depth = depth[order]
    sts = sts[order]

    screen = np.empty_like(xy)
    screen[:, 0] = xy[:, 0] + (width * 0.5)
    screen[:, 1] = (-xy[:, 1]) + (height * 0.56)

    dmin = float(depth.min())
    dmax = float(depth.max())
    drange = max(dmax - dmin, 1e-5)
    palette = {
        SURFEL_GROUNDED: np.array([120, 90, 60], dtype=np.float32),
        SURFEL_PLATEAU: np.array([80, 180, 255], dtype=np.float32),
        SURFEL_ASCENDED: np.array([70, 235, 120], dtype=np.float32),
    }

    for (x, y), z, state in zip(screen, depth, sts):
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= width or yi >= height:
            continue
        depth_norm = (z - dmin) / drange
        color = palette.get(int(state), palette[SURFEL_GROUNDED]).copy()
        color = np.clip(color * (0.55 + 0.60 * depth_norm), 0, 255).astype(np.uint8)
        radius_px = 1 if int(state) != SURFEL_ASCENDED else 2
        cv2.circle(canvas, (xi, yi), radius_px, tuple(int(c) for c in color), thickness=-1, lineType=cv2.LINE_AA)

    subtitle = (
        f"asc {int(np.count_nonzero(states == SURFEL_ASCENDED))}  "
        f"plat {int(np.count_nonzero(states == SURFEL_PLATEAU))}  "
        f"ground {int(np.count_nonzero(states == SURFEL_GROUNDED))}"
    )
    cv2.putText(canvas, subtitle, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)


def _snapshot_files(snapshots_dir: Path) -> list[Path]:
    return sorted(snapshots_dir.glob("surfels_state_f*.npz"))


def _camera_from_q(q: np.ndarray) -> tuple[float, float, float]:
    fx = float(q[2, 3])
    cx = float(-q[0, 3])
    cy = float(-q[1, 3])
    return fx, cx, cy


def _project_to_image(points: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    fx, cx, cy = _camera_from_q(q.astype(np.float32))
    z = points[:, 2].astype(np.float32)
    good = np.isfinite(points).all(axis=1) & (z > 1e-6)
    if not np.any(good):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    pts = points[good]
    z = pts[:, 2]
    uv = np.empty((len(pts), 2), dtype=np.float32)
    uv[:, 0] = (pts[:, 0] * fx / z) + cx
    uv[:, 1] = (pts[:, 1] * fx / z) + cy
    return uv, z


def _draw_overlay_panel(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    states: np.ndarray,
    *,
    q_matrix: np.ndarray,
    title: str,
    show_all: bool,
    crop_to_ascended: bool,
    crop_margin: int,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    overlay = canvas.copy()
    h, w = canvas.shape[:2]
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    if points.size == 0:
        return canvas
    mask = np.ones(len(points), dtype=bool) if show_all else (states == SURFEL_ASCENDED)
    pts = points[mask]
    sts = states[mask]
    if pts.size == 0:
        return canvas
    uv, depth = _project_to_image(pts, q_matrix)
    if uv.size == 0:
        return canvas
    crop = None
    if crop_to_ascended:
        asc_pts = points[states == SURFEL_ASCENDED]
        asc_uv, _ = _project_to_image(asc_pts, q_matrix)
        if asc_uv.size:
            xs = asc_uv[:, 0]
            ys = asc_uv[:, 1]
            x0 = max(0, int(np.floor(np.min(xs))) - int(crop_margin))
            y0 = max(0, int(np.floor(np.min(ys))) - int(crop_margin))
            x1 = min(w, int(np.ceil(np.max(xs))) + int(crop_margin))
            y1 = min(h, int(np.ceil(np.max(ys))) + int(crop_margin))
            if x1 > x0 and y1 > y0:
                crop = (x0, y0, x1, y1)
    order = np.argsort(depth)[::-1]
    uv = uv[order]
    depth = depth[order]
    sts = sts[order]
    dmin = float(depth.min())
    dmax = float(depth.max())
    drange = max(dmax - dmin, 1e-5)
    palette = {
        SURFEL_GROUNDED: np.array([120, 90, 60], dtype=np.float32),
        SURFEL_PLATEAU: np.array([80, 180, 255], dtype=np.float32),
        SURFEL_ASCENDED: np.array([70, 235, 120], dtype=np.float32),
    }
    for (x, y), z, state in zip(uv, depth, sts):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        depth_norm = (z - dmin) / drange
        color = palette.get(int(state), palette[SURFEL_GROUNDED]).copy()
        color = np.clip(color * (0.60 + 0.50 * (1.0 - depth_norm)), 0, 255).astype(np.uint8)
        radius_px = 1 if int(state) == SURFEL_GROUNDED else (2 if int(state) == SURFEL_PLATEAU else 3)
        cv2.circle(overlay, (xi, yi), radius_px, tuple(int(c) for c in color), thickness=-1, lineType=cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0.0)
    subtitle = (
        f"asc {int(np.count_nonzero(states == SURFEL_ASCENDED))}  "
        f"plat {int(np.count_nonzero(states == SURFEL_PLATEAU))}  "
        f"ground {int(np.count_nonzero(states == SURFEL_GROUNDED))}"
    )
    cv2.putText(canvas, subtitle, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if crop is not None:
        x0, y0, x1, y1 = crop
        cropped = canvas[y0:y1, x0:x1]
        if cropped.size:
            canvas = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", type=Path, required=True)
    ap.add_argument("--output-webm", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=810)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--pitch-deg", type=float, default=18.0)
    ap.add_argument("--yaw-deg", type=float, default=-90.0)
    ap.add_argument("--title", type=str, default="Animalexic Surfel Replay")
    ap.add_argument("--runtime-dir", type=Path, help="Runtime output dir containing left_input_fNNNN.png for image-space overlay replay")
    ap.add_argument("--calibration", type=Path, help="Calibration artifact used to project surfels back into image space")
    ap.add_argument("--crop-to-ascended", action="store_true", help="Crop the overlay view to the projected ascended support")
    ap.add_argument("--crop-margin", type=int, default=24)
    args = ap.parse_args()

    files = _snapshot_files(args.snapshots_dir)
    if not files:
        raise SystemExit(f"no snapshot files found in {args.snapshots_dir}")

    overlay_mode = args.runtime_dir is not None and args.calibration is not None
    artifact = load_calibration_artifact(args.calibration) if overlay_mode else None

    points_all = []
    for path in files:
        blob = np.load(path)
        pts = blob["pos"].astype(np.float32)
        if pts.size:
            points_all.append(pts)
    all_points = np.concatenate(points_all, axis=0) if points_all else np.zeros((0, 3), dtype=np.float32)
    if all_points.size == 0:
        raise SystemExit("snapshot files contain no points")

    center = all_points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_points - center[None, :], axis=1)))
    radius = max(radius, 1e-4)
    panel_w = args.width // 2
    panel_h = args.height
    scale = 0.42 * min(panel_w, panel_h) / radius
    yaw = math.radians(args.yaw_deg)
    pitch = math.radians(args.pitch_deg)

    frames_dir = args.frames_dir or (args.output_webm.parent / f"{args.output_webm.stem}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.output_webm.parent.mkdir(parents=True, exist_ok=True)

    left = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    right = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    for i, path in enumerate(files):
        blob = np.load(path)
        points = blob["pos"].astype(np.float32)
        states = blob["states"].astype(np.uint8)
        if overlay_mode:
            frame_idx = int(path.stem.split("_f", 1)[1])
            left_frame = cv2.imread(str(args.runtime_dir / f"left_input_f{frame_idx:04d}.png"), cv2.IMREAD_COLOR)
            if left_frame is None:
                left_frame = np.zeros((artifact.image_height, artifact.image_width, 3), dtype=np.uint8)
            left = _draw_overlay_panel(
                left_frame,
                points,
                states,
                q_matrix=artifact.q_matrix,
                title=f"{args.title} | all states",
                show_all=True,
                crop_to_ascended=bool(args.crop_to_ascended),
                crop_margin=int(args.crop_margin),
            )
            right = _draw_overlay_panel(
                left_frame,
                points,
                states,
                q_matrix=artifact.q_matrix,
                title=f"{args.title} | ascended only",
                show_all=False,
                crop_to_ascended=bool(args.crop_to_ascended),
                crop_margin=int(args.crop_margin),
            )
            if left.shape[1] != panel_w or left.shape[0] != panel_h:
                left = cv2.resize(left, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
                right = cv2.resize(right, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        else:
            _draw_panel(
                left,
                points,
                states,
                width=panel_w,
                height=panel_h,
                yaw=yaw,
                pitch=pitch,
                scale=scale,
                center=center,
                title=f"{args.title} | all states",
                show_all=True,
            )
            _draw_panel(
                right,
                points,
                states,
                width=panel_w,
                height=panel_h,
                yaw=yaw,
                pitch=pitch,
                scale=scale,
                center=center,
                title=f"{args.title} | ascended only",
                show_all=False,
            )
        frame = np.concatenate([left, right], axis=1)
        cv2.putText(
            frame,
            f"snapshot {i + 1}/{len(files)}",
            (args.width - 250, args.height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), frame)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "0",
        "-crf",
        "28",
        str(args.output_webm),
    ]
    subprocess.run(cmd, check=True)
    print(f"[render-surfel-replay] wrote frames: {frames_dir}")
    print(f"[render-surfel-replay] wrote webm: {args.output_webm}")


if __name__ == "__main__":
    main()
