#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

SURFEL_GROUNDED = 0
SURFEL_PLATEAU = 1
SURFEL_ASCENDED = 2


def _project_points(points: np.ndarray, yaw: float, pitch: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    yaw_rot = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    pitch_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    rotated = points @ yaw_rot.T @ pitch_rot.T
    depth = rotated[:, 2].copy()
    xy = rotated[:, :2] * scale
    return xy, depth


def _draw_panel(
    canvas: np.ndarray,
    points: np.ndarray,
    states: np.ndarray,
    yaw: float,
    pitch: float,
    *,
    title: str,
    width: int,
    height: int,
    show_all: bool,
) -> None:
    canvas[:] = (10, 10, 14)
    if points.size == 0:
        cv2.putText(canvas, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(canvas, "no points", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        return

    mask = np.ones(len(points), dtype=bool) if show_all else (states == SURFEL_ASCENDED)
    pts = points[mask]
    sts = states[mask]
    if pts.size == 0:
        cv2.putText(canvas, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(canvas, "no visible points", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        return

    centered = pts - pts.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    radius = max(radius, 1e-4)
    scale = 0.42 * min(width, height) / radius
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

    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    ascended = int(np.count_nonzero(states == SURFEL_ASCENDED))
    plateau = int(np.count_nonzero(states == SURFEL_PLATEAU))
    grounded = int(np.count_nonzero(states == SURFEL_GROUNDED))
    subtitle = f"asc {ascended}  plat {plateau}  ground {grounded}"
    cv2.putText(canvas, subtitle, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfel-state", type=Path, required=True)
    ap.add_argument("--output-webm", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=810)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--num-frames", type=int, default=180)
    ap.add_argument("--pitch-deg", type=float, default=18.0)
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--title", type=str, default="Animalexic Surfel Cloud")
    args = ap.parse_args()

    blob = np.load(args.surfel_state)
    points = blob["pos"].astype(np.float32)
    states = blob["states"].astype(np.uint8)

    frames_dir = args.frames_dir or (args.output_webm.parent / f"{args.output_webm.stem}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.output_webm.parent.mkdir(parents=True, exist_ok=True)

    panel_w = args.width // 2
    panel_h = args.height
    pitch = math.radians(args.pitch_deg)

    left = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    right = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    for i in range(max(1, args.num_frames)):
        t = i / max(1, args.num_frames)
        yaw = (2.0 * math.pi * args.turns * t) - (0.5 * math.pi)
        _draw_panel(
            left,
            points,
            states,
            yaw,
            pitch,
            title=f"{args.title} | all states",
            width=panel_w,
            height=panel_h,
            show_all=True,
        )
        _draw_panel(
            right,
            points,
            states,
            yaw,
            pitch,
            title=f"{args.title} | ascended only",
            width=panel_w,
            height=panel_h,
            show_all=False,
        )
        frame = np.concatenate([left, right], axis=1)
        cv2.putText(
            frame,
            f"frame {i + 1}/{args.num_frames}",
            (args.width - 220, args.height - 24),
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
    print(f"[render-surfel-webm] wrote frames: {frames_dir}")
    print(f"[render-surfel-webm] wrote webm: {args.output_webm}")


if __name__ == "__main__":
    main()
