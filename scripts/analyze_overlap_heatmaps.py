#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2


def load_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    r = arr[..., 0] > 127
    g = arr[..., 1] > 127
    b = arr[..., 2] > 127
    tp = g & ~r & ~b
    fp = r & ~g & ~b
    fn = b & ~r & ~g
    return np.stack([tp, fp, fn], axis=0)


def summarize(kind: str, files: list[Path], tile: int) -> dict:
    if not files:
        return {"kind": kind, "frames": 0}
    stack = np.array([load_mask(p) for p in files], dtype=np.uint8)
    # stack: F x 3 x H x W
    mean_maps = stack.mean(axis=0)
    h, w = mean_maps.shape[1:]
    tile_h = (h + tile - 1) // tile
    tile_w = (w + tile - 1) // tile
    hotspots = []
    for cls_idx, cls_name in [(1, 'fp'), (2, 'fn'), (0, 'tp')]:
        m = mean_maps[cls_idx]
        for ty in range(tile_h):
            y0 = ty * tile
            y1 = min(h, y0 + tile)
            for tx in range(tile_w):
                x0 = tx * tile
                x1 = min(w, x0 + tile)
                score = float(m[y0:y1, x0:x1].mean())
                hotspots.append({
                    'class': cls_name,
                    'tile_x': tx,
                    'tile_y': ty,
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1,
                    'score': score,
                })
    hotspots.sort(key=lambda x: x['score'], reverse=True)
    fp_mass = float(mean_maps[1].sum())
    fn_mass = float(mean_maps[2].sum())
    tp_mass = float(mean_maps[0].sum())
    return {
        'kind': kind,
        'frames': len(files),
        'shape': [int(h), int(w)],
        'tile': tile,
        'mean_tp_mass': tp_mass,
        'mean_fp_mass': fp_mass,
        'mean_fn_mass': fn_mass,
        'fp_to_tp': float(fp_mass / max(tp_mass, 1e-9)),
        'fn_to_tp': float(fn_mass / max(tp_mass, 1e-9)),
        'top_hotspots': hotspots[:15],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare-dir', type=Path, required=True)
    ap.add_argument('--tile', type=int, default=32)
    ap.add_argument('--output-json', type=Path)
    args = ap.parse_args()

    out = {}
    for kind in ['candidate', 'promoted']:
        files = sorted(args.compare_dir.glob(f'{kind}_overlap_f*.png'))
        out[kind] = summarize(kind, files, args.tile)
    text = json.dumps(out, indent=2)
    if args.output_json:
        args.output_json.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
