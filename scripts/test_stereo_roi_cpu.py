#!/usr/bin/env python3
"""
CPU harness to validate tie-breaks and severity flags for the stereo ROI spec.
This does NOT run Vulkan; it mirrors shaders/stereo_roi.comp logic for a tiny synthetic pair.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class StereoROIParams:
    d_min: int
    d_max: int
    min_texture: int
    max_cost: int
    min_conf_gap: int
    left_border_guard: int
    right_border_guard: int
    severity_low_texture: int
    severity_out_of_range: int
    severity_ambiguous: int


def census5x5(img: np.ndarray, x: int, y: int) -> int:
    center = img[y, x]
    bits = 0
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            if img[y + dy, x + dx] < center:
                bits |= 1 << bit
            bit += 1
    return bits


def run_stereo_roi(left: np.ndarray, right: np.ndarray, params: StereoROIParams, roi: Tuple[int, int, int, int]):
    x0, y0, w, h = roi
    disp = np.zeros_like(left, dtype=np.uint16)
    cost_min = np.zeros_like(left, dtype=np.uint16)
    conf_gap = np.zeros_like(left, dtype=np.uint16)
    valid = np.zeros_like(left, dtype=np.uint8)
    severity = np.zeros_like(left, dtype=np.uint8)

    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            if x < params.left_border_guard or x >= left.shape[1] - params.right_border_guard:
                severity[y, x] = params.severity_out_of_range
                continue
            cL = census5x5(left, x, y)
            patch = left[y - 2 : y + 3, x - 2 : x + 3]
            texture = patch.max() - patch.min()
            if texture < params.min_texture:
                severity[y, x] = params.severity_low_texture
                continue
            best = 1 << 31
            second = 1 << 31
            best_d = 0
            for d in range(params.d_min, params.d_max + 1):
                xr = x - d
                if xr < params.left_border_guard:
                    continue
                cR = census5x5(right, xr, y)
                c = bin(cL ^ cR).count("1")
                if c < best:
                    second = best
                    best = c
                    best_d = d
                elif c < second:
                    second = c
            if best == (1 << 31):
                severity[y, x] = params.severity_out_of_range
                continue
            cg = 0 if second == (1 << 31) else (second - best)
            v = 1
            sev = 0
            if best > params.max_cost:
                v = 0
                sev = max(sev, params.severity_out_of_range)
            if cg < params.min_conf_gap:
                v = 0
                sev = max(sev, params.severity_ambiguous)
            disp[y, x] = best_d << 8
            cost_min[y, x] = best
            conf_gap[y, x] = cg
            valid[y, x] = v
            severity[y, x] = sev
    return disp, cost_min, conf_gap, valid, severity


def main():
    # Synthetic pair: left has a bright square shifted by disparity=3
    w, h = 32, 16
    left = np.zeros((h, w), dtype=np.uint8)
    right = np.zeros_like(left)
    left[6:10, 10:14] = 200
    right[6:10, 7:11] = 200  # shift right image left by 3 => disparity 3

    params = StereoROIParams(
        d_min=0,
        d_max=8,
        min_texture=10,
        max_cost=24,
        min_conf_gap=2,
        left_border_guard=4,
        right_border_guard=4,
        severity_low_texture=2,
        severity_out_of_range=4,
        severity_ambiguous=3,
    )
    roi = (4, 4, 20, 8)
    disp, cost_min, conf_gap, valid, severity = run_stereo_roi(left, right, params, roi)

    # Check tie-break and validity
    assert disp[7, 12] == (3 << 8), f"expected disparity 3 at bright block, got {disp[7,12]}"
    assert valid[7, 12] == 1, "bright block should be valid"
    assert severity[7, 12] == 0, "bright block severity should be 0"

    # Pixels outside ROI remain untouched (zero severity)
    assert severity[0, 0] == 0

    print("stereo_roi CPU harness passed basic checks.")


if __name__ == "__main__":
    main()
