#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_mask(path: Path):
    import cv2
    import numpy as np

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def _load_gray(path: Path):
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _balanced_coords(oracle_valid, runtime_candidate, runtime_roi, max_samples, rng: random.Random):
    h, w = oracle_valid.shape
    positives = []
    hard_negatives = []
    other_negatives = []
    for y in range(h):
        for x in range(w):
            ov = int(oracle_valid[y, x] > 0)
            rc = int(runtime_candidate[y, x] > 0)
            rr = int(runtime_roi[y, x] > 0)
            coord = (y, x)
            if ov:
                positives.append(coord)
            elif rc or rr:
                hard_negatives.append(coord)
            else:
                other_negatives.append(coord)

    n = min(max_samples, len(positives) + len(hard_negatives) + len(other_negatives))
    if n == 0:
        return []
    target_pos = min(len(positives), max(1, n // 2))
    target_hard_neg = min(len(hard_negatives), max(1, (n - target_pos) // 2))
    target_other_neg = min(len(other_negatives), max(0, n - target_pos - target_hard_neg))
    if target_pos + target_hard_neg + target_other_neg < n:
        extra = n - (target_pos + target_hard_neg + target_other_neg)
        target_hard_neg = min(len(hard_negatives), target_hard_neg + extra)

    rng.shuffle(positives)
    rng.shuffle(hard_negatives)
    rng.shuffle(other_negatives)
    coords = positives[:target_pos] + hard_negatives[:target_hard_neg] + other_negatives[:target_other_neg]
    rng.shuffle(coords)
    return coords


def _iter_frame_indices(oracle_dir: Path, runtime_dir: Path):
    oracle_frames = {
        int(path.stem.split("_f")[-1])
        for path in oracle_dir.glob("valid_f*.png")
    }
    runtime_frames = {
        int(path.stem.split("_f")[-1])
        for path in runtime_dir.glob("promoted_mask_f*.png")
    }
    return sorted(oracle_frames & runtime_frames)


def _safe_mean(arr):
    import numpy as np
    return float(np.mean(arr)) if arr.size else 0.0


def _safe_std(arr):
    import numpy as np
    return float(np.std(arr)) if arr.size else 0.0


def _bbox_fill_ratio(region_mask, x, y, w, h):
    box = region_mask[y:y + h, x:x + w]
    if box.size == 0:
        return 0.0
    return float((box > 0).mean())


def _edge_distance_stats(edge_u8, region):
    vals = edge_u8[region].astype("float32") / 255.0
    return _safe_mean(vals), _safe_std(vals)


def _oracle_overlap(region, oracle_valid):
    import numpy as np
    inter = int(np.count_nonzero(region & (oracle_valid > 0)))
    area = int(np.count_nonzero(region))
    overlap = inter / max(area, 1)
    return overlap, inter, area


def export_region_dataset_for_frame(
    *,
    frame_idx: int,
    runtime_candidate,
    runtime_promoted,
    runtime_roi,
    oracle_valid,
    candidate_disp,
    candidate_cost,
    candidate_conf,
    candidate_lr_delta,
    candidate_median_delta,
    candidate_texture,
    candidate_disp_gradient,
    candidate_edge_distance=None,
    candidate_border_penalty=None,
):
    import cv2
    rows = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(runtime_candidate.astype("uint8"), connectivity=8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        region = labels == label
        if area <= 0:
            continue
        overlap_ratio, overlap_px, area_px = _oracle_overlap(region, oracle_valid)
        promoted_overlap_px = int(((region) & (runtime_promoted > 0)).sum())
        roi_overlap_px = int(((region) & (runtime_roi > 0)).sum())
        disp_vals = candidate_disp[region].astype("float32")
        cost_vals = candidate_cost[region].astype("float32")
        conf_vals = candidate_conf[region].astype("float32")
        lr_vals = candidate_lr_delta[region].astype("float32")
        med_vals = candidate_median_delta[region].astype("float32")
        tex_vals = candidate_texture[region].astype("float32")
        grad_vals = candidate_disp_gradient[region].astype("float32")
        fill_ratio = _bbox_fill_ratio(runtime_candidate > 0, x, y, w, h)
        bbox_area = int(w * h)
        row = {
            "frame_idx": frame_idx,
            "label": int(label),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "bbox_area": bbox_area,
            "area_px": area_px,
            "fill_ratio": fill_ratio,
            "roi_overlap_ratio": roi_overlap_px / max(area_px, 1),
            "promoted_overlap_ratio": promoted_overlap_px / max(area_px, 1),
            "oracle_overlap_ratio": overlap_ratio,
            "oracle_overlap_px": overlap_px,
            "oracle_label_loose": int(overlap_ratio >= 0.10),
            "oracle_label_strict": int(overlap_ratio >= 0.30),
            "disp_mean": _safe_mean(disp_vals),
            "disp_std": _safe_std(disp_vals),
            "cost_mean": _safe_mean(cost_vals),
            "cost_std": _safe_std(cost_vals),
            "conf_mean": _safe_mean(conf_vals),
            "conf_std": _safe_std(conf_vals),
            "lr_delta_mean": _safe_mean(lr_vals),
            "lr_delta_std": _safe_std(lr_vals),
            "median_delta_mean": _safe_mean(med_vals),
            "median_delta_std": _safe_std(med_vals),
            "texture_mean": _safe_mean(tex_vals),
            "texture_std": _safe_std(tex_vals),
            "disp_gradient_mean": _safe_mean(grad_vals),
            "disp_gradient_std": _safe_std(grad_vals),
        }
        if candidate_edge_distance is not None:
            edge_mean, edge_std = _edge_distance_stats(candidate_edge_distance, region)
            row["edge_distance_mean"] = edge_mean
            row["edge_distance_std"] = edge_std
        if candidate_border_penalty is not None:
            bp = candidate_border_penalty[region].astype("float32") / 255.0
            row["border_penalty_mean"] = _safe_mean(bp)
            row["border_penalty_std"] = _safe_std(bp)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", type=Path, required=True)
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl-regions", type=Path)
    ap.add_argument("--max-samples-per-frame", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl_regions is not None:
        args.output_jsonl_regions.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    sample_count = 0
    region_count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        region_out = args.output_jsonl_regions.open("w", encoding="utf-8") if args.output_jsonl_regions is not None else None
        try:
            for frame_idx in _iter_frame_indices(args.oracle_dir, args.runtime_dir):
                oracle_valid = _load_mask(args.oracle_dir / f"valid_f{frame_idx:04d}.png")
                runtime_promoted = _load_mask(args.runtime_dir / f"promoted_mask_f{frame_idx:04d}.png")
                runtime_candidate = _load_mask(args.runtime_dir / f"candidate_mask_f{frame_idx:04d}.png")
                runtime_roi = _load_mask(args.runtime_dir / f"roi_mask_f{frame_idx:04d}.png")
                candidate_disp = _load_gray(args.runtime_dir / f"candidate_disp_f{frame_idx:04d}.png")
                candidate_cost = _load_gray(args.runtime_dir / f"candidate_cost_f{frame_idx:04d}.png")
                candidate_conf = _load_gray(args.runtime_dir / f"candidate_conf_f{frame_idx:04d}.png")
                candidate_lr_delta = _load_gray(args.runtime_dir / f"candidate_lr_delta_f{frame_idx:04d}.png")
                candidate_median_delta = _load_gray(args.runtime_dir / f"candidate_median_delta_f{frame_idx:04d}.png")
                candidate_texture = _load_gray(args.runtime_dir / f"candidate_texture_f{frame_idx:04d}.png")
                candidate_disp_gradient = _load_gray(args.runtime_dir / f"candidate_disp_gradient_f{frame_idx:04d}.png")
                candidate_edge_distance = _load_gray(args.runtime_dir / f"candidate_edge_distance_f{frame_idx:04d}.png")
                candidate_border_penalty = _load_gray(args.runtime_dir / f"candidate_border_penalty_f{frame_idx:04d}.png")
                canonical_disp = _load_gray(args.runtime_dir / f"canonical_disp_f{frame_idx:04d}.png")
                if (
                    oracle_valid is None or runtime_promoted is None or runtime_candidate is None or runtime_roi is None
                    or candidate_disp is None or candidate_cost is None or candidate_conf is None
                    or candidate_lr_delta is None or candidate_median_delta is None or candidate_texture is None
                    or candidate_disp_gradient is None or candidate_edge_distance is None or candidate_border_penalty is None
                    or canonical_disp is None
                ):
                    continue
                coords = _balanced_coords(oracle_valid, runtime_candidate, runtime_roi, args.max_samples_per_frame, rng)
                for y, x in coords:
                    row = {
                        "frame_idx": frame_idx,
                        "x": x,
                        "y": y,
                        "candidate_disp_u8": int(candidate_disp[y, x]),
                        "candidate_cost_u8": int(candidate_cost[y, x]),
                        "candidate_conf_u8": int(candidate_conf[y, x]),
                        "candidate_lr_delta_u8": int(candidate_lr_delta[y, x]),
                        "candidate_median_delta_u8": int(candidate_median_delta[y, x]),
                        "candidate_texture_u8": int(candidate_texture[y, x]),
                        "candidate_disp_gradient_u8": int(candidate_disp_gradient[y, x]),
                        "candidate_edge_distance_u8": int(candidate_edge_distance[y, x]),
                        "candidate_border_penalty_u8": int(candidate_border_penalty[y, x]),
                        "canonical_disp_u8": int(canonical_disp[y, x]),
                        "runtime_roi": int(runtime_roi[y, x] > 0),
                        "runtime_candidate": int(runtime_candidate[y, x] > 0),
                        "runtime_promoted": int(runtime_promoted[y, x] > 0),
                        "oracle_valid": int(oracle_valid[y, x] > 0),
                        "candidate_oracle_hit": int((runtime_candidate[y, x] > 0) and (oracle_valid[y, x] > 0)),
                        "agreement": int((runtime_promoted[y, x] > 0) == (oracle_valid[y, x] > 0)),
                    }
                    out.write(json.dumps(row, sort_keys=True) + "\n")
                    sample_count += 1
                if region_out is not None:
                    region_rows = export_region_dataset_for_frame(
                        frame_idx=frame_idx,
                        runtime_candidate=runtime_candidate,
                        runtime_promoted=runtime_promoted,
                        runtime_roi=runtime_roi,
                        oracle_valid=oracle_valid,
                        candidate_disp=candidate_disp,
                        candidate_cost=candidate_cost,
                        candidate_conf=candidate_conf,
                        candidate_lr_delta=candidate_lr_delta,
                        candidate_median_delta=candidate_median_delta,
                        candidate_texture=candidate_texture,
                        candidate_disp_gradient=candidate_disp_gradient,
                        candidate_edge_distance=candidate_edge_distance,
                        candidate_border_penalty=candidate_border_penalty,
                    )
                    for row in region_rows:
                        region_out.write(json.dumps(row, sort_keys=True) + "\n")
                        region_count += 1
                frame_count += 1
        finally:
            if region_out is not None:
                region_out.close()

    print(f"[teacher] frames exported: {frame_count}")
    print(f"[teacher] samples exported: {sample_count}")
    if args.output_jsonl_regions is not None:
        print(f"[teacher] regions exported: {region_count}")
        print(f"[teacher] wrote region dataset: {args.output_jsonl_regions}")
    print(f"[teacher] wrote dataset: {args.output_jsonl}")


if __name__ == "__main__":
    main()
