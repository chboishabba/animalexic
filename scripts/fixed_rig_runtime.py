#!/usr/bin/env python3

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationArtifact:
    calibration_id: str
    image_width: int
    image_height: int
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    q_matrix: np.ndarray
    metadata: Dict[str, object]


def load_calibration_artifact(path: Path) -> CalibrationArtifact:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("calibration artifact must be a .npz file")
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing calibration metadata: {meta_path}")
    blob = np.load(path)
    metadata = json.loads(meta_path.read_text())
    return CalibrationArtifact(
        calibration_id=str(metadata["calibration_id"]),
        image_width=int(metadata["image_width"]),
        image_height=int(metadata["image_height"]),
        left_map_x=blob["left_map_x"],
        left_map_y=blob["left_map_y"],
        right_map_x=blob["right_map_x"],
        right_map_y=blob["right_map_y"],
        q_matrix=blob["Q"],
        metadata=metadata,
    )


def rectify_pair(left: np.ndarray, right: np.ndarray, artifact: CalibrationArtifact) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    if left.shape != (artifact.image_height, artifact.image_width):
        raise ValueError(
            f"left image shape {left.shape} does not match calibration {(artifact.image_height, artifact.image_width)}"
        )
    if right.shape != (artifact.image_height, artifact.image_width):
        raise ValueError(
            f"right image shape {right.shape} does not match calibration {(artifact.image_height, artifact.image_width)}"
        )
    rect_left = cv2.remap(left, artifact.left_map_x, artifact.left_map_y, interpolation=cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, artifact.right_map_x, artifact.right_map_y, interpolation=cv2.INTER_LINEAR)
    return rect_left, rect_right


def _tile_reduce(mask: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = mask.shape
    tile_h = (h + tile_size - 1) // tile_size
    tile_w = (w + tile_size - 1) // tile_size
    out = np.zeros((tile_h, tile_w), dtype=np.uint8)
    for ty in range(tile_h):
        y0 = ty * tile_size
        y1 = min(y0 + tile_size, h)
        for tx in range(tile_w):
            x0 = tx * tile_size
            x1 = min(x0 + tile_size, w)
            if np.any(mask[y0:y1, x0:x1] != 0):
                out[ty, tx] = 1
    return out


def _expand_tile_mask(tile_mask: np.ndarray, halo_tiles: int) -> np.ndarray:
    if halo_tiles <= 0:
        return tile_mask.copy()
    h, w = tile_mask.shape
    out = tile_mask.copy()
    active = np.argwhere(tile_mask != 0)
    for ty, tx in active:
        y0 = max(0, ty - halo_tiles)
        y1 = min(h, ty + halo_tiles + 1)
        x0 = max(0, tx - halo_tiles)
        x1 = min(w, tx + halo_tiles + 1)
        out[y0:y1, x0:x1] = 1
    return out


def _tile_mask_to_pixel_mask(tile_mask: np.ndarray, image_shape: Tuple[int, int], tile_size: int) -> np.ndarray:
    h, w = image_shape
    pixel = np.zeros((h, w), dtype=np.uint8)
    for ty in range(tile_mask.shape[0]):
        y0 = ty * tile_size
        y1 = min(y0 + tile_size, h)
        for tx in range(tile_mask.shape[1]):
            if tile_mask[ty, tx] == 0:
                continue
            x0 = tx * tile_size
            x1 = min(x0 + tile_size, w)
            pixel[y0:y1, x0:x1] = 1
    return pixel


def build_delta_roi(
    prev_frame: Optional[np.ndarray],
    curr_frame: np.ndarray,
    diff_threshold: int,
    min_luma: int,
    tile_size: int,
    tile_halo: int,
) -> Dict[str, object]:
    h, w = curr_frame.shape
    if prev_frame is None:
        pixel_mask = np.ones((h, w), dtype=np.uint8)
        diff = np.zeros((h, w), dtype=np.uint16)
    else:
        diff = np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16)).astype(np.uint16)
        bright = np.maximum(prev_frame, curr_frame) >= min_luma
        pixel_mask = ((diff >= diff_threshold) & bright).astype(np.uint8)

    tile_mask = _tile_reduce(pixel_mask, tile_size)
    tile_mask = _expand_tile_mask(tile_mask, tile_halo)
    roi_mask = _tile_mask_to_pixel_mask(tile_mask, curr_frame.shape, tile_size)

    tiles = []
    for ty in range(tile_mask.shape[0]):
        y0 = ty * tile_size
        y1 = min(y0 + tile_size, h)
        for tx in range(tile_mask.shape[1]):
            if tile_mask[ty, tx] == 0:
                continue
            x0 = tx * tile_size
            x1 = min(x0 + tile_size, w)
            tiles.append((x0, y0, x1 - x0, y1 - y0))

    return {
        "diff": diff,
        "pixel_mask": pixel_mask,
        "tile_mask": tile_mask,
        "roi_mask": roi_mask,
        "tiles": tiles,
    }


def promote_disparity(
    prev_promoted: Optional[np.ndarray],
    cand_disp: np.ndarray,
    cost_min: np.ndarray,
    conf_gap: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    tau_cost: int,
    tau_conf: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    promoted_mask = (
        (roi_mask != 0)
        & (valid_mask != 0)
        & (cost_min <= tau_cost)
        & (conf_gap >= tau_conf)
    )
    reject_mask = (roi_mask != 0) & (valid_mask == 0)
    abstain_mask = (roi_mask != 0) & (~promoted_mask) & (~reject_mask)

    if prev_promoted is None:
        canonical = np.zeros_like(cand_disp)
    else:
        canonical = prev_promoted.copy()
    canonical[promoted_mask] = cand_disp[promoted_mask]

    residual = np.where(promoted_mask, cost_min, 0).astype(np.uint16)
    promoted_costs = cost_min[promoted_mask]
    metrics = {
        "promoted_count": int(promoted_mask.sum()),
        "abstained_count": int(abstain_mask.sum()),
        "rejected_count": int(reject_mask.sum()),
        "roi_count": int((roi_mask != 0).sum()),
        "residual_mean": float(promoted_costs.mean()) if promoted_costs.size else 0.0,
        "residual_p95": float(np.percentile(promoted_costs, 95)) if promoted_costs.size else 0.0,
        "coverage_pct": (100.0 * float(promoted_mask.sum()) / float(max(1, roi_mask.sum()))),
        "residual": residual,
        "promoted_mask": promoted_mask.astype(np.uint8),
        "abstain_mask": abstain_mask.astype(np.uint8),
        "reject_mask": reject_mask.astype(np.uint8),
    }
    return canonical, metrics


def depth_from_disparity(disparity_q8: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    disp = disparity_q8.astype(np.float32) / 256.0
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0.0
    if not np.any(valid):
        return depth
    q = q_matrix.astype(np.float32)
    denom = q[3, 2] * disp[valid] + q[3, 3]
    safe = np.abs(denom) > 1e-6
    z = np.zeros_like(disp[valid], dtype=np.float32)
    z[safe] = (q[2, 3]) / denom[safe]
    depth[valid] = z
    return depth


@dataclass
class TemporalMergeParams:
    max_severity_promote: int = 1
    max_cost: float = 24.0
    min_gap: float = 4.0
    min_conf: float = 0.45
    tau_close_disp: float = 1.5
    conf_improvement_req: float = 0.15
    max_age_keep: int = 12
    min_stability_for_strong: int = 3
    smooth_alpha_weak: float = 0.35
    smooth_alpha_strong: float = 0.65
    min_evidence_frames: int = 2
    weak_conf_scale: float = 0.7


def confidence_from_cost_gap(cost: np.ndarray, gap: np.ndarray, max_cost: float, good_gap: float = 10.0) -> np.ndarray:
    cost_term = np.clip(1.0 - (cost.astype(np.float32) / max_cost), 0.0, 1.0)
    gap_term = np.clip(gap.astype(np.float32) / good_gap, 0.0, 1.0)
    return np.clip(0.6 * cost_term + 0.4 * gap_term, 0.0, 1.0)


def merge_disparity_state(
    prev_disp: np.ndarray,
    prev_conf: np.ndarray,
    prev_age: np.ndarray,
    prev_stability: np.ndarray,
    prev_valid: np.ndarray,
    prev_evidence: np.ndarray,
    cand_disp: np.ndarray,
    cand_valid: np.ndarray,
    cand_cost: np.ndarray,
    cand_gap: np.ndarray,
    cand_severity: np.ndarray,
    roi_mask: np.ndarray,
    params: TemporalMergeParams,
):
    """
    Merge candidate disparity into persistent state.
    Disparities are float32 (pixel disparity). Masks are bool or 0/1 arrays.
    """
    # Start from previous state
    new_disp = prev_disp.copy()
    new_conf = prev_conf.copy()
    new_age = prev_age.copy() + 1
    new_stability = np.maximum(prev_stability - 1, 0)
    new_valid = prev_valid.copy()
    evidence = prev_evidence.copy()

    roi = roi_mask.astype(bool)
    cand_valid_mask = cand_valid.astype(bool)
    severity_ok = cand_severity <= params.max_severity_promote
    cand_conf = confidence_from_cost_gap(cand_cost, cand_gap, params.max_cost)

    hard_ok = (
        roi
        & cand_valid_mask
        & severity_ok
        & (cand_disp > 0.0)
        & (cand_cost.astype(np.float32) <= params.max_cost)
        & (cand_gap.astype(np.float32) >= params.min_gap)
        & (cand_conf >= params.min_conf)
    )

    prev_exists = prev_valid.astype(bool)
    delta = np.abs(cand_disp - prev_disp)

    # Evidence accumulation (weaker threshold)
    increment = roi & cand_valid_mask & severity_ok & (cand_conf >= params.min_conf * params.weak_conf_scale)
    evidence = np.where(increment, np.minimum(evidence + 1, 255), np.where(roi, np.maximum(evidence - 1, 0), evidence))

    accept_close = hard_ok & (~prev_exists | (delta <= params.tau_close_disp))
    accept_better = (
        hard_ok
        & prev_exists
        & (delta > params.tau_close_disp)
        & (cand_conf >= (prev_conf + params.conf_improvement_req))
    )

    accept_temporal = (
        roi
        & cand_valid_mask
        & severity_ok
        & (evidence >= params.min_evidence_frames)
        & (cand_conf >= params.min_conf * params.weak_conf_scale)
    )

    accept = accept_close | accept_better | accept_temporal

    stability_next = np.where(
        accept & prev_exists & (delta <= params.tau_close_disp),
        np.minimum(prev_stability + 1, 255),
        np.where(accept, 1, new_stability),
    )

    strong = accept & (stability_next >= params.min_stability_for_strong)
    alpha = np.where(strong, params.smooth_alpha_strong, params.smooth_alpha_weak)

    blended = alpha * cand_disp + (1.0 - alpha) * prev_disp
    new_disp = np.where(accept, blended, new_disp)
    new_conf = np.where(accept, np.maximum(prev_conf, cand_conf), new_conf)
    new_age = np.where(accept, 0, new_age)
    new_stability = stability_next.astype(prev_stability.dtype)
    new_valid = np.where(accept, True, new_valid)

    expire = new_valid & (new_age > params.max_age_keep) & (~roi)
    new_valid = np.where(expire, False, new_valid)
    new_conf = np.where(expire, 0.0, new_conf)

    stats = {
        "accepted_pixels": int(np.count_nonzero(accept)),
        "accepted_close_pixels": int(np.count_nonzero(accept_close)),
        "accepted_better_pixels": int(np.count_nonzero(accept_better)),
        "accepted_temporal_pixels": int(np.count_nonzero(accept_temporal)),
        "expired_pixels": int(np.count_nonzero(expire)),
        "mean_candidate_conf": float(cand_conf[hard_ok].mean()) if np.any(hard_ok) else 0.0,
    }

    return new_disp, new_conf, new_age, new_stability, new_valid, evidence, cand_conf, accept, stats


class ReceiptStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                started_at REAL NOT NULL,
                source TEXT NOT NULL,
                calibration_id TEXT,
                config_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                frame_time REAL NOT NULL,
                stage_json TEXT NOT NULL,
                roi_tiles INTEGER NOT NULL,
                roi_pixels INTEGER NOT NULL,
                roi_coverage REAL NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                frame_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                roi_set_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                thresholds_json TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                residual_mean REAL NOT NULL,
                residual_p95 REAL NOT NULL,
                invariant_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id),
                FOREIGN KEY(frame_id) REFERENCES frames(id)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                frame_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id),
                FOREIGN KEY(frame_id) REFERENCES frames(id)
            );
            """
        )
        self.conn.commit()

    def create_run(self, source: str, calibration_id: Optional[str], config: Dict[str, object]) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at, source, calibration_id, config_json) VALUES (?, ?, ?, ?)",
            (time.time(), source, calibration_id, json.dumps(config, sort_keys=True)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def write_frame_metrics(
        self,
        run_id: int,
        frame_index: int,
        stage_metrics: Dict[str, float],
        roi_tiles: int,
        roi_pixels: int,
        roi_coverage: float,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO frames(run_id, frame_index, frame_time, stage_json, roi_tiles, roi_pixels, roi_coverage) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                frame_index,
                time.time(),
                json.dumps(stage_metrics, sort_keys=True),
                roi_tiles,
                roi_pixels,
                roi_coverage,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def write_receipt(
        self,
        run_id: int,
        frame_id: int,
        kind: str,
        roi_set_id: str,
        decision: str,
        thresholds: Dict[str, object],
        counts: Dict[str, object],
        residual_mean: float,
        residual_p95: float,
        invariants: Dict[str, object],
    ):
        self.conn.execute(
            "INSERT INTO receipts(run_id, frame_id, kind, roi_set_id, decision, thresholds_json, counts_json, residual_mean, residual_p95, invariant_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                frame_id,
                kind,
                roi_set_id,
                decision,
                json.dumps(thresholds, sort_keys=True),
                json.dumps(counts, sort_keys=True),
                residual_mean,
                residual_p95,
                json.dumps(invariants, sort_keys=True),
            ),
        )
        self.conn.commit()

    def write_artifact(self, run_id: int, frame_id: int, kind: str, path: Path):
        self.conn.execute(
            "INSERT INTO artifacts(run_id, frame_id, kind, path) VALUES (?, ?, ?, ?)",
            (run_id, frame_id, kind, str(path)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
