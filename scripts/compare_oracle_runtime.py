#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrameStats:
    frame_idx: int
    oracle_frame_idx: int | None = None
    runtime_frame_idx: int | None = None
    total_px: int | None = None

    oracle_valid_px: int | None = None
    oracle_valid_ratio: float | None = None
    oracle_disp_mean: float | None = None
    oracle_rectify_ms: float | None = None
    oracle_disparity_ms: float | None = None
    source_selected_index: int | None = None
    source_pts_time: float | None = None

    runtime_valid_px: int | None = None
    runtime_valid_ratio: float | None = None
    runtime_candidate_valid_px: int | None = None
    runtime_candidate_valid_ratio: float | None = None
    runtime_disp_mean: float | None = None
    runtime_cost_mean: float | None = None
    runtime_conf_mean: float | None = None
    runtime_roi_px: int | None = None
    runtime_promoted_px: int | None = None
    runtime_expired_px: int | None = None
    runtime_stereo_ms: float | None = None
    runtime_promote_ms: float | None = None

    coverage_gap_px: int | None = None
    coverage_gap_ratio: float | None = None
    candidate_gap_px: int | None = None
    candidate_gap_ratio: float | None = None
    overlap_iou: float | None = None
    candidate_overlap_iou: float | None = None
    oracle_only_px: int | None = None
    runtime_only_px: int | None = None
    intersection_px: int | None = None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _mean(xs):
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_oracle_summary(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", [])
    by_index = {}
    for row in frames:
        frame_idx = _safe_int(row.get("frame_idx", row.get("frame_index", row.get("frame"))))
        if frame_idx is None:
            continue
        valid_px = _safe_int(row.get("valid_px", row.get("valid_pixels", row.get("valid"))))
        total_px = _safe_int(row.get("total_px", row.get("total_pixels")))
        valid_ratio = _safe_float(row.get("valid_ratio", row.get("valid_pct")))
        if valid_ratio is not None and valid_ratio > 1.0:
            valid_ratio /= 100.0
        if valid_ratio is None and valid_px is not None and total_px:
            valid_ratio = valid_px / total_px
        by_index[frame_idx] = {
            "frame_idx": frame_idx,
            "valid_px": valid_px,
            "total_px": total_px,
            "valid_ratio": valid_ratio,
            "disp_mean": _safe_float(row.get("disp_mean")),
            "rectify_ms": _safe_float(row.get("rectify_ms")),
            "disparity_ms": _safe_float(row.get("disparity_ms")),
            "source_selected_index": _safe_int(row.get("source_selected_index", row.get("source_frame_index"))),
            "source_pts_time": _safe_float(row.get("source_pts_time")),
        }
    return payload, by_index


def _load_oracle_receipts(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        frame_idx = _safe_int(row.get("frame_idx", row.get("frame_index", row.get("frame"))))
        if frame_idx is None:
            continue
        valid_px = _safe_int(row.get("valid_px", row.get("valid_pixels", row.get("valid"))))
        total_px = _safe_int(row.get("total_px", row.get("total_pixels")))
        valid_ratio = _safe_float(row.get("valid_ratio", row.get("valid_pct")))
        if valid_ratio is not None and valid_ratio > 1.0:
            valid_ratio /= 100.0
        if valid_ratio is None and valid_px is not None and total_px:
            valid_ratio = valid_px / total_px
        out[frame_idx] = {
            "frame_idx": frame_idx,
            "valid_px": valid_px,
            "total_px": total_px,
            "valid_ratio": valid_ratio,
            "disp_mean": _safe_float(row.get("disp_mean")),
            "rectify_ms": _safe_float(row.get("rectify_ms")),
            "disparity_ms": _safe_float(row.get("disparity_ms")),
            "source_selected_index": _safe_int(row.get("source_selected_index", row.get("source_frame_index"))),
            "source_pts_time": _safe_float(row.get("source_pts_time")),
        }
    return out


def _merge_oracle(summary_rows: dict[int, dict[str, Any]], receipt_rows: dict[int, dict[str, Any]]):
    out: dict[int, dict[str, Any]] = {}
    for frame_idx in sorted(set(summary_rows) | set(receipt_rows)):
        merged = {}
        merged.update(receipt_rows.get(frame_idx, {}))
        merged.update(summary_rows.get(frame_idx, {}))
        out[frame_idx] = merged
    return out


def _select_runtime_run(con: sqlite3.Connection, run_id: int | None):
    if run_id is not None:
        row = con.execute(
            "SELECT id, source, calibration_id, config_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    else:
        row = con.execute(
            "SELECT id, source, calibration_id, config_json FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("no runtime run found in receipts DB")
    return {
        "run_id": int(row[0]),
        "source": row[1],
        "calibration_id": row[2],
        "config": json.loads(row[3]) if row[3] else {},
    }


def _load_runtime_frames_sqlite(path: Path, run_id: int | None) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    con = sqlite3.connect(str(path))
    try:
        runtime_run = _select_runtime_run(con, run_id)
        rows = con.execute(
            """
            SELECT
                f.frame_index,
                f.stage_json,
                f.roi_pixels,
                f.roi_coverage,
                r.counts_json,
                r.thresholds_json,
                r.invariant_json,
                r.residual_mean
            FROM frames f
            LEFT JOIN receipts r
              ON r.frame_id = f.id
             AND r.run_id = f.run_id
             AND r.kind = 'disparity'
            WHERE f.run_id = ?
            ORDER BY f.frame_index ASC
            """,
            (runtime_run["run_id"],),
        ).fetchall()
    finally:
        con.close()

    out: dict[int, dict[str, Any]] = {}
    total_px = int(runtime_run["config"].get("width", 0)) * int(runtime_run["config"].get("height", 0))
    for row in rows:
        frame_idx = int(row[0])
        stage = json.loads(row[1]) if row[1] else {}
        counts = json.loads(row[4]) if row[4] else {}
        thresholds = json.loads(row[5]) if row[5] else {}
        invariants = json.loads(row[6]) if row[6] else {}
        promoted_px = _safe_int(counts.get("promoted", counts.get("promoted_px")))
        candidate_valid_px = _safe_int(counts.get("candidate_valid_px", promoted_px))
        valid_ratio = None
        candidate_valid_ratio = None
        if promoted_px is not None and total_px > 0:
            valid_ratio = promoted_px / total_px
        if candidate_valid_px is not None and total_px > 0:
            candidate_valid_ratio = candidate_valid_px / total_px
        out[frame_idx] = {
            "frame_idx": frame_idx,
            "total_px": total_px if total_px > 0 else None,
            "valid_px": promoted_px,
            "valid_ratio": valid_ratio,
            "candidate_valid_px": candidate_valid_px,
            "candidate_valid_ratio": candidate_valid_ratio,
            "disp_mean": None,
            "cost_mean": _safe_float(row[7]),
            "conf_mean": None,
            "roi_px": _safe_int(row[2]),
            "roi_coverage": _safe_float(row[3]),
            "promoted_px": promoted_px,
            "expired_px": _safe_int(counts.get("expired", counts.get("expired_px"))),
            "thresholds": thresholds,
            "invariants": invariants,
            "stage": stage,
            "source_selected_index": _safe_int(stage.get("source_selected_index")),
            "source_pts_time": _safe_float(stage.get("source_pts_time")),
        }
    return runtime_run, out


def _load_runtime_jsonl(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    out: dict[int, dict[str, Any]] = {}
    config: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        frame_idx = _safe_int(row.get("frame_idx", row.get("frame_index", row.get("frame"))))
        if frame_idx is None:
            continue
        total_px = _safe_int(row.get("total_px", row.get("total_pixels")))
        valid_px = _safe_int(row.get("valid_px", row.get("valid_pixels", row.get("valid"))))
        valid_ratio = _safe_float(row.get("valid_ratio", row.get("valid_pct")))
        if valid_ratio is not None and valid_ratio > 1.0:
            valid_ratio /= 100.0
        if valid_ratio is None and valid_px is not None and total_px:
            valid_ratio = valid_px / total_px
        out[frame_idx] = {
            "frame_idx": frame_idx,
            "total_px": total_px,
            "valid_px": valid_px,
            "valid_ratio": valid_ratio,
            "candidate_valid_px": _safe_int(row.get("candidate_valid_px", valid_px)),
            "candidate_valid_ratio": _safe_float(row.get("candidate_valid_ratio", valid_ratio)),
            "disp_mean": _safe_float(row.get("disp_mean")),
            "cost_mean": _safe_float(row.get("cost_mean")),
            "conf_mean": _safe_float(row.get("conf_mean")),
            "roi_px": _safe_int(row.get("roi_px")),
            "roi_coverage": _safe_float(row.get("roi_coverage")),
            "promoted_px": _safe_int(row.get("promoted_px", row.get("promoted"))),
            "expired_px": _safe_int(row.get("expired_px", row.get("expired"))),
            "stage": {},
            "source_selected_index": _safe_int(row.get("source_selected_index", row.get("source_frame_index"))),
            "source_pts_time": _safe_float(row.get("source_pts_time")),
            "thresholds": row.get("thresholds", {}),
            "invariants": row.get("invariants", {}),
        }
    return {"run_id": None, "source": str(path), "calibration_id": None, "config": config}, out


def _load_mask(path: Path):
    import cv2
    import numpy as np

    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def _mask_metrics(oracle_mask_path: Path, runtime_mask_path: Path):
    try:
        oracle_mask = _load_mask(oracle_mask_path)
        runtime_mask = _load_mask(runtime_mask_path)
    except Exception:
        return None
    if oracle_mask is None or runtime_mask is None or oracle_mask.shape != runtime_mask.shape:
        return None
    oracle_valid = oracle_mask != 0
    runtime_valid = runtime_mask != 0
    intersection = int((oracle_valid & runtime_valid).sum())
    union = int((oracle_valid | runtime_valid).sum())
    oracle_only = int((oracle_valid & ~runtime_valid).sum())
    runtime_only = int((runtime_valid & ~oracle_valid).sum())
    return {
        "overlap_iou": (intersection / union) if union else 0.0,
        "oracle_only_px": oracle_only,
        "runtime_only_px": runtime_only,
        "intersection_px": intersection,
    }


def _write_disagreement_heatmap(oracle_mask_path: Path, runtime_mask_path: Path, out_path: Path):
    import cv2
    import numpy as np

    oracle_mask = _load_mask(oracle_mask_path)
    runtime_mask = _load_mask(runtime_mask_path)
    if oracle_mask is None or runtime_mask is None or oracle_mask.shape != runtime_mask.shape:
        return False
    oracle_valid = oracle_mask != 0
    runtime_valid = runtime_mask != 0
    heat = np.zeros((oracle_mask.shape[0], oracle_mask.shape[1], 3), dtype=np.uint8)
    # TP green, FP red, FN blue
    heat[oracle_valid & runtime_valid] = (0, 200, 0)
    heat[(~oracle_valid) & runtime_valid] = (220, 0, 0)
    heat[oracle_valid & (~runtime_valid)] = (0, 0, 220)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), heat)
    return True


def _join_key(row: dict[str, Any]):
    pts = _safe_float(row.get("source_pts_time"))
    if pts is not None:
        return ("pts_ms", int(round(pts * 1000.0)))
    selected = _safe_int(row.get("source_selected_index"))
    if selected is not None:
        return ("selected_index", selected)
    frame_idx = _safe_int(row.get("frame_idx"))
    if frame_idx is not None:
        return ("frame_idx", frame_idx)
    return None


def _join_candidates(row: dict[str, Any]) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    pts = _safe_float(row.get("source_pts_time"))
    if pts is not None:
        keys.append(("pts_ms", int(round(pts * 1000.0))))
    selected = _safe_int(row.get("source_selected_index"))
    if selected is not None:
        keys.append(("selected_index", selected))
    frame_idx = _safe_int(row.get("frame_idx"))
    if frame_idx is not None:
        keys.append(("frame_idx", frame_idx))
    return keys


def _join_frames(
    oracle_rows: dict[int, dict[str, Any]],
    runtime_rows: dict[int, dict[str, Any]],
    oracle_dir: Path,
    runtime_dir: Path,
) -> list[FrameStats]:
    runtime_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in runtime_rows.values():
        for key in _join_candidates(row):
            runtime_by_key.setdefault(key, []).append(row)

    matched_runtime_ids: set[int] = set()
    join_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for frame_idx in sorted(oracle_rows):
        oracle = oracle_rows[frame_idx]
        runtime = {}
        for key in _join_candidates(oracle):
            candidates = runtime_by_key.get(key, [])
            match = next((row for row in candidates if id(row) not in matched_runtime_ids), None)
            if match is not None:
                runtime = match
                matched_runtime_ids.add(id(match))
                break
        join_pairs.append((oracle, runtime))

    for frame_idx in sorted(runtime_rows):
        runtime = runtime_rows[frame_idx]
        if id(runtime) in matched_runtime_ids:
            continue
        join_pairs.append(({}, runtime))

    frames = []
    for oracle, runtime in join_pairs:
        frame_idx = _safe_int(oracle.get("frame_idx", runtime.get("frame_idx")))
        total_px = _safe_int(oracle.get("total_px", runtime.get("total_px")))
        oracle_valid_px = _safe_int(oracle.get("valid_px"))
        runtime_valid_px = _safe_int(runtime.get("valid_px"))
        runtime_candidate_valid_px = _safe_int(runtime.get("candidate_valid_px"))
        oracle_valid_ratio = _safe_float(oracle.get("valid_ratio"))
        runtime_valid_ratio = _safe_float(runtime.get("valid_ratio"))
        runtime_candidate_valid_ratio = _safe_float(runtime.get("candidate_valid_ratio"))
        gap_px = None
        if oracle_valid_px is not None and runtime_valid_px is not None:
            gap_px = oracle_valid_px - runtime_valid_px
        gap_ratio = None
        if oracle_valid_ratio is not None and runtime_valid_ratio is not None:
            gap_ratio = oracle_valid_ratio - runtime_valid_ratio
        candidate_gap_px = None
        if oracle_valid_px is not None and runtime_candidate_valid_px is not None:
            candidate_gap_px = oracle_valid_px - runtime_candidate_valid_px
        candidate_gap_ratio = None
        if oracle_valid_ratio is not None and runtime_candidate_valid_ratio is not None:
            candidate_gap_ratio = oracle_valid_ratio - runtime_candidate_valid_ratio

        oracle_frame_idx = _safe_int(oracle.get("frame_idx"))
        runtime_frame_idx = _safe_int(runtime.get("frame_idx"))
        frame = FrameStats(
            frame_idx=frame_idx if frame_idx is not None else -1,
            oracle_frame_idx=oracle_frame_idx,
            runtime_frame_idx=runtime_frame_idx,
            total_px=total_px,
            oracle_valid_px=oracle_valid_px,
            oracle_valid_ratio=oracle_valid_ratio,
            oracle_disp_mean=_safe_float(oracle.get("disp_mean")),
            oracle_rectify_ms=_safe_float(oracle.get("rectify_ms")),
            oracle_disparity_ms=_safe_float(oracle.get("disparity_ms")),
            source_selected_index=_safe_int(oracle.get("source_selected_index", runtime.get("source_selected_index"))),
            source_pts_time=_safe_float(oracle.get("source_pts_time", runtime.get("source_pts_time"))),
            runtime_valid_px=runtime_valid_px,
            runtime_valid_ratio=runtime_valid_ratio,
            runtime_candidate_valid_px=runtime_candidate_valid_px,
            runtime_candidate_valid_ratio=runtime_candidate_valid_ratio,
            runtime_disp_mean=_safe_float(runtime.get("disp_mean")),
            runtime_cost_mean=_safe_float(runtime.get("cost_mean")),
            runtime_conf_mean=_safe_float(runtime.get("conf_mean")),
            runtime_roi_px=_safe_int(runtime.get("roi_px")),
            runtime_promoted_px=_safe_int(runtime.get("promoted_px")),
            runtime_expired_px=_safe_int(runtime.get("expired_px")),
            runtime_stereo_ms=_safe_float(runtime.get("stage", {}).get("stereo_ms")),
            runtime_promote_ms=_safe_float(runtime.get("stage", {}).get("promote_ms")),
            coverage_gap_px=gap_px,
            coverage_gap_ratio=gap_ratio,
            candidate_gap_px=candidate_gap_px,
            candidate_gap_ratio=candidate_gap_ratio,
        )
        if oracle_frame_idx is not None and runtime_frame_idx is not None:
            mask_metrics = _mask_metrics(
                oracle_dir / f"valid_f{oracle_frame_idx:04d}.png",
                runtime_dir / f"promoted_mask_f{runtime_frame_idx:04d}.png",
            )
            if mask_metrics is not None:
                frame.overlap_iou = _safe_float(mask_metrics["overlap_iou"])
                frame.oracle_only_px = _safe_int(mask_metrics["oracle_only_px"])
                frame.runtime_only_px = _safe_int(mask_metrics["runtime_only_px"])
                frame.intersection_px = _safe_int(mask_metrics["intersection_px"])
            candidate_mask_metrics = _mask_metrics(
                oracle_dir / f"valid_f{oracle_frame_idx:04d}.png",
                runtime_dir / f"candidate_mask_f{runtime_frame_idx:04d}.png",
            )
            if candidate_mask_metrics is not None:
                frame.candidate_overlap_iou = _safe_float(candidate_mask_metrics["overlap_iou"])
        frames.append(frame)
    return frames


def _summarize(frames: list[FrameStats], oracle_summary_raw: dict[str, Any], runtime_meta: dict[str, Any]):
    both = [f for f in frames if f.oracle_valid_ratio is not None and f.runtime_valid_ratio is not None]
    high_oracle = [f for f in both if f.oracle_valid_ratio is not None and f.oracle_valid_ratio >= 0.20]
    summary = {
        "num_frames_joined": len(frames),
        "num_frames_both": len(both),
        "oracle_valid_ratio_mean": _mean(f.oracle_valid_ratio for f in both),
        "runtime_valid_ratio_mean": _mean(f.runtime_valid_ratio for f in both),
        "runtime_candidate_valid_ratio_mean": _mean(f.runtime_candidate_valid_ratio for f in both),
        "coverage_gap_ratio_mean": _mean(f.coverage_gap_ratio for f in both),
        "candidate_gap_ratio_mean": _mean(f.candidate_gap_ratio for f in both),
        "runtime_stereo_ms_mean": _mean(f.runtime_stereo_ms for f in both),
        "runtime_promote_ms_mean": _mean(f.runtime_promote_ms for f in both),
        "mask_overlap_iou_mean": _mean(f.overlap_iou for f in both),
        "candidate_mask_overlap_iou_mean": _mean(f.candidate_overlap_iou for f in both),
        "source_pts_time_count": sum(1 for f in both if f.source_pts_time is not None),
        "source_selected_index_count": sum(1 for f in both if f.source_selected_index is not None),
        "oracle_high_coverage_frames": [f.frame_idx for f in high_oracle],
        "oracle_high_coverage_mean": _mean(f.oracle_valid_ratio for f in high_oracle),
        "runtime_on_oracle_high_mean": _mean(f.runtime_valid_ratio for f in high_oracle),
        "runtime_roi_ratio_on_oracle_high_mean": _mean(
            (f.runtime_roi_px / f.total_px) if f.runtime_roi_px is not None and f.total_px else None
            for f in high_oracle
        ),
        "runtime_cost_mean_on_oracle_high": _mean(f.runtime_cost_mean for f in high_oracle),
        "runtime_conf_mean_on_oracle_high": _mean(f.runtime_conf_mean for f in high_oracle),
        "runtime_meta": runtime_meta,
        "oracle_summary_raw": oracle_summary_raw,
    }
    recommendations = _recommend(summary)
    return summary, recommendations


def _recommend(summary: dict[str, Any]):
    rec: dict[str, Any] = {"signals": [], "recommended_direction": {}}
    gap = _safe_float(summary.get("coverage_gap_ratio_mean"))
    roi = _safe_float(summary.get("runtime_roi_ratio_on_oracle_high_mean"))
    cost = _safe_float(summary.get("runtime_cost_mean_on_oracle_high"))
    conf = _safe_float(summary.get("runtime_conf_mean_on_oracle_high"))
    if gap is not None and gap > 0.10:
        rec["signals"].append("runtime under-covers oracle materially")
        rec["recommended_direction"]["min_conf"] = "decrease slightly"
        rec["recommended_direction"]["min_gap"] = "decrease slightly"
        rec["recommended_direction"]["max_cost"] = "increase slightly"
    if roi is not None and roi < 0.95:
        rec["signals"].append("runtime ROI excludes oracle-good regions")
        rec["recommended_direction"]["diff_threshold"] = "decrease"
        rec["recommended_direction"]["tile_halo"] = "increase"
    if cost is not None and cost > 2.0:
        rec["signals"].append("runtime accepted candidates are still relatively costly")
        rec["recommended_direction"]["stereo_matcher"] = "improve local evidence, not only merge thresholds"
    if conf is not None and conf < 1.5:
        rec["signals"].append("runtime confidence signal is weak on oracle-good frames")
        rec["recommended_direction"]["conf_calibration"] = "recompute against oracle labels"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-summary", type=Path, required=True)
    ap.add_argument("--oracle-receipts", type=Path, help="Optional oracle JSONL receipts")
    runtime = ap.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--runtime-sqlite", type=Path)
    runtime.add_argument("--runtime-jsonl", type=Path)
    ap.add_argument("--runtime-run-id", type=int, help="Specific runtime run id for sqlite input")
    ap.add_argument("--runtime-output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    oracle_summary_raw, oracle_summary_rows = _load_oracle_summary(args.oracle_summary)
    oracle_receipt_rows = _load_oracle_receipts(args.oracle_receipts) if args.oracle_receipts else {}
    oracle_rows = _merge_oracle(oracle_summary_rows, oracle_receipt_rows)

    if args.runtime_sqlite:
        runtime_meta, runtime_rows = _load_runtime_frames_sqlite(args.runtime_sqlite, args.runtime_run_id)
    else:
        runtime_meta, runtime_rows = _load_runtime_jsonl(args.runtime_jsonl)

    frames = _join_frames(
        oracle_rows,
        runtime_rows,
        args.oracle_summary.parent,
        args.runtime_output_dir,
    )
    summary, recommendations = _summarize(frames, oracle_summary_raw, runtime_meta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "joined_frames.json").write_text(
        json.dumps([asdict(frame) for frame in frames], indent=2, sort_keys=True)
    )
    for frame in frames:
        if frame.frame_idx < 0:
            continue
        if frame.oracle_frame_idx is None or frame.runtime_frame_idx is None:
            continue
        _write_disagreement_heatmap(
            args.oracle_summary.parent / f"valid_f{frame.oracle_frame_idx:04d}.png",
            args.runtime_output_dir / f"promoted_mask_f{frame.runtime_frame_idx:04d}.png",
            args.output_dir / f"promoted_overlap_f{frame.frame_idx:04d}.png",
        )
        _write_disagreement_heatmap(
            args.oracle_summary.parent / f"valid_f{frame.oracle_frame_idx:04d}.png",
            args.runtime_output_dir / f"candidate_mask_f{frame.runtime_frame_idx:04d}.png",
            args.output_dir / f"candidate_overlap_f{frame.frame_idx:04d}.png",
        )
    result = {"summary": summary, "recommendations": recommendations}
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"[compare] wrote summary: {args.output_dir / 'comparison_summary.json'}")
    print(f"[compare] frames joined: {summary['num_frames_joined']}")
    if summary["oracle_valid_ratio_mean"] is not None:
        print(f"[compare] oracle mean valid pct: {100.0 * summary['oracle_valid_ratio_mean']:.2f}")
    if summary["runtime_candidate_valid_ratio_mean"] is not None:
        print(f"[compare] runtime mean candidate pct: {100.0 * summary['runtime_candidate_valid_ratio_mean']:.2f}")
    if summary["runtime_valid_ratio_mean"] is not None:
        print(f"[compare] runtime mean promoted pct: {100.0 * summary['runtime_valid_ratio_mean']:.2f}")
    if summary["candidate_gap_ratio_mean"] is not None:
        print(f"[compare] mean candidate gap pct: {100.0 * summary['candidate_gap_ratio_mean']:.2f}")
    if summary["coverage_gap_ratio_mean"] is not None:
        print(f"[compare] mean promoted gap pct: {100.0 * summary['coverage_gap_ratio_mean']:.2f}")
    if summary["mask_overlap_iou_mean"] is not None:
        print(f"[compare] mean mask IoU: {summary['mask_overlap_iou_mean']:.3f}")
    if summary["candidate_mask_overlap_iou_mean"] is not None:
        print(f"[compare] mean candidate mask IoU: {summary['candidate_mask_overlap_iou_mean']:.3f}")


if __name__ == "__main__":
    main()
