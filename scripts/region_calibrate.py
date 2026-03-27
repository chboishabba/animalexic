#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_xy(rows, label_key="oracle_label_strict"):
    feats = []
    labels = []
    feature_names = [
        "area_px",
        "fill_ratio",
        "roi_overlap_ratio",
        "promoted_overlap_ratio",
        "disp_mean",
        "disp_std",
        "cost_mean",
        "cost_std",
        "conf_mean",
        "conf_std",
        "lr_delta_mean",
        "lr_delta_std",
        "median_delta_mean",
        "median_delta_std",
        "texture_mean",
        "texture_std",
        "disp_gradient_mean",
        "disp_gradient_std",
        "edge_distance_mean",
        "edge_distance_std",
        "border_penalty_mean",
        "border_penalty_std",
    ]
    for r in rows:
        feats.append([float(r.get(name, 0.0)) for name in feature_names])
        labels.append(int(r.get(label_key, 0)))
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.int64), feature_names


def balanced_sample(x, y, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return x, y
    n = min(len(pos), len(neg))
    keep = np.concatenate([rng.choice(pos, size=n, replace=False), rng.choice(neg, size=n, replace=False)])
    rng.shuffle(keep)
    return x[keep], y[keep]


def stump_fit(x, y):
    best = None
    for j in range(x.shape[1]):
        vals = np.unique(np.quantile(x[:, j], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
        for t in vals:
            pred = (x[:, j] >= t).astype(np.int64)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            cand = {"feature": j, "threshold": float(t), "f1": float(f1)}
            if best is None or cand["f1"] > best["f1"]:
                best = cand
    return best


def evaluate_rule(x, y, rule):
    pred = (x[:, rule["feature"]] >= rule["threshold"]).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--label-key", type=str, default="oracle_label_strict")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.input_jsonl)
    x, y, feature_names = build_xy(rows, label_key=args.label_key)
    x, y = balanced_sample(x, y, seed=args.seed)
    rule = stump_fit(x, y)
    metrics = evaluate_rule(x, y, rule)
    model = {
        "model_type": "region_stump_v1",
        "label_key": args.label_key,
        "feature_names": feature_names,
        "rule": {
            "feature_name": feature_names[rule["feature"]],
            "feature_index": int(rule["feature"]),
            "threshold": float(rule["threshold"]),
        },
        "metrics": metrics,
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
    args.output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
