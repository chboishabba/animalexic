#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset(rows, require_candidate=False):
    feats = []
    labels = []
    candidate_flags = []
    for row in rows:
        if require_candidate and not int(row.get('runtime_candidate', 0)):
            continue
        candidate_flag = int(row.get('runtime_candidate', 0))
        feats.append([
            1.0,
            float(row.get('runtime_roi', 0)),
            float(candidate_flag),
            float(row.get('candidate_disp_u8', 0)) / 255.0,
            float(row.get('candidate_cost_u8', 0)) / 255.0,
            float(row.get('candidate_conf_u8', 0)) / 255.0,
            float(row.get('candidate_lr_delta_u8', 255)) / 255.0,
            float(row.get('candidate_median_delta_u8', 255)) / 255.0,
            float(row.get('candidate_texture_u8', 0)) / 255.0,
            float(row.get('candidate_disp_gradient_u8', 255)) / 255.0,
            float(row.get('candidate_edge_distance_u8', 0)) / 255.0,
            float(row.get('candidate_border_penalty_u8', 255)) / 255.0,
        ])
        labels.append(float(row.get('oracle_valid', 0)))
        candidate_flags.append(candidate_flag)
    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    c = np.asarray(candidate_flags, dtype=np.uint8)
    return x, y, c


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(x, y, steps=600, lr=0.3, l2=1e-4):
    w = np.zeros(x.shape[1], dtype=np.float64)
    n = float(len(y))
    pos = max(1.0, float(np.count_nonzero(y == 1.0)))
    neg = max(1.0, float(np.count_nonzero(y == 0.0)))
    sample_weight = np.where(y == 1.0, neg / pos, 1.0)
    sample_weight /= sample_weight.mean()
    for _ in range(steps):
        p = sigmoid(x @ w)
        grad = (x.T @ ((p - y) * sample_weight)) / n
        grad += l2 * w
        w -= lr * grad
    return w


def standardize(x):
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    x_norm = (x - mu) / sigma
    x_norm[:, 0] = 1.0
    return x_norm, mu, sigma


def summarize_thresholds(prob, y, runtime_candidate):
    best = None
    for tau in np.linspace(0.1, 0.9, 17):
        pred = prob >= tau
        tp = int(np.count_nonzero(pred & (y == 1)))
        fp = int(np.count_nonzero(pred & (y == 0)))
        fn = int(np.count_nonzero((~pred) & (y == 1)))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        row = {
            'tau': float(tau),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'predicted_positive_rate': float(np.mean(pred)),
        }
        if best is None or row['f1'] > best['f1']:
            best = row
    candidate_prec = float(np.count_nonzero((runtime_candidate == 1) & (y == 1)) / max(1, np.count_nonzero(runtime_candidate == 1)))
    candidate_rec = float(np.count_nonzero((runtime_candidate == 1) & (y == 1)) / max(1, np.count_nonzero(y == 1)))
    return best, {'candidate_precision': candidate_prec, 'candidate_recall': candidate_rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-jsonl', type=Path, required=True)
    ap.add_argument('--output-json', type=Path, required=True)
    ap.add_argument('--require-candidate', action='store_true', help='Train only on runtime candidate-positive pixels')
    args = ap.parse_args()

    rows = load_rows(args.input_jsonl)
    if not rows:
        raise SystemExit('no teacher rows found')
    x, y, runtime_candidate = build_dataset(rows, require_candidate=args.require_candidate)
    if len(y) == 0:
        raise SystemExit('no rows left after filtering')
    x_norm, mu, sigma = standardize(x)
    w = fit_logreg(x_norm, y)
    prob = sigmoid(x_norm @ w)
    best, baseline = summarize_thresholds(prob, y, runtime_candidate)
    result = {
        'num_rows': int(len(y)),
        'weights': {
            'bias': float(w[0]),
            'runtime_roi': float(w[1]),
            'runtime_candidate': float(w[2]),
            'candidate_disp_u8': float(w[3]),
            'candidate_cost_u8': float(w[4]),
            'candidate_conf_u8': float(w[5]),
            'candidate_lr_delta_u8': float(w[6]),
            'candidate_median_delta_u8': float(w[7]),
            'candidate_texture_u8': float(w[8]),
            'candidate_disp_gradient_u8': float(w[9]),
            'candidate_edge_distance_u8': float(w[10]),
            'candidate_border_penalty_u8': float(w[11]),
        },
        'feature_mean': {
            'runtime_roi': float(mu[1]),
            'runtime_candidate': float(mu[2]),
            'candidate_disp_u8': float(mu[3]),
            'candidate_cost_u8': float(mu[4]),
            'candidate_conf_u8': float(mu[5]),
            'candidate_lr_delta_u8': float(mu[6]),
            'candidate_median_delta_u8': float(mu[7]),
            'candidate_texture_u8': float(mu[8]),
            'candidate_disp_gradient_u8': float(mu[9]),
            'candidate_edge_distance_u8': float(mu[10]),
            'candidate_border_penalty_u8': float(mu[11]),
        },
        'feature_std': {
            'runtime_roi': float(sigma[1]),
            'runtime_candidate': float(sigma[2]),
            'candidate_disp_u8': float(sigma[3]),
            'candidate_cost_u8': float(sigma[4]),
            'candidate_conf_u8': float(sigma[5]),
            'candidate_lr_delta_u8': float(sigma[6]),
            'candidate_median_delta_u8': float(sigma[7]),
            'candidate_texture_u8': float(sigma[8]),
            'candidate_disp_gradient_u8': float(sigma[9]),
            'candidate_edge_distance_u8': float(sigma[10]),
            'candidate_border_penalty_u8': float(sigma[11]),
        },
        'predicted_mean': float(prob.mean()),
        'oracle_positive_rate': float(y.mean()),
        'require_candidate': bool(args.require_candidate),
        'best_threshold': best,
        'candidate_baseline': baseline,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[calib] rows: {len(rows)}")
    print(f"[calib] oracle positive rate: {y.mean():.3f}")
    print(f"[calib] best threshold: {best['tau']:.2f} f1={best['f1']:.3f} precision={best['precision']:.3f} recall={best['recall']:.3f}")
    print(f"[calib] candidate baseline: precision={baseline['candidate_precision']:.3f} recall={baseline['candidate_recall']:.3f}")
    print(f"[calib] wrote: {args.output_json}")


if __name__ == '__main__':
    main()
