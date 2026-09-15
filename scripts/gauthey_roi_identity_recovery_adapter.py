"""Recover pooled Gauthey conventional-2P selected-row source identity.

This reproduces the selection semantics in murthylab/lightbead-analysis without
materializing the full 188000 x 668 stacked matrix.  Four source trial
pickles are processed independently, correlations are concatenated, and the
global top 0.5% (940 rows) is mapped back to trial / plane / cluster identity.

An optional compact 940 x 668 carrier can be supplied.  If the recovered rows
match it exactly, that supplies a strong same-object weld between the compact
published matrix and the recovered source-row identities.  Without that exact
comparison, the output is a reconstruction candidate only.

Pickle loading is opt-in because pickle is executable.  Use only exact,
checksum-verified source artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gauthey_functional_trajectory_adapter import (
    DATA_DOI,
    EXPECTED_ROWS,
    EXPECTED_TIME_SAMPLES,
    FIG3_PREPROCESSING_BLOB,
    PAPER_DOI,
    SOURCE_TRIALS,
    STIMULUS_BLOCK_END_S,
    STIMULUS_BLOCK_START_S,
    TWO_P_HZ,
    load_matrix,
)

EXPECTED_PLANES = 47
EXPECTED_CLUSTERS_PER_PLANE = 1000
EXPECTED_ROWS_PER_TRIAL = EXPECTED_PLANES * EXPECTED_CLUSTERS_PER_PLANE
EXPECTED_TOTAL_SOURCE_ROWS = EXPECTED_ROWS_PER_TRIAL * len(SOURCE_TRIALS)
SELECTION_PERCENT = 0.5
TAU_RISE_S = 0.050
TAU_DECAY_S = 0.140
KERNEL_DURATION_S = 1.0


@dataclass(frozen=True)
class TrialCarrier:
    canonical_name: str
    path: Path
    sha256: str
    matrix: np.ndarray


@dataclass(frozen=True)
class SelectedIdentity:
    pooled_selected_row: int
    global_row: int
    trial_index: int
    trial_name: str
    local_row: int
    plane_index: int
    cluster_index: int
    correlation: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_trusted_trial(path: Path, canonical_name: str) -> TrialCarrier:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict) or "dffs_corrected" not in obj:
        raise ValueError(f"{canonical_name}: expected dict with dffs_corrected")
    matrix = np.asarray(obj["dffs_corrected"], dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{canonical_name}: expected 2D dffs_corrected, got {matrix.shape}")
    if matrix.shape[0] != EXPECTED_ROWS_PER_TRIAL:
        raise ValueError(
            f"{canonical_name}: expected {EXPECTED_ROWS_PER_TRIAL} rows "
            f"(47 planes x 1000 clusters), got {matrix.shape[0]}"
        )
    if matrix.shape[1] < EXPECTED_TIME_SAMPLES:
        raise ValueError(
            f"{canonical_name}: expected at least {EXPECTED_TIME_SAMPLES} aligned samples, "
            f"got {matrix.shape[1]}"
        )
    matrix = matrix[:, :EXPECTED_TIME_SAMPLES]
    if not np.isfinite(matrix).all():
        raise ValueError(f"{canonical_name}: non-finite values in aligned source matrix")
    return TrialCarrier(canonical_name, path, _sha256(path), matrix)


def source_stimulus(samples: int = EXPECTED_TIME_SAMPLES) -> np.ndarray:
    """Reproduce functions.create_stim + GCaMP6f convolution for scope=2p."""
    idx = np.arange(samples, dtype=float)
    stimulus = np.zeros(samples, dtype=float)
    starts = np.asarray(STIMULUS_BLOCK_START_S) * TWO_P_HZ
    ends = np.asarray(STIMULUS_BLOCK_END_S) * TWO_P_HZ
    for start, end in zip(starts, ends):
        stimulus[(idx > start) & (idx < end)] = 1.0

    dt = 1.0 / TWO_P_HZ
    kernel_t = np.arange(0.0, KERNEL_DURATION_S, dt)
    kernel = (1.0 - np.exp(-kernel_t / TAU_RISE_S)) * np.exp(-kernel_t / TAU_DECAY_S)
    kernel /= np.max(kernel)
    return np.convolve(stimulus, kernel, mode="full")[:samples]


def row_correlations(matrix: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (matrix - means) / stds
    stim_norm = (stimulus - stimulus.mean()) / stimulus.std()
    return (normalized @ stim_norm) / normalized.shape[1]


def recover_identities(trials: Sequence[TrialCarrier]) -> tuple[list[SelectedIdentity], np.ndarray]:
    if len(trials) != len(SOURCE_TRIALS):
        raise ValueError(f"expected {len(SOURCE_TRIALS)} canonical source trials")

    stimulus = source_stimulus()
    corr_parts = [row_correlations(t.matrix, stimulus) for t in trials]
    correlations = np.concatenate(corr_parts)
    if correlations.shape != (EXPECTED_TOTAL_SOURCE_ROWS,):
        raise AssertionError("unexpected concatenated source-row count")

    valid_global = np.flatnonzero(~np.isnan(correlations))
    valid_corr = correlations[valid_global]
    index_cutoff = int(EXPECTED_TOTAL_SOURCE_ROWS * (SELECTION_PERCENT / 100.0))
    if index_cutoff != EXPECTED_ROWS:
        raise AssertionError(f"expected top-selection count {EXPECTED_ROWS}, got {index_cutoff}")

    order = np.argsort(valid_corr)
    selected_global = valid_global[order[-index_cutoff:]]
    selected_corr = valid_corr[order[-index_cutoff:]]

    identities: list[SelectedIdentity] = []
    selected_rows: list[np.ndarray] = []
    for pooled_idx, (global_row, corr) in enumerate(zip(selected_global, selected_corr)):
        trial_index = int(global_row // EXPECTED_ROWS_PER_TRIAL)
        local_row = int(global_row % EXPECTED_ROWS_PER_TRIAL)
        plane_index = local_row // EXPECTED_CLUSTERS_PER_PLANE
        cluster_index = local_row % EXPECTED_CLUSTERS_PER_PLANE
        identities.append(
            SelectedIdentity(
                pooled_selected_row=pooled_idx,
                global_row=int(global_row),
                trial_index=trial_index,
                trial_name=SOURCE_TRIALS[trial_index],
                local_row=local_row,
                plane_index=plane_index,
                cluster_index=cluster_index,
                correlation=float(corr),
            )
        )
        selected_rows.append(trials[trial_index].matrix[local_row])

    return identities, np.vstack(selected_rows)


def write_identity_csv(path: Path, identities: Sequence[SelectedIdentity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pooled_selected_row",
                "global_row",
                "trial_index",
                "trial_name",
                "local_row",
                "plane_index",
                "cluster_index",
                "correlation",
            ],
        )
        writer.writeheader()
        for row in identities:
            writer.writerow(row.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trials",
        nargs=4,
        type=Path,
        help="four trusted aligned conventional-2P trial pickles in canonical SOURCE_TRIALS order",
    )
    parser.add_argument("--trusted-pickle", action="store_true")
    parser.add_argument(
        "--compact-matrix",
        type=Path,
        help="optional compact 940x668 published carrier for exact same-object comparison",
    )
    parser.add_argument(
        "--trusted-compact-pickle",
        action="store_true",
        help="required if --compact-matrix is a pickle",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if not args.trusted_pickle:
        raise SystemExit("refusing source-trial pickle loading without --trusted-pickle")

    trials = [
        _load_trusted_trial(path, canonical)
        for path, canonical in zip(args.trials, SOURCE_TRIALS)
    ]
    identities, recovered_compact = recover_identities(trials)

    compact_exact_match: bool | None = None
    compact_sha256: str | None = None
    if args.compact_matrix is not None:
        compact, compact_sha256 = load_matrix(
            args.compact_matrix,
            trusted_pickle=args.trusted_compact_pickle,
        )
        compact_exact_match = bool(np.array_equal(recovered_compact, compact))
        if not compact_exact_match:
            raise ValueError(
                "recovered selected rows do not exactly equal supplied compact carrier; "
                "same-object pooled-row identity is not paid"
            )

    write_identity_csv(args.out, identities)
    receipt = {
        "schema": "animalexic-gauthey-pooled-roi-identity-recovery-v1",
        "paper_doi": PAPER_DOI,
        "data_doi": DATA_DOI,
        "source_preprocessing_blob": FIG3_PREPROCESSING_BLOB,
        "source_trial_order": list(SOURCE_TRIALS),
        "source_trial_sha256": [t.sha256 for t in trials],
        "source_trial_shape_after_truncation": [EXPECTED_ROWS_PER_TRIAL, EXPECTED_TIME_SAMPLES],
        "source_row_order_semantics": "47 planes x 1000 clusters, plane-major as constructed by Fig3_aligment.py",
        "selection": {
            "method": "source-equivalent zero-lag z-normalized correlation with convolved 13-block stimulus",
            "selection_percent": SELECTION_PERCENT,
            "selected_rows": len(identities),
            "expected_selected_rows": EXPECTED_ROWS,
        },
        "compact_same_object_check": {
            "supplied": args.compact_matrix is not None,
            "sha256": compact_sha256,
            "exact_array_equal": compact_exact_match,
        },
        "identity_status": (
            "exact-same-object-pooled-row-to-trial-plane-cluster"
            if compact_exact_match
            else "reconstructed-candidate-pending-exact-compact-weld"
        ),
        "boundary": {
            "trial_plane_cluster_identity_is_malecns_neuron_identity": False,
            "source_roi_identity_is_anatomical_registration": False,
            "correlation_selection_is_causal_necessity": False,
            "reconstruction_without_compact_exact_match_is_exact_pooled_identity": False,
            "exact_compact_match_creates_animalexic_promotion": False,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
