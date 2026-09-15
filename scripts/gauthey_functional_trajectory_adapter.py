"""Build a governed diagnostic trajectory from Gauthey whole-brain calcium data.

Scientific source:
W. Gauthey et al., "High-speed whole-brain imaging in Drosophila",
DOI:10.1038/s41467-026-72437-1.
Preprocessed data DOI:10.5281/zenodo.17618684.

DASHI source archaeology pins the pooled conventional-2P functional member as
940 selected-ROI x 668 time samples.  The source preprocessing also pins the
conventional-2P sampling rate and common stimulus-block timebase.  This pays a
protocol-aligned time coordinate for the pooled matrix; it does NOT recover the
source trial/plane/ROI identity of an individual pooled selected row.

The selected ROI rows are not registered MaleCNS neuron identities.  This
adapter makes a declared PCA-3 visual diagnostic over time and emits CANDIDATE
rows only.

Loading pickle is opt-in because pickle can execute arbitrary code.  Use only
with a trusted, checksum-verified source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from neural_observation_bridge import Decision
from state_space_trajectory_adapter import TrajectoryObservation, export_trajectory

PAPER_DOI = "10.1038/s41467-026-72437-1"
DATA_DOI = "10.5281/zenodo.17618684"
EXPECTED_ROWS = 940
EXPECTED_TIME_SAMPLES = 668
CANONICAL_MEMBER = "Data/Dffs/Audio correlated/dffs_audio_2p_corr_top05_all.pkl"

# Exact current public source-code identities inspected for this adapter.
LIGHTBEAD_REPO = "https://github.com/murthylab/lightbead-analysis"
LIGHTBEAD_MAIN_TREE = "55570f4ad028bfd19ab63d5b9b13430803cb277c"
FIG3_ALIGNMENT_BLOB = "f34fe193f0507b5ad7f3c05d6b8973c04a34cf02"
FIG3_PREPROCESSING_BLOB = "39a4ae6739b9e13040b971469616e908df41f502"
STIMULUS_FILE = "Data/Stimulus/highspeed_pulse_2_WG_paper_forplotting.mat"
STIMULUS_FILE_GIT_BLOB = "57a3052ba0a9b3503a04a6c50de5255fadcd4d19"

# Conventional-2P values from analysis_scripts/fig3_preprocessing.py.
TWO_P_HZ = 2.20337115787
FRAME_PERIOD_S = 1.0 / TWO_P_HZ
STIMULUS_BLOCK_START_S = (
    5.0, 25.0, 45.0, 65.0, 84.0, 103.0, 123.0,
    143.0, 163.0, 183.0, 202.99894, 222.99788, 242.99788,
)
STIMULUS_BLOCK_END_S = (
    15.0, 35.0, 55.0, 75.0, 94.0, 113.0, 133.0,
    153.0, 173.0, 192.99894, 212.99788, 232.99788, 252.99788,
)
SOURCE_TRIALS = (
    "GCaMP6f_12132024_a2_r2.pkl",
    "GCaMP6f_12132024_a2_r3.pkl",
    "GCaMP6f_12132024_a2_r4.pkl",
    "GCaMP6f_12202024_a1_r2.pkl",
)


def _array_from_object(obj: object) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float)
    to_numpy = getattr(obj, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=float)
    return np.asarray(obj, dtype=float)


def load_matrix(path: Path, *, trusted_pickle: bool) -> tuple[np.ndarray, str]:
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if len(archive.files) != 1:
            raise ValueError("NPZ must contain exactly one array or be converted explicitly")
        matrix = archive[archive.files[0]]
    elif suffix in {".pkl", ".pickle"}:
        if not trusted_pickle:
            raise ValueError(
                "refusing to load pickle without --trusted-pickle; "
                "verify source identity/checksum first"
            )
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        # The deposited export is a dictionary containing audio_correlated.
        if isinstance(obj, dict) and "audio_correlated" in obj:
            obj = obj["audio_correlated"]
        matrix = _array_from_object(obj)
    else:
        raise ValueError("supported inputs are .npy, single-array .npz, or trusted .pkl/.pickle")

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D selected-ROI x time matrix, got shape {matrix.shape}")
    if matrix.shape != (EXPECTED_ROWS, EXPECTED_TIME_SAMPLES):
        raise ValueError(
            "source shape does not match the repo-pinned Gauthey 2p pooled matrix: "
            f"expected {(EXPECTED_ROWS, EXPECTED_TIME_SAMPLES)}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("functional matrix contains non-finite values")
    return matrix, sha256


def pca3_time_trajectory(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA-3 over timepoints; rows remain features, not neuron identities."""
    samples = matrix.T  # 668 timepoints x 940 selected ROI rows
    centered = samples - samples.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    coordinates = centered @ basis
    return coordinates, singular_values[:3]


def stimulus_block_at(t_seconds: float) -> int | None:
    """Return 1-based source stimulus block index on the protocol timebase."""
    for idx, (start, end) in enumerate(
        zip(STIMULUS_BLOCK_START_S, STIMULUS_BLOCK_END_S), start=1
    ):
        if start <= t_seconds < end:
            return idx
    return None


def build_candidate_rows(
    matrix: np.ndarray,
    *,
    source_sha256: str,
    sample_rate_hz: float = TWO_P_HZ,
) -> tuple[list[TrajectoryObservation], dict]:
    if sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")

    coords, singular_values = pca3_time_trajectory(matrix)
    rows: list[TrajectoryObservation] = []
    for i, xyz in enumerate(coords):
        # Source code builds time_activity as frame_period, 2*frame_period, ...
        # rather than starting at zero.
        t = (i + 1) / sample_rate_hz
        block = stimulus_block_at(t) if np.isclose(sample_rate_hz, TWO_P_HZ) else None
        block_ref = "off" if block is None else str(block)
        provenance = (
            f"paper-doi:{PAPER_DOI};data-doi:{DATA_DOI};"
            f"archive-member:{CANONICAL_MEMBER};source-sha256:{source_sha256};"
            "source-shape:940-selected-roi-x-668-time;"
            "projection:timepoint-PCA3-mean-centered;"
            f"time-basis:aligned-seconds@{sample_rate_hz:g}Hz;"
            f"stimulus-block:{block_ref};"
            "pooled-source-trials:4;pooled-row-trial-plane-roi:unresolved;"
            "roi-identity:unregistered-to-MaleCNS"
        )
        rows.append(
            TrajectoryObservation(
                t=t,
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                channel="gauthey-2p-functional-pca3",
                provenance=provenance,
                decision=Decision.CANDIDATE,
                source_id=f"zenodo:{DATA_DOI}:2p-pooled:time:{i}",
                residual=None,
                receipt_id="diagnostic-projection-not-registration-or-promotion",
            )
        )

    receipt = {
        "schema": "animalexic-gauthey-functional-trajectory-source-v2",
        "paper_doi": PAPER_DOI,
        "data_doi": DATA_DOI,
        "canonical_archive_member": CANONICAL_MEMBER,
        "source_file_sha256": source_sha256,
        "source_shape": [EXPECTED_ROWS, EXPECTED_TIME_SAMPLES],
        "source_axis_semantics": "selected-roi-by-time",
        "trajectory_point_count": len(rows),
        "source_code": {
            "repository": LIGHTBEAD_REPO,
            "main_tree": LIGHTBEAD_MAIN_TREE,
            "fig3_alignment_blob": FIG3_ALIGNMENT_BLOB,
            "fig3_preprocessing_blob": FIG3_PREPROCESSING_BLOB,
            "stimulus_file": STIMULUS_FILE,
            "stimulus_file_git_blob": STIMULUS_FILE_GIT_BLOB,
        },
        "source_trials": list(SOURCE_TRIALS),
        "timebase": {
            "sample_rate_hz": sample_rate_hz,
            "canonical_2p_sample_rate_hz": TWO_P_HZ,
            "time_sample_zero_convention": "first aligned functional sample occurs at one frame period",
            "stimulus_block_start_s": list(STIMULUS_BLOCK_START_S),
            "stimulus_block_end_s": list(STIMULUS_BLOCK_END_S),
            "pooled_matrix_has_common_protocol_aligned_time_axis": True,
            "pooled_selected_row_source_trial_plane_roi_recovered": False,
        },
        "projection": {
            "name": "mean-centered PCA-3 over timepoints",
            "input_dimension": EXPECTED_ROWS,
            "output_dimension": 3,
            "leading_singular_values": [float(v) for v in singular_values],
            "projection_is_downstream_visual_diagnostic": True,
        },
        "promotion_state": "candidate_only",
        "boundary": {
            "protocol_time_alignment_is_row_identity": False,
            "pooled_row_index_is_source_trial_roi_identity": False,
            "selected_roi_row_is_malecns_neuron": False,
            "pca_coordinate_is_anatomical_coordinate": False,
            "visual_recurrence_is_same_neuron_population": False,
            "functional_activity_is_causal_necessity": False,
            "published_dataset_is_animalexic_promotion_receipt": False,
            "later_row_identity_registration_and_promotion_require_separate_receipts": True,
        },
    }
    return rows, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_matrix", type=Path)
    parser.add_argument("--trusted-pickle", action="store_true")
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=TWO_P_HZ,
        help="defaults to the source-pinned conventional-2P rate",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectory-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()

    matrix, source_sha256 = load_matrix(args.input_matrix, trusted_pickle=args.trusted_pickle)
    rows, source_receipt = build_candidate_rows(
        matrix,
        source_sha256=source_sha256,
        sample_rate_hz=args.sample_rate_hz,
    )
    trajectory_receipt = export_trajectory(
        rows,
        args.out,
        args.trajectory_receipt,
        promoted_only=False,
    )
    trajectory_receipt["source_receipt"] = str(args.source_receipt)
    args.trajectory_receipt.write_text(
        json.dumps(trajectory_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.source_receipt.write_text(
        json.dumps(source_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
