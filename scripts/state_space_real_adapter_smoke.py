"""Lightweight source-free smoke checks for state-space adapters.

No external dataset is downloaded.  The checks exercise schema discovery,
candidate-only semantics, the fail-closed Gauthey shape gate, and the
source-pinned conventional-2P stimulus/timebase receipt.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from arese_shared_manifold_adapter import discover_columns, read_arese_csv
from gauthey_functional_trajectory_adapter import (
    EXPECTED_ROWS,
    EXPECTED_TIME_SAMPLES,
    FIG3_ALIGNMENT_BLOB,
    FIG3_PREPROCESSING_BLOB,
    STIMULUS_BLOCK_START_S,
    STIMULUS_FILE_GIT_BLOB,
    TWO_P_HZ,
    load_matrix,
    stimulus_block_at,
)
from neural_observation_bridge import Decision
from state_space_trajectory_adapter import synthetic_fixture


def main() -> None:
    columns = discover_columns(["time_s", "UMAP1", "UMAP2", "UMAP3", "recording_id"])
    assert columns.x == "UMAP1"
    assert columns.y == "UMAP2"
    assert columns.z == "UMAP3"
    assert columns.time == "time_s"
    assert columns.individual == "recording_id"

    fixture = synthetic_fixture()
    assert any(row.decision is Decision.CANDIDATE for row in fixture)
    assert any(row.decision is Decision.PROMOTED for row in fixture)

    # Source-bound alignment facts, kept separate from pooled row identity.
    assert EXPECTED_ROWS == 940
    assert EXPECTED_TIME_SAMPLES == 668
    assert TWO_P_HZ == 2.20337115787
    assert len(STIMULUS_BLOCK_START_S) == 13
    assert stimulus_block_at(4.9) is None
    assert stimulus_block_at(5.0) == 1
    assert stimulus_block_at(25.0) == 2
    assert FIG3_ALIGNMENT_BLOB == "f34fe193f0507b5ad7f3c05d6b8973c04a34cf02"
    assert FIG3_PREPROCESSING_BLOB == "39a4ae6739b9e13040b971469616e908df41f502"
    assert STIMULUS_FILE_GIT_BLOB == "57a3052ba0a9b3503a04a6c50de5255fadcd4d19"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        arese = tmp_path / "arese_fixture.csv"
        with arese.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["time_s", "UMAP1", "UMAP2", "UMAP3", "recording_id"],
            )
            writer.writeheader()
            writer.writerow(
                {"time_s": "0.25", "UMAP1": "1", "UMAP2": "2", "UMAP3": "3", "recording_id": "WR_01"}
            )
        imported, source_receipt = read_arese_csv(
            arese,
            species="eurasian-wren",
            feature_space="mfcc",
        )
        assert len(imported) == 1
        assert imported[0].decision is Decision.CANDIDATE
        assert imported[0].t == 0.25
        assert source_receipt["promotion_state"] == "candidate_only"
        assert source_receipt["boundary"]["published_external_row_is_promoted_animalexic_state"] is False

        bad = tmp_path / "wrong_shape.npy"
        np.save(bad, np.zeros((3, 4), dtype=float), allow_pickle=False)
        try:
            load_matrix(bad, trusted_pickle=False)
        except ValueError as exc:
            assert "expected (940, 668)" in str(exc)
        else:
            raise AssertionError("Gauthey adapter accepted a wrong-shape matrix")

    print("state-space real-adapter smoke: PASS")


if __name__ == "__main__":
    main()
