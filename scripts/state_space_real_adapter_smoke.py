"""Lightweight source-free smoke checks for state-space adapters.

No external dataset is downloaded.  The checks exercise schema discovery,
candidate-only semantics, and the fail-closed Gauthey shape gate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from arese_shared_manifold_adapter import discover_columns
from gauthey_functional_trajectory_adapter import load_matrix
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

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "wrong_shape.npy"
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
