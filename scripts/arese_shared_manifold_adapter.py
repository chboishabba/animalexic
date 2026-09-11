"""Import Lucio Arese's published shared-acoustic-manifold CSV outputs.

Source coordinates:
- Lucio Arese, "Shared acoustic manifolds for exploratory comparison of
  passerine vocalizations", DOI:10.32942/X2W65N.
- Processed outputs: Zenodo DOI:10.5281/zenodo.18332166.

The published CSV is external evidence.  This adapter therefore emits
Animalexic CANDIDATE trajectory rows only.  Publication/data custody is not an
Animalexic promotion receipt.  A later runtime/governance stage may promote an
observation after its own invariants, residuals, and receipt checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from neural_observation_bridge import Decision
from state_space_trajectory_adapter import TrajectoryObservation, export_trajectory

PAPER_DOI = "10.32942/X2W65N"
DATA_DOI = "10.5281/zenodo.18332166"
PIPELINE_VERSION = "arese-shared-acoustic-manifolds-v4"

# Current paper v4 method receipt.  These are method/source facts, not locally
# recomputed guarantees about a particular CSV payload.
METHOD_RECEIPT = {
    "mfcc": {
        "input_dimensions": 120,
        "construction": "40 MFCC + delta + delta-delta",
        "pca_dimensions": 20,
        "umap_dimensions": 3,
        "umap_metric": "euclidean",
    },
    "chroma": {
        "input_dimensions": 80,
        "construction": "80-bin EDO chroma, L1-normalized",
        "pca_dimensions": 20,
        "umap_dimensions": 3,
        "umap_metric": "cosine",
    },
    "umap_neighbors": 30,
    "umap_min_dist": 0.1,
    "seed": 42,
    "coordinate_normalization": "shared global unit-cube",
    "overlays_not_embedding_inputs": ["RMS", "spectral_centroid", "CEC"],
}


@dataclass(frozen=True)
class ColumnMap:
    x: str
    y: str
    z: str
    time: str | None
    individual: str | None
    frame: str | None


def _first(fieldnames: Sequence[str], aliases: Sequence[str]) -> str | None:
    lower = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def discover_columns(fieldnames: Sequence[str]) -> ColumnMap:
    """Fail-closed schema discovery with common embedding-column aliases."""
    x = _first(fieldnames, ("x", "umap_x", "umap1", "umap_1", "embedding_x", "dim1"))
    y = _first(fieldnames, ("y", "umap_y", "umap2", "umap_2", "embedding_y", "dim2"))
    z = _first(fieldnames, ("z", "umap_z", "umap3", "umap_3", "embedding_z", "dim3"))
    if not (x and y and z):
        raise ValueError(
            "could not identify three embedding coordinate columns; "
            f"available columns={list(fieldnames)!r}"
        )
    return ColumnMap(
        x=x,
        y=y,
        z=z,
        time=_first(fieldnames, ("t", "time", "time_s", "timestamp", "frame_time")),
        individual=_first(fieldnames, ("individual", "individual_id", "recording_id", "id")),
        frame=_first(fieldnames, ("frame", "frame_index", "frame_idx", "index")),
    )


def _stable_row_id(path: Path, row_index: int, row: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"file": path.name, "row_index": row_index, "row": dict(row)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_arese_csv(
    path: Path,
    *,
    species: str,
    feature_space: str,
) -> tuple[list[TrajectoryObservation], dict]:
    if feature_space not in {"mfcc", "chroma"}:
        raise ValueError("feature_space must be 'mfcc' or 'chroma'")

    raw_bytes = path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    rows: list[TrajectoryObservation] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns = discover_columns(fieldnames)
        for i, row in enumerate(reader):
            # Preserve paper time when present; otherwise frame ordinal is an
            # explicit fallback coordinate rather than an invented sample time.
            if columns.time and row.get(columns.time, "").strip():
                t = float(row[columns.time])
                time_basis = f"source-column:{columns.time}"
            elif columns.frame and row.get(columns.frame, "").strip():
                t = float(row[columns.frame])
                time_basis = f"frame-ordinal-column:{columns.frame}"
            else:
                t = float(i)
                time_basis = "csv-row-ordinal"

            individual = row.get(columns.individual, "") if columns.individual else ""
            source_id = f"zenodo:{DATA_DOI}:{path.name}:row:{i}"
            provenance = (
                f"paper-doi:{PAPER_DOI};data-doi:{DATA_DOI};"
                f"pipeline:{PIPELINE_VERSION};species:{species};"
                f"feature:{feature_space};individual:{individual};"
                f"time-basis:{time_basis};file-sha256:{file_sha256};"
                f"row-sha256:{_stable_row_id(path, i, row)}"
            )
            rows.append(
                TrajectoryObservation(
                    t=t,
                    x=float(row[columns.x]),
                    y=float(row[columns.y]),
                    z=float(row[columns.z]),
                    channel=f"birdsong-{species}-{feature_space}",
                    provenance=provenance,
                    decision=Decision.CANDIDATE,
                    source_id=source_id,
                    residual=None,
                    receipt_id="external-published-data-not-animalexic-promotion",
                )
            )

    source_receipt = {
        "schema": "animalexic-arese-shared-manifold-source-v1",
        "paper_doi": PAPER_DOI,
        "data_doi": DATA_DOI,
        "pipeline_version": PIPELINE_VERSION,
        "species": species,
        "feature_space": feature_space,
        "source_filename": path.name,
        "source_file_sha256": file_sha256,
        "row_count": len(rows),
        "method": METHOD_RECEIPT,
        "promotion_state": "candidate_only",
        "boundary": {
            "published_external_row_is_promoted_animalexic_state": False,
            "umap_coordinate_is_physical_coordinate": False,
            "embedding_neighborhood_is_taxonomic_identity": False,
            "descriptor_overlay_changes_embedding_geometry": False,
            "later_promotion_requires_separate_governance_receipt": True,
        },
    }
    return rows, source_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--species", required=True, help="source species/group label")
    parser.add_argument("--feature-space", choices=("mfcc", "chroma"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectory-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()

    rows, source_receipt = read_arese_csv(
        args.input_csv,
        species=args.species,
        feature_space=args.feature_space,
    )

    # Candidate diagnostic export is intentional.  The generic adapter's
    # promoted-only default is bypassed only so downstream inspection can see
    # externally published candidate geometry without calling it canonical.
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
