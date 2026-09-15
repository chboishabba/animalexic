"""Governed trajectory export for Animalexic visual/state-space consumers.

This adapter turns time-indexed runtime observations into a provenance-preserving
CSV ABI that DASHI's downstream state-space renderer can consume. It does not
promote geometry or semantics itself: candidate/promoted/abstain/reject status is
retained explicitly and canonical exports can be restricted to promoted rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from neural_observation_bridge import Decision


@dataclass(frozen=True)
class TrajectoryObservation:
    t: float
    x: float
    y: float
    z: float
    channel: str
    provenance: str
    decision: Decision = Decision.CANDIDATE
    source_id: str = ""
    residual: float | None = None
    receipt_id: str = ""

    def canonical(self) -> bool:
        return self.decision is Decision.PROMOTED


def _canonical_payload(rows: Sequence[TrajectoryObservation]) -> bytes:
    payload = []
    for row in rows:
        item = asdict(row)
        item["decision"] = row.decision.value
        payload.append(item)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def export_trajectory(
    rows: Iterable[TrajectoryObservation],
    csv_path: Path,
    receipt_path: Path,
    *,
    promoted_only: bool = True,
) -> dict:
    materialized = list(rows)
    exported = [r for r in materialized if (r.canonical() or not promoted_only)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "t", "x", "y", "z", "channel", "provenance",
        "decision", "source_id", "residual", "receipt_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in exported:
            writer.writerow({
                "t": row.t,
                "x": row.x,
                "y": row.y,
                "z": row.z,
                "channel": row.channel,
                "provenance": row.provenance,
                "decision": row.decision.value,
                "source_id": row.source_id,
                "residual": "" if row.residual is None else row.residual,
                "receipt_id": row.receipt_id,
            })

    digest = hashlib.sha256(_canonical_payload(exported)).hexdigest()
    receipt = {
        "schema": "animalexic-state-space-trajectory-v1",
        "input_count": len(materialized),
        "export_count": len(exported),
        "promoted_only": promoted_only,
        "channels": sorted({r.channel for r in exported}),
        "decisions_present": sorted({r.decision.value for r in materialized}),
        "canonical_payload_sha256": digest,
        "csv_path": str(csv_path),
        "boundary": {
            "candidate_geometry_is_canonical": False,
            "visual_proximity_implies_physical_or_anatomical_proximity": False,
            "recurrence_implies_semantic_meaning": False,
            "functional_trace_identity_implies_connectome_neuron_identity": False,
            "promotion_requires_upstream_governance": True,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def synthetic_fixture() -> list[TrajectoryObservation]:
    """Tiny deterministic governance fixture; not measured biological data."""
    return [
        TrajectoryObservation(0.0, 0.0, 0.0, 0.0, "neural", "synthetic", Decision.CANDIDATE, "demo:0"),
        TrajectoryObservation(0.1, 0.1, 0.2, 0.0, "neural", "synthetic", Decision.PROMOTED, "demo:1", 0.02, "receipt:1"),
        TrajectoryObservation(0.2, 0.2, 0.1, 0.1, "motor", "synthetic", Decision.ABSTAIN, "demo:2", 0.4, "receipt:2"),
        TrajectoryObservation(0.3, 0.3, 0.2, 0.2, "effector", "synthetic", Decision.PROMOTED, "demo:3", 0.03, "receipt:3"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true", help="export deterministic synthetic fixture")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--include-nonpromoted", action="store_true")
    args = parser.parse_args()

    if not args.fixture:
        raise SystemExit("Current CLI intentionally supports only --fixture; import export_trajectory from runtime producers.")

    export_trajectory(
        synthetic_fixture(),
        args.out,
        args.receipt,
        promoted_only=not args.include_nonpromoted,
    )


if __name__ == "__main__":
    main()
