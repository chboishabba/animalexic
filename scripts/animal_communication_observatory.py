"""Shared-world wildlife observatory association layer.

Consumes candidate visual animal tracks and passive acoustic localization
candidates that already live in the same candidate world frame. Produces
many-to-many audiovisual association receipts without promoting same-object,
vocal-emitter, individual, or semantic identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VisualTrackCandidate:
    track_id: str
    position_world_m: tuple[float, float, float]
    position_sigma_m: float
    species_hypothesis: str | None
    individual_hypothesis: str | None
    provenance_reference: str
    status: str = "candidate"
    same_object_identity_paid: bool = False
    species_identity_paid: bool = False
    individual_identity_paid: bool = False


@dataclass(frozen=True)
class SharedWorldAVAssociationCandidate:
    acoustic_event_id: str
    visual_track_id: str
    spatial_distance_m: float
    combined_position_sigma_m: float
    normalized_distance: float
    temporal_alignment_reference: str
    visual_provenance_reference: str
    status: str = "candidate"
    spatial_overlap_is_evidence: bool = True
    same_object_identity_paid: bool = False
    vocal_emitter_identity_paid: bool = False
    individual_identity_paid: bool = False
    semantic_meaning_paid: bool = False
    many_to_many_association_retained: bool = True


def _point(value, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite three-coordinate point")
    return point


def associate_visual_acoustic_candidates(
    acoustic_candidate,
    visual_tracks: Iterable[VisualTrackCandidate],
    *,
    max_normalized_distance: float,
    temporal_alignment_reference: str,
) -> list[SharedWorldAVAssociationCandidate]:
    """Return every spatially compatible visual/acoustic candidate relation."""
    if getattr(acoustic_candidate, "status", None) != "candidate":
        raise ValueError("acoustic localization must remain candidate")
    if max_normalized_distance <= 0 or not np.isfinite(max_normalized_distance):
        raise ValueError("max_normalized_distance must be positive and finite")
    if not temporal_alignment_reference:
        raise ValueError("temporal_alignment_reference must be non-empty")

    acoustic_position = _point(acoustic_candidate.position_world_m, "acoustic position")
    acoustic_sigma = float(acoustic_candidate.position_sigma_m)
    if acoustic_sigma < 0 or not np.isfinite(acoustic_sigma):
        raise ValueError("acoustic position_sigma_m must be non-negative and finite")

    out: list[SharedWorldAVAssociationCandidate] = []
    for track in visual_tracks:
        if track.status != "candidate":
            continue
        if not track.track_id:
            raise ValueError("visual track_id must be non-empty")
        visual_position = _point(track.position_world_m, "visual position")
        visual_sigma = float(track.position_sigma_m)
        if visual_sigma < 0 or not np.isfinite(visual_sigma):
            raise ValueError("visual position_sigma_m must be non-negative and finite")

        distance = float(np.linalg.norm(visual_position - acoustic_position))
        combined_sigma = float(np.hypot(acoustic_sigma, visual_sigma))
        if combined_sigma == 0.0:
            normalized = 0.0 if distance == 0.0 else float("inf")
        else:
            normalized = distance / combined_sigma
        if normalized > max_normalized_distance:
            continue

        out.append(
            SharedWorldAVAssociationCandidate(
                acoustic_event_id=str(acoustic_candidate.event_id),
                visual_track_id=track.track_id,
                spatial_distance_m=distance,
                combined_position_sigma_m=combined_sigma,
                normalized_distance=float(normalized),
                temporal_alignment_reference=temporal_alignment_reference,
                visual_provenance_reference=track.provenance_reference,
            )
        )

    out.sort(key=lambda candidate: (candidate.normalized_distance, candidate.visual_track_id))
    return out


def observatory_receipt_payload(
    acoustic_candidate,
    visual_tracks: Iterable[VisualTrackCandidate],
    associations: Iterable[SharedWorldAVAssociationCandidate],
    *,
    scene_id: str,
) -> dict[str, object]:
    """Emit an append-only JSON-serializable candidate receipt packet."""
    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    return {
        "schema": "animalexic-animal-communication-observatory-v1",
        "scene_id": scene_id,
        "decision": "candidate",
        "acoustic_localization": asdict(acoustic_candidate),
        "visual_tracks": [asdict(track) for track in visual_tracks],
        "av_associations": [asdict(association) for association in associations],
        "boundary": {
            "spatial_overlap_is_same_object_proof": False,
            "localization_is_species_identity": False,
            "visual_track_is_vocal_emitter_identity": False,
            "av_association_is_semantic_meaning": False,
            "active_acoustic_probe_used": False,
            "many_to_many_association_retained": True,
        },
    }
