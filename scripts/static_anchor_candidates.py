from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from scripts.static_anchor_association import StaticAnchorObservation


@dataclass(frozen=True)
class StaticFeatureObservation:
    camera_id: int
    time_s: float
    feature_id: str
    descriptor: tuple[float, ...]
    static_confidence: float
    provenance: str
    point_local_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class AnchorIdentityCandidate:
    source_camera_id: int
    target_camera_id: int
    source_feature_id: str
    target_feature_id: str
    descriptor_distance: float
    descriptor_margin: float
    mutual_nearest: bool
    ambiguous: bool
    source_provenance: str
    target_provenance: str
    status: str
    same_object_paid: bool = False


@dataclass(frozen=True)
class CandidateAnchorTrack:
    track_id: str
    camera_id: int
    members: tuple[StaticFeatureObservation, ...]
    provenance_chain: tuple[str, ...]
    status: str = "candidate"
    same_object_paid: bool = False


@dataclass(frozen=True)
class SameObjectAnchorReceipt:
    anchor_id: str
    source_feature_id: str
    target_feature_id: str
    source_provenance: str
    target_provenance: str
    receipt_ref: str


def _descriptor(observation: StaticFeatureObservation) -> np.ndarray:
    value = np.asarray(observation.descriptor, dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("descriptor must be a finite non-empty vector")
    if not 0.0 <= float(observation.static_confidence) <= 1.0:
        raise ValueError("static_confidence must be in [0,1]")
    if not math.isfinite(float(observation.time_s)):
        raise ValueError("time_s must be finite")
    return value


def _distance(a: StaticFeatureObservation, b: StaticFeatureObservation) -> float:
    da = _descriptor(a)
    db = _descriptor(b)
    if da.shape != db.shape:
        raise ValueError("descriptor dimensions must match")
    return float(np.linalg.norm(da - db))


def propose_cross_camera_candidates(
    source_observations,
    target_observations,
    *,
    max_descriptor_distance: float,
    min_margin: float,
    min_static_confidence: float = 0.9,
) -> list[AnchorIdentityCandidate]:
    """Propose cross-camera anchor identity without paying same-object identity."""
    if max_descriptor_distance < 0 or min_margin < 0:
        raise ValueError("descriptor thresholds must be non-negative")
    source = [
        value for value in source_observations
        if float(value.static_confidence) >= min_static_confidence
    ]
    target = [
        value for value in target_observations
        if float(value.static_confidence) >= min_static_confidence
    ]
    if not source or not target:
        return []

    distances = np.empty((len(source), len(target)), dtype=np.float64)
    for i, src in enumerate(source):
        for j, dst in enumerate(target):
            if src.camera_id == dst.camera_id:
                raise ValueError("cross-camera candidates require distinct cameras")
            distances[i, j] = _distance(src, dst)

    source_best = np.argmin(distances, axis=1)
    target_best = np.argmin(distances, axis=0)
    out = []
    for i, j in enumerate(source_best):
        best = float(distances[i, j])
        if best > max_descriptor_distance:
            continue
        ordered = np.sort(distances[i])
        second = float(ordered[1]) if len(ordered) > 1 else math.inf
        margin = second - best
        mutual = int(target_best[j]) == i
        ambiguous = margin < min_margin
        if not mutual and not ambiguous:
            continue
        src = source[i]
        dst = target[int(j)]
        out.append(
            AnchorIdentityCandidate(
                source_camera_id=int(src.camera_id),
                target_camera_id=int(dst.camera_id),
                source_feature_id=src.feature_id,
                target_feature_id=dst.feature_id,
                descriptor_distance=best,
                descriptor_margin=margin,
                mutual_nearest=mutual,
                ambiguous=ambiguous,
                source_provenance=src.provenance,
                target_provenance=dst.provenance,
                status="abstain" if ambiguous or not mutual else "candidate",
                same_object_paid=False,
            )
        )
    return out


def build_temporal_candidate_tracks(
    observations,
    *,
    max_time_delta_s: float,
    max_descriptor_distance: float,
    min_margin: float,
    min_static_confidence: float = 0.9,
) -> list[CandidateAnchorTrack]:
    if max_time_delta_s < 0 or max_descriptor_distance < 0 or min_margin < 0:
        raise ValueError("track thresholds must be non-negative")
    eligible = [
        value for value in observations
        if float(value.static_confidence) >= min_static_confidence
    ]
    for value in eligible:
        _descriptor(value)

    by_camera: dict[int, list[StaticFeatureObservation]] = {}
    for value in eligible:
        by_camera.setdefault(int(value.camera_id), []).append(value)

    tracks = []
    next_track = 0
    for camera_id in sorted(by_camera):
        remaining = sorted(
            by_camera[camera_id],
            key=lambda value: (float(value.time_s), value.feature_id),
        )
        while remaining:
            current = remaining.pop(0)
            members = [current]
            while remaining:
                last = members[-1]
                admissible = [
                    value for value in remaining
                    if 0.0 < float(value.time_s) - float(last.time_s) <= max_time_delta_s
                ]
                if not admissible:
                    break
                distances = sorted(
                    ((_distance(last, value), value) for value in admissible),
                    key=lambda item: (item[0], item[1].feature_id),
                )
                best_distance, best = distances[0]
                second_distance = distances[1][0] if len(distances) > 1 else math.inf
                margin = float(second_distance - best_distance)
                if best_distance > max_descriptor_distance or margin < min_margin:
                    break
                members.append(best)
                remaining.remove(best)
            tracks.append(
                CandidateAnchorTrack(
                    track_id=f"camera{camera_id}-candidate-{next_track}",
                    camera_id=camera_id,
                    members=tuple(members),
                    provenance_chain=tuple(value.provenance for value in members),
                    status="candidate",
                    same_object_paid=False,
                )
            )
            next_track += 1
    return tracks


def _paid_point(observation: StaticFeatureObservation) -> tuple[float, float, float]:
    if observation.point_local_m is None:
        raise ValueError("same-object payment requires a local 3D anchor point")
    point = np.asarray(observation.point_local_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("anchor point must be finite xyz")
    return tuple(float(x) for x in point)


def materialise_paid_anchor_observations(
    candidate: AnchorIdentityCandidate,
    receipt: SameObjectAnchorReceipt,
    source_observation: StaticFeatureObservation,
    target_observation: StaticFeatureObservation,
) -> tuple[StaticAnchorObservation, StaticAnchorObservation]:
    """Cross the identity seam only with an exact external same-object receipt."""
    if candidate.status != "candidate" or candidate.ambiguous:
        raise ValueError("only unambiguous identity candidates may receive payment")
    if not receipt.anchor_id or not receipt.receipt_ref:
        raise ValueError("same-object receipt requires anchor_id and receipt_ref")
    expected = (
        candidate.source_feature_id,
        candidate.target_feature_id,
        candidate.source_provenance,
        candidate.target_provenance,
    )
    paid = (
        receipt.source_feature_id,
        receipt.target_feature_id,
        receipt.source_provenance,
        receipt.target_provenance,
    )
    actual = (
        source_observation.feature_id,
        target_observation.feature_id,
        source_observation.provenance,
        target_observation.provenance,
    )
    if paid != expected or actual != expected:
        raise ValueError("same-object receipt does not match candidate feature/provenance identity")
    if int(source_observation.camera_id) != candidate.source_camera_id:
        raise ValueError("source camera does not match candidate")
    if int(target_observation.camera_id) != candidate.target_camera_id:
        raise ValueError("target camera does not match candidate")

    source_anchor = StaticAnchorObservation(
        anchor_id=receipt.anchor_id,
        camera_id=str(source_observation.camera_id),
        time_s=float(source_observation.time_s),
        point_local_m=_paid_point(source_observation),
        is_static=True,
        confidence=float(source_observation.static_confidence),
        provenance_ref=f"{source_observation.provenance}|same-object:{receipt.receipt_ref}",
        status="candidate",
    )
    target_anchor = StaticAnchorObservation(
        anchor_id=receipt.anchor_id,
        camera_id=str(target_observation.camera_id),
        time_s=float(target_observation.time_s),
        point_local_m=_paid_point(target_observation),
        is_static=True,
        confidence=float(target_observation.static_confidence),
        provenance_ref=f"{target_observation.provenance}|same-object:{receipt.receipt_ref}",
        status="candidate",
    )
    return source_anchor, target_anchor
