"""Governed static-anchor association and robust cross-camera weld fitting.

This module sits between landmark/track producers and ``cross_camera_world_weld``.
It only associates observations whose upstream anchor identity already agrees;
it does not infer semantic/same-object identity from descriptor similarity.  The
robust fitter uses deterministic minimal-set hypotheses to reject geometric
outliers while preserving candidate-only world-weld semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from scripts.cross_camera_world_weld import WorldWeldCandidate, estimate_world_weld


@dataclass(frozen=True)
class StaticAnchorObservation:
    anchor_id: str
    camera_id: str
    time_s: float
    point_local_m: tuple[float, float, float]
    is_static: bool
    confidence: float
    provenance_ref: str
    status: str = "candidate"


@dataclass(frozen=True)
class StaticAnchorPair:
    anchor_id: str
    source_camera_id: str
    target_camera_id: str
    source_time_s: float
    target_time_s: float
    source_point_m: tuple[float, float, float]
    target_point_m: tuple[float, float, float]
    source_provenance_ref: str
    target_provenance_ref: str
    confidence: float
    time_delta_s: float
    status: str = "candidate"


@dataclass(frozen=True)
class RobustWorldWeldReceipt:
    weld: WorldWeldCandidate
    pair_count: int
    inlier_count: int
    outlier_count: int
    inlier_anchor_ids: tuple[str, ...]
    outlier_anchor_ids: tuple[str, ...]
    same_object_identity_required: bool
    robust_outlier_rejection: bool
    similarity_scale_exposed: bool
    status: str = "candidate"


def _validate_observation(obs: StaticAnchorObservation) -> None:
    if not obs.anchor_id:
        raise ValueError("anchor_id must be non-empty")
    if not obs.camera_id:
        raise ValueError("camera_id must be non-empty")
    if not obs.provenance_ref:
        raise ValueError("provenance_ref must be non-empty")
    if obs.status != "candidate":
        raise ValueError("anchor observations must remain candidate")
    if not np.isfinite(obs.time_s):
        raise ValueError("anchor time must be finite")
    point = np.asarray(obs.point_local_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("anchor point must be a finite 3-vector")
    if not np.isfinite(obs.confidence) or not (0.0 <= obs.confidence <= 1.0):
        raise ValueError("anchor confidence must lie in [0, 1]")


def associate_static_anchors(
    source_observations: Iterable[StaticAnchorObservation],
    target_observations: Iterable[StaticAnchorObservation],
    *,
    max_time_delta_s: float = 0.25,
    min_confidence: float = 0.0,
) -> list[StaticAnchorPair]:
    """Associate already-identified static anchors across two camera maps.

    Exact ``anchor_id`` equality is a required upstream same-object receipt.  If
    an identity appears multiple times, the closest admissible timestamp pair is
    selected deterministically.  Dynamic, low-confidence, stale, or promoted
    observations are not silently admitted.
    """
    if max_time_delta_s < 0 or not np.isfinite(max_time_delta_s):
        raise ValueError("max_time_delta_s must be finite and non-negative")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must lie in [0, 1]")

    source = list(source_observations)
    target = list(target_observations)
    for obs in source + target:
        _validate_observation(obs)

    source_by_id: dict[str, list[StaticAnchorObservation]] = {}
    target_by_id: dict[str, list[StaticAnchorObservation]] = {}
    for obs in source:
        if obs.is_static and obs.confidence >= min_confidence:
            source_by_id.setdefault(obs.anchor_id, []).append(obs)
    for obs in target:
        if obs.is_static and obs.confidence >= min_confidence:
            target_by_id.setdefault(obs.anchor_id, []).append(obs)

    pairs: list[StaticAnchorPair] = []
    for anchor_id in sorted(set(source_by_id) & set(target_by_id)):
        candidates = []
        for a in source_by_id[anchor_id]:
            for b in target_by_id[anchor_id]:
                dt = abs(float(a.time_s) - float(b.time_s))
                if dt <= max_time_delta_s:
                    combined_confidence = min(float(a.confidence), float(b.confidence))
                    candidates.append((dt, -combined_confidence, a.provenance_ref, b.provenance_ref, a, b))
        if not candidates:
            continue
        dt, neg_conf, _, _, a, b = min(candidates, key=lambda item: item[:4])
        pairs.append(
            StaticAnchorPair(
                anchor_id=anchor_id,
                source_camera_id=a.camera_id,
                target_camera_id=b.camera_id,
                source_time_s=float(a.time_s),
                target_time_s=float(b.time_s),
                source_point_m=tuple(float(x) for x in a.point_local_m),
                target_point_m=tuple(float(x) for x in b.point_local_m),
                source_provenance_ref=a.provenance_ref,
                target_provenance_ref=b.provenance_ref,
                confidence=float(-neg_conf),
                time_delta_s=float(dt),
                status="candidate",
            )
        )
    return pairs


def _pair_arrays(pairs: list[StaticAnchorPair]):
    src = np.asarray([p.source_point_m for p in pairs], dtype=np.float64)
    dst = np.asarray([p.target_point_m for p in pairs], dtype=np.float64)
    return src, dst


def _residuals(weld: WorldWeldCandidate, pairs: list[StaticAnchorPair]) -> np.ndarray:
    src, dst = _pair_arrays(pairs)
    R = np.asarray(weld.rotation_target_from_source, dtype=np.float64).reshape(3, 3)
    t = np.asarray(weld.translation_target_from_source_m, dtype=np.float64).reshape(3)
    predicted = (float(weld.scale) * (R @ src.T)).T + t
    return np.linalg.norm(predicted - dst, axis=1)


def robust_estimate_world_weld(
    pairs: Iterable[StaticAnchorPair],
    *,
    max_residual_m: float,
    min_inliers: int = 3,
    allow_scale: bool = False,
    max_hypotheses: int = 4096,
) -> RobustWorldWeldReceipt:
    """Fit an SE(3)/Sim(3) world weld while rejecting anchor outliers.

    Hypotheses are enumerated deterministically from 3-anchor minimal subsets,
    then the best consensus is refit on all inliers.  This is intentionally a
    bounded robust-estimation layer, not bundle adjustment or landmark identity
    discovery.
    """
    values = list(pairs)
    if max_residual_m <= 0 or not np.isfinite(max_residual_m):
        raise ValueError("max_residual_m must be positive and finite")
    if min_inliers < 3:
        raise ValueError("min_inliers must be at least three")
    if len(values) < min_inliers:
        raise ValueError("not enough anchor pairs for required consensus")
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    for pair in values:
        if pair.status != "candidate":
            raise ValueError("world weld requires candidate anchor pairs")

    best_indices: tuple[int, ...] | None = None
    best_count = -1
    best_rms = float("inf")
    tested = 0

    for hypothesis_indices in combinations(range(len(values)), 3):
        if tested >= max_hypotheses:
            break
        tested += 1
        hypothesis_pairs = [values[i] for i in hypothesis_indices]
        try:
            hypothesis = estimate_world_weld(
                *_pair_arrays(hypothesis_pairs),
                allow_scale=allow_scale,
            )
        except ValueError:
            continue
        residual = _residuals(hypothesis, values)
        inliers = tuple(int(i) for i in np.flatnonzero(residual <= max_residual_m))
        if not inliers:
            continue
        rms = float(np.sqrt(np.mean(np.square(residual[list(inliers)]))))
        if len(inliers) > best_count or (len(inliers) == best_count and rms < best_rms):
            best_indices = inliers
            best_count = len(inliers)
            best_rms = rms

    if best_indices is None or best_count < min_inliers:
        raise ValueError("no robust static-anchor consensus meets min_inliers")

    consensus_pairs = [values[i] for i in best_indices]
    weld = estimate_world_weld(*_pair_arrays(consensus_pairs), allow_scale=allow_scale)
    residual = _residuals(weld, values)
    final_indices = tuple(int(i) for i in np.flatnonzero(residual <= max_residual_m))
    if len(final_indices) < min_inliers:
        raise ValueError("refit lost required static-anchor consensus")
    if final_indices != best_indices:
        consensus_pairs = [values[i] for i in final_indices]
        weld = estimate_world_weld(*_pair_arrays(consensus_pairs), allow_scale=allow_scale)
        residual = _residuals(weld, values)
        final_indices = tuple(int(i) for i in np.flatnonzero(residual <= max_residual_m))
        if len(final_indices) < min_inliers:
            raise ValueError("final refit lost required static-anchor consensus")

    final_set = set(final_indices)
    inlier_ids = tuple(values[i].anchor_id for i in final_indices)
    outlier_ids = tuple(values[i].anchor_id for i in range(len(values)) if i not in final_set)
    return RobustWorldWeldReceipt(
        weld=weld,
        pair_count=len(values),
        inlier_count=len(final_indices),
        outlier_count=len(values) - len(final_indices),
        inlier_anchor_ids=inlier_ids,
        outlier_anchor_ids=outlier_ids,
        same_object_identity_required=True,
        robust_outlier_rejection=True,
        similarity_scale_exposed=bool(allow_scale),
        status="candidate",
    )
