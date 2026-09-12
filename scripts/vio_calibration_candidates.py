from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TimedVectorSample:
    time_s: float
    value: tuple[float, ...]


@dataclass(frozen=True)
class ClockOffsetCandidate:
    offset_s: float
    rms_residual: float
    residual_margin: float
    overlap_count: int
    ambiguous: bool
    status: str
    clock_alignment_paid: bool = False


def _validated_series(samples):
    values = list(samples)
    if len(values) < 2:
        raise ValueError("at least two timed vector samples are required")
    times = np.asarray([float(sample.time_s) for sample in values], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("sample timestamps must be finite and strictly increasing")
    vectors = [np.asarray(sample.value, dtype=np.float64) for sample in values]
    if any(vector.ndim != 1 or vector.size == 0 for vector in vectors):
        raise ValueError("sample vectors must be non-empty one-dimensional values")
    width = vectors[0].shape
    if any(vector.shape != width for vector in vectors):
        raise ValueError("sample vectors must have equal dimension")
    matrix = np.vstack(vectors)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sample vectors must be finite")
    return times, matrix


def _interpolate_matrix(source_times, source_values, query_times):
    columns = [
        np.interp(query_times, source_times, source_values[:, column])
        for column in range(source_values.shape[1])
    ]
    return np.stack(columns, axis=1)


def estimate_clock_offset_candidate(
    reference_samples,
    sensor_samples,
    *,
    search_offsets_s,
    min_overlap: int,
    min_residual_margin: float,
    max_rms_residual: float | None = None,
) -> ClockOffsetCandidate:
    """Estimate a candidate time offset by residual over overlapping motion.

    The sign convention is: ``sensor_time + offset_s`` is compared with the
    reference clock. The result remains candidate-only; a downstream alignment
    receipt must still pay use of this offset in VIO correction.
    """
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least two")
    if min_residual_margin < 0 or not math.isfinite(min_residual_margin):
        raise ValueError("min_residual_margin must be finite and non-negative")
    if max_rms_residual is not None and (
        max_rms_residual < 0 or not math.isfinite(max_rms_residual)
    ):
        raise ValueError("max_rms_residual must be finite and non-negative")

    reference_times, reference_values = _validated_series(reference_samples)
    sensor_times, sensor_values = _validated_series(sensor_samples)
    offsets = [float(offset) for offset in search_offsets_s]
    if not offsets or not all(math.isfinite(offset) for offset in offsets):
        raise ValueError("search_offsets_s must contain finite values")

    scored = []
    for offset in offsets:
        aligned_sensor_times = sensor_times + offset
        start = max(reference_times[0], aligned_sensor_times[0])
        stop = min(reference_times[-1], aligned_sensor_times[-1])
        mask = (reference_times >= start) & (reference_times <= stop)
        query_times = reference_times[mask]
        if len(query_times) < min_overlap:
            continue
        interpolated = _interpolate_matrix(
            aligned_sensor_times,
            sensor_values,
            query_times,
        )
        residual = reference_values[mask] - interpolated
        rms = float(np.sqrt(np.mean(np.square(residual))))
        scored.append((rms, offset, int(len(query_times))))

    if not scored:
        raise ValueError("no candidate clock offset has sufficient overlap")
    scored.sort(key=lambda row: (row[0], abs(row[1]), row[1]))
    best_rms, best_offset, best_overlap = scored[0]
    second_rms = scored[1][0] if len(scored) > 1 else math.inf
    margin = float(second_rms - best_rms)
    ambiguous = margin < min_residual_margin
    if max_rms_residual is not None and best_rms > max_rms_residual:
        ambiguous = True

    return ClockOffsetCandidate(
        offset_s=float(best_offset),
        rms_residual=float(best_rms),
        residual_margin=margin,
        overlap_count=best_overlap,
        ambiguous=bool(ambiguous),
        status="abstain" if ambiguous else "candidate",
        clock_alignment_paid=False,
    )
