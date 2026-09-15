"""Candidate-only passive acoustic localization for Animalexic scenes.

This module estimates a source position and unknown emission time from
synchronized microphone arrival times.  It is deliberately an observer, not an
identity or semantic authority: a good TDOA fit narrows the source-location
fibre but does not identify the animal that emitted the signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PassiveAcousticLocalizationCandidate:
    event_id: str
    position_world_m: tuple[float, float, float]
    position_sigma_m: float
    emission_time_s: float
    rms_tdoa_residual_s: float
    microphone_count: int
    sound_speed_m_s: float
    clock_calibration_reference: str
    microphone_geometry_reference: str
    propagation_medium_reference: str
    status: str = "candidate"
    clock_synchronization_paid: bool = True
    microphone_geometry_paid: bool = True
    medium_model_paid: bool = True
    unique_emitter_paid: bool = False
    species_identity_paid: bool = False
    active_probe_used: bool = False


def _validate_geometry(microphones: np.ndarray) -> None:
    if microphones.ndim != 2 or microphones.shape[1:] != (3,):
        raise ValueError("microphone_positions_m must be an Nx3 array")
    if len(microphones) < 4:
        raise ValueError("at least four synchronized microphones are required")
    if not np.all(np.isfinite(microphones)):
        raise ValueError("microphone coordinates must be finite")
    centered = microphones - microphones.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-10) < 3:
        raise ValueError("microphone geometry must span three dimensions")


def localize_tdoa_candidate(
    *,
    event_id: str,
    microphone_positions_m: Sequence[Sequence[float]],
    arrival_times_s: Sequence[float],
    sound_speed_m_s: float,
    clock_calibration_reference: str,
    microphone_geometry_reference: str,
    propagation_medium_reference: str,
    max_iterations: int = 100,
    convergence_position_m: float = 1e-9,
    convergence_time_s: float = 1e-12,
) -> PassiveAcousticLocalizationCandidate:
    """Estimate a candidate 3-D source position from synchronized arrivals.

    The unknown emission time is fitted jointly with position.  The returned
    uncertainty is an observation-scale diagnostic derived from the RMS timing
    residual multiplied by sound speed; it is not a calibrated confidence
    region and must not be treated as one without an external calibration
    receipt.
    """
    if not event_id:
        raise ValueError("event_id must be non-empty")
    if not clock_calibration_reference:
        raise ValueError("clock_calibration_reference must be non-empty")
    if not microphone_geometry_reference:
        raise ValueError("microphone_geometry_reference must be non-empty")
    if not propagation_medium_reference:
        raise ValueError("propagation_medium_reference must be non-empty")
    if sound_speed_m_s <= 0 or not np.isfinite(sound_speed_m_s):
        raise ValueError("sound_speed_m_s must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    microphones = np.asarray(microphone_positions_m, dtype=np.float64)
    arrival_times = np.asarray(arrival_times_s, dtype=np.float64)
    _validate_geometry(microphones)
    if arrival_times.shape != (len(microphones),):
        raise ValueError("arrival_times_s must contain one time per microphone")
    if not np.all(np.isfinite(arrival_times)):
        raise ValueError("arrival times must be finite")

    # Centroid initialization is deterministic and works well for sources near
    # the array aperture.  This tranche intentionally avoids pretending to be
    # a globally optimal hyperbolic-localization solver.
    position = microphones.mean(axis=0)
    initial_ranges = np.linalg.norm(microphones - position, axis=1)
    emission_time = float(np.median(arrival_times - initial_ranges / sound_speed_m_s))

    for _ in range(max_iterations):
        delta = position[None, :] - microphones
        ranges = np.linalg.norm(delta, axis=1)
        if np.any(ranges <= 1e-12):
            # Move infinitesimally off a microphone to keep the Jacobian finite.
            position = position + np.asarray([1e-9, -1e-9, 1e-9])
            continue

        predicted = emission_time + ranges / sound_speed_m_s
        residual = arrival_times - predicted
        jacobian = np.empty((len(microphones), 4), dtype=np.float64)
        jacobian[:, :3] = delta / (sound_speed_m_s * ranges[:, None])
        jacobian[:, 3] = 1.0
        step, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        if not np.all(np.isfinite(step)):
            raise ValueError("TDOA localization produced a non-finite update")
        position = position + step[:3]
        emission_time = emission_time + float(step[3])
        if (
            float(np.linalg.norm(step[:3])) <= convergence_position_m
            and abs(float(step[3])) <= convergence_time_s
        ):
            break

    final_ranges = np.linalg.norm(microphones - position, axis=1)
    final_residual = arrival_times - (
        emission_time + final_ranges / sound_speed_m_s
    )
    rms_residual = float(np.sqrt(np.mean(final_residual * final_residual)))
    if not np.isfinite(rms_residual):
        raise ValueError("TDOA localization produced a non-finite residual")

    return PassiveAcousticLocalizationCandidate(
        event_id=event_id,
        position_world_m=tuple(float(x) for x in position),
        position_sigma_m=max(float(sound_speed_m_s) * rms_residual, 1e-9),
        emission_time_s=float(emission_time),
        rms_tdoa_residual_s=rms_residual,
        microphone_count=int(len(microphones)),
        sound_speed_m_s=float(sound_speed_m_s),
        clock_calibration_reference=clock_calibration_reference,
        microphone_geometry_reference=microphone_geometry_reference,
        propagation_medium_reference=propagation_medium_reference,
    )
