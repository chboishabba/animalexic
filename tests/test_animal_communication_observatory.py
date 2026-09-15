import math

import numpy as np

from scripts.passive_acoustic_localization import localize_tdoa_candidate
from scripts.animal_communication_observatory import (
    VisualTrackCandidate,
    associate_visual_acoustic_candidates,
)


def _arrival_times(source, microphones, sound_speed=343.0, emission_time=1.25):
    source = np.asarray(source, dtype=float)
    microphones = np.asarray(microphones, dtype=float)
    return emission_time + np.linalg.norm(microphones - source, axis=1) / sound_speed


def test_tdoa_localization_recovers_candidate_position_without_promotion():
    microphones = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [4.0, 4.0, 2.0],
        ],
        dtype=float,
    )
    source = np.asarray([1.2, 1.7, 1.1], dtype=float)
    arrival_times = _arrival_times(source, microphones)

    candidate = localize_tdoa_candidate(
        event_id="call-17",
        microphone_positions_m=microphones,
        arrival_times_s=arrival_times,
        sound_speed_m_s=343.0,
        clock_calibration_reference="clock:synthetic-exact",
        microphone_geometry_reference="array:synthetic-5mic",
        propagation_medium_reference="air:343mps-fixture",
    )

    assert candidate.status == "candidate"
    assert candidate.active_probe_used is False
    assert candidate.unique_emitter_paid is False
    assert candidate.species_identity_paid is False
    assert candidate.clock_synchronization_paid is True
    assert candidate.microphone_geometry_paid is True
    assert np.linalg.norm(np.asarray(candidate.position_world_m) - source) < 0.03
    assert candidate.rms_tdoa_residual_s < 1e-5


def test_shared_world_association_keeps_many_to_many_candidates():
    acoustic = type(
        "AcousticCandidate",
        (),
        {
            "event_id": "call-9",
            "position_world_m": (2.0, 2.0, 1.0),
            "position_sigma_m": 0.8,
            "status": "candidate",
        },
    )()
    tracks = [
        VisualTrackCandidate(
            track_id="bird-a",
            position_world_m=(1.9, 2.1, 1.1),
            position_sigma_m=0.5,
            species_hypothesis="magpie",
            individual_hypothesis=None,
            provenance_reference="cam:A",
        ),
        VisualTrackCandidate(
            track_id="bird-b",
            position_world_m=(2.5, 2.0, 1.2),
            position_sigma_m=0.5,
            species_hypothesis="magpie",
            individual_hypothesis=None,
            provenance_reference="cam:B",
        ),
    ]

    associations = associate_visual_acoustic_candidates(
        acoustic,
        tracks,
        max_normalized_distance=1.0,
        temporal_alignment_reference="sync:fixture",
    )

    assert [a.visual_track_id for a in associations] == ["bird-a", "bird-b"]
    assert all(a.status == "candidate" for a in associations)
    assert all(a.same_object_identity_paid is False for a in associations)
    assert all(a.vocal_emitter_identity_paid is False for a in associations)
    assert all(a.semantic_meaning_paid is False for a in associations)
    assert all(a.many_to_many_association_retained is True for a in associations)


def test_far_visual_track_is_not_forced_into_association():
    acoustic = type(
        "AcousticCandidate",
        (),
        {
            "event_id": "call-10",
            "position_world_m": (0.0, 0.0, 0.0),
            "position_sigma_m": 0.5,
            "status": "candidate",
        },
    )()
    tracks = [
        VisualTrackCandidate(
            track_id="far-bird",
            position_world_m=(20.0, 0.0, 0.0),
            position_sigma_m=0.5,
            species_hypothesis="unknown",
            individual_hypothesis=None,
            provenance_reference="cam:far",
        )
    ]

    assert associate_visual_acoustic_candidates(
        acoustic,
        tracks,
        max_normalized_distance=2.0,
        temporal_alignment_reference="sync:fixture",
    ) == []
