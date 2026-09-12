import unittest

from scripts.static_anchor_candidates import (
    StaticFeatureObservation,
    build_temporal_candidate_tracks,
    propose_cross_camera_candidates,
)


class StaticAnchorCandidateTests(unittest.TestCase):
    def feature(self, camera, time_s, feature_id, descriptor, provenance):
        return StaticFeatureObservation(
            camera_id=camera,
            time_s=time_s,
            feature_id=feature_id,
            descriptor=tuple(descriptor),
            static_confidence=0.99,
            provenance=provenance,
        )

    def test_mutual_distinct_descriptor_match_is_candidate_not_identity_payment(self):
        source = [self.feature(0, 0.0, "s0", [0.0, 0.0], "src0")]
        target = [
            self.feature(1, 0.01, "t0", [0.01, 0.0], "dst0"),
            self.feature(1, 0.01, "t1", [2.0, 2.0], "dst1"),
        ]
        candidates = propose_cross_camera_candidates(
            source,
            target,
            max_descriptor_distance=0.1,
            min_margin=0.2,
        )
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.source_feature_id, "s0")
        self.assertEqual(candidate.target_feature_id, "t0")
        self.assertEqual(candidate.status, "candidate")
        self.assertFalse(candidate.same_object_paid)
        self.assertEqual(candidate.source_provenance, "src0")
        self.assertEqual(candidate.target_provenance, "dst0")

    def test_ambiguous_descriptor_match_abstains(self):
        source = [self.feature(0, 0.0, "s0", [0.0, 0.0], "src")]
        target = [
            self.feature(1, 0.0, "a", [0.01, 0.0], "a"),
            self.feature(1, 0.0, "b", [-0.01, 0.0], "b"),
        ]
        candidates = propose_cross_camera_candidates(
            source,
            target,
            max_descriptor_distance=0.1,
            min_margin=0.05,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].status, "abstain")
        self.assertTrue(candidates[0].ambiguous)
        self.assertFalse(candidates[0].same_object_paid)

    def test_temporal_tracks_retain_candidate_status_and_provenance(self):
        observations = [
            self.feature(0, 0.0, "f0", [0.0, 0.0], "p0"),
            self.feature(0, 0.1, "f1", [0.01, 0.0], "p1"),
            self.feature(0, 0.2, "f2", [0.02, 0.0], "p2"),
        ]
        tracks = build_temporal_candidate_tracks(
            observations,
            max_time_delta_s=0.15,
            max_descriptor_distance=0.05,
            min_margin=0.01,
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual([member.feature_id for member in tracks[0].members], ["f0", "f1", "f2"])
        self.assertEqual(tracks[0].status, "candidate")
        self.assertFalse(tracks[0].same_object_paid)
        self.assertEqual(tracks[0].provenance_chain, ("p0", "p1", "p2"))


if __name__ == "__main__":
    unittest.main()
