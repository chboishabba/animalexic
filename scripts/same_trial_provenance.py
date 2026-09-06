"""Same-trial provenance graph for governed multimodal animal observations.

This module is intentionally domain-neutral: Drosophila neural imaging, stereo
pose, contact, acoustic, or physiological channels may corroborate one another
without being independent when they share an animal, trial, registration, or
preprocessing root.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Mapping, TypeVar


class RootKind(str, Enum):
    SUBJECT = "subject"
    TRIAL = "trial"
    ACQUISITION = "acquisition"
    REGISTRATION = "registration"
    PREPROCESSING = "preprocessing"
    MODEL = "model"
    DATASET = "dataset"


@dataclass(frozen=True)
class Root:
    root_id: str
    kind: RootKind
    stable_reference: str


ArtifactT = TypeVar("ArtifactT", bound=Hashable)


@dataclass(frozen=True)
class EvidenceDependenceGraph:
    roots_by_artifact: Mapping[ArtifactT, frozenset[Root]]

    def roots(self, artifact: ArtifactT) -> frozenset[Root]:
        return self.roots_by_artifact.get(artifact, frozenset())

    def shared_roots(self, left: ArtifactT, right: ArtifactT) -> frozenset[Root]:
        return self.roots(left) & self.roots(right)

    def independent(self, left: ArtifactT, right: ArtifactT) -> bool:
        left_roots = self.roots(left)
        right_roots = self.roots(right)
        # Missing provenance is epistemic uncertainty, never proof of independence.
        return bool(left_roots and right_roots) and left_roots.isdisjoint(right_roots)

    def same_trial(self, left: ArtifactT, right: ArtifactT) -> bool:
        return any(root.kind is RootKind.TRIAL for root in self.shared_roots(left, right))

    def same_subject(self, left: ArtifactT, right: ArtifactT) -> bool:
        return any(root.kind is RootKind.SUBJECT for root in self.shared_roots(left, right))

    def shares_pipeline(self, left: ArtifactT, right: ArtifactT) -> bool:
        return any(
            root.kind in {RootKind.REGISTRATION, RootKind.PREPROCESSING, RootKind.MODEL}
            for root in self.shared_roots(left, right)
        )


@dataclass(frozen=True)
class TrialBundle:
    subject_id: str
    trial_id: str
    structural_artifact: str
    functional_artifact: str
    body_artifact: str
    behaviour_artifact: str
    registration_artifact: str


def can_claim_independent_replication(
    graph: EvidenceDependenceGraph, left: ArtifactT, right: ArtifactT
) -> bool:
    return graph.independent(left, right)
