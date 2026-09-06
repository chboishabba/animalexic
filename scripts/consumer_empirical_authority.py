"""Consumer-scoped empirical authority for governed Animalexic experiments.

This is the domain-generic counterpart of the MaleCNS real-data benchmark gate.
An observation can be present, digested, and useful diagnostically without being
artifact-authority verified for a particular consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class VerificationLevel(str, Enum):
    MISSING = "missing"
    PRESENT = "present"
    DIGESTED = "digested"
    HASH_VERIFIED = "hash_verified"


class EvidenceResolution(str, Enum):
    OBSERVATION = "observation"
    REGION_OR_MODULE = "region_or_module"
    INDIVIDUAL_IDENTITY = "individual_identity"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class ConsumerAuthorityPolicy:
    name: str
    required_artifacts: tuple[str, ...]
    minimum_resolution: EvidenceResolution
    active: bool = True


@dataclass(frozen=True)
class GovernedEmpiricalBundle:
    policy: ConsumerAuthorityPolicy
    verification: Mapping[str, VerificationLevel]
    resolution: EvidenceResolution
    held_out_verified: bool
    null_control_verified: bool

    @property
    def artifact_verified(self) -> bool:
        return all(
            self.verification.get(k) is VerificationLevel.HASH_VERIFIED
            for k in self.policy.required_artifacts
        )

    @property
    def empirically_promotable(self) -> bool:
        if not self.policy.active:
            return False
        return self.artifact_verified and self.held_out_verified and self.null_control_verified


def transfer_requires_receipt(source_consumer: str, target_consumer: str) -> bool:
    return source_consumer != target_consumer


def same_region_implies_same_individual() -> bool:
    return False


def repository_identifier_implies_exact_file() -> bool:
    return False
