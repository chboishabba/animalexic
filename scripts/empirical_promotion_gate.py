"""Run-mode promotion gate shared by Animalexic-style governed experiments.

This keeps pipeline execution, diagnostic simulation and empirical promotion as
separate authority levels. It is intentionally domain-generic and can wrap the
Drosophila benchmark without importing its concrete manifest implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunMode(str, Enum):
    DRY_RUN = "dry_run"
    MOCK = "mock"
    SYNTHETIC = "synthetic"
    REAL_UNVERIFIED = "real_unverified"
    REAL_HASH_VERIFIED = "real_hash_verified"


class Authority(str, Enum):
    PIPELINE = "pipeline"
    DIAGNOSTIC = "diagnostic"
    EMPIRICAL_CANDIDATE = "empirical_candidate"
    EMPIRICAL = "empirical"


@dataclass(frozen=True)
class EmpiricalRunReceipt:
    mode: RunMode
    inputs_verified: bool
    registration_verified: bool
    split_verified: bool
    output_verified: bool
    synthetic_inputs_present: bool = False

    @property
    def artifact_verified(self) -> bool:
        return all(
            (
                self.inputs_verified,
                self.registration_verified,
                self.split_verified,
                self.output_verified,
            )
        )

    @property
    def authority(self) -> Authority:
        if self.mode in {RunMode.DRY_RUN, RunMode.MOCK}:
            return Authority.PIPELINE
        if self.mode is RunMode.SYNTHETIC or self.synthetic_inputs_present:
            return Authority.DIAGNOSTIC
        if self.mode is RunMode.REAL_UNVERIFIED or not self.artifact_verified:
            return Authority.EMPIRICAL_CANDIDATE
        return Authority.EMPIRICAL

    @property
    def empirical_promotion_allowed(self) -> bool:
        return self.authority is Authority.EMPIRICAL


def governed_promotion(candidate_policy_passed: bool, receipt: EmpiricalRunReceipt) -> bool:
    """Both candidate-level governance and empirical run authority are required."""

    return candidate_policy_passed and receipt.empirical_promotion_allowed
