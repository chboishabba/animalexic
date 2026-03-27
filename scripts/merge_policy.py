#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MergePolicy:
    max_cost: float = 20.0
    min_gap: float = 5.0
    min_conf: float = 0.55
    tau_close_disp: float = 1.0
    conf_improvement_req: float = 0.12
    max_age_keep: int = 14
    min_evidence_frames: int = 2
    weak_conf_scale: float = 0.70
    max_severity_promote: int = 1
    tau_oracle_disp: float = 2.0
    min_region_fill: float = 0.20
    max_region_var: float = 16.0
    min_oracle_overlap: float = 0.25


@dataclass
class CandidateEvidence:
    valid: bool
    disparity: float
    cost: float
    gap: float
    confidence: float
    severity: int = 0
    prior_valid: bool = False
    prior_disparity: float = 0.0
    prior_confidence: float = 0.0


@dataclass
class OracleEvidence:
    valid: bool
    disparity: float = 0.0


def hard_ok(cand: CandidateEvidence, policy: MergePolicy) -> bool:
    return (
        cand.valid
        and cand.disparity > 0.0
        and cand.cost <= policy.max_cost
        and cand.gap >= policy.min_gap
        and cand.confidence >= policy.min_conf
        and cand.severity <= policy.max_severity_promote
    )


def online_promotion_decision(cand: CandidateEvidence, policy: MergePolicy) -> str:
    if not hard_ok(cand, policy):
        return "abstain"
    if cand.prior_valid and abs(cand.disparity - cand.prior_disparity) <= policy.tau_close_disp:
        return "promote_strong"
    if (not cand.prior_valid) or cand.confidence >= cand.prior_confidence + policy.conf_improvement_req:
        return "promote_weak"
    return "abstain"


def oracle_conditioned_decision(
    cand: CandidateEvidence,
    oracle: OracleEvidence,
    policy: MergePolicy,
) -> str:
    if not hard_ok(cand, policy):
        return "abstain"
    if oracle.valid and abs(cand.disparity - oracle.disparity) <= policy.tau_oracle_disp:
        return "promote_strong"
    if not oracle.valid:
        return "reject_oracle_conflict"
    return "abstain"


def region_accept(fill_ratio: float, variance: float, oracle_overlap: float | None, policy: MergePolicy) -> bool:
    if fill_ratio < policy.min_region_fill:
        return False
    if variance > policy.max_region_var:
        return False
    if oracle_overlap is not None and oracle_overlap < policy.min_oracle_overlap:
        return False
    return True


def propose_param_update(metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if metrics.get("fn_rate", 0.0) > 0.20:
        out["min_conf"] = -0.03
        out["min_gap"] = -0.5
        out["max_cost"] = +2.0
    if metrics.get("fp_rate", 0.0) > 0.10:
        out["min_conf"] = out.get("min_conf", 0.0) + 0.02
        out["min_gap"] = out.get("min_gap", 0.0) + 0.5
    if metrics.get("roi_miss_rate", 0.0) > 0.10:
        out["diff_threshold"] = -2.0
        out["tile_halo"] = +1.0
    if metrics.get("temporal_flicker", 0.0) > 0.15:
        out["tau_close_disp"] = +0.5
        out["max_age_keep"] = +2.0
    return out


def profile_calibrated_tight() -> MergePolicy:
    return MergePolicy(
        max_cost=20.0,
        min_gap=5.0,
        min_conf=0.55,
        tau_close_disp=1.0,
        conf_improvement_req=0.12,
        max_age_keep=14,
        min_evidence_frames=2,
        weak_conf_scale=0.70,
    )


def profile_calibrated_loose() -> MergePolicy:
    return MergePolicy(
        max_cost=26.0,
        min_gap=3.0,
        min_conf=0.38,
        tau_close_disp=1.5,
        conf_improvement_req=0.08,
        max_age_keep=12,
        min_evidence_frames=2,
        weak_conf_scale=0.70,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(asdict(profile_calibrated_loose()), indent=2, sort_keys=True))
