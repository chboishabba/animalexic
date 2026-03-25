# Governance and Promotion

This document defines how candidate results become canonical, and what must be recorded for traceability.

## Roles
- **Substrate**: raw inputs (frames, timestamps, device metadata).
- **Candidate**: derived hypotheses (disparity, masks, body pose, splats).
- **Promoted**: canonical state accepted into S_t.
- **Abstain**: decision to keep prior canonical state.
- **Reject**: candidate is invalid and should not be reused.

## Promotion rules (apply per type)
Promotion is allowed only when all required invariants pass:
- Reprojection residual below threshold for the relevant ROI.
- Temporal consistency over a short window.
- Multi-view agreement (when available).
- Confidence above per-type threshold.
- No violated hard invariants (e.g., geometry sanity, non-negativity).

If any required invariant fails → Abstain (retain previous promoted state).  
If inputs are inconsistent or corrupt → Reject.

## Recompute guards
- Do **not** recompute global geometry every frame. Trigger only when:
  - drift/error exceeds bound,
  - sync error spike,
  - sustained residual growth,
  - landmark mismatch persists.
- Prefer ROI recompute driven by motion masks or residual hotspots.

## Receipts (audit trail)
Every promotion/abstention/rejection logs a Receipt containing:
- Inputs: frame IDs/timestamps, candidate refs, prior promoted refs.
- Kernels executed + parameters + ROIs.
- Invariants evaluated + results + thresholds.
- Residual metrics (mean/max, histogram if cheap).
- Decision: promote/abstain/reject + reason string.
- Hash/checksum of inputs and outputs for replay.

Receipts are append-only; canonical state changes must reference a Receipt.

## Separation of concerns
- SPIR-V kernels: fast, heuristic, local; cannot mutate promoted state.
- DASHI/runtime: owns promotion, invariants, receipts, and state lattice.

## Default decisions
- Absence of sufficient evidence ⇒ Abstain (never silent promotion).
- Promotion may increase structure; demotion requires explicit governance action.

## Minimal invariant library (starting set)
- `Inv_Reproj(depth, cams, frames, roi)`: mean/95th residual < τ_r.
- `Inv_Temporal(depth_t, depth_t-1, flow, roi)`: warped difference < τ_t.
- `Inv_Consistency(depth, mask_conf)`: candidate confidence > τ_c and coverage > τ_cov.
- `Inv_Geometry(poses)`: baseline / FOV sane; determinant / norms bounded.
- `Inv_BodyPose(body3d)`: joint lengths within anatomical range.

Thresholds are per-regime (fixed rig vs phone) and configurable.
