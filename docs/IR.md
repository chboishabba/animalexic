# Project IR (Semantic ↔ SPIR-V Bridge)

This IR is the contract between the DASHI semantic layer and the SPIR-V execution layer. It is hardware-agnostic but execution-aware: every node carries provenance, confidence, and promotion status.

## Core value types
- `Image`: 2D array with format metadata (e.g., RGBA8, NV12, R16F).
- `Mask`: binary/soft mask aligned to an Image; may carry confidence.
- `Field2D`: per-pixel scalar/vector field (e.g., disparity, residual).
- `Field3D`: sparse/dense 3D field (e.g., depth volume, surfels).
- `PointSet`: XYZ (+ optional normal/color/confidence) points.
- `SplatSet`: point-like elements with radius/extent and opacity; used for fast preview.
- `CameraModel`: intrinsics, extrinsics, rolling-shutter params, timestamp sync info.
- `Flow`: optical flow / warp field.
- `Residual`: per-pixel or per-region reprojection/consistency error.

## Wrappers
- `Candidate<T>`: proposed value with confidence, provenance, timestamp.
- `Promoted<T>`: canonical value; immutable except via governance transitions.
- `Provenance`: kernel list, inputs, regions, parameters, checksums.
- `Receipt`: record of promotion/abstention decision (inputs, invariants, residuals, reason).
- `InvariantCheck`: named predicate with thresholds/bounds; must be replayable.

## State tuple (per time t)
S_t = (G_t, V_t, D_t, B_t, E_t, P_t)
- G_t: geometry state (CameraModel, calibration confidences) — slow
- V_t: visual substrate (frames, masks, flow, residuals) — fast
- D_t: spatial field (disparity/depth/PointSet/SplatSet) — medium
- B_t: body state (pose, landmarks, orientation) — medium
- E_t: event/semantic state (behaviour tokens, motifs) — slow/medium
- P_t: provenance/promotion state (candidate vs promoted tags) — canonical

Timescale split:
- Slow: G_t, background scene priors
- Medium: D_t, B_t
- Fast: V_t (per-frame residuals, masks)
- Canonical semantic: promoted subsets of D_t, B_t, E_t

## Lattice (truth levels)
substrate → candidate → promoted  
Transitions must be explicit; abstain/reject are first-class outcomes. Fast kernels cannot mutate promoted state directly.

## Kernel plan nodes (lowerable to SPIR-V)
- `frame_diff(Image a, Image b) -> Mask/Residual`
- `threshold(Mask/Residual) -> Mask`
- `warp(Image/Field2D, Flow/CameraModel) -> Image/Field2D`
- `stereo_roi(Image L, Image R, ROI, CameraModel) -> Field2D disparity + Residual`
- `flow(Image a, Image b) -> Flow` (optional backend)
- `splat_render(SplatSet/PointSet, CameraModel) -> Image/Field2D`
- `reduce(Field2D/Mask, op=sum/max/mean) -> scalar(s)`
- `compose_masks(Mask...) -> Mask`
- `update_state(prev Field/PointSet, Mask, delta Field) -> Field/PointSet`

Each node records: input refs, output refs, ROI, parameters, and is pure/deterministic at this level.

## Governance hooks (interfaces)
- `promote(Candidate<T>, invariants[], receipts[]) -> Promoted<T> | Abstain | Reject`
- `invariant_check(T, InvariantCheck) -> pass/fail + metrics`
- `receipt_log(Receipt)` → append-only store (for audit/replay)

## Provenance requirements
For every Candidate or Promoted value:
- source frames and timestamps
- kernels executed + parameters
- regions/ROIs touched
- checksums or hashes of inputs
- residual metrics used for decision
- promotion decision and reason

## Execution notes
- SPIR-V layer only implements bounded local operators (above kernels).
- Governance, promotion, and lattice transitions remain in the DASHI layer / runtime scheduler.
- CPU fallback is allowed; IR remains the same.
