# Promotion Rules (Concrete, Minimal)

Keep DASHI above the kernels: kernels emit candidates; promotion happens here.

## Disparity promotion (fixed-rig regime)
Inputs: candidate disparity, cost_min, conf_gap, valid_mask, severity.

Decision per pixel:
- Promote if: valid_mask==1 AND severity==0 AND cost_min <= τ_cost AND conf_gap >= τ_conf.
- Abstain if: severity > 0 OR cost_min > τ_cost OR conf_gap < τ_conf.
- Reject if: inputs missing/corrupt or ROI mismatch.

Outputs:
- promoted disparity (immutable until next promotion)
- residual detail = cost_min (kept for diagnostics)
- receipt with: frame ids, ROI, thresholds, mean/95th residual in ROI, decision.

Regime-specific thresholds (examples):
- τ_cost: 24 (Census 5x5 Hamming)
- τ_conf: 3

## Disparity promotion (phone/Regime B)
Stricter:
- require multi-view agreement (if two phones overlap): |d1 - d2| < τ_mv within ROI.
- require temporal consistency: |d_t - warp(d_{t-1})| < τ_temp.
- otherwise abstain (do not overwrite).

## Motion mask promotion
- Promote if mask is 1 and severity==0.
- Abstain on low-luma severity.
- Residual = diff_out kept for instrumentation.

## Geometry update guard
- Recompute extrinsics only if:
  - mean reprojection residual over ROI > τ_geom for N consecutive frames, or
  - sync error spike detected.
- Otherwise reuse previous geometry (slow state).

## Provenance / receipt minimal fields
- frame ids/timestamps
- ROI set id
- thresholds used
- counts of promoted / abstained / rejected pixels in ROI
- hashes of input images (or tiles) for replay
- list of invariants evaluated + pass/fail

## Lattice mapping
- severity: u8, max-join; severity>0 blocks promotion.
- promotion only increases canonical structure; abstain retains prior promoted state.

## Implementation note
- Promotion happens on host/runtime side; SPIR-V kernels never mutate promoted state.
