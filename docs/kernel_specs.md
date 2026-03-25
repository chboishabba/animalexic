# Kernel Specs (SPIR-V Targets)

Minimal executable kernels to implement next, following the IR/governance split. These are algorithm specs, not code.

## frame_diff (motion mask)
- Inputs: Image L_t (R8/RGBA8), Image L_{t-1}, threshold τ (scalar), optional ROI mask.
- Output: Mask M (uint8) where M=1 if |L_t - L_{t-1}| > τ per channel (luma if available), else 0; Residual R (abs diff) optional.
- Notes:
  - Prefer luma (NV12 → Y) to reduce bandwidth.
  - Local_size: 16×16 or 32×8; coalesced linear read.
  - Optional dilation step can reuse core_mask kernels.

## stereo_roi (Census + winner-takes-all)
- Inputs: Rectified Image L, Image R (R8), ROI mask (optional), disparity range [d_min, d_max], window size w (e.g., 5×5), Census bitwidth (e.g., 24–32 bits).
- Outputs: Field2D disparity (int16/float), Residual (matching cost).
- Steps:
  1) For each pixel in ROI: compute Census(L) over window.
  2) For each disparity d in [d_min, d_max]: shift R by d; Census(R); cost = Hamming(Census(L), Census(R)).
  3) Pick d with min cost; write disparity and cost.
  4) Optional: left-right consistency check flag.
- Notes:
  - Use packed uint32 census per pixel; Hamming via popcount.
  - Early exit if ROI mask=0.
  - Local_size: 8×8 or 16×8 to balance registers vs occupancy.

## warp_depth
- Reuse existing `warp_affine_2d.spv` / `warp_piecewise.spv`.
- Inputs: depth/disparity field, flow or rectification warp, ROI.
- Output: warped field and validity mask.

## splat_render (lightweight splats)
- Inputs: PointSet/SplatSet (XYZ(+radius,color,opacity)), CameraModel.
- Output: Image (RGBA or depth+color), optional coverage mask.
- Minimal version: project point → screen pixel; depth test (nearest) or simple over compositing with small radius (e.g., 1–2 px).
- Notes: start with atomic min on depth buffer for simplicity.

## Optional (Phase 2)
- flow_lk: pyramid Lucas–Kanade 2×2 or 3×3 patches for coarse flow to warp previous depth.
- sgm_agg: directional path aggregation (P1/P2 penalties) on disparity; only if quality needs bump after baseline.
