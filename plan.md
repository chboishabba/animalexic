# Plan — Stereo 3D 4K60 Pipeline

## Goal
Build a GPU-first stereo → 3D pipeline that can run live at 4K60 for a fixed stereo rig, with an extensible path for opportunistic multi-view phone captures of a moving dog.

## Context
- Source chat: “Stereo 3D 4K60 Pipeline” (online UUID 69bce770-5540-8398-9257-b1c7da6a1d11; canonical df0976e7ec5aabd4809189993f76cf4be8331b0b; 2026-03-20 06:22–06:40 UTC).
- Two regimes identified:
  - Regime A — fixed stereo rig: calibrate/rectify once, extrinsics are persistent state, per-frame depth/disparity with delta-only processing.
  - Regime B — opportunistic multi-view (phones): time + viewpoint alignment, pose recovery, fusion, handling rolling shutter/lens differences; throughput secondary to robustness.

## Success criteria
- Regime A: sustained ≥60 fps at 4K per eye on target GPU; depth quality visually stable; latency <50 ms end-to-end.
- Regime B: successful reconstruction of the dog from two unsynchronised phones with acceptable temporal coherence; clearly documented degradation when sync/overlap is poor.
- Instrumented metrics: FPS, latency breakdown, percent of pixels processed (delta gating), disparity error on a small ground-truth clip if available.

## Milestones

### M1 — Baseline Regime A pipeline
- Select stereo matcher (see M0 decision below) and run: calibration → rectification → disparity/depth → reprojection.
- Implement delta gating (frame_t - frame_{t-1}) to reduce work; measure effect.
- Add temporal merge + accumulation of disparity/depth into persistent canonical state with promote/abstain receipts.
- Metrics pass: FPS and latency breakdown on a fixed test clip; identify bottlenecks.

### M2 — 4K60 optimisation
- Tune matcher parameters and GPU settings; ensure memory fits.
- Parallelise/overlap stages (streamed rectification + stereo + reprojection).
- Validate stability over 5+ minute runs; record thermal/GPU utilisation.

### M3 — Regime B ingest and fusion
- Dual-phone capture ingest: timestamp alignment (audio clap / gyro), feature-based sync fallback.
- Viewpoint alignment: essential matrix + scale; rolling shutter compensation heuristic.
- Fusion: triangulation + temporal smoothing; handle partial overlap gracefully.
- Failure modes: low-overlap detection, desync detection, fallback to best single-view cues.

### M4 — Evaluation & dataset
- Assemble two evaluation sets:
  - Fixed rig: kennel/room clip with a moving dog.
  - Dual-phone: two owners filming the same dog with drifting viewpoints.
- Define scoring: FPS/latency (A), qualitative fusion/temporal stability (B), optional disparity error if ground truth exists.

## M0 — Stereo matcher decision (now)
- Oracle baseline: **OpenCV stereo** for calibration/rectification/disparity/reprojection sanity checks.
- Action: implement the OpenCV SBS oracle first; keep Retinify as a later benchmark/comparison candidate rather than the system-of-record.
- Open item: confirm exact GPU model + available VRAM to tune parameters for the eventual Vulkan/Retinify comparison path.

## Implementation outline (Regime A)
- Calibrate + rectify once; persist intrinsics/extrinsics.
- Per frame: load stereo pair → delta mask (abs diff vs previous frame) → stereo matching constrained to delta regions → fill/denoise → depth map → optional point cloud.
- Instrument timings per stage; log percent pixels processed after delta mask.

## Implementation outline (Regime B)
- Ingest phone videos → audio/gyro based coarse sync → feature match keyframes → estimate relative pose (E matrix + scale) → per-frame delta masks → stereo/multi-view disparity where overlap exists → triangulate → temporal smoothing.
- Periodically refresh extrinsics when pose drift exceeds a threshold; otherwise treat extrinsics as slow-changing state.

## Risks & mitigations
- Hitting 4K60: start with delta gating + GPU profiler; be ready to drop resolution/ROI temporarily while tuning.
- Rolling shutter/unsynced phones: use shorter exposure / higher shutter speed, apply rudimentary rolling-shutter correction only if needed.
- Thermal throttling: monitor clocks; keep a lower-power profile for long runs.

## Immediate actions (next 3 working sessions)
- Verify the ascended voxel set from the exact guarded downstream path before changing any guard parameters: overlay ascended voxels on the promoted point cloud, and measure nearest-promoted-point residuals versus plateau/grounded voxels.
- The first verifier pass is in place (`scripts/voxel_quality.py`); it shows ascended voxels are still slightly noisier than plateau on nearest-promoted-cloud residuals, so the next guard tuning should start with `h_a` rather than lowering `tau_a`.
- The first `h_a` bump (`2 -> 3`) reduced ascended count (`26 -> 20`) but made ascended residual slightly worse (`0.3379 -> 0.3585`), so the next guard knob should likely be temporal weighting (`beta`) or another persistence adjustment rather than lowering `tau_a`.
- If the ascended set is coherent, tune only one guard knob at a time:
  - first try increasing temporal persistence (`h_a`)
  - if needed, then lower `tau_a` slightly
- Keep the guarded downstream seam conservative until the quality checks pass. Surfel output is now available, but the first surfel verifier run shows ascended surfels are worse than plateau on promoted-cloud residuals, so surfel work should focus on guard/support correctness before any densification.
- Surfel verification now distinguishes anchor position from merged centroid. The current best corrected run still has worse ascended centroid residual than plateau (`0.0178` vs `0.0000`), so the next surfel task is merge-geometry correction rather than more threshold sweeps.
- Cross-frame-only surfel merging materially improved the surfel regime: the current best run recovers 9 ascended surfels with centroid residual `0.0024` instead of `0.0164+`, so the next surfel task is no longer same-frame merge cleanup. It is verifier/governance refinement for multi-frame surfels versus singleton plateau surfels.
- The verifier now makes that distinction explicitly. On the current cross-frame baseline, ascended centroid residual (`0.0024`) is better than the non-promoted multi-frame comparison set (`0.0233`), so the next surfel task is controlled densification from this fixed baseline.
- That first densification pass is now complete. The preferred new baseline is `tau_a=0.02`, which yields `71` ascended surfels while maintaining a clear centroid-residual advantage over the non-promoted multi-frame set. The next surfel task is a second one-knob sweep from that baseline, most likely `beta`.
- The second densification pass is also complete. `beta=1.00` on the `tau_a=0.02` baseline yields `103` ascended surfels while keeping the centroid-residual advantage over the non-promoted multi-frame set, so the next surfel task is a third knob from that baseline rather than revisiting `tau_a` or `beta`.
- The third densification pass is also complete. `gamma_neighbor=1.50` on the `tau_a=0.02`, `beta=1.00` baseline yields `157` ascended surfels while preserving the centroid-residual margin over the non-promoted multi-frame set, so the next surfel task is a support-radius refinement or a bounded continuation of neighbor support from this baseline.
- The support-radius pass is also complete. Tightening to `pos_eps=0.15` on the current baseline yields `165` ascended surfels while still preserving the centroid-residual margin, so the next surfel task is a geometry-aware merge refinement such as a depth-consistency gate.
- The surfel path now has measured early-stop instrumentation. On the current preferred 24-frame lossless clip, the guarded stop rule does not trigger once warmup/support gates are enforced, so the clip still adds useful structure through its end; the next ingest experiment should extend the same-object segment rather than shorten it.
- That stop instrumentation now uses a governed residual-margin metric instead of raw ascended residual: keep longer ingest runs only while `nonpromoted_multiframe_mean_residual - ascended_mean_residual` stays positive, with the current preferred baseline verifying a margin of `+0.00965`.
- Use the calibrated oracle as a teacher on source-aligned runs and optimize for overlap / agreement (`IoU`, false negatives, false positives), not coverage alone.
- Improve candidate placement with richer evidence and region-level reasoning before attempting another learned promotion gate.
- Keep learned confidence calibration opt-in until it beats the heuristic decomposed-evidence baseline on aligned compare metrics.
- Use disagreement heatmaps (`candidate_overlap_fNNNN.png`, `promoted_overlap_fNNNN.png`) as the primary debugging surface for FN/FP structure before changing runtime policy.
- Generate a calibration artifact and validate rectified/canonical depth on a local stereo clip; self-calibration is the default end-user path, with board calibration as an optional higher-quality route.
- Capture or download two short clips for evaluation sets (fixed rig + dual-phone).
- Add lightweight instrumentation wrapper to log FPS/latency/pixel coverage per stage.
- Stand up Retinify locally and run a 4K test clip (even synthetic) to get baseline FPS once the OpenCV oracle and calibrated local validation are stable.
- Runtime profiles are now explicit: `demo`, `demo_loose`, `calibrated`, `strict`; use `calibrated` for artifact-backed local validation and keep demo profiles for synthetic sanity only.
- Calibration artifact sources now include both board-based calibration and self-calibration from matched stereo imagery.

## Vulkan assets available (from ../dashiCORE)
- IO / convert: `spv/nv12_to_r8.spv`, `spv/nv12_to_rgba.spv`, `spv/write_image.spv` (plus preview vert/frag) — good for ingest + debug blits.
- Geometry/warp: `spv/warp_affine_2d.spv`, `spv/warp_piecewise.spv` — useful for rectification / reprojection.
- Masking/threshold helpers: `gpu_shaders/core_mask*.comp` and `spv/decode_threshold_maxbuf.spv` (can be repurposed for motion masks).
- Math / reduction: `spv/reduce_*`, `spv/prefix_scan_blocks.spv`, `spv/axpy.spv`, `spv/mul_scalar.spv` — handy for per-stage metrics and buffer ops.
- Sparse/dense primitives: `spv/push.spv`, `spv/pop.spv`, `spv/scatter_add_atomic.spv` — for ROI writes/reads if needed.
- Gap: no stereo block-matching/matching-specific shader yet; we’ll author a small disparity kernel (block or census) and keep it alongside these SPVs.

## Other DASHI repos inspected
- `../dashiRTX`: Python-only light-transport / PDA-MDL demos and figures; no Vulkan/SPIR-V or stereo-ready kernels to reuse.

## Governance & IR (added)
- `docs/IR.md`: project IR bridging DASHI semantics ↔ SPIR-V; defines core types, candidate/promoted wrappers, kernel plan nodes, provenance, and the S_t state tuple with lattice.
- `docs/governance.md`: promotion/abstention/rejection rules, receipts, invariants, and recompute guards.
- `docs/promotion_rules.md`: concrete, minimal promotion logic (regime-specific thresholds, severity gating, receipts) keeping DASHI above kernels.
- Next blocking item before disparity shader: keep scheduler/governance aligned with this IR (no promoted-state mutation from kernels).
