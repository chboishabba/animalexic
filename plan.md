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
- Candidate: **Retinify** (GPU-accelerated stereo pipeline; architecturally aligned with 4K60).
- Action: proceed with Retinify as first implementation; keep one fallback lightweight classical matcher ready if Retinify underperforms on our GPU.
- Open item: confirm exact GPU model + available VRAM to tune parameters.

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
- Stand up Retinify locally and run a 4K test clip (even synthetic) to get baseline FPS.
- Capture or download two short clips for evaluation sets (fixed rig + dual-phone).
- Add lightweight instrumentation wrapper to log FPS/latency/pixel coverage per stage.

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
