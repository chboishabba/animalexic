# Architecture

## Top-Level Shape
The repo is structured as a governed stereo runtime, not just a raw matcher. Fast candidate generation happens per frame, while canonical state updates happen through explicit governance and receipt logging.

## Runtime Layers
- Ingest and dispatch: [scripts/run_stereo_dispatch.py](/home/c/Documents/code/animalexic/scripts/run_stereo_dispatch.py) streams frames from local files or YouTube, handles SBS splitting, optional auto-resolution probing, ROI selection, and the overall runtime loop.
- Fixed-rig core: [scripts/fixed_rig_runtime.py](/home/c/Documents/code/animalexic/scripts/fixed_rig_runtime.py) owns calibration loading, rectification, delta ROI construction, disparity promotion, temporal merge, depth conversion, and receipt persistence.
- Self-calibration: [scripts/self_calibrate_stereo.py](/home/c/Documents/code/animalexic/scripts/self_calibrate_stereo.py) bootstraps `fixed_rig_selfcal_v1` artifacts from stereo imagery when explicit board calibration is unavailable.
- Oracle/reference path: [scripts/opencv_sbs_oracle.py](/home/c/Documents/code/animalexic/scripts/opencv_sbs_oracle.py) provides rectified disparity/depth reference outputs for comparison and tuning.
- Comparison and supervision: [scripts/compare_oracle_runtime.py](/home/c/Documents/code/animalexic/scripts/compare_oracle_runtime.py), [scripts/merge_policy.py](/home/c/Documents/code/animalexic/scripts/merge_policy.py), and [scripts/oracle_teacher.py](/home/c/Documents/code/animalexic/scripts/oracle_teacher.py) turn artifact disagreement into explicit tuning and teacher signals.
- Calibration experiments: [scripts/oracle_calibrate_confidence.py](/home/c/Documents/code/animalexic/scripts/oracle_calibrate_confidence.py) fits a lightweight confidence model from balanced teacher exports, but that model is still an explicit experiment rather than the default runtime path.

## State Model
- Slow state: calibration, intrinsics, extrinsics, and other geometry assumptions.
- Fast state: frames, frame-difference masks, ROI tiles, candidate disparity, and per-frame residuals.
- Canonical state: promoted disparity/depth and associated receipts.

This split matches the repo IR in [docs/IR.md](/home/c/Documents/code/animalexic/docs/IR.md) and the governance contract in [docs/governance.md](/home/c/Documents/code/animalexic/docs/governance.md).

## Processing Flow
1. Load or derive stereo geometry.
2. Stream stereo frames and optionally rectify them.
3. Build motion-gated ROI tiles, unless full-frame mode is forced.
4. Generate candidate disparity with CPU or SGBM-backed matching.
5. Compute decomposed evidence channels for each candidate field (cost, confidence proxy, LR delta, local median delta, texture, disparity gradient).
6. Apply governance thresholds and temporal merge rules.
7. Compare runtime artifacts against the oracle using source-aware alignment keys (`source_pts_time`, `source_selected_index`) instead of frame ordinal whenever available.
8. Persist receipts and write debug artifacts for comparison.

## Current Bottleneck
- Candidate and promoted coverage are now close enough that governance is no longer the dominant choke point.
- The main remaining gap is spatial agreement with the oracle: candidate placement and overlap (`IoU(C, O)`), not raw coverage.
- Learned per-pixel confidence is currently opt-in only because the first learned gate underperformed the heuristic decomposed-evidence path on aligned compare metrics.
- Disagreement heatmaps are now part of the comparison workflow so FN/FP structure can be inspected directly instead of only reading scalar metrics.

## Vulkan Path
Current Python runtime keeps Vulkan integration as a readiness hook. The intended path is:
- ingest/convert via `../dashiCORE`
- motion masking
- warp/rectification
- stereo ROI disparity kernel authored in this repo
- preview/output writeback

The Vulkan/SPIR-V layer must remain below governance. Kernels may propose candidates, but promotion and canonical state mutation stay in the scheduler/runtime layer.
