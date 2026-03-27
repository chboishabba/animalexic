# Changelog

## 2026-03-26

- Added a standalone OpenCV SBS oracle baseline (`scripts/opencv_sbs_oracle.py`) for rectification, disparity, and optional point-cloud preview.
- Added `--auto-res` so the runtime can probe a source's native decoded resolution and scale to that instead of relying on a fixed test size.
- Added bootstrap self-calibration to the stereo runtime so a valid first pair can seed a calibration artifact when feature matches are available.
- Added `--full-frame-roi` to force full-frame stereo search for static or low-motion scenes, while keeping motion-gated ROI as the default compute-scope optimization.
- Made PNG artifact writing independent of Pillow so the runtime can emit outputs in a minimal environment.
- Tightened fallback behavior so self-calibration failures and frame-acquisition timeouts degrade cleanly into unrectified demo mode instead of crashing.
- Added explicit startup progress logs and heartbeat output for YouTube resolution, ffprobe auto-res probing, and first-frame waits so long stalls are visible instead of silent.
- Cleaned up ffmpeg subprocess ownership so the runtime and oracle close streamers on exit or Ctrl-C, suppress raw broken-pipe spam, and kill stuck ffmpeg children instead of leaving the terminal wedged.
- Added `scripts/compare_oracle_runtime.py` to compare oracle coverage against runtime receipts, with optional mask-IoU metrics when debug mask artifacts are present.
- Expanded `scripts/compare_oracle_runtime.py` into a repo-ready comparison tool that can join oracle summary/receipts with runtime sqlite or JSONL, emit per-frame joined metrics, and recommend threshold directions from oracle disagreement.
- Reduced log spam by suppressing ffmpeg's repeated-line noise and by only emitting wait heartbeats when frame acquisition is actually blocked.
- Loosened the calibrated runtime's CPU census candidate thresholds and print the active stereo thresholds on frame 0 so calibrated runs expose more candidate signal and are easier to tune against the oracle.
- Added runtime `--disp-min` / `--disp-max` controls and widened the calibrated default disparity search range to `128` so the CPU census matcher is not capped below the oracle's search range.
- Added `scripts/merge_policy.py` with explicit online/oracle-conditioned promotion rules and parameter-update heuristics, and `scripts/oracle_teacher.py` to export oracle-conditioned supervision from runtime/oracle mask artifacts.
- Added runtime matcher selection with calibrated `auto -> opencv_sgbm`, keeping CPU census available for sweeps/debug. Calibrated runs can now use an SGBM-backed candidate generator without changing the governed promotion/receipt layer.

## 2026-03-27

- Added source-aware oracle/runtime alignment by propagating ffmpeg-selected frame metadata through the oracle and runtime paths and teaching `scripts/compare_oracle_runtime.py` to join by source timing/index before falling back to frame ordinal.
- Suppressed `showinfo` log spam while preserving frame metadata extraction so aligned runs stay readable.
- Relaxed the calibrated SGBM promotion profile enough that promoted coverage now tracks candidate coverage instead of collapsing far below it.
- Persisted decomposed SGBM evidence maps (`candidate_lr_delta`, `candidate_median_delta`, `candidate_texture`, `candidate_disp_gradient`) alongside candidate cost/confidence artifacts.
- Expanded `scripts/oracle_teacher.py` and `scripts/oracle_calibrate_confidence.py` to export/train on richer evidence features with balanced sampling and feature standardization.
- Added an opt-in learned confidence model path for calibrated SGBM runs via `--confidence-model`; left it disabled by default because the first learned gate regressed aligned compare metrics relative to the heuristic decomposed-evidence baseline.
- Removed promotion leakage from the calibration feature set by dropping `runtime_promoted` from the learned confidence trainer inputs.
- Added FN/FP/TP disagreement heatmaps to `scripts/compare_oracle_runtime.py` so oracle/runtime placement errors can be inspected visually per frame.
- Added region-level acceptance to the governed merge path using connected support, component fill ratio, and local disparity variance; calibrated SGBM runs now reject many small/incoherent promotion islands before canonical update.
- Added CLI overrides for region-level acceptance (`--region-min-pixels`, `--region-max-disp-std`, `--region-min-fill-ratio`) and swept a first set of thresholds on the aligned house segment; the current calibrated SGBM default now uses `region_min_pixels=40` as the best tested tradeoff.
- Added `scripts/analyze_overlap_heatmaps.py` to aggregate TP/FP/FN heatmaps into tile-level hotspots; current region-filtered runs show dominant false negatives in the center/right image band and persistent false positives near the lower/right borders.
- Added edge-distance and border-penalty evidence maps to the SGBM runtime and teacher export. First border-aware matcher pass slightly improved candidate IoU but regressed promoted IoU on the aligned house segment, so it remains experimental and not a validated new default direction.
- Exported edge-distance and border-penalty features through `scripts/oracle_teacher.py` and retrained the confidence calibrator on the richer teacher set. The resulting logistic model is still degenerate (`tau=0.10`, predicts nearly everything positive), so these channels are not yet sufficient to rescue the learned gate.
- Added region-level teacher export to `scripts/oracle_teacher.py` and a first `scripts/region_calibrate.py` trainer that fits a simple region stump from connected-component overlap labels.
- Added optional region-model scoring to the calibrated merge path via `--region-model` plus strong/weak region-score thresholds, with runtime receipts now recording region-score settings and strong/weak kept-region counts.
- Validated the first region-model runtime on the aligned house segment. It regressed against the tuned hard region-filter baseline (`candidate IoU ~0.022`, `promoted IoU ~0.020` vs baseline `~0.0269` / `~0.0237`), so region-model scoring is wired but not yet a better default.
- Added a cheap one-step morphological candidate expansion to the calibrated SGBM path before region filtering. The first constrained `3x3` expansion improved the aligned house-segment compare to roughly `candidate IoU ~0.027` and `promoted IoU ~0.026`, outperforming the earlier edge-aware and region-model runs while keeping the implementation O(N).
- Added CLI controls for the cheap SGBM expansion support thresholds (`--expand-texture-min`, `--expand-lr-max`, `--expand-median-max`) and ran a first 5-point sweep. The default support thresholds remained the best tested promoted-IoU tradeoff; looser settings increased coverage but degraded overlap, and tighter LR/median thresholds generally hurt promoted IoU.
- Tried a hotspot-targeted interior horizontal expansion pass on top of the cheap SGBM expansion. It regressed the aligned compare (`candidate IoU ~0.023`, `promoted IoU ~0.022`), so the current default remains the simple constrained `3x3` expansion rather than a wider interior growth rule.
- Added a guarded downstream voxel prototype:
  - `scripts/voxel_guard.py` provides deterministic DDA traversal plus a minimal `grounded/plateau/ascended` voxel state machine
  - `scripts/promoted_depth_to_voxel.py` projects promoted runtime pixels into candidate voxel hits and guards them into plateau/ascended voxel states
  - first validated run on `outputs/runtime_npbi_expand24` produced `outputs/voxel_expand24/voxel_state.npz`, `voxel_summary.json`, and PLY previews with `137` plateau voxels and `195` ascended voxels
- Added lossless promoted-depth exports to the runtime (`promoted_depth_f*.npz` and `promoted_depth.npz`) containing canonical disparity, promoted mask, and depth arrays. Updated `scripts/promoted_depth_to_voxel.py` to prefer those NPZ artifacts over the older PNG approximation path. First NPZ-backed run on `outputs/runtime_npbi_expand24_lossless` produced `35` plateau voxels and `47` ascended voxels.
- Reworked the downstream voxel prototype to use the exact guarded accumulation equations:
  - per-voxel evidence `e_t(v)` is now accumulated from `c_t * g_t * o_t * r_t`
  - temporal evidence/hit state uses `E_t` / `H_t` with persistence scoring `S_t`
  - the guard now uses `grounded` / `plateau` / `ascended` plus the residual constraint `rho_t(v) <= epsilon_rho`
  - the lossless 24-frame validation run on `outputs/runtime_npbi_expand24_lossless` at `stride=16` produced `23` plateau voxels and `26` ascended voxels in `outputs/voxel_expand24_exact24`
- Added `scripts/voxel_quality.py` to verify the ascended voxel set against the promoted point cloud. The first exact-24 validation run reported ascended residuals slightly worse than plateau (`0.3379` vs `0.3145`), so the next tuning step is to try `h_a` before lowering `tau_a`.
