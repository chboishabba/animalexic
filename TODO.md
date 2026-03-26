# TODO

- Stand up Retinify locally and confirm it runs on the available GPU; note VRAM headroom.
- Capture or source two evaluation sets: (a) fixed rig dog clip; (b) dual-phone dog clip with drifting viewpoints.
- Generate a calibration artifact and validate rectification on a local stereo clip. Prefer self-calibration for end-user flow; keep ChArUco as an optional higher-quality path.
- Prototype Regime A loop per `plan.md`: calibration/rectification → disparity with delta gating → instrumentation (FPS/latency/pixel coverage). (temporal merge/promotion/accumulation added; demo/relative 3D milestone reached; next confirm calibrated local stability)
- Instrumented profiling: log per-stage timings and delta-mask coverage at 4K.
- Validate the new calibration artifact schema (`fixed_rig_calibration_v1`) and prefer `--merge-profile calibrated` for local artifact-backed runs.
- Validate the self-calibration artifact flow (`fixed_rig_selfcal_v1`) on a local stereo pair or SBS frame and measure how well rectification improves coverage.
- Tune temporal merge thresholds (cost/gap/close-disp/age) on clean SBS CGI and a real fixed-rig clip; document preferred defaults.
- Tune evidence accumulation parameters (min_evidence_frames, weak_conf_scale, decay) for relative 3D stability.
- Draft failure-handling heuristics for Regime B (low overlap, desync, rolling shutter) and choose refresh policy for extrinsics drift.
- Decide when to add `CHANGELOG.md` (once first code lands) and keep notes ready.
- Wire Vulkan path via ../dashiCORE: ingest (nv12_to_r8/rgba) → motion mask → warp/rectify (warp_affine_2d) → disparity kernel (to be authored) → write_image preview.
- Add a minimal stereo disparity compute shader (block or census) compatible with the existing Vulkan dispatcher layouts.
- Author SPIR-V kernels per `docs/kernel_specs.md`: frame_diff (motion mask), stereo_roi (census WTA), splat_render (atomic depth/min compositing); reuse existing warp/reduce kernels.
