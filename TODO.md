# TODO

- Stand up Retinify locally and confirm it runs on the available GPU; note VRAM headroom.
- Capture or source two evaluation sets: (a) fixed rig dog clip; (b) dual-phone dog clip with drifting viewpoints.
- Prototype Regime A loop per `plan.md`: calibration/rectification → disparity with delta gating → instrumentation (FPS/latency/pixel coverage).
- Instrumented profiling: log per-stage timings and delta-mask coverage at 4K.
- Draft failure-handling heuristics for Regime B (low overlap, desync, rolling shutter) and choose refresh policy for extrinsics drift.
- Decide when to add `CHANGELOG.md` (once first code lands) and keep notes ready.
- Wire Vulkan path via ../dashiCORE: ingest (nv12_to_r8/rgba) → motion mask → warp/rectify (warp_affine_2d) → disparity kernel (to be authored) → write_image preview.
- Add a minimal stereo disparity compute shader (block or census) compatible with the existing Vulkan dispatcher layouts.
- Author SPIR-V kernels per `docs/kernel_specs.md`: frame_diff (motion mask), stereo_roi (census WTA), splat_render (atomic depth/min compositing); reuse existing warp/reduce kernels.
