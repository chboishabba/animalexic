# COMPACTIFIED_CONTEXT

- Source chat: "Stereo 3D 4K60 Pipeline"
- Online thread UUID: 69bce770-5540-8398-9257-b1c7da6a1d11
- Canonical thread ID: df0976e7ec5aabd4809189993f76cf4be8331b0b
- Source: local archive (`pull_to_structurer.py` → `chat_context_resolver.py`, 2026-03-25)
- Chat window: 2026-03-20 06:22:06–06:40:15 UTC

## Main points pulled from the thread

- Two operating regimes to design for from day one:
  - **Regime A — fixed stereo rig**: calibrate + rectify once, treat extrinsics as persistent state, run depth/correspondence per frame; this is the plausible path to live 4K60.
  - **Regime B — opportunistic multi-view (phones)**: handle time alignment, viewpoint alignment, pose recovery, triangulation/fusion, rolling-shutter and lens differences; expect lower throughput and higher algorithmic complexity.
- Architecture should be two-timescale: slow layer maintains intrinsics/extrinsics; fast layer does per-frame disparity/flow/temporal updates under fixed geometry.
- Throughput tricks: process frame deltas/changed regions instead of full frames; keep extrinsics as slow-changing latent constants; prefer GPU-first, classical or lightweight stereo over heavy NeRF/diffusion.
- Use the “change map” idea (frame_t - frame_{t-1}) to limit work to motion regions, especially when scenes are mostly static.

## Open questions / follow-ups

- Pick concrete stereo matcher/library (e.g., Retinify or similar) and target GPU budget for 4K60.
- Define evaluation clips (fixed-rig kennel/room + dual-phone opportunistic captures).
- Decide how to persist and refresh extrinsics for Regime B (frequency, confidence, and failure handling).
- Capture benchmarks for delta-processing savings vs full-frame stereo at 4K.
