# Spec

## Objective
Build a GPU-first stereo-to-3D pipeline with two explicit operating regimes:

- Regime A: fixed stereo rig targeting live 4K60 on a stable camera pair.
- Regime B: opportunistic multi-view phone capture of a moving dog, where robustness matters more than throughput.

## Source Context
- Chat title: `Stereo 3D 4K60 Pipeline`
- Online UUID: `69bce770-5540-8398-9257-b1c7da6a1d11`
- Canonical thread ID: `df0976e7ec5aabd4809189993f76cf4be8331b0b`
- Source used: `db` (`~/chat_archive.sqlite`)
- Refreshed: `2026-03-26`
- Companion chat title: `ZKP Framing Analysis`
- Companion online UUID: `69c498c6-9514-839b-bce3-583cb6c168e5`
- Companion canonical thread ID: `e73a07606b58ec64b19007e3230e459e1283204e`
- Companion source used: `db` (`~/chat_archive.sqlite`)
- Companion refreshed: `2026-03-27`

## Functional Requirements
- For Regime A, support calibration, rectification, disparity/depth estimation, and governed promotion into canonical state.
- Prefer motion-gated delta ROI processing by default, with an explicit full-frame override for static scenes.
- Persist calibration/extrinsics as slow-changing state instead of recomputing geometry every frame.
- Record promotion, abstention, and rejection decisions in an auditable receipt store.
- Support self-calibration as the default end-user path, with board-based calibration as an optional higher-quality route.
- Maintain an oracle comparison path so runtime behavior can be tuned against a calibrated OpenCV reference.

## Non-Goals
- Treating uncalibrated demo output as metric depth.
- Recomputing full global geometry every frame in the fixed-rig regime.
- Depending on heavy NeRF or diffusion systems for the real-time path.

## Success Criteria
- Regime A reaches a plausible path toward sustained `>=60 fps` at `4K` per eye with latency below `50 ms`.
- Runtime outputs remain governance-backed: candidate generation stays separate from promoted canonical state.
- Local calibrated validation is stable on real stereo clips, not only synthetic or YouTube SBS demos.
- Oracle-vs-runtime artifacts are available to drive matcher and promotion tuning.
- Runtime evaluation is measured primarily by oracle agreement (`IoU`, false negatives, false positives), not raw coverage alone.

## Current Decisions
- OpenCV SBS oracle is the baseline reference path and system sanity check.
- Calibrated local validation is the current execution milestone.
- Retinify remains a later benchmark/comparison candidate, not the current system-of-record.
- Vulkan integration should reuse existing assets from `../dashiCORE` for ingest, masking, warp, and preview, while authoring stereo-specific kernels locally.
- The current problem is candidate placement / oracle agreement, not promotion collapse.
- Learned confidence calibration remains experimental and opt-in only until it beats the heuristic decomposed-evidence baseline on aligned compare metrics.
