# Regime B — Live Multicamera / Handheld Roadmap

This document is the current execution frontier for the opportunistic multi-view / handheld path described in `plan.md`. It is deliberately narrower than the historical plan and should be updated as receipts move from implementation to empirical validation.

## Current state

| Layer | Status | Boundary |
| --- | --- | --- |
| Known-pose multicamera carrier | implemented | Issue-20 metadata -> `CameraObservation` |
| Controlled pose perturbation | implemented | experiment input only; original metadata remains ground truth |
| Image relative-pose recovery | implemented | static visual correspondences -> candidate `R, t_hat` |
| Metric scale gate | implemented | recovered pose cannot enter metric world without explicit scale receipt |
| Issue-20 image-pair operator | implemented | archive execution still blocked by binary acquisition in current environment |
| IMU preintegration | implemented | candidate inertial prior only |
| Visual-inertial segment correction | implemented | residual-gated correction; may abstain |
| Local camera trajectory composition | implemented | deterministic candidate `SE(3)` composition |
| Cross-camera `SE(3)` / `Sim(3)` weld | implemented | candidate shared-world transform |
| Static-anchor association | implemented for explicit upstream IDs | exact `anchor_id`, static/confidence/time gates, provenance retained |
| Robust weld outlier rejection | implemented | deterministic minimal-set consensus + refit |
| Per-ray world camera origins in voxel guard | implemented | optional `(N,3)` origin fibre; zero-origin SBS fallback preserved |
| World-weld -> existing guard adapter | implemented | world keyframes + observations -> existing frame points/origins/weights/residuals contract |
| Guard transport comparison/frontier | implemented | state agreement, ascended IoU, score residual, change mask, signed `{-1,0,+1}` frontier |
| Known/perturbed/recovered/welded sensitivity run | unpaid | comparison surface exists; controlled end-to-end guard runs remain to execute |
| Descriptor/track anchor identity discovery | unpaid | similarity cannot auto-promote to same-object identity |
| Online camera/IMU extrinsic estimation | unpaid | current correction consumes supplied extrinsic |
| Camera/IMU clock-offset estimation | unpaid | current correction consumes supplied alignment receipt |
| Online IMU bias estimation | unpaid | biases are explicit input coordinates |
| Multi-keyframe optimized VIO / loop closure | unpaid | candidate composition is not BA/SLAM |
| Shared-world voxel/surfel quality validation | unpaid | origin carrier/adapter exists; end-to-end quality receipt does not |
| Rolling-shutter refinement | unpaid | explicit debt remains in `CameraModel` |
| Real dual-phone / N-phone field validation | unpaid | required before handheld claims |

## dashiRTX cross-pollination

The sibling `chboishabba/dashiRTX` work provides a useful architecture, not a replacement geometry model. Its PDA/MDL light-transport toy explicitly transports depth/radiance between camera poses, measures reprojection error, retains a signed ternary frontier, importance and persistent state, and targets refresh/refinement where transport error is consumer-visible. Its roadmap similarly prioritizes quality-targeted `render -> error -> refine -> retrain` loops.

Animalexic now reuses that pattern over geometry only:

```text
known-pose oracle guard
  -> candidate pose/world-weld transport
  -> existing voxel guard
  -> state/score residual
  -> signed {-1,0,+1} frontier
  -> refine the pose/weld fibre causing consumer-visible error
```

This does **not** import radiance semantics, MDL scores as truth, or dashiRTX rendering authority. It is a cross-domain transport/refinement pattern only.

## Highest-alpha next sequence

1. **Execute the controlled guard sensitivity portfolio.** Run the same world observations through known pose, controlled perturbations, image-recovered pose, and robust-welded pose. Measure ascended IoU, state agreement, score residual and the signed frontier.
2. **Attribute the frontier to the producer fibre.** Separate camera-origin error, orientation error, world-weld residual, scale error and observation residual instead of optimizing one undifferentiated geometry score.
3. **Quality-targeted pose/weld refinement.** Borrow the dashiRTX refinement discipline: refine only fibres that change the downstream consumer surface; stop when the consumer residual is within its policy bound rather than demanding globally perfect pose.
4. **Anchor identity discovery as a candidate producer.** Add static feature/track proposals with explicit ambiguity and provenance. Descriptor or geometric similarity may propose identity; only a governed same-object receipt may pay it.
5. **Temporal anchor tracks.** Preserve anchor identity across time and reject identity switches before cross-camera welding.
6. **Estimate clock/extrinsic/bias coordinates.** Move supplied VIO calibration coordinates into learned candidates one at a time, each with a residual-based abstention route.
7. **Real phone capture.** Two unsynchronised phones first; then N cameras of heterogeneous type. Keep rolling shutter and weak overlap as explicit degradation coordinates.

## Non-collapse rules

- descriptor similarity != same-object identity
- temporal proximity != same-object identity
- low weld residual != correct identity
- `Sim(3)` alignment != metric scale paid
- corrected candidate trajectory != promoted VIO
- world-welded camera origin != promoted voxel
- lower guard transport residual != correct pose
- better ascended IoU != physical truth
- dashiRTX lower MDL / render error != Animalexic geometry truth
- successful synthetic recovery != Issue-20 archive validation
- successful Issue-20 validation != handheld-phone validation

## Empirical blockers

- The contributed Issue-20 `test_data.tar.xz` identity is known, but this execution environment cannot materialize the binary archive through the current connector. Treat this as acquisition debt, not implementation debt.
- A real dual-phone capture with synchronized frame/IMU logs is still required for field validation.
- Agda owners are source-wired in `dashi_agda`; a fresh Agda kernel receipt still requires an Agda-capable execution environment.
