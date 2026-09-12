# Regime B — Live Multicamera / Handheld Roadmap

This document is the current execution frontier for the opportunistic multi-view / handheld path described in `plan.md`. It distinguishes implementation, controlled synthetic receipts, explicit acceptance/payment, and real empirical validation.

## Current state

| Layer | Status | Boundary |
| --- | --- | --- |
| Known-pose multicamera carrier | implemented | Issue-20 metadata -> `CameraObservation` |
| Controlled pose perturbation | implemented | original metadata remains oracle only |
| Image relative-pose recovery | implemented | static correspondences -> candidate `R, t_hat` |
| Metric scale gate | implemented | scale-free pose cannot enter metric world |
| Issue-20 image-pair operator | implemented | archive bytes still unavailable here |
| IMU preintegration | implemented | candidate inertial prior |
| Residual-gated visual-inertial segment correction | implemented | incompatible observers abstain |
| Candidate local trajectory composition | implemented | deterministic `SE(3)` transport only |
| Clock-offset candidate | implemented | overlap/residual-margin gate; ambiguous motion abstains |
| Stationary gyro-bias candidate | implemented | per-axis noise gate |
| Camera<-IMU rotation candidate | implemented | rank/excitation + residual gate |
| Camera<-IMU lever-arm candidate | implemented | `AX=XB` translation solve; full-rank rotational excitation required |
| Gravity-referenced accelerometer-bias candidate | implemented | expected rest force is explicit `-R^T g` |
| Calibration candidate acceptance seams | partially implemented | clock, gyro bias, camera/IMU rotation explicit receipts; lever-arm/accel field acceptance remains debt |
| Translation pose graph | implemented | anchored global least squares; loop residual gate |
| Rotation pose graph | implemented | anchored chordal `SO(3)` averaging; loop residual gate |
| Pose-graph -> existing trajectory carrier | implemented | representation adapter; no joint optimization claim |
| Fixed-rotation visual-inertial P/V smoother | implemented | one global visual+inertial linear solve with rotations fixed |
| Full nonlinear VIO / BA | unpaid | rotations, P/V, biases and calibration are not jointly optimized |
| Cross-camera `SE(3)` / `Sim(3)` weld | implemented | candidate shared-world transform |
| Static-anchor association with explicit IDs | implemented | static/confidence/time/provenance gates |
| Robust weld outlier consensus/refit | implemented | inlier/outlier anchor identity retained |
| Descriptor anchor proposals | implemented candidate-only | mutual/distinctive similarity never pays identity |
| Temporal descriptor tracks | implemented candidate-only | continuity never pays identity |
| Exact same-object payment seam | implemented | exact feature IDs + provenance + external receipt required |
| Per-ray world camera origins in voxel guard | implemented | zero-origin SBS fallback retained |
| World-weld -> existing guard adapter | implemented | no second geometry governance path |
| Guard transport comparison/frontier | implemented | agreement, ascended IoU, score residual, changed-state mask, ternary frontier |
| Controlled guard sensitivity portfolio | executed in exact-content mirror | synthetic only; not Issue-20/phone evidence |
| Pareto refinement frontier | implemented | no weighted scalar collapse |
| Quality-targeted geometry refinement | implemented | terminates `within_policy`, `consumer_plateau`, or `max_steps` |
| Rolling-shutter row-time pose transport | implemented | candidate row pose from supplied readout model |
| Rolling-shutter readout time/direction candidate | implemented | row-span + timing-residual gates |
| Rolling-shutter readout acceptance | implemented | exact candidate-reference receipt required |
| Shared-world voxel/surfel field-quality validation | unpaid | end-to-end real carrier still missing |
| Real dual-phone / N-phone validation | unpaid | required before handheld claims |

## Controlled receipts now available

The synthetic guard sensitivity probe compares the same guarded consumer under isolated geometry defects. Under its explicit synthetic policy, the oracle is exact while 25 cm origin error, 10 deg yaw, +10% scale, and +5 observation residual all violate consumer policy. The Pareto-maximal defect in that carrier is the yaw/orientation fibre. Halving yaw eventually hits a quantized guard plateau: producer error keeps shrinking while three guard cells remain changed, so the refinement terminates `consumer_plateau` rather than inventing closure.

The camera/IMU hand-eye translation probe recovers a known `(0.12,-0.04,0.08) m` lever arm from a rank-3 stacked system to floating-point error; pure translation has rank 0 and is rejected. Translation and rotation pose-graph probes likewise separate consistent loops from inconsistent loops by explicit consumer residual gates.

These are controlled implementation receipts, not field tolerances.

## dashiRTX cross-pollination

The sibling `chboishabba/dashiRTX` work contributes a structural pattern, not rendering truth for Animalexic:

```text
transport -> consumer error/frontier -> localise active fibre -> targeted refinement
```

Animalexic applies that pattern to camera/world geometry:

```text
known-pose oracle guard
  -> candidate pose/world-weld transport
  -> existing voxel guard
  -> state/score residual
  -> signed {-1,0,+1} frontier
  -> refine only the geometry fibre visible to the consumer
```

No radiance semantics or MDL score is promoted into camera/world truth.

## Highest-alpha remaining sequence

1. **Acquire/execute the Issue-20 binary archive.** Run known pose, recovered pose and robust-welded pose through the exact same guard and replace synthetic sensitivity coordinates with dataset receipts.
2. **Run a real two-phone capture with frame + IMU logs.** Exercise learned clock, camera/IMU rotation, lever arm, gyro/accel bias and rolling-shutter candidates against a real carrier.
3. **Measure shared-world voxel and surfel degradation.** Preserve the dashiRTX-style consumer frontier and attribute failures back to origin/orientation/scale/timing/weld fibres.
4. **Only if the real carrier requires it, add joint nonlinear VIO/BA.** The current bounded graph/smoother stack intentionally stops short of claiming full VIO.
5. **Then generalise from two phones to N heterogeneous cameras.** Keep weak overlap, independent clocks, rolling shutter and scale debt explicit.

## Non-collapse rules

- descriptor similarity != same-object identity
- temporal continuity != same-object identity
- low weld residual != correct identity
- `Sim(3)` alignment != metric scale paid
- calibration candidate != accepted calibration
- accepted calibration != field validation
- translation/rotation pose graphs != full VIO
- fixed-rotation P/V smoother != nonlinear BA/SLAM
- world-welded camera origin != promoted voxel
- lower guard transport residual != correct pose
- better ascended IoU != physical truth
- smaller producer perturbation != consumer closure
- dashiRTX lower MDL/render error != Animalexic geometry truth
- successful synthetic recovery != Issue-20 archive validation
- successful Issue-20 validation != handheld-phone validation

## Hard blockers in this environment

- The contributed Issue-20 `test_data.tar.xz` identity is known, but the current connector cannot materialize the binary archive. Treat this as acquisition debt, not algorithmic debt.
- `animalexic` has no GitHub workflow directory on this branch, so pushes do not currently yield a branch CI receipt here.
- Container DNS cannot clone/download the GitHub branch, so focused Python checks have used exact-content mirrors or standalone equation probes rather than a true checkout.
- A real dual-phone capture with synchronized frame/IMU logs is not available in this session.
- The Agda owners are source-wired in `dashi_agda`, but this execution environment has no fresh Agda kernel receipt for these new owners.
- The `dashi_agda` Animalexic branch has diverged from newer `master`; reconciliation should occur before treating aggregate compilation against current master as paid.
