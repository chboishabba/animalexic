# animalexic

Experiments in human:animal communication, embodied observation, governed multimodal inference, and time-indexed multi-camera geometry.

Animalexic uses candidate/promote/abstain/reject semantics across observation layers. The current Drosophila cross-pollination applies the same governance to connectome, registered functional imaging, effector/body state, behavioural motifs, same-trial provenance dependence, and consumer-indexed evidence promotion.

## Stereo and multi-camera geometry

The geometry runtime has two related regimes:

- **Regime A — fixed/SBS stereo:** calibration/rectification -> motion-gated disparity -> governed depth -> voxel/surfel accumulation.
- **Regime B — opportunistic multi-view / handheld:** per-camera visual-inertial trajectory -> cross-camera world-frame weld -> time alignment -> governed multi-view geometry.

The controlled bridge from A to B is the four-camera Pixeltovoxelprojector Issue #20 dataset. `scripts/issue20_multiview.py` consumes known camera poses and emits candidate cross-camera voxel evidence without bypassing governance.

For handheld cameras, `scripts/camera_pose_fibre.py` treats camera pose as a **time-indexed candidate fibre**, not a one-off constant. Static-scene visual tracks may support pose; moving animal pixels are excluded by default from pose evidence. IMU measurements are priors rather than authoritative camera pose. Metric scale, clock alignment, and rolling-shutter uncertainty remain explicit debt until paid.

The controlled bridge from exact poses to learned poses is executable:

- `perturb_camera_observation(...)` creates an explicit synthetic pose perturbation while retaining the source observation as ground truth.
- `recover_relative_pose_from_correspondences(...)` recovers calibrated two-view rotation and translation direction from static correspondences while leaving metric scale unpaid unless a named baseline is supplied.
- `materialise_recovered_camera_observation(...)` admits only metric-paid recovered poses into the same `CameraObservation` / ray-projection contract used by the known-pose Issue-20 producer.
- `scripts/issue20_pose_recovery.py` adds the image-facing layer: SIFT matching over pose-admissible static regions, calibrated pair recovery, conversion of known metadata into validation geometry, and post-hoc rotation/translation-direction scoring. Known poses are an oracle only and are not fed into the recovery solve.

The handheld lane now has both an inertial predictor and a bounded visual correction path in `scripts/visual_inertial_pose.py`:

- `preintegrate_imu_prior(...)` integrates timestamped gyro/specific-force measurements into candidate rotation/velocity/position deltas.
- `correct_visual_inertial_segment(...)` converts the inertial increment through a supplied camera<-IMU rigid transform, checks it against a static-scene visual relative pose, applies visual rotation only when the residual gate passes, applies visual metric translation only when scale is explicitly paid, and returns `abstain` on incompatible observations.
- `compose_candidate_trajectory(...)` accumulates accepted corrected intervals into time-indexed local camera keyframes using deterministic SE(3) transport.

Cross-camera alignment now has its own producer in `scripts/cross_camera_world_weld.py`:

- `estimate_world_weld(...)` estimates an `SE(3)` weld from shared static 3D anchors when metric scale is already paid, or an explicit `Sim(3)` weld when scale must remain a coordinate.
- geometrically degenerate/collinear anchor sets fail closed;
- `apply_world_weld_to_trajectory(...)` transports candidate local keyframes into the target shared world without changing their candidate status.

This is **not yet promoted/full VIO or field-ready multicam fusion**. The correction path consumes supplied camera/IMU extrinsic and clock-alignment receipts; it does not yet estimate those quantities online. The world weld currently assumes already-associated same-object static anchors and has no robust outlier/temporal association layer. Online bias estimation, multi-keyframe optimization, loop closure, rolling-shutter correction, real-phone validation, and downstream shared-world voxel/surfel validation remain unpaid.

This is **consumer-contract parity, not evidence parity**: known metadata and image-recovered pose may feed the same downstream ray/voxel/surfel machinery once metric scale is paid, but their provenance, uncertainty, validation status, and remaining debt stay distinct.

The Regime-B ladder is now:

```text
known-pose multicam                         [implemented]
  -> controlled perturbed known pose       [implemented]
  -> image-recovered relative pose         [implemented; rendered-image validation]
  -> Issue-20 image-pair recovery operator [implemented; archive execution blocked on binary acquisition]
  -> IMU preintegration prior              [implemented]
  -> bounded visual-inertial correction    [implemented; synthetic validation]
  -> candidate local camera trajectory     [implemented; deterministic composition]
  -> rigid/similarity cross-camera weld    [implemented; synthetic validation]
  -> static-anchor association/outliers    [unpaid]
  -> online extrinsic/clock/bias estimation[unpaid]
  -> multi-keyframe optimized VIO          [unpaid]
  -> shared-world voxel/surfel validation  [unpaid]
  -> rolling-shutter refinement            [unpaid]
  -> fully handheld multicam fusion        [unpaid]
```

Pose adequacy is consumer-indexed: a pose can be sufficient for coarse ray/voxel intersection while still being insufficient for fine surfel fusion or body-pose reconstruction. See `plan.md` and `docs/IR.md` for the existing `G_t` geometry state and promotion boundary.
