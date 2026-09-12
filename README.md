# animalexic

Experiments in human:animal communication, embodied observation, governed multimodal inference, and time-indexed multi-camera geometry.

Animalexic uses candidate/promote/abstain/reject semantics across observation layers. The current Drosophila cross-pollination applies the same governance to connectome, registered functional imaging, effector/body state, behavioural motifs, same-trial provenance dependence, and consumer-indexed evidence promotion.

## Stereo and multi-camera geometry

The geometry runtime has two related regimes:

- **Regime A — fixed/SBS stereo:** calibration/rectification -> motion-gated disparity -> governed depth -> voxel/surfel accumulation.
- **Regime B — opportunistic multi-view / handheld:** per-camera visual-inertial trajectory -> cross-camera world-frame weld -> time alignment -> governed multi-view geometry.

The controlled bridge from A to B is the four-camera Pixeltovoxelprojector Issue #20 dataset. `scripts/issue20_multiview.py` consumes known camera poses and emits candidate cross-camera voxel evidence without bypassing governance.

For handheld cameras, `scripts/camera_pose_fibre.py` treats camera pose as a **time-indexed candidate fibre**, not a one-off constant. Static-scene visual tracks may support pose; moving animal pixels are excluded by default from pose evidence. IMU measurements are priors rather than authoritative camera pose. Metric scale, clock alignment, and rolling-shutter uncertainty remain explicit debt until paid.

The same module now also owns the controlled bridge from exact poses to learned poses:

- `perturb_camera_observation(...)` creates an explicit synthetic pose perturbation while retaining the source observation as ground truth.
- `recover_relative_pose_from_correspondences(...)` recovers calibrated two-view rotation and translation direction from static correspondences while leaving metric scale unpaid unless a named baseline is supplied.
- `materialise_recovered_camera_observation(...)` admits only metric-paid recovered poses into the same `CameraObservation` / ray-projection contract used by the known-pose Issue-20 producer.

This is **consumer-contract parity, not evidence parity**: known metadata and image-recovered pose may feed the same downstream ray/voxel/surfel machinery, but their provenance, uncertainty, validation status, and remaining debt stay distinct.

The Regime-B ladder is now:

```text
known-pose multicam                         [implemented]
  -> controlled perturbed known pose       [implemented]
  -> image-recovered relative pose         [implemented; synthetic probe]
  -> real Issue-20 archive recovery        [validation debt]
  -> visual-inertial per-camera trajectory [next implementation rung]
  -> cross-camera shared-world weld        [unpaid]
  -> clock / rolling-shutter refinement    [unpaid]
  -> fully handheld multicam fusion        [unpaid]
```

Pose adequacy is consumer-indexed: a pose can be sufficient for coarse ray/voxel intersection while still being insufficient for fine surfel fusion or body-pose reconstruction. See `plan.md` and `docs/IR.md` for the existing `G_t` geometry state and promotion boundary.
