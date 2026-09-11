# animalexic

Experiments in human:animal communication, embodied observation, governed multimodal inference, and time-indexed multi-camera geometry.

Animalexic uses candidate/promote/abstain/reject semantics across observation layers. The current Drosophila cross-pollination applies the same governance to connectome, registered functional imaging, effector/body state, behavioural motifs, same-trial provenance dependence, and consumer-indexed evidence promotion.

## Stereo and multi-camera geometry

The geometry runtime has two related regimes:

- **Regime A — fixed/SBS stereo:** calibration/rectification -> motion-gated disparity -> governed depth -> voxel/surfel accumulation.
- **Regime B — opportunistic multi-view / handheld:** per-camera visual-inertial trajectory -> cross-camera world-frame weld -> time alignment -> governed multi-view geometry.

The controlled bridge from A to B is the four-camera Pixeltovoxelprojector Issue #20 dataset. `scripts/issue20_multiview.py` consumes known camera poses and emits candidate cross-camera voxel evidence without bypassing governance.

For handheld cameras, `scripts/camera_pose_fibre.py` treats camera pose as a **time-indexed candidate fibre**, not a one-off constant. Static-scene visual tracks may support pose; moving animal pixels are excluded by default from pose evidence. IMU measurements are priors rather than authoritative camera pose. Metric scale, clock alignment, and rolling-shutter uncertainty remain explicit debt until paid.

The intended Regime-B ladder is:

```text
known-pose multicam
  -> perturbed known pose
  -> image-recovered relative pose
  -> visual-inertial per-camera trajectory
  -> cross-camera shared-world weld
  -> clock / rolling-shutter refinement
  -> fully handheld multicam fusion
```

Pose adequacy is consumer-indexed: a pose can be sufficient for coarse ray/voxel intersection while still being insufficient for fine surfel fusion or body-pose reconstruction. See `plan.md` and `docs/IR.md` for the existing `G_t` geometry state and promotion boundary.
