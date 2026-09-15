# animalexic

Experiments in human:animal communication, embodied observation, and governed multimodal inference.

The current Drosophila cross-pollination reuses Animalexic's candidate/promote/abstain/reject semantics for connectome, registered functional imaging, effector/body state, behavioural motifs, same-trial provenance dependence, and consumer-indexed evidence promotion.

## Governed state-space trajectories

`state_space_trajectory_adapter.py` exposes the same runtime governance to downstream temporal/manifold visualisation. A trajectory row carries time, embedded coordinates, channel, provenance, decision state, source identity, residual, and receipt identity. Candidate, abstained, and rejected observations remain inspectable but do not silently become canonical geometry; promoted-only export is the default.

This provides a runtime ABI for DASHI's birdsong/fly state-space visualisation work without changing Animalexic's core rule: fast producers may propose observations, while canonical state mutation requires explicit governance and an append-only receipt. Visual proximity is not anatomical/physical proximity, recurrence is not semantic meaning, and a functional trace identity is not a connectome-neuron identity unless separately receipted.

### Real published birdsong producer

`scripts/arese_shared_manifold_adapter.py` imports the processed CSV outputs associated with Lucio Arese, *Shared acoustic manifolds for exploratory comparison of passerine vocalizations* (EcoEvoRxiv preprint v4; DOI `10.32942/X2W65N`; data DOI `10.5281/zenodo.18332166`). The adapter preserves exact source-file SHA-256 and row-level provenance, discovers common UMAP coordinate-column spellings fail-closed, and emits the same governed trajectory ABI.

The current v4 paper pipeline is recorded as source metadata rather than silently recomputed: 120-D MFCC (`40 + delta + delta-delta`) or 80-D chroma -> PCA-20 -> UMAP-3D, with `n_neighbors=30`, `min_dist=0.1`, seed `42`; MFCC uses Euclidean distance and chroma cosine. RMS, spectral centroid, and CEC are retained as visualization-overlay semantics, not embedding inputs.

Published Zenodo rows enter Animalexic as **candidate-only** observations. The importer intentionally uses non-promoted diagnostic export so they can be inspected/rendered without being mistaken for canonical Animalexic state. A later promotion requires a separate runtime governance receipt; publication or a DOI cannot pay that obligation.

Example once a Zenodo CSV is local:

```bash
python scripts/arese_shared_manifold_adapter.py path/to/manifold.csv \
  --species eurasian-wren --feature-space mfcc \
  --out outputs/arese_wren_mfcc.csv \
  --trajectory-receipt outputs/arese_wren_mfcc.trajectory.json \
  --source-receipt outputs/arese_wren_mfcc.source.json
```

### Real Drosophila functional producer

`scripts/gauthey_functional_trajectory_adapter.py` consumes the repo-pinned preprocessed functional carrier for Wayan Gauthey et al., *High-speed whole-brain imaging in Drosophila* (DOI `10.1038/s41467-026-72437-1`; preprocessed-data DOI `10.5281/zenodo.17618684`). The expected pooled 2p member is `Data/Dffs/Audio correlated/dffs_audio_2p_corr_top05_all.pkl` with shape `940 x 668`, interpreted as **940 selected ROI rows x 668 time samples**.

The public source-code archaeology now pays the common conventional-2P protocol timebase rather than leaving time as a guessed CLI parameter. `Fig3_aligment.py` aligns every source trial through the sound-server frame and I2C clock and retains aligned activity/audio times. `fig3_preprocessing.py` then truncates four named 47,000-row trials to 668 samples at `2.20337115787 Hz`, constructs the thirteen stimulus blocks on that aligned timebase, stacks ROI rows and selects the top 0.5% stimulus-correlated rows. The adapter records the exact public repository tree, alignment/preprocessing blobs, stimulus file/blob, source trial order and block schedule.

The adapter treats each timepoint as a 940-dimensional functional observation, mean-centres the selected-ROI coordinates, and computes a downstream PCA-3 trajectory for visual inspection. This PCA is an Animalexic/DASHI visual diagnostic, not a claim made by the Gauthey paper. Output rows are candidate-only and retain the source hash, DOI, archive member, projection identity, stimulus-block coordinate and unresolved row/anatomical registration boundary.

Important: the source pickle loader is opt-in via `--trusted-pickle` because pickle can execute arbitrary code. Verify source custody/checksum first. The adapter also accepts non-pickle `.npy` or single-array `.npz` forms with the same exact 940x668 shape.

Example after extracting/verifying the pinned member:

```bash
python scripts/gauthey_functional_trajectory_adapter.py \
  path/to/dffs_audio_2p_corr_top05_all.pkl --trusted-pickle \
  --out outputs/gauthey_2p_pca3.csv \
  --trajectory-receipt outputs/gauthey_2p_pca3.trajectory.json \
  --source-receipt outputs/gauthey_2p_pca3.source.json
```

Protocol/timebase recovery is deliberately weaker than source-row identity: the compact published pickle contains the selected 940 traces but not the selection indices that identify which source trial/plane/cluster supplied each row.

### Recover pooled selected-row identity

`scripts/gauthey_roi_identity_recovery_adapter.py` implements the current highest-alpha reverse dependency. It reproduces the published preprocessing selection without allocating the full `188000 x 668` stack in memory:

1. load the four canonical aligned conventional-2P trial dictionaries;
2. truncate each `dffs_corrected` matrix to the same 668-sample window;
3. reproduce the source thirteen-block stimulus and GCaMP6f convolution;
4. compute the source-equivalent zero-lag z-normalized correlation per trial;
5. concatenate only the 188,000 scalar correlation values and select the global top 0.5% = 940 rows;
6. map each selected global index to `(trial, local row, plane, cluster)` using 47 planes x 1000 clusters per trial;
7. reconstruct the selected 940x668 carrier in source selection order.

The result remains a **reconstruction candidate** unless the exact compact published matrix is also supplied. With `--compact-matrix`, the producer requires exact `numpy.array_equal` between the reconstructed and published 940x668 carriers. Any mismatch fails closed. Only an exact match pays the narrow claim:

`pooled selected row -> exact source trial / plane / cluster identity`.

It still does **not** pay `source ROI -> MaleCNS neuron`, anatomical registration, causal necessity, or Animalexic promotion.

Example once the exact source trial artifacts and compact carrier are locally available and verified:

```bash
python scripts/gauthey_roi_identity_recovery_adapter.py \
  trial_a2_r2.pkl trial_a2_r3.pkl trial_a2_r4.pkl trial_a1_r2.pkl \
  --trusted-pickle \
  --compact-matrix dffs_audio_2p_corr_top05_all.pkl \
  --trusted-compact-pickle \
  --out outputs/gauthey_2p_source_identity.csv \
  --receipt outputs/gauthey_2p_source_identity.receipt.json
```

The standing firewalls are: protocol alignment != pooled-row source identity; selected ROI row != MaleCNS neuron; PCA coordinate != anatomical coordinate; visual recurrence != same neuron population; functional activation != causal necessity; published dataset != Animalexic promotion receipt.

### Source-free smoke check

`python scripts/state_space_real_adapter_smoke.py` checks the Arese UMAP-column discovery and candidate-only import boundary, the governed decision vocabulary, the fail-closed Gauthey 940x668 shape gate, and the exact source-pinned conventional-2P sampling/block/blob coordinates without downloading or fabricating either scientific dataset.
