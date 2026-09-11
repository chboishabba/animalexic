# animalexic

Experiments in human:animal communication, embodied observation, and governed multimodal inference.

The current Drosophila cross-pollination reuses Animalexic's candidate/promote/abstain/reject semantics for connectome, registered functional imaging, effector/body state, behavioural motifs, same-trial provenance dependence, and consumer-indexed evidence promotion.

## Governed state-space trajectories

`state_space_trajectory_adapter.py` exposes the same runtime governance to downstream temporal/manifold visualisation. A trajectory row carries time, embedded coordinates, channel, provenance, decision state, source identity, residual, and receipt identity. Candidate, abstained, and rejected observations remain inspectable but do not silently become canonical geometry; promoted-only export is the default.

This provides a runtime ABI for DASHI's birdsong/fly state-space visualisation work without changing Animalexic's core rule: fast producers may propose observations, while canonical state mutation requires explicit governance and an append-only receipt. Visual proximity is not anatomical/physical proximity, recurrence is not semantic meaning, and a functional trace identity is not a connectome-neuron identity unless separately receipted.

### Real published birdsong producer

`scripts/arese_shared_manifold_adapter.py` imports the processed CSV outputs associated with Lucio Arese, *Shared acoustic manifolds for exploratory comparison of passerine vocalizations* (DOI `10.32942/X2W65N`; data DOI `10.5281/zenodo.18332166`). The adapter preserves exact source-file SHA-256 and row-level provenance, discovers common UMAP coordinate-column spellings fail-closed, and emits the same governed trajectory ABI.

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

The adapter treats each timepoint as a 940-dimensional functional observation, mean-centres the selected-ROI coordinates, and computes a downstream PCA-3 trajectory for visual inspection. This PCA is an Animalexic/DASHI visual diagnostic, not a claim made by the Gauthey paper. Output rows are candidate-only and retain the source hash, DOI, archive member, projection identity, and the unresolved registration boundary.

Important: the source pickle loader is opt-in via `--trusted-pickle` because pickle can execute arbitrary code. Verify source custody/checksum first. The adapter also accepts non-pickle `.npy` or single-array `.npz` forms with the same exact 940x668 shape.

Example after extracting/verifying the pinned member:

```bash
python scripts/gauthey_functional_trajectory_adapter.py \
  path/to/dffs_audio_2p_corr_top05_all.pkl --trusted-pickle \
  --out outputs/gauthey_2p_pca3.csv \
  --trajectory-receipt outputs/gauthey_2p_pca3.trajectory.json \
  --source-receipt outputs/gauthey_2p_pca3.source.json
```

The standing firewalls are: selected ROI row != MaleCNS neuron; PCA coordinate != anatomical coordinate; visual recurrence != same neuron population; functional activation != causal necessity; published dataset != Animalexic promotion receipt.
