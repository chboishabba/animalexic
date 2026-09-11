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
