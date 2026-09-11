# animalexic

Experiments in human:animal communication, embodied observation, and governed multimodal inference.

The current Drosophila cross-pollination reuses Animalexic's candidate/promote/abstain/reject semantics for connectome, registered functional imaging, effector/body state, behavioural motifs, same-trial provenance dependence, and consumer-indexed evidence promotion.

## Governed state-space trajectories

`state_space_trajectory_adapter.py` exposes the same runtime governance to downstream temporal/manifold visualisation. A trajectory row carries time, embedded coordinates, channel, provenance, decision state, source identity, residual, and receipt identity. Candidate, abstained, and rejected observations remain inspectable but do not silently become canonical geometry; promoted-only export is the default.

This provides a runtime ABI for DASHI's birdsong/fly state-space visualisation work without changing Animalexic's core rule: fast producers may propose observations, while canonical state mutation requires explicit governance and an append-only receipt. Visual proximity is not anatomical/physical proximity, recurrence is not semantic meaning, and a functional trace identity is not a connectome-neuron identity unless separately receipted.
