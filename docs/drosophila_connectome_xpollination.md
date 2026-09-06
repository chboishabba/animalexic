# Drosophila connectome / functional-imaging x-pollination

Animalexic's runtime remains a governed embodied-observation system.  The Drosophila work in `dashi_agda` and `dashiBRAIN` supplies a particularly strong biological instantiation of the same observation/promotion architecture.

## Scientific source attribution rule

Every external scientific source recorded here or in runtime experiment metadata must carry:

1. author or consortium;
2. title; and
3. DOI or another stable identifier when DOI is unavailable.

A URL may be retained as a convenience locator but does not replace a stable identifier.

Canonical sources for this bridge include:

- Berg et al., **Sexual dimorphism in the complete Drosophila male central nervous system connectome**, DOI `10.1016/j.cell.2026.08.015`.
- Bates et al.; BANC-FlyWire Consortium, **Distributed control circuits across a brain-and-cord connectome**, DOI `10.1038/s41586-026-10735-w`.
- Gauthey, Lin, Ahmed, Leifer, Murthy, Thiberge et al., **High-speed whole-brain imaging in Drosophila**, DOI `10.1038/s41467-026-72437-1`.
- Brezovec, Berger, Hao, Lin, Ahmed, Pacheco, Thiberge, Murthy, Clandinin, **BIFROST: A method for registering diverse imaging datasets of the Drosophila brain**, DOI `10.1073/pnas.2322687121`.
- Turner, Mann, Clandinin, **The connectome predicts resting-state functional connectivity across the Drosophila brain**, DOI `10.1016/j.cub.2021.03.004`.
- Azevedo et al., **Connectomic reconstruction of a female Drosophila ventral nerve cord**, DOI `10.1038/s41586-024-07389-x`.
- Lesser, Azevedo, Phelps et al., **Synaptic architecture of leg and wing premotor control networks in Drosophila**, DOI `10.1038/s41586-024-07600-z`.

## State alignment

Animalexic currently uses

```text
S_t = (G_t, V_t, D_t, B_t, E_t, P_t)
```

with geometry, visual substrate, spatial field, body state, event/semantic state and provenance/promotion state.

The Drosophila adapter refines this as:

```text
G_t : atlas/registration/geometry state
V_t : optical/ephys/behaviour sensor substrate
D_t : registered neural/spatial field and connectome coordinates
B_t : effector/body/kinematic state
E_t : behavioural motifs and later semantic hypotheses
P_t : receipts, residuals, provenance and promotion status
```

The connectome itself belongs primarily to the slow structural constraints on `D_t`/latent dynamics; it is not an instantaneous neural-state observation.

## Governance reuse

The same lattice applies:

```text
substrate -> candidate -> promoted
                    \-> abstain
                    \-> reject
```

A fast connectome kernel, calcium analysis, pose estimator or behaviour classifier may propose a candidate.  It may not directly mutate canonical state.

Promotion should be conditioned on explicit receipts for:

- source dataset/version and hashes;
- functional-to-connectome registration;
- residual/admissibility thresholds;
- relevant cross-modal consistency;
- experiment/predictor commit and output hash.

## Multimodal does not mean independent

Optical calcium and behavioural motion observed from the same animal are distinct modalities but share upstream provenance.  They should therefore be useful jointly without being counted as independent evidence merely because their modality labels differ.

```text
same-fly acquisition
   |-- calcium trace
   `-- motion trace
```

The shared root must remain explicit in the provenance graph.

## Behavioural motifs

Animalexic's existing behavioural-syllable rule transfers directly:

```text
recurrent pose/motion motif != semantic meaning
```

A Drosophila turning, gait, grooming, wing or proboscis motif can be a reproducible dynamical object without already being named as an intention or communicative act.

## Closed-loop biological target

The strongest shared target is:

```text
stimulus
  -> connectome-constrained neural state
  -> registered functional observation
  -> descending/VNC drive
  -> muscle/effector state
  -> body kinematics
  -> behavioural motif
  -> sensory return
```

Animalexic supplies the observation, promotion, abstention, provenance and defeasible semantic layers around that chain.  Drosophila supplies unusually rich structural and functional producers for the chain itself.

## Non-promotion boundaries

```text
structural edge != functional correlation
functional correlation != behavioural event
behavioural motif != communicative act
communicative act != semantic meaning
single modality != multimodal confirmation
multiple modalities != independent evidence when upstream provenance is shared
```
