# Cross-Model Verdict: Strike VideoMAE-L vs V-JEPA 2

## Main finding

On the harder object-side manipulation target, `V-JEPA 2 Large` is stronger than
`VideoMAE-L` in both **peak timing** and **peak magnitude**.

This is the strongest current cross-model result because it shows that the
cross-model distinction is not only about where the signal peaks. On
`Strike / object_direction_3d`, predictive video pretraining also produces the
better final decoder.

## Strike / `object_direction_3d`

| Model | Seeds | L0 | L8 | Peak R^2 | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 Large | 1 | 0.521 | 0.774 | **0.813** | **12 / 24** | **0.500** | **0.812** |
| VideoMAE-L | 1 | 0.450 | 0.725 | 0.788 | 23 / 24 | 0.958 | 0.788 |

## Interpretation

- `V-JEPA 2` rescues Strike object direction at intermediate depth.
- `VideoMAE-L` does improve object-direction decoding, but it does so only by
  refining the signal until the final layer.
- Unlike Push end-effector direction, where `VideoMAE-L` reached a higher final
  `R^2`, Strike object direction is a regime where `V-JEPA 2` is better in
  both respects:
  - earlier accessibility
  - stronger peak decoding

This makes the cross-model story stronger, not weaker:

- `Push / ee_direction_3d` showed that `VideoMAE-L` can match or exceed final
  decoding strength without producing a PEZ.
- `Strike / object_direction_3d` shows that on the harder contact-conditioned
  object target, `V-JEPA 2` both preserves the PEZ-like mid-depth regime and
  beats `VideoMAE-L` outright.

## Paper claim supported by Push + Strike together

The current cross-model evidence supports the following claim:

> predictive video pretraining produces a distinctive intermediate-depth
> manipulation regime that masked-video pretraining does not reproduce.

More specifically:

- for easier end-effector kinematic targets, `VideoMAE-L` can reach higher final
  `R^2` but only at the last layer
- for harder object-side contact-conditioned targets, `V-JEPA 2` is both
  earlier and better

So the best current paper statement is:

> PEZ is objective-specific, and its strongest advantage appears on harder
> contact-conditioned object dynamics.

## Immediate next step

- add the Strike row to the main cross-model paper table
- update the paper text so the cross-model section is no longer a Push-only
  observation but a two-target objective-specificity result
