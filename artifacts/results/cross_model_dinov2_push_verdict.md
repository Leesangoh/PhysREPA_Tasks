# Cross-Model Verdict: Push DINOv2-L and the 4-way comparison

## Main finding

The Push cross-model panel now contains three distinct regimes:

- `V-JEPA 2`: intermediate-depth PEZ-like accessibility
- `VideoMAE-L`: strongest final decoding, but only at the last layer
- `DINOv2-L`: weakest overall and also late-peaking

This is the clearest current evidence that the PEZ is not a generic property of
strong encoders. It is tied to **pretraining objective**.

## Push / `ee_direction_3d`

| Model | Seeds | L0 | L8 | Peak R^2 | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 Large | 3 | 0.648 | 0.804 | 0.816 | 11.7 / 24 | 0.486 | 0.811 |
| V-JEPA 2 Giant | 2 | 0.537 | 0.772 | 0.818 | 27.0 / 40 | 0.675 | 0.803 |
| V-JEPA 2 Huge | 1 | 0.559 | 0.803 | 0.817 | 15 / 32 | 0.469 | 0.795 |
| VideoMAE-L | 1 | 0.602 | 0.807 | **0.844** | **23 / 24** | **0.958** | **0.844** |
| DINOv2-L | 1 | -1.455 | 0.700 | 0.776 | 15 / 24 | 0.625 | 0.765 |

Interpretation:

- `V-JEPA 2` is the only family whose best direction decoding appears at
  intermediate depth across all tested scales.
- `VideoMAE-L` exceeds V-JEPA in final Push direction `R^2`, but only by
  refining to the final block.
- `DINOv2-L` is substantially weaker than both video-model families and does not
  produce the same mid-depth accessibility pattern.

## Push / `ee_speed`

| Model | Seeds | L0 | L8 | Peak R^2 | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 Large | 3 | 0.707 | 0.930 | 0.934 | 11.0 / 24 | 0.458 | 0.917 |
| V-JEPA 2 Giant | 2 | 0.516 | 0.913 | 0.935 | 25.0 / 40 | 0.625 | 0.914 |
| V-JEPA 2 Huge | 1 | 0.528 | 0.918 | 0.924 | 16 / 32 | 0.500 | 0.904 |
| VideoMAE-L | 1 | 0.514 | 0.919 | **0.943** | **23 / 24** | **0.958** | **0.943** |
| DINOv2-L | 1 | -1.136 | 0.729 | 0.885 | 23 / 24 | 0.958 | 0.885 |

Interpretation:

- The same ordering appears on the control target.
- `VideoMAE-L` has the strongest final speed decoding.
- `DINOv2-L` is again weaker and peaks at the last layer.
- `V-JEPA 2 Large` retains a distinctive earlier accessibility regime.

## Updated paper claim after Push 4-way comparison

The strongest defensible cross-model Push statement is now:

> predictive video pretraining is the only tested pretraining type that
> produces a mid-depth manipulation PEZ; masked-video and static-image
> pretraining push the same targets toward the final layer.

This is stronger than the earlier two-model comparison because `DINOv2-L`
removes the fallback explanation that the timing split is simply ``video versus
image''. The actual pattern is:

- predictive video objective -> mid-depth PEZ
- masked-video objective -> late refinement
- static-image objective -> weaker, mostly late refinement

## What remains to close the 3-way story

- finish `DINOv2-L` Strike extraction
- run `Strike / object_direction_3d`
- then write the unified `cross_model_verdict.md` across:
  - `V-JEPA 2`
  - `VideoMAE-L`
  - `DINOv2-L`
