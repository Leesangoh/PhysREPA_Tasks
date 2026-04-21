# Cross-Model Verdict: Push VideoMAE-L vs V-JEPA 2

## Main finding

`VideoMAE-L` does **not** reproduce a Push PEZ-style mid-depth regime on the
main manipulation-direction target. Instead, it keeps improving until the last
layer.

This is the strongest current cross-model result because it separates
**objective-specific representational timing** from the generic fact of being a
video model.

## Push / `ee_direction_3d`

| Model | Seeds | L0 | L8 | Peak R^2 | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 Large | 3 | 0.648 | 0.804 | 0.816 | 11.7 / 24 | 0.486 | 0.811 |
| V-JEPA 2 Giant | 2 | 0.537 | 0.772 | 0.818 | 27.0 / 40 | 0.675 | 0.803 |
| V-JEPA 2 Huge | 1 | 0.559 | 0.803 | 0.817 | 15 / 32 | 0.469 | 0.795 |
| VideoMAE-L | 1 | 0.602 | 0.807 | **0.844** | **23 / 24** | **0.958** | **0.844** |

Interpretation:

- V-JEPA 2 preserves a stable direction peak magnitude (`~0.817`) across scale
  while peaking at intermediate depth.
- `VideoMAE-L` reaches a **higher final direction R^2** (`0.844`) but shows no
  intermediate plateau or mid-depth peak.
- The key distinction is therefore **not peak strength**, but **where the
  representation becomes maximally linearly accessible**.

## Push / `ee_speed`

| Model | Seeds | L0 | L8 | Peak R^2 | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 Large | 3 | 0.707 | 0.930 | 0.934 | 11.0 / 24 | 0.458 | 0.917 |
| V-JEPA 2 Giant | 2 | 0.516 | 0.913 | 0.935 | 25.0 / 40 | 0.625 | 0.914 |
| V-JEPA 2 Huge | 1 | 0.528 | 0.918 | 0.924 | 16 / 32 | 0.500 | 0.904 |
| VideoMAE-L | 1 | 0.514 | 0.919 | **0.943** | **23 / 24** | **0.958** | **0.943** |

Interpretation:

- The control target shows the same qualitative pattern.
- `VideoMAE-L` attains the strongest final speed decoding, but only at the last
  layer.
- V-JEPA 2 concentrates both direction and speed decoding much earlier.

## Paper claim supported by this result

This result rules out the simple reviewer alternative:

> ``Any strong video backbone will show a manipulation PEZ.''

That statement is false under the current benchmark.

- `V-JEPA 2` produces a **mid-depth PEZ-like regime**
- `VideoMAE-L` produces **late monotonic refinement**

So the current evidence supports the narrower and stronger claim:

> manipulation PEZ is **objective-specific**, not just a generic consequence of
> video pretraining.

## Honest caveats

- `VideoMAE-L` is currently measured on `Push` only and with one probe seed.
- The strongest claim at this stage is therefore:
  - objective-specificity on `Push / ee_direction_3d`
  - not yet a full cross-model generalization across all tasks
- `Strike / object_direction_3d` is the next required cross-model stage.

## Immediate next step

- finish `VideoMAE-L` Strike extraction
- run `Strike / object_direction_3d`
- test whether the same qualitative split holds:
  - V-JEPA 2: mid-depth object-direction rescue
  - VideoMAE-L: late monotonic refinement or weaker object-side signal
