# Scale Law Verdict

## Scope

This document summarizes the completed Push scale comparison for:

- `ee_direction_3d`
- `ee_speed`

under the same PEZ-informed recipe:

- `resid_post`
- `temporal_last_patch`
- token-patch flattening
- `trainable` 20-HP sweep
- 5-fold `GroupKFold` by `episode_id`
- `zscore`

Models:

- `V-JEPA 2 Large` (`24` layers, `1024` dim, `3` probe seeds)
- `V-JEPA 2 Giant` (`40` layers, `1408` dim, `2` probe seeds)
- `V-JEPA 2 Huge` (`32` layers, `1280` dim, `1` probe seed)


## Main result

The data do **not** support a monotonic depth law.

What is preserved across scale is the **peak magnitude** of direction decoding:

- `Large`: `0.8165 ± 0.0006`
- `Giant`: `0.8183 ± 0.0030`
- `Huge`: `0.8168`

What changes is **where** the peak occurs, and that depth pattern is
non-monotonic:

- `Large`: `L11.7 / 24 = 0.486`
- `Giant`: `L27.0 / 40 = 0.675`
- `Huge`: `L15 / 32 = 0.469`

So the correct claim is:

> Scale preserves the existence and strength of the Push direction PEZ-like
> regime, but peak depth is architecture-sensitive rather than monotonic.


## Full summary

### `ee_direction_3d`

| Model | Seeds | L0 | L8 | Peak R² | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| Large | 3 | `0.6478 ± 0.0071` | `0.8043 ± 0.0022` | `0.8165 ± 0.0006` | `11.7 ± 0.9` | `0.486 ± 0.039` | `0.8107 ± 0.0016` |
| Giant | 2 | `0.5369 ± 0.0159` | `0.7716 ± 0.0046` | `0.8183 ± 0.0030` | `27.0 ± 3.0` | `0.675 ± 0.075` | `0.8032 ± 0.0049` |
| Huge | 1 | `0.5586` | `0.8025` | `0.8168` | `15` | `0.469` | `0.7950` |

Direction interpretation:

- Peak `R^2` is effectively invariant across scale.
- Giant pushes the best-decoding layer much deeper than Large.
- Huge does **not** continue that trend; it returns to nearly the same
  fractional depth as Large.
- Therefore the strongest empirical statement is not
  “bigger model -> deeper PEZ,” but
  “direction PEZ survives scaling, while its depth depends on architecture.”


### `ee_speed`

| Model | Seeds | L0 | L8 | Peak R² | Peak layer | Peak depth | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| Large | 3 | `0.7071 ± 0.0274` | `0.9302 ± 0.0032` | `0.9340 ± 0.0033` | `11.0 ± 1.6` | `0.458 ± 0.068` | `0.9170 ± 0.0023` |
| Giant | 2 | `0.5156 ± 0.0019` | `0.9132 ± 0.0002` | `0.9347 ± 0.0043` | `25.0 ± 0.0` | `0.625 ± 0.000` | `0.9136 ± 0.0043` |
| Huge | 1 | `0.5278` | `0.9180` | `0.9241` | `16` | `0.500` | `0.9041` |

Speed interpretation:

- Speed also remains strongly decodable across scale.
- Giant again moves the peak later than Large.
- Huge again returns to a near-half-depth peak.
- Peak magnitude is stable to within about one percentage point, much more
  stable than peak location.


## Honest interpretation

The cleanest interpretation is that **peak magnitude and peak depth are governed
by different factors**.

- Pretraining makes the kinematic signal strong at all three scales.
- Architecture family appears to influence where that signal becomes maximally
  linearly aligned.

This interpretation fits the actual model families:

- `Large` and `Huge` are both standard `vit_large / vit_huge` variants.
- `Giant` is a `vit_giant_xformers` variant with a different backbone design.

So the current evidence is more consistent with:

> architecture-sensitive representational timing

than with:

> a monotonic parameter-count law.

This matters scientifically because a monotonic story would have been simpler
but wrong. The scale result is still useful: it shows that the PEZ-like regime
is robust across model sizes while rejecting a naive “deeper-is-bigger” law.


## What this means for the paper

The paper should make three scale-law claims and no stronger ones:

1. `Push ee_direction_3d` remains strongly PEZ-like across `Large`, `Giant`,
   and `Huge`.
2. Peak magnitude is nearly invariant across scale (`~0.817`).
3. Peak depth is non-monotonic: `Large ≈ Huge < Giant`.

The paper should **not** claim a monotonic scale law.


## Figure / table specification

### Main table

Table title:

> Push scale comparison across V-JEPA 2 Large, Giant, and Huge.

Columns:

- model
- layers
- embedding dim
- seeds
- target
- `L0`
- `L8`
- peak `R^2`
- peak layer
- peak depth
- last

### Main figure

Figure title:

> Scale preserves peak strength but not peak depth.

Panel A:

- overlay of `ee_direction_3d` curves for `Large`, `Giant`, `Huge`
- x-axis: absolute layer index
- annotate peak for each curve

Panel B:

- same curves plotted against fractional depth (`layer / num_layers`)
- this makes the `Large ≈ Huge` versus `Giant` split visually obvious

Panel C:

- `ee_speed` overlay in the same normalized-depth coordinates

Panel D:

- bar plot of peak `R^2` and peak depth by model


## Next implication

The scale result strengthens the paper, but it changes the narrative.

The headline is no longer:

> larger models push PEZ deeper.

The stronger and more accurate headline is:

> manipulation PEZ is robust across scale, while its depth depends on
> architecture family.
