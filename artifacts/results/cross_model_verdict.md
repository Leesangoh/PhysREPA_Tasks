# Unified Cross-Model Verdict

## Scope

This summary consolidates the cross-model results that landed cleanly in the
current study:

- Push:
  - `ee_direction_3d`
  - `ee_speed`
- Strike:
  - `object_direction_3d`
  - surrogate-contact event targets
    - `contact_happening`
    - `contact_force_proxy`

Backbones:

- `V-JEPA 2` (`Large`, plus `Giant/Huge` for Push scale)
- `VideoMAE-L`
- `DINOv2-L`

## Push: predictive video pretraining is the only tested family with a mid-depth PEZ

| Model | Target | Peak | Peak layer | Peak depth |
| --- | --- | ---: | ---: | ---: |
| `V-JEPA 2 Large` | `ee_direction_3d` | `0.816` | `11.7 / 24` | `0.486` |
| `V-JEPA 2 Giant` | `ee_direction_3d` | `0.818` | `27 / 40` | `0.675` |
| `V-JEPA 2 Huge` | `ee_direction_3d` | `0.817` | `15 / 32` | `0.469` |
| `VideoMAE-L` | `ee_direction_3d` | **`0.844`** | `23 / 24` | `0.958` |
| `DINOv2-L` | `ee_direction_3d` | `0.776` | `15 / 24` | `0.625` |
| `V-JEPA 2 Large` | `ee_speed` | `0.934` | `11.0 / 24` | `0.458` |
| `V-JEPA 2 Giant` | `ee_speed` | `0.935` | `25 / 40` | `0.625` |
| `V-JEPA 2 Huge` | `ee_speed` | `0.924` | `16 / 32` | `0.500` |
| `VideoMAE-L` | `ee_speed` | **`0.943`** | `23 / 24` | `0.958` |
| `DINOv2-L` | `ee_speed` | `0.885` | `23 / 24` | `0.958` |

### Push interpretation

- `VideoMAE-L` can exceed `V-JEPA 2` in final `R^2`, but only through
  last-layer refinement.
- `DINOv2-L` is weaker overall and also late-peaking.
- The distinctive `V-JEPA 2` property on Push is not final accuracy alone.
- It is the existence of a strong **intermediate-depth accessibility regime**.

## Strike object direction: predictive video pretraining is both earlier and stronger

| Model | Target | Peak | Peak layer | Peak depth |
| --- | --- | ---: | ---: | ---: |
| `V-JEPA 2 Large` | `object_direction_3d` | **`0.813`** | **`12 / 24`** | **`0.500`** |
| `VideoMAE-L` | `object_direction_3d` | `0.788` | `23 / 24` | `0.958` |

### Strike-object interpretation

- On the harder object-side target, `V-JEPA 2` does not just peak earlier.
- It also peaks higher.
- This rules out the idea that earlier `V-JEPA` peaks are only an efficiency
  difference followed by matching final performance.

## Strike Tier-B force panel: predictive video pretraining is strongest on interaction magnitude

| Model | `contact_happening` peak AUC | Peak layer | `contact_force_proxy` peak `R^2` | Peak layer |
| --- | ---: | ---: | ---: | ---: |
| `V-JEPA 2 Large` | `0.9987` | `L14` | **`0.2204`** | `L20` |
| `VideoMAE-L` | `0.9965` | `L23` | `0.1980` | `L19` |
| `DINOv2-L` | `0.9879` | `L16` | `0.1537` | `L18` |

### Tier-B interpretation

- `contact_happening` is nearly saturated for all tested families and mostly
  acts as an upper-bound sanity target.
- The informative target is `contact_force_proxy`.
- Here the ranking is:
  - `V-JEPA 2 > VideoMAE-L > DINOv2-L`
- This extends the objective-specificity story beyond observable kinematics.

## Final claim supported by the current cross-model evidence

The current cross-model panel supports three linked conclusions.

1. Predictive video pretraining is the only tested family that reliably
   produces a manipulation PEZ-like intermediate-depth regime on observable
   kinematic targets.
2. Masked-video and static-image pretraining often recover the same targets
   only through late refinement.
3. On the harder Tier-B interaction-magnitude target, predictive video
   pretraining is also the strongest tested family in absolute decoding
   strength.

This is stronger than a generic ``video models encode manipulation'' story. The
evidence now supports an objective-sensitive claim about both
**representational timing** and **implicit interaction-magnitude decoding**.
