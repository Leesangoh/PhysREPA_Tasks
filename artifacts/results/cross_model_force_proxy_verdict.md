# Cross-Model Tier-B Verdict: Strike Contact-Force Proxy

## Setup

- Task: `strike`
- Surrogate labels:
  - `contact_happening` (binary acceleration-spike event)
  - `contact_force_proxy` (`mass * ||object_acceleration||`)
- Readout:
  - window-level event probe
  - patch-mean token features
  - 5-fold `GroupKFold`
  - trainable 20-HP sweep
  - `zscore`
- Models:
  - `V-JEPA 2 Large` (`phase3_events_strike`, full strike cache)
  - `VideoMAE-L` (`phase3_events_videomae_strike`, matched `1000`-episode subset)
  - `DINOv2-L` (`phase3_events_dino_strike`, matched `1000`-episode subset)

## Main result

| Model | `contact_happening` peak AUC | Peak layer | `contact_force_proxy` peak `R^2` | Peak layer |
| --- | ---: | ---: | ---: | ---: |
| `V-JEPA 2 Large` | `0.9987` | `L14` | **`0.2204`** | `L20` |
| `VideoMAE-L` | `0.9965` | `L23` | `0.1980` | `L19` |
| `DINOv2-L` | `0.9879` | `L16` | `0.1537` | `L18` |

## Interpretation

Three patterns are stable.

1. `contact_happening` is nearly saturated for all three models.
   - `V-JEPA 2`: `0.9987 @ L14`
   - `VideoMAE-L`: `0.9965 @ L23`
   - `DINOv2-L`: `0.9879 @ L16`
   This target remains an upper-bound surrogate because it is tightly coupled to visible motion spikes.

2. `contact_force_proxy` is the scientifically meaningful Tier-B target.
   - `V-JEPA 2`: `0.2204 @ L20`
   - `VideoMAE-L`: `0.1980 @ L19`
   - `DINOv2-L`: `0.1537 @ L18`

3. The ordering is consistent with the kinematic cross-model story but stronger in scientific value:
   - predictive video pretraining is best
   - masked-video pretraining is intermediate
   - static-image pretraining is weakest

## Quantitative gaps

- `V-JEPA 2` vs `VideoMAE-L`
  - absolute gap: `+0.0224 R^2`
  - relative to `VideoMAE-L`: `+11.3%`
- `V-JEPA 2` vs `DINOv2-L`
  - absolute gap: `+0.0667 R^2`
  - relative to `DINOv2-L`: `+43.4%`
- `VideoMAE-L` vs `DINOv2-L`
  - absolute gap: `+0.0443 R^2`
  - relative to `DINOv2-L`: `+28.8%`

## What this shows

- This is the first three-way cross-model Tier-B panel in the project.
- The objective-specificity story now extends beyond observable kinematics.
- `V-JEPA 2` is not only the only tested family with an intermediate-depth PEZ on manipulation direction.
- It is also the strongest tested family on an implicit interaction-magnitude target.

The key paper claim is therefore stronger:

> Predictive video pretraining is not only distinctive in representational timing on observable manipulation kinematics; it also yields the strongest decoding of implicit interaction magnitude among the tested objectives.

## Important caveat

- `VideoMAE-L` and `DINOv2-L` were run on a matched `1000`-episode subset to keep the force panel storage-safe and directly comparable.
- The current `V-JEPA 2 Large` force baseline comes from the previously committed full-cache run rather than a matched subset rerun.
- The ranking is still informative, but a perfectly matched three-way panel would require rerunning the `V-JEPA 2` event probe on the same `1000`-episode subset.

## Paper implication

- The kinematic cross-model section established:
  - `V-JEPA 2` = mid-depth PEZ
  - `VideoMAE-L` = late refinement
  - `DINOv2-L` = weaker and late
- The new force panel shows:
  - the same objective family that produces the most manipulation-specific representational timing also produces the strongest interaction-magnitude decoding
- This upgrades the paper from a Tier-A timing story to a Tier-A + Tier-B representation story.
