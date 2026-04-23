# Phase 2d: 2D vs 3D Direction Comparison

Date: 2026-04-19

## Question

The PEZ paper uses a planar direction target because the Kubric physics dataset is overhead 2D motion.
Manipulation data is not planar. This follow-up rerun asks whether PhysProbe direction targets should be treated as:

- 2D XY-angle (`sin/cos`)
- or full 3D unit direction (`vx, vy, vz / ||v||`)

This document compares the two formulations on tasks where token-patch caches already existed:

- `reach`
- `strike`

`push` 3D is still pending because the Push token cache had been deleted earlier to recover disk space during the Strike extraction incident.

## Shared Recipe

- model: `V-JEPA 2 Large`
- capture: `resid_post`
- pooling: `temporal_last_patch`
- feature type: `token_patch`
- probe: `trainable 20-HP sweep`
- CV: `5-fold GroupKFold by episode_id`
- norm: `zscore`

## Reach: 2D vs 3D

| Target | L0 | L8 | Peak R2 | Peak layer | Last | Classification |
|---|---:|---:|---:|---:|---:|---|
| `ee_direction_sincos` | 0.302 | 0.353 | 0.396 | 20 | 0.395 | `intermediate` |
| `ee_direction_3d` | 0.435 | 0.536 | 0.553 | 11 | 0.502 | `PEZ-like` |

Interpretation:

- 3D direction is **clearly better** than the planar XY target on Reach
- the peak rises from `0.396` to `0.553`
- the peak layer moves from very late (`L20`) to mid-depth (`L11`)

Conclusion for Reach:

- 2D was under-specifying the task
- 3D direction is the correct formulation for a 3D manipulation trajectory

## Strike: 2D vs 3D

### End-effector direction

| Target | L0 | L8 | Peak R2 | Peak layer | Last | Classification |
|---|---:|---:|---:|---:|---:|---|
| `ee_direction_sincos` | 0.697 | 0.871 | 0.885 | 11 | 0.877 | `PEZ-like` |
| `ee_direction_3d` | 0.623 | 0.818 | 0.849 | 22 | 0.847 | `intermediate` |

Interpretation:

- 2D is **better** than 3D for Strike end-effector direction if the criterion is PEZ shape
- 3D still gives very high absolute decodability (`0.849`)
- but the peak moves too late (`L22`) to be counted as PEZ-like under the preregistered classifier

### Object direction

| Target | L0 | L8 | Peak R2 | Peak layer | Last | Classification |
|---|---:|---:|---:|---:|---:|---|
| `object_direction_sincos` | -0.220 | 0.048 | 0.084 | 23 | 0.084 | `fail` |
| `object_direction_3d` | 0.521 | 0.774 | 0.813 | 12 | 0.812 | `PEZ-like` |

Interpretation:

- this is the strongest reversal in the entire night shift
- the planar 2D object-direction target was essentially useless
- the 3D object-direction target becomes highly decodable and PEZ-like

Conclusion for Strike:

- for end-effector direction, 2D sin/cos remains the cleaner PEZ-shape target
- for object direction, 3D is dramatically superior and likely the only scientifically valid choice

## Takeaway

### What changed after moving to 3D

1. `reach / ee_direction` improved enough to become PEZ-like
2. `strike / object_direction` was rescued from complete failure to a strong PEZ-like signal
3. `strike / ee_direction` remained strong, but 2D retained the cleaner mid-depth PEZ shape

### Best current interpretation

- 2D direction was a good direct analog for the Kubric paper setting
- 3D direction is the more principled target for manipulation
- the target choice is **not** neutral:
  - it can suppress real 3D object-motion structure if forced into XY only

## Pending

- `push / ee_direction_3d`
- `push / object_direction_3d`

These are deferred until the Push token cache is re-extracted.

## Provisional Verdict

- For manipulation, **3D direction should be treated as the default scientific target**
- 2D direction is still useful as a Kubric-paper analog, but it is not sufficient on its own for final interpretation
