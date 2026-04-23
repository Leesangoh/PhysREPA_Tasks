# Phase 2c Overall Verdict: Push vs Reach vs Strike

Date: 2026-04-19

## Objective

Evaluate whether a PEZ-aligned probing recipe from the PEZ reproduction project transfers to PhysProbe manipulation data.

Shared recipe:

- features: `resid_post + temporal_last_patch`
- model: `V-JEPA 2 Large`
- probe: `trainable 20-HP sweep`
- CV: `5-fold GroupKFold by episode_id`
- norm: `zscore`
- primary PEZ analog targets:
  - `ee_direction_sincos`
  - `ee_speed`
  - `ee_accel_magnitude`
- integrity target:
  - `fake_mod5`

## Task-by-Task Summary

| Task | Target | L0 | L8 | Peak R2 | Peak layer | Last | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| Push | `ee_direction_sincos` | 0.555 | 0.789 | 0.807 | 13 | 0.797 | `PEZ-like` |
| Push | `ee_speed` | 0.671 | 0.928 | 0.931 | 13 | 0.920 | `always-linear` by interpretation |
| Push | `ee_accel_magnitude` | 0.347 | 0.673 | 0.694 | 14 | 0.688 | `intermediate` |
| Push | `object_direction_sincos` | -0.220 | 0.048 | 0.084 | 23 | 0.084 | `fail` |
| Reach | `ee_direction_sincos` | 0.302 | 0.353 | 0.396 | 20 | 0.395 | `intermediate` |
| Reach | `ee_speed` | 0.598 | 0.813 | 0.825 | 17 | 0.797 | `intermediate` |
| Reach | `ee_accel_magnitude` | 0.717 | 0.854 | 0.873 | 10 | 0.835 | `PEZ-like` |
| Reach | `fake_mod5` | -0.565 | -0.673 | -0.410 | 3 | -0.470 | `never-linear` |
| Strike | `ee_direction_sincos` | 0.697 | 0.871 | 0.885 | 11 | 0.877 | `PEZ-like` |
| Strike | `ee_speed` | 0.883 | 0.957 | 0.963 | 11 | 0.951 | `always-linear` |
| Strike | `ee_accel_magnitude` | 0.664 | 0.876 | 0.896 | 12 | 0.878 | `PEZ-like` |
| Strike | `fake_mod5` | -0.462 | -0.349 | -0.290 | 12 | -0.306 | `never-linear` |

## Main Scientific Findings

### 1. PEZ-like emergence is present on end-effector kinematics, but not uniformly across tasks

Strongest positive cases:

- `push / ee_direction_sincos`:
  - `peak=0.807 @ L13`
- `strike / ee_direction_sincos`:
  - `peak=0.885 @ L11`

These are clear mid-depth rises with high peak R2 and mild late decline, consistent with a manipulation-domain PEZ analog.

### 2. Contact-rich Strike is the strongest overall reproduction

`strike` is the first task where all three kinematic PEZ analogs line up cleanly:

- `ee_direction_sincos`: strong PEZ-like
- `ee_speed`: shallow/high, effectively always-linear
- `ee_accel_magnitude`: strong PEZ-like
- `fake_mod5`: negative everywhere

This is the cleanest evidence so far that the PEZ methodology transfers to manipulation data in a contact-heavy setting.

### 3. Reach is a valid integrity/generalization check, but not a strong PEZ direction case

`reach` does **not** reproduce the Push/Strike direction-emergence story:

- `ee_direction_sincos` peaks only at `0.396 @ L20`

But it is still valuable because:

- `ee_accel_magnitude` remains strongly decodable (`0.873 @ L10`)
- `fake_mod5` is negative at all layers, which strongly supports leak-free evaluation

Interpretation:

- PEZ-like direction emergence is not a universal artifact of all arm motion
- richer interaction structure appears to matter

### 4. Static physics and object-side direction remain hard

From Push:

- static `mass / friction / surface_friction` stayed `never-linear`
- `object_direction_sincos` failed badly (`peak=0.084 @ L23`)

This suggests the current success story is specifically:

- **end-effector kinematics**
- especially in **interaction-rich tasks**

not a general “all physics parameters become PEZ-like” claim.

## Integrity Checks

The strongest integrity evidence:

- `reach / fake_mod5`: negative at every layer
- `strike / fake_mod5`: negative at every layer

This makes a leakage explanation much less plausible.

## Overall Verdict

### What is supported

- The PEZ probing methodology **does transfer** to PhysProbe manipulation data.
- The strongest transfer is on **end-effector kinematic variables**.
- The clearest positive task is **Strike**; Push is also positive but more partial.

### What is not supported

- Static physics parameters do not emerge under the current recipe.
- Object-side direction is not recovered by the linear flatten readout.
- Direction emergence is not universal across all tasks; Reach is weaker.

## Ranked Tasks by PEZ Alignment

1. `strike` — strongest full kinematic reproduction
2. `push` — clear ee-side PEZ, but object/static failures
3. `reach` — useful integrity/generalization check, not a strong direction-PEZ case

## Recommended Next Step

If work continues, the most justified next experiment is:

1. `Huge` model on `strike` and/or `push`

Rationale:

- PEZ Figure 6 analog can now be tested on the strongest manipulation tasks
- this is more valuable than reviving the expensive attentive pilot
