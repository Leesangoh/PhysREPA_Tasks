# Phase 2c Comparison: Push vs Reach

Date: 2026-04-18

## Scope

This note compares the Phase 2 token-patch PEZ-aligned probing recipe across two manipulation tasks:

- `push` (contact-capable rigid-body interaction)
- `reach` (no task-level physics randomization; arm-motion generality test)

Shared recipe:

- model: `V-JEPA 2 Large`
- feature cache: `resid_post + temporal_last_patch`
- probe: `trainable 20-HP sweep`
- CV: `5-fold GroupKFold by episode_id`
- norm: `zscore`

## Summary Table

| Task | Target | L0 | L8 | Peak R2 | Peak layer | Last | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| Push | `ee_direction_sincos` | 0.555 | 0.789 | 0.807 | 13 | 0.797 | `PEZ-like` |
| Push | `ee_speed` | 0.671 | 0.928 | 0.931 | 13 | 0.920 | `intermediate` |
| Push | `ee_accel_magnitude` | 0.347 | 0.673 | 0.694 | 14 | 0.688 | `PEZ-like` |
| Push | `object_direction_sincos` | -0.220 | 0.048 | 0.084 | 23 | 0.084 | `never-linear` |
| Reach | `ee_direction_sincos` | 0.302 | 0.353 | 0.396 | 20 | 0.395 | `intermediate` |
| Reach | `ee_speed` | 0.598 | 0.813 | 0.825 | 17 | 0.797 | `intermediate` |
| Reach | `ee_accel_magnitude` | 0.717 | 0.854 | 0.873 | 10 | 0.835 | `PEZ-like` |
| Reach | `fake_mod5` | -0.565 | -0.673 | -0.410 | 3 | -0.470 | `never-linear` |

## Main Findings

### 1. Push reproduces an arm-side PEZ analog

`push / ee_direction_sincos` is the clearest manipulation-domain direction analog found so far:

- low-to-moderate shallow baseline (`L0=0.555`)
- strong mid-depth rise (`L8=0.789`)
- peak in the middle of the stack (`L13`)
- mild late decline (`0.807 -> 0.797`)

This is the strongest evidence so far that a PEZ-like pattern exists in manipulation features, but only for a subset of targets.

### 2. Reach does not reproduce the same direction-emergence story

`reach / ee_direction_sincos` does not show the Push-style PEZ curve:

- `L0=0.302`
- `L8=0.353`
- peak only `0.396 @ L20`

Interpretation:

- the arm-direction PEZ seen in Push is not a generic arm-motion effect
- interaction context appears to matter

### 3. Reach validates pipeline integrity

`reach / fake_mod5` is negative at every layer:

- `L0=-0.565`
- `L8=-0.673`
- peak `-0.410 @ L3`

This is the strongest integrity check so far that the token-patch probe is not leaking episode identity.

### 4. Reach acceleration is strongly decodable

`reach / ee_accel_magnitude` peaks at `0.873 @ L10`, which satisfies the pre-registered classifier's `PEZ-like` rule.

However, its shallow baseline is already high (`L0=0.717`), so this is better interpreted as:

- a strong mid-depth refinement curve
- not a clean low-baseline emergence in the Figure 2(c) sense

## Verdict

### Push

- `ee_direction_sincos`: PEZ-like
- `ee_accel_magnitude`: PEZ-like by classifier
- `ee_speed`: strongly decodable but not a clean `always-linear` control under the strict `L0 >= 0.8` rule
- `object_direction_sincos`: failed

Overall verdict:

- `Scenario A partial`
- PEZ-aligned behavior is present on the end-effector side, not on the object side

### Reach

- `ee_direction_sincos`: not PEZ-like
- `ee_accel_magnitude`: classifier-positive but with high shallow baseline
- `fake_mod5`: correct negative control

Overall verdict:

- the Phase 2 recipe generalizes as a *valid probe pipeline*
- it does **not** generalize as a universal arm-direction PEZ detector

## Implication for Next Steps

Priority remains:

1. finish the Push attentive pilot
2. test whether attentive readout can rescue `object_direction_sincos`
3. if not, move to a new interaction-heavy task (`strike`) rather than over-generalizing from Reach
