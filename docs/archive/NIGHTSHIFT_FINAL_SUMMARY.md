# NIGHTSHIFT Final Summary

Date: 2026-04-19

## One-Line Verdict

This night shift established a **real PEZ transfer signal on PhysProbe manipulation data**, strongest on **end-effector kinematics** in **contact-rich tasks** (`strike`, then `push`), while static physics and object-side direction remained unrecovered.

## Final Outcomes

### Push

Best positive result:

- `ee_direction_sincos`
  - `L0=0.555`
  - `L8=0.789`
  - `peak=0.807 @ L13`
  - `last=0.797`
  - `classification=PEZ-like`

Other findings:

- `ee_pos`: shallow/high control success
- `ee_speed`: high from early layers
- `object_direction_sincos`: failed (`peak=0.084 @ L23`)
- static `mass / friction / surface_friction`: never-linear

Interpretation:

- Push is a **partial** manipulation-domain PEZ reproduction
- the signal lives on the end-effector side, not the object/static side

### Reach

Key findings:

- `ee_direction_sincos` is weak:
  - `peak=0.396 @ L20`
- `ee_accel_magnitude` is strong:
  - `peak=0.873 @ L10`
- `fake_mod5` negative everywhere

Interpretation:

- Reach validates the pipeline
- Reach does **not** support a universal “all arm motion has PEZ-like direction emergence” claim

### Strike

Strongest task of the night:

- `ee_direction_sincos`
  - `L0=0.697`
  - `L8=0.871`
  - `peak=0.885 @ L11`
  - `last=0.877`
  - `classification=PEZ-like`
- `ee_speed`
  - `L0=0.883`
  - `peak=0.963 @ L11`
  - `classification=always-linear`
- `ee_accel_magnitude`
  - `L0=0.664`
  - `peak=0.896 @ L12`
  - `classification=PEZ-like`
- `fake_mod5`
  - negative at all layers

Interpretation:

- Strike is the **clearest positive transfer** of PEZ methodology to manipulation data

## What Changed the Answer

### 1. Mean-pooled features were not enough

Phase 1 on Push mean-pooled cache made static physics look non-decodable and did not reveal the strongest kinematic story.

### 2. PEZ-aligned token-patch features mattered

The decisive recipe was:

- `resid_post`
- `temporal_last_patch`
- token-patch flatten
- `trainable 20-HP`
- `5-fold GroupKFold by episode_id`
- `zscore`

### 3. Manipulation direction required `sin/cos`, not scalar angle

This was the most important target-parameterization lesson:

- scalar angle failed on Push direction
- `sin/cos` recovered a clear PEZ-like curve

Interpretation:

- circular wrap hurts manipulation much more than simple ballistic PEZ videos

## Blockers Encountered

### Public contact labels are unusable

`contact_flag`, `contact_force`, and related channels were all-zero in the public data audit.

Consequence:

- event-aligned force probe was blocked

### Token cache storage pressure

`push` and `strike` token-patch caches consumed ~terabyte-scale storage.

Operational consequence:

- Push cache had to be deleted to free `/mnt` space
- Strike extraction ended with a single corrupted tail file due to disk pressure
- this was repaired by quarantining `002895.safetensors`

### Attentive pilot was too expensive

A one-layer attentive pilot on Push ran for hours without delivering enough incremental value.

Decision:

- terminate attentive pilot
- prioritize `strike`

## Task Tree

### Completed

- `Phase 1`
  - Push mean-pooled baseline
- `Phase 2`
  - Push token-patch probe
  - direction `sin/cos` rescue
- `Phase 2c`
  - Reach token-patch extraction + probe
  - Strike token-patch extraction + probe
- Night-shift comparative verdict:
  - Push vs Reach vs Strike

### Blocked

- `Phase 3` force/contact probe
  - blocked by all-zero public contact labels

### Stopped by choice

- Push attentive pilot
  - compute-expensive
  - low marginal value relative to Strike

## Recommended Next Step

1. Run `Huge` on `strike` and/or `push`

Reason:

- Figure 6 analog is now the most information-dense next experiment
- the best manipulation tasks are identified
- further linear re-runs are no longer the bottleneck

## Key Files

- `artifacts/results/EXPERIMENT_RESULTS_phase2_push.md`
- `artifacts/results/EXPERIMENT_RESULTS_phase2_push_vs_reach.md`
- `artifacts/results/EXPERIMENT_RESULTS_phase2c_overall.md`
- `artifacts/results/verdict_phase2_push.json`
- `artifacts/results/verdict_phase2c_reach_reach.json`
- `artifacts/results/verdict_phase2c_strike_strike.json`
- `docs/archive/NIGHTSHIFT_LOG.md`
