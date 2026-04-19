# F5 Design: Frame Shuffle Stress Test for PhysProbe PEZ Results

## Scope

Round 1 of `NIGHTSHIFT2_PROTOCOL.md` tests whether the current positive PhysProbe PEZ curves are driven by:

1. true temporal structure, or
2. static visual correlates that survive token-patch probing.

This design applies to the strongest existing positive targets:

- `push / ee_direction_3d`
- `push / ee_speed`
- `strike / ee_direction_3d`
- `strike / object_direction_3d`

The baseline results already exist in committed Phase 2d artifacts.

## Evidence Base

Evidence-based:

- Push and Strike token-patch caches already exist at:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike`
- Disk is tight:
  - `/mnt/md1/solee` usage was `95%` at start of Nightshift2.
  - Push raw token cache size: `1006G`
  - Strike raw token cache size: `1.5T`
- Reach token cache is already absent, so the protocol's first deletion fallback has already effectively happened.

Hypothesis:

- If current PEZ-like results depend on temporal causality, frame shuffling inside each 16-frame window should materially reduce peak `R^2` and/or delay/flatten the emergence curve.
- If curves mostly survive shuffle, the current positive results are more compatible with static scene correlates or framewise appearance cues.

## Proposed Shuffle Implementation

### Window-level shuffle

For each sampled 16-frame window:

- preserve the frame set
- permute frame order with a deterministic RNG
- feed the permuted clip through the same extractor used for the original cache

This destroys temporal causality while preserving:

- object identity
- background
- camera viewpoint
- per-frame appearance statistics

### Deterministic seed

Use:

- global seed: `42`
- derived per-window seed: hash of `(task, episode_id, window_start, 42)`

This guarantees:

- reproducibility across reruns
- different windows receive different shuffles
- identical rerun outputs for the same episode/window

### Extraction settings

Keep the current best-performing token recipe fixed:

- model: `large`
- residual capture: `resid_post`
- transform: `resize`
- pooling: `temporal_last_patch`
- output dtype: `float16`

### Output roots

- Push shuffled cache:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_shuffled/push`
- Strike shuffled cache:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_shuffled/strike`

## Probe Settings

Keep the probe recipe matched to the successful Phase 2d runs:

- `feature-type = token_patch`
- flattened patch tokens
- `trainable` solver
- 20 HP sweep:
  - LR in `{1e-4, 3e-4, 1e-3, 3e-3, 5e-3}`
  - WD in `{0.01, 0.1, 0.4, 0.8}`
- 5-fold `GroupKFold`
- group key: `episode_id`
- normalization: `zscore`

## Sanity Checks

### S1. Shuffle really changes order

For the first extracted episode:

- record original `window_starts`
- record the first shuffled frame permutation
- confirm the permutation is not identity

### S2. Determinism

Rerun one sampled episode twice and confirm identical shuffled indices.

### S3. Cache integrity

Check:

- keys exist for all 24 layers
- tensor shapes match original token cache
- dtype remains `float16`

### S4. Probe integrity

For shuffled probing:

- fake target or negative control is not required if original target already has a known baseline
- train/val episode overlap must remain zero

## Expected Outcomes

### Push

#### `ee_direction_3d`

Evidence-based baseline:

- `L0 = 0.652`
- peak `0.817 @ L11`

Expected under shuffle:

- peak should drop noticeably if temporal ordering matters
- if curve remains near original, current result is likely dominated by framewise or posture cues

#### `ee_speed`

Expected under shuffle:

- weaker drop than direction is plausible
- but a substantial drop would strengthen the "temporal" interpretation

### Strike

#### `ee_direction_3d`

Evidence-based baseline:

- peak `0.849 @ L22`

Expected under shuffle:

- large degradation is expected if the late-layer PEZ-like signal depends on motion sequence

#### `object_direction_3d`

Evidence-based baseline:

- peak `0.813 @ L12`

Expected under shuffle:

- this is the strongest temporal test
- if object direction survives shuffle, the current result is likely not truly temporal

## Decision Criteria

Observation-based interpretation will follow:

- `Strong temporal evidence`:
  - peak drop `>= 0.15` absolute, or
  - peak layer shifts later while the curve weakens materially
- `Mixed evidence`:
  - peak drop in `[0.05, 0.15)`
- `Mostly static correlate`:
  - peak drop `< 0.05` and shape remains similar

These thresholds are for interpretation only; raw numbers will be reported directly.

## Storage Plan

Storage is the main operational risk.

### Constraint

At Nightshift2 start:

- available space on `/mnt/md1/solee`: `199G`

That is not enough to hold:

- existing raw token caches, plus
- full shuffled caches for Push and Strike

### Mitigation

Before deleting any raw cache needed for later CKA:

1. keep the committed probe verdicts as the scientific baseline
2. save compact CKA-ready summaries later in Round 2 if needed
3. recycle raw caches sequentially for shuffle extraction

Round 1 execution plan under this constraint:

1. run Push shuffled extraction/probe
2. delete temporary shuffled Push cache after verdict
3. run Strike shuffled extraction/probe
4. preserve only the final verdict artifacts, not all intermediate raw caches

If further storage pressure appears:

- original Push raw cache can be deleted after shuffled Push is complete because its probe results are already committed
- original Strike raw cache can be deferred until after Strike shuffled verdict lands

## Open Review Questions for Claude

1. Is window-level shuffle sufficient, or should there also be a control with a fixed reversed order?
2. For F5 verdict, is absolute peak drop the right primary metric, or should area-under-curve difference be added?
3. Given storage pressure, is it acceptable to treat the already committed original probe curves as the baseline rather than preserving all original raw caches through Round 1?

## Codex Response to Anticipated Critique

1. Reversed-order is weaker than random shuffle because it preserves monotonicity; random shuffle is the stronger causality test.
2. Peak drop and layer shift are easier to interpret and map directly to PEZ claims; AUC can be added later if needed.
3. Yes. The scientific comparison is against the committed original Phase 2d curves, not against raw cache bytes.
