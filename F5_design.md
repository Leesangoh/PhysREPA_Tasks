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

## [NEURIPS MODE]

This round is designed not only to detect a difference, but to support or refute a paper-quality claim under reviewer scrutiny.

Mandatory standards for this round:

- multi-seed shuffling, not single-seed only
- effect-size reporting, not just curve plots
- explicit alternative-hypothesis handling
- paired statistical comparison where feasible

## [ORAL MODE]

### Main claim for Round 1

If frame shuffle materially degrades the strongest manipulation direction curves, then the current PhysProbe PEZ-analog signals depend on temporal ordering rather than only static appearance.

### Killer evidence chain

To support that claim, Round 1 needs all of the following:

1. **Reproducible degradation across shuffle seeds**
   - not a one-seed accident
2. **Material effect size**
   - not just a tiny but statistically non-zero difference
3. **Target specificity**
   - stronger degradation for direction-like targets than for trivial/static correlates

### Counter-evidence scenarios

1. Shuffled curves stay near original
   - interpretation: static/framewise correlate dominates
2. Only one target degrades while others do not
   - interpretation: claim becomes target-specific, not a broad manipulation PEZ claim
3. Results are highly unstable across seeds
   - interpretation: current evidence is too brittle for an oral-level story

### Reviewer attack to answer

Reviewer attack:

- "This is just deeper-is-better. Your model has more information in later layers, and your direction curves are not proof of temporal encoding."

Round-1 answer if successful:

- shuffle preserves per-frame appearance while destroying temporal order
- if the same probe recipe collapses under shuffle, then later-layer success cannot be explained by static appearance alone

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

Use the seed set:

- `42`
- `123`
- `2024`

For each seed, derive a deterministic per-window seed from `(global_seed, episode_id, window_start)`.

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
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_shuffled_seed{seed}/push`
- Strike shuffled cache:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_shuffled_seed{seed}/strike`

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

## Statistical Comparison Plan

### Primary effect sizes

For each target and seed:

- `delta_peak = peak(original) - peak(shuffled)`
- `delta_Lpez = R^2_original(L_pez) - R^2_shuffled(L_pez)`
- `delta_L0 = R^2_original(L0) - R^2_shuffled(L0)`
- `delta_peak_layer = peak_layer(shuffled) - peak_layer(original)`

Where `L_pez` is task/target specific:

- Push `ee_direction_3d`: use `L11`
- Push `ee_speed`: use original peak layer
- Strike `ee_direction_3d`: use `L22`
- Strike `object_direction_3d`: use `L12`

### Uncertainty

If fold-level scores are available:

- use paired bootstrap over folds
- report bootstrap CI for `delta_peak` and `delta_Lpez`

If only per-seed summaries are available:

- aggregate across the 3 shuffle seeds
- treat seed as the uncertainty axis
- report mean +/- std across seeds

### Decision rule

The round supports a temporal-causality claim only if:

- the shuffled degradation is consistent across seeds, and
- the effect is material in magnitude, not just statistically non-zero

### Oral-mode threshold interpretation

- `>= 30%` degradation from original at the key PEZ layer or peak:
  - strong causal temporal evidence
- `10% or less` degradation:
  - strong evidence for static/framewise contribution
- `15-25%` degradation:
  - mixed temporal+static interpretation

### Multiple-comparison plan

If layer-wise significance testing is performed:

- control family-wise error with Bonferroni over 24 layers, or
- report FDR-adjusted `q` values as a secondary analysis

Primary narrative will still rely on effect sizes and CIs, not only `p` values.

## Reviewer Attack Model

Primary reviewer attack:

- "This is not PEZ; deeper layers simply contain more information, and shuffle does not matter."

Counter-experiment in this round:

- multi-seed frame shuffle with exact same token-patch probe

Interpretation rule:

- if shuffle does not materially degrade the curve, the default interpretation is static/framewise correlation
- if shuffle degrades the curve strongly and consistently, temporal causality is supported

## Story Update Goal

If Round 1 succeeds, the evidence chain becomes:

1. token-patch probing reveals PEZ-like direction emergence
2. fake targets remain negative
3. frame shuffle degrades the same targets materially

That would move the claim from "representation contains information" to "representation depends on temporal ordering."

If Round 1 is only mixed:

- the story remains publishable, but the claim becomes:
  - "manipulation PEZ-like signals combine temporal and static cues"
  - not a pure causal temporal mechanism

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

1. run one seed at a time
2. probe immediately after extraction for that seed
3. keep compact CSV/statistical outputs
4. delete temporary shuffled cache before the next seed
5. preserve only final verdict artifacts, not all intermediate raw caches

If further storage pressure appears:

- original Push raw cache can be deleted after shuffled Push is complete because its probe results are already committed
- original Strike raw cache can be deferred until after Strike shuffled verdict lands

## Open Review Questions for Claude

1. Is window-level shuffle sufficient, or should there also be a control with a fixed reversed order?
2. Should `delta_peak` or `delta_Lpez` be the primary headline statistic?
3. Is 3 seeds enough given the storage ceiling, or is a fourth seed necessary for reviewer confidence?

## Codex Response to Anticipated Critique

1. Reversed-order is weaker than random shuffle because it preserves monotonicity; random shuffle is the stronger causality test.
2. `delta_Lpez` maps directly onto the PEZ claim, while `delta_peak` captures total degradation; both should be reported.
3. Given the current `95%` disk usage, 3 seeds run sequentially with cache recycling is the highest-rigor feasible design without stalling the whole nightshift.
