# F5 Frame Shuffle Verdict

## Scope

This verdict evaluates the strongest current PhysProbe transfer claim under the Round 1 frame-shuffle stress test:

- task: `push`
- model: `V-JEPA 2 Large`
- feature recipe: `resid_post + temporal_last_patch + token_patch flatten`
- probe recipe: `trainable 20-HP sweep + 5-fold GroupKFold + zscore`
- targets:
  - `ee_direction_3d`
  - `ee_speed`

Original baseline was rerun with 3 probe seeds:

- `42`
- `123`
- `2024`

Shuffled extraction was run with 2 frame-order seeds:

- shuffle seed `42`
- shuffle seed `123`

Each shuffled extraction was probed with the same 3 probe seeds, yielding 6 paired shuffled comparisons.

## Main Claim

Frame shuffle provides **mixed but genuine temporal evidence** for `push / ee_direction_3d`.

The strongest evidence is **not** a large peak-`R^2` collapse, but a **robust peak-layer shift**:

- original peak near `L11-L13`
- shuffled peak near `L22-L23`

This is consistent across both shuffled extraction seeds and all probe seeds.

## Killer Evidence Chain

### E1. Original baseline is tight across probe seeds

`ee_direction_3d` original baseline:

- `L0 = 0.648 +/- 0.007`
- `L8 = 0.804 +/- 0.002`
- `peak = 0.816 +/- 0.001`
- `peak layer = 11.7 +/- 0.9`

This rules out a "seed artifact" explanation for the original curve.

### E2. Shuffled direction consistently moves the peak late

`ee_direction_3d` shuffled:

- shuffle `42`: `peak = 0.730 +/- 0.001`, `peak layer = 22.7 +/- 0.5`
- shuffle `123`: `peak = 0.738 +/- 0.001`, `peak layer = 22.0 +/- 0.0`

Paired against the original baseline:

- `delta_peak ~= 0.078 - 0.086`
- `delta_Lpez (L11) ~= 0.086 - 0.089`
- `delta_peak_layer ~= +10.3 to +11.0`

The peak-layer shift is the most robust signal in the whole experiment.

### E3. Train `R^2` remains at ceiling

`ee_direction_3d`:

- original train `R^2` near `1.0`
- shuffled train `R^2` also near `1.0`

Therefore the shuffled result is **not** explained by a trivial optimization failure.

The representation still supports fitting, but the validation-optimal layer structure changes substantially.

## Quantitative Summary

Source tables:

- `artifacts/results/frame_shuffle_summary.csv`
- `artifacts/results/frame_shuffle_paired_deltas.csv`

### `ee_direction_3d`

| condition | L0 | L8 | peak | peak layer | last |
|---|---:|---:|---:|---:|---:|
| original | `0.648 +/- 0.007` | `0.804 +/- 0.002` | `0.816 +/- 0.001` | `11.7 +/- 0.9` | `0.811 +/- 0.002` |
| shuffle42 | `0.589 +/- 0.005` | `0.716 +/- 0.001` | `0.730 +/- 0.001` | `22.7 +/- 0.5` | `0.729 +/- 0.002` |
| shuffle123 | `0.595 +/- 0.008` | `0.716 +/- 0.002` | `0.738 +/- 0.001` | `22.0 +/- 0.0` | `0.734 +/- 0.003` |

Interpretation:

- peak drop is modest (`~9-11%`)
- but the peak-layer shift is large and consistent
- this is **strong qualitative temporal evidence** with **moderate quantitative degradation**

### `ee_speed`

| condition | L0 | L8 | peak | peak layer | last |
|---|---:|---:|---:|---:|---:|
| original | `0.707 +/- 0.027` | `0.930 +/- 0.003` | `0.934 +/- 0.003` | `11.0 +/- 1.6` | `0.917 +/- 0.002` |
| shuffle42 | `0.793 +/- 0.014` | `0.912 +/- 0.004` | `0.914 +/- 0.003` | `9.0 +/- 0.8` | `0.906 +/- 0.003` |
| shuffle123 | `0.802 +/- 0.006` | `0.911 +/- 0.001` | `0.913 +/- 0.000` | `10.0 +/- 0.8` | `0.905 +/- 0.006` |

Interpretation:

- peak drop is small (`~0.02`)
- `L0` increases under shuffle by `~0.086 to 0.095`
- this does **not** support a clean temporal-causality story for `ee_speed`

## Why Does `ee_speed` L0 Increase?

Observation:

- `ee_speed` is the anomalous target
- shuffle raises shallow decode while leaving peak nearly intact

Evidence-based facts:

- train `R^2` also stays near ceiling under shuffle
- peak layer shifts slightly earlier, not later
- direction behaves differently under the same manipulation

Most plausible hypothesis:

- `temporal_last_patch` introduces a bias toward whatever motion evidence lands in the final temporal bucket
- random frame order can place high-speed frames into that bucket more often, effectively acting like a hard augmentation for speed magnitude

Alternative hypothesis:

- `ee_speed` is already dominated by framewise arm pose/blur cues, so destroying temporal order barely matters

Current judgment:

- this is a reviewer-sensitive anomaly
- `ee_speed` should **not** be used as the flagship causal-temporal result

## Reviewer Attack and Response

### Attack

"This is not PEZ; deeper layers just have more information. Your frame shuffle barely changes peak `R^2`."

### Response

That criticism is valid for `ee_speed`, but incomplete for `ee_direction_3d`.

For `ee_direction_3d`, shuffle does two nontrivial things simultaneously:

1. lowers the score at the original PEZ layer (`L11`) by `~0.086-0.089`
2. pushes the best layer to the end of the network (`L22-L23`)

If this were only generic deeper-is-better accumulation, the original and shuffled peak layers would not be expected to separate so dramatically and consistently.

## Counter-Evidence Scenario

This round would fail as a temporal claim if:

- shuffled direction had stayed near `L11-L13`, or
- the late peak shift appeared only for one shuffle seed, or
- the original baseline itself were unstable across probe seeds

None of those happened.

## Final Verdict

### `ee_direction_3d`

**Mixed temporal evidence, strong enough to matter.**

- Quantitatively: modest degradation
- Qualitatively: large, robust late-layer shift

This is best described as:

- **temporal order shapes where direction is best decoded**
- not
- **temporal order fully determines whether direction is decodable at all**

### `ee_speed`

**Mostly static/framewise under the current recipe.**

It is not a strong temporal-causality result.

## Seed-2024 Extraction Decision

Decision:

- do **not** run a third shuffled extraction before Round 2

Reason:

- two shuffled extraction seeds already agree closely on the main finding:
  - `ee_direction_3d` peak `~0.73-0.74`
  - peak layer `~22-23`
- the next highest-value experiment is now Round 2 CKA
- a third shuffled extraction remains a valid follow-up if reviewers demand more extraction-level variance

## What This Means for the Paper Story

Current strongest story is no longer:

- "all manipulation kinematic signals are temporally causal"

It is:

- "3D direction-like manipulation signals show temporally sensitive layer organization, while speed remains more compatible with static/framewise cues"

That is a narrower claim, but a more defensible one.

## Next Experiment

Immediate next step:

- Round 2 F3 CKA across tasks

Reason:

- if PEZ-like tasks align around the same layers, that would strengthen the "task-general kinematic layer" story

Follow-up counter-experiment after CKA if needed:

- single-frame/static-only control for `push / ee_direction_3d`
