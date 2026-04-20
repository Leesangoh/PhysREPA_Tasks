# Round 2 Verdict: Cross-Task CKA Rejects the "Task-General PEZ Layer" Hypothesis

## Scope

This verdict analyzes the completed Push-Strike CKA run:

- input snapshots:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_cka_bal600/push`
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_cka_bal600/strike`
- model:
  - V-JEPA 2 Large
- representation:
  - `resid_post`
  - `temporal_last_patch`
  - flattened patch tokens
- sample count:
  - `600` episodes per task
- metric:
  - linear CKA with bootstrap CIs

Artifacts:

- `artifacts/results/cka_push_strike/cka_cross_task.csv`
- `artifacts/results/cka_push_strike/cka_cross_task_lines.png`
- `artifacts/results/cka_push_strike/cka_cross_task_heatmap.png`

## Main hypothesis tested

Original Round 2 hypothesis:

> The layers that support the strongest PEZ-like kinematic decoding in Push and
> Strike should also be the layers with the highest cross-task representation
> similarity.

This was the "task-general kinematic layer" hypothesis.

## Verdict

**Rejected.**

The highest Push-Strike similarity occurs in **early layers**, not in the
PEZ-like layers identified by probing.

## Evidence

### 1. Peak Push-Strike CKA is early, not mid-depth

Peak same-layer cross-task similarity:

- `L3 CKA = 0.6459`
- bootstrap `95% CI = [0.6058, 0.6860]`

This is the global maximum in `push-strike`.

### 2. PEZ-zone CKA is lower than early CKA

PEZ-zone summary (`L8-L13`):

- mean `CKA = 0.5645`
- min/max over the zone:
  - `0.5567` to `0.5693`

By comparison, early-layer summary (`L0-L4`) is:

- mean `CKA = 0.6158`

Difference:

- `early_mean - pez_mean = 0.0513`

### 3. Late layers do not recover the early peak

Late summary (`L20-L23`):

- mean `CKA = 0.5671`

This is essentially flat with the PEZ-zone values and still lower than early
layers:

- `early_mean - late_mean = 0.0487`

### 4. The early-vs-PEZ gap is not just noise

Layer-wise bootstrap intervals show a consistent gap:

- `L3: [0.6058, 0.6860]`
- `L8: [0.5309, 0.6015]`
- `L9: [0.5321, 0.5983]`
- `L10: [0.5338, 0.5999]`
- `L11: [0.5225, 0.5916]`
- `L12: [0.5294, 0.5952]`
- `L13: [0.5327, 0.6020]`

Interpretation:

- the `L3` lower bound is above the upper bound of every PEZ-layer CI except
  `L13`, where the intervals are still essentially disjoint up to rounding
- this is strong evidence that cross-task similarity is genuinely higher in the
  early visual/generic regime than in the PEZ-like decoding regime

This is not merely a visual impression from the line plot.

## Updated interpretation

The original "task-general PEZ layer" story does **not** fit the data.

A better evidence-based interpretation is:

1. **Early layers are generic**
   - Push and Strike still share strong visual/spatiotemporal structure there
   - this yields the highest cross-task CKA
2. **PEZ-like layers are more task-specialized**
   - they support strong kinematic decoding within each task
   - but Push and Strike are **less** similar to each other there
3. **Late layers flatten into a plateau**
   - cross-task similarity does not re-peak
   - this argues against a simple “shared late semantic layer” story

In short:

> the probing results suggest that PEZ-like layers are where useful task-specific
> kinematic structure becomes more separable, not where tasks become maximally
> similar.

## Why this is still valuable

This negative result is scientifically useful.

If the hypothesis had been supported, the story would have been:

- "manipulation has a shared task-general PEZ layer"

But the actual result supports a more nuanced and arguably more interesting
story:

- **generic early layers**
- **task-specializing PEZ layers**
- **late plateau rather than re-convergence**

That is a stronger mechanism-level claim than merely reporting that two tasks
both show positive direction curves.

## Reviewer attack and response

### Reviewer attack

> "If CKA is lower in the PEZ zone, doesn't that mean there is no PEZ at all?"

### Response

No.

PEZ is defined here by **layer-wise probe behavior**, not by maximum
cross-task similarity.

The probing results already showed:

- Push `ee_direction_3d`: PEZ-like
- Strike `ee_direction_3d`: PEZ-like
- Strike `object_direction_3d`: strong 3D rescue

CKA answers a different question:

- whether the successful layers are also **shared across tasks**

The answer is **no**.

So the correct conclusion is not "PEZ is false", but:

> PEZ-like kinematic emergence exists, but it is more task-specific than
> originally hypothesized.

## Critical self-review

### Would a reviewer say this is just deeper-is-better?

For CKA specifically: no, the data actually argues *against* a monotonic
deeper-is-better story.

- CKA peaks early at `L3`
- then drops and stays flat through PEZ and late layers

So the CKA result is not a generic depth accumulation result.

### Strongest alternative hypothesis

Alternative:

> Push and Strike share low-level appearance statistics, but diverge once the
> representation begins to encode task-specific dynamics.

Current verdict:

- this alternative is **supported**

### What experiment best challenges that alternative next?

The next strongest test is **Round 4 random-init baseline**, not split-by-value.

Reason:

- the main reviewer vulnerability now is not value stratification
- it is whether the observed task-specific PEZ curves require learned V-JEPA 2
  structure at all

If random-init features fail to reproduce the direction curves, that would
support the claim that V-JEPA pretraining creates a task-specific manipulation
kinematic representation.

## Recommended next step

**Choose Round 4 random-init baseline before Round 3 split-by-value.**

Why:

1. It is more central to the reviewer question "is this learned or generic?"
2. After CKA, the story is now about specialization, not global alignment
3. Split-by-value is secondary until the representation itself is shown to
   require pretraining

## Bottom line

Round 2 falsifies the cleanest cross-task story:

> there is **not** a single task-general PEZ layer shared by Push and Strike.

The better current narrative is:

> manipulation PEZ is **task-specific specialization**, emerging after a more
> generic early visual regime and before a flat late plateau.
