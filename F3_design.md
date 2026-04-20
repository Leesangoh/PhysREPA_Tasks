# F3 Design: Cross-Task CKA for a Task-General Kinematic Layer

## Scope

Round 2 of `NIGHTSHIFT2_PROTOCOL.md` tests whether the strongest manipulation
signals found so far align across tasks at specific layers, rather than merely
improving monotonically with depth.

Primary tasks for this round:

- `push`
- `strike`

Optional extension after the first verdict:

- `reach`

Primary representation source:

- original token-patch cache
- `resid_post`
- `temporal_last_patch`
- ViT-Large, `24` layers

Balanced comparison size:

- `600` episodes per task

## [NEURIPS MODE]

This round is not just a visualization exercise.

It must answer a reviewer-facing question:

> Do the positive Push/Strike direction results point to a shared
> task-general kinematic layer, or are they just two independent examples of
> deeper-is-better decoding?

## [ORAL MODE]

### Main claim for Round 2

If cross-task similarity peaks near the same layers that maximize kinematic
decodability, then V-JEPA 2 Large contains a task-general kinematic subspace,
not just task-specific memorization.

### Killer evidence chain

To support that claim, Round 2 needs all of the following:

1. **High same-layer Push-Strike CKA in the PEZ-like region**
   - specifically around the layers where `ee_direction_3d` or
     `object_direction_3d` peak in the probing results
2. **Lower CKA in shallow layers**
   - to rule out a trivial "everything is similar everywhere" story
3. **Lower CKA again or flattening in very late layers**
   - to separate a task-general kinematic band from generic depth accumulation

### Counter-evidence scenarios

1. Push-Strike CKA is flat across depth
   - interpretation: no privileged kinematic layer
2. Push-Strike CKA simply rises monotonically to the output
   - interpretation: deeper-is-better remains a viable explanation
3. CKA is high for all task pairs and all layers
   - interpretation: feature geometry is too globally similar for layer-specific
     claims

### Reviewer attack to answer

Reviewer attack:

> "Your PEZ-like direction curves may just reflect stronger late-layer
> representations. Why think there is a shared kinematic layer at all?"

Round-2 answer if successful:

- because cross-task similarity is not flat
- and not purely monotonic
- but concentrated near the same middle/deeper layers that maximize decoding

## Data Plan

### Inputs

Balanced compact snapshots:

- Push:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_cka_bal600/push`
- Strike:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_cka_bal600/strike`

Each snapshot contains:

- `episode_ids.npy`
- `layer_{00..23}.dat`
- `meta.txt`

Representation shape per layer:

- `600 x 262144`

### Why compact snapshots

The raw token caches are too large to keep for repeated CKA analysis after F5
shuffle runs. Compact snapshots preserve the per-episode, per-layer flattened
token-patch representation while making CKA feasible on a balanced sample.

## CKA Formula

Use **linear CKA** with centered Gram matrices.

For feature matrices `X` and `Y` of shape `N x D`:

1. row-center `X` and `Y`
2. compute Gram matrices:
   - `Kx = X X^T`
   - `Ky = Y Y^T`
3. double-center the Gram matrices
4. compute:
   - `CKA(X, Y) = <Kx, Ky>_F / (||Kx||_F ||Ky||_F)`

Expected sanity properties:

- symmetry: `CKA(X, Y) == CKA(Y, X)` up to numerical tolerance
- self-similarity: `CKA(X, X) = 1`

## Statistical Plan

### Bootstrap

Use bootstrap resampling over episode indices.

- `B = 1000` bootstrap resamples
- resample the `600` aligned episode rows with replacement
- recompute CKA per layer per task pair

Report:

- bootstrap mean
- `95%` percentile CI

### Effect-size summaries

For Push-Strike CKA:

- `CKA_L0-L4_mean`
- `CKA_L10-L13_mean`
- `CKA_L20-L23_mean`
- `delta_mid_minus_shallow`
- `delta_mid_minus_late`

### Decision thresholds

Evidence-based interpretation rules:

- if `delta_mid_minus_shallow > 0.05` and CI excludes `0`
  - supports a task-general kinematic layer above shallow baselines
- if `delta_mid_minus_late > 0.03` and CI excludes `0`
  - supports a localized kinematic band rather than monotonic depth gain
- if both deltas are near `0`
  - weak evidence for layer specificity

These thresholds are heuristic and will be reported as such.

## Visualization Plan

### Main figure

Layer-wise line plot:

- `Push-Strike CKA`
- optional `Push-Reach`, `Strike-Reach` later
- shaded `95%` bootstrap CI
- vertical markers at:
  - `L11` (Push `ee_direction_3d` peak)
  - `L12` (Strike `object_direction_3d` peak)
  - `L22` (Strike `ee_direction_3d` peak)

### Secondary figure

Task-pair x layer heatmap:

- rows: task pairs
- columns: layers
- values: bootstrap mean CKA

## Sanity Checks

1. **Dimensional consistency**
   - all task snapshots must share the same `feature_dim`
2. **Episode count parity**
   - compare only balanced sets of `600`
3. **Self-CKA**
   - each task against itself should be `~1.0`
4. **Symmetry**
   - `CKA(push, strike)` and `CKA(strike, push)` should match numerically

## Immediate Execution Plan

1. Preserve the already completed Push `600`-episode snapshot
2. Create an equivalent Strike `600`-episode snapshot
3. Implement `compute_cka.py`
4. Run Push-Strike CKA first
5. Write `cka_cross_task_verdict.md`
6. Only then decide whether Reach adds enough value to justify the extra run

## Why this matters after F5

F5 gave mixed temporal evidence:

- `ee_direction_3d` did not collapse under shuffle
- but its peak layer shifted robustly from `L11-L13` to `L22-L23`

That finding is important but still reviewer-vulnerable.

Round 2 strengthens the story only if it shows that the same kinematic layers
also align across tasks. If CKA does not localize to those layers, the paper
story weakens to "task-specific decodability" rather than "task-general
kinematic emergence."
