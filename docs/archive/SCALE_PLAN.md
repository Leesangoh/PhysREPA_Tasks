# SCALE_PLAN

## Goal

Run a model-scale PEZ experiment on **Push / `ee_direction_3d`** using the exact best-recipe pipeline:

- `resid_post`
- `temporal_last_patch`
- token-patch features
- `trainable` 20-HP sweep
- 5-fold `GroupKFold` by `episode_id`
- `zscore`

The main scientific question is whether the **peak layer shifts systematically with model scale**:

- `V-JEPA 2 Large` (`24` layers, `1024` dim)
- `V-JEPA 2 Giant` (`40` layers, `1408` dim)
- `V-JEPA 2 Huge` (`32` layers, `1280` dim)

The intended paper contribution is a **Figure 6 analog for manipulation kinematics**:

> Does the manipulation-direction PEZ move deeper as model capacity increases?


## Current Audit

### Checkpoints

Available locally at `/mnt/md1/solee/checkpoints/vjepa2/`:

- `vitl.pt` (`4.8G`)
- `vitg.pt` (`16G`)
- `vith.pt` (`9.7G`)

### Existing code support

#### `extract_token_features.py`

- `large`: supported
- `giant`: supported
- `huge`: **not yet supported**

Current `MODEL_CONFIGS` only includes:

- `large -> vit_large, depth=24, dim=1024`
- `giant -> vit_giant_xformers, depth=40, dim=1408`

`vjepa2` upstream **does** expose `vit_huge`, and the local codebase confirms:

- factory: `vit_huge`
- depth: `32`
- embed dim: `1280`

So the blocker is not model availability; it is a small local code extension.

#### `probe_physprobe.py`

- `large`: supported
- `giant`: supported
- `huge`: **not yet supported**

Current `MODEL_CONFIGS` only includes:

- `large -> tag vitl, num_layers 24, dim 1024`
- `giant -> tag vitg, num_layers 40, dim 1408`

### Disk state

Current free space on `/mnt/md1/solee`: about `184G`.

Large token-patch pretrained Push cache is **not** currently present.

Large caches still occupying space:

- `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_randominit_seed0/push` ~ `1006G`
- `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike` ~ `1.5T`

Implication:

- `giant` and `huge` scale runs are **not** possible as a parallel cached setup.
- The scale experiment must be run with **strict storage rotation**.


## Scientific Hypotheses

### H_scale_1

Pretrained `ee_direction_3d` remains PEZ-like at all three scales.

Success criterion:

- each model reaches peak `R^2 >= 0.70`
- peak depth lies in the broad mid/later band `0.20 <= depth <= 0.75`

### H_scale_2

The peak layer shifts deeper with scale.

Operational prediction:

- `Large` peak depth < `Huge` peak depth
- ideally `Large < Giant < Huge`

This does not require monotonic absolute `R^2`; the critical variable is **where** the best decoding occurs.

### H_scale_3

Scale changes **layer location** more strongly than it changes whether the signal exists.

Expected pattern:

- strong signal across all scales
- systematic movement of the emergence/peak layer


## Why this experiment matters

This is currently the most direct oral-tier addition because it addresses a reviewer question that remains open:

> Is the observed manipulation PEZ just one model's idiosyncratic curve, or is it a scale-sensitive representational phenomenon?

If the answer is scale-sensitive, then the paper moves from:

- "V-JEPA 2 Large shows PEZ-like kinematic emergence"

to:

- "Manipulation PEZ obeys a model-scale law inside the V-JEPA 2 family."


## Required code changes before launch

### 1. Add `huge` to `extract_token_features.py`

Add:

- `factory = vit_huge`
- `checkpoint = /mnt/md1/solee/checkpoints/vjepa2/vith.pt`
- `embed_dim = 1280`
- `depth = 32`
- `img_size = 256`

No other recipe changes.

### 2. Add `huge` to `probe_physprobe.py`

Add:

- `tag = vith`
- `num_layers = 32`
- `dim = 1280`

### 3. No methodological changes

Do **not** change:

- pooling
- normalization
- CV
- solver
- target definition

This is a pure scale ablation, not a new recipe search.


## Storage-safe execution plan

Because free space is currently too low, the run must be staged.

### Stage 0. Free space

Delete caches whose scientific value is already committed:

Priority order:

1. `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_randominit_seed0/push`
2. if needed, `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike`

Do **not** delete:

- CKA snapshots
- current result artifacts

### Stage 1. Reconstruct Large baseline only if needed

Preferred path:

- reuse existing committed `Large` Push `ee_direction_3d` results

Do **not** re-extract `Large` unless a direct apples-to-apples rerun is needed for seed alignment.

### Stage 2. Giant

1. extract Push token-patch cache with `--model giant`
2. run probe for:
   - `ee_direction_3d`
   - optional secondary target: `ee_speed`
3. save outputs
4. summarize verdict
5. delete raw Giant cache after outputs are safely written

### Stage 3. Huge

1. add `huge` support
2. extract Push token-patch cache with `--model huge`
3. run probe for:
   - `ee_direction_3d`
   - optional secondary target: `ee_speed`
4. save outputs
5. summarize verdict
6. delete raw Huge cache after outputs are safely written


## Exact run plan

### Giant extraction

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /isaac-sim/python.sh \
  /home/solee/physrepa_tasks/extract_token_features.py \
  --task push \
  --model giant \
  --capture resid_post \
  --pooling temporal_last_patch \
  --output-root /mnt/md1/solee/features/physprobe_vitg_tokenpatch
```

### Giant probe

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /isaac-sim/python.sh \
  /home/solee/physrepa_tasks/probe_physprobe.py \
  --task push \
  --model giant \
  --feature-type token_patch \
  --feature-root /mnt/md1/solee/features/physprobe_vitg_tokenpatch \
  --targets ee_direction_3d ee_speed \
  --device cuda:0 \
  --run-tag scale_giant
```

### Huge extraction

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /isaac-sim/python.sh \
  /home/solee/physrepa_tasks/extract_token_features.py \
  --task push \
  --model huge \
  --capture resid_post \
  --pooling temporal_last_patch \
  --output-root /mnt/md1/solee/features/physprobe_vith_tokenpatch
```

### Huge probe

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /isaac-sim/python.sh \
  /home/solee/physrepa_tasks/probe_physprobe.py \
  --task push \
  --model huge \
  --feature-type token_patch \
  --feature-root /mnt/md1/solee/features/physprobe_vith_tokenpatch \
  --targets ee_direction_3d ee_speed \
  --device cuda:0 \
  --run-tag scale_huge
```


## Outputs

### Results

- `artifacts/results/probe_push_ee_direction_3d_giant_scale_giant.csv`
- `artifacts/results/probe_push_ee_speed_giant_scale_giant.csv`
- `artifacts/results/probe_push_ee_direction_3d_huge_scale_huge.csv`
- `artifacts/results/probe_push_ee_speed_huge_scale_huge.csv`

### Summary

- `artifacts/results/scale_law_verdict.md`
- `artifacts/results/scale_law_summary.csv`

### Figures

- `artifacts/figures/scale_law_push_direction.png`
- `artifacts/figures/scale_law_push_speed.png`


## What the verdict must report

For each model:

- `L0`
- `L8` (or nearest early reference for deeper models)
- `peak R^2`
- `peak layer`
- `peak depth = peak_layer / num_layers`

Primary comparison:

- `Large vs Giant vs Huge` on `ee_direction_3d`

Secondary comparison:

- `Large vs Giant vs Huge` on `ee_speed`


## Reviewer-facing interpretations

### If peak depth increases with scale

This is the strongest outcome.

Claim:

> Manipulation-direction PEZ is scale-sensitive inside the V-JEPA 2 family.

This would materially strengthen the oral-tier story.

### If absolute `R^2` changes but peak depth does not

Claim:

> Scale changes representation quality but not where the computation appears.

This is still useful, but weaker.

### If neither depth nor quality changes materially

Claim:

> The manipulation PEZ observed in Large is robust but not scale-sensitive.

This is publishable but not a headline oral result.


## Risks

### Risk 1. Storage exhaustion

Mitigation:

- sequential extract -> probe -> delete loop
- no simultaneous Giant + Huge raw caches

### Risk 2. Huge support bug

Mitigation:

- add `huge` support first
- one-episode dry-run before full extraction

### Risk 3. Probe inconsistency across scales

Mitigation:

- keep recipe identical
- no per-model tuning


## Claude review checklist

- Is `ee_direction_3d` alone sufficient as the main scale-law target?
- Should `ee_speed` remain a secondary target or be dropped to save time?
- Is deleting the `strike` token-patch cache acceptable once the current paper figures are preserved?
- Do we want a `Large` rerun for exact seed-matched comparability, or is the existing committed baseline sufficient?


## Recommendation

Proceed after Claude review with the following strict order:

1. add `huge` code support
2. free disk
3. run `Giant`
4. delete Giant raw cache
5. run `Huge`
6. write `scale_law_verdict.md`

This is the highest-impact next experiment that does not require new data collection.
