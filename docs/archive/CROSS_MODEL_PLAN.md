# CROSS_MODEL_PLAN

## Goal

Run the highest-leverage oral-tier follow-up: a **cross-model manipulation PEZ benchmark** that tests whether the current results are specific to V-JEPA 2 or reflect a broader property of pretrained visual/video encoders.

The benchmark is intentionally narrow:

- **Push / `ee_direction_3d`**: main positive manipulation-direction target
- **Strike / `object_direction_3d`**: hardest and most convincing 3D rescue target
- **Push / `ee_speed`**: control target that should remain highly decodable

The primary comparison is:

- **V-JEPA 2 Large**: predictive video model, current best baseline
- **VideoMAE-L**: masked video reconstruction baseline
- **DINOv2-L**: image-only self-supervised baseline
- **Optional Hiera-L**: hierarchical MAE image baseline

The paper question is:

> Does manipulation PEZ require predictive video pretraining, or does any strong visual backbone produce the same curves?


## 1. Exact checkpoints

### A. V-JEPA 2 Large

Already local:

- checkpoint: `/mnt/md1/solee/checkpoints/vjepa2/vitl.pt`
- source repo: local clone at `/home/solee/vjepa2`

Used as the fixed reference baseline.


### B. VideoMAE-L

Chosen as the **first non-V-JEPA baseline** because it is also a video model and therefore tests pretraining objective rather than simply modality.

Checkpoint:

- Hugging Face model: `MCG-NJU/videomae-large`
- model page: `https://huggingface.co/MCG-NJU/videomae-large`

Planned download command:

```bash
env HF_HOME=/mnt/md1/solee/hf_cache /isaac-sim/python.sh -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='MCG-NJU/videomae-large', local_dir='/mnt/md1/solee/checkpoints/cross_model/videomae-large', local_dir_use_symlinks=False)"
```

Expected need:

- **requires network approval**
- **not currently local**


### C. DINOv2-L

Chosen as the **static-image control**.

Checkpoint:

- Hugging Face model: `facebook/dinov2-large`
- model page: `https://huggingface.co/facebook/dinov2-large`

Planned download command:

```bash
env HF_HOME=/mnt/md1/solee/hf_cache /isaac-sim/python.sh -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/dinov2-large', local_dir='/mnt/md1/solee/checkpoints/cross_model/dinov2-large', local_dir_use_symlinks=False)"
```

Expected need:

- **requires network approval**
- **not currently local**


### D. Optional Hiera-L

Optional because it is lower priority than VideoMAE-L and DINOv2-L. It is useful if we want a second image-pretrained baseline with hierarchical spatial reduction.

Preferred checkpoint:

- Hugging Face model: `facebook/hiera-large-224-mae-hf`
- model page: `https://huggingface.co/facebook/hiera-large-224-mae-hf`

Alternative if the above integration is awkward:

- `facebook/hiera_base_224.mae_in1k_ft_in1k`
- model page: `https://huggingface.co/facebook/hiera_base_224.mae_in1k_ft_in1k`

Planned download command:

```bash
env HF_HOME=/mnt/md1/solee/hf_cache /isaac-sim/python.sh -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/hiera-large-224-mae-hf', local_dir='/mnt/md1/solee/checkpoints/cross_model/hiera-large-224-mae-hf', local_dir_use_symlinks=False)"
```

Expected need:

- **requires network approval**
- **not currently local**


## 2. Fairness rules

This experiment must be fair where fairness is possible, and explicit where it is not.

### Shared invariants across all models

These must remain fixed:

- same tasks
- same episode set
- same window definition
  - `16` frames per window
  - stride `4`
- same targets
  - Push `ee_direction_3d`
  - Strike `object_direction_3d`
  - Push `ee_speed`
- same probe protocol
  - `trainable` 20-HP sweep
  - 5-fold `GroupKFold` by `episode_id`
  - `zscore`
- same evaluation outputs
  - `L0`
  - `L8` or nearest early reference
  - peak `R^2`
  - peak layer
  - peak depth
  - last-layer `R^2`


### Model-native preprocessing is allowed

Exact patch size, resolution, and normalization **cannot** be made identical across these architectures without introducing artificial handicaps.

So the fairness rule is:

> each model gets its native inference preprocessing, but identical downstream probing and identical task/target definitions.

This means:

- `V-JEPA 2 Large`: `256x256`, native transform
- `VideoMAE-L`: `224x224`, native VideoMAE image processor
- `DINOv2-L`: `224x224`, native DINOv2 image processor
- `Hiera-L`: `224x224`, native Hiera preprocessing


### Video versus image models are not given fake temporal access

This is the most important fairness rule.

- **Video models** (`V-JEPA 2`, `VideoMAE`) receive the full 16-frame window.
- **Image models** (`DINOv2`, optional `Hiera`) receive **only the last frame of each 16-frame window**.

Why:

- repeating a frame 16 times would manufacture fake temporal tokens
- averaging features over all 16 frames would give the image model more temporal evidence than a single-image backbone actually uses
- the current V-JEPA best recipe is already `temporal_last_patch`, so last-frame image control is the cleanest comparison

This must be stated explicitly in the paper:

> image encoders are evaluated as static controls, not as equal-information video baselines.


### Compare fractional depth, not only raw layer index

Layer counts differ:

- V-JEPA 2 Large: `24`
- VideoMAE-L: likely `24`
- DINOv2-L: `24`
- Hiera-L: hierarchical and not directly comparable layer-for-layer

Therefore the main comparison axis should be:

- `peak_depth = peak_layer / num_layers`

Raw peak layer should still be reported, but the interpretation should rely on fractional depth.


## 3. Recipe reuse versus required adaptations

## Reuse exactly

The following should be reused without change:

- task/target loaders
- 5-fold grouped CV
- `trainable` probe with 20-HP sweep
- `zscore`
- artifact layout and CSV schema
- figure generation style


## Adaptations required

### A. New extraction driver

Create a new extractor, e.g.:

- `extract_cross_model_features.py`

Reason:

- `extract_token_features.py` is tightly coupled to V-JEPA 2 residual hooks
- cross-model support should not be bolted into that script in a messy way


### B. Readout definition per model

#### V-JEPA 2 Large

Keep existing best recipe:

- `resid_post`
- `temporal_last_patch`

#### VideoMAE-L

Best-matching analog:

- use encoder hidden states from every transformer block
- use the last temporal slice of patch tokens
- preserve spatial patch tokens

Working name in code:

- `videomae_last_patch`

There is no strict `resid_post` equivalent, so the fairness rule is:

> use the last hidden state after each block, before any classification head.

#### DINOv2-L

Best-matching analog:

- use hidden states from every transformer block
- discard CLS token
- preserve patch tokens from the **last frame only**

Working name in code:

- `dinov2_last_patch`

Again, no strict `resid_post`; use per-block hidden states before the final image-level head.

#### Optional Hiera-L

Best-matching analog:

- request intermediate feature maps
- spatially flatten the final stage patch grid for the last frame
- compare against V-JEPA using fractional depth only

This model is the least fair layerwise comparison because of its hierarchical token reduction. If it is added, it should be framed as:

> architectural robustness test, not a one-to-one layer-depth match.


### C. Probe model config map

`probe_physprobe.py` currently knows:

- `large`
- `giant`
- `huge`

Cross-model probing should not overload these names. Create a new model config namespace, e.g.:

- `vjepa2_large`
- `videomae_large`
- `dinov2_large`
- `hiera_large`

or create a separate cross-model probe driver to avoid contaminating the current V-JEPA-only logic.


## 4. Disk / GPU budget

These are rough planning numbers based on the current V-JEPA token-patch runs.

### V-JEPA 2 Large

Already done.

### VideoMAE-L

Expected:

- 24 layers
- ~1024 dim
- 16-frame window
- 224 resolution

Estimated raw token-patch cache:

- roughly `0.8T - 1.2T` for Push
- similar or slightly smaller than current V-JEPA Large token cache

GPU:

- extraction: `1` A6000
- probe: `1-2` A6000 if memory allows, otherwise sequential seeds

### DINOv2-L

Expected:

- 24 layers
- ~1024 dim
- single last frame only
- 224 resolution

Estimated raw patch cache:

- roughly `0.15T - 0.35T` for Push
- much smaller because no temporal tokens

GPU:

- extraction: `1` A6000
- probe: `1` A6000

### Optional Hiera-L

Expected:

- hierarchical tokens
- likely smaller raw cache than DINOv2-L if we store only final-stage intermediates

Estimated raw patch cache:

- roughly `0.1T - 0.3T`


## 5. Sequential execution plan

This must be storage-safe.

### Stage 0. Free space

Delete completed caches first:

1. `/mnt/md1/solee/features/physprobe_vith_tokenpatch`

Then maintain the rule:

- **extract -> probe -> summarize -> delete raw cache**

No overlapping multi-terabyte caches.


### Stage 1. VideoMAE-L on Push

1. download checkpoint
2. implement extractor support
3. extract Push token-patch cache
4. probe:
   - Push `ee_direction_3d`
   - Push `ee_speed`
5. write `cross_model_videomae_push_verdict.md`
6. delete raw VideoMAE cache

This stage tests the most important question first:

> does another video-pretrained model show the same manipulation direction PEZ?


### Stage 2. VideoMAE-L on Strike

1. extract Strike token-patch cache
2. probe:
   - Strike `object_direction_3d`
3. summarize
4. delete raw cache

This is the decisive object-side comparison.


### Stage 3. DINOv2-L on Push

1. download checkpoint
2. implement extractor support
3. extract last-frame patch cache
4. probe:
   - Push `ee_direction_3d`
   - Push `ee_speed`
5. summarize
6. delete raw cache

This stage establishes the image-only control.


### Stage 4. DINOv2-L on Strike

1. extract last-frame patch cache
2. probe:
   - Strike `object_direction_3d`
3. summarize
4. delete raw cache


### Stage 5. Optional Hiera-L

Only after `VideoMAE-L` and `DINOv2-L` are stable.


## 6. Figure / table specs for the paper

### Main figure

Proposed title:

> Predictive video pretraining is not interchangeable with static or masked-video baselines.

Panels:

#### Panel A. Push `ee_direction_3d`

- overlay:
  - V-JEPA 2 Large
  - VideoMAE-L
  - DINOv2-L
- x-axis: fractional depth
- annotate peak `R^2` and peak depth

#### Panel B. Strike `object_direction_3d`

- same overlay
- this is the highest-leverage object-side comparison

#### Panel C. Push `ee_speed`

- control target
- tests whether all models can decode the easier kinematic variable

#### Panel D. Temporal sensitivity summary

- optional bar plot:
  - original peak depth
  - shuffled peak depth
  - or no shuffle for image models

Interpretation:

- if V-JEPA 2 and VideoMAE both work but DINOv2 fails:
  - temporal/video pretraining matters
- if V-JEPA 2 beats VideoMAE and DINOv2:
  - predictive video pretraining matters specifically
- if all three work similarly:
  - PEZ is broader than V-JEPA 2, which is still a valuable result


### Main table

Columns:

- model
- modality (`video` / `image`)
- objective (`predictive` / `masked reconstruction` / `contrastive / SSL image`)
- target
- `L0`
- `L8`
- peak `R^2`
- peak layer
- peak depth
- last

Rows:

- Push `ee_direction_3d`
- Strike `object_direction_3d`
- Push `ee_speed`


## 7. Risks and mitigations

### Risk 1. Cross-model extraction semantics are not perfectly matched

Mitigation:

- compare only pre-head hidden states
- compare fractional depth, not raw layer count only
- clearly label image models as static controls


### Risk 2. VideoMAE-L gives strong results, weakening the V-JEPA-specific story

Mitigation:

- that is still a paper-strengthening result
- the claim would shift from
  - “V-JEPA 2 uniquely shows manipulation PEZ”
  to
  - “manipulation PEZ is a property of temporally pretrained video models”


### Risk 3. DINOv2-L also performs strongly

Mitigation:

- then the strongest claim becomes:
  - direction may be partially recoverable from strong image features
- but Push shuffle and event results still isolate temporal dependence
- the paper becomes more nuanced, not weaker


### Risk 4. Hiera integration complexity slows the project

Mitigation:

- treat Hiera as optional
- do not block `VideoMAE-L` and `DINOv2-L` on Hiera


### Risk 5. Disk pressure returns

Mitigation:

- strict extract -> probe -> delete loop
- no concurrent raw caches for different models


## 8. Recommendation

Run **VideoMAE-L first**, then **DINOv2-L**.

Reason:

- `VideoMAE-L` is the cleanest first test of whether the phenomenon depends on
  predictive video pretraining specifically, or survives in a masked-video
  baseline.
- `DINOv2-L` is the cleanest static-image control.

That pair alone is enough to transform the current paper from:

- a strong V-JEPA 2 case study

into:

- a comparative study of how pretraining objective shapes manipulation PEZ.
