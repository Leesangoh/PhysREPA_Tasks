# PEZ to PhysProbe Plan

Internal planning document for transferring the PEZ reproduction methodology from `/home/solee/pez` to manipulation-oriented PhysProbe analysis in `/home/solee/physrepa_tasks`.

This document is based on:
- `/home/solee/pez/README.md`
- `/home/solee/pez/artifacts/results/final_reproduction_report.md`
- `/home/solee/pez/artifacts/results/final_verdict.md`
- `/home/solee/pez/artifacts/results/final_verdict_fig2b.md`
- `/home/solee/pez/artifacts/results/intphys_deep_rootcause.md`
- `/home/solee/pez/artifacts/results/fig2b_failure_analysis.md`
- `/home/solee/pez/artifacts/results/figure6_verdict.md`
- `/home/solee/pez/artifacts/results/figure8_verdict.md`
- `/home/solee/pez/constants.py`
- `/home/solee/pez/step1_generate.py`
- `/home/solee/pez/step2_extract.py`
- `/home/solee/pez/step3_probe.py`
- `/home/solee/CLAUDE.md`
- `/home/solee/physrepa_tasks/analysis/probe_sweep_v2.py`
- `/home/solee/physrepa_tasks/analysis/probe_sweep_v3.py`


## Part 1: What the PEZ reproduction actually did

This section is intentionally detailed. The goal is not to summarize the paper abstractly, but to record which concrete choices changed the outcome during reproduction.


### 1. Global PEZ pipeline

The PEZ reproduction converged to a three-stage pipeline:

1. `step1_generate.py`
   - Generate synthetic ball videos and ground-truth labels.
   - Backends:
     - `kubric` (paper-faithful target backend)
     - `pyrender` (fallback only)
   - Physics:
     - PyBullet simulation at `240 Hz`
     - `10` substeps per rendered frame
   - Rendering:
     - Blender/Kubric
     - `16` frames
     - `24 fps`
     - `256 x 256`
   - Camera:
     - overhead perspective
     - floor visible extent `[-8, 8]^2`
     - `camera_z = 10.0`
   - Datasets:
     - velocity: `8 directions x 7 speeds x 7 start positions = 392 clips`
     - acceleration: `8 directions x 5 accelerations x 7 start positions = 280 clips`
   - Labels:
     - screen-space `vx_px`, `vy_px`, `ax_px`, `ay_px`
     - scalar `speed`
     - scalar `accel_magnitude`
     - `direction_rad`
     - projected start coordinates `pos_x_px`, `pos_y_px`

2. `step2_extract.py`
   - Extract frozen V-JEPA 2 hidden representations.
   - Supported models:
     - Large (`24` layers, `1024` dim)
     - Huge (`32` layers, `1280` dim)
     - Giant (`40` layers, `1408` dim)
   - Supported capture conventions:
     - `resid_pre`
     - `resid_post`
   - Supported preprocessing:
     - `resize`
     - `eval_preproc`
   - Supported pooling:
     - `mean`
     - `temporal_last`
     - `temporal_first`
     - `temporal_diff`
     - `temporal_last_patch`
     - `temporal_diff_patch`

3. `step3_probe.py`
   - Perform layer-wise linear probing with grouped 5-fold CV.
   - Supported probe sets:
     - Figure 2(c): `speed`, `direction`, `acceleration`
     - Figure 2(b): `velocity_xy`, `acceleration_xy`
     - axis-wise auxiliaries: `velocity_x`, `velocity_y`, `acceleration_x`, `acceleration_y`
   - Supported direction targets:
     - `sincos`
     - `angle`
     - `vxy`
   - Supported grouping:
     - `position`
     - `condition`
     - `video`
     - `direction`
     - `magnitude`
     - `pixel_region`
     - `spatial_sector`
     - `spatial_cluster`
     - `direction_spatial_sector`
     - `magnitude_spatial_sector`
   - Supported solvers:
     - trainable `20 HP sweep` (Appendix B style)
     - `ridge`
     - `adamw100` weak-probe variant


### 2. Exact synthetic dataset spec used in PEZ reproduction

From `constants.py` and `step1_generate.py`, the final synthetic setup was:

- `N_FRAMES = 16`
- `FPS = 24`
- `RESOLUTION = 256`
- `BALL_RADIUS_M = 0.3`
- velocity directions: `[0, 45, 90, 135, 180, 225, 270, 315]`
- speeds: `[1, 2, 3, 4, 5, 6, 7]` m/s
- accelerations: `[2, 4, 6, 8, 10]` m/s^2
- `N_START_POSITIONS = 7`
- `SEED = 42`
- floor size: `8.0 m`
- camera height: `10.0`
- PyBullet ball mass: `1.0 kg`
- friction and restitution effectively zeroed

Important refinement discovered during reproduction:

- Start positions had to be sampled freshly per `(direction, speed)` or `(direction, acceleration)` pair.
- Reusing one global set of 7 start positions across all conditions was a mismatch and was corrected.

Important limitation:

- Even after fixing PyBullet and start-position sampling, Figure 2(b) remained fragile.
- Therefore dataset realism alone did not explain the full reproduction gap.


### 3. Figure 2(c) Polar: what succeeded

#### Dataset/targets

- Same Kubric ball dataset as above.
- Targets:
  - `speed`
  - `direction`
  - `acceleration magnitude`

#### Feature extraction

The final best Figure 2(c) config was:

- capture: `resid_post`
- transform: `resize`
- pooling: `temporal_last`
- model: `vjepa2_L`

This is important because earlier attempts assumed `resid_pre` would be the most paper-faithful choice. In practice:

- `resid_pre` fixed some layer-0 interpretation issues
- but the best paper-like Figure 2(c) curve came from `resid_post + temporal_last`

#### Probe setup

Best config:

- solver: `trainable`
- HP search:
  - `lr in {1e-4, 3e-4, 1e-3, 3e-3, 5e-3}`
  - `wd in {0.01, 0.1, 0.4, 0.8}`
- CV: `5-fold GroupKFold`
- grouping: `direction_spatial_sector`
- direction target: `angle`
- normalization: `zscore`
- run name:
  - `fig2c_iter11_residpost_tlast_dirsector_angle`

#### Reproduction success criterion

Figure 2(c) was ultimately judged as reproduced because the best run matched the paper's textual criteria:

- `direction` becomes reliably decodable at the PEZ marker
- `direction` peaks in the middle layers
- `direction` weakens toward output
- scalar magnitudes are already decodable from early layers

Best metrics:

- speed:
  - `L0 = 0.895`
  - `L8 = 0.983`
  - peak `0.988 @ L19`
- direction:
  - `L0 = 0.326`
  - `L8 = 0.816`
  - peak `0.876 @ L16`
  - late decline `0.876 -> 0.835`
- acceleration:
  - `L0 = 0.866`
  - `L8 = 0.974`
  - peak `0.986 @ L20`

#### Hidden detail discovered by iteration

These choices were critical and not explicit in the paper:

- `temporal_last` pooling mattered a lot
- `angle` outperformed `sincos`
- `direction_spatial_sector` mattered more than naive `position` grouping
- `resid_post` beat naive `resid_pre` for the final paper-like polar curve

This is the main lesson for PhysProbe: paper-level PEZ claims can hinge on under-specified evaluation choices, not only on model or dataset.


### 4. Figure 2(b) Cartesian: what failed

#### Targets

- `velocity_xy = (vx_px, vy_px)`
- `acceleration_xy = (ax_px, ay_px)`

#### What happened

Figure 2(b) never converged to one coherent global recipe.

Best partial configs were probe-specific:

- `velocity_xy`
  - capture: `resid_post`
  - pooling: `temporal_last_patch`
  - grouping: `magnitude_spatial_sector`
  - target: `vxy`
  - norm: `center`
  - solver: `trainable`
  - best run:
    - `fig2b_iter23_velocity_residpost_tlastpatch_magsector_center`
- `acceleration_xy`
  - capture: `resid_post`
  - pooling: `temporal_last`
  - grouping: `magnitude`
  - target: `vxy`
  - norm: `center`
  - solver: `trainable`
  - best run:
    - `fig2b_iter16_accel_residpost_tlast_magnitude_center`

Best metrics:

- `velocity_xy`
  - `L0 = 0.527`
  - `L8 = 0.908`
  - peak `0.926 @ L12`
- `acceleration_xy`
  - `L0 = 0.454`
  - `L8 = 0.915`
  - peak `0.944 @ L21`

#### Why Figure 2(b) failed

From `fig2b_failure_analysis.md`:

1. Cartesian targets were too easy in the current synthetic setup.
   - discrete low-cardinality `(vx, vy)` / `(ax, ay)`
   - episode-constant labels
   - single ball, fixed camera, fixed background
2. Spatial holdout was too weak for Cartesian vectors.
3. Different Cartesian probes preferred different recipes.
4. Mean pooling and strong readouts made shallow motion cues too accessible.

#### Key lesson for PhysProbe

Some targets are intrinsically too linearly aligned with early visual features.
For PhysProbe, this means:

- dynamic kinematic variables may saturate immediately
- they should not be mistaken for “emergent physics”
- they should be used as always-decodable controls, not as primary PEZ targets


### 5. Figure 1 IntPhys: what initially failed and how it was fixed

#### Initial mistake

The first IntPhys reproduction used:

- full public `dev`
- `resid_pre`
- `16` frames
- clip-level binary accuracy

This produced only:

- `L8 clip accuracy = 73.9%`
- peak `77.2% @ L18`

That looked like a reproduction failure.

#### Root cause

From `intphys_deep_rootcause.md`, the main issue was metric mismatch:

- public IntPhys benchmark is grouped by scene
- scene-relative accuracy matters more than naive clip accuracy
- the paper text says “binary classification” but does not clearly specify whether Figure 1 is evaluated clip-wise or scene-relatively

#### Corrected setup

Best Figure 1 config:

- dataset: full public IntPhys `dev`
- label: possible vs impossible
- metric: `scene-relative accuracy`
- capture: `resid_pre`
- transform: `resize`
- frames: `16`
- probe: trainable linear probe

Best metrics:

- scene-relative accuracy:
  - `L0 = 0.733`
  - `L8 = 1.000`
  - peak `1.000`
- clip-level accuracy remained much lower:
  - `L8 = 0.736`
  - peak around `0.769`

#### Key lesson for PhysProbe

Evaluation semantics can dominate apparent success/failure.

For PhysProbe, this means:

- before declaring PEZ absent or present, metric definition must be aligned with the scientific question
- simple per-window or per-episode `R^2` may be insufficient for some target families


### 6. Figure 6 and Figure 8: what was possible with public resources

#### Figure 6

Status:

- overall-only reproduction
- not full figure

Why:

- public IntPhys resources do not expose the subtask mapping for:
  - object permanence
  - shape constancy
  - spatiotemporal continuity

What was done:

- compared Large / Giant / Huge
- reproduced overall IntPhys row with linear probes

Interpretation:

- overall trend reproducible
- fine-grained subtask rows not reproducible from public resources

#### Figure 8

Status:

- overall-only attentive reproduction

What was added:

- `step_intphys_attentive.py`
- token-preserving attentive pipeline
- `temporal_last_patch`-style token features

Large/Huge/Giant overall results:

- Large:
  - `L8 relative = 82.2%`
  - peak `88.9% @ L10`
- Huge:
  - `L8 relative = 68.9%`
  - peak `86.7% @ L16`
- Giant:
  - `L8 relative = 68.9%`
  - peak `87.8% @ L21`

#### Key lesson for PhysProbe

Token-level attentive probing can reveal stronger emergence structure than naive mean-pooling, but it is much more expensive.

This matters directly for PhysProbe:

- start from existing mean-pooled features
- but be prepared to re-extract token-level features if scalar physics parameters do not show meaningful layer structure


### 7. What the PEZ reproduction teaches us overall

#### Paper-explicit choices

These were explicit in the paper or easy to reconstruct:

- 16-frame ball videos
- 256 resolution
- layer-wise probing
- linear probe
- 20-HP sweep
- 5-fold grouped CV

#### Hidden details that turned out to be critical

These were not sufficiently specified in the paper and strongly changed outcomes:

- residual capture point: `resid_pre` vs `resid_post`
- temporal pooling: `mean` vs `temporal_last`
- grouping key:
  - `position`
  - `condition`
  - `spatial_sector`
  - combinations
- target parameterization:
  - `angle`
  - `sincos`
  - `vxy`
- patch-level vs pooled probe
- IntPhys metric:
  - clip accuracy vs scene-relative accuracy

#### Core meta-lesson

PEZ is not “one script + one default recipe”.
It is a family of sensitive evaluation phenomena whose visibility depends on:

- target design
- split semantics
- pooling choice
- metric choice

This is the correct mindset to bring into PhysProbe.


## Part 2: Applying PEZ methodology to PhysProbe (manipulation data)

This section maps the PEZ lessons onto `/home/solee/physrepa_tasks`.


### 1. Current PhysProbe assets and constraints

From `CLAUDE.md` and `physrepa_tasks`:

- tasks:
  - `push`
  - `strike`
  - `peg_insert`
  - `nut_thread`
  - `drawer`
  - `reach`
- total episodes: `12,100`
- total frames: about `2.4M`
- dataset root:
  - `/mnt/md1/solee/data/isaac_physrepa_v2/step0`
- pre-extracted features:
  - `/mnt/md1/solee/features/physprobe_vitg/{task}/`
  - `/mnt/md1/solee/features/physprobe_vitl/{task}/`
- existing feature form:
  - mean-pooled safetensors
  - ViT-G: `40` layers, `1408` dim
  - ViT-L: `24` layers, `1024` dim
  - clips are `16` frames with stride `4`

Known dataset caveats:

- `Reach` has no physics randomization and must be treated as a negative control.
- `Drawer` varies only `drawer_joint_damping`.
- Factory tasks (`PegInsert`, `NutThread`) tend to have poor or zero direct contact-force supervision.
- `phase` is frequently useless / constant.


### 2. Target variable mapping: PEZ concepts -> PhysProbe concepts

The PEZ paper had three broad target classes:

1. always-decodable motion magnitude variables
   - speed
   - acceleration magnitude
2. harder structural variable with emergence
   - direction
3. plausibility variable
   - IntPhys possible/impossible

For PhysProbe, the closest mapping is:

#### A. Always-decodable controls

These correspond to PEZ speed/acceleration controls.
They should likely be linearly decodable from shallow layers.

Use:

- `ee_velocity`
- `ee_acceleration`
- `object_velocity`
- `object_acceleration`
- `handle_velocity`
- `peg_velocity`
- `nut_velocity`
- `drawer_joint_vel`
- `drawer_joint_pos`
- `drawer_opening_extent`
- `ee_to_target_distance`
- `ee_to_object_distance`
- `object_to_target_distance`
- `ball_planar_travel_distance`
- `insertion_depth`
- `axial_progress`

Interpretation:

- if these are strong at L0/L1, that is not PEZ evidence
- these serve as sanity checks that the feature cache and probe are functioning

#### B. Candidate “direction analogs” for manipulation

PEZ direction was the target that became globally available only after intermediate processing.
For PhysProbe, the best analog is likely not a kinematic state but a latent physical parameter that:

- affects dynamics
- is not trivially readable from one frame
- requires integrating behavior over time

Candidate emergent targets:

- Push:
  - `object_0_mass`
  - `object_0_static_friction`
  - `object_0_dynamic_friction`
  - `surface_static_friction`
  - `surface_dynamic_friction`
- Strike:
  - `object_0_mass`
  - `object_0_static_friction`
  - `object_0_dynamic_friction`
  - `object_0_restitution`
  - `surface_* friction`
- PegInsert:
  - `peg_mass`
  - `peg_static_friction`
  - `peg_dynamic_friction`
  - `hole_static_friction`
  - `hole_dynamic_friction`
- NutThread:
  - `nut_mass`
  - `nut_static_friction`
  - `nut_dynamic_friction`
  - `bolt_static_friction`
  - `bolt_dynamic_friction`
- Drawer:
  - `drawer_joint_damping`
  - optionally `drawer_handle_mass` only if variance is real in the dataset

Working hypothesis:

- these static physics parameters are the best PhysProbe analog of PEZ direction:
  - low shallow-layer decodability
  - possible middle-layer emergence if model internalizes interaction physics

#### C. Possible/impossible analog

There is no direct possible/impossible benchmark in PhysProbe.
The nearest analog would be:

- success/failure discrimination
- or consistency/violation labels from synthetic perturbations

But Step 0 data does not currently expose a public benchmark of that type.
So for the first PhysProbe PEZ pass, the main focus should be static physics parameter regression.


### 3. Task-by-task target plan

#### Push

Primary PEZ candidates:

- `object_0_mass`
- `object_0_static_friction`
- `object_0_dynamic_friction`
- `surface_static_friction`
- `surface_dynamic_friction`

Control variables:

- `ee_velocity`
- `ee_acceleration`
- `object_velocity`
- `object_acceleration`
- `ee_to_object_distance`
- `object_to_target_distance`

Interpretation:

- Push is the cleanest rigid-body task and should be the first target for probing static physics emergence.

#### Strike

Primary PEZ candidates:

- `object_0_mass`
- `object_0_static_friction`
- `object_0_dynamic_friction`
- `object_0_restitution`
- `surface_* friction`

Control variables:

- `ee_velocity`
- `object_velocity`
- `object_acceleration`
- `ball_planar_travel_distance`

Interpretation:

- restitution and friction in contact-rich collision are especially promising PEZ-like targets.

#### PegInsert

Primary PEZ candidates:

- `peg_mass`
- `peg_static_friction`
- `peg_dynamic_friction`
- `hole_static_friction`
- `hole_dynamic_friction`

Control variables:

- `insertion_depth`
- `peg_hole_lateral_error`
- `peg_velocity`

Interpretation:

- insertion kinematics are always-decodable controls
- static friction/mass may require richer contact reasoning

#### NutThread

Primary PEZ candidates:

- `nut_mass`
- `nut_static_friction`
- `nut_dynamic_friction`
- `bolt_static_friction`
- `bolt_dynamic_friction`

Control variables:

- `axial_progress`
- `nut_bolt_relative_angle`
- `nut_velocity`

Interpretation:

- relative-angle and axial-progress curves are useful dynamic controls
- friction/mass should be the real emergence targets

#### Drawer

Primary PEZ candidate:

- `drawer_joint_damping`

Controls:

- `drawer_joint_pos`
- `drawer_joint_vel`
- `drawer_opening_extent`
- `ee_velocity`

Interpretation:

- Drawer is structurally limited.
- Damping is the only real static physics target that matters.

#### Reach

Primary use:

- negative control only

Use:

- fake target such as `episode_index mod 5`
- permuted static target

Interpretation:

- any apparent emergence here is likely leakage or artifact


### 4. Feature strategy for PhysProbe

#### Immediate strategy

Start with existing mean-pooled features.

Reasons:

- feature caches already exist for ViT-L and ViT-G
- costs are low
- this is enough to answer first-order questions:
  - which variables are always decodable?
  - which are never decodable?
  - which show layer-selective emergence?

Relevant existing code:

- `analysis/probe_sweep_v2.py`
- `analysis/probe_sweep_v3.py`

#### Expected limitation

PEZ reproduction showed that mean-pooling is not always enough.

Specifically:

- Figure 2(c) needed careful temporal pooling
- Figure 2(b) sometimes needed patch-level probing
- Figure 8 required token-preserving attentive probing

Therefore for PhysProbe:

- mean-pooled existing features are the starting point
- but not the final ceiling
- if static physics parameters remain flat or noisy, next escalation should be:
  - per-window features
  - per-token / patch-preserving extraction
  - attentive probe

This should be written into the plan from the start so “failure on mean-pool” is not misread as “no PEZ”.


### 5. Grouping strategy for PhysProbe

PEZ reproduction made grouping a first-class variable.
For PhysProbe, grouping must prevent leakage while staying scientifically meaningful.

#### Default grouping

- `GroupKFold` by `episode_id`

Reason:

- windows from the same episode must not leak across folds
- this is directly analogous to avoiding video-window leakage

#### Secondary grouping candidates

- physics bucket grouping
  - if static param values are discrete or bucketable
  - useful for testing out-of-distribution generalization
- scene seed grouping
  - if metadata exposes environment seed or domain-randomization seed
- episode family grouping
  - if multiple episodes are generated from the same underlying randomized setup

#### Practical recommendation

Phase 1:

- use `episode_id` grouping only

Phase 2:

- add target-conditioned holdouts for the most interesting static parameters:
  - leave-one-mass-bin-out
  - leave-one-friction-bin-out

This is the direct PhysProbe analog of PEZ's magnitude/direction/spatial holdouts.


### 6. Probe design for PhysProbe

#### Core probe

Use the same Appendix-B-like trainable linear probe logic already implemented in:

- `analysis/probe_sweep_v3.py`

Keep:

- 20 HP sweep
- GroupKFold 5-fold
- per-fold best hyperparameter selection
- `R^2` as the regression metric

#### Target form

For static physics parameters:

- scalar regression target

For dynamic controls:

- scalar or vector, depending on variable
- existing `window + xyz` mode from `probe_sweep_v3.py` is already useful

#### Model priority

Start with:

- ViT-L first

Then:

- ViT-G

Reason:

- PEZ paper shows model-size effects
- but ViT-L is cheaper and aligns with the most extensively debugged PEZ reproduction


### 7. Judgment criteria for PEZ-like behavior in PhysProbe

Every `(task, variable)` should be assigned to one of three outcome classes:

#### A. PEZ-like emergence

Criteria:

- low or near-zero `L0`
- clear middle-layer rise
- identifiable peak not at final layer
- possible late decline

Interpretation:

- candidate “direction analog”
- latent physical parameter becomes readable only after intermediate processing

#### B. Always-linear / always-decodable

Criteria:

- moderate/high `L0`
- high all the way through
- no meaningful emergence transition

Interpretation:

- likely direct kinematic or appearance-linked signal
- analogous to PEZ speed / acceleration magnitude controls

#### C. Never-decodable

Criteria:

- `R^2` near zero at all layers
- no robust middle-layer structure

Interpretation:

- target may be absent from features
- dataset variance may be too weak
- label quality may be poor

#### Summary metrics to compute per curve

For each `(task, variable, model)`:

- `L0 R^2`
- `peak R^2`
- `peak layer`
- late-layer score
- `late decline = peak - last`
- binary flags:
  - `emerges`
  - `always_decodable`
  - `never_decodable`

Suggested initial thresholds:

- `always_decodable`:
  - `L0 >= 0.4` and `peak >= 0.6`
- `emerges`:
  - `L0 < 0.2`
  - `peak >= 0.3`
  - `peak layer` in middle 40% of layers
- `never_decodable`:
  - `peak < 0.1`

Thresholds should be treated as provisional and tuned after Push.


### 8. Known gaps to encode before starting

- Drawer:
  - only damping truly varies
- Factory tasks:
  - contact-force channels may be zero or uninformative
- Reach:
  - no true physics randomization
  - must be negative control only
- phase:
  - often degenerate / not useful
- existing feature caches are mean-pooled:
  - may hide token-level emergence
- current PhysProbe results already show some zero-valued targets:
  - should audit target variance before spending compute


## Part 3: Concrete execution plan

The plan below assumes the project starts from existing mean-pooled feature caches and uses PEZ-faithful probing logic first.


### [STEP 1] Write `physrepa_tasks/probe_physprobe.py`

Goal:

- create one task-focused driver script rather than continue overloading `analysis/probe_sweep_v3.py`

Implementation source:

- borrow training/CV logic from:
  - `/home/solee/pez/step3_probe.py`
  - `/home/solee/physrepa_tasks/analysis/probe_sweep_v3.py`

Minimum interface:

```bash
python probe_physprobe.py \
  --task push \
  --model_size large \
  --feature_root /mnt/md1/solee/features/physprobe_vitl/push \
  --target object_0_mass \
  --probing_level episode \
  --solver trainable \
  --output_dir results_physprobe/
```

Output:

- `results_{task}_{target}_{model}.csv`
  - columns:
    - `layer`
    - `r2_mean`
    - `r2_std`
    - `best_lr`
    - `best_wd`
    - `n_samples`
- optional plot:
  - `{task}_{target}_{model}_curve.png`

Required features:

- scalar regression
- 5-fold `GroupKFold`
- grouping by `episode_id`
- Appendix-B trainable sweep
- optional ridge sanity mode


### [STEP 2] Run Push first

Why Push first:

- simplest rigid-body task
- cleanest static targets
- no articulated geometry complication

Primary static targets:

- `object_0_mass`
- `object_0_static_friction`
- `object_0_dynamic_friction`
- `surface_static_friction`
- `surface_dynamic_friction`

Control targets:

- `ee_velocity`
- `ee_acceleration`
- `object_velocity`
- `object_acceleration`

Setup:

- model: `ViT-L`
- existing cache:
  - `/mnt/md1/solee/features/physprobe_vitl/push`
- probing level:
  - start with `episode`
- grouping:
  - `episode_id`
- solver:
  - trainable 20-HP

Expected:

- dynamic controls: always-decodable
- static physics params: candidate PEZ-like or never-decodable


### [STEP 3] Extend to all tasks

Tasks:

- `strike`
- `peg_insert`
- `nut_thread`
- `drawer`
- `reach`

Priority order:

1. `strike`
2. `peg_insert`
3. `nut_thread`
4. `drawer`
5. `reach`

Reason:

- Strike is the strongest contact-rich rigid-body analog.
- Factory tasks are the most interesting but also noisier.
- Drawer is constrained.
- Reach is the negative control.

Outputs:

- one CSV per `(task, target, model)`
- summary plots per task
- consolidated table across all tasks


### [STEP 4] Detect PEZ patterns automatically

Write a summarizer that reads all per-target CSVs and computes:

- `L0`
- `peak`
- `peak layer`
- `last`
- `late decline`
- class:
  - `pez_like`
  - `always_linear`
  - `never_linear`

Desired summary artifact:

- `PEZ_IN_PHYSPROBE_SUMMARY.csv`

Desired figure:

- overlay by target type:
  - mass-like targets
  - friction-like targets
  - damping-like targets
  - dynamic controls

This is the PhysProbe analog of asking:

- which variables behave like PEZ direction?
- which behave like PEZ speed?


### [STEP 5] Negative control

Use `Reach`.

Since Reach has no physics randomization:

- real static physics targets should not exist
- any claimed emergence on static physics is suspicious

Possible control labels:

- `episode_index mod 5`
- randomly permuted pseudo-target

Expected:

- `R^2 ~ 0` at all layers

If not:

- leakage or grouping bug is likely


### [STEP 6] Cross-model comparison

After ViT-L works on Push/Strike:

- repeat on ViT-G

Question:

- does larger model size shift emergence earlier/later?
- does larger model size sharpen static-physics peaks?

This is directly motivated by PEZ Figure 6 / Figure 8:

- model size matters
- but effect can differ by probe family


### [STEP 7] Escalation path if mean-pooled features fail

If static physics parameters are flat, noisy, or contradictory:

1. switch to `window` mode in `probe_sweep_v3.py`
   - episode mean -> window samples
2. keep vector-valued dynamic targets in `xyz`
3. add token-level extraction for one task
   - Push first
4. try attentive probe for the strongest candidate target family

This is the direct lesson from:

- PEZ Figure 8
- Figure 2(c) temporal pooling sensitivity
- Figure 2(b) patch-level sensitivity


### [STEP 8] Final report

Write:

- `PEZ_IN_PHYSPROBE_RESULTS.md`

Include:

- task-by-task verdict
- per-target verdict
- strongest PEZ-like variables
- strongest always-decodable controls
- negative-control sanity check
- model-size effect


## Part 4: Open questions

### 1. Why expect PEZ in manipulation at all?

The original paper studies:

- synthetic ball physics
- free-body motion
- IntPhys plausibility

Manipulation adds:

- contact-rich interaction
- articulated objects
- robot embodiment
- partial observability
- policy-conditioned trajectories

Therefore:

- PEZ may exist
- but it may move in depth
- or fragment by task
- or apply only to some static parameters, not all


### 2. Could PEZ location differ by task?

Yes.

Likely:

- Push/Strike:
  - earlier emergence for rigid-body mass/restitution
- Peg/Nut:
  - later emergence because contact geometry is harder
- Drawer:
  - maybe no clear PEZ because only one scalar physics target varies


### 3. Which model first: ViT-L or ViT-G?

Recommendation:

- start with `ViT-L`

Reason:

- cheaper
- already best understood from PEZ reproduction
- easier to debug before scaling

Then:

- run `ViT-G` only after the Push pipeline is stable


### 4. Do we need attentive probing?

Not at first.

Use mean-pooled features for:

- broad screening
- quick identification of candidate targets

Add attentive probing only if:

- static physics targets are ambiguous
- dynamic controls dominate
- layer curves look flat despite strong intuition that physics should be encoded


### 5. What is the most likely manipulation analog of PEZ direction?

Current hypothesis:

- static interaction parameters:
  - mass
  - friction
  - restitution
  - damping

not:

- raw kinematics
- position
- velocity
- obvious geometry progress variables

This should be treated as the central hypothesis to test.


### 6. What would count as a strong PhysProbe result?

A convincing positive result would look like:

- dynamic controls:
  - high from shallow layers
- static physics parameters:
  - low L0
  - middle-layer peak
  - possible late decline
- Reach negative control:
  - flat near zero

That would be the strongest manipulation-domain analog of PEZ.
