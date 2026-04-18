# EXPERIMENT_DESIGN

Pre-registered experiment design for applying PEZ probing methodology to PhysProbe manipulation data.

This document is intentionally written **before** implementing the new probe driver. The goal is to freeze hypotheses, metrics, decision rules, and escalation criteria in advance so that the interpretation of results is not driven by post-hoc confirmation bias.

Companion documents:

- [PEZ_TO_PHYSPROBE_PLAN.md](./PEZ_TO_PHYSPROBE_PLAN.md)
- archived historical code: `./archive_data_collection/`

Primary source of PEZ lessons:

- `/home/solee/pez/README.md`
- `/home/solee/pez/artifacts/results/final_reproduction_report.md`
- `/home/solee/pez/artifacts/results/final_verdict.md`
- `/home/solee/pez/artifacts/results/final_verdict_fig2b.md`
- `/home/solee/pez/artifacts/results/intphys_deep_rootcause.md`
- `/home/solee/pez/artifacts/results/fig2b_failure_analysis.md`
- `/home/solee/pez/artifacts/results/figure6_verdict.md`
- `/home/solee/pez/artifacts/results/figure8_verdict.md`


## §0. Pre-registered Hypotheses

The hypotheses below are the main scientific claims to test. Each one has:

- a binary prediction
- a success criterion
- a falsification criterion

The intent is to decide **before seeing PhysProbe probe curves** what counts as support or failure.


### H1 — Robot state is linearly decodable from shallow layers

**PEZ analog**

- This is the PhysProbe analog of PEZ `speed` / `acceleration magnitude`:
  always-decodable controls.

**Targets**

- `ee_velocity`
- `ee_acceleration`
- `object_velocity`
- `object_acceleration`
- `peg_velocity`
- `nut_velocity`
- `drawer_joint_pos`
- `drawer_joint_vel`
- `drawer_opening_extent`
- `ee_to_target_distance`
- `ee_to_object_distance`
- `object_to_target_distance`

**Binary prediction**

- These curves should be high from the earliest layers and should not require a PEZ-like onset.

**Success criteria**

- `R^2(L0) >= 0.8` for the strongest kinematic controls
- or at minimum `R^2(L0) >= 0.5` with monotonic/high curve and no delayed jump
- peak does **not** need to occur near one-third depth

**Falsification criteria**

- most control variables show:
  - `R^2(L0) < 0.2`
  - and first `R^2 >= 0.5` only in middle layers

That would imply the feature cache or target loading is broken, or the probing unit is misaligned.


### H2 — Static physics parameters show PEZ-like emergence at intermediate depth

**PEZ analog**

- This is the manipulation analog of PEZ `direction`.

**Targets**

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
  - `surface_static_friction`
  - `surface_dynamic_friction`
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

**Binary prediction**

- At least some static physics variables should:
  - start near zero at shallow layers
  - rise in the middle third of depth
  - peak in the middle layers
  - optionally weaken toward the output

**Success criteria**

- `R^2(L0) < 0.2`
- `peak R^2 >= 0.3`
- `peak layer fraction in [0.2, 0.6]`
- `late decline >= 0.02` preferred but not required

**Falsification criteria**

- all static physics variables are either:
  - `R^2(L0) >= 0.5` from the start, or
  - `peak R^2 < 0.1` everywhere

That would imply either:

- no manipulation-domain PEZ analog is present in mean-pooled features, or
- mean-pooled features are too weak a representation level.


### H3 — Reach negative control shows no PEZ-like emergence

**Reason**

- Reach has no true physics randomization in this dataset.

**Targets**

- fake targets:
  - `episode_index mod 5`
  - shuffled pseudo-regression target
- optional dynamic control:
  - `ee_velocity`

**Binary prediction**

- No fake target should show a PEZ-like curve.

**Success criteria**

- `peak R^2 < 0.1` for all fake targets

**Falsification criteria**

- fake target gets:
  - `peak R^2 >= 0.2`
  - or clean middle-layer rise

This would strongly suggest leakage, grouping failure, or target misalignment.


### H4 — Drawer damping is weaker and subtler than mass/friction in rigid-body tasks

**Reason**

- Drawer varies only `drawer_joint_damping`.
- The drawer task also contains strong trivial kinematic signals (`drawer_joint_pos`, `drawer_joint_vel`).

**Binary prediction**

- `drawer_joint_damping` should either:
  - show a low-amplitude PEZ-like curve
  - or remain weak / ambiguous

**Success criteria**

- `peak R^2` lower than the best Push/Strike static-physics peaks
- curve still classifiable as `PEZ-like` or `intermediate`

**Falsification criteria**

- `drawer_joint_damping` is among the strongest static-physics curves in the entire benchmark

That would be suspicious and likely indicate leakage.


### H5 — Factory tasks show weaker friction emergence than rigid-body tasks

**Reason**

- PegInsert / NutThread have contact-rich geometry, but known issues:
  - contact-force supervision is not reliable
  - some direct force channels are zero or weak

**Binary prediction**

- friction emergence in PegInsert/NutThread should be weaker or noisier than Push/Strike.

**Success criteria**

- lower `peak R^2` or later peak layer than Push/Strike

**Falsification criteria**

- Factory friction curves are uniformly stronger and cleaner than Push/Strike across both ViT-L and ViT-G


### H6 — Model size shifts the effective PEZ location

**Reason**

- In PEZ reproduction:
  - Large most often showed paper-like onset around `L8`
  - Giant/Huge often shifted the strongest point later

**Binary prediction**

- ViT-G should show equal or later peaks than ViT-L for the same target family.

**Success criteria**

- `peak_layer_G >= peak_layer_L` for the majority of PEZ-like static targets

**Falsification criteria**

- ViT-G consistently peaks earlier than ViT-L with no gain in robustness


### H7 — Mean-pooled features are enough for controls, but may be insufficient for subtle static physics

**Reason**

- PEZ Figure 2(c) and Figure 8 showed that pooling/tokenization details matter.

**Binary prediction**

- mean-pooled existing features should already recover:
  - strong control curves
  - some static-physics signal
- but some static physics may remain weak enough to justify Phase 2 token-level extraction.

**Success criteria**

- controls succeed on mean-pool
- at least one static parameter is `intermediate` or `PEZ-like`

**Falsification criteria**

- all static physics are zero/flat and all dynamic controls are also weak

That would point to deeper implementation or alignment problems.


### H8 — Episode-level grouping is the minimum valid CV scheme

**Reason**

- Existing feature files contain multiple windows per episode.
- Windows from the same episode must not cross folds.

**Binary prediction**

- GroupKFold by `episode_id` is the minimum acceptable protocol.

**Success criteria**

- all Phase 1 runs group by `episode_id`
- no window from a held-out episode appears in training

**Falsification criteria**

- any run mixes windows from the same episode across train/val


### H9 — A PEZ-like result should be rare, not ubiquitous

**Reason**

- If every target shows a PEZ-like curve, the criterion is too weak.

**Binary prediction**

- most dynamic control variables will be `always-linear`
- most fake/noisy variables will be `never-linear`
- only a subset of static physics parameters should be `PEZ-like`

**Success criteria**

- the final classification is sparse and interpretable

**Falsification criteria**

- nearly everything is PEZ-like or nearly nothing is classifiable


### H10 — Contact-rich targets may need token-level Phase 2 before any strong conclusion

**Reason**

- PEZ Figure 8 showed that attentive / token-preserving probes can uncover structure that mean-pooling suppresses.

**Binary prediction**

- if static-physics curves are weak-but-shaped under mean-pool, token-level extraction may sharpen them.

**Success criteria**

- Phase 1 identifies a concrete target family worth escalating

**Falsification criteria**

- Phase 1 is already decisive for all targets


## §1. Experimental Design


### 1.1 Factors to vary

#### Task

- `push`
- `strike`
- `peg_insert`
- `nut_thread`
- `drawer`
- `reach`

#### Target variable

Two broad groups:

1. static physics parameters
2. dynamic robot/object state controls

#### Model

- Phase 1: `ViT-L`
- Phase 3: `ViT-G`

#### Feature representation

- Phase 1:
  - existing mean-pooled safetensors
- Phase 2:
  - token-level re-extraction if needed


### 1.2 Fixed constants from PEZ lessons

These are fixed at the beginning unless a documented escalation criterion is triggered.

- solver:
  - `trainable 20-HP sweep`
- CV:
  - `5-fold GroupKFold`
- grouping:
  - by `episode_id`
- target metric:
  - `R^2`
- normalization:
  - `zscore`
- selection metric:
  - validation `R^2`
- probing level:
  - start at `episode` for static params
  - keep `window` in reserve for dynamic/vector follow-up

Rationale:

- `ridge` was useful for quick sanity in PEZ, but the final, stable protocol was trainable 20-HP
- clip accuracy was wrong for IntPhys; PhysProbe should use `R^2` for continuous parameters from the start
- GroupKFold semantics must be explicit up front


### 1.3 Explicit NOT-do list

These are paths that consumed time in PEZ or are already known to be poor defaults for this project.

- do **not** use binary clip accuracy for any discrimination-style PhysProbe task
- do **not** use naive KFold without episode grouping
- do **not** interpret dynamic controls as PEZ evidence
- do **not** conclude “no PEZ” after mean-pool only if static targets show weak structured curves
- do **not** mix multiple windows from one episode across folds
- do **not** start with ViT-G before ViT-L is stable
- do **not** treat `phase` as a meaningful target by default
- do **not** assume contact-force channels are valid in Factory tasks
- do **not** use target parameterizations more complex than the raw scalar value for scalar physics parameters
- do **not** infer success from a single visually nice curve without pre-registered criteria


## §2. Lessons from PEZ — Concrete Carryovers

The table below is the main pre-baked set of lessons to carry into PhysProbe.

| Lesson | What happened in PEZ | How to apply to PhysProbe |
|---|---|---|
| L1. Residual readout matters | `resid_pre` vs `resid_post` changed layer-0 accessibility and curve shape | if future PhysProbe re-extraction is needed, compare both readouts |
| L2. Temporal pooling matters | `mean` often hid PEZ; `temporal_last` unlocked Figure 2(c) | if mean-pooled PhysProbe features are inconclusive, Phase 2 should prioritize `temporal_last` |
| L3. Patch-preserving features can matter | Figure 2(b) velocity needed `temporal_last_patch` | subtle manipulation physics may need token-level or patch-level features |
| L4. Grouping key is not a detail | different grouping choices changed whether PEZ appeared | PhysProbe must lock `episode_id` grouping first and treat other groupings as explicit ablations |
| L5. Target parameterization changes conclusions | `angle` beat `sincos`; `vxy` behaved differently | use raw scalar physics parameters first; avoid unnecessary reparameterizations |
| L6. Metric choice can completely invert verdict | IntPhys failed under clip accuracy and matched under scene-relative accuracy | PhysProbe metric choice must be justified in advance; for regression use `R^2` |
| L7. Some targets are trivially shallow | Figure 2(b) Cartesian targets were too easy | dynamic kinematics in PhysProbe are controls, not primary PEZ targets |
| L8. One figure may not admit one global recipe | Figure 2(b) velocity and acceleration needed different best configs | do not force one universal interpretation across all PhysProbe targets |
| L9. Negative controls are mandatory | Reach-style or fake targets catch leakage | run Reach fake-target probes early, not at the end |
| L10. Model size changes layer position | Large/Giant/Huge had different onset and peak locations | compare ViT-L and ViT-G only after target families are identified |
| L11. Public paper descriptions can be under-specified | PEZ required 24+ iterations to uncover hidden choices | write down every PhysProbe choice explicitly from the start |
| L12. Mean-pooled success is only Phase 1 | attentive probe and token-preserving features changed IntPhys/Figure 8 conclusions | Phase 2 token-level escalation should be planned now, not improvised later |


## §3. Experimental Protocol


### 3.1 Dataset structure confirmed before design

This section records actual local dataset structure checks performed before writing this document.

#### Static physics metadata lives in `meta/episodes.jsonl`

Examples:

- Push episodes contain:
  - `object_0_mass`
  - `object_0_static_friction`
  - `object_0_dynamic_friction`
  - `surface_static_friction`
  - `surface_dynamic_friction`
- Strike adds:
  - `object_0_restitution`
- PegInsert contains:
  - `peg_mass`
  - `peg_static_friction`
  - `peg_dynamic_friction`
  - `hole_static_friction`
  - `hole_dynamic_friction`
- NutThread contains:
  - `nut_mass`
  - `nut_static_friction`
  - `nut_dynamic_friction`
  - `bolt_static_friction`
  - `bolt_dynamic_friction`
- Drawer contains:
  - `drawer_handle_mass`
  - `drawer_joint_damping`
  - `handle_static_friction`
  - `handle_dynamic_friction`
- Reach contains no static physics keys

Important:

- `physics_gt` is **not** stored in `episodes.jsonl`
- dynamic GT must be loaded from per-episode parquet files

#### Dynamic `physics_gt.*` lives in parquet

Verified examples:

- Push parquet columns include:
  - `physics_gt.ee_position`
  - `physics_gt.ee_velocity`
  - `physics_gt.ee_acceleration`
  - `physics_gt.object_position`
  - `physics_gt.object_velocity`
  - `physics_gt.object_acceleration`
  - `physics_gt.contact_flag`
  - `physics_gt.contact_force`
  - `physics_gt.object_to_target_distance`
  - `physics_gt.phase`
- Strike adds:
  - `physics_gt.ball_planar_travel_distance`
- PegInsert adds:
  - `physics_gt.peg_velocity`
  - `physics_gt.insertion_depth`
  - `physics_gt.peg_hole_lateral_error`
- NutThread adds:
  - `physics_gt.axial_progress`
  - `physics_gt.nut_bolt_relative_angle`
- Drawer adds:
  - `physics_gt.drawer_joint_pos`
  - `physics_gt.drawer_joint_vel`
  - `physics_gt.handle_velocity`
  - `physics_gt.drawer_opening_extent`
- Reach includes:
  - `physics_gt.ee_to_target_distance`

#### Existing feature caches are real and complete

Verified:

- `physprobe_vitl`:
  - push `1500`
  - strike `3000`
  - peg_insert `2500`
  - nut_thread `2500`
  - drawer `2000`
  - reach `600`
- same counts for `physprobe_vitg`

Feature file structure:

- one `.safetensors` file per episode
- window-level vectors already exist inside each file
- example:
  - `window_starts` shape `(58,)`
  - ViT-L keys like `layer_0_window_0` with shape `(1024,)`
  - ViT-G keys like `layer_0_window_0` with shape `(1408,)`

Conclusion:

- Phase 1 can run without any new feature extraction
- the existing cache is episode-organized and window-aware


### 3.2 Phase 1: ViT-L mean-pooled/window-derived probing

This is the minimal first pass using current assets.

#### [P1-1] Push task first

Targets:

- static:
  - `object_0_mass`
  - `object_0_static_friction`
  - `object_0_dynamic_friction`
  - `surface_static_friction`
  - `surface_dynamic_friction`
- controls:
  - `ee_velocity`
  - `ee_acceleration`
  - `object_velocity`
  - `object_acceleration`
  - `ee_to_object_distance`
  - `object_to_target_distance`

Config:

- model: `ViT-L`
- features:
  - `/mnt/md1/solee/features/physprobe_vitl/push/`
- grouping:
  - `episode_id`
- solver:
  - trainable 20-HP
- metric:
  - `R^2`

Expected purpose:

- validate the pipeline on the easiest rigid-body task
- establish the first separation between:
  - always-linear controls
  - candidate PEZ-like static params


#### [P1-2] Strike task

Targets:

- static:
  - `object_0_mass`
  - `object_0_static_friction`
  - `object_0_dynamic_friction`
  - `object_0_restitution`
  - `surface_static_friction`
  - `surface_dynamic_friction`
- controls:
  - `ee_velocity`
  - `object_velocity`
  - `object_acceleration`
  - `ball_planar_travel_distance`

Expected purpose:

- test collision-rich rigid-body physics
- especially `restitution`, which is a strong candidate for a manipulation PEZ analog


#### [P1-3] PegInsert / NutThread

Targets:

- PegInsert static:
  - `peg_mass`
  - `peg_static_friction`
  - `peg_dynamic_friction`
  - `hole_static_friction`
  - `hole_dynamic_friction`
- PegInsert controls:
  - `peg_velocity`
  - `insertion_depth`
  - `peg_hole_lateral_error`

- NutThread static:
  - `nut_mass`
  - `nut_static_friction`
  - `nut_dynamic_friction`
  - `bolt_static_friction`
  - `bolt_dynamic_friction`
- NutThread controls:
  - `nut_velocity`
  - `axial_progress`
  - `nut_bolt_relative_angle`

Important warning:

- do not over-interpret contact-force channels here


#### [P1-4] Drawer

Targets:

- static:
  - `drawer_joint_damping`
- controls:
  - `drawer_joint_pos`
  - `drawer_joint_vel`
  - `drawer_opening_extent`
  - `handle_velocity`
  - `ee_velocity`

Expected purpose:

- test whether a subtle scalar dynamic parameter can emerge at all under mean-pooled features


#### [P1-5] Reach negative control

Targets:

- real dynamic control:
  - `ee_velocity`
  - `ee_acceleration`
  - `ee_to_target_distance`
- fake targets:
  - `episode_index mod 5`
  - shuffled pseudo-target

Expected purpose:

- verify that PEZ-like findings are not artifacts of leakage or overfit


### 3.3 Phase 2: token-level escalation if Phase 1 is inconclusive

Token-level work should **not** be done immediately.
It is triggered only if Phase 1 gives suggestive but inconclusive results.

#### Trigger for Phase 2

Escalate only if at least one of the following is true:

1. a static physics parameter shows:
   - `0.1 <= peak R^2 < 0.3`
   - and a clear mid-layer shape
2. dynamic controls succeed strongly, but all static params remain weak
3. ViT-L and ViT-G disagree in a way that suggests pooling is hiding structure

#### [P2-1] Re-extraction decision

If triggered, write a new extractor in the style of PEZ `step2_extract.py`:

- retain token-level or patch-level structure
- allow:
  - `temporal_last`
  - `temporal_last_patch`
  - optionally `temporal_diff`

#### [P2-2] New probe modes

Add:

- token-level attentive probe
- patch-level mean vs temporal-last comparisons

#### [P2-3] Scope control

Do **not** re-extract all tasks first.

Start with:

- `push`
- then `strike`


### 3.4 Phase 3: cross-model comparison

Only after ViT-L has produced stable target families:

- rerun best task/target subsets on `ViT-G`

Suggested order:

1. Push
2. Strike
3. whichever static parameter showed the clearest PEZ-like shape

Goal:

- build a Figure-6-style analog for manipulation
- ask whether larger models shift or sharpen PEZ-like emergence


## §4. Metrics + Pass/Fail Decision Tree


### 4.1 Primary summary per curve

For each `(task, target, model)` compute:

- `L0`
- `L4`
- `L8`
- `peak_r2`
- `peak_layer`
- `last_r2`
- `late_decline = peak_r2 - last_r2`
- `peak_fraction = peak_layer / (num_layers - 1)`


### 4.2 Pre-registered classifier

```python
def classify_curve(r2_per_layer):
    L0 = float(r2_per_layer[0])
    peak_r2 = float(max(r2_per_layer))
    peak_layer = int(np.argmax(r2_per_layer))
    peak_fraction = peak_layer / max(1, len(r2_per_layer) - 1)
    late_decline = peak_r2 - float(r2_per_layer[-1])

    if L0 >= 0.8:
        return "always-linear"
    if peak_r2 < 0.3:
        return "never-linear"
    if peak_r2 >= 0.5 and 0.2 <= peak_fraction <= 0.6:
        return "PEZ-like"
    return "intermediate"
```

Interpretation:

- `always-linear`
  - PEZ speed/acceleration analog
- `PEZ-like`
  - candidate manipulation PEZ direction analog
- `never-linear`
  - absent / too subtle / label weak
- `intermediate`
  - worth follow-up, but not pre-registered success


### 4.3 Decision table

| Observation | Interpretation | Next action |
|---|---|---|
| control variable has `L0 >= 0.8` | expected | no escalation |
| static physics target is `PEZ-like` | main positive result | include in summary panel |
| static physics target is `intermediate` | ambiguous | candidate for Phase 2 |
| static physics target is `never-linear` but controls are strong | likely no shallow-decoding of that target | do not escalate unless scientifically important |
| fake Reach target is non-zero | leakage alarm | stop and audit grouping/target alignment |
| ViT-G peaks later than ViT-L | supports H6 | include cross-model analysis |
| all static params are always-linear | likely target too easy or dataset confounded | inspect parameter-target correlations |


## §5. Artifacts + Outputs

The new pipeline should write into `./artifacts/`.

Recommended structure:

- `artifacts/results/probe_{task}_{target}_{model}.csv`
- `artifacts/results/summary_{task}_{model}.json`
- `artifacts/figures/curves_{task}_{model}.png`
- `artifacts/figures/pez_analog_panel_{model}.png`
- `artifacts/tables/target_catalog.csv`

Final interpretation document:

- `EXPERIMENT_RESULTS.md`

This should contain:

- hypothesis-by-hypothesis outcome
- task-by-task summary
- PEZ-like targets list
- always-linear controls list
- negative-control check
- escalation decisions


## §6. Timeline Estimate

The estimate below is based on the actual PEZ reproduction cost plus the fact that PhysProbe already has cached features.

### Phase 1

- 6 tasks
- ViT-L only
- existing window-level mean-pooled caches
- 5-fold GroupKFold
- trainable 20-HP

Estimated cost:

- `~1-2 GPU-days`

Reason:

- no re-extraction needed
- main cost is repeated probing across many `(task, target)` pairs


### Phase 2

- token-level or patch-level re-extraction on 1-2 tasks

Estimated cost:

- `~1 GPU-day`

Reason:

- extraction cost returns
- but only for selected tasks if escalation is triggered


### Phase 3

- ViT-G rerun on selected best tasks

Estimated cost:

- `~1-2 GPU-days`

Reason:

- model depth is larger
- more layers and larger hidden width


### Total planned envelope

- minimal path:
  - `~1 GPU-day`
- likely realistic path:
  - `~2-4 GPU-days`
- upper bound with token-level escalation:
  - `~4-5 GPU-days`


## §7. Risks + Contingencies


### Risk 1 — All static physics params are never-linear

**Early signal**

- controls succeed strongly
- all static params `peak R^2 < 0.1`

**Interpretation**

- PEZ may not transfer to manipulation static parameters under mean-pooled features
- or the labels are too weak/noisy

**Mitigation**

- escalate only the most scientifically plausible targets to Phase 2 token-level probing


### Risk 2 — Reach negative control fails

**Early signal**

- fake target gets structured non-zero curve

**Interpretation**

- leakage, grouping error, or episode misalignment

**Mitigation**

- stop
- audit window-to-episode mapping
- audit fold assignment
- audit target loading


### Risk 3 — Mean-pool hides the only interesting signal

**Early signal**

- static targets are not zero, but only weakly structured
- dynamic controls are very strong

**Interpretation**

- same pattern as PEZ before token-level refinements

**Mitigation**

- Phase 2:
  - `temporal_last`
  - `temporal_last_patch`
  - attentive probe


### Risk 4 — Static targets are confounded by trivial correlates

**Early signal**

- static target looks always-linear from `L0`

**Interpretation**

- target may be visually correlated with trajectory amplitude, object travel distance, or contact timing

**Mitigation**

- inspect per-target correlation against dynamic controls
- compare held-out buckets if available


### Risk 5 — Too many targets, not enough interpretability

**Early signal**

- large number of CSVs with no consolidated judgment

**Interpretation**

- results become anecdotal instead of scientific

**Mitigation**

- enforce the pre-registered classifier
- summarize by target family


## Appendix A. Actual local resource checks used for this design

### A.1 Meta structure

Confirmed:

- static physics parameter keys are in:
  - `/home/solee/data/data/isaac_physrepa_v2/step0/{task}/meta/episodes.jsonl`
- dynamic `physics_gt.*` keys are in parquet under:
  - `/home/solee/data/data/isaac_physrepa_v2/step0/{task}/data/chunk-*/episode_*.parquet`

### A.2 Feature cache existence

Confirmed:

- `/mnt/md1/solee/features/physprobe_vitl/{task}/`
- `/mnt/md1/solee/features/physprobe_vitg/{task}/`

exist for all 6 tasks and match expected episode counts.

### A.3 Feature file format

Confirmed on `push/000000.safetensors`:

- ViT-L:
  - per-window vectors of shape `(1024,)`
  - `window_starts` shape `(58,)`
- ViT-G:
  - per-window vectors of shape `(1408,)`
  - `window_starts` shape `(58,)`

This means the current cache already supports:

- episode-level aggregation
- window-level probing

without new extraction.

