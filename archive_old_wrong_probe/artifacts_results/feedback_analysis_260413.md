# Analysis of `260413_feedback.md` Against Current PhysProbe Status

Date: 2026-04-19

Scope of reference experiments used for this analysis:

- `Phase 1`: Push mean-pooled probing
- `Phase 2`: Push token-patch probing
- `Phase 2c`: Reach / Strike token-patch probing
- `Phase 2d`: 3D direction reruns on Push / Reach / Strike

Relevant result files:

- `artifacts/results/EXPERIMENT_RESULTS_phase1_push.md`
- `artifacts/results/EXPERIMENT_RESULTS_phase2_push.md`
- `artifacts/results/EXPERIMENT_RESULTS_phase2c_overall.md`
- `artifacts/results/EXPERIMENT_RESULTS_phase2d_direction3d_overall.md`
- `artifacts/results/verdict_phase2_push.json`
- `artifacts/results/verdict_phase2c_reach_reach.json`
- `artifacts/results/verdict_phase2c_strike_strike.json`
- `artifacts/results/verdict_phase2d_direction3d_push.json`
- `artifacts/results/verdict_phase2d_direction3d_reach.json`
- `artifacts/results/verdict_phase2d_direction3d_strike.json`

---

## 1. Per-feedback analysis

### F1. DTW pitfalls and alternative metrics

**Feedback summary**

- Do not naively use unrestricted DTW for physics validation.
- Alternatives suggested:
  - contact-event alignment
  - phase-space analysis
  - windowed DTW
  - Fréchet / Chamfer distance

**Status**

- `미수행` for the dynamic time-warping family itself
- `부분 수행` only in the sense that current work already moved away from naive framewise scoring and into PEZ-style probing with grouped CV

**Current relevance**

- The current pipeline is not evaluating generated-vs-ground-truth trajectories frame-by-frame.
- It is evaluating **layerwise linear decodability** from frozen representations.
- So unrestricted DTW is **not** the right next move for the current core question.
- The part of F1 that still matters is the broader warning:
  - if the next step becomes **window-level** or **event-level** trajectory comparison, metric choice matters a lot.

**Priority**

- `Medium`

**Why**

- It is not the bottleneck for the current PEZ finding.
- The current positive results (`push`, `strike`, and 3D direction reruns) were obtained without DTW.
- But if the project expands toward event-aligned or temporal-trajectory diagnostics, F1 becomes important immediately.

**Concrete execution plan if run**

Recommended version:

1. Skip unrestricted DTW.
2. Use one of:
   - `windowed DTW` with a 1-2 window band
   - `phase-space` comparison for `(position, velocity)` trajectories
3. Apply first to:
   - `push / ee_direction_3d`
   - `strike / ee_direction_3d`
4. Compare:
   - raw time-aligned error
   - band-limited DTW
   - phase-space distance

**Bottom line**

- F1 is still valid conceptually, but it is **not the best immediate next experiment** for the current PEZ-transfer question.

---

### F2. Pre / During / Post contact PEZ analysis

**Feedback summary**

- Split clips into:
  - pre-contact
  - during-contact
  - post-contact
- Probe each regime separately.
- If R² is high only during contact, that would be strong evidence for contact-specific PEZ.

**Status**

- `현재 데이터로 불가능` for the exact proposed version

**Evidence**

- Public dataset audit during night shift found:
  - `contact_flag = 0` for all scanned Push episodes
  - `contact_force = 0`
  - `contact_point = 0`
  - same issue appeared in sampled `strike`, `peg_insert`, `nut_thread`, `drawer`
- This already blocked Phase 3 force/contact probing and was documented in:
  - `artifacts/results/PHASE3_BLOCKED.md`

**Current relevance**

- High conceptually, because the strongest positive task so far is `strike`, which is contact-heavy.
- If real contact annotations existed, F2 would be one of the best follow-up analyses.

**Priority**

- `Medium`

**Why not High**

- Exact implementation is blocked by the public labels.
- The project should not burn time pretending the public `contact_flag` is usable when it is all zero.

**Concrete execution plan if attempted anyway**

Only with a workaround. Two viable substitutes:

1. **Pseudo-contact segmentation from kinematics**
   - `push`: use object-speed onset, object-acceleration spikes, or `ee_to_object_distance` threshold
   - `strike`: use `object_acceleration` spike or `ball_planar_travel_distance` onset
   - `peg_insert`: use `insertion_depth` onset
   - `drawer`: use `drawer_joint_vel` onset
2. Build:
   - `pre-event`
   - `during-event`
   - `post-event`
   windows from those surrogate events
3. Re-run token-patch probe on each subset

**Bottom line**

- Direct F2 is blocked.
- A surrogate event-aligned version is possible, but only after carefully designing pseudo-contact labels.

---

### F3. CKA across tasks and layers

**Feedback summary**

- Compute cross-task representational similarity layer-by-layer.
- Hypothesis: PEZ layers may align more strongly across related tasks than non-PEZ layers.

**Status**

- `미수행`

**Current relevance**

- Very high.
- This is one of the cleanest next analyses because the required data already exists:
  - token-patch caches for `push` and `strike`
  - and a completed 3D direction verdict for `reach`
- The current results already imply a structured pattern:
  - `strike` is strongest positive
  - `push` is partial positive
  - `reach` is a weaker but still informative case
- CKA would test whether the “good” layers align across tasks.

**Priority**

- `High`

**Why**

- No new model extraction is strictly required for `push` and `strike`
- The analysis directly addresses the professor feedback
- It complements the current probing results instead of duplicating them

**Concrete execution plan**

Recommended configuration:

- features: use token-patch cache collapsed to the same per-episode flattened vectors used by the probes
- tasks:
  - `push`
  - `strike`
  - optionally `reach`
- model:
  - `large` first
- sample balancing:
  - randomly subsample each task to the same episode count
  - recommend `N = 600` to match Reach if including Reach
- compute:
  - linear CKA for each `(task_i, task_j, layer)`
- outputs:
  - task×task×layer heatmap
  - line plots for `push-strike`, `push-reach`, `strike-reach`

**Bottom line**

- F3 is one of the best immediate next experiments.

---

### F4. Physics parameter effects and split-by-value probing

**Feedback summary**

- A: split episodes by physical parameter bins (low/med/high friction, etc.) and compare probe behavior
- B: probe the physical parameters themselves as targets

**Status**

- `부분 수행`

**What is already done**

- F4-B has already been attempted in Phase 1 / Push:
  - `mass`
  - `obj_friction`
  - `surface_friction`
- Result:
  - all stayed `never-linear` under the tested recipe

**What is not done**

- F4-A split-by-value probing has not been run.

**Current relevance**

- Moderate.
- Current evidence already says static physical parameters are weak or absent under current settings.
- Split-by-value may still be informative, but it is a more secondary diagnostic than CKA or shuffle.

**Priority**

- `Medium`

**Why**

- It could reveal whether kinematic PEZ curves differ systematically by friction or mass regime.
- But it is unlikely to overturn the core conclusion that static params themselves are not strongly decodable.

**Concrete execution plan**

Two separate analyses:

1. **F4-A split-by-value**
   - task: `push` first
   - bins:
     - `mass` tertiles
     - `obj_friction` tertiles
     - `surface_friction` tertiles
   - target:
     - `ee_direction_3d`
     - `ee_speed`
     - `ee_accel_magnitude`
   - compare `R² vs layer` across low/med/high bins
2. **F4-B expanded direct probing**
   - rerun direct static-target probing on `strike` as well
   - maybe also `drawer / damping`

**Bottom line**

- F4 is still worth doing, but not before F3/F5.

---

### F5. Frame shuffle experiment

**Feedback summary**

- Shuffle frame order inside clips.
- If probing survives, the model may be relying on static appearance rather than temporal causality.

**Status**

- `미수행`

**Current relevance**

- Very high.
- This is one of the cleanest ways to stress-test whether the current positive PEZ-like curves are genuinely temporal.
- This matters especially because:
  - `push / ee_direction_3d` is positive
  - `strike / ee_direction_3d` and `object_direction_3d` are strongly positive
- Without shuffle, a reviewer could still argue that the probe is exploiting static visual correlates.

**Priority**

- `High`

**Why**

- It directly tests temporal causality.
- It uses the same analysis framework already in place.
- It speaks to the strongest remaining skepticism about the positive results.

**Concrete execution plan**

Recommended minimal version:

1. Re-extract token-patch cache with shuffled frames for:
   - `push`
   - `strike`
2. Keep everything else fixed:
   - `resid_post`
   - `temporal_last_patch`
   - same model (`large`)
   - same trainable 20-HP probe
3. Probe only:
   - `ee_direction_3d`
   - `object_direction_3d` for `strike`
   - `ee_speed`
4. Compare:
   - original vs shuffled `R² vs layer`

Expected useful outcomes:

- if curves collapse, the current PEZ result is genuinely temporal
- if curves persist, then static cue dependence is stronger than currently assumed

**Bottom line**

- F5 is arguably the most important next validation experiment after finishing the present token-patch direction story.

---

### F6. Probing pseudo code / core protocol

**Feedback summary**

- Basic probing loop:
  - task × layer × variable
  - GroupKFold by episode
  - ridge / probe
  - save mean score per layer

**Status**

- `이미 수행`

**Current relevance**

- The current project already implements the same structural idea, but with an improved recipe learned from PEZ reproduction:
  - `trainable 20-HP sweep`
  - grouped CV by episode
  - sanity checks against leakage
  - token-patch features
  - 3D direction target where appropriate

**Priority**

- `Low`

**Why**

- The pseudo code itself is not a next experiment.
- It has already been operationalized and extended.

**Concrete execution plan if wanting a literal replication**

- add a simple `ridge(alpha=1.0)` baseline mode to `probe_physprobe.py`
- run it on:
  - `push / ee_direction_3d`
  - `strike / ee_direction_3d`
  - `reach / ee_direction_3d`
- compare with current `trainable 20-HP`

**Bottom line**

- F6 is already functionally addressed.

---

## 2. Feasibility matrix

| Feedback | Status | Feasibility now | Main blocker / enabler |
|---|---|---|---|
| F1 | Partial concept only | Additional work needed | Current pipeline is not a trajectory-alignment setup |
| F2 | Blocked in direct form | Surrogate-only | Public `contact_flag/contact_force` all zero |
| F3 | Not run | Immediately feasible | Existing task caches and features are enough |
| F4 | Partial | Immediately feasible | Split-by-value analysis not yet run |
| F5 | Not run | Additional work needed | Requires shuffled token-patch re-extraction |
| F6 | Already done in spirit | Completed | Could add literal ridge baseline, but not necessary |

Quick classification:

- **Already completed in spirit**: `F6`
- **Immediately runnable with current data**: `F3`, `F4`
- **Runnable but needs new extraction / engineering**: `F1`, `F5`
- **Directly blocked by public labels**: `F2`

---

## 3. Priority-ranked next steps with time estimates

### 1. F5 Frame shuffle validation

- Priority: `High`
- Why:
  - strongest test of temporal causality
  - most directly strengthens the positive PEZ-transfer claim
- Estimated cost:
  - `push` shuffle token extraction: ~1.5-2h
  - `strike` shuffle token extraction: ~2-3h
  - probe runs: ~1-2h total

### 2. F3 CKA cross-task similarity

- Priority: `High`
- Why:
  - no new extraction required for `push` and `strike`
  - directly addresses representational similarity across tasks
- Estimated cost:
  - feature loading + CKA compute + plotting: ~1-3h depending on sample count

### 3. F4-A split-by-value probing

- Priority: `Medium`
- Why:
  - may explain why static params are not directly decodable
  - could reveal condition-dependent PEZ strength
- Estimated cost:
  - `push` first-pass split analysis: ~1-2h

### 4. F1 phase-space / bounded-DTW add-on

- Priority: `Medium`
- Why:
  - useful if the project moves toward temporal validation beyond decoding
  - not central to current PEZ-transfer verdict
- Estimated cost:
  - metric implementation + first diagnostic: ~2-4h

### 5. F2 surrogate event-aligned contact analysis

- Priority: `Medium`
- Why:
  - scientifically interesting, especially for `strike`
  - but blocked in exact form
- Estimated cost:
  - pseudo-contact labeling + first run: ~3-6h

### 6. Literal ridge baseline for F6

- Priority: `Low`
- Why:
  - mostly a documentation/ablation exercise
  - low expected scientific upside
- Estimated cost:
  - ~30-60 min

---

## 4. Recommended execution order

Recommended order from the current state of the project:

1. **F5 Frame shuffle on `push` and `strike`**
   - highest-value validation
   - directly tests whether the current positive results are temporal rather than static

2. **F3 CKA on `push`, `strike`, optionally `reach`**
   - quantify cross-task layer similarity
   - especially test whether `push` and `strike` align most strongly near their positive layers

3. **F4-A split-by-value on `push`**
   - analyze whether PEZ-like kinematic curves vary by `mass` / `friction` regime

4. **F2 surrogate event-aligned analysis**
   - only after defining a robust pseudo-contact labeling scheme

5. **F1 bounded-DTW / phase-space**
   - useful if the project explicitly pivots from decoding to trajectory-consistency validation

6. **Optional F6 ridge baseline**
   - low priority cleanup ablation

## Recommendation in one sentence

If only one next experiment should be run, it should be **F5 frame shuffle on the strongest positive tasks (`push`, `strike`) using the final 3D direction targets**.
