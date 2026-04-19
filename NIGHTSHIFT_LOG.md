# Nightshift Log — Phase 2 PEZ reproduction attempt

Autonomous operation log while user is away (~12 hours).
All major decisions, Codex discussions, hypothesis tests recorded here.

## Starting state (2026-04-18 evening)

- Phase 1 (mean-pool, 24 layer ViT-L): mass/friction all never-linear, ee_pos control always-linear ✓
- Phase 2 launched: 1500-ep token-patch cache complete at /mnt/md1/solee/features/physprobe_vitl_tokenpatch/push (~1 TB)
- Two probes running in parallel:
  - **Probe A** (GPU 0, static H2): mass, obj_friction, surface_friction, ee_pos, object_pos
  - **Probe B** (GPU 2, PEZ Fig 2c direct analog): ee_speed, ee_accel_magnitude, **ee_direction**, object_{speed,accel_magnitude,direction}
- Both in ingestion phase, ETA ~1-1.5 hours each

## Hypothesis (user prediction)

User expects PEZ pattern to appear — speed/accel always-linear from L0, direction emerges at mid-layer (~1/3 depth ≈ layer 8) with R²>=0.5.

If PEZ appears → proceed to force/event-aligned probes (Phase 3)
If not → code-level audit first (not premature conclusion)

## Entry format

Each entry:
```
## [YYYY-MM-DD HH:MM] <topic>
<observation>
<action>
<next>
```

---

## [2026-04-18 14:06 UTC] Nightshift start / C1
Observed two active Phase 2 Push probes:
- static token-patch run (`run-tag=phase2`) alive: bash PID 3951607, python PID 3951612
- kinematic token-patch run (`run-tag=phase2_kinematic`) alive: bash PID 4002921, python PID 4002926
Both are currently in token-cache ingestion rather than fold sweeps.

Action:
- Keep both runs untouched.
- Monitor for first landed Phase 2 CSV / sanity / verdict artifact.

Next:
- Record `[FIRST CSV landed]` immediately when any phase2 result file appears.

## [2026-04-18 14:16 UTC] STEP 2-P1 / kinematic target extension
Added six Push dynamic scalar targets to `probe_physprobe.py` as direct PEZ Fig. 2(c) analogs:
- `ee_speed`
- `ee_accel_magnitude`
- `ee_direction`
- `object_speed`
- `object_accel_magnitude`
- `object_direction`

Implementation choice:
- speed / accel magnitude = episode mean of per-frame vector norms
- direction = circular mean of per-frame XY velocity angle using `atan2(mean(sin), mean(cos))`

Action:
- Keep angle as scalar target (not sin/cos), following the PEZ reproduction lesson.

Next:
- Sanity-check target variance and token cache shape before full parallel run.

## [2026-04-18 14:18 UTC] STEP 2-P2 / sanity checks
Sanity checks passed.

Observed:
- token cache shape confirmed on real file: `layer_0_window_0.shape = (256, 1024)`, dtype `float16`
- `window_starts[:5] = [0, 4, 8, 12, 16]`
- new targets all have non-zero variance on real Push episodes

Sample target variance on 8 episodes:
- `ee_speed`: `1.03e-4`
- `ee_accel_magnitude`: `3.88e-4`
- `ee_direction`: `1.99e-1`
- `object_speed`: `1.08e-5`
- `object_accel_magnitude`: `1.50e-3`
- `object_direction`: `5.92e-1`

CONSENSUS:
- No reason to block the kinematic PEZ-analog run.

Next:
- Launch full kinematic token-patch probe in parallel with the existing static run.

## [2026-04-18 14:19 UTC] [LAUNCHING LONG RUN: phase2_kinematic]
Started full Push token-patch kinematic probe in parallel.

Command family:
- `probe_physprobe.py --task push --model large --feature-type token_patch --feature-root /mnt/md1/solee/features/physprobe_vitl_tokenpatch --targets ee_speed ee_accel_magnitude ee_direction object_speed object_accel_magnitude object_direction --run-tag phase2_kinematic`

Resource decision:
- Existing static token-patch run left untouched
- Kinematic run moved to a separate GPU to avoid interference

Current status snapshot:
- static run still in token-cache ingestion
- kinematic run entered normal token-cache ingestion path

Next:
- Watch for the first landed `phase2` CSV / verdict artifact.

## [2026-04-18 15:08 UTC] static phase2 run entered probing
Observed the existing static token-patch run complete token-cache ingestion:
- feature load reached `1500 / 1500`
- parquet target load reached `1500 / 1500`
- run transitioned into `Probe [mass]`

Interpretation:
- The static Phase 2 path is now in the actual 5-fold / 20-HP sweeps.
- First landed CSV should appear after the first target finishes.

Next:
- Record `[FIRST CSV landed]` as soon as any `probe_push_*phase2*.csv` file is written.

## [2026-04-18 15:14 UTC] static phase2 sweep timing
Observed early `Probe [mass]` timing on the static token-patch run:
- layer 1 finished at ~`144 s`
- layer 2 cumulative ~`203 s`
- layer 3 cumulative ~`273 s`
- layer 4 cumulative ~`387 s`

Interpretation:
- Static token-patch probing is materially slower than mean-pool Phase 1.
- Expect multi-hour runtime for the full static target set.
- Kinematic run is likely to have similar per-layer cost once it exits ingestion.

Action:
- No protocol change.
- Continue passive monitoring and wait for the first landed CSV.

Next:
- Record `[FIRST CSV landed]` when target 1 completes for any phase2 run.

## [2026-04-18 15:20 UTC] preliminary scenario-B audit
Pre-computed a line-by-line solver audit target against `/home/solee/pez/step3_probe.py`.

Current finding:
- no obvious trainable-solver divergence detected yet

Matched items:
- `LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]`
- `WD_GRID = [0.01, 0.1, 0.4, 0.8]`
- `MAX_EPOCHS = 400`
- `PATIENCE = 40`
- `normalize_train_val()` logic
- `compute_r2()` logic
- full-batch manual Adam path in `fit_trainable_batched()`
- GroupKFold by group ids (PhysProbe uses episode id)

Known non-mismatch differences:
- PhysProbe currently aggregates windows to episode-level before probing
- token-patch mode currently flattens `(256, 1024)` to one vector rather than using patch-wise attentive readout
- target semantics differ (real manipulation state vs synthetic ball physics)

Interpretation:
- If PEZ-like direction fails later, the first suspicion should not be “wrong optimizer implementation”.

Next:
- Wait for the first CSV before escalating into a formal scenario-B verdict.

## [2026-04-18 15:27 UTC] [FIRST CSV landed]
First Phase 2 artifact landed:
- `artifacts/results/probe_push_mass_large_token_patch_phase2.csv`
- matching figure: `artifacts/figures/curve_push_mass_large_token_patch_phase2.png`

Key metrics:
- `L0 = -0.1451`
- `L8 = 0.0239`
- `peak = 0.0991 @ layer 14`
- `last = 0.0728`

Interpretation:
- Push `mass` remains `never-linear` under the full token-patch PEZ recipe.
- This is negative evidence for static-physics emergence in Push.
- It does **not** yet decide Scenario A/B because the direct PEZ analogs are the kinematic targets, especially `ee_direction` / `object_direction`.

Action:
- Keep both runs untouched.
- Prioritize the kinematic run for verdict purposes.

Next:
- Wait for the first landed kinematic CSV, especially a direction target.

## [2026-04-18 14:38 UTC] [CLAUDE AUDIT] Phase 2 mass worse than Phase 1 — flag

First landed Phase 2 CSV: `probe_push_mass_large_token_patch_phase2.csv`

Comparison:
- Phase 1 mean-pool mass: L0=0.058, peak L19=0.203
- Phase 2 token-patch mass: L0=**-0.145** (negative!), peak L14=0.099

Concern:
- Phase 2 (more info) should be ≥ Phase 1 (less info).
- L0 negative R² → probe worse than constant prediction → overfitting signal
- Feature dim exploded: 262144 (256 patches × 1024 D) vs 1024 in Phase 1
- With WD grid capped at 0.8, 262k-dim probe may be under-regularized

Possible causes (ranked):
1. **Under-regularization** — probe flatten to 262k dim, sweep max WD=0.8 insufficient
2. **temporal_last_patch too narrow** — keeps last tube only; mass is episode-static, full temporal mean would be more informative
3. Feature extraction bug (less likely — speed/accel probe hasn't landed yet, will be stronger sanity)

Decision:
- Wait for kinematic CSV first (speed/accel/direction) — those are primary PEZ analogs
- Direction PEZ pattern is the key verdict, not mass
- If direction emerges → mass weirdness is secondary concern
- If direction ALSO fails → trigger code audit (Scenario B)

Next:
- Monitor for kinematic CSV landing

## [2026-04-18 14:41 UTC] [FIRST CSV landed] Confirmed on disk

File:
- `artifacts/results/probe_push_mass_large_token_patch_phase2.csv`

Metrics:
- `L0 = -0.1451`
- `L8 = 0.0239`
- `peak = 0.0991 @ L14`
- `last = 0.0728`

Classification:
- `never-linear`

Interpretation:
- The full token-patch PEZ recipe does **not** rescue Push `mass`.
- Static physics remains weakly decodable at best, and worse than Phase 1 mean-pool.
- This sharpens the need to use `ee_direction` / `object_direction` as the primary Scenario A/B trigger.

Status:
- static run still active on `obj_friction`
- kinematic run still active on `ee_speed`

## [2026-04-18 14:44 UTC] [FIRST CSV landed] Kinematic control behaves PEZ-like-in-control-sense

File:
- `artifacts/results/probe_push_ee_speed_large_token_patch_phase2_kinematic.csv`

Metrics:
- `L0 = 0.6711`
- `L8 = 0.9284`
- `peak = 0.9312 @ L13`
- `last = 0.9201`
- `first R² >= 0.8 = L1`

Classification:
- `always-linear`

Interpretation:
- `ee_speed` matches the expected PEZ Fig. 2(c) control family: scalar magnitude is highly decodable from the input.
- This is the first positive sign that the token-patch Phase 2 pipeline is behaving in the right qualitative regime.
- Scenario A is still undecided until `ee_direction` or `object_direction` lands.

## [2026-04-18 14:57 UTC] [CSV landed] Mixed Phase 2 evidence

Files:
- `artifacts/results/probe_push_obj_friction_large_token_patch_phase2.csv`
- `artifacts/results/probe_push_ee_accel_magnitude_large_token_patch_phase2_kinematic.csv`

`obj_friction` metrics:
- `L0 = -0.2647`
- `L8 = -0.1305`
- `peak = -0.0626 @ L3`
- `last = -0.1389`
- classification: `never-linear`

`ee_accel_magnitude` metrics:
- `L0 = 0.3468`
- `L8 = 0.6734`
- `peak = 0.6943 @ L14`
- `last = 0.6879`
- `first R² >= 0.8 = none`
- provisional classification: `PEZ-like` by peak-depth rule, but **not** `always-linear`

Interpretation:
- Static friction is even less recoverable than mass under token-patch features.
- `ee_accel_magnitude` is clearly more decodable than static physics, but it does **not** yet satisfy the strict H1-style expectation (`L0 >= 0.5`).
- The decisive verdict is still `ee_direction` / `object_direction`. If direction emerges while speed stays shallow-high, the PEZ-analog story remains alive even with weaker acceleration.

## [2026-04-18 15:10 UTC] [CSV landed] `ee_direction` strongly disfavors Scenario A

File:
- `artifacts/results/probe_push_ee_direction_large_token_patch_phase2_kinematic.csv`

Metrics:
- `L0 = 0.0658`
- `L8 = 0.3816`
- `peak = 0.4383 @ L23`
- `last = 0.4383`
- `first R² >= 0.5 = none`
- `peak depth = 0.958`

Classification:
- `intermediate` by simple thresholding, but **not PEZ-like**

Interpretation:
- The shallow baseline is good (`L0` low), but the signal never reaches the preregistered `R² >= 0.5` bar and peaks at the final layer instead of the middle.
- If `object_direction` behaves similarly, Scenario B should trigger immediately after the kinematic run completes.
- Keep the current run untouched until `object_direction` lands; do not declare final verdict yet.

## [2026-04-18 15:15 UTC] [CLAUDE DISCUSSION] Scalar-angle wrap as next diagnostic

Question from watcher:
- Could the failed `ee_direction` curve be an artifact of scalar-angle wrap, given that manipulation trajectories can revisit the `-pi/pi` boundary more often than the PEZ synthetic ball setting?

Response:
- Yes, this is a credible **next diagnostic**.
- In PEZ, scalar `angle` beat `(sin, cos)` because the motion dataset had clean straight-line trajectories with little practical wrap ambiguity.
- In PhysProbe Push, the end-effector/object motion can curve, stall, reverse, or cluster near boundary angles, so raw scalar-angle regression can understate a representation that is direction-aware.

Decision:
- Do **not** interrupt the current scalar-angle baseline run.
- Let the current kinematic run finish completely.
- If final scalar-angle verdict remains weak, launch a direct `sin/cos` direction variant as the next Scenario-B ablation.

CONSENSUS:
- Treat scalar-angle as the preregistered baseline.
- Treat `sin/cos` as the first manipulation-specific follow-up if scalar-angle fails.

## [2026-04-18 15:39 UTC] [VERDICT: B] Scalar-angle direction fails Push PEZ criterion

Kinematic final metrics:

`ee_direction`
- `L0 = 0.0658`
- `L8 = 0.3816`
- `peak = 0.4383 @ L23`
- `peak depth = 0.958`

`object_direction`
- `L0 = -0.4051`
- `L8 = -0.0331`
- `peak = 0.0287 @ L23`
- `peak depth = 0.958`

Decision:
- Scenario A is rejected.
- Scenario B triggers.

Evidence:
- Neither direction target reaches `R² >= 0.5`.
- Both direction targets peak at the final layer rather than in the middle.
- `ee_speed` is high from shallow layers, so the token-patch pipeline is not globally broken.
- Therefore the failure is specific to the direction target/setup, not to feature ingestion.

Next action:
- Reuse the completed solver audit (already logged above).
- Launch `sin/cos` direction as the first Scenario-B ablation using the **same** token-patch cache and trainable 20-HP solver.

## [2026-04-18 15:40 UTC] [DECISION POINT] Launching sin/cos direction ablation

Planned ablation:
- task: `push`
- model: `large`
- feature type: `token_patch`
- cache: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch`
- targets: `ee_direction_sincos`, `object_direction_sincos`
- solver: unchanged (`trainable`, 20 HP)
- CV: unchanged (5-fold GroupKFold by episode)

Rationale:
- Manipulation may invalidate scalar-angle regression due to circular wrap.
- This isolates target parameterization as the only changed factor.

## [2026-04-18 15:44 UTC] [LAUNCHING LONG RUN: phase2_kinematic_sincos]

Command:
- `env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/physrepa_tasks/probe_physprobe.py --task push --model large --feature-type token_patch --feature-root /mnt/md1/solee/features/physprobe_vitl_tokenpatch --targets ee_direction_sincos object_direction_sincos --device cuda:0 --run-tag phase2_kinematic_sincos`

Notes:
- Same token-patch cache as the scalar-angle run.
- Same solver, folds, normalization, and selection metric.
- Only changed factor: direction target parameterization (`angle` -> `sin/cos`).

## [2026-04-18 17:21 UTC] [CONSENSUS] Watcher and run results agree on ee-side PEZ rescue

Consensus:
- `ee_direction_sincos` is the first clear manipulation-domain PEZ analog found in Push.
- Scalar-angle was the wrong parameterization for this domain because of circular wrap.
- `ee_direction_sincos` should be counted as an `H2`-style emergence target, not as an `H1` control.
- `ee_pos` remains the primary `H1` control (`L0 = 0.9440`).

Recorded outcome:
- `Scenario A partial`
- success scope: `ee-side`
- failure scope: `object-side direction`, `static physics`

## [2026-04-18 17:31 UTC] [PHASE3 BLOCKED] Contact labels appear to be absent in public PhysProbe

Motivation:
- After the Phase 2 partial success, the planned next step was an event-aligned force/contact probe.

Audit results:
- `push` full scan, `physics_gt.contact_flag`: `1500 / 1500` episodes checked, `0` non-zero episodes, `max mean = 0.0`
- `push` sampled contact channels:
  - `contact_flag = 0`
  - `contact_force = 0`
  - `contact_finger_l_object_flag = 0`
  - `contact_finger_l_object_force = 0`
  - `contact_object_surface_flag = 0`
  - `contact_object_surface_force = 0`
  - `contact_point = 0`
- sampled `strike`, `peg_insert`, `nut_thread`, `drawer` episodes also showed all-zero `contact_flag` and `contact_point`

Decision:
- Do **not** launch a force/contact probe on all-zero labels.
- Treat Phase 3 as dataset-blocked for the currently available public PhysProbe release.

Implication:
- The next meaningful path is not force probing on these labels.
- Future follow-up should either:
  1. obtain a PhysProbe release with populated contact labels, or
  2. define an alternative event target from non-zero kinematic/object-state channels.

## [2026-04-18 17:52 UTC] [PHASE 2c-A] Attentive probe driver added

Added:
- `probe_physprobe_attentive.py`

Design:
- reuses PhysProbe token-patch cache
- reuses `load_targets()` from `probe_physprobe.py`
- attentive readout = `AttentivePooler(depth=4, num_heads=16) + linear regression head`
- same CV protocol:
  - 5-fold `GroupKFold` by episode id
  - z-score normalization
  - LR x WD sweep over the same 20 HP grid

## [2026-04-18 18:10 UTC] [PHASE 2c-B] Full attentive sweep deemed too expensive; switching to targeted pilot

Observed runtime:
- `ee_direction_sincos`
- 3 layers only: `{0, 8, 13}`
- elapsed wall time before first layer completed: `13m51s`

Interpretation:
- A full attentive sweep over `24 layers x 3 targets x 20 HP x 5 folds` is not a good use of the remaining night budget.

Decision:
- Stop the broad dry-run.
- Replace it with a targeted attentive pilot on the most diagnostic layer first:
  - `layer 13`
  - targets: `ee_direction_sincos`, `object_direction_sincos`, `ee_speed`
- Purpose:
  - test whether attentive readout rescues weak object-side direction
  - keep Priority 1 alive without burning the entire night on an oversized sweep

## [2026-04-18 18:14 UTC] [LAUNCHING LONG RUN: attentive_pilot_l13]

Attentive pilot:
- task: `push`
- model: `large`
- layer: `13`
- targets:
  - `ee_direction_sincos`
  - `object_direction_sincos`
  - `ee_speed`
- session id: `12734`

## [2026-04-18 18:14 UTC] [LAUNCHING LONG RUN: reach_token_extract]

Reach token extraction:
- task: `reach`
- model: `large`
- recipe: `resid_post + temporal_last_patch`
- output root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/reach`
- session id: `45442`

## [2026-04-18 18:17 UTC] [BLOCKED] Reach extract hit sandbox filesystem boundary

Observed:
- sandboxed invocation could read data but failed to create `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/reach`
- error: `OSError: [Errno 30] Read-only file system`

Decision:
- Relaunch Reach extraction with escalation so the cache can be written to `/mnt`

## [2026-04-18 18:18 UTC] [LAUNCHING LONG RUN: reach_token_extract_escalated]

Reach token extraction relaunched successfully with writable `/mnt`.
- session id: `35540`

## [2026-04-18 19:05 UTC] [PHASE 2c-C] Reach extraction complete; launching probe next

Observed:
- Reach token extraction completed successfully
- final progress: `600 / 600`
- cache path: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/reach`

Attentive pilot status:
- Push attentive pilot is alive on GPU 0
- it completed single-layer token loading and entered training for `ee_direction_sincos`

Decision:
- Add Reach kinematic targets to `probe_physprobe.py`
- launch Reach token-patch probe immediately on a separate GPU
- targets:
  - `ee_direction_sincos`
  - `ee_speed`
  - `ee_accel_magnitude`
  - `fake_mod5` (negative-control fallback in place of static physics)

## [2026-04-18 15:21 UTC] [CSV landed] Object-side kinematics are weaker than arm-side kinematics

File:
- `artifacts/results/probe_push_object_speed_large_token_patch_phase2_kinematic.csv`

Metrics:
- `L0 = -0.2682`
- `L8 = 0.3301`
- `peak = 0.3978 @ L12`
- `last = 0.3814`
- `first R² >= 0.8 = none`

Classification:
- `intermediate` (not `always-linear`)

Interpretation:
- The arm-side control (`ee_speed`) is easy from shallow layers, but the object-side speed signal is not.
- This makes Push look less like the single-ball PEZ regime and more like an interaction-mediated system where even some kinematic quantities emerge gradually.
- This strengthens the case for keeping both arm-side and object-side targets in the final Phase 2 comparison, rather than treating all kinematics as simple controls.

## [2026-04-18 15:27 UTC] [CSV landed] Object acceleration clears 0.5, but too late

File:
- `artifacts/results/probe_push_object_accel_magnitude_large_token_patch_phase2_kinematic.csv`

Metrics:
- `L0 = 0.1995`
- `L8 = 0.5001`
- `peak = 0.5224 @ L20`
- `last = 0.5124`
- `first R² >= 0.8 = none`

Classification:
- `intermediate`

Interpretation:
- This is the first object-side kinematic target to cross the preregistered `peak >= 0.5` threshold.
- However, the peak is too late (`L20 / 24 = 0.83` depth), so it still fails the PEZ-like mid-depth criterion.
- The current pattern remains: object-side dynamics become decodable, but not with paper-like timing.

## [2026-04-18 15:03 UTC] [CLAUDE AUDIT] First kinematic CSV — ee_speed positive signal

`probe_push_ee_speed_large_token_patch_phase2_kinematic.csv` landed.

ee_speed curve:
- L0 = 0.671
- L8 = 0.928
- peak L13 = 0.931

Assessment:
- PEZ paper `speed` expected L0 >= 0.8 (always-linear control). Ours is 0.671 — lower but reasonable, not pathological.
- Clean monotonic rise to L8 (+0.26 delta), then flat near peak.
- NOT classic "always-linear from input" — mild emergence. But speed IS encoded.

obj_friction still weak:
- L0 = -0.265 (worse than constant)
- Same under-regularization hypothesis likely.

Decision:
- Wait for ee_direction — that's the primary PEZ verdict target.
- If direction emerges sharply at ~L8 with peak 0.5+, Scenario A (PEZ reproduced).

## [2026-04-18 15:24 UTC] [CLAUDE AUDIT] ee_direction landed — partial PEZ, with critical hypothesis

ee_direction: L0=0.066, L8=0.382, **peak L23=0.438**

Classification per EXPERIMENT_DESIGN §4:
- L0 low ✓
- Peak >= 0.5 ✗ (0.438)
- Peak at depth > 60% ✗ (L23 is last layer, 96% depth)
- Verdict: "intermediate" — partial PEZ, not classic pattern

Compare to PEZ Fig 2c Large:
- Direction: L0 ≈ 0.33, peak L16 = 0.876, late decline to L23=0.835
- Ours: L0 = 0.066, peak L23 = 0.438, no decline

Two qualitative differences:
1. Our peak is half of paper's (0.44 vs 0.88)
2. Paper peaks mid then declines; ours still rising at L23

Hypothesis — **circular wrap problem on scalar angle target**:
- ee_direction computed as atan2(vx,vy) ∈ [-π, π]
- zscore normalization assumes linear scalar; +π and -π are opposite zscored values but physically identical
- PEZ paper Kubric ball has overhead 2D straight-line motion with wrap-unaware angle mostly in clean quadrant
- Manipulation end-effector likely has much wider direction spread → more wraparound events

This might explain:
- Low peak (probe learns to minimize L2 but wraparound creates artificial jumps)
- Late peak (only deepest layer has enough capacity to approximate circular target with linear weights)

Proposed test (for Codex):
- Re-probe ee_direction with target = (sin(angle), cos(angle)) 2D vector
- Wrapping-free, MSE naturally circular-aware
- If this lifts peak significantly → confirms hypothesis

Note from PEZ history: paper lesson preferred scalar angle over sincos. BUT that was for simpler motion (ball). Manipulation may need sincos.

Also pending:
- object_speed, object_accel_magnitude, object_direction
- surface_friction Phase 2

Next action:
- Wait for object_direction (same hypothesis test — if object_direction also peaks late, pattern confirmed)
- Then nudge Codex to run sin/cos re-probe (quick — reuses same token cache)

## [2026-04-18 17:20 UTC] [CLAUDE AUDIT] SINCOS CONFIRMED PEZ PATTERN for ee_direction

Result comparison (circular wrap hypothesis TEST):

| Metric | Scalar angle | Sin/cos |
|---|---|---|
| L0 | 0.066 | **0.555** |
| L8 | 0.382 | **0.789** |
| peak | 0.438 @ L23 | **0.807 @ L13** (54% depth) |
| L23 | 0.438 | **0.797** (decline present) |

Paper PEZ Fig 2c Large direction (reference):
- L0 ≈ 0.33, peak L16 ≈ 0.876, L23 ≈ 0.835

Our sin/cos ee_direction peak = 0.807 (mid-layer, 54% depth) with late decline to 0.797.
**This is a paper-like PEZ curve for arm-side direction on manipulation data.**

VERDICT UPDATE:
- Scenario A (partial): **PEZ reproduced on ee_direction via sincos target**
- Circular wrap hypothesis CONFIRMED
- PEZ paper's scalar-angle lesson does NOT generalize to manipulation (which has wider direction distribution)

Remaining gaps:
- object_direction still fails (peak 0.084 @ L23) — object is small, motion dominated by contact periods
- static physics (mass/friction) still fail — under-regularization or truly not encoded

Note on Codex verdict marker:
- Codex's verdict file says "H1 FAIL" but this is misclassification:
  - H1 (always-decodable controls) should be evaluated on ee_pos (L0=0.944 ✓) and ee_speed (L0=0.671 marginal)
  - ee_direction_sincos is NOT an H1 control, it's a H2 emerge target — and it PASSED
- Need Codex to fix verdict logic

Next action decisions:
1. Have Codex fix verdict classification (direction → H2 not H1)
2. Declare Scenario A partial reproduction
3. Proceed to Phase 3 (event-aligned force probe) per protocol
4. Optional: sincos variant on ee_speed / ee_accel (though scalar variant was fine)

## [2026-04-18 18:34 UTC] [CONSENSUS] Phase 3 blocked, pivot to attentive Push + Reach generality

Decision:
- Phase 3 force/contact probe remains blocked by all-zero public labels.
- Next best use of the night shift:
  1. Push attentive probe on existing token-patch cache
  2. Reach token-patch extraction and PEZ-analog probe

Rationale:
- object_direction_sincos still fails under linear flatten (`peak=0.084 @ L23`)
- PEZ Figure 8 showed attentive readout can recover structure hidden from linear pooling
- Reach provides a clean generality test for arm-side kinematic PEZ without contact confounds

## [2026-04-18 18:41 UTC] [LAUNCHING LONG RUN: push_attentive_pilot_l13]

Started attentive diagnostic pilot on Push using existing token-patch cache.

Config:
- cache: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
- task: `push`
- model: `large`
- target layers: `13` only
- targets:
  - `ee_direction_sincos`
  - `object_direction_sincos`
  - `ee_speed`

Reason for restricted scope:
- a wider attentive sweep was too expensive for same-night turnaround
- the scientific question is narrow: does attentive readout rescue the failed object-direction signal around the known ee-side PEZ layer?

## [2026-04-18 19:07 UTC] [LAUNCHING LONG RUN: reach_token_extract_escalated]

The first attempt to write Reach token-patch cache hit the sandbox filesystem boundary on `/mnt/md1/solee`.
Relaunched with escalated filesystem access using the same PEZ-aligned extraction recipe:
- `resid_post`
- `temporal_last_patch`
- output root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/reach`

## [2026-04-18 20:11 UTC] [PHASE 2c-C] Reach extraction complete; launching probe next

Reach token-patch extraction finished successfully:
- task: `reach`
- episodes: `600 / 600`
- cache root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/reach`

Probe plan:
- task: `reach`
- model: `large`
- feature type: `token_patch`
- targets:
  - `ee_direction_sincos`
  - `ee_speed`
  - `ee_accel_magnitude`
  - `fake_mod5` (negative-control fallback in place of nonexistent Reach mass randomization)

## [2026-04-18 20:24 UTC] [LAUNCHING LONG RUN: reach_phase2c_probe]

Started Reach token-patch probe on a separate GPU from the attentive Push pilot.

Current expectation:
- `ee_direction_sincos` should show the same manipulation-domain PEZ-like mid-depth structure observed on Push ee-side
- `ee_speed` should be shallow/high from early layers
- `fake_mod5` should stay near chance / non-decodable

## [2026-04-18 20:31 UTC] [STATUS]

Live jobs:
- Push attentive pilot (`layer 13`, three targets) is still active on GPU 0
- Reach token-patch probe is actively loading features and has begun ingesting episode caches

No attentive or Reach result CSV has landed yet at this timestamp.

## [2026-04-18 21:07 UTC] [VERDICT: Reach probe complete]

Reach Phase 2c token-patch probe finished and wrote all expected outputs:
- `probe_reach_ee_direction_sincos_large_token_patch_phase2c_reach.csv`
- `probe_reach_ee_speed_large_token_patch_phase2c_reach.csv`
- `probe_reach_ee_accel_magnitude_large_token_patch_phase2c_reach.csv`
- `probe_reach_fake_mod5_large_token_patch_phase2c_reach.csv`
- `verdict_phase2c_reach_reach.json`
- `EXPERIMENT_RESULTS_phase2c_reach_reach.md`

Key numbers:
- `ee_direction_sincos`: `L0=0.302`, `L8=0.353`, `peak=0.396 @ L20` -> not PEZ-like
- `ee_speed`: `L0=0.598`, `L8=0.813`, `peak=0.825 @ L17` -> strongly decodable but not strict-H1
- `ee_accel_magnitude`: `L0=0.717`, `L8=0.854`, `peak=0.873 @ L10` -> classifier-positive PEZ-like / mid-depth refinement
- `fake_mod5`: negative at every layer -> integrity check passed

Interpretation:
- Push ee-side PEZ does **not** trivially generalize to Reach direction
- Reach confirms the token-patch probe is scientifically sane because the fake target remains non-decodable
- The strongest next discriminator is still the Push attentive pilot, not another linear re-run

## [2026-04-18 21:10 UTC] [STATUS] attentive pilot health check

Push attentive pilot has now been running for >2h, but it is not obviously hung:
- python process still active
- CPU usage ~`980%`
- GPU 0 at ~`100%` utilization and ~`7.7 GiB` memory used
- no output CSV has landed yet

Conclusion:
- do **not** kill yet
- wait for first attentive result before starting a new overlapping high-cost probe

## [2026-04-18 21:12 UTC] [CONSENSUS]

After Reach:
- the linear token-patch story is now clear enough to stop re-running linear variants
- next high-value question is whether attentive readout rescues failed object-side direction on Push
- if attentive still fails, the next task should be `strike`, not `huge-on-push`

## [2026-04-18 21:22 UTC] [PHASE 2c-D] Push vs Reach verdict committed

Committed and pushed:
- Reach result files
- comparative report: `artifacts/results/EXPERIMENT_RESULTS_phase2_push_vs_reach.md`

High-level comparison:
- Push `ee_direction_sincos`: PEZ-like
- Reach `ee_direction_sincos`: not PEZ-like
- Reach `ee_accel_magnitude`: classifier-positive PEZ-like / mid-depth refinement
- Reach `fake_mod5`: negative at all layers (pipeline integrity passed)

Implication:
- PEZ-aligned kinematic emergence is not universal across manipulation tasks
- interaction-rich Push remains the strongest positive case

## [2026-04-18 21:24 UTC] [DECISION POINT]

Attentive pilot remains active with no landed outputs, but is still consuming CPU/GPU and therefore is not treated as hung.

GPUs 1-3 are idle. To use the remaining night-shift window efficiently, the next task is:
- launch `strike` token-patch extraction with the same PEZ-aligned recipe

Rationale:
- `strike` is the best next candidate for object-side dynamics because it is more contact-dominant than `push`
- `huge-on-push` is lower priority until the attentive rescue question is resolved

## [2026-04-18 21:28 UTC] [LAUNCH ATTEMPT] strike_token_extract

First Strike extraction launch failed immediately because `extract_token_features.py` does not expose `--capture` / `--pooling` as CLI flags.

Resolution:
- confirmed from source that the script already hard-codes the intended PEZ recipe:
  - `resid_post`
  - `temporal_last_patch`
- relaunched with only the supported arguments

## [2026-04-18 21:29 UTC] [LAUNCHING LONG RUN: strike_token_extract]

Started Strike token-patch extraction:
- task: `strike`
- model: `large`
- output root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch`
- recipe: script-internal `resid_post + temporal_last_patch`

This uses an idle GPU in parallel with the still-running Push attentive pilot.

## [2026-04-18 21:36 UTC] [CONSENSUS] Kill Push attentive pilot

Decision:
- terminate the long-running Push attentive pilot

Reasoning:
- the pilot was restricted to a single diagnostic layer (`13`) yet still exceeded ~3h wall time
- existing Push linear token-patch results already answer the main scientific question:
  - `ee_direction_sincos` shows a PEZ-like curve
  - `object_direction_sincos` fails under the baseline readout
- attentive rescue of object-direction would be nice-to-have, not necessary for the current night-shift objective
- keeping the pilot alive risks wasting CPU time and slowing the newly launched `strike` extraction

Action:
- free resources for higher-value next steps (`strike`, later possibly `huge`)

## [2026-04-18 21:40 UTC] [ACTION CONFIRMED] attentive pilot terminated

Follow-up:
- the first `pkill -f probe_physprobe_attentive.py` attempt did not match the wrapper command line
- by the time the direct PID kill was attempted, the attentive pilot processes were already gone
- process table check confirmed no surviving `probe_physprobe_attentive` / `attentive_pilot_l13` processes

Resource state after termination:
- GPU 0 is now idle
- `strike` token extraction continues on GPU 1

Operational conclusion:
- attentive object-direction rescue is dropped from the current night-shift scope
- the night shift now focuses on completing `strike` extraction and the next contact-heavy linear probe

## [2026-04-18 21:48 UTC] [BLOCKED-RESOLVED] strike extraction ended with partial cache

Strike token extraction did not reach `3000 / 3000`.

Observed failure:
- extraction stopped at `2895 / 3000`
- traceback ended in `safetensors_rust.SafetensorError: No space left on device`
- cache audit shows `2896` completed `.safetensors` episode files under:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike`

Decision:
- do **not** attempt to recover the missing `104` episodes tonight
- proceed directly to probing on the available `2896` cached episodes

Rationale:
- statistical power remains high
- `probe_physprobe.py` enumerates episodes from existing safetensors, so partial caches are handled naturally
- time is better spent extracting the scientific verdict than repairing the tail of the cache

## [2026-04-18 21:50 UTC] [LAUNCHING LONG RUN: strike_phase2c_probe]

Started Strike token-patch probe on the partial `2896`-episode cache.

Config:
- task: `strike`
- model: `large`
- feature type: `token_patch`
- targets:
  - `ee_direction_sincos`
  - `ee_speed`
  - `ee_accel_magnitude`
  - `fake_mod5`
- CV: `5-fold GroupKFold by episode_id`
- solver: `trainable 20-HP`
- norm: `zscore`

Goal:
- test whether the Push ee-side PEZ analog transfers to a more contact-heavy manipulation task
- use `fake_mod5` again as an integrity guardrail

## [2026-04-18 21:58 UTC] [BLOCKED-RESOLVED] first Strike probe launch failed on target spec

The first Strike probe attempt exited immediately with:
- `ValueError: Unknown targets for strike: ['ee_direction_sincos', 'ee_speed', 'ee_accel_magnitude', 'fake_mod5']`

Cause:
- `probe_physprobe.py` had these kinematic targets added for `push` and `reach`, but not yet for `strike`

Resolution:
- added minimal Strike target support for:
  - `ee_speed`
  - `ee_accel_magnitude`
  - `ee_direction_sincos`
  - `fake_mod5`
- kept scope intentionally narrow to preserve the preregistered comparison with Push/Reach

## [2026-04-18 22:00 UTC] [LAUNCHING LONG RUN: strike_phase2c_probe_retry]

Relaunched Strike probe after target-spec fix.

Observed early status:
- feature loader started successfully on the partial `2896`-episode cache
- initial throughput ~`2.2 s / episode` during feature ingestion
- GPU 0 active again for probe compute

## [2026-04-19 00:05 UTC] [BLOCKED-RESOLVED] Strike probe crash root cause identified

The retried Strike probe died during feature loading.

Observed traceback:
- `SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered`

Audit result:
- scanned all `2896` Strike safetensors
- exactly **one** corrupted file was found:
  - `002895.safetensors`
- corruption pattern is consistent with the earlier disk-full extraction crash

Decision:
- quarantine the single bad file instead of re-extracting the whole tail

Action taken:
- moved:
  - `002895.safetensors`
  - -> `002895.safetensors.corrupt`
- valid cache count becomes `2895`

## [2026-04-19 00:08 UTC] [LAUNCHING LONG RUN: strike_phase2c_probe_clean2895]

Relaunched Strike token-patch probe on the cleaned `2895`-episode cache.

Initial status check:
- feature loader restarted successfully
- early throughput again ~`2.25 s / episode`
- GPU 0 active for probe compute

Interpretation:
- this was a cache-integrity failure, not a methodological failure
- if the run now completes, the Strike result can be treated as statistically valid despite missing `105` episodes

## [2026-04-19 02:10 UTC] [VERDICT: Strike probe complete]

Strike Phase 2c finished successfully on the cleaned `2895`-episode cache.

Key results:
- `ee_direction_sincos`
  - `L0=0.697`
  - `L8=0.871`
  - `peak=0.885 @ L11`
  - `last=0.877`
  - `classification=PEZ-like`
- `ee_speed`
  - `L0=0.883`
  - `L8=0.957`
  - `peak=0.963 @ L11`
  - `classification=always-linear`
- `ee_accel_magnitude`
  - `L0=0.664`
  - `L8=0.876`
  - `peak=0.896 @ L12`
  - `classification=PEZ-like`
- `fake_mod5`
  - negative at all layers
  - integrity check passed

Scientific conclusion:
- `strike` is the strongest positive PEZ transfer case observed in PhysProbe so far
- contact-rich interaction appears to help, not hurt, the emergence signal on end-effector kinematics

Operational conclusion:
- the night shift now has a complete three-task comparison:
  - `push`: partial positive
  - `reach`: integrity/generalization check
  - `strike`: strongest positive

## [2026-04-18 ~20:30 UTC] [CLAUDE AUDIT] Reach direction much weaker than Push

First Reach CSV landed: `probe_reach_ee_direction_sincos_large_token_patch_phase2c_reach.csv`

Comparison:
| Metric | Push | Reach |
|---|---|---|
| L0 | 0.555 | 0.302 |
| L8 | 0.789 | 0.353 |
| Peak | 0.807 @ L13 (54% depth) | 0.396 @ L20 (83%) |
| L23 | 0.797 | 0.395 |

Observations:
- Reach peak 0.396 is less than half of Push's 0.807
- Reach peak at 83% depth, not mid-layer — unlike Push
- Reach has no object interaction, so direction variance is much lower (robot goes toward target)
- Reach has 600 ep vs Push 1500 — less training data

Interpretation:
- NOT a clean PEZ reproduction on Reach
- BUT direction signal is present (L0 not zero)
- Suggests task complexity matters: PEZ-like emergence needs complex kinematics to be learnable in mid-layers
- This is expected behavior given Reach's simplicity

Still pending:
- Reach ee_speed, ee_accel_magnitude, fake_mod5
- Push attentive pilot L13 results

Decision on next steps:
- Wait for more Reach CSVs to complete the generality assessment
- Once both Reach + attentive done → decide on Strike task (contact-heavy, between Push and Reach complexity)
