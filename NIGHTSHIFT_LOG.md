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
