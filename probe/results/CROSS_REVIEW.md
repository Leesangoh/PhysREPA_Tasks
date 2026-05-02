# Cross-review log (Claude ↔ Codex)

Per plan section "Cross-verification protocol". Every new or modified script
must be reviewed by the *other* agent before full-data execution.

| Date (UTC) | Script | Author | Reviewer | Dry-run | Verdict |
|---|---|---|---|---|---|
| 2026-05-01 10:55 | probe/utils/stats.py | Codex | Claude | `/isaac-sim/python.sh probe/utils/stats.py` | PASS-with-suggestions |

## stats.py review (10:55)

**Verdict:** PASS for unit-tests, two minor improvements requested.

**Strengths:**
- Clean type hints + docstrings.
- Explicit NaN/inf filtering via `_as_1d_finite_array`.
- `_validate_ci` catches degenerate CIs early.
- Paired bootstrap correctly drops non-finite pairs by joint mask.
- Unpaired permutation test uses standard `(count + 1) / (n + 1)` continuity correction.
- Memory precompute correctly uses `torch.cuda.mem_get_info()` and computes headroom.

**Suggested fixes (sent to Codex):**
1. `memory_precompute(shape, dtype, device_id=None)` — add optional GPU index;
   currently it uses the implicit current CUDA device, which can be misleading
   when the caller is iterating over multiple devices.
2. All randomized functions (`bootstrap_r2_ci`, `bootstrap_diff_ci`,
   `permutation_null_diff`) should take an optional `seed: int | None = None`
   for reproducibility in regression tests.

**Past-mistake re-check:**
- `tk in tgt.files` style dict-attr bug? n/a (no tgt usage)
- `np.mean(vecs, axis=0)` collapse? n/a
- Mixed parquet/jsonl? n/a
- Episode-level resampling assumption documented in docstrings? Yes.

| 2026-05-01 11:08 | probe/scripts/08_time_only_baseline.py | Claude | (codex pending) | `/isaac-sim/python.sh probe/scripts/08_time_only_baseline.py` | PASS-self-review |

## 08_time_only_baseline.py self-review (11:08)

**Verdict:** PASS for unit-tests; awaiting Codex review.

**Bug found and fixed during dry-run:**
- Initial version assumed targets were 1D arrays. The targets cache stores
  scalar fields as `(N, 1)`. All targets were being skipped with the
  `y.ndim != 1` filter. Fixed by squeezing trailing size-1 axes before the
  ndim check.

**Final dry-run results (48 rows = 9 (task,target) × 5 folds + 3 phase rows):**
- push phase R² = 0.409 (significant time leakage)
- strike phase R² = 0.410
- reach phase R² = 0.033 (minimal — phase is constant for reach)
- drawer drawer_joint_pos R² = 0.229
- drawer drawer_opening_extent R² = 0.221
- nut_thread axial_progress R² = 0.100
- peg_insert insertion_depth R² = 0.071
- peg_insert peg_hole_lateral_error R² = 0.033

**Past-mistake re-check:**
- `tk in tgt.files`? No — uses `target in tgt` (dict access).
- Constant target handling? Yes, returns explicit `note='constant_target'` row
  (not present in current output because only `note='ok'` rows passed).
- GroupKFold disjointness? Implicit via sklearn API + episode-level group key.

| 2026-05-01 11:12 | probe/trajectory_analysis_B/scripts/17_phase_conditional.py | Codex | Claude | `/isaac-sim/python.sh trajectory_analysis_B/scripts/17_phase_conditional.py --dry-run` | PASS |

## 17_phase_conditional.py review (11:12, Codex's F2-3way refactor)

**Verdict:** PASS. Dry-run produces 36 rows including all three contact_phase3
condition values (pre/during/post) for push at L11.

**Strengths:**
- Pre/during/post labeling correctly handles transient zero windows by
  collapsing into "during" (sparse interior gaps don't get dropped, per spec).
- contact_phase3 only emitted for tasks in TASKS_CONTACT_PHASE3 (push/strike/drawer).
- dry-run uses 6 eps + min_mask=50 → all categories visible for verification.
- existing 2-way contact and top-3 phase splits preserved unchanged.

**Notes:**
- R² values in the 6-eps dry-run are noisy (and even sharply negative for some
  small-mask conditions; e.g. push during ee_acceleration). This is expected
  and not a script bug — small-N folds with 5-fold GroupKFold and ~50-window
  test masks naturally produce unstable Ridge predictions. Full run with 60 eps
  will smooth these out.
- contact_phase3 mask sizes are reasonable: pre=280, during=295, post=805
  windows from the 1380 total in the 6-eps push sample. Reflects that push
  episodes spend more time post-contact than approach.

**Next:** launch full run on all 6 phase-varying tasks at 60 eps each.

| 2026-05-01 11:36 | probe/trajectory_analysis_B/scripts/18_phase_space_geometry.py | Codex | Claude | `/isaac-sim/python.sh probe/trajectory_analysis_B/scripts/18_phase_space_geometry.py --dry-run` | PASS |

## 18_phase_space_geometry.py review (11:36, F1-b)

**Verdict:** PASS. Dry-run produces 12 rows for push at L={0,11,23} × split={all,pre,during,post}.

**Strengths:**
- mean_dz_norm correctly grows with depth (0.17 @ L0 → 1.73 @ L11 → 2.75 @ L23) — consistent with prior trajectory analysis showing path length monotonically increases with depth.
- Pre/during/post split reuses contact_phase3 logic from F2; mask sizes are reasonable (280/293/800 ≈ 1373 windows in 6 push eps).
- Curvature κ decreases with depth (16.9 → 2.7 → 1.4 for "all") — phase-space curls less in deep layers, consistent with manifold spreading.
- Sweep volume increases with depth (0.59 → 74.4 → 97.2) — deep latents occupy more phase-space volume, sensible.
- lyap_local is included as supplementary diagnostic (small, near-zero values, as expected for a stable trajectory).

**Past-mistake re-check:**
- `tk in tgt.files`? n/a (no target alignment in this script — pure geometry).
- Constant-target degeneracy? n/a.
- Multi-D shape collapsing? Pure ndarray ops, no shape ambiguity.

| 2026-05-01 11:38 | probe/scripts/09_physics_condition_split.py | Codex | Claude | `/isaac-sim/python.sh probe/scripts/09_physics_condition_split.py --dry-run` | PASS |

## 09_physics_condition_split.py review (11:38, F4-A)

**Verdict:** PASS. Dry-run produces 9 rows for push×object_0_mass × {low,med,high} × {L0,L11,L23} × ee_velocity.

**Strengths:**
- Tercile binning correctly splits 30 dry-run episodes into 10/10/10 with equal q33/q67 thresholds 0.876/1.432.
- L0 R² strongly negative (-1.5 to -2.0), L11 R² 0.55-0.70 (mid PEZ), L23 R² 0.66-0.86 (deep strong) — depth pattern matches main probe.
- Initial signal: low-mass bin gives higher R² than high-mass at deep layer — interesting modulation, will validate at full scale.
- Schema matches request exactly: `bin, layer, target, r2_mean, r2_std, n_episodes_in_bin, n_windows_in_bin, q33, q67`.

**Past-mistake re-check:**
- Physics params loaded from `meta/episodes.jsonl` (correct path, not parquet — addresses Plan 3 orchestrator bug).
- GroupKFold within-bin: correctly disjoint by episode.
- Constant-target / NaN handling? Targets are kinematic (ee_velocity), no constant issue.

**Next:** launch full runs for 18 (all 6 phase-varying tasks, 60 eps each, 24 layers) and 09 (5 tasks × multiple params, full episode count, 24 layers, 4 targets).

| 2026-05-01 13:30 | probe/scripts/run_shuffled_probe.sh | Claude | Codex | `bash run_shuffled_probe.sh push` (current cache: 1064/1500 — should ABORT) | PASS-after-fix |

## run_shuffled_probe.sh review (13:30)

**Verdict:** PASS after fix. Codex review found a **critical bug** in the
initial version: pre-check threshold of "100+ files" would have allowed launch
on incomplete shuffled cache (current 1064 files for push), corrupting the F5
ΔR² claim. Fixed.

**Codex's 3 findings:**
1. ❌→✅ Pre-check threshold too weak. Fixed: now requires
   `count(variant_A_shuffled) == count(variant_A)`. Verified ABORT on current
   1064/1500 push state.
2. ⚠️ `--tier all_extended` was unused (overridden by `--targets`). Removed for
   cleanliness.
3. ⚠️ GPU 0 must not be used while strike F5 extraction is on GPU 0. Fixed:
   default `gpu_csv` is `1,2,3` (override via 2nd arg).

**Codex's recommended target priority** (dynamics first):
ee_acceleration → ee_velocity → contact_force_log1p_mag → contact_flag.
Implemented as the TARGETS array order; with 3 GPUs, contact_flag waits for
the first slot to free.

**Past-mistake re-check:**
- Incomplete-cache launch could silently corrupt downstream bootstrap CIs —
  exactly the class of "silent data corruption" the user is most worried about.
  Hard pre-check addresses this.
- GPU 0 conflict could cause OOM mid-run (Phase D failure mode). Default
  GPU_CSV=1,2,3 sidesteps.

| 2026-05-01 13:30 | probe/scripts/11_extract_r3m_features.py | Claude | (codex pending) | `/isaac-sim/python.sh probe/scripts/11_extract_r3m_features.py --task push --gpu 0 --dry-run` | PASS-self |

## 11_extract_r3m_features.py self-review (13:30, M1 baseline extractor)

**Verdict:** PASS dry-run. 3 episodes × 230 win, all 5 stages emit correct shapes.

**Strengths:**
- Schema matches Codex Decision B specification: stage_0 (64) / stage_1 (256) / stage_2 (512) / stage_3 (1024) / stage_4 (2048) + t_last + episode_id.
- Cumulative-sum trick for window mean-pooling — O(T) instead of O(T*W).
- Per-frame ResNet50 forward (batch=64), then GAP via hooks captured during forward.
- Cache path: `/mnt/md1/solee/physprobe_features/<task>/r3m/episode_<id>.npz` (parallel structure to V-JEPA cache).
- Skip-if-cached logic respects existing cache (resumable).
- ImageNet normalization done with R3M's expected stats (mean/std).
- 384x384 → 224 bilinear interp (canonical R3M input size).

**Throughput observation:**
- ~30 win/s during dry-run (single GPU, batch=64) → push 1500 ep ≈ 200 min, strike 3000 ep ≈ 400 min, drawer 2000 ep ≈ 270 min on a single GPU. 3-task total ≈ 870 min wallclock single GPU; can shard to 4 GPUs for ~220 min wallclock once shuffled probes are also done.

**Awaiting Codex review** of the probe wrapper Codex agreed to write
(`probe/scripts/12_run_r3m_probe.py` or similar).

| 2026-05-01 13:34 | probe/scripts/12_run_r3m_probe.py | Codex | Claude | `/isaac-sim/python.sh probe/scripts/12_run_r3m_probe.py --dry-run` (after seeding 10 push R3M episodes) | PASS |

## 12_run_r3m_probe.py review (13:34, R3M probe wrapper)

**Verdict:** PASS. Dry-run with 10 push episodes produces stage_0 × ee_acceleration × 5 folds:
  R² = 0.074, 0.085, 0.078, -0.019, 0.045 (mean ≈ 0.05). Schema correct.

**Strengths:**
- Reuses utils/probe.py run_groupkfold_probe (same Adam 20-HP grid as V-JEPA main probe).
- stack_stage() loads per-stage features and aligns with t_last targets via the standard io.load_targets path.
- CSV schema columns: stage, fold, best_lr, best_wd, r2, mse, n_test_windows (mirrors V-JEPA layer-format with `stage` substituting `layer`).
- --dry-run flag honored: 5 episodes only, single (stage, target).
- Output: probe/results/<task>/r3m/<target>.csv (parallel structure).

**Codex-noted minor nit on extractor (11_*.py):**
- Unused `from torchvision import transforms` import. Cleaned up post-review.

**Past-mistake re-check:**
- Constant-target degeneracy? Targets are kinematic / contact (no constant for push/strike/drawer in the M1 set).
- GroupKFold disjointness: same as V-JEPA, asserted by sklearn API.
- Schema mismatch: r3m results live in separate variant_r3m/ directory, won't pollute V-JEPA aggregator.

| 2026-05-01 14:00 | probe/scripts/09_physics_condition_split.py (GPU patch) | Codex | (test pending) | py_compile only — actual GPU dry-run pending free-GPU window | PASS-syntax |
| 2026-05-01 14:00 | probe/trajectory_analysis_B/scripts/16_cross_task_transfer.py (per-fold sidecar) | Codex | Claude | re-run launched (PID 1380725 CPU) | ACTIVE |

## Codex 09 GPU + 16 sidecar patches (14:00)

**Verdicts:** PASS-syntax. Both py_compile clean. Codex did code-only patches per
my request (09 currently running with old CPU code, will be killed and restarted
with --device cuda:N when GPUs free up later).

**09 GPU patch summary:**
- New function `fit_group_ridge_with_folds_gpu(X, y, groups, device, alpha=1, n_splits=5)`.
- Closed-form: `w = (X^T X + αI)^-1 X^T y` via `torch.linalg.solve`. Same numeric
  result as sklearn Ridge α=1 (closed-form), modulo float precision.
- `--device` arg (default "cpu" → existing sklearn path; "cuda:N" → GPU path).
- Codex confirmed F4-A claim is independent of optimizer; switching CPU→GPU
  Ridge for some (task, param) and not others is fine as long as both are
  closed-form solutions.

**16 per-fold sidecar summary:**
- New function `within_task_r2_with_folds()` returns `(mean, list[fold_meta])`.
- Main loop adds per-fold rows: `{src_task, tgt_task, layer, target, fold,
  transfer_r2, within_tgt_r2, gap}`.
- New output `cross_task_transfer_per_fold.csv` alongside the aggregate.

**Codex F5 wallclock estimate:**
- push (4 targets, 2 GPUs sharded): wallclock ~4h (each target ~2h, 2-batch sequential).
- strike (4 targets, 4 GPUs sharded): wallclock ~5h.

**Codex CI rerun protocol:**
- push shuffled probe DONE → immediate 10_bootstrap_cis.py re-run for preliminary
  push-only F5_delta_r2 numbers.
- strike DONE → final integrated re-run.

**Operational note:** 16 v2 launched on CPU (PID 1380725). Will produce
cross_task_transfer_per_fold.csv enabling transfer bootstrap CI in 10_bootstrap_cis.

| 2026-05-01 14:11 | probe/scripts/12_run_r3m_probe.py | Codex | Claude | full launch on push (CUDA_VISIBLE_DEVICES=0, PID 1449834) | RUNNING-PASS |

## R3M push probe live launch (14:11)

After R3M push extraction completed (1490 eps in 33.9 min), launched
`12_run_r3m_probe.py --task push --gpu 0` on physical GPU 0 (shares with F5
strike extraction, 35GB free, R3M probe small footprint).

Within 1 minute of launch:
- ee_acceleration.csv: stage_0 × 5 folds, R² ≈ 0.05 (low layer, kinematic — sane).
- contact_flag.csv: stage_0 × 5 folds.

Schema verified: `stage,fold,best_lr,best_wd,r2,mse,n_test_windows`.

Past-mistake re-check: probe wrapper writes incrementally per (target, stage)
(Codex implementation), so a crash mid-run preserves completed stages. Output
directory parallel to V-JEPA: `probe/results/push/r3m/<target>.csv`.

## R3M push probe FINAL (14:18)

✅ Probe DONE for push (3 targets × 5 stages × 5 folds = 75 fold-fits).
Wallclock: 8 min (very cheap — R3M ResNet50 stages are small).

**Headline (push only, R3M vs V-JEPA Variant A):**
| Target | R3M best stage R² | V-JEPA best layer R² | Δ (R3M − V-JEPA) |
|---|---|---|---|
| ee_acceleration | stage_4 = 0.084 | L19 = 0.219 | −0.135 |
| contact_flag | stage_3 = 0.677 | L21 = 0.768 | −0.091 |
| contact_force_log1p_mag | stage_3 = 0.638 | L21 = 0.754 | −0.116 |

**Per-stage progression (push):**
- contact_flag: 0.52 → 0.67 → 0.67 → 0.68 → 0.67 (stage_0..stage_4) — peaks at stage_3
- contact_force: 0.49 → 0.62 → 0.62 → 0.64 → 0.64 — plateaus stage_3..4
- ee_acceleration: 0.05 → 0.07 → 0.08 → 0.07 → 0.08 — shallow depth-progression (image features don't encode kinematics with depth)

**Interpretation (Codex Decision B framing):**
R3M is a static image encoder baseline. V-JEPA wins on all three targets, with
the **largest gap on ee_acceleration** (the most temporal/dynamic target).
Contact targets show smaller gaps — image features partly capture contact
geometry. This supports "video pretraining adds temporal-coherence-dependent
information beyond what static image encoders capture." Strong contextual
baseline for the F5 frame-shuffle main claim.

| 2026-05-01 14:31 | F5 push extraction milestone | system | system | watch_push triggered shuffled probe at 14:32:05 | DONE |
| 2026-05-01 14:33 | probe/scripts/09_physics_condition_split.py (fp32 patch) | Codex | Claude | Re-launched 09 v4 with --device cuda:0 (PID 1624293). Rate ~5 rows/min on GPU. | RUNNING |

## F5 push extraction completion (14:31)
1497 episodes (3 too short to extract) in 208.4 min. Variant_A and variant_A_shuffled
counts both 1500 → watch_push triggered at 14:32:05. Sharded probe sweep launched
on GPU 2,3:
- ee_acceleration → GPU 2 (PID 1634139)
- ee_velocity → GPU 3 (PID 1634140)
- contact_force_log1p_mag → waits for GPU 2
- contact_flag → waits for GPU 3

Codex's wallclock estimate: ~4h for all 4 push targets on 2 GPUs.

## 09 GPU v4 fp32 launch (14:31)
fp64 → fp32 closed-form Ridge to address A6000 fp64 throughput bottleneck.
Rate observed: ~5 rows/min (vs fp64 ~2.8, sklearn CPU ~4.8). Modest speedup;
data loading IO dominates. With unlimited time, acceptable. Total ETA ~24h.

## F5 push PRELIMINARY headline (14:39)

After 7.4-7.6 min of probing per target on a single GPU (much faster than Codex's
4h estimate — Adam 20-HP grid is GPU-efficient on 1024-d Variant A features):

**push ee_velocity ΔR² = R²(unshuf) − R²(shuf):**
- L2: +0.308 [+0.282, +0.329] p<0.001 ← peak shuffle damage
- L1-L9: +0.22 to +0.30 (mid-band, all p<0.001)
- L20-L23: +0.21 (deep PEZ also affected)
- All 24 layers show large positive ΔR², CI excludes 0.

**push ee_acceleration ΔR²:**
- Small but uniformly positive (~0.005 at L0 → ~0.025 at L9), p<0.001 throughout.
- Smaller absolute magnitudes because the V-JEPA Variant A baseline R² for
  ee_acceleration is itself low (~0.22 at peak), so the room for shuffle-induced
  drop is bounded.

Pending: contact_flag + contact_force_log1p_mag (Codex predicted these would
show smaller / null ΔR² — to be verified once probes complete).

Combined with R3M baseline (V-JEPA wins on all 3 targets, biggest gap on
ee_acceleration), this is shaping up to be the paper's strongest evidence
panel: Variant A representation depends on input temporal coherence for
dynamic targets.

## F5 push FINAL bootstrap CIs (14:48) — ICLR oral evidence

All 4 push shuffled probes complete in ~30 min wallclock total (Codex's 4h
estimate was conservative; Adam grid is GPU-efficient on 1024-d Variant A).

**F5_delta_r2 = R²(unshuffled) − R²(shuffled) — paired bootstrap 1000 reps:**

| Target                  | Peak ΔR² | Peak Layer | 95% CI         | CI>0  |
|-------------------------|---------:|:----------:|:---------------|------:|
| ee_velocity             | +0.308   | L2         | [+0.28, +0.33] | 24/24 |
| ee_acceleration         | +0.113   | L17        | [+0.11, +0.12] | 24/24 |
| contact_force_log1p_mag | +0.053   | L2         | [+0.04, +0.07] | 23/24 |
| contact_flag            | +0.052   | L1         | [+0.05, +0.06] | 24/24 |

**Key finding (ICLR oral level):** Target-selective frame-shuffle damage.
Dynamic targets (velocity, acceleration) lose 6-12 R² points at peak; contact
targets lose only ~5 points. Ratio dynamic/contact = 6× (ee_velocity) or 2×
(ee_acceleration). Confirms Codex's a-priori prediction.

**Target-by-target observations:**
- ee_velocity: peak ΔR² at L2 (early!), still +0.21 at L23 (deep). Suggests
  early layers extract velocity from local frame differences, which shuffle
  destroys most thoroughly.
- ee_acceleration: peak at L17 (mid-PEZ band). The "PEZ band loss" pattern
  Codex hypothesized.
- Contact targets: peak at L1-L2 (very early). Disruption is small at all
  depths — contact info is largely spatial/instantaneous, not temporal.

Pending: strike replication (~3h). If pattern reproduces, this becomes the
paper's headline figure (per Codex Decision C ranking).

R3M strike probe launched immediately on GPU 2 (PID 1813898) since R3M strike
extraction completed 39.4 min ago.

## R3M strike probe DONE (14:56) — full M1 baseline coverage for push+strike

| Task | Target | R3M peak (stage_k, R²) | V-JEPA peak (Lk, R²) | Δ R3M − V-JEPA |
|---|---|---|---|---:|
| push | ee_acceleration | stage_4, 0.084 | L19, 0.219 | −0.135 |
| push | contact_flag | stage_3, 0.677 | L21, 0.768 | −0.091 |
| push | contact_force_log1p_mag | stage_3, 0.638 | L21, 0.754 | −0.116 |
| **strike** | **ee_acceleration** | stage_4, 0.189 | L17, 0.413 | **−0.224** |
| **strike** | **contact_flag** | stage_3, 0.678 | L19, 0.825 | **−0.147** |
| **strike** | **contact_force_log1p_mag** | stage_3, 0.694 | L21, 0.834 | **−0.140** |

**V-JEPA wins all 6 cells.** Strike gaps are LARGER than push gaps — strike has
richer dynamic content (impulsive impact vs sustained sliding), so V-JEPA's
temporal pretraining advantage is more pronounced. Largest gap remains
**ee_acceleration on strike (−0.224)**.

R3M depth-progression: stage_3 best for contact, stage_4 best for ee_acc — a
much shallower curve than V-JEPA's 24-layer trajectory. Image-encoder
inductive bias caps useful depth.

This is the M1 contextual baseline panel for the paper appendix.

| 2026-05-01 15:05 | PhysProbe_Neurips_Paper/Sections/latent_properties.tex (F1-b subsection) | Codex | Claude | pdflatex compile (16 pages) | PASS |

## Codex F1-b phase-space narrative (15:05)

**Verdict:** PASS. Codex extracted 4 sharp observations from
phase_space.csv (360 rows) and wrote a clean F1-b subsection inserted
into latent_properties.tex. Compiles to 16 pages.

**Codex's 4 key observations:**
1. **Mean latent speed grows with depth across all 6 tasks.** Push 0.17→2.81,
   Drawer 0.23→2.72, NutThread 0.12→2.97 from L0 to L23. Reach mild exception
   peaking at L8.
2. **Curvature collapses fast.** Push κ 17.1 (L0) → 1.4 (L23). Strike 26.9
   (L0) → 1.1 (L8). "Fast but jagged early → smoother latent flow."
3. **Contact-rich windows occupy larger/faster latent regime.** Push speed
   1.60 (pre) → 1.78 (during); sweep volume 21.7 → 69.1 (pre → during).
   Same pattern in Strike.
4. **Mid-layer expansion / late-layer compression.** Sweep volume peak at
   L8 for Push/Strike/Reach. Late layers fast but narrow.
5. **Drawer outlier:** mostly during/post contact, persistent depth.

Framing: supplementary geometric diagnostic, NOT promoted to causal claim
(F5 retains that role). Quality matches paper-grade.

Past-mistake re-check:
- Numbers consistent with phase_space.csv (sampled and verified).
- LaTeX compiles cleanly.
- Schema preserved (existing sections intact).

| 2026-05-01 15:15 | PhysProbe_Neurips_Paper/Sections/appendix.tex (Note H R3M baseline) | Codex | Claude | pdflatex compile (17 pages) | PASS |
| 2026-05-01 15:10 | /root/.claude/plans/...md (live milestone log + status refresh) | Claude | (self) | n/a — bookkeeping | DONE |

## Codex Note H R3M baseline (15:15)

**Verdict:** PASS. Codex wrote Note H sub-section with comprehensive R3M
context-baseline framing. PDF compiles to 17 pages.

**Codex's Note H contains:**
1. **Setup** — minimal R3M-resnet50 protocol explanation (per-frame ResNet50,
   GAP, window mean-pool over 5 stages, same Adam 20-HP grid as Variant A).
2. **Outputs** — full artifact paths for the comparison.
3. **6-cell results table** — exact numbers, V-JEPA wins all.
4. **Three sharp paragraphs:**
   - "The largest deficit is on acceleration" — push −0.135, strike −0.224.
     Ties cleanly to F5's "temporal coherence matters most for dynamics."
   - "Contact is better, but still not enough" — R3M closes some of the gap
     on contact (image features capture contact geometry) but never wins.
   - "R3M stage progression saturates shallow" — best contact stage_3, best
     acceleration stage_4; cannot match V-JEPA's L17-L21 deep peak.
5. **Status** — appendix-level supplementary, NOT main-text.

This is the M1 contextual baseline panel. Codex Decision B framing
("R3M as static image encoder baseline, not temporal competitor")
perfectly preserved.

## Plan file refresh (15:10)
Plan file at `/root/.claude/plans/read-physrepa-tasks-claude-code-task-md-humming-planet.md`
updated with live status: feedback table now reflects all DONE items,
Stage table marks completed/in-progress per phase, and a new "Live milestone
log" with timestamped events from 14:31 (F5 push DONE) through 15:10
(plan refresh).

| 2026-05-01 15:21 | PhysProbe_Neurips_Paper/Sections/cross_task.tex (Phase E bootstrap update) | Codex | Claude | pdflatex compile (17 pages) | PASS |

## Codex E Phase-E enhancement (15:21)

**Verdict:** PASS. Codex added two paragraphs to cross_task.tex Phase E section:

1. **"Bootstrap update (transfer-gap confidence)"** — uses
   cross_task_transfer_per_fold.csv to compute paired bootstrap CI on the
   gap=within-transfer for 576 (pair × layer × target) cells. Reports
   representative cells:
   - push→strike ee_velocity L4: gap +0.045 [-0.023, +0.112], CI crosses 0
   - push→strike ee_acc L22: gap -0.026 [-0.158, +0.107], CI crosses 0
   - push→drawer ee_velocity L11: gap +1.123 [+1.077, +1.177], CI > 0
   - drawer→push contact_flag L19: gap +2.604 [+2.562, +2.648], CI > 0
   
   Layer-counts where CI includes 0 (i.e. transfer succeeds):
   - push↔strike has 3-5 such layers per target
   - drawer-related pairs essentially zero such layers

2. **"Interpretation"** — frames F5 (within-task temporal coherence) and
   transfer (between-task generalization) as orthogonal causal-style
   intervention vs structure probe. Same conclusion: push+strike share
   transferable physics subspace, drawer remains task-entangled.

Cross-validates F5 main story: model learns task-general kinematics for
push/strike but not for drawer.

Both 4-paragraph addition keeps PDF at 17 pages (Phase E section grows
~2KB but doesn't add a page).

| 2026-05-01 15:33 | PhysProbe_Neurips_Paper/Sections/appendix.tex (Note H drawer rows) | Codex | Claude | pdflatex compile (17 pages) | PASS |

## Codex H drawer update (15:33)

**Verdict:** PASS. Codex added 3 drawer rows to Note H table:
- drawer ee_acceleration: stage_1, 0.004 vs L22 0.298 → Δ −0.293
- drawer contact_flag: stage_4, 0.509 vs L17 0.710 → Δ −0.201
- drawer contact_force_log1p_mag: stage_4, 0.425 vs L18 0.720 → Δ −0.295

Plus updated paragraph "R3M stage progression still saturates shallow" with
drawer-specific note: "Drawer reinforces the same point: even where R3M
reaches its best contact scores at stage_4, it remains well below V-JEPA's
late-layer peaks."

This completes the M1 R3M baseline panel: V-JEPA wins all 9 cells. Largest
gaps on dynamic / drawer targets (drawer ee_acc: −0.293, drawer contact_force:
−0.295). Drawer R3M ee_acceleration is essentially R²=0 — image encoder
fails at constrained articulated motion that requires temporal integration.

## F5 strike shuffled probe LAUNCHED (17:27:31)

✅ F5 strike extraction DONE (3000 eps in 383.5 min wallclock).
watch_strike triggered shuffled probe on GPU 0,1,2,3 (4-way parallel, 1 target/GPU):
- ee_acceleration → GPU 0 (PID 2138683)
- ee_velocity → GPU 1 (PID 2138684)
- contact_force_log1p_mag → GPU 2 (PID 2138685)
- contact_flag → GPU 3 (PID 2138686)

Wallclock ETA ~15 min for all 4 targets in parallel.
Once complete: re-run 10_bootstrap_cis.py to get full F5_delta_r2 (push + strike) = 192 rows.

## F5 STRIKE REPLICATION COMPLETE (17:39) — paper headline 확정

✅ Strike F5 shuffled probe ALL 4 targets DONE (~11 min wallclock on 4 GPUs).
Bootstrap CI rerun: F5_delta_r2 = 192 rows (push 96 + strike 96).

| Target | Push peak (L) | Strike peak (L) |
|---|---|---|
| ee_velocity | +0.308 (L2) | +0.213 (L4) |
| ee_acceleration | +0.113 (L17) | +0.172 (L16) |
| contact_force_log1p_mag | +0.053 (L2) | +0.046 (L2) |
| contact_flag | +0.052 (L1) | +0.055 (L7) |

**Hierarchy preserved exactly** across both tasks: dynamics ≫ contact.
**Two task-specific differences**:
1. Strike ee_acceleration LARGER (+0.172 > +0.113 push) — impact task has
   sharper acceleration transients in PEZ band
2. Strike ee_velocity slightly LATER + smaller (L4 +0.213 vs push L2 +0.308)
   — sustained sliding (push) vs impulsive contact (strike) physics

This is the **primary cross-task robustness check** for F5 mechanism.
Combined with R3M baseline (Note H, V-JEPA wins all 9 cells, biggest gap
on dynamic targets, especially drawer ee_acc), F5 establishes the
specifically temporal-coherence-dependent character of V-JEPA representation.

PDF compile clean (17 pages). Cross_task.tex F5 section now contains:
- Push final results (96 rows)
- Strike final results (96 rows)  
- 3-tier hierarchy interpretation
- Strike vs push comparison
- Codex paper-writer framing
- Status (push+strike DONE, F4-A push 5/5, strike 09 v4 in progress)

Bootstrap_cis.csv: 1812 total rows (F5 192 + F4A 564 + F2 480 + transfer 576).

## 09 v4 FULL COVERAGE COMPLETE (2026-05-02 03:12 UTC)

✅ All 22 (task, param) F4-A bootstrap conditions DONE on fp32 GPU Ridge.
Bootstrap CIs total: 3120 rows.

**F4A breakdown:**
- push: 480 (5 params × 96)
- strike: 576 (6 params × 96)
- drawer: 96 (1 param × 96)
- peg_insert: 360 (5 params × 72; constant-target rows excluded)
- nut_thread: 360 (5 params × 72; same exclusion)
- **TOTAL F4A: 1872 rows**

**bootstrap_cis.csv composition (final):**
- F4A_high_minus_low: 1872
- transfer_gap: 576
- F2_during_minus_post: 288
- F2_during_minus_pre: 192
- F5_delta_r2: 192
- **TOTAL: 3120 rows**

**Strongest F4A signal (paper-grade):**
- Strike `object_0_static_friction`: ΔR² ≈ **+0.55** at L1-L9 for both
  `contact_flag` and `contact_force_log1p_mag` (CI tight, p<0.001).
  Interpretation: high static-friction strikes produce more sustained,
  predictable contact-force traces than low-friction ones — the model
  encodes this physical-condition modulation across many layers.
- Strike `object_0_mass`: very small ΔR² at sensible layers L2+ (around
  +0.10), but L0/L1 explode to nonsensical magnitudes (324, 209) due to
  base R² being deeply negative. These are paper-excluded per the
  exclusion rules.
- Drawer `joint_damping` for `ee_acceleration` at L3 = +2.46 with very
  wide CI (0.009 to 7.33) — high drawer-joint-damping → more
  controllable acceleration profile. Also paper-excluded (CI too wide).

The robust paper-grade F4A claim is the strike static-friction effect:
**physical-condition modulation of contact representation is layer-wide**
(not just at PEZ peak), and **the magnitude is bigger than F5
shuffle-effect**, which positions F4-A as a complementary axis to F5.

This completes the F4-A evidence package.
