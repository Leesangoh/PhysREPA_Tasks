# CLAUDE.md — physrepa_tasks

Guidance for Claude Code when working in `/home/solee/physrepa_tasks/`. See the parent `/home/solee/CLAUDE.md` for environment-wide constraints (Isaac Sim Python, torch pins, dataset paths).

## Repo purpose

Isaac Lab simulation environments + data-collection pipeline for the **PhysProbe** dataset (Step 0), and the layer-wise linear probing experiments that consume V-JEPA 2 / DINOv2 features extracted from those episodes. GitHub: `Leesangoh/PhysREPA_Tasks`.

A methodology reset happened on **2026-04-23**: the previous `probe_physprobe.py` family used episode-level aggregation, which collapsed every episode to a single (feature, target) pair and destroyed timestep-level signal. All affected code/results are quarantined under `archive_old_wrong_probe/`. The new spec lives in `claude_code_task.md` and has not yet been implemented. See `/root/.claude/projects/-home-solee/memory/project_physprobe_incident_2026-04-23.md` for the incident memo.

## Canonical dataset

**Use `/home/solee/data/data/isaac_physrepa_v2_recollected_2026-04-23/`** — full re-collection with the contact-GT fix (bodyfilters approach) applied. This supersedes the older `/home/solee/data/data/isaac_physrepa_v2/step0/` dataset (collected 2026-04-09, contact GT zero-filled). All probing / feature-extraction work should target the recollected path.

| Task | Episodes | Contact GT (verified) |
|------|----------|------------------------|
| push | 1,500 | `contact_force` max 132.8 N, nonzero 24.5% |
| strike | 3,000 | max 53.5 N, nonzero 2.9% |
| peg_insert | 2,500 | max 5.3 N, nonzero 100% |
| nut_thread | 2,500 | max 11.7 N, nonzero 100% |
| drawer | 2,000 | max 429.3 N, nonzero 26.9% — **finger-handle contact populated** (richer than original step0, where only `drawer_joint_damping` varied) |
| reach | 600 | no contact channels (negative control, expected) |

Total: 12,100 episodes — same counts as the dataset table in the parent CLAUDE.md, but with valid contact GT. Format unchanged (LeRobot V2: parquet + MP4 + meta, dual 384×384 cameras).

The parent `/home/solee/CLAUDE.md` "PhysProbe Dataset (Step 0)" section still points at the old path and lists "Factory tasks: contact forces are zeros" as a known limitation — that limitation is now obsolete for the recollected dataset.

## Top-level files

| File | Role |
|------|------|
| `README.md` | Methodology-reset note. Points at the audit trail and the new spec. |
| `claude_code_task.md` | **Authoritative spec** for the new probing pipeline. 16-frame sliding windows; three feature-pooling variants (A 1024-d spatiotemporal mean, B 8192-d spatial mean, C 262144-d fallback); 5-fold GroupKFold; 20-config Adam sweep with 100 epochs; 11 stop conditions; integrity checks 12a–12d. Tiered execution: Priority-1 Push/Strike/Reach → decision rule → Priority-2 PegInsert/NutThread/Drawer. **No episode aggregation.** |
| `__init__.py` | Re-exports the `mdp` shim. |
| `.gitignore` | Excludes `PhysProbe_Neurips_Paper/`, `artifacts/action_control_features/`, old logs, figure variants. |

## Directory map

### `archive_data_collection/` — **ACTIVE (misnamed)**

Production data-collection code despite the `archive_` prefix. Treat as live.

- `collect_sample_data.py` (142 KB) — main entry point; sequential + parallel `collect_task_parallel`.
- `verify_dataset.py`, `verify_env.py`, `verify_factory*.py` — 5-level dataset verification (L0 randomization → L4 correlation).
- `eval_rl_policy.py`, `rollout_rl_policy.py` — RL rollout utilities.
- `extract_shuffled_features.py` — shuffled-feature extraction for control experiments.
- `test_all_envs.py`, `test_lift_env.py` — env smoke tests.
- `envs/` — 10 Isaac Lab task configs: `push_env_cfg.py`, `strike_env_cfg.py`, `reach_env_cfg.py`, `pick_place_env_cfg.py`, `peg_insert_env_cfg.py`, `nut_thread_env_cfg.py`, `drawer_env_cfg.py`, `lift_env_cfg.py`, `factory_camera_env.py`.
- `mdp/` — `observations.py` (physics GT), `events.py`, `sync_marker.py`.
- `policies/` — scripted policies for all tasks (incl. `Step0PushPolicy`, `Step0StrikePolicy`).
- `rl_envs/` — RL task registrations (`push_rl_cfg.py`, `strike_rl_cfg.py`, `pick_place_rl_cfg.py`, `stack_rl_cfg.py`) + `agents/`.
- `utils/` — `rl_games_policy.py` (LSTM policy wrapper for Factory tasks).
- `analysis/` — 8 scripts including 3 versioned `probe_sweep_v*.py` variants (methodology iteration debt), trajectory analysis, layer-curvature analysis.
- `results/`, `results_v2/`, `results_v3/`, `archive_v1/` — historical analysis runs. Three `results*` directories = stale.
- `STATUS.md` (2026-03-19) — lift/pick/push/stack verified with IK-relative control, 7D actions, binary gripper.

### `archive_old_wrong_probe/` — Stale, kept for audit

Pre-2026-04-23 probing code with the episode-aggregation bug. Do not resurrect unless explicitly recovering provenance.

- `scripts/` — 15 files incl. the buggy `probe_physprobe.py`, `probe_physprobe_attentive.py`, `compare_surrogate_native_force.py`, action/event probes, CKA, OOD diagnostics.
- `artifacts_results/` (e.g. `cka_push_strike/`), `artifacts_figures/`, `artifacts_logs/` (`force_proxy_multiseed/`, `native_force_multiseed/`) — all invalidated.

### `artifacts/` — ACTIVE outputs / audit summaries

- `results/contact_inference_summary.csv` and `contact_inference_summary_sample50.csv` (generated 2026-04-20) — proxy evaluation across 5 tasks. `contact_gt_max_nonzero_fraction` is 0.0 everywhere, **but this is stale**: the CSV was computed against the main dataset under `/home/solee/data/data/isaac_physrepa_v2/step0/` (collected 2026-04-09, before the contact-GT fix). The contact columns exist in those parquets but are zero-filled.
- `figures/` — ~80 PNGs incl. `gt_distribution_*.png` validation plots for reach / nut_thread / peg_insert / push.
- `notebooks/contact_inference_explore.py`.
- `native_force_recollection_pilot{,_bodyfilters,_hand,_objectsensor}/` (2026-04-22) — pilot recollections that **do** populate contact GT. Strike, 1 episode each. Status:
  - `pilot_bodyfilters` ✅ — `physics_gt.contact_force` max ≈ 39.8 N, nonzero fraction ≈ 4%. **This is the working fix.**
  - `pilot_objectsensor` — only `contact_object_surface_force` populated (max ≈ 51.6 N, ≈ 33%); finger-object channel still zero.
  - `pilot`, `pilot_hand` — contact channels still zero (failed variants).
- `action_control_features/` — gitignored.

**Implication:** the contact-GT issue *is* solved (bodyfilters approach), but the main step0 dataset has **not yet been re-collected** with the fix. Probing/proxy work that depends on contact GT must either (a) wait for re-collection, or (b) run on the bodyfilters pilot data.

### `PhysProbe_Neurips_Paper/` — Paper draft (nested git, gitignored)

`main.tex`, compiled `main.pdf` (2026-04-23, 1.04 MB), `Sections/`, `Figures/`, `references.bib`, `neurips_2026.sty`, `math_commands.tex`, `tools/make_main_figures.py`. `token.txt` present (likely HF auth — do not commit).

### `docs/` — Historical, superseded

`docs/archive/` contains: `NIGHTSHIFT_LOG.md`, `NIGHTSHIFT2_PROTOCOL.md`, `PEZ_TO_PHYSPROBE_PLAN.md`, `CROSS_MODEL_PLAN.md`, `SCALE_PLAN.md`, `F3_design.md`, `F5_design.md`, `REVISION_PLAN.md`, `260413_feedback.md`, `ORAL_STRATEGY.md`. All superseded by `claude_code_task.md`.

### `envs/`, `mdp/`, `policies/`, `rl_envs/`, `utils/` — Import shims

Each is a thin `__init__.py` re-exporting from the matching `archive_data_collection/` subdir, so `from physrepa_tasks.envs import ...` keeps working after the reorg. Don't add real code here — put it in `archive_data_collection/` and extend the shim.

## Working rules

- **Never reintroduce episode-level aggregation** in any probing pipeline. Targets and features must stay timestep- (or window-) aligned per the spec.
- **Don't add new code under `archive_old_wrong_probe/`** — it's quarantined for audit only.
- **New collection / env code goes under `archive_data_collection/`** (misnamed but live), and a re-export added to the matching root shim if it should be importable as `physrepa_tasks.<pkg>`.
- **Probing implementation is still TODO.** Before writing it, re-read `claude_code_task.md` end-to-end — the integrity checks and HP grid are exact.
- **Contact GT status**: resolved. Use `/home/solee/data/data/isaac_physrepa_v2_recollected_2026-04-23/` (see "Canonical dataset" above). The 2026-04-20 audit CSV under `artifacts/results/` and any analysis pointing at `isaac_physrepa_v2/step0/` predate the fix and should not be trusted for contact-dependent claims.
- **Three `results*` directories under `archive_data_collection/`** are stale; don't write new outputs there. Use `artifacts/results/`.
- **Run scripts via `/isaac-sim/python.sh`** (parent CLAUDE.md constraint). Lint with the Isaac-GR00T toolchain only when touching shared code.

## Quick command reference

```bash
# Step0 collection (scripted policy)
/isaac-sim/python.sh archive_data_collection/collect_sample_data.py \
  --task push --num_episodes 1500 --num_envs 16 --step0 \
  --output_dir <OUT> --headless

# Factory parallel collection (RL policy)
/isaac-sim/python.sh archive_data_collection/collect_sample_data.py \
  --task peg_insert --num_episodes 2500 --num_envs 200 \
  --filter_success --rl_checkpoint <CKPT> --output_dir <OUT> --headless

# Verification
/isaac-sim/python.sh archive_data_collection/verify_dataset.py \
  --task push --data_dir <DIR> --level all
```

Parallel collection caps (from parent CLAUDE.md): Push/Strike `num_envs<=16`, Drawer `<=32`, Factory `200` ok. `env_spacing=5.0`, `skip_frames=5` after reset.
