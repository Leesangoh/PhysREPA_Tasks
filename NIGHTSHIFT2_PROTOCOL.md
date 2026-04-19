# Nightshift 2 Autonomous Protocol (2026-04-19)

User away ~12 hours. Execute feedback plan autonomously.
Priority #1: accuracy + publication-quality rigor. No mistakes.

## Execution Order (planned)

### Round 1: F5 Frame Shuffle (HIGH priority)

Goal: Test whether current positive PEZ curves are temporal vs static visual correlation.

**Step 1.1** Extract Push shuffled token cache
- Modify `extract_token_features.py` to support `--shuffle-frames` flag
- Shuffled frame order preserves visual content but destroys temporal causality
- Use fixed random seed (e.g., 42) for reproducibility
- Sanity check: verify shuffle is deterministic given seed + episode_id
- Output: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_shuffled/push/`

**Step 1.2** Probe Push shuffled cache
- Targets: `ee_direction_3d`, `ee_speed` (most positive targets to stress-test)
- Same config: resid_post + temporal_last_patch + trainable 20-HP + 5-fold GroupKFold + zscore
- Compare R² curve: original vs shuffled

**Step 1.3** Extract + probe Strike shuffled
- Same process for Strike (2895 episodes)
- Targets: `ee_direction_3d`, `object_direction_3d` (Strike's strongest positive)

**Rigor checks (MANDATORY)**:
- Verify shuffle changes frame order (not just relabeling)
- Verify seed reproducibility across calls
- Compare L0 values — if L0 R² stays high on shuffle → static visual correlate confirmed
- Compare peak R² and peak layer — if peak drops significantly → temporal causality confirmed

### Round 2: F3 CKA Cross-Task (HIGH priority)

Goal: Test whether "good" (PEZ) layers align across tasks.

**Step 2.1** Implement CKA
- Linear CKA formula: `||K_X K_Y||_F^2 / (||K_X K_X||_F ||K_Y K_Y||_F)` where K_X = X X^T
- Or HSIC-based: mean-centered
- Library: compute directly or use `cka_modules` from GitHub

**Step 2.2** Compute task×task×layer matrix
- Tasks: Push, Strike (both with existing caches), optionally Reach
- Sample balancing: subsample each task to min(N_push, N_strike, N_reach) = 600
- Per-episode aggregated feature (flattened 262144-d) per layer
- Compute CKA(task_i layer_l, task_j layer_l) for all layer l

**Step 2.3** Visualize
- 3D heatmap: task_i × task_j × layer → CKA score
- Line plot per task pair: CKA vs layer
- Test hypothesis: "PEZ layers (L11-L13 based on our best results) have higher cross-task CKA than shallow L0-L4 or late L23"

**Rigor checks**:
- Matrix diagonal CKA(task, task, layer) should be 1.0 (sanity)
- Symmetry: CKA(A,B) ≈ CKA(B,A)
- Bootstrap confidence intervals (resample episodes)

### Round 3: F4-A Split-by-Value (MEDIUM priority if time)

Goal: Does PEZ strength vary with physics parameter regime?

**Step 3.1** Split Push episodes by physics param tertiles
- mass_low / mass_med / mass_high
- obj_friction same
- surface_friction same

**Step 3.2** Probe each subset
- Target: `ee_direction_3d` (our strongest signal)
- Compare R² curves across bins

**Rigor checks**:
- Ensure bin sample sizes adequate (≥100 per bin for 5-fold CV)
- Report per-bin L0, peak, peak_layer

## Discussion Protocol

Codex ↔ Claude critical discussion rules:
1. Before starting each Round, Codex writes design doc (hypothesis, recipe, rigor checks)
2. Claude reviews + flags issues → Codex responds + iterate
3. After each Round, Codex writes verdict. Claude audits.
4. Any ambiguity → pause + write to NIGHTSHIFT2_LOG.md, no autonomous decision

## Writing Standards

All markdown docs must:
- Cite source data paths
- Report exact numbers (no approximations)
- Distinguish observation from interpretation
- Mark every claim as "evidence-based" vs "hypothesis"
- Include negative controls + sanity check results

## Autonomous Decision Scope

**Auto-approve** (no user confirmation):
- Run experiments per Protocol
- Delete own intermediate caches after results committed
- Use any GPU that's free
- Commit + push to origin/main
- Create new analysis scripts

**PAUSE and log** (don't auto-act):
- Storage >95% full → delete Reach cache first, then decide
- Repeated silent crash (>2 times same cause) → log + stop
- Results dramatically different from Phase 2d (e.g., all R²<0) → audit pipeline before commit

## Discussion Log

File: `NIGHTSHIFT2_LOG.md` (append only)
Format: `## [UTC time] [who: Claude/Codex] [topic] + content`

## Expected Output

By end of 12h:
- F5 verdict: frame_shuffle_verdict.md
- F3 verdict: cka_cross_task_verdict.md
- F4-A verdict: split_by_value_verdict.md (if time)
- NIGHTSHIFT2_LOG.md with all decisions logged

## Quality Gate

Before user wakes:
- [ ] All CSVs have shape consistent with ORIGINAL phase2d
- [ ] fake_mod5 negative control still negative (if re-run)
- [ ] Every verdict file references original data paths
- [ ] NIGHTSHIFT2_LOG has no "blocked and gave up" entries

## [NEURIPS MODE]

Goal escalation:

- not just execute ablations
- produce reviewer-resistant evidence for a paper-quality claim

### Mandatory additions for every round

1. **Critical self-review**
   - After every major result, Codex must answer:
     - "Would a reviewer call this generic depth accumulation rather than PEZ?"
     - "What alternative hypothesis explains this result?"
     - "What counter-experiment would falsify that alternative?"

2. **Counter-experiment planning**
   - Every verdict file must include:
     - main claim
     - strongest reviewer attack
     - next experiment that would defeat that attack

3. **Story arc awareness**
   - Each round must explicitly update:
     - what the paper's main claim would be
     - what the current evidence chain is
     - what link in the chain is still weak

4. **Statistical rigor**
   - Prefer confidence intervals or fold-level variability over single-point metrics
   - For stochastic ablations, use at least 3 seeds unless blocked by compute or storage
   - Compare original vs ablated curves with paired statistics where feasible

### Round 1 F5 upgrade

Frame shuffle is now required to include:

- at least 3 seeds: `42`, `123`, `2024`
- exact same probe recipe as original
- effect-size summary:
  - peak drop
  - `R^2` drop at PEZ layer
  - peak-layer shift
- paired test plan:
  - compare original vs shuffled fold scores per layer
  - or paired bootstrap/permutation over seeds x folds when available

### Reviewer-standard interpretation rule

If a shuffled curve remains close to the original:

- default interpretation is **static or framewise correlate**
- not "temporal PEZ"

If a positive curve survives only under one parameterization:

- default interpretation is **parameterization-sensitive**
- not yet robust enough for a broad claim
