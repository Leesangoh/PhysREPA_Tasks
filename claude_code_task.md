You are implementing a layer-wise linear probing experiment from scratch for the
PhysProbe project. The goal is to measure where in V-JEPA 2 Large's 24
residual layers observable kinematic state (position, velocity, speed,
direction, acceleration, acceleration magnitude) becomes maximally linearly
accessible — i.e., the Physics Emergence Zone (PEZ) for manipulation tasks.

CRITICAL CONTEXT — READ FIRST
The previous implementation contained a methodological error: it averaged
features over windows AND targets over an entire episode, collapsing every
episode to a single (feature, target) pair. This destroyed all timestep-level
signal. Do NOT replicate that. The new pipeline must be window-aligned at the
sample level: ONE (feature, target) sample per window. ZERO episode-level
aggregation anywhere. The dataset has been regenerated and is published at
https://huggingface.co/datasets/leesangoh/dynamics-probing — it preserves
per-timestep ground-truth kinematic state per episode.

This experiment runs THREE FEATURE POOLING VARIANTS in a TIERED execution
order. Two of them (A and B) are cached from a SINGLE V-JEPA 2 forward pass.
The third (C) is a fallback that requires re-running V-JEPA 2 with
layer-streaming. The default execution path is "A only"; B and C are only
unlocked by the decision rule below.

============================================================
SPEC (DO NOT DEVIATE)
============================================================

0. TOP-LEVEL EXECUTION ORDER

   Step 1) V-JEPA forward over ALL SIX tasks (Push, Strike, Reach, Drawer,
           PegInsert, NutThread) → cache Variant A (1024-d) AND Variant B
           (8192-d) features per (window, layer). DO NOT cache Variant C.
   Step 2) Run probe sweep with Variant A on all six tasks. Start with Push
           (sanity reference); the rest can run in any order.
   Step 3) Apply the decision rule (section 16) to the Variant A results:
             HEALTHY  → ship A as the main result. B/C optional.
             MARGINAL → run probe sweep with Variant B on the same tasks.
             FAILED   → re-run integrity checks. If still bad, escalate to
                        Variant C with layer-streaming feature extraction.

   Each variant uses the SAME windows, SAME targets, SAME folds, SAME probe
   protocol, SAME metrics. The ONLY difference is the pooling operation
   applied to the per-window per-layer activation tensor.

1. BACKBONE
   - Model: V-JEPA 2 Large (HuggingFace: facebook/vjepa2-vit-l).
   - Frozen, eval mode, no gradient. torch.inference_mode() for all forwards.
   - Layers: probe ALL 24 residual stream layers (L0 = patch_embed output;
     Lk for k≥1 = post-block residual addition output of block k-1, pre-final-LN).
   - d_model: 1024.
   - Input resolution: 256 × 256 (Isaac Sim native rendering, NOT 224×224).
   - Patch size: 16 × 16 → spatial token grid 16 × 16 = 256 tokens per
     temporal slice.
   - Tubelet stride: 2 (V-JEPA 2 default; 16 input frames → 8 temporal tokens).
   - Token grid per window per layer: 8 temporal × 256 spatial = 2,048 tokens.
   - Hook point: post-block residual addition (= "residual stream"), pre-final-LN.
     Use forward hooks. Dump shapes for the first hook output and confirm
     [B, 2048, 1024]. Document your hook choice in REPORT.md.

2. WINDOW CONSTRUCTION
   - Window size: 16 input frames.
   - Stride: 1 (max sliding window).
   - For an episode of length T, generate windows for t = 15, 16, ..., T-1.
     Each window covers frames [t-15, t] (16 frames). One window per t.
     This yields (T - 15) windows per episode.

3. FEATURE POOLING — THREE VARIANTS

   For every window, every layer, the V-JEPA hook gives an activation tensor
   shaped [2048, 1024] which equals [T_tok=8, N=256, D=1024].

   --- Variant A (spatiotemporal full mean) ---
     # act shape: [T_tok=8, N=256, D=1024] (or flat [2048, 1024])
     feat_A = act.reshape(8, 256, 1024).mean(dim=(0, 1))   # [1024]
     # Equivalently:
     # feat_A = act.reshape(2048, 1024).mean(dim=0)        # [1024]
     # → Final feature dim per (window, layer): 1,024
     # → This is the original PEZ paper recipe (Appendix B "spatiotemporally
     #   pooled activations") and matches our reproduction study.

   --- Variant B (spatial mean only, temporal preserved) ---
     # act shape: [T_tok=8, N=256, D=1024]
     feat_B = act.reshape(8, 256, 1024).mean(dim=1)        # [8, 1024]
     feat_B = feat_B.flatten()                              # [8192]
     # → Final feature dim per (window, layer): 8,192 (= 8 × 1024)
     # → Preserves temporal axis, collapses 16×16 spatial tokens.
     # → Justification: manipulation may rely on impact-timing dynamics that
     #   spatiotemporal mean would erase.

   --- Variant C (temporal mean only, spatial preserved) — FALLBACK ---
     # act shape: [T_tok=8, N=256, D=1024]
     feat_C = act.reshape(8, 256, 1024).mean(dim=0)        # [256, 1024]
     feat_C = feat_C.flatten()                              # [262144]
     # → Final feature dim per (window, layer): 262,144 (= 256 × 1024)
     # → Preserves spatial token grid, collapses temporal axis.
     # → DO NOT cache C at extraction time. Only re-extract on demand
     #   with layer-streaming if both A and B fail.

   --- POOLING IDENTITY (must hold within float16 tolerance) ---
     A == B.reshape(8, 1024).mean(dim=0)         # spatial-then-temporal vs full
     A == C.reshape(256, 1024).mean(dim=0)       # temporal-then-spatial vs full
   This identity is asserted in integrity check 12a.

   --- Caching rules ---
     • Step 1 forward (all six tasks): from each activation tensor compute
       BOTH A and B in-place, write to disk, then DROP the activation tensor.
     • Cache layout (per task, per variant):
         cache/<task>/variant_A/episode_<id>.npz
           contains: feats [N_win, 24, 1024]      (float16)
                     episode_id [N_win]
                     t_last [N_win]
         cache/<task>/variant_B/episode_<id>.npz
           contains: feats [N_win, 24, 8192]      (float16)
                     episode_id [N_win]
                     t_last [N_win]
     • Variant C (only if reached): re-run V-JEPA, layer-by-layer:
         cache/<task>/variant_C/layer_<L>.npz
           contains: feats [N_win_total_in_task, 262144]   (float16)
                     episode_id [N_win_total_in_task]
                     t_last [N_win_total_in_task]
       Process exactly one layer at a time; release before moving on.
     • Save manifest at cache/manifest.json with backbone hash, dataset
       commit, input_resolution=[256,256], window_size=16, stride=1,
       spatial_tokens_per_frame_group=256, temporal_tokens_per_window=8,
       tasks_extracted, variants_cached, feature_dtype, seed,
       extraction_start_time, extraction_end_time.

4. TARGETS (Tier A; per window; computed from frame t = window's last frame)

   Always (every task):
     ee_position       (3D, world frame, m)
     ee_velocity       (3D, world frame, m/s)
     ee_speed          (1D, ‖ee_velocity‖)
     ee_direction      (3D, unit vector, ee_velocity / ‖ee_velocity‖)
     ee_acceleration   (3D, world frame, m/s²)
     ee_accel_mag      (1D, ‖ee_acceleration‖)

   For tasks with object (Push, Strike, PegInsert, NutThread, Drawer):
     obj_position, obj_velocity, obj_speed, obj_direction,
     obj_acceleration, obj_accel_mag (same dims/definitions as EE)

   Reach has no object → EE targets only.

   TARGET ALIGNMENT (critical):
     Each window covers frames [t-15, t]. Target = ground-truth value AT
     FRAME t (the LAST frame of the window). NOT a mean within window.
     NOT first frame. Single timestep at last frame.

   DIRECTION TARGET MASKING:
     For ee_direction and obj_direction: if ‖velocity‖ < 1e-4 at frame t,
     mark as NaN. Mask these out of the direction probe (training, val, test).
     Keep them in speed/velocity probes (zero is a valid target there).

5. PROBE PROTOCOL (paper-exact, Appendix B)

   Probe form: f(h) = W·h + b (linear, single layer).
   Loss: MSE.
   Optimizer: Adam (torch.optim.Adam). NOT sklearn Ridge.
     • Exception: if Variant C is reached AND Adam wall time is infeasible,
       Sangoh may approve a sklearn Ridge alpha sweep
       (logspace(-3, 3, 20)). Document this deviation explicitly.
   Hyperparameter sweep: 20 configs = 5 LR × 4 WD.
     LR ∈ {1e-4, 3e-4, 1e-3, 3e-3, 5e-3}
     WD ∈ {0.01, 0.1, 0.4, 0.8}
   Epochs: 100 uniform across all targets.
   Batch size: 1024 (or full batch if N_train < 2048).
   Best HP selection: held-out validation slice within train fold (10% of
     train EPISODES, not 10% of windows — episode-level split). Pick the
     config with lowest validation MSE.
   Per (variant, layer, target): one independent probe trained 5 times
     (once per fold). All folds use identical HP sweep.

6. CROSS-VALIDATION

   sklearn.model_selection.GroupKFold(n_splits=5).
   Group key = episode_id. Two windows from the same episode MUST end up
   in the same fold (either both train or both test).
   Hard assertion at start of each fold:
     assert set(train_episode_ids).isdisjoint(set(test_episode_ids))
   Same assertion for the inner train/val split:
     assert set(inner_train_episode_ids).isdisjoint(set(inner_val_episode_ids))

7. NORMALIZATION (within-fold z-score; inner_train statistics only)

   For each fold:
     - Compute feat_mean, feat_std on inner_train features only.
     - Compute targ_mean, targ_std on inner_train targets only.
     - Normalize inner_train, inner_val, test using these statistics.
     - Train probe in normalized space.
     - At inference, predict in normalized space, then UNNORMALIZE before
       computing R² and MSE: pred_unscaled = pred * targ_std + targ_mean.
     - R² and MSE are reported in original units, not normalized space.
   Never use full-dataset statistics — would leak.

8. METRICS

   For each fold, on the test set (pooled across all test windows):
     y_pred_unscaled = unnormalize(model(X_test_normalized))
     y_test_unscaled = y[test_idx]   # original unscaled

     r2 = sklearn.metrics.r2_score(
              y_test_unscaled, y_pred_unscaled,
              multioutput='variance_weighted')
     mse = ((y_test_unscaled - y_pred_unscaled) ** 2).mean()

   For 1D targets (speed, accel_mag): standard scalar R² (multioutput moot).
   For ee_direction / obj_direction (3D unit vector): also report mean
     cosine similarity as a SECONDARY metric — pred_dir / ‖pred_dir‖ vs
     target unit vector.
   Report mean ± std across the 5 folds.

   POOLING POLICY: pool ALL test windows in a fold for R²/MSE computation
   (sklearn r2_score default behavior). Do NOT compute per-episode R² and
   then average.

9. TASKS — run all six in one sweep

   Push       (1,500 episodes)   ee + obj
   Strike     (3,000 episodes)   ee + obj (hammer)
   Reach      (600 episodes)     ee only (no object)
   Drawer     (2,000 episodes)   ee + obj (handle)
   PegInsert  (2,500 episodes)   ee + obj (peg). Note: contact_peg_socket
                                  force is reconstructed as residual
                                  (F_peg_total - F_finger_l - F_finger_r);
                                  noisier — not used in Tier-A but flag if
                                  encountered.
   NutThread  (2,500 episodes)   ee + obj (nut)

   Run order: start with Push (sanity / negative-control reference), then
   the rest in any order Claude Code finds convenient. No staged "Priority
   1 / Priority 2" gate — process all six.

10. OUTPUT STRUCTURE

    results/
      manifest.json              # backbone hash, dataset hash, seed, config snapshot
      <task>/
        variant_A/               # spatiotemporal mean, 1024-dim
          ee_position.csv        # columns: layer, fold, best_lr, best_wd,
                                 #          r2, mse, n_test_windows
          ee_velocity.csv
          ee_speed.csv
          ee_direction.csv       # extra column: cos_sim_mean
          ee_acceleration.csv
          ee_accel_mag.csv
          obj_*.csv              # if applicable
          _summary.csv           # layer × target →
                                 #   r2_mean, r2_std, mse_mean, mse_std
        variant_B/               # spatial mean only, 8192-dim
          ... (same structure as variant_A; only created if A is MARGINAL/FAILED)
        variant_C/               # temporal mean only, 262144-dim, FALLBACK
          ... (same structure; only created if A and B both fail)

11. REPRODUCIBILITY

    Seed everything: numpy=42, torch=42, torch.cuda=42,
    torch.backends.cudnn.deterministic=True, cudnn.benchmark=False,
    sklearn random_state=42. DataLoader workers fixed seed via worker_init_fn.

12. INTEGRITY CHECKS (run BEFORE main sweep — block sweep until all pass)

    a) Feature integrity (1 Push episode):
       - Run V-JEPA 2 on every window in that episode.
       - Assert variant A shape [N_windows, 24, 1024], no NaN/Inf.
       - Assert variant B shape [N_windows, 24, 8192], no NaN/Inf.
       - Compute per-layer mean/std; assert layer 0 stat ≠ layer 23 stat
         (mean diff > std/10).
       - POOLING IDENTITY:
           feat_A == feat_B.reshape(8, 1024).mean(dim=0)
         within float16 tolerance (~1e-3) or float32 tolerance (~1e-6).
       - If Variant C is later run: also assert
           feat_A == feat_C.reshape(256, 1024).mean(dim=0)
         in the same tolerance.

    b) Target integrity (1 Push episode):
       - velocity ≈ finite_diff(position) within tolerance (~1e-2 m/s @ 30Hz).
       - acceleration ≈ finite_diff(velocity) within similar tolerance.
       - speed = ‖velocity‖.
       - direction = velocity / speed where speed > 1e-4, NaN otherwise.
       - No NaN/Inf in any target except by design in direction.

    c) GroupKFold integrity:
       - Build the 5-fold split; for each fold assert
         set(train_episode_ids).isdisjoint(set(test_episode_ids)).
       - Same assertion for inner train/val split.

    d) Negative control (Push, layer 12, ee_velocity, variant A only):
       - Shuffle targets at the EPISODE level (each episode gets another
         random episode's full target time series; within-episode timestep
         order preserved).
       - Run full 5-fold probe with this shuffled target.
       - Assert mean R² < 0.05 across folds. If higher, halt and report —
         pipeline is leaking.

13. STOP CONDITIONS — halt and report immediately

    - NaN/Inf in features or targets.
    - Pooling identity fails (Variant A ≠ temporal-mean(B), or ≠ spatial-mean(C)).
    - Negative control R² ≥ 0.05.
    - Any GroupKFold fold has train/test episode overlap.
    - Best-layer R² < 0.05 across ALL Tier-A targets in variant A on Push
      (Tier A should reach R² > 0.3 at the best layer; otherwise pipeline
      is broken).
    - Adam training diverges (loss NaN) for ALL 20 HP configs on a single
      (variant, layer, target, fold) — pause and inspect.
    - Storage hits a pre-agreed quota (default 1 TB) — pause and ask before
      proceeding.

14. COMPUTE / STORAGE BUDGET

    Estimated wall time per task (single A100):
      Variant A (1024-dim):    feature ~6-8 h  +  probe sweep ~10-30 h
      Variant B (8192-dim):    probe sweep ~100-300 h (~4-12 days; only
                               run if escalated)
      Variant C (262144-dim):  probe sweep impractical with Adam at the
                               default budget; consider Ridge fallback.

    Estimated cache storage (float16, per task; assumes T_avg=200, stride=1):
      Variant A: Push ~13 GB, Strike ~26 GB, Reach ~5 GB,
                 Drawer ~17 GB, PegInsert ~22 GB, NutThread ~22 GB
      Variant B: Push ~104 GB, Strike ~208 GB, Reach ~42 GB,
                 Drawer ~139 GB, PegInsert ~173 GB, NutThread ~173 GB
      Variant C: Push ~3.33 TB, Strike ~6.66 TB, ... (use layer-streaming →
                 peak ~5.8 GB per layer per task)

    All-six-tasks A+B cache total ≈ 944 GB. Tell Sangoh BEFORE allocating
    > 1 TB. If Variant C is to be reached, use layer-streaming so peak
    on-disk footprint stays per-task per-layer.

15. DELIVERABLES

    Code layout (suggested):
      scripts/
        00_integrity_checks.py   # 12a–12d, must pass before sweep
        01_extract_features.py   # forward V-JEPA 2; cache A and B per task
        01b_extract_variant_C.py # ONLY if needed; layer-streaming
        02_build_targets.py      # parse dataset; build per-window targets
        03_run_probe.py          # 5-fold GroupKFold Adam sweep; write CSVs
        04_aggregate_results.py  # build _summary.csv per (task, variant)
      configs/
        push.yaml, strike.yaml, reach.yaml, peg_insert.yaml,
        nut_thread.yaml, drawer.yaml
      utils/
        models.py, data.py, metrics.py, io.py
    Results in results/<task>/<variant>/.
    Final REPORT.md in results/ with:
      - Per-task per-variant per-target layer-vs-R² table.
      - PEZ peak layer for each (task, variant, target).
      - Whether all six tasks show consistent PEZ structure.
      - If multiple variants ran: agreement of PEZ peak layer across A/B/C.
      - All deviations from this spec, with justification.
      - Anything that broke. Anything surprising. Be honest.

16. DECISION RULE FOR VARIANT ESCALATION (after Variant A sweep over all
    six tasks is done)

    HEALTHY:
      - Push ee_velocity best layer R² > 0.5 AND
      - Push ee_position best layer R² > 0.5 AND
      - Strike ee_velocity best layer R² > 0.4 AND
      - PEZ peak observed in mid-depth (layer 6–18 range)
      → Ship Variant A as the main result. Variant B/C optional appendix.

    MARGINAL:
      - 1–2 of the HEALTHY criteria miss, but R² > 0.2 on basic targets
        AND a PEZ peak is identifiable somewhere
      → Run Variant B probe sweep on the same tasks. Compare A vs B
        layer-wise R². Decide with Sangoh.

    FAILED:
      - All basic targets have R² < 0.2 OR
      - Layer 0 already saturates (R² > 0.9 from L0 onwards) — implausible
        and signals leakage
      → Re-run integrity checks (12a–12d). If they still pass, escalate
        to Variant B; if Variant B also fails, then Variant C with
        layer-streaming.

============================================================
EXECUTION DISCIPLINE
============================================================

- Implement step by step. After each step (integrity checks, feature
  extraction for one task, single (variant, layer, target) probe sweep),
  STOP and show me the output before moving on.
- Do NOT silently skip integrity checks.
- Do NOT silently change the spec. If something is ambiguous or impossible,
  pause and ask. Do not guess.
- Do NOT escalate to Variant B or C without my explicit greenlight after
  the decision rule (section 16) is applied.
- Prefer simple, auditable code over clever code. The previous pipeline
  failed silently due to over-cleverness. Readability > performance here.
- Assume nothing about array shapes — print and assert at every boundary.

START
1. Read the dataset README at:
     https://huggingface.co/datasets/leesangoh/dynamics-probing
   Confirm the per-episode timestep-level structure of the kinematic
   ground truth. Note any field that does not exist (e.g., if some task
   does not have obj_acceleration, document this and pause).
2. Run integrity checks 12a–12d on a single Push episode BEFORE writing
   any sweep code. (Variant A and Variant B both extracted; Variant C
   is NOT extracted at this stage — its identity check waits.)
3. After integrity checks pass, extract features (Variant A and B) for
   Push only, layer 12 only, on a small slice (~10 episodes). Train ONE
   (variant=A, layer=12, target=ee_velocity) probe end-to-end with the
   full 20-HP sweep. Show me the resulting R²/MSE per fold and best HP.
   THEN STOP.
4. After I confirm step 3 looks healthy, scale to Variant A full Push
   sweep across all 24 layers and all targets. Then run the remaining
   five tasks (Strike, Reach, Drawer, PegInsert, NutThread) — order
   flexible.
5. After Variant A sweep across all six tasks is done, apply the decision
   rule (section 16). Surface the decision to me and wait for greenlight
   before running Variant B (or, if needed, Variant C with layer-streaming).