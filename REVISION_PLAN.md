# Paper Revision Plan

You are helping execute a revision plan for the current paper submission. Your job is to improve the current paper conservatively and defensibly. Do not expand scope unless explicitly authorized.

========================
GLOBAL OBJECTIVE
========================
Focus on making the current paper more defensible for acceptance first.
Priority order:
1) Layer 1: acceptance hardening
2) Layer 2: spotlight-strengthening only after Layer 1 is completed and reviewed
3) Layer 3 is NOT part of the current paper. Treat Layer 3 only as a follow-up paper backlog. Do not start it unless explicitly approved by a human.

========================
HARD GUARDRAILS
========================
1) Do not start Layer 2.2 or any Layer 3 task before Layer 1 is fully completed and a human explicitly approves moving on.
2) After each layer, stop and produce a concise human-review checkpoint. Do not auto-continue.
3) Do only the tasks explicitly listed below. No scope creep, no "nice-to-have" side experiments, no extra benchmarks beyond the listed deliverables.
4) Use pre-registration style reporting:
   - Report effect sizes, confidence intervals, and negative results symmetrically.
   - Do not optimize wording toward positive conclusions.
   - For every task, if the expected effect is absent, report that directly and adjust the claim downward.
5) Maintain the current paper's restrained tone.
   - Avoid overclaiming language such as "prove", "causal grounding", "we show why X exists" unless the evidence truly supports it.
   - Do not rewrite the paper into a mechanism/causality paper.
6) Reproducibility requirements for every experiment:
   - Record seeds, hyperparameters, dataset split, environment version, feature extraction config, and any failure cases.
7) Task 1.1 is scoped ONLY to the current 2-task OOD control results (Push and Drawer). Do not generalize it to future tasks unless separately instructed later.

========================
CURRENT PAPER SCOPE
========================
The current paper is a 3-task core + 3-task peripheral paper.
Core tasks under the final probing recipe:
- Push
- Strike
- Reach

Peripheral / scope-expansion / OOD follow-up tasks:
- PegInsert
- NutThread
- Drawer

This distinction must be reflected consistently in abstract, introduction, contributions, and benchmark framing.

========================
LAYER 1: ACCEPTANCE HARDENING
========================

--------------------------------
Task 1.1: OOD control statistical rigor
--------------------------------
Objective:
Strengthen the OOD control claims in the current 2-task OOD action-chunk regression results.

Inputs:
- Existing OOD action-chunk regression results
- 3 seeds per representation per task
- Push split: 412 / 88 / 89 / 911
- Drawer split: 487 / 104 / 105 / 300
- Per-episode OOD R² predictions for each representation

Deliverables:
1) ood_pairwise_bootstrap.py
   Run paired bootstrap by episode with 1000 resamples for these comparisons:

   Push:
   - vjepa_fusion vs vjepa_last
   - vjepa_fusion vs vjepa_pez
   - vjepa_fusion vs videomae_best
   - videomae_best vs dino_mid

   Drawer:
   - same pair set

   Output for each pair:
   - mean gap
   - 95% CI

2) drawer_ood_diagnosis.py
   Diagnose why Drawer OOD R² may exceed IID R².
   Check:
   - per-episode R² distribution: central vs damping-extreme
   - action variance: central vs damping-extreme
   - action chunk predictability: central vs damping-extreme
   - episode length distribution
   - action autocorrelation

3) Updated Table 10 with CI-based reporting
4) One appendix paragraph describing the Drawer diagnosis

Acceptance / reporting rule:
- Replace vague claims with CI-based statements.
- Example style:
  "Fusion vs late gap = X with 95% CI [L, U]."
- If CI overlaps zero, report that the current evidence does not support a clear improvement.

--------------------------------
Task 1.2: Contribution framing tone-down
--------------------------------
Objective:
Make the benchmark contribution and actual analysis scope fully aligned across the paper.

Deliverables:
1) Abstract revision:
   Whenever the six-task benchmark is mentioned, explicitly distinguish:
   - three tasks fully analyzed under the final probing recipe (Push, Strike, Reach)
   - three tasks used for scope-expansion / OOD follow-ups (PegInsert, NutThread, Drawer)

2) Contribution bullet 1 revision:
   Keep "12,100 successful episodes" but clearly separate:
   - fully swept / main-result tasks
   - scope-expansion / follow-up tasks

3) Introduction revision:
   Introduce the Manipulation Physics Benchmark with the same honest scope distinction used in Section 3.4.

4) Ensure Section 3.4's scope honesty is mirrored in the abstract and introduction.

Acceptance / reporting rule:
- A reviewer reading only the abstract + contributions should not mistake this for a fully matched 6-task paper.

--------------------------------
Task 1.3: Strike random-init null
--------------------------------
Objective:
Extend the "learned property, not architecture only" claim from Push to Strike.

Inputs:
- Existing Strike token-patch extraction pipeline
- V-JEPA 2 Large architecture with model_seed=0 random initialization

Deliverables:
1) Extract random-init V-JEPA 2 Large Strike features using:
   - resid_post
   - temporal_last_patch
   - 16-frame window
   - stride 4

2) Run 3-seed probing (seeds 42, 123, 2024) for:
   - ee_direction_3d
   - object_direction_3d
   - ee_speed

3) Extend Table 5 with Strike rows:
   - pretrained vs random-init
   - peak R²
   - peak layer
   - last-layer R²
   - train R²

Acceptance / reporting rule:
- Report whether random-init shows lower validation peak and/or later peak layer.
- If Strike does NOT show a clear drop, report that as a task-dependent exception rather than forcing a general claim.

--------------------------------
Task 1.4: Huge multi-seed
--------------------------------
Objective:
Make the scale-law claim about non-monotonic peak depth defensible.

Inputs:
- V-JEPA 2 Huge feature extraction pipeline
- Push dataset

Deliverables:
1) Add 2 more probe seeds for Huge so total seeds = 3
2) Targets:
   - ee_direction_3d
   - ee_speed
3) Update Table 6 to mean ± std
4) If OOM occurs, use gradient checkpointing and/or smaller batch size; document the mitigation

Acceptance / reporting rule:
- If Huge still supports non-monotonic peak depth, report it.
- If the previous non-monotonicity looks unstable after 3 seeds, weaken or retract the claim.

========================
STOP POINT AFTER LAYER 1
========================
After finishing all Layer 1 tasks:
1) Produce a concise summary with:
   - what changed
   - what claims got stronger
   - what claims got weaker or were retracted
   - updated risk assessment for the current paper
2) Stop.
3) Wait for explicit human approval before starting Layer 2.

========================
LAYER 2: SPOTLIGHT-STRENGTHENING
========================
Start Layer 2 only after human approval.

--------------------------------
Task 2.1: 6-task full token-patch sweep
--------------------------------
Objective:
Bring PegInsert, NutThread, and Drawer into the same final recipe analysis regime as the core tasks.

Inputs:
- RGB videos for PegInsert, NutThread, Drawer
- timestep kinematics
- episode-level physics parameters
- token-patch extraction pipeline:
  resid_post + temporal_last_patch, 256×256 input, 16-frame stride 4

Deliverables:
1) Feature extraction for each task × model × layer:
   - V-JEPA 2 Large
   - VideoMAE-L
   - DINOv2-L

2) Tier A probing:
   - ee_pos
   - ee_speed
   - ee_accel_magnitude
   - ee_direction_sincos
   - ee_direction_3d
   - object_direction_3d
   - task-specific targets:
     PegInsert: insertion_depth, peg_hole_relative_position
     NutThread: axial_progress, nut_bolt_relative_angle
     Drawer: drawer_position, handle_grasp_state

3) Tier B probing for applicable physics variables:
   - object_mass
   - object_friction
   - surface_friction
   - damping
   - restitution

4) 3 probe seeds per target
5) 5-fold GroupKFold
6) 20-HP sweep
7) fake_mod5 negative control per task
8) Extended tables for 2D vs 3D direction and cross-model comparisons

Acceptance / reporting rule:
- Do not assume the same PEZ pattern transfers.
- Report each new task as:
  - clear PEZ-like
  - broad plateau
  - late-refinement
  - weak/no evidence
- Task-specific heterogeneity is an acceptable outcome.

--------------------------------
Task 2.2: Model panel extension (careful taxonomy)
--------------------------------
Objective:
Expand model comparison without making an unjustified family-level predictive-video claim.

Important taxonomy rule:
- Hiera should be treated as masked-video / MAE-like, not predictive-video.
- InternVideo2 is multi-objective and should NOT be used as a clean family-control model unless explicitly approved later.
- Predictive-video remains effectively n=1 in the current paper if only V-JEPA 2 is available.
- Therefore, do NOT make a strong family-level claim such as "predictive-video objectives generally produce PEZ."
- Allowed claim style:
  "Among tested models, V-JEPA 2 shows a distinctive mid-depth pattern; whether this extends to other predictive-video models remains open."

Deliverables:
1) Add Hiera as an additional masked-video-style comparator
2) Optionally add MAE ViT-L as an image baseline extension
3) Run on Push + Strike only:
   Push:
   - ee_direction_3d
   - ee_speed
   Strike:
   - ee_direction_3d
   - object_direction_3d
   - native_contact_force if already available

4) Extend the cross-model table accordingly

Acceptance / reporting rule:
- Strong family-level claim allowed only for masked-video if the evidence supports it with n≥2.
- Predictive-video claim must remain model-specific unless a second clean predictive-video model is actually available.

--------------------------------
Task 2.3: Native force expansion beyond Strike
--------------------------------
Objective:
Reduce dependence of Tier B native-force claims on Strike alone.

Inputs:
- Isaac Lab environments
- corrected Step-0 export pipeline

Deliverables:
1) Audit contact_* channels for:
   - Push
   - PegInsert
   - Drawer

2) Fix collector where needed:
   - object-side robot-contact sensor
   - task-specific body filtering

3) Recollect matched 1000-episode subsets for:
   - Push
   - PegInsert
   - Drawer

4) Native supervision:
   - contact_force_native
   - contact_happening_native

5) Probe:
   - V-JEPA 2 Large
   - VideoMAE-L
   - DINOv2-L
   - 3 seeds each

6) Paired bootstrap CIs for cross-model gaps
7) Extend Table 8 across tasks

Acceptance / reporting rule:
- Report whether the ranking
  V-JEPA 2 > VideoMAE-L > DINOv2-L
  is:
  - consistent across tasks
  - partially task-dependent
  - not supported outside Strike
- Any of the three outcomes is acceptable. Do not force consistency.

--------------------------------
Task 2.4: Static physics diagnostic
--------------------------------
Objective:
Determine whether static physics decoding failure reflects absent information or inaccessible information.

Inputs:
- Push static physics labels:
  - mass
  - object_friction
  - surface_friction
- existing token-patch features

Deliverables:
1) Split-by-value experiment:
   - 3-bin split: low / mid / high
   - train on low + high, test on mid
   - run for mass, object_friction, surface_friction
   - compare against random-episode split

2) Non-linear probe:
   - 2-layer MLP, hidden size 512
   - same HP sweep protocol
   - compare linear vs MLP by layer

3) Window-level event probing:
   - per-window feature to episode label
   - contact windows vs non-contact windows analyzed separately

4) Cross-task static physics probing where feasible:
   - Strike
   - PegInsert

5) Expand Section 5.1 / Table 12 accordingly

Acceptance / reporting rule:
For each target, conclude one of:
- no evidence the information is present
- some evidence it is present but not linearly accessible
- evidence is localized to contact windows
- evidence is task-dependent
Do not collapse these outcomes into a single narrative.

--------------------------------
Task 2.5: Recipe robustness ablation
--------------------------------
Objective:
Defend the token-patch recipe as a principled choice rather than an arbitrary engineering trick.

Inputs:
- configurable V-JEPA 2 Large extraction pipeline

Deliverables:
Run a controlled ablation on Push with:
Targets:
- ee_direction_3d
- ee_speed

Axes:
- Pooling: mean, max, temporal_last_patch, temporal_mean, CLS, spatial_center, token_flatten
- Residual: resid_pre, resid_mid, resid_post, attention_out, mlp_out
- Temporal window: 8, 16, 32
- Spatial resolution: 224, 256, 288

Outputs:
1) peak R²
2) peak layer
3) heatmap for pooling × residual
4) top-5 vs bottom-5 recipe table
5) short interpretation paragraph explaining why token-patch may suit manipulation

Acceptance / reporting rule:
- Report sensitivity honestly.
- If multiple recipes perform similarly, say so.
- Do not overstate uniqueness of the chosen recipe.

========================
STOP POINT AFTER LAYER 2
========================
After finishing any approved Layer 2 tasks:
1) Produce a concise summary:
   - what strengthened the paper
   - what remained weak
   - whether the current paper should stop here
   - whether any follow-up paper ideas became more justified
2) Stop.
3) Wait for explicit human instruction.

========================
LAYER 3: FOLLOW-UP PAPER BACKLOG ONLY
========================
Do NOT execute Layer 3 for the current paper.
Treat the following only as future-paper planning topics:
- causal intervention on PEZ layer
- mechanistic theory beyond correlational probing
- closed-loop policy learning with PEZ-aware design
- sim-to-real or sim-to-sim transfer
- any narrative pivot toward causal/mechanistic framing

Do not start any of these unless explicitly instructed in a separate future project.

========================
REPORTING FORMAT FOR EVERY TASK
========================
For each completed task, return:
1) What was run
2) Any deviations from plan
3) Main numerical results
4) Effect size + CI where applicable
5) Whether the original claim was strengthened, weakened, or retracted
6) Exact text suggestions for paper edits
7) Reproducibility metadata

========================
FINAL INSTRUCTION
========================
Begin with Layer 1 only.
Do not start Layer 2.2 or any Layer 3 work.
Do not continue past Layer 1 without explicit human approval.
