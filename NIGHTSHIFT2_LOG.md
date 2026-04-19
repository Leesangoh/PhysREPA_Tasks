## [2026-04-19 03:46 UTC] [who: Codex] [NIGHTSHIFT2 START]
- Started Nightshift2 under `NIGHTSHIFT2_PROTOCOL.md`.
- Priority order accepted: Round 1 F5 frame shuffle -> Round 2 F3 CKA -> Round 3 F4-A split-by-value.

## [2026-04-19 03:46 UTC] [who: Codex] [startup audit]
- Push token cache present: `1500` safetensors at `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
- Strike token cache present: `2895` valid safetensors at `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike`
- Disk state: `/mnt/md1/solee` at `95%` usage with `199G` free
- GPU state at audit: GPUs `0-3` idle
- Reach cache already absent, so protocol's "delete Reach first if storage >95%" fallback is already effectively satisfied.

## [2026-04-19 03:47 UTC] [who: Codex] [Round 1 Step 1.1 design]
- Wrote `F5_design.md`.
- Design includes deterministic shuffle (`seed=42`), window-level permutation, PEZ-style probe parity, sanity checks, interpretation thresholds, and storage mitigation plan.
- Open review questions for Claude were embedded in the design doc.

## [2026-04-19 03:48 UTC] [who: Codex] [storage discussion]
- Main risk for Round 1 is storage, not compute.
- Existing raw caches plus full shuffled caches cannot coexist on `/mnt/md1/solee`.
- Planned workaround: preserve scientific baseline via committed Phase 2d probe results, recycle raw caches sequentially for shuffled extraction, and keep only final verdict artifacts.

## [2026-04-19 03:55 UTC] [who: Claude] [review escalation]
- Reviewer-quality concern raised: single-seed shuffle is not enough for a NeurIPS-quality claim.
- Required upgrades:
  - at least 3 shuffle seeds
  - paired statistical comparison plan
  - explicit reviewer-attack framing
  - effect-size reporting, not just raw curve overlays

## [2026-04-19 03:56 UTC] [who: Codex] [[NEURIPS MODE]]
- Accepted the stricter objective: paper-quality evidence over simple ablation execution.
- Upgraded `NIGHTSHIFT2_PROTOCOL.md` and `F5_design.md` to require:
  - seeds `42`, `123`, `2024`
  - paired/effect-size analysis plan
  - explicit reviewer attack model
  - counter-experiment interpretation rules

## [2026-04-19 04:10 UTC] [who: Claude] [oral escalation]
- Review standard escalated from paper-quality to oral-quality.
- Required additions:
  - mechanism-level claim framing
  - killer evidence chain
  - counter-evidence scenarios
  - Round 4 random-init null baseline after Round 1

## [2026-04-19 04:11 UTC] [who: Codex] [[ORAL MODE CONFIRMED]]
- Upgraded `NIGHTSHIFT2_PROTOCOL.md` and `F5_design.md` again.
- Round 1 now explicitly targets a causal-temporal claim, not just an ablation delta.
- Added oral-mode threshold interpretation:
  - `>= 30%` degradation => strong temporal evidence
  - `<= 10%` degradation => static/framewise evidence
  - `15-25%` degradation => mixed mechanism
- Added Round 4 placeholder: random-init V-JEPA 2 null baseline.

## [2026-04-19 04:01 UTC] [who: Codex] [F5 sanity result]
- `--shuffle-frames` implementation passed the first deterministic sanity check.
- Two independent 1-episode dry-runs produced the same first-window permutation:
  - `episode=000000`
  - `window_start=0`
  - `seed=42000126`
  - `permutation=[15, 11, 5, 6, 3, 1, 7, 13, 10, 4, 9, 14, 12, 0, 2, 8]`
- This confirms:
  - frame order is genuinely changed (not identity)
  - shuffling is deterministic under the same global seed

## [2026-04-19 04:02 UTC] [who: Codex] [next decision]
- Before deleting large raw caches for multi-seed shuffled extraction, one open operational question remains:
  - whether to snapshot compact original Push/Strike features now for Round 2 CKA, or accept later re-extraction cost
- Default leaning: snapshot a balanced compact representation first if it can be done quickly and without destabilizing Round 1.

## [2026-04-19 04:07 UTC] [who: Codex] [Round 2 insurance before cache recycling]
- Chose to preserve Round 2 feasibility before deleting original Push raw cache.
- Launched compact Push CKA snapshot job:
  - script: `prepare_cka_snapshot.py`
  - input: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
  - output: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_cka/push`
- Rationale:
  - F5 requires cache recycling because storage is tight
  - CKA later still needs original non-shuffled representations
