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
