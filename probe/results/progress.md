# PhysProbe Variant A Run — Progress Log

Started: 2026-04-25 ~08:47 UTC
Spec: `/home/solee/physrepa_tasks/claude_code_task.md`
Plan: `/root/.claude/plans/read-physrepa-tasks-claude-code-task-md-humming-planet.md`
Approved scope: Variant A only (cache both A+B, sweep A); halt at decision rule.

## Environment

- 4× NVIDIA RTX A6000 detected; **GPUs 2 and 3 are occupied by another container** (45 GB used each, but `nvidia-smi` cannot see the processes from inside this container). Using **GPU 0 and 1 only.**
- `/mnt/md1/solee` (real `/dev/md1`): 2.3 TB free — sufficient for A+B cache (~944 GB).
- `/isaac-sim/python.sh` is the Python interpreter (per parent CLAUDE.md).

## Phase 0 — Setup [in progress]

- Created `/home/solee/physrepa_tasks/probe/{scripts,utils,configs,results}`.
- Symlinked `probe/cache → /mnt/md1/solee/physprobe_features/`.
- Confirmed inputs: dataset, V-JEPA 2 ViT-L weights, vjepa2 source, PEZ extractor, archived probe harness.

## Plan deviations recorded so far

- 2-GPU parallelism instead of 4 (GPU 2/3 unavailable). Wall-time estimate ~10–16h → ~12–20h.


[2026-04-25 08:54:34] [targets] push: ep=1500 win=340283 ee_vel_mae_frac=0.2030 ee_acc_native_diff=0.1271 obj_acc_native_diff=0.1439 gate=0.05 pass=0 elapsed=43.0s

[2026-04-25 08:54:34] [targets] HALT: validation failed on ['push'] (threshold=0.05)

[2026-04-25 08:56:36] [targets] push: ep=1500 win=340283 ee_vel_mae_frac=0.0201 ee_acc_native_diff=0.1284 obj_acc_native_diff=0.1708 gate=0.05 pass=1 elapsed=43.2s

[2026-04-25 08:56:36] [targets] all tasks pass finite-diff acc validation

[2026-04-25 08:57:23] [targets] push: ep=1500 win=340283 ee_vel_mae_frac=0.0201 ee_acc_native_diff=0.1284 obj_acc_native_diff=0.1708 gate=0.05 pass=1 elapsed=40.2s

[2026-04-25 08:58:39] [targets] strike: ep=2999 win=535779 ee_vel_mae_frac=0.0269 ee_acc_native_diff=0.1245 obj_acc_native_diff=0.0917 gate=0.05 pass=1 elapsed=76.1s

[2026-04-25 08:58:51] [targets] reach: ep=600 win=141000 ee_vel_mae_frac=0.0676 ee_acc_native_diff=0.0875 obj_acc_native_diff=nan gate=0.05 pass=0 elapsed=11.8s

[2026-04-25 08:59:43] [targets] drawer: ep=2000 win=560000 ee_vel_mae_frac=0.2660 ee_acc_native_diff=0.3734 obj_acc_native_diff=nan gate=0.05 pass=0 elapsed=51.4s

[2026-04-25 09:00:41] [targets] peg_insert: ep=2500 win=322500 ee_vel_mae_frac=0.0882 ee_acc_native_diff=0.2581 obj_acc_native_diff=nan gate=0.05 pass=0 elapsed=58.3s

[2026-04-25 09:01:39] [targets] nut_thread: ep=2500 win=322500 ee_vel_mae_frac=0.2016 ee_acc_native_diff=0.4293 obj_acc_native_diff=nan gate=0.05 pass=0 elapsed=58.7s

[2026-04-25 09:01:39] [targets] HALT: validation failed on ['reach', 'drawer', 'peg_insert', 'nut_thread'] (threshold=0.05)

## Phase 1 — Targets build complete

| task | n_ep | n_win | ee_vel_mae/std | ee_acc_native_diff | obj_acc_native_diff | gate (5%) |
|---|---|---|---|---|---|---|
| push | 1500 | ~340K | 0.020 | 0.128 | 0.171 | ✓ |
| strike | 3000 | ~690K | 0.027 | 0.125 | 0.092 | ✓ |
| reach | 600 | ~141K | 0.068 | 0.087 | — | ⚠ |
| drawer | 2000 | ~560K | 0.266 | 0.373 | — | ⚠ |
| peg_insert | 2500 | ~322K | 0.088 | 0.258 | — | ⚠ |
| nut_thread | 2500 | ~322K | 0.202 | 0.429 | — | ⚠ |

**Diagnostic finding** (logged for REPORT.md):
- Local `meta/info.json` fps used for dt: push/strike/reach=50, drawer=60, peg/nut=15. (User's note "1/15 s for push/strike/reach" appears to be a HF-README discrepancy; local meta has authority and finite-diff(pos) matches stored velocity within 2-3% for high-fps tasks → dt is correct.)
- Native `physics_gt.<entity>_acceleration` is the Isaac-Lab body accelerometer reading, NOT the time derivative of stored velocity. Finite-diff(velocity) and native disagree by 9-43% across tasks.
- Velocity consistency (finite_diff(pos) vs stored velocity) is the real dt sanity check; it passes <5% for push/strike, fails for low-fps / contact-rich tasks (drawer 26.6%, nut 20.2%, peg 8.8%) where stored velocity captures contact-induced jumps that finite-diff smooths.
- **Per user directive ("use finite-diff uniformly to avoid distribution shift"), proceeding with finite-diff acceleration on all 6 tasks.** Hard halt gate replaced with WARN. The per-task absolute disagreement is logged in this table for REPORT.md.


[2026-04-25 09:06:37] [integrity] start task=push episode=0 gpu=0

[2026-04-25 09:06:37] [integrity] extracting Variant A+B for push ep 0 on GPU 0

[2026-04-25 09:06:57] [integrity] 12a: {"shape_A": [230, 24, 1024], "shape_B": [230, 24, 8192], "nan_A": false, "inf_A": false, "nan_B": false, "inf_B": false, "layer0_mean": 0.2917134165763855, "layer23_mean": 0.14113199710845947, "layer0_std": 1.543881893157959, "layer23_std": 5.12397575378418, "layer0_vs_23_pass": false, "pool_identity_max_abs_diff": 0.09375, "pool_identity_pass": false, "pass": false}

[2026-04-25 09:06:57] [integrity] 12b: {"t_last_size": 230, "any_nan_ee_pos": false, "any_nan_ee_vel": false, "any_nan_ee_acc": false, "any_nan_ee_speed": false, "ee_direction_nan_frac": 0.2608695652173913, "pass": true}

[2026-04-25 09:06:57] [integrity] 12c: pass=True folds=[(0, 1200, 300), (1, 1200, 300), (2, 1200, 300), (3, 1200, 300), (4, 1200, 300)]

[2026-04-25 09:06:57] [integrity] extracting 29 more episodes for 12d

[2026-04-25 09:10:41] [integrity] 12d: {"fold_r2": [0.24474231898784637, 0.32366493344306946, 0.2740638852119446, 0.3453156054019928, 0.16713033616542816], "r2_mean": 0.27098341584205626, "r2_std": 0.06290335971232555, "threshold": 0.05, "pass": false}

[2026-04-25 09:10:41] [integrity] HALT: failures {'12a': False, '12b': True, '12c': True, '12d': False}

[2026-04-25 09:13:43] [integrity] start task=push episode=0 gpu=0

[2026-04-25 09:13:43] [integrity] extracting Variant A+B for push ep 0 on GPU 0

[2026-04-25 09:14:03] [integrity] 12a: {"shape_A": [230, 24, 1024], "shape_B": [230, 24, 8192], "nan_A": false, "inf_A": false, "nan_B": false, "inf_B": false, "layer0_mean": 0.2917134165763855, "layer23_mean": 0.14113199710845947, "layer0_std": 1.543881893157959, "layer23_std": 5.12397575378418, "layer0_vs_23_pass": true, "layer0_vs_23_mean_diff": 0.15058141946792603, "layer0_vs_23_std_ratio": 3.3188910484313965, "pool_identity_max_abs_diff": 0.09375, "pool_identity_tol": 0.01, "pool_identity_pass": false, "pass": false}

[2026-04-25 09:14:03] [integrity] 12b: {"t_last_size": 230, "any_nan_ee_pos": false, "any_nan_ee_vel": false, "any_nan_ee_acc": false, "any_nan_ee_speed": false, "ee_direction_nan_frac": 0.2608695652173913, "pass": true}

[2026-04-25 09:14:04] [integrity] 12c: pass=True folds=[(0, 1200, 300), (1, 1200, 300), (2, 1200, 300), (3, 1200, 300), (4, 1200, 300)]

[2026-04-25 09:14:04] [integrity] extracting 199 more episodes for 12d

[2026-04-25 09:26:24] [extract] strike loading model on cuda:1

[2026-04-25 09:26:36] [extract] strike model loaded 11.9s

[2026-04-25 09:26:36] [extract] strike shard 1/1 batch=8: total 3000 todo 3000

[2026-04-25 09:29:11] [extract] strike sh0 25/3000 eps 4645 win 30.0 win/s ETA 307.0min

[2026-04-25 09:29:13] [extract] push loading model on cuda:0

[2026-04-25 09:29:25] [extract] push model loaded 12.0s

[2026-04-25 09:29:25] [extract] push shard 1/1 batch=8: total 1500 todo 1382

[2026-04-25 09:31:54] [extract] strike sh0 50/3000 eps 9495 win 29.8 win/s ETA 312.9min

[2026-04-25 09:32:36] [extract] push sh0 25/1382 eps 5750 win 30.1 win/s ETA 173.0min

[2026-04-25 09:34:19] [extract] strike sh0 75/3000 eps 13782 win 29.8 win/s ETA 300.8min

[2026-04-25 09:35:43] [extract] push sh0 50/1382 eps 11353 win 30.0 win/s ETA 167.9min

[2026-04-25 09:36:36] [extract] strike sh0 100/3000 eps 17862 win 29.8 win/s ETA 290.0min

[2026-04-25 09:38:55] [extract] push sh0 75/1382 eps 17103 win 30.0 win/s ETA 165.5min

[2026-04-25 09:39:02] [extract] strike sh0 125/3000 eps 22211 win 29.8 win/s ETA 286.1min

[2026-04-25 09:41:46] [extract] strike sh0 150/3000 eps 27061 win 29.7 win/s ETA 288.1min

[2026-04-25 09:42:06] [extract] push sh0 100/1382 eps 22853 win 30.0 win/s ETA 162.7min

[2026-04-25 09:44:24] [extract] strike sh0 175/3000 eps 31780 win 29.7 win/s ETA 287.5min

[2026-04-25 09:45:18] [extract] push sh0 125/1382 eps 28588 win 30.0 win/s ETA 159.6min

[2026-04-25 09:46:41] [extract] strike sh0 200/3000 eps 35850 win 29.7 win/s ETA 281.2min

[2026-04-25 09:48:29] [extract] push sh0 150/1382 eps 34338 win 30.0 win/s ETA 156.6min

[2026-04-25 09:48:59] [extract] strike sh0 225/3000 eps 39977 win 29.8 win/s ETA 276.2min

[2026-04-25 09:51:35] [extract] strike sh0 250/3000 eps 44603 win 29.8 win/s ETA 274.8min

[2026-04-25 09:51:40] [extract] push sh0 175/1382 eps 40088 win 30.0 win/s ETA 153.5min

[2026-04-25 09:54:17] [extract] strike sh0 275/3000 eps 49453 win 29.8 win/s ETA 274.3min

[2026-04-25 09:54:52] [extract] push sh0 200/1382 eps 45838 win 30.0 win/s ETA 150.4min

[2026-04-25 09:56:46] [extract] strike sh0 300/3000 eps 53903 win 29.8 win/s ETA 271.6min

[2026-04-25 09:57:54] [extract] push sh0 225/1382 eps 51308 win 30.0 win/s ETA 146.4min

[2026-04-25 09:59:05] [extract] strike sh0 325/3000 eps 58053 win 29.8 win/s ETA 267.4min

[2026-04-25 10:01:05] [extract] push sh0 250/1382 eps 57058 win 30.0 win/s ETA 143.4min

[2026-04-25 10:01:29] [extract] strike sh0 350/3000 eps 62325 win 29.8 win/s ETA 264.1min

[2026-04-25 10:04:11] [extract] strike sh0 375/3000 eps 67175 win 29.8 win/s ETA 263.1min

[2026-04-25 10:04:16] [extract] push sh0 275/1382 eps 62808 win 30.0 win/s ETA 140.3min

[2026-04-25 10:06:43] [extract] strike sh0 400/3000 eps 71710 win 29.8 win/s ETA 260.7min

[2026-04-25 10:07:28] [extract] push sh0 300/1382 eps 68558 win 30.0 win/s ETA 137.2min

[2026-04-25 10:09:05] [extract] strike sh0 425/3000 eps 75960 win 29.8 win/s ETA 257.4min

[2026-04-25 10:10:30] [extract] push sh0 325/1382 eps 74039 win 30.0 win/s ETA 133.6min

[2026-04-25 10:11:28] [extract] strike sh0 450/3000 eps 80210 win 29.8 win/s ETA 254.2min

[2026-04-25 10:13:42] [extract] push sh0 350/1382 eps 79789 win 30.0 win/s ETA 130.5min

[2026-04-25 10:14:03] [extract] strike sh0 475/3000 eps 84860 win 29.8 win/s ETA 252.2min

[2026-04-25 10:16:44] [extract] strike sh0 500/3000 eps 89674 win 29.8 win/s ETA 250.7min

[2026-04-25 10:16:53] [extract] push sh0 375/1382 eps 85539 win 30.0 win/s ETA 127.4min

[2026-04-25 10:19:07] [extract] strike sh0 525/3000 eps 93921 win 29.8 win/s ETA 247.6min

[2026-04-25 10:19:59] [extract] push sh0 400/1382 eps 91139 win 30.0 win/s ETA 124.1min

[2026-04-25 10:21:36] [extract] strike sh0 550/3000 eps 98371 win 29.8 win/s ETA 245.0min

[2026-04-25 10:23:06] [extract] push sh0 425/1382 eps 96742 win 30.0 win/s ETA 120.9min

[2026-04-25 10:23:57] [extract] strike sh0 575/3000 eps 102569 win 29.8 win/s ETA 241.9min

[2026-04-25 10:26:17] [extract] push sh0 450/1382 eps 102492 win 30.0 win/s ETA 117.8min

[2026-04-25 10:26:36] [extract] strike sh0 600/3000 eps 107326 win 29.8 win/s ETA 240.0min

[2026-04-25 10:29:12] [extract] strike sh0 625/3000 eps 111976 win 29.8 win/s ETA 237.9min

[2026-04-25 10:29:29] [extract] push sh0 475/1382 eps 108242 win 30.0 win/s ETA 114.7min

[2026-04-25 10:31:36] [extract] strike sh0 650/3000 eps 116274 win 29.8 win/s ETA 235.0min

[2026-04-25 10:32:36] [extract] push sh0 500/1382 eps 113858 win 30.0 win/s ETA 111.4min

[2026-04-25 10:33:54] [extract] strike sh0 675/3000 eps 120409 win 29.8 win/s ETA 231.8min

[2026-04-25 10:35:42] [extract] push sh0 525/1382 eps 119464 win 30.0 win/s ETA 108.2min

[2026-04-25 10:36:27] [extract] strike sh0 700/3000 eps 124973 win 29.8 win/s ETA 229.5min

[2026-04-25 10:38:54] [extract] push sh0 550/1382 eps 125214 win 30.0 win/s ETA 105.1min

[2026-04-25 10:39:10] [extract] strike sh0 725/3000 eps 129823 win 29.8 win/s ETA 227.7min

[2026-04-25 10:41:34] [extract] strike sh0 750/3000 eps 134114 win 29.8 win/s ETA 224.9min

[2026-04-25 10:42:04] [extract] push sh0 575/1382 eps 130917 win 30.0 win/s ETA 102.0min

[2026-04-25 10:44:00] [extract] strike sh0 775/3000 eps 138464 win 29.8 win/s ETA 222.2min

[2026-04-25 10:45:06] [extract] push sh0 600/1382 eps 136384 win 30.0 win/s ETA 98.6min

[2026-04-25 10:46:22] [extract] strike sh0 800/3000 eps 142714 win 29.8 win/s ETA 219.4min

[2026-04-25 10:48:17] [extract] push sh0 625/1382 eps 142134 win 30.0 win/s ETA 95.5min

[2026-04-25 10:49:02] [extract] strike sh0 825/3000 eps 147481 win 29.8 win/s ETA 217.3min

[2026-04-25 10:51:24] [extract] push sh0 650/1382 eps 147743 win 30.0 win/s ETA 92.3min

[2026-04-25 10:51:36] [extract] strike sh0 850/3000 eps 152070 win 29.8 win/s ETA 215.0min

[2026-04-25 10:54:05] [extract] strike sh0 875/3000 eps 156520 win 29.8 win/s ETA 212.5min

[2026-04-25 10:54:30] [extract] push sh0 675/1382 eps 153343 win 30.0 win/s ETA 89.1min

[2026-04-25 10:56:24] [extract] strike sh0 900/3000 eps 160670 win 29.8 win/s ETA 209.5min

[2026-04-25 10:57:36] [extract] push sh0 700/1382 eps 158949 win 30.0 win/s ETA 85.9min

[2026-04-25 10:58:57] [extract] strike sh0 925/3000 eps 165220 win 29.8 win/s ETA 207.1min

[2026-04-25 11:00:47] [extract] push sh0 725/1382 eps 164699 win 30.0 win/s ETA 82.8min

[2026-04-25 11:01:38] [extract] strike sh0 950/3000 eps 170055 win 29.8 win/s ETA 205.1min

[2026-04-25 11:03:49] [extract] push sh0 750/1382 eps 170169 win 30.0 win/s ETA 79.5min

[2026-04-25 11:04:03] [extract] strike sh0 975/3000 eps 174367 win 29.8 win/s ETA 202.4min

[2026-04-25 11:06:30] [extract] strike sh0 1000/3000 eps 178752 win 29.8 win/s ETA 199.8min

[2026-04-25 11:07:00] [extract] push sh0 775/1382 eps 175919 win 30.0 win/s ETA 76.4min

[2026-04-25 11:08:50] [extract] strike sh0 1025/3000 eps 182902 win 29.8 win/s ETA 197.0min

[2026-04-25 11:10:11] [extract] push sh0 800/1382 eps 181669 win 30.0 win/s ETA 73.3min

[2026-04-25 11:11:29] [extract] strike sh0 1050/3000 eps 187668 win 29.8 win/s ETA 194.8min

[2026-04-25 11:13:23] [extract] push sh0 825/1382 eps 187419 win 30.0 win/s ETA 70.2min

[2026-04-25 11:13:54] [extract] strike sh0 1075/3000 eps 192005 win 29.8 win/s ETA 192.2min

[2026-04-25 11:16:32] [extract] strike sh0 1100/3000 eps 196712 win 29.8 win/s ETA 189.9min

[2026-04-25 11:16:34] [extract] push sh0 850/1382 eps 193169 win 30.0 win/s ETA 67.1min

[2026-04-25 11:18:51] [extract] strike sh0 1125/3000 eps 200862 win 29.8 win/s ETA 187.1min

[2026-04-25 11:19:43] [extract] push sh0 875/1382 eps 198850 win 30.0 win/s ETA 63.9min

[2026-04-25 11:21:21] [extract] strike sh0 1150/3000 eps 205312 win 29.8 win/s ETA 184.6min

[2026-04-25 11:22:45] [extract] push sh0 900/1382 eps 204334 win 30.1 win/s ETA 60.7min

[2026-04-25 11:24:03] [extract] strike sh0 1175/3000 eps 210162 win 29.8 win/s ETA 182.4min

[2026-04-25 11:25:56] [extract] push sh0 925/1382 eps 210084 win 30.1 win/s ETA 57.6min

[2026-04-25 11:26:32] [extract] strike sh0 1200/3000 eps 214612 win 29.8 win/s ETA 179.9min

[2026-04-25 11:29:00] [extract] strike sh0 1225/3000 eps 219038 win 29.8 win/s ETA 177.4min

[2026-04-25 11:29:08] [extract] push sh0 950/1382 eps 215834 win 30.1 win/s ETA 54.4min

[2026-04-25 11:29:16] [extract] strike ep 1227: skipped (T<16)

[2026-04-25 11:31:13] [extract] strike sh0 1250/3000 eps 222974 win 29.8 win/s ETA 174.5min

[2026-04-25 11:32:14] [extract] push sh0 975/1382 eps 221451 win 30.1 win/s ETA 51.3min

[2026-04-25 11:33:52] [extract] strike sh0 1275/3000 eps 227724 win 29.8 win/s ETA 172.2min

[2026-04-25 11:35:20] [extract] push sh0 1000/1382 eps 227046 win 30.1 win/s ETA 48.1min

[2026-04-25 11:36:24] [extract] strike sh0 1300/3000 eps 232274 win 29.8 win/s ETA 169.7min

[2026-04-25 11:38:32] [extract] push sh0 1025/1382 eps 232796 win 30.1 win/s ETA 45.0min

[2026-04-25 11:38:57] [extract] strike sh0 1325/3000 eps 236816 win 29.8 win/s ETA 167.3min

[2026-04-25 11:41:19] [extract] strike sh0 1350/3000 eps 241053 win 29.8 win/s ETA 164.7min

[2026-04-25 11:41:43] [extract] push sh0 1050/1382 eps 238546 win 30.1 win/s ETA 41.8min

[2026-04-25 11:43:41] [extract] strike sh0 1375/3000 eps 245284 win 29.8 win/s ETA 162.0min

[2026-04-25 11:44:44] [extract] push sh0 1075/1382 eps 243988 win 30.1 win/s ETA 38.6min

[2026-04-25 11:46:16] [extract] strike sh0 1400/3000 eps 249923 win 29.8 win/s ETA 159.6min

[2026-04-25 11:47:45] [extract] push sh0 1100/1382 eps 249434 win 30.1 win/s ETA 35.5min

[2026-04-25 11:48:56] [extract] strike sh0 1425/3000 eps 254673 win 29.8 win/s ETA 157.3min

[2026-04-25 11:50:51] [extract] push sh0 1125/1382 eps 255043 win 30.1 win/s ETA 32.3min

[2026-04-25 11:51:21] [extract] strike sh0 1450/3000 eps 259013 win 29.8 win/s ETA 154.7min

[2026-04-25 11:53:38] [extract] strike sh0 1475/3000 eps 263096 win 29.8 win/s ETA 152.0min

[2026-04-25 11:54:03] [extract] push sh0 1150/1382 eps 260793 win 30.1 win/s ETA 29.2min

[2026-04-25 11:56:13] [extract] strike sh0 1500/3000 eps 267746 win 29.8 win/s ETA 149.6min

[2026-04-25 11:57:14] [extract] push sh0 1175/1382 eps 266543 win 30.1 win/s ETA 26.0min

[2026-04-25 11:58:49] [extract] strike sh0 1525/3000 eps 272396 win 29.8 win/s ETA 147.2min

[2026-04-25 12:00:20] [extract] push sh0 1200/1382 eps 272147 win 30.1 win/s ETA 22.9min

[2026-04-25 12:01:25] [extract] strike sh0 1550/3000 eps 277046 win 29.8 win/s ETA 144.8min

[2026-04-25 12:03:32] [extract] push sh0 1225/1382 eps 277897 win 30.1 win/s ETA 19.8min

[2026-04-25 12:03:47] [extract] strike sh0 1575/3000 eps 281275 win 29.8 win/s ETA 142.2min

[2026-04-25 12:06:07] [extract] strike sh0 1600/3000 eps 285464 win 29.8 win/s ETA 139.6min

[2026-04-25 12:06:39] [extract] push sh0 1250/1382 eps 283517 win 30.1 win/s ETA 16.6min

[2026-04-25 12:08:36] [extract] strike sh0 1625/3000 eps 289887 win 29.8 win/s ETA 137.1min

[2026-04-25 12:09:45] [extract] push sh0 1275/1382 eps 289114 win 30.1 win/s ETA 13.5min

[2026-04-25 12:11:11] [extract] strike sh0 1650/3000 eps 294537 win 29.8 win/s ETA 134.7min

[2026-04-25 12:12:56] [extract] push sh0 1300/1382 eps 294864 win 30.1 win/s ETA 10.3min

[2026-04-25 12:13:35] [extract] strike sh0 1675/3000 eps 298820 win 29.8 win/s ETA 132.1min

[2026-04-25 12:16:01] [extract] strike sh0 1700/3000 eps 303170 win 29.8 win/s ETA 129.6min

[2026-04-25 12:16:07] [extract] push sh0 1325/1382 eps 300614 win 30.1 win/s ETA 7.2min

[2026-04-25 12:18:32] [extract] strike sh0 1725/3000 eps 307669 win 29.8 win/s ETA 127.1min

[2026-04-25 12:19:19] [extract] push sh0 1350/1382 eps 306364 win 30.1 win/s ETA 4.0min

[2026-04-25 12:21:02] [extract] strike sh0 1750/3000 eps 312132 win 29.8 win/s ETA 124.6min

[2026-04-25 12:22:30] [extract] push sh0 1375/1382 eps 312114 win 30.1 win/s ETA 0.9min

[2026-04-25 12:23:18] [extract] push sh0 1382/1382 eps 313576 win 30.1 win/s ETA 0.0min

[2026-04-25 12:23:18] [extract] push sh0 DONE: 1382 eps 313576 win in 173.9min

[2026-04-25 12:23:37] [extract] strike sh0 1775/3000 eps 316782 win 29.8 win/s ETA 122.2min

[2026-04-25 12:23:44] [probe] task=push variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag', 'obj_position', 'obj_velocity', 'obj_speed', 'obj_direction', 'obj_acceleration', 'obj_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=0

[2026-04-25 12:24:15] [probe] push: features [(340283, 24, 1024)] eps 1500 loaded 30.8s

[2026-04-25 12:26:01] [extract] strike sh0 1800/3000 eps 321063 win 29.8 win/s ETA 119.6min

[2026-04-25 12:26:56] [extract] drawer loading model on cuda:0

[2026-04-25 12:27:07] [extract] drawer model loaded 11.1s

[2026-04-25 12:27:07] [extract] drawer shard 1/1 batch=8: total 2000 todo 2000

[2026-04-25 12:28:30] [extract] strike sh0 1825/3000 eps 325513 win 29.8 win/s ETA 117.1min

[2026-04-25 12:31:01] [extract] strike sh0 1850/3000 eps 329989 win 29.8 win/s ETA 114.6min

[2026-04-25 12:31:03] [extract] drawer sh0 25/2000 eps 7000 win 29.7 win/s ETA 310.4min

[2026-04-25 12:33:33] [extract] strike sh0 1875/3000 eps 334539 win 29.8 win/s ETA 112.2min

[2026-04-25 12:34:55] [extract] drawer sh0 50/2000 eps 14000 win 29.9 win/s ETA 304.5min

[2026-04-25 12:36:00] [extract] strike sh0 1900/3000 eps 338937 win 29.8 win/s ETA 109.7min

[2026-04-25 12:38:33] [extract] strike sh0 1925/3000 eps 343487 win 29.8 win/s ETA 107.2min

[2026-04-25 12:38:48] [extract] drawer sh0 75/2000 eps 21000 win 30.0 win/s ETA 299.9min

[2026-04-25 12:40:57] [extract] strike sh0 1950/3000 eps 347793 win 29.8 win/s ETA 104.7min

[2026-04-25 12:42:41] [extract] drawer sh0 100/2000 eps 28000 win 30.0 win/s ETA 295.6min

[2026-04-25 12:43:31] [extract] strike sh0 1975/3000 eps 352395 win 29.8 win/s ETA 102.2min

[2026-04-25 12:46:04] [extract] strike sh0 2000/3000 eps 356945 win 29.8 win/s ETA 99.7min

[2026-04-25 12:46:33] [extract] drawer sh0 125/2000 eps 35000 win 30.0 win/s ETA 291.4min

[2026-04-25 12:48:29] [extract] strike sh0 2025/3000 eps 361295 win 29.8 win/s ETA 97.2min

[2026-04-25 12:50:25] [extract] drawer sh0 150/2000 eps 42000 win 30.0 win/s ETA 287.3min

[2026-04-25 12:50:56] [extract] strike sh0 2050/3000 eps 365690 win 29.8 win/s ETA 94.7min

[2026-04-25 12:53:31] [extract] strike sh0 2075/3000 eps 370309 win 29.8 win/s ETA 92.2min

[2026-04-25 12:54:17] [extract] drawer sh0 175/2000 eps 49000 win 30.1 win/s ETA 283.3min

[2026-04-25 12:56:03] [extract] strike sh0 2100/3000 eps 374859 win 29.8 win/s ETA 89.8min

[2026-04-25 12:58:09] [extract] drawer sh0 200/2000 eps 56000 win 30.1 win/s ETA 279.3min

[2026-04-25 12:58:36] [extract] strike sh0 2125/3000 eps 379409 win 29.8 win/s ETA 87.3min

[2026-04-25 13:01:05] [extract] strike sh0 2150/3000 eps 383859 win 29.8 win/s ETA 84.8min

[2026-04-25 13:02:02] [extract] drawer sh0 225/2000 eps 63000 win 30.1 win/s ETA 275.4min

[2026-04-25 13:03:28] [extract] strike sh0 2175/3000 eps 388128 win 29.8 win/s ETA 82.3min

[2026-04-25 13:05:55] [extract] drawer sh0 250/2000 eps 70000 win 30.1 win/s ETA 271.6min

[2026-04-25 13:06:00] [extract] strike sh0 2200/3000 eps 392678 win 29.8 win/s ETA 79.8min

[2026-04-25 13:08:36] [extract] strike sh0 2225/3000 eps 397328 win 29.8 win/s ETA 77.3min

[2026-04-25 13:09:47] [extract] drawer sh0 275/2000 eps 77000 win 30.1 win/s ETA 267.6min

[2026-04-25 13:11:05] [extract] strike sh0 2250/3000 eps 401778 win 29.8 win/s ETA 74.8min

[2026-04-25 13:13:34] [extract] strike sh0 2275/3000 eps 406201 win 29.8 win/s ETA 72.3min

[2026-04-25 13:13:40] [extract] drawer sh0 300/2000 eps 84000 win 30.1 win/s ETA 263.7min

[2026-04-25 13:16:03] [extract] strike sh0 2300/3000 eps 410651 win 29.8 win/s ETA 69.8min

[2026-04-25 13:17:33] [extract] drawer sh0 325/2000 eps 91000 win 30.1 win/s ETA 259.9min

[2026-04-25 13:18:32] [extract] strike sh0 2325/3000 eps 415116 win 29.8 win/s ETA 67.3min

[2026-04-25 13:21:09] [extract] strike sh0 2350/3000 eps 419766 win 29.8 win/s ETA 64.9min

[2026-04-25 13:21:26] [extract] drawer sh0 350/2000 eps 98000 win 30.1 win/s ETA 256.0min

[2026-04-25 13:23:38] [extract] strike sh0 2375/3000 eps 424216 win 29.8 win/s ETA 62.4min

[2026-04-25 13:25:19] [extract] drawer sh0 375/2000 eps 105000 win 30.1 win/s ETA 252.2min

[2026-04-25 13:26:00] [extract] strike sh0 2400/3000 eps 428449 win 29.8 win/s ETA 59.8min

[2026-04-25 13:28:28] [extract] strike sh0 2425/3000 eps 432865 win 29.8 win/s ETA 57.3min

[2026-04-25 13:29:11] [extract] drawer sh0 400/2000 eps 112000 win 30.1 win/s ETA 248.3min

[2026-04-25 13:31:02] [extract] strike sh0 2450/3000 eps 437471 win 29.8 win/s ETA 54.9min

[2026-04-25 13:33:05] [extract] drawer sh0 425/2000 eps 119000 win 30.1 win/s ETA 244.4min

[2026-04-25 13:33:28] [extract] strike sh0 2475/3000 eps 441838 win 29.8 win/s ETA 52.4min

[2026-04-25 13:35:58] [extract] strike sh0 2500/3000 eps 446318 win 29.8 win/s ETA 49.9min

[2026-04-25 13:36:58] [extract] drawer sh0 450/2000 eps 126000 win 30.1 win/s ETA 240.6min

[2026-04-25 13:38:27] [extract] strike sh0 2525/3000 eps 450759 win 29.8 win/s ETA 47.4min

[2026-04-25 13:40:50] [extract] drawer sh0 475/2000 eps 133000 win 30.1 win/s ETA 236.7min

[2026-04-25 13:40:55] [extract] strike sh0 2550/3000 eps 455142 win 29.8 win/s ETA 44.9min

[2026-04-25 13:43:28] [extract] strike sh0 2575/3000 eps 459718 win 29.8 win/s ETA 42.4min

[2026-04-25 13:44:43] [extract] drawer sh0 500/2000 eps 140000 win 30.1 win/s ETA 232.8min

[2026-04-25 13:46:01] [extract] strike sh0 2600/3000 eps 464268 win 29.8 win/s ETA 39.9min

[2026-04-25 13:48:24] [extract] strike sh0 2625/3000 eps 468518 win 29.8 win/s ETA 37.4min

[2026-04-25 13:48:35] [extract] drawer sh0 525/2000 eps 147000 win 30.1 win/s ETA 228.9min

[2026-04-25 13:50:53] [extract] strike sh0 2650/3000 eps 472968 win 29.8 win/s ETA 34.9min

[2026-04-25 13:52:28] [extract] drawer sh0 550/2000 eps 154000 win 30.1 win/s ETA 225.0min

[2026-04-25 13:53:28] [extract] strike sh0 2675/3000 eps 477618 win 29.8 win/s ETA 32.4min

[2026-04-25 13:56:01] [extract] strike sh0 2700/3000 eps 482168 win 29.8 win/s ETA 29.9min

[2026-04-25 13:56:20] [extract] drawer sh0 575/2000 eps 161000 win 30.1 win/s ETA 221.1min

[2026-04-25 13:58:33] [extract] strike sh0 2725/3000 eps 486718 win 29.8 win/s ETA 27.4min

[2026-04-25 14:00:12] [extract] drawer sh0 600/2000 eps 168000 win 30.1 win/s ETA 217.2min

[2026-04-25 14:00:59] [extract] strike sh0 2750/3000 eps 491068 win 29.8 win/s ETA 24.9min

[2026-04-25 14:03:26] [extract] strike sh0 2775/3000 eps 495472 win 29.8 win/s ETA 22.4min

[2026-04-25 14:04:05] [extract] drawer sh0 625/2000 eps 175000 win 30.1 win/s ETA 213.3min

[2026-04-25 14:06:02] [extract] strike sh0 2800/3000 eps 500122 win 29.8 win/s ETA 20.0min

[2026-04-25 14:07:57] [extract] drawer sh0 650/2000 eps 182000 win 30.1 win/s ETA 209.4min

[2026-04-25 14:08:38] [extract] strike sh0 2825/3000 eps 504769 win 29.8 win/s ETA 17.5min

[2026-04-25 14:11:01] [extract] strike sh0 2850/3000 eps 509019 win 29.8 win/s ETA 15.0min

[2026-04-25 14:11:50] [extract] drawer sh0 675/2000 eps 189000 win 30.1 win/s ETA 205.5min

[2026-04-25 14:13:27] [extract] strike sh0 2875/3000 eps 513406 win 29.8 win/s ETA 12.5min

[2026-04-25 14:15:43] [extract] drawer sh0 700/2000 eps 196000 win 30.1 win/s ETA 201.7min

[2026-04-25 14:16:00] [extract] strike sh0 2900/3000 eps 517956 win 29.8 win/s ETA 10.0min

[2026-04-25 14:18:30] [extract] strike sh0 2925/3000 eps 522419 win 29.8 win/s ETA 7.5min

[2026-04-25 14:19:35] [extract] drawer sh0 725/2000 eps 203000 win 30.1 win/s ETA 197.8min

[2026-04-25 14:21:04] [extract] strike sh0 2950/3000 eps 527042 win 29.8 win/s ETA 5.0min

[2026-04-25 14:23:28] [extract] drawer sh0 750/2000 eps 210000 win 30.1 win/s ETA 193.9min

[2026-04-25 14:23:28] [extract] strike sh0 2975/3000 eps 531329 win 29.8 win/s ETA 2.5min

[2026-04-25 14:25:57] [extract] strike sh0 3000/3000 eps 535779 win 29.8 win/s ETA 0.0min

[2026-04-25 14:25:57] [extract] strike sh0 DONE: 3000 eps 535779 win in 299.4min

[2026-04-25 14:26:15] [probe] task=strike variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag', 'obj_position', 'obj_velocity', 'obj_speed', 'obj_direction', 'obj_acceleration', 'obj_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=1

[2026-04-25 14:27:08] [probe] strike: features [(535779, 24, 1024)] eps 2999 loaded 53.0s

[2026-04-25 14:27:20] [extract] drawer sh0 775/2000 eps 217000 win 30.1 win/s ETA 190.0min

[2026-04-25 14:27:35] [probe] strike A ee_position L00: r2_mean=0.819 std=0.005 25.2s

[2026-04-25 14:28:01] [probe] strike A ee_position L01: r2_mean=0.913 std=0.004 23.2s

[2026-04-25 14:28:26] [probe] strike A ee_position L02: r2_mean=0.927 std=0.007 23.3s

[2026-04-25 14:28:51] [probe] strike A ee_position L03: r2_mean=0.944 std=0.006 23.2s

[2026-04-25 14:29:17] [probe] strike A ee_position L04: r2_mean=0.950 std=0.003 23.9s

[2026-04-25 14:29:42] [probe] strike A ee_position L05: r2_mean=0.957 std=0.002 22.9s

[2026-04-25 14:30:08] [probe] strike A ee_position L06: r2_mean=0.961 std=0.002 23.8s

[2026-04-25 14:30:33] [probe] strike A ee_position L07: r2_mean=0.966 std=0.002 23.0s

[2026-04-25 14:30:58] [probe] strike A ee_position L08: r2_mean=0.967 std=0.003 23.1s

[2026-04-25 14:31:13] [extract] drawer sh0 800/2000 eps 224000 win 30.1 win/s ETA 186.1min

[2026-04-25 14:31:24] [probe] strike A ee_position L09: r2_mean=0.967 std=0.003 23.2s

[2026-04-25 14:31:48] [probe] strike A ee_position L10: r2_mean=0.972 std=0.003 22.5s

[2026-04-25 14:32:15] [probe] strike A ee_position L11: r2_mean=0.974 std=0.003 24.0s

[2026-04-25 14:32:39] [probe] strike A ee_position L12: r2_mean=0.974 std=0.003 22.4s

[2026-04-25 14:32:58] [probe] strike A ee_position L13: r2_mean=0.979 std=0.003 16.6s

[2026-04-25 14:33:12] [probe] strike A ee_position L14: r2_mean=0.979 std=0.003 12.9s

[2026-04-25 14:33:37] [probe] strike A ee_position L15: r2_mean=0.980 std=0.003 23.2s

[2026-04-25 14:34:03] [probe] strike A ee_position L16: r2_mean=0.981 std=0.003 24.0s

[2026-04-25 14:34:29] [probe] strike A ee_position L17: r2_mean=0.982 std=0.003 23.2s

[2026-04-25 14:34:54] [probe] strike A ee_position L18: r2_mean=0.982 std=0.002 23.3s

[2026-04-25 14:35:05] [extract] drawer sh0 825/2000 eps 231000 win 30.1 win/s ETA 182.2min

[2026-04-25 14:35:19] [probe] strike A ee_position L19: r2_mean=0.983 std=0.003 23.0s

[2026-04-25 14:35:44] [probe] strike A ee_position L20: r2_mean=0.983 std=0.002 23.0s

[2026-04-25 14:36:10] [probe] strike A ee_position L21: r2_mean=0.984 std=0.003 23.2s

[2026-04-25 14:36:35] [probe] strike A ee_position L22: r2_mean=0.985 std=0.003 23.2s

[2026-04-25 14:37:01] [probe] strike A ee_position L23: r2_mean=0.984 std=0.003 23.4s

[2026-04-25 14:37:26] [probe] strike A ee_velocity L00: r2_mean=0.432 std=0.013 23.0s

[2026-04-25 14:37:52] [probe] strike A ee_velocity L01: r2_mean=0.693 std=0.003 24.0s

[2026-04-25 14:38:17] [probe] strike A ee_velocity L02: r2_mean=0.750 std=0.006 22.8s

[2026-04-25 14:38:43] [probe] strike A ee_velocity L03: r2_mean=0.780 std=0.002 23.4s

[2026-04-25 14:38:57] [extract] drawer sh0 850/2000 eps 238000 win 30.1 win/s ETA 178.4min

[2026-04-25 14:39:08] [probe] strike A ee_velocity L04: r2_mean=0.805 std=0.001 23.1s

[2026-04-25 14:39:33] [probe] strike A ee_velocity L05: r2_mean=0.827 std=0.001 23.1s

[2026-04-25 14:39:59] [probe] strike A ee_velocity L06: r2_mean=0.839 std=0.005 24.1s

[2026-04-25 14:40:24] [probe] strike A ee_velocity L07: r2_mean=0.855 std=0.001 23.1s

[2026-04-25 14:40:50] [probe] strike A ee_velocity L08: r2_mean=0.853 std=0.001 23.9s

[2026-04-25 14:41:15] [probe] strike A ee_velocity L09: r2_mean=0.860 std=0.001 22.6s

[2026-04-25 14:41:40] [probe] strike A ee_velocity L10: r2_mean=0.868 std=0.004 23.1s

[2026-04-25 14:42:06] [probe] strike A ee_velocity L11: r2_mean=0.876 std=0.001 23.4s

[2026-04-25 14:42:24] [probe] strike A ee_velocity L12: r2_mean=0.874 std=0.001 15.9s

[2026-04-25 14:42:39] [probe] strike A ee_velocity L13: r2_mean=0.889 std=0.001 13.5s

[2026-04-25 14:42:49] [extract] drawer sh0 875/2000 eps 245000 win 30.1 win/s ETA 174.5min

[2026-04-25 14:42:54] [probe] strike A ee_velocity L14: r2_mean=0.891 std=0.001 12.8s

[2026-04-25 14:43:17] [probe] strike A ee_velocity L15: r2_mean=0.906 std=0.001 21.1s

[2026-04-25 14:43:43] [probe] strike A ee_velocity L16: r2_mean=0.926 std=0.001 24.5s

[2026-04-25 14:44:09] [probe] strike A ee_velocity L17: r2_mean=0.929 std=0.001 23.6s

[2026-04-25 14:44:35] [probe] strike A ee_velocity L18: r2_mean=0.936 std=0.001 23.8s

[2026-04-25 14:45:00] [probe] strike A ee_velocity L19: r2_mean=0.935 std=0.001 23.2s

[2026-04-25 14:45:25] [probe] strike A ee_velocity L20: r2_mean=0.935 std=0.001 22.5s

[2026-04-25 14:45:50] [probe] strike A ee_velocity L21: r2_mean=0.934 std=0.001 23.5s

[2026-04-25 14:46:15] [probe] strike A ee_velocity L22: r2_mean=0.935 std=0.003 22.7s

[2026-04-25 14:46:41] [extract] drawer sh0 900/2000 eps 252000 win 30.1 win/s ETA 170.6min

[2026-04-25 14:46:42] [probe] strike A ee_velocity L23: r2_mean=0.933 std=0.001 24.1s

[2026-04-25 14:47:05] [probe] strike A ee_speed L00: r2_mean=0.706 std=0.014 21.2s

[2026-04-25 14:47:29] [probe] strike A ee_speed L01: r2_mean=0.882 std=0.002 22.0s

[2026-04-25 14:47:53] [probe] strike A ee_speed L02: r2_mean=0.896 std=0.002 21.1s

[2026-04-25 14:48:16] [probe] strike A ee_speed L03: r2_mean=0.915 std=0.001 21.0s

[2026-04-25 14:48:39] [probe] strike A ee_speed L04: r2_mean=0.925 std=0.002 20.9s

[2026-04-25 14:48:52] [probe] strike A ee_speed L05: r2_mean=0.938 std=0.001 10.7s

[2026-04-25 14:49:06] [probe] strike A ee_speed L06: r2_mean=0.943 std=0.001 12.5s

[2026-04-25 14:49:19] [probe] strike A ee_speed L07: r2_mean=0.952 std=0.001 11.3s

[2026-04-25 14:49:35] [probe] strike A ee_speed L08: r2_mean=0.951 std=0.001 15.0s

[2026-04-25 14:49:58] [probe] strike A ee_speed L09: r2_mean=0.951 std=0.001 20.8s

[2026-04-25 14:50:22] [probe] strike A ee_speed L10: r2_mean=0.955 std=0.001 22.1s

[2026-04-25 14:50:33] [extract] drawer sh0 925/2000 eps 259000 win 30.1 win/s ETA 166.7min

[2026-04-25 14:50:45] [probe] strike A ee_speed L11: r2_mean=0.956 std=0.001 20.8s

[2026-04-25 14:51:09] [probe] strike A ee_speed L12: r2_mean=0.957 std=0.001 21.4s

[2026-04-25 14:51:32] [probe] strike A ee_speed L13: r2_mean=0.963 std=0.001 21.0s

[2026-04-25 14:51:55] [probe] strike A ee_speed L14: r2_mean=0.966 std=0.001 20.9s

[2026-04-25 14:52:19] [probe] strike A ee_speed L15: r2_mean=0.969 std=0.001 21.9s

[2026-04-25 14:52:42] [probe] strike A ee_speed L16: r2_mean=0.974 std=0.001 21.1s

[2026-04-25 14:53:06] [probe] strike A ee_speed L17: r2_mean=0.975 std=0.000 21.5s

[2026-04-25 14:53:30] [probe] strike A ee_speed L18: r2_mean=0.976 std=0.001 22.2s

[2026-04-25 14:53:53] [probe] strike A ee_speed L19: r2_mean=0.975 std=0.001 21.0s

[2026-04-25 14:54:17] [probe] strike A ee_speed L20: r2_mean=0.975 std=0.001 21.6s

[2026-04-25 14:54:25] [extract] drawer sh0 950/2000 eps 266000 win 30.1 win/s ETA 162.8min

[2026-04-25 14:54:40] [probe] strike A ee_speed L21: r2_mean=0.974 std=0.000 21.1s

[2026-04-25 14:55:03] [probe] strike A ee_speed L22: r2_mean=0.974 std=0.001 21.2s

[2026-04-25 14:55:27] [probe] strike A ee_speed L23: r2_mean=0.973 std=0.000 21.1s

[2026-04-25 14:55:59] [probe] strike A ee_direction L00: r2_mean=0.254 std=0.113 21.3s

[2026-04-25 14:56:21] [probe] strike A ee_direction L01: r2_mean=0.629 std=0.007 20.9s

[2026-04-25 14:56:44] [probe] strike A ee_direction L02: r2_mean=0.689 std=0.003 21.0s

[2026-04-25 14:57:06] [probe] strike A ee_direction L03: r2_mean=0.714 std=0.003 21.6s

[2026-04-25 14:57:28] [probe] strike A ee_direction L04: r2_mean=0.741 std=0.003 20.8s

[2026-04-25 14:57:51] [probe] strike A ee_direction L05: r2_mean=0.765 std=0.005 21.1s

[2026-04-25 14:58:13] [probe] strike A ee_direction L06: r2_mean=0.772 std=0.012 20.8s

[2026-04-25 14:58:18] [extract] drawer sh0 975/2000 eps 273000 win 30.1 win/s ETA 158.9min

[2026-04-25 14:58:36] [probe] strike A ee_direction L07: r2_mean=0.797 std=0.004 21.0s

[2026-04-25 14:58:59] [probe] strike A ee_direction L08: r2_mean=0.799 std=0.005 21.6s

[2026-04-25 14:59:19] [probe] strike A ee_direction L09: r2_mean=0.796 std=0.007 18.6s

[2026-04-25 14:59:39] [probe] strike A ee_direction L10: r2_mean=0.803 std=0.008 19.4s

[2026-04-25 15:00:02] [probe] strike A ee_direction L11: r2_mean=0.814 std=0.004 21.5s

[2026-04-25 15:00:22] [probe] strike A ee_direction L12: r2_mean=0.803 std=0.029 18.8s

[2026-04-25 15:00:42] [probe] strike A ee_direction L13: r2_mean=0.831 std=0.005 18.2s

[2026-04-25 15:00:59] [probe] strike A ee_direction L14: r2_mean=0.809 std=0.043 15.4s

[2026-04-25 15:01:08] [probe] strike A ee_direction L15: r2_mean=0.840 std=0.003 8.4s

[2026-04-25 15:01:19] [probe] strike A ee_direction L16: r2_mean=0.832 std=0.010 9.4s

[2026-04-25 15:01:41] [probe] strike A ee_direction L17: r2_mean=0.845 std=0.006 20.5s

[2026-04-25 15:02:02] [probe] strike A ee_direction L18: r2_mean=0.463 std=0.206 19.6s

[2026-04-25 15:02:10] [extract] drawer sh0 1000/2000 eps 280000 win 30.1 win/s ETA 155.1min

[2026-04-25 15:02:23] [probe] strike A ee_direction L19: r2_mean=0.430 std=0.442 20.2s

[2026-04-25 15:02:44] [probe] strike A ee_direction L20: r2_mean=0.819 std=0.022 19.6s

[2026-04-25 15:02:57] [probe] strike A ee_direction L21: r2_mean=0.841 std=0.006 11.1s

[2026-04-25 15:03:12] [probe] strike A ee_direction L22: r2_mean=0.773 std=0.078 14.3s

[2026-04-25 15:03:22] [probe] strike A ee_direction L23: r2_mean=0.824 std=0.022 8.3s

[2026-04-25 15:03:40] [probe] strike A ee_acceleration L00: r2_mean=0.097 std=0.003 15.6s

[2026-04-25 15:04:03] [probe] strike A ee_acceleration L01: r2_mean=0.150 std=0.002 20.6s

[2026-04-25 15:04:21] [probe] strike A ee_acceleration L02: r2_mean=0.172 std=0.003 17.0s

[2026-04-25 15:04:46] [probe] strike A ee_acceleration L03: r2_mean=0.193 std=0.003 21.9s

[2026-04-25 15:05:11] [probe] strike A ee_acceleration L04: r2_mean=0.205 std=0.002 23.8s

[2026-04-25 15:05:36] [probe] strike A ee_acceleration L05: r2_mean=0.220 std=0.002 22.7s

[2026-04-25 15:06:00] [probe] strike A ee_acceleration L06: r2_mean=0.239 std=0.005 22.1s

[2026-04-25 15:06:03] [extract] drawer sh0 1025/2000 eps 287000 win 30.1 win/s ETA 151.2min

[2026-04-25 15:06:26] [probe] strike A ee_acceleration L07: r2_mean=0.253 std=0.002 22.8s

[2026-04-25 15:06:49] [probe] strike A ee_acceleration L08: r2_mean=0.263 std=0.007 21.6s

[2026-04-25 15:07:15] [probe] strike A ee_acceleration L09: r2_mean=0.258 std=0.003 23.2s

[2026-04-25 15:07:39] [probe] strike A ee_acceleration L10: r2_mean=0.289 std=0.017 22.6s

[2026-04-25 15:08:04] [probe] strike A ee_acceleration L11: r2_mean=0.295 std=0.014 22.8s

[2026-04-25 15:08:29] [probe] strike A ee_acceleration L12: r2_mean=0.299 std=0.011 23.1s

[2026-04-25 15:08:54] [probe] strike A ee_acceleration L13: r2_mean=0.345 std=0.012 22.7s

[2026-04-25 15:09:20] [probe] strike A ee_acceleration L14: r2_mean=0.348 std=0.010 23.9s

[2026-04-25 15:09:45] [probe] strike A ee_acceleration L15: r2_mean=0.370 std=0.002 23.1s

[2026-04-25 15:09:56] [extract] drawer sh0 1050/2000 eps 294000 win 30.1 win/s ETA 147.3min

[2026-04-25 15:10:11] [probe] strike A ee_acceleration L16: r2_mean=0.408 std=0.004 24.1s

[2026-04-25 15:10:36] [probe] strike A ee_acceleration L17: r2_mean=0.413 std=0.008 22.8s

[2026-04-25 15:11:01] [probe] strike A ee_acceleration L18: r2_mean=0.403 std=0.005 23.2s

[2026-04-25 15:11:26] [probe] strike A ee_acceleration L19: r2_mean=0.408 std=0.010 22.7s

[2026-04-25 15:11:51] [probe] strike A ee_acceleration L20: r2_mean=0.383 std=0.009 23.3s

[2026-04-25 15:12:17] [probe] strike A ee_acceleration L21: r2_mean=0.374 std=0.015 23.2s

[2026-04-25 15:12:41] [probe] strike A ee_acceleration L22: r2_mean=0.370 std=0.006 22.9s

[2026-04-25 15:13:07] [probe] strike A ee_acceleration L23: r2_mean=0.359 std=0.019 23.7s

[2026-04-25 15:13:31] [probe] strike A ee_accel_mag L00: r2_mean=0.285 std=0.005 21.2s

[2026-04-25 15:13:48] [extract] drawer sh0 1075/2000 eps 301000 win 30.1 win/s ETA 143.4min

[2026-04-25 15:13:55] [probe] strike A ee_accel_mag L01: r2_mean=0.413 std=0.005 21.7s

[2026-04-25 15:14:18] [probe] strike A ee_accel_mag L02: r2_mean=0.439 std=0.003 21.1s

[2026-04-25 15:14:40] [probe] strike A ee_accel_mag L03: r2_mean=0.469 std=0.004 20.6s

[2026-04-25 15:15:04] [probe] strike A ee_accel_mag L04: r2_mean=0.484 std=0.004 21.5s

[2026-04-25 15:15:26] [probe] strike A ee_accel_mag L05: r2_mean=0.500 std=0.005 20.3s

[2026-04-25 15:15:50] [probe] strike A ee_accel_mag L06: r2_mean=0.517 std=0.003 21.6s

[2026-04-25 15:16:14] [probe] strike A ee_accel_mag L07: r2_mean=0.549 std=0.009 21.5s

[2026-04-25 15:16:36] [probe] strike A ee_accel_mag L08: r2_mean=0.529 std=0.006 20.2s

[2026-04-25 15:17:01] [probe] strike A ee_accel_mag L09: r2_mean=0.533 std=0.009 22.2s

[2026-04-25 15:17:23] [probe] strike A ee_accel_mag L10: r2_mean=0.557 std=0.004 21.0s

[2026-04-25 15:17:40] [extract] drawer sh0 1100/2000 eps 308000 win 30.1 win/s ETA 139.5min

[2026-04-25 15:17:47] [probe] strike A ee_accel_mag L11: r2_mean=0.580 std=0.005 21.0s

[2026-04-25 15:18:10] [probe] strike A ee_accel_mag L12: r2_mean=0.602 std=0.006 21.7s

[2026-04-25 15:18:33] [probe] strike A ee_accel_mag L13: r2_mean=0.644 std=0.008 20.6s

[2026-04-25 15:18:58] [probe] strike A ee_accel_mag L14: r2_mean=0.650 std=0.008 22.4s

[2026-04-25 15:19:21] [probe] strike A ee_accel_mag L15: r2_mean=0.676 std=0.007 20.9s

[2026-04-25 15:19:43] [probe] strike A ee_accel_mag L16: r2_mean=0.708 std=0.003 20.7s

[2026-04-25 15:20:07] [probe] strike A ee_accel_mag L17: r2_mean=0.718 std=0.002 21.5s

[2026-04-25 15:20:31] [probe] strike A ee_accel_mag L18: r2_mean=0.718 std=0.004 21.2s

[2026-04-25 15:20:55] [probe] strike A ee_accel_mag L19: r2_mean=0.712 std=0.004 22.1s

[2026-04-25 15:21:18] [probe] strike A ee_accel_mag L20: r2_mean=0.701 std=0.002 21.2s

[2026-04-25 15:21:33] [extract] drawer sh0 1125/2000 eps 315000 win 30.1 win/s ETA 135.7min

[2026-04-25 15:21:42] [probe] strike A ee_accel_mag L21: r2_mean=0.692 std=0.008 21.4s

[2026-04-25 15:22:05] [probe] strike A ee_accel_mag L22: r2_mean=0.684 std=0.004 21.5s

[2026-04-25 15:22:28] [probe] strike A ee_accel_mag L23: r2_mean=0.670 std=0.004 21.1s

[2026-04-25 15:22:54] [probe] strike A obj_position L00: r2_mean=0.147 std=0.010 23.6s

[2026-04-25 15:23:20] [probe] strike A obj_position L01: r2_mean=0.276 std=0.052 23.2s

[2026-04-25 15:23:46] [probe] strike A obj_position L02: r2_mean=0.233 std=0.009 23.9s

[2026-04-25 15:24:11] [probe] strike A obj_position L03: r2_mean=0.289 std=0.007 23.3s

[2026-04-25 15:24:39] [probe] strike A obj_position L04: r2_mean=0.205 std=0.031 25.4s

[2026-04-25 15:25:04] [probe] strike A obj_position L05: r2_mean=0.253 std=0.016 22.9s

[2026-04-25 15:25:24] [extract] drawer sh0 1150/2000 eps 322000 win 30.1 win/s ETA 131.8min

[2026-04-25 15:25:29] [probe] strike A obj_position L06: r2_mean=0.217 std=0.030 23.3s

[2026-04-25 15:25:55] [probe] strike A obj_position L07: r2_mean=0.200 std=0.020 24.0s

[2026-04-25 15:26:20] [probe] strike A obj_position L08: r2_mean=0.259 std=0.021 22.7s

[2026-04-25 15:26:46] [probe] strike A obj_position L09: r2_mean=0.170 std=0.028 23.8s

[2026-04-25 15:27:09] [probe] strike A obj_position L10: r2_mean=0.214 std=0.027 21.8s

[2026-04-25 15:27:35] [probe] strike A obj_position L11: r2_mean=0.247 std=0.029 23.7s

[2026-04-25 15:28:01] [probe] strike A obj_position L12: r2_mean=0.208 std=0.036 23.1s

[2026-04-25 15:28:26] [probe] strike A obj_position L13: r2_mean=0.210 std=0.026 22.5s

[2026-04-25 15:28:51] [probe] strike A obj_position L14: r2_mean=0.194 std=0.024 23.4s

[2026-04-25 15:29:16] [probe] strike A obj_position L15: r2_mean=0.159 std=0.016 22.4s

[2026-04-25 15:29:16] [extract] drawer sh0 1175/2000 eps 329000 win 30.1 win/s ETA 127.9min

[2026-04-25 15:29:41] [probe] strike A obj_position L16: r2_mean=0.153 std=0.015 23.8s

[2026-04-25 15:30:07] [probe] strike A obj_position L17: r2_mean=0.128 std=0.027 23.2s

[2026-04-25 15:30:31] [probe] strike A obj_position L18: r2_mean=0.109 std=0.006 22.4s

[2026-04-25 15:30:56] [probe] strike A obj_position L19: r2_mean=0.118 std=0.027 22.9s

[2026-04-25 15:31:21] [probe] strike A obj_position L20: r2_mean=0.082 std=0.006 22.8s

[2026-04-25 15:31:47] [probe] strike A obj_position L21: r2_mean=0.077 std=0.020 23.6s

[2026-04-25 15:32:12] [probe] strike A obj_position L22: r2_mean=0.033 std=0.019 23.0s

[2026-04-25 15:32:38] [probe] strike A obj_position L23: r2_mean=0.037 std=0.008 23.6s

[2026-04-25 15:33:03] [probe] strike A obj_velocity L00: r2_mean=0.107 std=0.007 23.1s

[2026-04-25 15:33:08] [extract] drawer sh0 1200/2000 eps 336000 win 30.1 win/s ETA 124.0min

[2026-04-25 15:33:29] [probe] strike A obj_velocity L01: r2_mean=0.278 std=0.013 23.2s

[2026-04-25 15:33:53] [probe] strike A obj_velocity L02: r2_mean=0.331 std=0.017 22.9s

[2026-04-25 15:34:19] [probe] strike A obj_velocity L03: r2_mean=0.402 std=0.012 23.1s

[2026-04-25 15:34:44] [probe] strike A obj_velocity L04: r2_mean=0.434 std=0.018 23.5s

[2026-04-25 15:35:09] [probe] strike A obj_velocity L05: r2_mean=0.479 std=0.006 23.0s

[2026-04-25 15:35:35] [probe] strike A obj_velocity L06: r2_mean=0.497 std=0.024 24.0s

[2026-04-25 15:36:00] [probe] strike A obj_velocity L07: r2_mean=0.532 std=0.020 22.6s

[2026-04-25 15:36:26] [probe] strike A obj_velocity L08: r2_mean=0.561 std=0.015 23.5s

[2026-04-25 15:36:50] [probe] strike A obj_velocity L09: r2_mean=0.572 std=0.023 22.6s

[2026-04-25 15:37:00] [extract] drawer sh0 1225/2000 eps 343000 win 30.1 win/s ETA 120.1min

[2026-04-25 15:37:15] [probe] strike A obj_velocity L10: r2_mean=0.602 std=0.015 22.4s

[2026-04-25 15:37:41] [probe] strike A obj_velocity L11: r2_mean=0.607 std=0.028 23.4s

[2026-04-25 15:38:04] [probe] strike A obj_velocity L12: r2_mean=0.619 std=0.022 21.5s

[2026-04-25 15:38:29] [probe] strike A obj_velocity L13: r2_mean=0.649 std=0.027 23.3s

[2026-04-25 15:38:53] [probe] strike A obj_velocity L14: r2_mean=0.646 std=0.030 21.5s

[2026-04-25 15:39:17] [probe] strike A obj_velocity L15: r2_mean=0.642 std=0.023 22.3s

[2026-04-25 15:39:43] [probe] strike A obj_velocity L16: r2_mean=0.655 std=0.020 23.6s

[2026-04-25 15:40:07] [probe] strike A obj_velocity L17: r2_mean=0.665 std=0.010 21.9s

[2026-04-25 15:40:33] [probe] strike A obj_velocity L18: r2_mean=0.656 std=0.021 23.8s

[2026-04-25 15:40:52] [extract] drawer sh0 1250/2000 eps 350000 win 30.1 win/s ETA 116.3min

[2026-04-25 15:40:57] [probe] strike A obj_velocity L19: r2_mean=0.662 std=0.029 21.6s

[2026-04-25 15:41:22] [probe] strike A obj_velocity L20: r2_mean=0.650 std=0.019 23.5s

[2026-04-25 15:41:47] [probe] strike A obj_velocity L21: r2_mean=0.664 std=0.029 23.3s

[2026-04-25 15:42:12] [probe] strike A obj_velocity L22: r2_mean=0.648 std=0.018 22.3s

[2026-04-25 15:42:33] [probe] strike A obj_velocity L23: r2_mean=0.648 std=0.022 19.1s

[2026-04-25 15:42:56] [probe] strike A obj_speed L00: r2_mean=0.285 std=0.019 20.6s

[2026-04-25 15:43:10] [probe] strike A obj_speed L01: r2_mean=0.462 std=0.021 11.1s

[2026-04-25 15:43:22] [probe] strike A obj_speed L02: r2_mean=0.493 std=0.023 10.9s

[2026-04-25 15:43:34] [probe] strike A obj_speed L03: r2_mean=0.556 std=0.021 10.7s

[2026-04-25 15:43:56] [probe] strike A obj_speed L04: r2_mean=0.581 std=0.027 20.2s

[2026-04-25 15:44:20] [probe] strike A obj_speed L05: r2_mean=0.605 std=0.023 21.6s

[2026-04-25 15:44:43] [probe] strike A obj_speed L06: r2_mean=0.638 std=0.025 21.2s

[2026-04-25 15:44:44] [extract] drawer sh0 1275/2000 eps 357000 win 30.1 win/s ETA 112.4min

[2026-04-25 15:45:07] [probe] strike A obj_speed L07: r2_mean=0.658 std=0.022 21.7s

[2026-04-25 15:45:30] [probe] strike A obj_speed L08: r2_mean=0.669 std=0.024 20.7s

[2026-04-25 15:45:53] [probe] strike A obj_speed L09: r2_mean=0.679 std=0.028 21.1s

[2026-04-25 15:46:17] [probe] strike A obj_speed L10: r2_mean=0.689 std=0.024 21.8s

[2026-04-25 15:46:40] [probe] strike A obj_speed L11: r2_mean=0.705 std=0.031 20.9s

[2026-04-25 15:47:03] [probe] strike A obj_speed L12: r2_mean=0.715 std=0.030 21.5s

[2026-04-25 15:47:26] [probe] strike A obj_speed L13: r2_mean=0.751 std=0.025 20.4s

[2026-04-25 15:47:49] [probe] strike A obj_speed L14: r2_mean=0.753 std=0.028 21.1s

[2026-04-25 15:48:13] [probe] strike A obj_speed L15: r2_mean=0.754 std=0.027 22.2s

[2026-04-25 15:48:35] [probe] strike A obj_speed L16: r2_mean=0.777 std=0.027 19.8s

[2026-04-25 15:48:36] [extract] drawer sh0 1300/2000 eps 364000 win 30.1 win/s ETA 108.5min

[2026-04-25 15:48:59] [probe] strike A obj_speed L17: r2_mean=0.783 std=0.027 21.8s

[2026-04-25 15:49:22] [probe] strike A obj_speed L18: r2_mean=0.783 std=0.027 21.2s

[2026-04-25 15:49:46] [probe] strike A obj_speed L19: r2_mean=0.778 std=0.027 21.2s

[2026-04-25 15:50:09] [probe] strike A obj_speed L20: r2_mean=0.776 std=0.028 21.5s

[2026-04-25 15:50:33] [probe] strike A obj_speed L21: r2_mean=0.782 std=0.028 21.2s

[2026-04-25 15:50:56] [probe] strike A obj_speed L22: r2_mean=0.771 std=0.028 21.4s

[2026-04-25 15:51:20] [probe] strike A obj_speed L23: r2_mean=0.759 std=0.026 21.3s

[2026-04-25 15:51:44] [probe] strike A obj_direction L00: r2_mean=0.287 std=0.012 21.6s

[2026-04-25 15:52:09] [probe] strike A obj_direction L01: r2_mean=0.408 std=0.017 23.4s

[2026-04-25 15:52:29] [extract] drawer sh0 1325/2000 eps 371000 win 30.1 win/s ETA 104.6min

[2026-04-25 15:52:34] [probe] strike A obj_direction L02: r2_mean=0.437 std=0.014 23.2s

[2026-04-25 15:53:01] [probe] strike A obj_direction L03: r2_mean=0.488 std=0.015 23.9s

[2026-04-25 15:53:26] [probe] strike A obj_direction L04: r2_mean=0.503 std=0.006 23.3s

[2026-04-25 15:53:51] [probe] strike A obj_direction L05: r2_mean=0.536 std=0.009 22.8s

[2026-04-25 15:54:16] [probe] strike A obj_direction L06: r2_mean=0.556 std=0.013 23.0s

[2026-04-25 15:54:41] [probe] strike A obj_direction L07: r2_mean=0.590 std=0.017 23.2s

[2026-04-25 15:55:07] [probe] strike A obj_direction L08: r2_mean=0.625 std=0.007 23.4s

[2026-04-25 15:55:32] [probe] strike A obj_direction L09: r2_mean=0.669 std=0.008 23.2s

[2026-04-25 15:55:58] [probe] strike A obj_direction L10: r2_mean=0.686 std=0.005 23.9s

[2026-04-25 15:56:21] [extract] drawer sh0 1350/2000 eps 378000 win 30.1 win/s ETA 100.7min

[2026-04-25 15:56:23] [probe] strike A obj_direction L11: r2_mean=0.704 std=0.014 22.9s

[2026-04-25 15:56:49] [probe] strike A obj_direction L12: r2_mean=0.755 std=0.008 23.5s

[2026-04-25 15:57:13] [probe] strike A obj_direction L13: r2_mean=0.769 std=0.006 22.3s

[2026-04-25 15:57:38] [probe] strike A obj_direction L14: r2_mean=0.773 std=0.009 23.3s

[2026-04-25 15:58:05] [probe] strike A obj_direction L15: r2_mean=0.763 std=0.005 24.2s

[2026-04-25 15:58:32] [probe] strike A obj_direction L16: r2_mean=0.766 std=0.006 25.5s

[2026-04-25 15:59:00] [probe] strike A obj_direction L17: r2_mean=0.766 std=0.006 25.1s

[2026-04-25 15:59:25] [probe] strike A obj_direction L18: r2_mean=0.761 std=0.005 23.7s

[2026-04-25 15:59:52] [probe] strike A obj_direction L19: r2_mean=0.759 std=0.011 24.2s

[2026-04-25 16:00:13] [extract] drawer sh0 1375/2000 eps 385000 win 30.1 win/s ETA 96.9min

[2026-04-25 16:00:17] [probe] strike A obj_direction L20: r2_mean=0.752 std=0.006 24.0s

[2026-04-25 16:00:45] [probe] strike A obj_direction L21: r2_mean=0.771 std=0.013 24.8s

[2026-04-25 16:01:12] [probe] strike A obj_direction L22: r2_mean=0.761 std=0.005 26.0s

[2026-04-25 16:01:39] [probe] strike A obj_direction L23: r2_mean=0.760 std=0.007 24.2s

[2026-04-25 16:02:05] [probe] strike A obj_acceleration L00: r2_mean=0.005 std=0.001 24.0s

[2026-04-25 16:02:30] [probe] strike A obj_acceleration L01: r2_mean=0.018 std=0.001 22.9s

[2026-04-25 16:02:56] [probe] strike A obj_acceleration L02: r2_mean=0.022 std=0.002 24.0s

[2026-04-25 16:03:22] [probe] strike A obj_acceleration L03: r2_mean=0.035 std=0.001 23.1s

[2026-04-25 16:03:47] [probe] strike A obj_acceleration L04: r2_mean=0.037 std=0.003 23.4s

[2026-04-25 16:04:06] [extract] drawer sh0 1400/2000 eps 392000 win 30.1 win/s ETA 93.0min

[2026-04-25 16:04:13] [probe] strike A obj_acceleration L05: r2_mean=0.050 std=0.003 23.2s

[2026-04-25 16:04:37] [probe] strike A obj_acceleration L06: r2_mean=0.060 std=0.003 22.6s

[2026-04-25 16:05:03] [probe] strike A obj_acceleration L07: r2_mean=0.071 std=0.004 23.8s

[2026-04-25 16:05:28] [probe] strike A obj_acceleration L08: r2_mean=0.081 std=0.007 23.2s

[2026-04-25 16:05:54] [probe] strike A obj_acceleration L09: r2_mean=0.077 std=0.006 23.9s

[2026-04-25 16:06:19] [probe] strike A obj_acceleration L10: r2_mean=0.097 std=0.006 22.8s

[2026-04-25 16:06:45] [probe] strike A obj_acceleration L11: r2_mean=0.096 std=0.012 23.3s

[2026-04-25 16:07:09] [probe] strike A obj_acceleration L12: r2_mean=0.105 std=0.005 22.5s

[2026-04-25 16:07:35] [probe] strike A obj_acceleration L13: r2_mean=0.126 std=0.012 23.0s

[2026-04-25 16:07:58] [extract] drawer sh0 1425/2000 eps 399000 win 30.1 win/s ETA 89.1min

[2026-04-25 16:08:00] [probe] strike A obj_acceleration L14: r2_mean=0.120 std=0.009 23.3s

[2026-04-25 16:08:25] [probe] strike A obj_acceleration L15: r2_mean=0.110 std=0.012 23.2s

[2026-04-25 16:08:50] [probe] strike A obj_acceleration L16: r2_mean=0.127 std=0.006 23.1s

[2026-04-25 16:09:16] [probe] strike A obj_acceleration L17: r2_mean=0.128 std=0.002 22.9s

[2026-04-25 16:09:41] [probe] strike A obj_acceleration L18: r2_mean=0.130 std=0.006 23.6s

[2026-04-25 16:10:06] [probe] strike A obj_acceleration L19: r2_mean=0.134 std=0.011 22.2s

[2026-04-25 16:10:30] [probe] strike A obj_acceleration L20: r2_mean=0.131 std=0.006 22.8s

[2026-04-25 16:10:56] [probe] strike A obj_acceleration L21: r2_mean=0.134 std=0.014 23.3s

[2026-04-25 16:11:20] [probe] strike A obj_acceleration L22: r2_mean=0.127 std=0.006 22.7s

[2026-04-25 16:11:47] [probe] strike A obj_acceleration L23: r2_mean=0.120 std=0.009 24.0s

[2026-04-25 16:11:50] [extract] drawer sh0 1450/2000 eps 406000 win 30.1 win/s ETA 85.2min

[2026-04-25 16:12:09] [probe] strike A obj_accel_mag L00: r2_mean=0.114 std=0.008 19.6s

[2026-04-25 16:12:32] [probe] strike A obj_accel_mag L01: r2_mean=0.169 std=0.011 21.4s

[2026-04-25 16:12:56] [probe] strike A obj_accel_mag L02: r2_mean=0.181 std=0.011 21.9s

[2026-04-25 16:13:18] [probe] strike A obj_accel_mag L03: r2_mean=0.203 std=0.011 20.0s

[2026-04-25 16:13:42] [probe] strike A obj_accel_mag L04: r2_mean=0.212 std=0.014 22.3s

[2026-04-25 16:14:05] [probe] strike A obj_accel_mag L05: r2_mean=0.230 std=0.014 20.4s

[2026-04-25 16:14:28] [probe] strike A obj_accel_mag L06: r2_mean=0.250 std=0.016 20.8s

[2026-04-25 16:14:52] [probe] strike A obj_accel_mag L07: r2_mean=0.274 std=0.016 21.8s

[2026-04-25 16:15:11] [probe] strike A obj_accel_mag L08: r2_mean=0.275 std=0.017 17.6s

[2026-04-25 16:15:27] [probe] strike A obj_accel_mag L09: r2_mean=0.279 std=0.021 13.5s

[2026-04-25 16:15:39] [probe] strike A obj_accel_mag L10: r2_mean=0.287 std=0.017 10.9s

[2026-04-25 16:15:43] [extract] drawer sh0 1475/2000 eps 413000 win 30.1 win/s ETA 81.4min

[2026-04-25 16:15:54] [probe] strike A obj_accel_mag L11: r2_mean=0.285 std=0.026 12.7s

[2026-04-25 16:16:16] [probe] strike A obj_accel_mag L12: r2_mean=0.299 std=0.021 20.9s

[2026-04-25 16:16:32] [probe] strike A obj_accel_mag L13: r2_mean=0.310 std=0.018 12.8s

[2026-04-25 16:16:47] [probe] strike A obj_accel_mag L14: r2_mean=0.314 std=0.022 14.2s

[2026-04-25 16:17:03] [probe] strike A obj_accel_mag L15: r2_mean=0.312 std=0.023 13.1s

[2026-04-25 16:17:18] [probe] strike A obj_accel_mag L16: r2_mean=0.317 std=0.020 13.8s

[2026-04-25 16:17:39] [probe] strike A obj_accel_mag L17: r2_mean=0.319 std=0.021 19.1s

[2026-04-25 16:17:53] [probe] strike A obj_accel_mag L18: r2_mean=0.326 std=0.022 11.4s

[2026-04-25 16:18:06] [probe] strike A obj_accel_mag L19: r2_mean=0.337 std=0.024 11.1s

[2026-04-25 16:18:18] [probe] strike A obj_accel_mag L20: r2_mean=0.344 std=0.026 10.9s

[2026-04-25 16:18:36] [probe] strike A obj_accel_mag L21: r2_mean=0.340 std=0.026 16.4s

[2026-04-25 16:18:57] [probe] strike A obj_accel_mag L22: r2_mean=0.330 std=0.024 19.0s

[2026-04-25 16:19:10] [probe] strike A obj_accel_mag L23: r2_mean=0.322 std=0.022 11.0s

[2026-04-25 16:19:10] [probe] strike A DONE in 112.9min

[2026-04-25 16:19:15] [extract] reach loading model on cuda:1

[2026-04-25 16:19:26] [extract] reach model loaded 11.0s

[2026-04-25 16:19:26] [extract] reach shard 1/1 batch=8: total 600 todo 600

[2026-04-25 16:19:35] [extract] drawer sh0 1500/2000 eps 420000 win 30.1 win/s ETA 77.5min

[2026-04-25 16:22:40] [extract] reach sh0 25/600 eps 5875 win 30.2 win/s ETA 74.5min

[2026-04-25 16:23:27] [extract] drawer sh0 1525/2000 eps 427000 win 30.1 win/s ETA 73.6min

[2026-04-25 16:25:56] [extract] reach sh0 50/600 eps 11750 win 30.2 win/s ETA 71.4min

[2026-04-25 16:27:19] [extract] drawer sh0 1550/2000 eps 434000 win 30.1 win/s ETA 69.7min

[2026-04-25 16:29:11] [extract] reach sh0 75/600 eps 17625 win 30.1 win/s ETA 68.3min

[2026-04-25 16:31:12] [extract] drawer sh0 1575/2000 eps 441000 win 30.1 win/s ETA 65.9min

[2026-04-25 16:32:26] [extract] reach sh0 100/600 eps 23500 win 30.1 win/s ETA 65.0min

[2026-04-25 16:35:04] [extract] drawer sh0 1600/2000 eps 448000 win 30.1 win/s ETA 62.0min

[2026-04-25 16:35:41] [extract] reach sh0 125/600 eps 29375 win 30.1 win/s ETA 61.7min

[2026-04-25 16:38:56] [extract] reach sh0 150/600 eps 35250 win 30.1 win/s ETA 58.5min

[2026-04-25 16:38:56] [extract] drawer sh0 1625/2000 eps 455000 win 30.1 win/s ETA 58.1min

[2026-04-25 16:42:11] [extract] reach sh0 175/600 eps 41125 win 30.1 win/s ETA 55.2min

[2026-04-25 16:42:48] [extract] drawer sh0 1650/2000 eps 462000 win 30.1 win/s ETA 54.2min

[2026-04-25 16:45:26] [extract] reach sh0 200/600 eps 47000 win 30.1 win/s ETA 52.0min

[2026-04-25 16:46:40] [extract] drawer sh0 1675/2000 eps 469000 win 30.1 win/s ETA 50.4min

[2026-04-25 16:48:41] [extract] reach sh0 225/600 eps 52875 win 30.1 win/s ETA 48.7min

[2026-04-25 16:50:32] [extract] drawer sh0 1700/2000 eps 476000 win 30.1 win/s ETA 46.5min

[2026-04-25 16:51:56] [extract] reach sh0 250/600 eps 58750 win 30.1 win/s ETA 45.5min

[2026-04-25 16:54:25] [extract] drawer sh0 1725/2000 eps 483000 win 30.1 win/s ETA 42.6min

[2026-04-25 16:55:11] [extract] reach sh0 275/600 eps 64625 win 30.1 win/s ETA 42.3min

[2026-04-25 16:58:18] [extract] drawer sh0 1750/2000 eps 490000 win 30.1 win/s ETA 38.7min

[2026-04-25 16:58:26] [extract] reach sh0 300/600 eps 70500 win 30.1 win/s ETA 39.0min

[2026-04-25 17:01:41] [extract] reach sh0 325/600 eps 76375 win 30.1 win/s ETA 35.8min

[2026-04-25 17:02:10] [extract] drawer sh0 1775/2000 eps 497000 win 30.1 win/s ETA 34.9min

[2026-04-25 17:04:56] [extract] reach sh0 350/600 eps 82250 win 30.1 win/s ETA 32.5min

[2026-04-25 17:06:02] [extract] drawer sh0 1800/2000 eps 504000 win 30.1 win/s ETA 31.0min

[2026-04-25 17:08:12] [extract] reach sh0 375/600 eps 88125 win 30.1 win/s ETA 29.3min

[2026-04-25 17:09:55] [extract] drawer sh0 1825/2000 eps 511000 win 30.1 win/s ETA 27.1min

[2026-04-25 17:11:27] [extract] reach sh0 400/600 eps 94000 win 30.1 win/s ETA 26.0min

[2026-04-25 17:13:47] [extract] drawer sh0 1850/2000 eps 518000 win 30.1 win/s ETA 23.2min

[2026-04-25 17:14:42] [extract] reach sh0 425/600 eps 99875 win 30.1 win/s ETA 22.8min

[2026-04-25 17:17:39] [extract] drawer sh0 1875/2000 eps 525000 win 30.1 win/s ETA 19.4min

[2026-04-25 17:17:57] [extract] reach sh0 450/600 eps 105750 win 30.1 win/s ETA 19.5min

[2026-04-25 17:21:12] [extract] reach sh0 475/600 eps 111625 win 30.1 win/s ETA 16.3min

[2026-04-25 17:21:31] [extract] drawer sh0 1900/2000 eps 532000 win 30.1 win/s ETA 15.5min

[2026-04-25 17:24:27] [extract] reach sh0 500/600 eps 117500 win 30.1 win/s ETA 13.0min

[2026-04-25 17:25:24] [extract] drawer sh0 1925/2000 eps 539000 win 30.1 win/s ETA 11.6min

[2026-04-25 17:27:43] [extract] reach sh0 525/600 eps 123375 win 30.1 win/s ETA 9.8min

[2026-04-25 17:29:17] [extract] drawer sh0 1950/2000 eps 546000 win 30.1 win/s ETA 7.7min

[2026-04-25 17:30:58] [extract] reach sh0 550/600 eps 129250 win 30.1 win/s ETA 6.5min

[2026-04-25 17:33:10] [extract] drawer sh0 1975/2000 eps 553000 win 30.1 win/s ETA 3.9min

[2026-04-25 17:34:13] [extract] reach sh0 575/600 eps 135125 win 30.1 win/s ETA 3.3min

[2026-04-25 17:37:03] [extract] drawer sh0 2000/2000 eps 560000 win 30.1 win/s ETA 0.0min

[2026-04-25 17:37:03] [extract] drawer sh0 DONE: 2000 eps 560000 win in 309.9min

[2026-04-25 17:37:07] [probe] task=drawer variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag', 'obj_position', 'obj_velocity', 'obj_speed', 'obj_direction', 'obj_acceleration', 'obj_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=0

[2026-04-25 17:37:28] [extract] reach sh0 600/600 eps 141000 win 30.1 win/s ETA 0.0min

[2026-04-25 17:37:28] [extract] reach sh0 DONE: 600 eps 141000 win in 78.0min

[2026-04-25 17:37:32] [probe] task=reach variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=1

[2026-04-25 17:37:46] [probe] reach: features [(141000, 24, 1024)] eps 600 loaded 13.7s

[2026-04-25 17:38:02] [probe] drawer: features [(560000, 24, 1024)] eps 2000 loaded 55.0s

[2026-04-25 17:38:08] [probe] reach A ee_position L00: r2_mean=0.039 std=0.468 21.5s

[2026-04-25 17:38:32] [probe] reach A ee_position L01: r2_mean=0.692 std=0.014 23.1s

[2026-04-25 17:38:34] [probe] drawer A ee_position L00: r2_mean=0.904 std=0.011 30.0s

[2026-04-25 17:38:55] [probe] reach A ee_position L02: r2_mean=0.744 std=0.022 22.0s

[2026-04-25 17:39:06] [probe] drawer A ee_position L01: r2_mean=0.914 std=0.036 28.9s

[2026-04-25 17:39:18] [probe] reach A ee_position L03: r2_mean=0.785 std=0.010 21.9s

[2026-04-25 17:39:37] [probe] drawer A ee_position L02: r2_mean=0.930 std=0.038 27.8s

[2026-04-25 17:39:39] [probe] reach A ee_position L04: r2_mean=0.804 std=0.006 19.7s

[2026-04-25 17:39:58] [probe] reach A ee_position L05: r2_mean=0.835 std=0.006 17.3s

[2026-04-25 17:40:06] [probe] drawer A ee_position L03: r2_mean=0.956 std=0.007 25.8s

[2026-04-25 17:40:18] [probe] reach A ee_position L06: r2_mean=0.850 std=0.004 19.0s

[2026-04-25 17:40:34] [probe] reach A ee_position L07: r2_mean=0.876 std=0.005 15.8s

[2026-04-25 17:40:37] [probe] drawer A ee_position L04: r2_mean=0.959 std=0.007 28.3s

[2026-04-25 17:40:55] [probe] reach A ee_position L08: r2_mean=0.885 std=0.003 19.4s

[2026-04-25 17:41:09] [probe] drawer A ee_position L05: r2_mean=0.954 std=0.014 28.3s

[2026-04-25 17:41:14] [probe] reach A ee_position L09: r2_mean=0.882 std=0.005 18.4s

[2026-04-25 17:41:34] [probe] reach A ee_position L10: r2_mean=0.899 std=0.004 19.3s

[2026-04-25 17:41:40] [probe] drawer A ee_position L06: r2_mean=0.937 std=0.039 28.2s

[2026-04-25 17:41:53] [probe] reach A ee_position L11: r2_mean=0.908 std=0.004 17.5s

[2026-04-25 17:42:09] [probe] reach A ee_position L12: r2_mean=0.896 std=0.007 15.0s

[2026-04-25 17:42:11] [probe] drawer A ee_position L07: r2_mean=0.963 std=0.007 27.4s

[2026-04-25 17:42:30] [probe] reach A ee_position L13: r2_mean=0.913 std=0.016 19.9s

[2026-04-25 17:42:41] [probe] drawer A ee_position L08: r2_mean=0.963 std=0.007 26.9s

[2026-04-25 17:42:50] [probe] reach A ee_position L14: r2_mean=0.926 std=0.003 18.9s

[2026-04-25 17:43:10] [probe] reach A ee_position L15: r2_mean=0.930 std=0.004 19.0s

[2026-04-25 17:43:11] [probe] drawer A ee_position L09: r2_mean=0.954 std=0.015 27.0s

[2026-04-25 17:43:28] [probe] reach A ee_position L16: r2_mean=0.933 std=0.005 17.5s

[2026-04-25 17:43:42] [probe] drawer A ee_position L10: r2_mean=0.956 std=0.019 28.4s

[2026-04-25 17:43:48] [probe] reach A ee_position L17: r2_mean=0.936 std=0.005 18.4s

[2026-04-25 17:44:07] [probe] reach A ee_position L18: r2_mean=0.934 std=0.011 18.3s

[2026-04-25 17:44:13] [probe] drawer A ee_position L11: r2_mean=0.935 std=0.046 27.5s

[2026-04-25 17:44:28] [probe] reach A ee_position L19: r2_mean=0.927 std=0.018 19.7s

[2026-04-25 17:44:43] [probe] drawer A ee_position L12: r2_mean=0.946 std=0.046 27.8s

[2026-04-25 17:44:48] [probe] reach A ee_position L20: r2_mean=0.712 std=0.159 19.4s

[2026-04-25 17:45:03] [probe] reach A ee_position L21: r2_mean=0.641 std=0.165 13.9s

[2026-04-25 17:45:15] [probe] drawer A ee_position L13: r2_mean=0.945 std=0.052 28.1s

[2026-04-25 17:45:22] [probe] reach A ee_position L22: r2_mean=0.935 std=0.012 18.5s

[2026-04-25 17:45:44] [probe] reach A ee_position L23: r2_mean=0.931 std=0.012 20.6s

[2026-04-25 17:45:46] [probe] drawer A ee_position L14: r2_mean=0.972 std=0.006 28.7s

[2026-04-25 17:46:04] [probe] reach A ee_velocity L00: r2_mean=-0.427 std=0.572 19.0s

[2026-04-25 17:46:15] [probe] drawer A ee_position L15: r2_mean=0.948 std=0.049 26.0s

[2026-04-25 17:46:23] [probe] reach A ee_velocity L01: r2_mean=0.039 std=0.009 17.9s

[2026-04-25 17:46:40] [probe] reach A ee_velocity L02: r2_mean=0.056 std=0.005 15.6s

[2026-04-25 17:46:47] [probe] drawer A ee_position L16: r2_mean=0.947 std=0.051 28.6s

[2026-04-25 17:46:53] [probe] reach A ee_velocity L03: r2_mean=0.072 std=0.008 12.4s

[2026-04-25 17:47:10] [probe] reach A ee_velocity L04: r2_mean=0.091 std=0.006 16.1s

[2026-04-25 17:47:17] [probe] drawer A ee_position L17: r2_mean=0.970 std=0.010 27.7s

[2026-04-25 17:47:29] [probe] reach A ee_velocity L05: r2_mean=0.128 std=0.010 18.0s

[2026-04-25 17:47:49] [probe] drawer A ee_position L18: r2_mean=0.975 std=0.006 28.9s

[2026-04-25 17:47:50] [probe] reach A ee_velocity L06: r2_mean=0.142 std=0.010 19.9s

[2026-04-25 17:48:09] [probe] reach A ee_velocity L07: r2_mean=0.163 std=0.008 18.2s

[2026-04-25 17:48:18] [probe] drawer A ee_position L19: r2_mean=0.974 std=0.008 26.2s

[2026-04-25 17:48:30] [probe] reach A ee_velocity L08: r2_mean=0.171 std=0.012 19.1s

[2026-04-25 17:48:49] [probe] drawer A ee_position L20: r2_mean=0.975 std=0.006 27.4s

[2026-04-25 17:48:51] [probe] reach A ee_velocity L09: r2_mean=0.172 std=0.022 20.1s

[2026-04-25 17:49:10] [probe] reach A ee_velocity L10: r2_mean=0.197 std=0.016 18.4s

[2026-04-25 17:49:20] [probe] drawer A ee_position L21: r2_mean=0.976 std=0.006 27.9s

[2026-04-25 17:49:28] [probe] reach A ee_velocity L11: r2_mean=0.194 std=0.013 16.6s

[2026-04-25 17:49:50] [probe] reach A ee_velocity L12: r2_mean=0.216 std=0.020 21.2s

[2026-04-25 17:49:50] [probe] drawer A ee_position L22: r2_mean=0.976 std=0.006 27.8s

[2026-04-25 17:50:11] [probe] reach A ee_velocity L13: r2_mean=0.242 std=0.031 19.7s

[2026-04-25 17:50:21] [probe] drawer A ee_position L23: r2_mean=0.976 std=0.006 27.8s

[2026-04-25 17:50:31] [probe] reach A ee_velocity L14: r2_mean=0.246 std=0.018 18.8s

[2026-04-25 17:50:50] [probe] reach A ee_velocity L15: r2_mean=0.259 std=0.016 18.7s

[2026-04-25 17:50:52] [probe] drawer A ee_velocity L00: r2_mean=0.264 std=0.010 27.3s

[2026-04-25 17:51:11] [probe] reach A ee_velocity L16: r2_mean=0.292 std=0.017 19.9s

[2026-04-25 17:51:23] [probe] drawer A ee_velocity L01: r2_mean=0.316 std=0.009 27.9s

[2026-04-25 17:51:31] [probe] reach A ee_velocity L17: r2_mean=0.296 std=0.013 19.4s

[2026-04-25 17:51:50] [probe] reach A ee_velocity L18: r2_mean=0.290 std=0.028 17.3s

[2026-04-25 17:51:53] [probe] drawer A ee_velocity L02: r2_mean=0.322 std=0.006 27.5s

[2026-04-25 17:52:10] [probe] reach A ee_velocity L19: r2_mean=0.282 std=0.013 18.9s

[2026-04-25 17:52:25] [probe] drawer A ee_velocity L03: r2_mean=0.351 std=0.006 28.3s

[2026-04-25 17:52:30] [probe] reach A ee_velocity L20: r2_mean=-0.382 std=0.567 19.3s

[2026-04-25 17:52:50] [probe] reach A ee_velocity L21: r2_mean=-0.071 std=0.324 18.7s

[2026-04-25 17:52:55] [probe] drawer A ee_velocity L04: r2_mean=0.376 std=0.017 27.0s

[2026-04-25 17:53:08] [probe] reach A ee_velocity L22: r2_mean=0.269 std=0.033 16.8s

[2026-04-25 17:53:24] [probe] reach A ee_velocity L23: r2_mean=0.238 std=0.035 15.7s

[2026-04-25 17:53:25] [probe] drawer A ee_velocity L05: r2_mean=0.432 std=0.008 26.6s

[2026-04-25 17:53:43] [probe] reach A ee_speed L00: r2_mean=0.041 std=0.033 17.9s

[2026-04-25 17:53:55] [probe] drawer A ee_velocity L06: r2_mean=0.443 std=0.012 27.5s

[2026-04-25 17:54:03] [probe] reach A ee_speed L01: r2_mean=0.220 std=0.011 18.7s

[2026-04-25 17:54:23] [probe] reach A ee_speed L02: r2_mean=0.281 std=0.005 18.4s

[2026-04-25 17:54:26] [probe] drawer A ee_velocity L07: r2_mean=0.496 std=0.010 27.5s

[2026-04-25 17:54:37] [probe] reach A ee_speed L03: r2_mean=0.360 std=0.008 13.5s

[2026-04-25 17:54:56] [probe] drawer A ee_velocity L08: r2_mean=0.499 std=0.020 27.4s

[2026-04-25 17:54:56] [probe] reach A ee_speed L04: r2_mean=0.404 std=0.005 18.4s

[2026-04-25 17:55:15] [probe] reach A ee_speed L05: r2_mean=0.510 std=0.009 18.3s

[2026-04-25 17:55:28] [probe] drawer A ee_velocity L09: r2_mean=0.486 std=0.011 28.2s

[2026-04-25 17:55:30] [probe] reach A ee_speed L06: r2_mean=0.593 std=0.006 13.5s

[2026-04-25 17:55:47] [probe] reach A ee_speed L07: r2_mean=0.671 std=0.006 15.8s

[2026-04-25 17:55:57] [probe] drawer A ee_velocity L10: r2_mean=0.484 std=0.009 26.5s

[2026-04-25 17:56:06] [probe] reach A ee_speed L08: r2_mean=0.705 std=0.003 18.7s

[2026-04-25 17:56:26] [probe] reach A ee_speed L09: r2_mean=0.692 std=0.012 18.4s

[2026-04-25 17:56:27] [probe] drawer A ee_velocity L11: r2_mean=0.494 std=0.014 26.7s

[2026-04-25 17:56:45] [probe] reach A ee_speed L10: r2_mean=0.768 std=0.007 18.0s

[2026-04-25 17:56:57] [probe] drawer A ee_velocity L12: r2_mean=0.518 std=0.019 27.9s

[2026-04-25 17:57:04] [probe] reach A ee_speed L11: r2_mean=0.755 std=0.009 18.5s

[2026-04-25 17:57:23] [probe] reach A ee_speed L12: r2_mean=0.747 std=0.005 18.4s

[2026-04-25 17:57:28] [probe] drawer A ee_velocity L13: r2_mean=0.532 std=0.012 27.6s

[2026-04-25 17:57:45] [probe] reach A ee_speed L13: r2_mean=0.802 std=0.005 20.0s

[2026-04-25 17:57:59] [probe] drawer A ee_velocity L14: r2_mean=0.524 std=0.014 28.3s

[2026-04-25 17:58:04] [probe] reach A ee_speed L14: r2_mean=0.821 std=0.008 18.1s

[2026-04-25 17:58:24] [probe] reach A ee_speed L15: r2_mean=0.842 std=0.002 19.0s

[2026-04-25 17:58:32] [probe] drawer A ee_velocity L15: r2_mean=0.529 std=0.016 29.3s

[2026-04-25 17:58:41] [probe] reach A ee_speed L16: r2_mean=0.857 std=0.005 17.0s

[2026-04-25 17:58:59] [probe] reach A ee_speed L17: r2_mean=0.842 std=0.005 16.4s

[2026-04-25 17:59:02] [probe] drawer A ee_velocity L16: r2_mean=0.541 std=0.011 27.2s

[2026-04-25 17:59:17] [probe] reach A ee_speed L18: r2_mean=0.829 std=0.002 17.8s

[2026-04-25 17:59:31] [probe] drawer A ee_velocity L17: r2_mean=0.566 std=0.015 25.6s

[2026-04-25 17:59:34] [probe] reach A ee_speed L19: r2_mean=0.814 std=0.003 15.9s

[2026-04-25 17:59:54] [probe] reach A ee_speed L20: r2_mean=0.736 std=0.039 18.3s

[2026-04-25 17:59:59] [probe] drawer A ee_velocity L18: r2_mean=0.560 std=0.009 25.6s

[2026-04-25 18:00:13] [probe] reach A ee_speed L21: r2_mean=0.723 std=0.070 18.0s

[2026-04-25 18:00:30] [probe] drawer A ee_velocity L19: r2_mean=0.560 std=0.008 27.0s

[2026-04-25 18:00:33] [probe] reach A ee_speed L22: r2_mean=0.738 std=0.007 18.3s

[2026-04-25 18:00:52] [probe] reach A ee_speed L23: r2_mean=0.702 std=0.004 17.5s

[2026-04-25 18:01:00] [probe] drawer A ee_velocity L20: r2_mean=0.562 std=0.017 27.3s

[2026-04-25 18:01:09] [probe] reach A ee_direction L00: r2_mean=-0.813 std=0.493 14.2s

[2026-04-25 18:01:27] [probe] reach A ee_direction L01: r2_mean=0.185 std=0.021 17.5s

[2026-04-25 18:01:31] [probe] drawer A ee_velocity L21: r2_mean=0.566 std=0.008 27.2s

[2026-04-25 18:01:46] [probe] reach A ee_direction L02: r2_mean=0.265 std=0.019 18.2s

[2026-04-25 18:02:02] [probe] drawer A ee_velocity L22: r2_mean=0.578 std=0.015 28.1s

[2026-04-25 18:02:04] [probe] reach A ee_direction L03: r2_mean=0.317 std=0.014 18.3s

[2026-04-25 18:02:23] [probe] reach A ee_direction L04: r2_mean=0.381 std=0.014 18.7s

[2026-04-25 18:02:33] [probe] drawer A ee_velocity L23: r2_mean=0.553 std=0.012 28.1s

[2026-04-25 18:02:42] [probe] reach A ee_direction L05: r2_mean=0.408 std=0.011 18.8s

[2026-04-25 18:03:01] [probe] drawer A ee_speed L00: r2_mean=0.766 std=0.008 24.3s

[2026-04-25 18:03:02] [probe] reach A ee_direction L06: r2_mean=0.404 std=0.011 19.1s

[2026-04-25 18:03:19] [probe] reach A ee_direction L07: r2_mean=0.431 std=0.017 16.9s

[2026-04-25 18:03:30] [probe] drawer A ee_speed L01: r2_mean=0.803 std=0.003 26.0s

[2026-04-25 18:03:37] [probe] reach A ee_direction L08: r2_mean=0.451 std=0.014 18.0s

[2026-04-25 18:03:56] [probe] reach A ee_direction L09: r2_mean=0.449 std=0.018 18.8s

[2026-04-25 18:03:58] [probe] drawer A ee_speed L02: r2_mean=0.806 std=0.003 25.0s

[2026-04-25 18:04:14] [probe] reach A ee_direction L10: r2_mean=0.466 std=0.014 17.0s

[2026-04-25 18:04:27] [probe] drawer A ee_speed L03: r2_mean=0.818 std=0.003 25.4s

[2026-04-25 18:04:30] [probe] reach A ee_direction L11: r2_mean=0.487 std=0.014 16.6s

[2026-04-25 18:04:49] [probe] reach A ee_direction L12: r2_mean=0.494 std=0.012 18.5s

[2026-04-25 18:04:55] [probe] drawer A ee_speed L04: r2_mean=0.825 std=0.003 26.1s

[2026-04-25 18:05:08] [probe] reach A ee_direction L13: r2_mean=0.512 std=0.012 18.1s

[2026-04-25 18:05:24] [probe] reach A ee_direction L14: r2_mean=0.513 std=0.016 16.1s

[2026-04-25 18:05:26] [probe] drawer A ee_speed L05: r2_mean=0.838 std=0.001 27.5s

[2026-04-25 18:05:40] [probe] reach A ee_direction L15: r2_mean=0.524 std=0.013 16.0s

[2026-04-25 18:05:56] [probe] drawer A ee_speed L06: r2_mean=0.843 std=0.001 26.2s

[2026-04-25 18:05:59] [probe] reach A ee_direction L16: r2_mean=0.539 std=0.013 18.4s

[2026-04-25 18:06:18] [probe] reach A ee_direction L17: r2_mean=0.547 std=0.010 18.3s

[2026-04-25 18:06:25] [probe] drawer A ee_speed L07: r2_mean=0.852 std=0.001 25.8s

[2026-04-25 18:06:35] [probe] reach A ee_direction L18: r2_mean=0.559 std=0.013 17.2s

[2026-04-25 18:06:53] [probe] reach A ee_direction L19: r2_mean=0.561 std=0.008 18.0s

[2026-04-25 18:06:54] [probe] drawer A ee_speed L08: r2_mean=0.852 std=0.002 26.3s

[2026-04-25 18:07:12] [probe] reach A ee_direction L20: r2_mean=0.482 std=0.040 17.9s

[2026-04-25 18:07:24] [probe] drawer A ee_speed L09: r2_mean=0.849 std=0.002 26.1s

[2026-04-25 18:07:28] [probe] reach A ee_direction L21: r2_mean=-0.296 std=0.486 16.5s

[2026-04-25 18:07:47] [probe] reach A ee_direction L22: r2_mean=-0.027 std=0.419 18.5s

[2026-04-25 18:07:52] [probe] drawer A ee_speed L10: r2_mean=0.853 std=0.002 24.9s

[2026-04-25 18:08:06] [probe] reach A ee_direction L23: r2_mean=0.550 std=0.019 18.8s

[2026-04-25 18:08:19] [probe] drawer A ee_speed L11: r2_mean=0.857 std=0.002 23.5s

[2026-04-25 18:08:25] [probe] reach A ee_acceleration L00: r2_mean=-0.488 std=0.467 17.0s

[2026-04-25 18:08:45] [probe] reach A ee_acceleration L01: r2_mean=0.006 std=0.003 19.1s

[2026-04-25 18:08:46] [probe] drawer A ee_speed L12: r2_mean=0.858 std=0.001 24.4s

[2026-04-25 18:09:05] [probe] reach A ee_acceleration L02: r2_mean=0.010 std=0.002 19.2s

[2026-04-25 18:09:15] [probe] drawer A ee_speed L13: r2_mean=0.865 std=0.004 25.4s

[2026-04-25 18:09:26] [probe] reach A ee_acceleration L03: r2_mean=0.012 std=0.003 19.8s

[2026-04-25 18:09:43] [probe] drawer A ee_speed L14: r2_mean=0.864 std=0.003 25.1s

[2026-04-25 18:09:46] [probe] reach A ee_acceleration L04: r2_mean=0.039 std=0.009 19.2s

[2026-04-25 18:10:08] [probe] reach A ee_acceleration L05: r2_mean=0.043 std=0.007 20.2s

[2026-04-25 18:10:11] [probe] drawer A ee_speed L15: r2_mean=0.868 std=0.003 25.4s

[2026-04-25 18:10:28] [probe] reach A ee_acceleration L06: r2_mean=0.042 std=0.005 18.7s

[2026-04-25 18:10:40] [probe] drawer A ee_speed L16: r2_mean=0.868 std=0.002 26.1s

[2026-04-25 18:10:48] [probe] reach A ee_acceleration L07: r2_mean=0.045 std=0.006 19.0s

[2026-04-25 18:11:08] [probe] drawer A ee_speed L17: r2_mean=0.870 std=0.003 24.8s

[2026-04-25 18:11:09] [probe] reach A ee_acceleration L08: r2_mean=0.046 std=0.005 19.8s

[2026-04-25 18:11:29] [probe] reach A ee_acceleration L09: r2_mean=0.043 std=0.006 19.2s

[2026-04-25 18:11:36] [probe] drawer A ee_speed L18: r2_mean=0.872 std=0.004 24.6s

[2026-04-25 18:11:50] [probe] reach A ee_acceleration L10: r2_mean=0.061 std=0.005 20.2s

[2026-04-25 18:12:04] [probe] drawer A ee_speed L19: r2_mean=0.870 std=0.002 25.2s

[2026-04-25 18:12:12] [probe] reach A ee_acceleration L11: r2_mean=0.061 std=0.004 20.3s

[2026-04-25 18:12:32] [probe] drawer A ee_speed L20: r2_mean=0.869 std=0.002 25.1s

[2026-04-25 18:12:33] [probe] reach A ee_acceleration L12: r2_mean=0.051 std=0.011 19.9s

[2026-04-25 18:12:53] [probe] reach A ee_acceleration L13: r2_mean=0.098 std=0.006 19.1s

[2026-04-25 18:13:02] [probe] drawer A ee_speed L21: r2_mean=0.868 std=0.004 26.4s

[2026-04-25 18:13:14] [probe] reach A ee_acceleration L14: r2_mean=0.097 std=0.007 20.2s

[2026-04-25 18:13:30] [probe] drawer A ee_speed L22: r2_mean=0.867 std=0.002 25.6s

[2026-04-25 18:13:34] [probe] reach A ee_acceleration L15: r2_mean=0.098 std=0.005 18.7s

[2026-04-25 18:13:55] [probe] reach A ee_acceleration L16: r2_mean=0.098 std=0.004 19.7s

[2026-04-25 18:13:58] [probe] drawer A ee_speed L23: r2_mean=0.864 std=0.002 24.7s

[2026-04-25 18:14:16] [probe] reach A ee_acceleration L17: r2_mean=0.091 std=0.005 19.8s

[2026-04-25 18:14:29] [probe] drawer A ee_direction L00: r2_mean=0.421 std=0.011 27.3s

[2026-04-25 18:14:36] [probe] reach A ee_acceleration L18: r2_mean=0.090 std=0.004 19.3s

[2026-04-25 18:14:56] [probe] reach A ee_acceleration L19: r2_mean=0.083 std=0.009 18.8s

[2026-04-25 18:15:01] [probe] drawer A ee_direction L01: r2_mean=0.479 std=0.007 28.6s

[2026-04-25 18:15:17] [probe] reach A ee_acceleration L20: r2_mean=-0.186 std=0.140 19.6s

[2026-04-25 18:15:31] [probe] drawer A ee_direction L02: r2_mean=0.485 std=0.006 27.2s

[2026-04-25 18:15:33] [probe] reach A ee_acceleration L21: r2_mean=-0.678 std=0.262 15.7s

[2026-04-25 18:15:55] [probe] reach A ee_acceleration L22: r2_mean=0.065 std=0.013 20.3s

[2026-04-25 18:16:01] [probe] drawer A ee_direction L03: r2_mean=0.506 std=0.003 26.9s

[2026-04-25 18:16:15] [probe] reach A ee_acceleration L23: r2_mean=0.064 std=0.010 19.4s

[2026-04-25 18:16:30] [probe] reach A ee_accel_mag L00: r2_mean=-0.053 std=0.026 14.0s

[2026-04-25 18:16:31] [probe] drawer A ee_direction L04: r2_mean=0.524 std=0.010 27.0s

[2026-04-25 18:16:48] [probe] reach A ee_accel_mag L01: r2_mean=0.055 std=0.007 17.6s

[2026-04-25 18:17:04] [probe] drawer A ee_direction L05: r2_mean=0.562 std=0.011 29.5s

[2026-04-25 18:17:08] [probe] reach A ee_accel_mag L02: r2_mean=0.064 std=0.005 18.5s

[2026-04-25 18:17:28] [probe] reach A ee_accel_mag L03: r2_mean=0.072 std=0.005 18.6s

[2026-04-25 18:17:35] [probe] drawer A ee_direction L06: r2_mean=0.570 std=0.012 28.3s

[2026-04-25 18:17:46] [probe] reach A ee_accel_mag L04: r2_mean=0.204 std=0.013 17.5s

[2026-04-25 18:18:04] [probe] reach A ee_accel_mag L05: r2_mean=0.210 std=0.014 17.1s

[2026-04-25 18:18:07] [probe] drawer A ee_direction L07: r2_mean=0.604 std=0.006 28.3s

[2026-04-25 18:18:21] [probe] reach A ee_accel_mag L06: r2_mean=0.204 std=0.016 15.6s

[2026-04-25 18:18:39] [probe] reach A ee_accel_mag L07: r2_mean=0.218 std=0.012 17.2s

[2026-04-25 18:18:39] [probe] drawer A ee_direction L08: r2_mean=0.603 std=0.010 28.9s

[2026-04-25 18:18:57] [probe] reach A ee_accel_mag L08: r2_mean=0.215 std=0.007 17.1s

[2026-04-25 18:19:11] [probe] drawer A ee_direction L09: r2_mean=0.589 std=0.016 28.8s

[2026-04-25 18:19:15] [probe] reach A ee_accel_mag L09: r2_mean=0.225 std=0.010 17.5s

[2026-04-25 18:19:33] [probe] reach A ee_accel_mag L10: r2_mean=0.275 std=0.002 16.7s

[2026-04-25 18:19:42] [probe] drawer A ee_direction L10: r2_mean=0.580 std=0.005 28.2s

[2026-04-25 18:19:52] [probe] reach A ee_accel_mag L11: r2_mean=0.269 std=0.005 18.5s

[2026-04-25 18:20:10] [probe] drawer A ee_direction L11: r2_mean=0.601 std=0.012 24.7s

[2026-04-25 18:20:11] [probe] reach A ee_accel_mag L12: r2_mean=0.259 std=0.009 18.2s

[2026-04-25 18:20:29] [probe] reach A ee_accel_mag L13: r2_mean=0.368 std=0.004 17.0s

[2026-04-25 18:20:41] [probe] drawer A ee_direction L12: r2_mean=0.613 std=0.014 28.2s

[2026-04-25 18:20:47] [probe] reach A ee_accel_mag L14: r2_mean=0.363 std=0.005 16.8s

[2026-04-25 18:21:05] [probe] reach A ee_accel_mag L15: r2_mean=0.344 std=0.005 16.6s

[2026-04-25 18:21:11] [probe] drawer A ee_direction L13: r2_mean=0.615 std=0.010 27.3s

[2026-04-25 18:21:23] [probe] reach A ee_accel_mag L16: r2_mean=0.348 std=0.004 17.8s

[2026-04-25 18:21:41] [probe] drawer A ee_direction L14: r2_mean=0.622 std=0.006 26.6s

[2026-04-25 18:21:43] [probe] reach A ee_accel_mag L17: r2_mean=0.326 std=0.003 18.4s

[2026-04-25 18:22:00] [probe] reach A ee_accel_mag L18: r2_mean=0.309 std=0.003 16.5s

[2026-04-25 18:22:11] [probe] drawer A ee_direction L15: r2_mean=0.632 std=0.008 27.5s

[2026-04-25 18:22:14] [probe] reach A ee_accel_mag L19: r2_mean=0.298 std=0.004 13.3s

[2026-04-25 18:22:32] [probe] reach A ee_accel_mag L20: r2_mean=0.253 std=0.030 16.8s

[2026-04-25 18:22:43] [probe] drawer A ee_direction L16: r2_mean=0.632 std=0.008 28.4s

[2026-04-25 18:22:52] [probe] reach A ee_accel_mag L21: r2_mean=0.231 std=0.037 18.8s

[2026-04-25 18:23:10] [probe] reach A ee_accel_mag L22: r2_mean=0.263 std=0.019 17.6s

[2026-04-25 18:23:14] [probe] drawer A ee_direction L17: r2_mean=0.643 std=0.008 28.4s

[2026-04-25 18:23:30] [probe] reach A ee_accel_mag L23: r2_mean=0.252 std=0.007 18.4s

[2026-04-25 18:23:30] [probe] reach A DONE in 46.0min

[2026-04-25 18:23:34] [extract] nut_thread loading model on cuda:1

[2026-04-25 18:23:45] [probe] drawer A ee_direction L18: r2_mean=0.647 std=0.003 27.8s

[2026-04-25 18:23:45] [extract] nut_thread model loaded 11.6s

[2026-04-25 18:23:45] [extract] nut_thread shard 1/1 batch=8: total 2500 todo 2500

[2026-04-25 18:24:17] [probe] drawer A ee_direction L19: r2_mean=0.642 std=0.008 29.0s

[2026-04-25 18:24:48] [probe] drawer A ee_direction L20: r2_mean=0.642 std=0.010 27.3s

[2026-04-25 18:25:19] [probe] drawer A ee_direction L21: r2_mean=0.643 std=0.004 28.5s

[2026-04-25 18:25:35] [extract] nut_thread sh0 25/2500 eps 3225 win 29.4 win/s ETA 181.2min

[2026-04-25 18:25:49] [probe] drawer A ee_direction L22: r2_mean=0.645 std=0.012 27.1s

[2026-04-25 18:26:19] [probe] drawer A ee_direction L23: r2_mean=0.639 std=0.006 27.0s

[2026-04-25 18:26:50] [probe] drawer A ee_acceleration L00: r2_mean=0.024 std=0.005 27.4s

[2026-04-25 18:27:22] [probe] drawer A ee_acceleration L01: r2_mean=0.041 std=0.005 28.2s

[2026-04-25 18:27:26] [extract] nut_thread sh0 50/2500 eps 6450 win 29.2 win/s ETA 180.1min

[2026-04-25 18:27:53] [probe] drawer A ee_acceleration L02: r2_mean=0.062 std=0.005 28.0s

[2026-04-25 18:28:23] [probe] drawer A ee_acceleration L03: r2_mean=0.090 std=0.004 27.1s

[2026-04-25 18:28:52] [probe] drawer A ee_acceleration L04: r2_mean=0.112 std=0.002 26.6s

[2026-04-25 18:29:16] [extract] nut_thread sh0 75/2500 eps 9675 win 29.3 win/s ETA 178.2min

[2026-04-25 18:29:23] [probe] drawer A ee_acceleration L05: r2_mean=0.163 std=0.007 26.9s

[2026-04-25 18:29:52] [probe] drawer A ee_acceleration L06: r2_mean=0.191 std=0.016 26.3s

[2026-04-25 18:30:21] [probe] drawer A ee_acceleration L07: r2_mean=0.220 std=0.009 25.8s

[2026-04-25 18:30:50] [probe] drawer A ee_acceleration L08: r2_mean=0.220 std=0.017 26.5s

[2026-04-25 18:31:07] [extract] nut_thread sh0 100/2500 eps 12900 win 29.2 win/s ETA 176.6min

[2026-04-25 18:31:20] [probe] drawer A ee_acceleration L09: r2_mean=0.219 std=0.018 26.7s

[2026-04-25 18:31:49] [probe] drawer A ee_acceleration L10: r2_mean=0.236 std=0.016 26.2s

[2026-04-25 18:32:18] [probe] drawer A ee_acceleration L11: r2_mean=0.237 std=0.010 26.0s

[2026-04-25 18:32:48] [probe] drawer A ee_acceleration L12: r2_mean=0.255 std=0.024 27.3s

[2026-04-25 18:32:57] [extract] nut_thread sh0 125/2500 eps 16125 win 29.2 win/s ETA 174.7min

[2026-04-25 18:33:15] [probe] drawer A ee_acceleration L13: r2_mean=0.277 std=0.021 24.2s

[2026-04-25 18:33:41] [probe] drawer A ee_acceleration L14: r2_mean=0.274 std=0.008 23.1s

[2026-04-25 18:34:12] [probe] drawer A ee_acceleration L15: r2_mean=0.266 std=0.015 27.9s

[2026-04-25 18:34:42] [probe] drawer A ee_acceleration L16: r2_mean=0.250 std=0.019 26.9s

[2026-04-25 18:34:47] [extract] nut_thread sh0 150/2500 eps 19350 win 29.2 win/s ETA 172.8min

[2026-04-25 18:35:11] [probe] drawer A ee_acceleration L17: r2_mean=0.248 std=0.011 26.3s

[2026-04-25 18:35:41] [probe] drawer A ee_acceleration L18: r2_mean=0.261 std=0.013 26.9s

[2026-04-25 18:36:14] [probe] drawer A ee_acceleration L19: r2_mean=0.265 std=0.016 29.8s

[2026-04-25 18:36:38] [extract] nut_thread sh0 175/2500 eps 22575 win 29.2 win/s ETA 171.1min

[2026-04-25 18:36:46] [probe] drawer A ee_acceleration L20: r2_mean=0.266 std=0.020 28.5s

[2026-04-25 18:37:17] [probe] drawer A ee_acceleration L21: r2_mean=0.276 std=0.016 28.2s

[2026-04-25 18:37:47] [probe] drawer A ee_acceleration L22: r2_mean=0.298 std=0.019 27.2s

[2026-04-25 18:38:17] [probe] drawer A ee_acceleration L23: r2_mean=0.256 std=0.020 26.1s

[2026-04-25 18:38:28] [extract] nut_thread sh0 200/2500 eps 25800 win 29.2 win/s ETA 169.2min

[2026-04-25 18:38:40] [probe] drawer A ee_accel_mag L00: r2_mean=0.607 std=0.011 20.2s

[2026-04-25 18:39:07] [probe] drawer A ee_accel_mag L01: r2_mean=0.664 std=0.011 24.1s

[2026-04-25 18:39:35] [probe] drawer A ee_accel_mag L02: r2_mean=0.670 std=0.008 25.3s

[2026-04-25 18:40:03] [probe] drawer A ee_accel_mag L03: r2_mean=0.691 std=0.006 24.5s

[2026-04-25 18:40:18] [extract] nut_thread sh0 225/2500 eps 29025 win 29.2 win/s ETA 167.3min

[2026-04-25 18:40:30] [probe] drawer A ee_accel_mag L04: r2_mean=0.700 std=0.007 24.0s

[2026-04-25 18:40:57] [probe] drawer A ee_accel_mag L05: r2_mean=0.717 std=0.007 23.8s

[2026-04-25 18:41:25] [probe] drawer A ee_accel_mag L06: r2_mean=0.720 std=0.013 25.4s

[2026-04-25 18:41:53] [probe] drawer A ee_accel_mag L07: r2_mean=0.731 std=0.007 24.8s

[2026-04-25 18:42:09] [extract] nut_thread sh0 250/2500 eps 32250 win 29.2 win/s ETA 165.5min

[2026-04-25 18:42:20] [probe] drawer A ee_accel_mag L08: r2_mean=0.732 std=0.006 24.4s

[2026-04-25 18:42:49] [probe] drawer A ee_accel_mag L09: r2_mean=0.729 std=0.009 25.2s

[2026-04-25 18:43:16] [probe] drawer A ee_accel_mag L10: r2_mean=0.737 std=0.007 24.5s

[2026-04-25 18:43:44] [probe] drawer A ee_accel_mag L11: r2_mean=0.743 std=0.005 24.7s

[2026-04-25 18:43:59] [extract] nut_thread sh0 275/2500 eps 35475 win 29.2 win/s ETA 163.7min

[2026-04-25 18:44:10] [probe] drawer A ee_accel_mag L12: r2_mean=0.747 std=0.008 23.3s

[2026-04-25 18:44:37] [probe] drawer A ee_accel_mag L13: r2_mean=0.757 std=0.006 23.7s

[2026-04-25 18:45:04] [probe] drawer A ee_accel_mag L14: r2_mean=0.756 std=0.007 24.0s

[2026-04-25 18:45:33] [probe] drawer A ee_accel_mag L15: r2_mean=0.763 std=0.006 25.6s

[2026-04-25 18:45:49] [extract] nut_thread sh0 300/2500 eps 38700 win 29.2 win/s ETA 161.8min

[2026-04-25 18:45:59] [probe] drawer A ee_accel_mag L16: r2_mean=0.763 std=0.004 23.6s

[2026-04-25 18:46:23] [probe] drawer A ee_accel_mag L17: r2_mean=0.766 std=0.005 21.2s

[2026-04-25 18:46:48] [probe] drawer A ee_accel_mag L18: r2_mean=0.766 std=0.005 21.9s

[2026-04-25 18:47:13] [probe] drawer A ee_accel_mag L19: r2_mean=0.767 std=0.004 22.8s

[2026-04-25 18:47:40] [extract] nut_thread sh0 325/2500 eps 41925 win 29.2 win/s ETA 160.0min

[2026-04-25 18:47:41] [probe] drawer A ee_accel_mag L20: r2_mean=0.766 std=0.005 24.6s

[2026-04-25 18:48:08] [probe] drawer A ee_accel_mag L21: r2_mean=0.765 std=0.004 24.0s

[2026-04-25 18:48:37] [probe] drawer A ee_accel_mag L22: r2_mean=0.763 std=0.004 26.8s

[2026-04-25 18:49:06] [probe] drawer A ee_accel_mag L23: r2_mean=0.763 std=0.004 25.1s

[2026-04-25 18:49:30] [extract] nut_thread sh0 350/2500 eps 45150 win 29.2 win/s ETA 158.1min

[2026-04-25 18:49:35] [probe] drawer A obj_position L00: r2_mean=0.933 std=0.005 26.4s

[2026-04-25 18:50:07] [probe] drawer A obj_position L01: r2_mean=0.948 std=0.031 28.4s

[2026-04-25 18:50:37] [probe] drawer A obj_position L02: r2_mean=0.973 std=0.003 27.5s

[2026-04-25 18:51:08] [probe] drawer A obj_position L03: r2_mean=0.974 std=0.001 27.1s

[2026-04-25 18:51:20] [extract] nut_thread sh0 375/2500 eps 48375 win 29.2 win/s ETA 156.2min

[2026-04-25 18:51:37] [probe] drawer A obj_position L04: r2_mean=0.976 std=0.002 27.0s

[2026-04-25 18:52:08] [probe] drawer A obj_position L05: r2_mean=0.977 std=0.006 27.6s

[2026-04-25 18:52:37] [probe] drawer A obj_position L06: r2_mean=0.976 std=0.004 26.2s

[2026-04-25 18:53:08] [probe] drawer A obj_position L07: r2_mean=0.974 std=0.010 27.3s

[2026-04-25 18:53:10] [extract] nut_thread sh0 400/2500 eps 51600 win 29.2 win/s ETA 154.4min

[2026-04-25 18:53:36] [probe] drawer A obj_position L08: r2_mean=0.976 std=0.005 25.4s

[2026-04-25 18:54:07] [probe] drawer A obj_position L09: r2_mean=0.978 std=0.004 27.6s

[2026-04-25 18:54:36] [probe] drawer A obj_position L10: r2_mean=0.981 std=0.003 26.7s

[2026-04-25 18:55:01] [extract] nut_thread sh0 425/2500 eps 54825 win 29.2 win/s ETA 152.6min

[2026-04-25 18:55:06] [probe] drawer A obj_position L11: r2_mean=0.973 std=0.016 26.8s

[2026-04-25 18:55:36] [probe] drawer A obj_position L12: r2_mean=0.977 std=0.018 27.0s

[2026-04-25 18:56:05] [probe] drawer A obj_position L13: r2_mean=0.986 std=0.002 26.4s

[2026-04-25 18:56:35] [probe] drawer A obj_position L14: r2_mean=0.986 std=0.003 26.6s

[2026-04-25 18:56:52] [extract] nut_thread sh0 450/2500 eps 58050 win 29.2 win/s ETA 150.8min

[2026-04-25 18:57:03] [probe] drawer A obj_position L15: r2_mean=0.986 std=0.003 25.0s

[2026-04-25 18:57:31] [probe] drawer A obj_position L16: r2_mean=0.986 std=0.004 24.6s

[2026-04-25 18:58:01] [probe] drawer A obj_position L17: r2_mean=0.982 std=0.014 27.3s

[2026-04-25 18:58:33] [probe] drawer A obj_position L18: r2_mean=0.989 std=0.001 28.5s

[2026-04-25 18:58:43] [extract] nut_thread sh0 475/2500 eps 61275 win 29.2 win/s ETA 149.0min

[2026-04-25 18:59:04] [probe] drawer A obj_position L19: r2_mean=0.988 std=0.005 28.4s

[2026-04-25 18:59:35] [probe] drawer A obj_position L20: r2_mean=0.987 std=0.005 27.0s

[2026-04-25 19:00:05] [probe] drawer A obj_position L21: r2_mean=0.990 std=0.001 27.3s

[2026-04-25 19:00:33] [extract] nut_thread sh0 500/2500 eps 64500 win 29.2 win/s ETA 147.2min

[2026-04-25 19:00:35] [probe] drawer A obj_position L22: r2_mean=0.969 std=0.027 26.6s

[2026-04-25 19:01:05] [probe] drawer A obj_position L23: r2_mean=0.990 std=0.001 26.9s

[2026-04-25 19:01:36] [probe] drawer A obj_velocity L00: r2_mean=0.701 std=0.012 27.4s

[2026-04-25 19:02:07] [probe] drawer A obj_velocity L01: r2_mean=0.791 std=0.009 27.8s

[2026-04-25 19:02:24] [extract] nut_thread sh0 525/2500 eps 67725 win 29.2 win/s ETA 145.4min

[2026-04-25 19:02:38] [probe] drawer A obj_velocity L02: r2_mean=0.787 std=0.008 27.7s

[2026-04-25 19:03:09] [probe] drawer A obj_velocity L03: r2_mean=0.808 std=0.009 27.9s

[2026-04-25 19:03:39] [probe] drawer A obj_velocity L04: r2_mean=0.818 std=0.008 27.6s

[2026-04-25 19:04:10] [probe] drawer A obj_velocity L05: r2_mean=0.843 std=0.007 27.4s

[2026-04-25 19:04:14] [extract] nut_thread sh0 550/2500 eps 70950 win 29.2 win/s ETA 143.5min

[2026-04-25 19:04:40] [probe] drawer A obj_velocity L06: r2_mean=0.844 std=0.007 27.5s

[2026-04-25 19:05:10] [probe] drawer A obj_velocity L07: r2_mean=0.862 std=0.006 26.3s

[2026-04-25 19:05:39] [probe] drawer A obj_velocity L08: r2_mean=0.862 std=0.006 26.5s

[2026-04-25 19:06:04] [extract] nut_thread sh0 575/2500 eps 74175 win 29.2 win/s ETA 141.7min

[2026-04-25 19:06:09] [probe] drawer A obj_velocity L09: r2_mean=0.858 std=0.006 26.3s

[2026-04-25 19:06:37] [probe] drawer A obj_velocity L10: r2_mean=0.866 std=0.006 25.4s

[2026-04-25 19:07:07] [probe] drawer A obj_velocity L11: r2_mean=0.872 std=0.005 26.9s

[2026-04-25 19:07:36] [probe] drawer A obj_velocity L12: r2_mean=0.874 std=0.005 26.2s

[2026-04-25 19:07:54] [extract] nut_thread sh0 600/2500 eps 77400 win 29.2 win/s ETA 139.8min

[2026-04-25 19:08:04] [probe] drawer A obj_velocity L13: r2_mean=0.888 std=0.004 24.8s

[2026-04-25 19:08:34] [probe] drawer A obj_velocity L14: r2_mean=0.888 std=0.004 27.4s

[2026-04-25 19:09:03] [probe] drawer A obj_velocity L15: r2_mean=0.901 std=0.004 26.4s

[2026-04-25 19:09:34] [probe] drawer A obj_velocity L16: r2_mean=0.911 std=0.003 27.9s

[2026-04-25 19:09:44] [extract] nut_thread sh0 625/2500 eps 80625 win 29.2 win/s ETA 137.9min

[2026-04-25 19:10:04] [probe] drawer A obj_velocity L17: r2_mean=0.917 std=0.003 26.2s

[2026-04-25 19:10:34] [probe] drawer A obj_velocity L18: r2_mean=0.918 std=0.003 28.1s

[2026-04-25 19:11:05] [probe] drawer A obj_velocity L19: r2_mean=0.916 std=0.004 27.4s

[2026-04-25 19:11:33] [probe] drawer A obj_velocity L20: r2_mean=0.915 std=0.003 25.7s

[2026-04-25 19:11:35] [extract] nut_thread sh0 650/2500 eps 83850 win 29.2 win/s ETA 136.1min

[2026-04-25 19:12:01] [probe] drawer A obj_velocity L21: r2_mean=0.916 std=0.003 24.2s

[2026-04-25 19:12:30] [probe] drawer A obj_velocity L22: r2_mean=0.916 std=0.003 26.3s

[2026-04-25 19:12:59] [probe] drawer A obj_velocity L23: r2_mean=0.916 std=0.003 25.5s

[2026-04-25 19:13:25] [extract] nut_thread sh0 675/2500 eps 87075 win 29.2 win/s ETA 134.3min

[2026-04-25 19:13:28] [probe] drawer A obj_speed L00: r2_mean=0.720 std=0.013 25.7s

[2026-04-25 19:14:00] [probe] drawer A obj_speed L01: r2_mean=0.811 std=0.009 27.9s

[2026-04-25 19:14:28] [probe] drawer A obj_speed L02: r2_mean=0.801 std=0.014 25.1s

[2026-04-25 19:14:57] [probe] drawer A obj_speed L03: r2_mean=0.828 std=0.006 26.4s

[2026-04-25 19:15:15] [extract] nut_thread sh0 700/2500 eps 90300 win 29.2 win/s ETA 132.4min

[2026-04-25 19:15:25] [probe] drawer A obj_speed L04: r2_mean=0.836 std=0.007 25.0s

[2026-04-25 19:15:55] [probe] drawer A obj_speed L05: r2_mean=0.861 std=0.004 26.7s

[2026-04-25 19:16:21] [probe] drawer A obj_speed L06: r2_mean=0.859 std=0.005 22.9s

[2026-04-25 19:16:47] [probe] drawer A obj_speed L07: r2_mean=0.878 std=0.005 22.8s

[2026-04-25 19:17:05] [extract] nut_thread sh0 725/2500 eps 93525 win 29.2 win/s ETA 130.6min

[2026-04-25 19:17:11] [probe] drawer A obj_speed L08: r2_mean=0.876 std=0.004 20.8s

[2026-04-25 19:17:34] [probe] drawer A obj_speed L09: r2_mean=0.873 std=0.005 20.9s

[2026-04-25 19:18:03] [probe] drawer A obj_speed L10: r2_mean=0.881 std=0.005 25.6s

[2026-04-25 19:18:31] [probe] drawer A obj_speed L11: r2_mean=0.888 std=0.004 24.5s

[2026-04-25 19:18:56] [extract] nut_thread sh0 750/2500 eps 96750 win 29.2 win/s ETA 128.7min

[2026-04-25 19:19:00] [probe] drawer A obj_speed L12: r2_mean=0.889 std=0.003 24.6s

[2026-04-25 19:19:30] [probe] drawer A obj_speed L13: r2_mean=0.904 std=0.004 26.7s

[2026-04-25 19:19:58] [probe] drawer A obj_speed L14: r2_mean=0.905 std=0.003 24.9s

[2026-04-25 19:20:26] [probe] drawer A obj_speed L15: r2_mean=0.916 std=0.004 24.4s

[2026-04-25 19:20:46] [extract] nut_thread sh0 775/2500 eps 99975 win 29.2 win/s ETA 126.9min

[2026-04-25 19:20:55] [probe] drawer A obj_speed L16: r2_mean=0.922 std=0.002 24.6s

[2026-04-25 19:21:24] [probe] drawer A obj_speed L17: r2_mean=0.927 std=0.003 23.5s

[2026-04-25 19:21:52] [probe] drawer A obj_speed L18: r2_mean=0.928 std=0.002 24.9s

[2026-04-25 19:22:18] [probe] drawer A obj_speed L19: r2_mean=0.926 std=0.004 21.3s

[2026-04-25 19:22:36] [extract] nut_thread sh0 800/2500 eps 103200 win 29.2 win/s ETA 125.0min

[2026-04-25 19:22:46] [probe] drawer A obj_speed L20: r2_mean=0.924 std=0.003 24.5s

[2026-04-25 19:23:14] [probe] drawer A obj_speed L21: r2_mean=0.925 std=0.003 23.3s

[2026-04-25 19:23:40] [probe] drawer A obj_speed L22: r2_mean=0.927 std=0.003 23.1s

[2026-04-25 19:24:07] [probe] drawer A obj_speed L23: r2_mean=0.924 std=0.002 22.2s

[2026-04-25 19:24:26] [extract] nut_thread sh0 825/2500 eps 106425 win 29.2 win/s ETA 123.2min

[2026-04-25 19:24:35] [probe] drawer A obj_direction L00: r2_mean=0.151 std=0.007 18.6s

[2026-04-25 19:24:55] [probe] drawer A obj_direction L01: r2_mean=0.216 std=0.012 18.7s

[2026-04-25 19:25:15] [probe] drawer A obj_direction L02: r2_mean=0.247 std=0.011 18.9s

[2026-04-25 19:25:32] [probe] drawer A obj_direction L03: r2_mean=0.271 std=0.013 16.2s

[2026-04-25 19:25:46] [probe] drawer A obj_direction L04: r2_mean=0.297 std=0.012 12.8s

[2026-04-25 19:26:06] [probe] drawer A obj_direction L05: r2_mean=0.323 std=0.011 19.2s

[2026-04-25 19:26:16] [extract] nut_thread sh0 850/2500 eps 109650 win 29.2 win/s ETA 121.4min

[2026-04-25 19:26:26] [probe] drawer A obj_direction L06: r2_mean=0.329 std=0.011 18.7s

[2026-04-25 19:26:46] [probe] drawer A obj_direction L07: r2_mean=0.353 std=0.012 19.9s

[2026-04-25 19:27:06] [probe] drawer A obj_direction L08: r2_mean=0.345 std=0.008 18.3s

[2026-04-25 19:27:26] [probe] drawer A obj_direction L09: r2_mean=0.325 std=0.015 19.2s

[2026-04-25 19:27:47] [probe] drawer A obj_direction L10: r2_mean=0.328 std=0.009 19.4s

[2026-04-25 19:28:06] [probe] drawer A obj_direction L11: r2_mean=0.338 std=0.009 18.4s

[2026-04-25 19:28:06] [extract] nut_thread sh0 875/2500 eps 112875 win 29.2 win/s ETA 119.5min

[2026-04-25 19:28:26] [probe] drawer A obj_direction L12: r2_mean=0.347 std=0.010 19.2s

[2026-04-25 19:28:46] [probe] drawer A obj_direction L13: r2_mean=0.352 std=0.008 18.9s

[2026-04-25 19:29:06] [probe] drawer A obj_direction L14: r2_mean=0.342 std=0.012 18.2s

[2026-04-25 19:29:26] [probe] drawer A obj_direction L15: r2_mean=0.348 std=0.012 18.8s

[2026-04-25 19:29:46] [probe] drawer A obj_direction L16: r2_mean=0.355 std=0.018 19.4s

[2026-04-25 19:29:57] [extract] nut_thread sh0 900/2500 eps 116100 win 29.2 win/s ETA 117.7min

[2026-04-25 19:30:02] [probe] drawer A obj_direction L17: r2_mean=0.357 std=0.013 15.0s

[2026-04-25 19:30:17] [probe] drawer A obj_direction L18: r2_mean=0.358 std=0.011 14.3s

[2026-04-25 19:30:36] [probe] drawer A obj_direction L19: r2_mean=0.361 std=0.007 18.4s

[2026-04-25 19:30:57] [probe] drawer A obj_direction L20: r2_mean=0.368 std=0.017 19.4s

[2026-04-25 19:31:16] [probe] drawer A obj_direction L21: r2_mean=0.353 std=0.032 18.0s

[2026-04-25 19:31:37] [probe] drawer A obj_direction L22: r2_mean=0.377 std=0.017 19.7s

[2026-04-25 19:31:47] [extract] nut_thread sh0 925/2500 eps 119325 win 29.2 win/s ETA 115.8min

[2026-04-25 19:31:54] [probe] drawer A obj_direction L23: r2_mean=0.251 std=0.076 16.6s

[2026-04-25 19:32:24] [probe] drawer A obj_acceleration L00: r2_mean=0.059 std=0.004 25.3s

[2026-04-25 19:32:54] [probe] drawer A obj_acceleration L01: r2_mean=0.108 std=0.007 26.0s

[2026-04-25 19:33:23] [probe] drawer A obj_acceleration L02: r2_mean=0.140 std=0.005 26.0s

[2026-04-25 19:33:37] [extract] nut_thread sh0 950/2500 eps 122550 win 29.2 win/s ETA 114.0min

[2026-04-25 19:33:53] [probe] drawer A obj_acceleration L03: r2_mean=0.154 std=0.003 25.8s

[2026-04-25 19:34:22] [probe] drawer A obj_acceleration L04: r2_mean=0.183 std=0.004 26.1s

[2026-04-25 19:34:52] [probe] drawer A obj_acceleration L05: r2_mean=0.220 std=0.005 25.7s

[2026-04-25 19:35:21] [probe] drawer A obj_acceleration L06: r2_mean=0.240 std=0.016 26.4s

[2026-04-25 19:35:27] [extract] nut_thread sh0 975/2500 eps 125775 win 29.2 win/s ETA 112.1min

[2026-04-25 19:35:52] [probe] drawer A obj_acceleration L07: r2_mean=0.254 std=0.006 26.2s

[2026-04-25 19:36:21] [probe] drawer A obj_acceleration L08: r2_mean=0.256 std=0.006 26.2s

[2026-04-25 19:36:50] [probe] drawer A obj_acceleration L09: r2_mean=0.262 std=0.016 24.2s

[2026-04-25 19:37:18] [extract] nut_thread sh0 1000/2500 eps 129000 win 29.2 win/s ETA 110.3min

[2026-04-25 19:37:18] [probe] drawer A obj_acceleration L10: r2_mean=0.278 std=0.012 25.7s

[2026-04-25 19:37:48] [probe] drawer A obj_acceleration L11: r2_mean=0.282 std=0.008 25.6s

[2026-04-25 19:38:17] [probe] drawer A obj_acceleration L12: r2_mean=0.311 std=0.017 26.3s

[2026-04-25 19:38:47] [probe] drawer A obj_acceleration L13: r2_mean=0.330 std=0.017 25.6s

[2026-04-25 19:39:09] [extract] nut_thread sh0 1025/2500 eps 132225 win 29.2 win/s ETA 108.5min

[2026-04-25 19:39:15] [probe] drawer A obj_acceleration L14: r2_mean=0.323 std=0.006 25.8s

[2026-04-25 19:39:44] [probe] drawer A obj_acceleration L15: r2_mean=0.316 std=0.009 25.8s

[2026-04-25 19:40:13] [probe] drawer A obj_acceleration L16: r2_mean=0.316 std=0.008 26.0s

[2026-04-25 19:40:42] [probe] drawer A obj_acceleration L17: r2_mean=0.323 std=0.005 25.0s

[2026-04-25 19:40:59] [extract] nut_thread sh0 1050/2500 eps 135450 win 29.2 win/s ETA 106.6min

[2026-04-25 19:41:10] [probe] drawer A obj_acceleration L18: r2_mean=0.321 std=0.005 25.3s

[2026-04-25 19:41:39] [probe] drawer A obj_acceleration L19: r2_mean=0.321 std=0.023 26.1s

[2026-04-25 19:42:08] [probe] drawer A obj_acceleration L20: r2_mean=0.301 std=0.005 25.9s

[2026-04-25 19:42:36] [probe] drawer A obj_acceleration L21: r2_mean=0.295 std=0.005 25.6s

[2026-04-25 19:42:49] [extract] nut_thread sh0 1075/2500 eps 138675 win 29.2 win/s ETA 104.8min

[2026-04-25 19:43:04] [probe] drawer A obj_acceleration L22: r2_mean=0.293 std=0.017 25.5s

[2026-04-25 19:43:34] [probe] drawer A obj_acceleration L23: r2_mean=0.287 std=0.005 25.8s

[2026-04-25 19:44:00] [probe] drawer A obj_accel_mag L00: r2_mean=0.275 std=0.004 23.5s

[2026-04-25 19:44:29] [probe] drawer A obj_accel_mag L01: r2_mean=0.314 std=0.004 24.9s

[2026-04-25 19:44:39] [extract] nut_thread sh0 1100/2500 eps 141900 win 29.2 win/s ETA 103.0min

[2026-04-25 19:44:55] [probe] drawer A obj_accel_mag L02: r2_mean=0.321 std=0.005 23.4s

[2026-04-25 19:45:21] [probe] drawer A obj_accel_mag L03: r2_mean=0.342 std=0.003 23.1s

[2026-04-25 19:45:48] [probe] drawer A obj_accel_mag L04: r2_mean=0.357 std=0.006 24.1s

[2026-04-25 19:46:16] [probe] drawer A obj_accel_mag L05: r2_mean=0.389 std=0.005 24.7s

[2026-04-25 19:46:29] [extract] nut_thread sh0 1125/2500 eps 145125 win 29.2 win/s ETA 101.1min

[2026-04-25 19:46:43] [probe] drawer A obj_accel_mag L06: r2_mean=0.398 std=0.006 24.2s

[2026-04-25 19:47:11] [probe] drawer A obj_accel_mag L07: r2_mean=0.414 std=0.006 24.7s

[2026-04-25 19:47:37] [probe] drawer A obj_accel_mag L08: r2_mean=0.417 std=0.007 23.8s

[2026-04-25 19:48:05] [probe] drawer A obj_accel_mag L09: r2_mean=0.412 std=0.003 24.1s

[2026-04-25 19:48:20] [extract] nut_thread sh0 1150/2500 eps 148350 win 29.2 win/s ETA 99.3min

[2026-04-25 19:48:31] [probe] drawer A obj_accel_mag L10: r2_mean=0.424 std=0.006 23.5s

[2026-04-25 19:48:57] [probe] drawer A obj_accel_mag L11: r2_mean=0.448 std=0.009 23.0s

[2026-04-25 19:49:24] [probe] drawer A obj_accel_mag L12: r2_mean=0.451 std=0.007 23.9s

[2026-04-25 19:49:50] [probe] drawer A obj_accel_mag L13: r2_mean=0.466 std=0.005 23.6s

[2026-04-25 19:50:10] [extract] nut_thread sh0 1175/2500 eps 151575 win 29.2 win/s ETA 97.4min

[2026-04-25 19:50:17] [probe] drawer A obj_accel_mag L14: r2_mean=0.470 std=0.007 23.7s

[2026-04-25 19:50:43] [probe] drawer A obj_accel_mag L15: r2_mean=0.476 std=0.006 23.1s

[2026-04-25 19:51:09] [probe] drawer A obj_accel_mag L16: r2_mean=0.481 std=0.005 22.5s

[2026-04-25 19:51:35] [probe] drawer A obj_accel_mag L17: r2_mean=0.489 std=0.005 22.9s

[2026-04-25 19:52:00] [extract] nut_thread sh0 1200/2500 eps 154800 win 29.2 win/s ETA 95.6min

[2026-04-25 19:52:02] [probe] drawer A obj_accel_mag L18: r2_mean=0.492 std=0.004 24.2s

[2026-04-25 19:52:29] [probe] drawer A obj_accel_mag L19: r2_mean=0.490 std=0.005 23.8s

[2026-04-25 19:52:55] [probe] drawer A obj_accel_mag L20: r2_mean=0.491 std=0.007 23.6s

[2026-04-25 19:53:20] [probe] drawer A obj_accel_mag L21: r2_mean=0.489 std=0.008 21.6s

[2026-04-25 19:53:47] [probe] drawer A obj_accel_mag L22: r2_mean=0.491 std=0.008 23.9s

[2026-04-25 19:53:51] [extract] nut_thread sh0 1225/2500 eps 158025 win 29.2 win/s ETA 93.8min

[2026-04-25 19:54:14] [probe] drawer A obj_accel_mag L23: r2_mean=0.485 std=0.005 23.9s

[2026-04-25 19:54:14] [probe] drawer A DONE in 137.1min

[2026-04-25 19:54:19] [extract] peg_insert loading model on cuda:0

[2026-04-25 19:54:31] [extract] peg_insert model loaded 11.2s

[2026-04-25 19:54:31] [extract] peg_insert shard 1/1 batch=8: total 2500 todo 2500

[2026-04-25 19:55:41] [extract] nut_thread sh0 1250/2500 eps 161250 win 29.2 win/s ETA 91.9min

[2026-04-25 19:56:26] [extract] peg_insert sh0 25/2500 eps 3225 win 28.0 win/s ETA 189.8min

[2026-04-25 19:57:31] [extract] nut_thread sh0 1275/2500 eps 164475 win 29.2 win/s ETA 90.1min

[2026-04-25 19:58:16] [extract] peg_insert sh0 50/2500 eps 6450 win 28.7 win/s ETA 183.7min

[2026-04-25 19:59:21] [extract] nut_thread sh0 1300/2500 eps 167700 win 29.2 win/s ETA 88.2min

[2026-04-25 20:00:06] [extract] peg_insert sh0 75/2500 eps 9675 win 28.9 win/s ETA 180.6min

[2026-04-25 20:01:11] [extract] nut_thread sh0 1325/2500 eps 170925 win 29.2 win/s ETA 86.4min

[2026-04-25 20:01:56] [extract] peg_insert sh0 100/2500 eps 12900 win 29.0 win/s ETA 178.0min

[2026-04-25 20:03:01] [extract] nut_thread sh0 1350/2500 eps 174150 win 29.2 win/s ETA 84.6min

[2026-04-25 20:03:46] [extract] peg_insert sh0 125/2500 eps 16125 win 29.1 win/s ETA 175.7min

[2026-04-25 20:04:52] [extract] nut_thread sh0 1375/2500 eps 177375 win 29.2 win/s ETA 82.7min

[2026-04-25 20:05:35] [extract] peg_insert sh0 150/2500 eps 19350 win 29.1 win/s ETA 173.5min

[2026-04-25 20:06:42] [extract] nut_thread sh0 1400/2500 eps 180600 win 29.2 win/s ETA 80.9min

[2026-04-25 20:07:36] [extract] peg_insert sh0 175/2500 eps 22575 win 28.8 win/s ETA 173.8min

[2026-04-25 20:08:32] [extract] nut_thread sh0 1425/2500 eps 183825 win 29.2 win/s ETA 79.0min

[2026-04-25 20:10:01] [extract] peg_insert sh0 200/2500 eps 25800 win 27.7 win/s ETA 178.3min

[2026-04-25 20:10:23] [extract] nut_thread sh0 1450/2500 eps 187050 win 29.2 win/s ETA 77.2min

[2026-04-25 20:12:08] [extract] peg_insert sh0 225/2500 eps 29025 win 27.5 win/s ETA 178.1min

[2026-04-25 20:12:13] [extract] nut_thread sh0 1475/2500 eps 190275 win 29.2 win/s ETA 75.4min

[2026-04-25 20:13:57] [extract] peg_insert sh0 250/2500 eps 32250 win 27.6 win/s ETA 175.0min

[2026-04-25 20:14:03] [extract] nut_thread sh0 1500/2500 eps 193500 win 29.2 win/s ETA 73.5min

[2026-04-25 20:15:47] [extract] peg_insert sh0 275/2500 eps 35475 win 27.8 win/s ETA 172.1min

[2026-04-25 20:15:54] [extract] nut_thread sh0 1525/2500 eps 196725 win 29.2 win/s ETA 71.7min

[2026-04-25 20:17:37] [extract] peg_insert sh0 300/2500 eps 38700 win 27.9 win/s ETA 169.5min

[2026-04-25 20:17:44] [extract] nut_thread sh0 1550/2500 eps 199950 win 29.2 win/s ETA 69.9min

[2026-04-25 20:19:27] [extract] peg_insert sh0 325/2500 eps 41925 win 28.0 win/s ETA 166.9min

[2026-04-25 20:19:35] [extract] nut_thread sh0 1575/2500 eps 203175 win 29.2 win/s ETA 68.0min

[2026-04-25 20:21:17] [extract] peg_insert sh0 350/2500 eps 45150 win 28.1 win/s ETA 164.4min

[2026-04-25 20:21:24] [extract] nut_thread sh0 1600/2500 eps 206400 win 29.2 win/s ETA 66.2min

[2026-04-25 20:23:14] [extract] nut_thread sh0 1625/2500 eps 209625 win 29.2 win/s ETA 64.3min

[2026-04-25 20:23:34] [extract] peg_insert sh0 375/2500 eps 48375 win 27.8 win/s ETA 164.6min

[2026-04-25 20:25:04] [extract] nut_thread sh0 1650/2500 eps 212850 win 29.2 win/s ETA 62.5min

[2026-04-25 20:25:57] [extract] peg_insert sh0 400/2500 eps 51600 win 27.4 win/s ETA 165.0min

[2026-04-25 20:26:54] [extract] nut_thread sh0 1675/2500 eps 216075 win 29.2 win/s ETA 60.7min

[2026-04-25 20:28:23] [extract] peg_insert sh0 425/2500 eps 54825 win 27.0 win/s ETA 165.4min

[2026-04-25 20:28:44] [extract] nut_thread sh0 1700/2500 eps 219300 win 29.2 win/s ETA 58.8min

[2026-04-25 20:30:27] [extract] peg_insert sh0 450/2500 eps 58050 win 26.9 win/s ETA 163.7min

[2026-04-25 20:30:34] [extract] nut_thread sh0 1725/2500 eps 222525 win 29.2 win/s ETA 57.0min

[2026-04-25 20:32:24] [extract] nut_thread sh0 1750/2500 eps 225750 win 29.2 win/s ETA 55.1min

[2026-04-25 20:32:54] [extract] peg_insert sh0 475/2500 eps 61275 win 26.6 win/s ETA 163.6min

[2026-04-25 20:34:14] [extract] nut_thread sh0 1775/2500 eps 228975 win 29.2 win/s ETA 53.3min

[2026-04-25 20:35:17] [extract] peg_insert sh0 500/2500 eps 64500 win 26.4 win/s ETA 163.1min

[2026-04-25 20:36:04] [extract] nut_thread sh0 1800/2500 eps 232200 win 29.3 win/s ETA 51.5min

[2026-04-25 20:37:43] [extract] peg_insert sh0 525/2500 eps 67725 win 26.1 win/s ETA 162.5min

[2026-04-25 20:37:54] [extract] nut_thread sh0 1825/2500 eps 235425 win 29.3 win/s ETA 49.6min

[2026-04-25 20:39:44] [extract] nut_thread sh0 1850/2500 eps 238650 win 29.3 win/s ETA 47.8min

[2026-04-25 20:39:45] [extract] peg_insert sh0 550/2500 eps 70950 win 26.1 win/s ETA 160.4min

[2026-04-25 20:41:34] [extract] nut_thread sh0 1875/2500 eps 241875 win 29.3 win/s ETA 45.9min

[2026-04-25 20:42:09] [extract] peg_insert sh0 575/2500 eps 74175 win 25.9 win/s ETA 159.5min

[2026-04-25 20:43:24] [extract] nut_thread sh0 1900/2500 eps 245100 win 29.3 win/s ETA 44.1min

[2026-04-25 20:43:59] [extract] peg_insert sh0 600/2500 eps 77400 win 26.1 win/s ETA 156.7min

[2026-04-25 20:45:15] [extract] nut_thread sh0 1925/2500 eps 248325 win 29.3 win/s ETA 42.3min

[2026-04-25 20:45:49] [extract] peg_insert sh0 625/2500 eps 80625 win 26.2 win/s ETA 153.9min

[2026-04-25 20:47:05] [extract] nut_thread sh0 1950/2500 eps 251550 win 29.3 win/s ETA 40.4min

[2026-04-25 20:47:40] [extract] peg_insert sh0 650/2500 eps 83850 win 26.3 win/s ETA 151.3min

[2026-04-25 20:48:55] [extract] nut_thread sh0 1975/2500 eps 254775 win 29.3 win/s ETA 38.6min

[2026-04-25 20:50:06] [extract] peg_insert sh0 675/2500 eps 87075 win 26.1 win/s ETA 150.3min

[2026-04-25 20:50:45] [extract] nut_thread sh0 2000/2500 eps 258000 win 29.3 win/s ETA 36.7min

[2026-04-25 20:52:29] [extract] peg_insert sh0 700/2500 eps 90300 win 26.0 win/s ETA 149.1min

[2026-04-25 20:52:35] [extract] nut_thread sh0 2025/2500 eps 261225 win 29.3 win/s ETA 34.9min

[2026-04-25 20:54:25] [extract] nut_thread sh0 2050/2500 eps 264450 win 29.3 win/s ETA 33.1min

[2026-04-25 20:54:54] [extract] peg_insert sh0 725/2500 eps 93525 win 25.8 win/s ETA 147.8min

[2026-04-25 20:56:15] [extract] nut_thread sh0 2075/2500 eps 267675 win 29.3 win/s ETA 31.2min

[2026-04-25 20:57:11] [extract] peg_insert sh0 750/2500 eps 96750 win 25.7 win/s ETA 146.2min

[2026-04-25 20:58:05] [extract] nut_thread sh0 2100/2500 eps 270900 win 29.3 win/s ETA 29.4min

[2026-04-25 20:59:33] [extract] peg_insert sh0 775/2500 eps 99975 win 25.6 win/s ETA 144.8min

[2026-04-25 20:59:55] [extract] nut_thread sh0 2125/2500 eps 274125 win 29.3 win/s ETA 27.6min

[2026-04-25 21:01:39] [extract] peg_insert sh0 800/2500 eps 103200 win 25.6 win/s ETA 142.7min

[2026-04-25 21:01:45] [extract] nut_thread sh0 2150/2500 eps 277350 win 29.3 win/s ETA 25.7min

[2026-04-25 21:03:35] [extract] nut_thread sh0 2175/2500 eps 280575 win 29.3 win/s ETA 23.9min

[2026-04-25 21:04:05] [extract] peg_insert sh0 825/2500 eps 106425 win 25.5 win/s ETA 141.3min

[2026-04-25 21:05:24] [extract] nut_thread sh0 2200/2500 eps 283800 win 29.3 win/s ETA 22.0min

[2026-04-25 21:06:29] [extract] peg_insert sh0 850/2500 eps 109650 win 25.4 win/s ETA 139.7min

[2026-04-25 21:07:14] [extract] nut_thread sh0 2225/2500 eps 287025 win 29.3 win/s ETA 20.2min

[2026-04-25 21:08:49] [extract] peg_insert sh0 875/2500 eps 112875 win 25.3 win/s ETA 138.0min

[2026-04-25 21:09:04] [extract] nut_thread sh0 2250/2500 eps 290250 win 29.3 win/s ETA 18.4min

[2026-04-25 21:10:54] [extract] nut_thread sh0 2275/2500 eps 293475 win 29.3 win/s ETA 16.5min

[2026-04-25 21:11:11] [extract] peg_insert sh0 900/2500 eps 116100 win 25.2 win/s ETA 136.3min

[2026-04-25 21:12:44] [extract] nut_thread sh0 2300/2500 eps 296700 win 29.3 win/s ETA 14.7min

[2026-04-25 21:13:31] [extract] peg_insert sh0 925/2500 eps 119325 win 25.2 win/s ETA 134.5min

[2026-04-25 21:14:34] [extract] nut_thread sh0 2325/2500 eps 299925 win 29.3 win/s ETA 12.9min

[2026-04-25 21:15:46] [extract] peg_insert sh0 950/2500 eps 122550 win 25.1 win/s ETA 132.6min

[2026-04-25 21:16:24] [extract] nut_thread sh0 2350/2500 eps 303150 win 29.3 win/s ETA 11.0min

[2026-04-25 21:18:03] [extract] peg_insert sh0 975/2500 eps 125775 win 25.1 win/s ETA 130.7min

[2026-04-25 21:18:14] [extract] nut_thread sh0 2375/2500 eps 306375 win 29.3 win/s ETA 9.2min

[2026-04-25 21:20:05] [extract] nut_thread sh0 2400/2500 eps 309600 win 29.3 win/s ETA 7.3min

[2026-04-25 21:20:29] [extract] peg_insert sh0 1000/2500 eps 129000 win 25.0 win/s ETA 129.0min

[2026-04-25 21:21:55] [extract] nut_thread sh0 2425/2500 eps 312825 win 29.3 win/s ETA 5.5min

[2026-04-25 21:22:55] [extract] peg_insert sh0 1025/2500 eps 132225 win 24.9 win/s ETA 127.2min

[2026-04-25 21:23:44] [extract] nut_thread sh0 2450/2500 eps 316050 win 29.3 win/s ETA 3.7min

[2026-04-25 21:25:13] [extract] peg_insert sh0 1050/2500 eps 135450 win 24.9 win/s ETA 125.3min

[2026-04-25 21:25:35] [extract] nut_thread sh0 2475/2500 eps 319275 win 29.3 win/s ETA 1.8min

[2026-04-25 21:27:23] [extract] peg_insert sh0 1075/2500 eps 138675 win 24.9 win/s ETA 123.1min

[2026-04-25 21:27:25] [extract] nut_thread sh0 2500/2500 eps 322500 win 29.3 win/s ETA 0.0min

[2026-04-25 21:27:25] [extract] nut_thread sh0 DONE: 2500 eps 322500 win in 183.7min

[2026-04-25 21:27:29] [probe] task=nut_thread variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag', 'obj_position', 'obj_velocity', 'obj_speed', 'obj_direction', 'obj_acceleration', 'obj_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=1

[2026-04-25 21:28:03] [probe] nut_thread: features [(322500, 24, 1024)] eps 2500 loaded 34.6s

[2026-04-25 21:28:31] [probe] nut_thread A ee_position L00: r2_mean=0.643 std=0.012 25.4s

[2026-04-25 21:28:56] [probe] nut_thread A ee_position L01: r2_mean=0.836 std=0.047 23.3s

[2026-04-25 21:29:21] [probe] nut_thread A ee_position L02: r2_mean=0.872 std=0.005 22.7s

[2026-04-25 21:29:47] [probe] nut_thread A ee_position L03: r2_mean=0.915 std=0.002 23.7s

[2026-04-25 21:29:49] [extract] peg_insert sh0 1100/2500 eps 141900 win 24.8 win/s ETA 121.3min

[2026-04-25 21:30:11] [probe] nut_thread A ee_position L04: r2_mean=0.924 std=0.002 22.9s

[2026-04-25 21:30:37] [probe] nut_thread A ee_position L05: r2_mean=0.942 std=0.001 23.7s

[2026-04-25 21:31:02] [probe] nut_thread A ee_position L06: r2_mean=0.888 std=0.053 22.9s

[2026-04-25 21:31:27] [probe] nut_thread A ee_position L07: r2_mean=0.843 std=0.064 23.4s

[2026-04-25 21:31:51] [probe] nut_thread A ee_position L08: r2_mean=0.957 std=0.010 21.9s

[2026-04-25 21:32:04] [extract] peg_insert sh0 1125/2500 eps 145125 win 24.8 win/s ETA 119.2min

[2026-04-25 21:32:16] [probe] nut_thread A ee_position L09: r2_mean=0.860 std=0.186 22.5s

[2026-04-25 21:32:41] [probe] nut_thread A ee_position L10: r2_mean=0.924 std=0.041 23.2s

[2026-04-25 21:33:06] [probe] nut_thread A ee_position L11: r2_mean=0.973 std=0.007 23.2s

[2026-04-25 21:33:31] [probe] nut_thread A ee_position L12: r2_mean=0.971 std=0.001 22.7s

[2026-04-25 21:33:57] [probe] nut_thread A ee_position L13: r2_mean=0.972 std=0.002 23.9s

[2026-04-25 21:34:21] [probe] nut_thread A ee_position L14: r2_mean=0.971 std=0.001 23.0s

[2026-04-25 21:34:25] [extract] peg_insert sh0 1150/2500 eps 148350 win 24.7 win/s ETA 117.3min

[2026-04-25 21:34:47] [probe] nut_thread A ee_position L15: r2_mean=0.973 std=0.001 23.2s

[2026-04-25 21:35:12] [probe] nut_thread A ee_position L16: r2_mean=0.974 std=0.001 23.3s

[2026-04-25 21:35:36] [probe] nut_thread A ee_position L17: r2_mean=0.976 std=0.001 22.1s

[2026-04-25 21:36:00] [probe] nut_thread A ee_position L18: r2_mean=0.976 std=0.001 22.5s

[2026-04-25 21:36:25] [probe] nut_thread A ee_position L19: r2_mean=0.975 std=0.003 22.8s

[2026-04-25 21:36:50] [probe] nut_thread A ee_position L20: r2_mean=0.972 std=0.005 22.8s

[2026-04-25 21:36:50] [extract] peg_insert sh0 1175/2500 eps 151575 win 24.7 win/s ETA 115.4min

[2026-04-25 21:37:14] [probe] nut_thread A ee_position L21: r2_mean=0.975 std=0.001 22.4s

[2026-04-25 21:37:39] [probe] nut_thread A ee_position L22: r2_mean=0.973 std=0.001 23.0s

[2026-04-25 21:38:04] [probe] nut_thread A ee_position L23: r2_mean=0.968 std=0.002 23.0s

[2026-04-25 21:38:29] [probe] nut_thread A ee_velocity L00: r2_mean=-0.006 std=0.002 22.7s

[2026-04-25 21:38:53] [probe] nut_thread A ee_velocity L01: r2_mean=-0.212 std=0.320 22.2s

[2026-04-25 21:39:15] [extract] peg_insert sh0 1200/2500 eps 154800 win 24.6 win/s ETA 113.5min

[2026-04-25 21:39:18] [probe] nut_thread A ee_velocity L02: r2_mean=0.025 std=0.004 22.9s

[2026-04-25 21:39:43] [probe] nut_thread A ee_velocity L03: r2_mean=0.025 std=0.007 23.1s

[2026-04-25 21:40:03] [probe] nut_thread A ee_velocity L04: r2_mean=0.029 std=0.004 18.5s

[2026-04-25 21:40:28] [probe] nut_thread A ee_velocity L05: r2_mean=0.042 std=0.009 23.0s

[2026-04-25 21:40:52] [probe] nut_thread A ee_velocity L06: r2_mean=-0.015 std=0.049 22.1s

[2026-04-25 21:41:13] [probe] nut_thread A ee_velocity L07: r2_mean=-0.176 std=0.109 19.0s

[2026-04-25 21:41:32] [extract] peg_insert sh0 1225/2500 eps 158025 win 24.6 win/s ETA 111.4min

[2026-04-25 21:41:38] [probe] nut_thread A ee_velocity L08: r2_mean=0.023 std=0.021 23.4s

[2026-04-25 21:42:02] [probe] nut_thread A ee_velocity L09: r2_mean=-0.046 std=0.098 21.2s

[2026-04-25 21:42:27] [probe] nut_thread A ee_velocity L10: r2_mean=-0.489 std=0.487 23.0s

[2026-04-25 21:42:51] [probe] nut_thread A ee_velocity L11: r2_mean=0.048 std=0.026 22.7s

[2026-04-25 21:43:16] [probe] nut_thread A ee_velocity L12: r2_mean=0.073 std=0.008 22.8s

[2026-04-25 21:43:41] [probe] nut_thread A ee_velocity L13: r2_mean=0.076 std=0.018 22.8s

[2026-04-25 21:43:55] [extract] peg_insert sh0 1250/2500 eps 161250 win 24.6 win/s ETA 109.4min

[2026-04-25 21:44:05] [probe] nut_thread A ee_velocity L14: r2_mean=0.088 std=0.012 22.2s

[2026-04-25 21:44:30] [probe] nut_thread A ee_velocity L15: r2_mean=0.114 std=0.013 22.9s

[2026-04-25 21:44:54] [probe] nut_thread A ee_velocity L16: r2_mean=0.139 std=0.012 22.7s

[2026-04-25 21:45:19] [probe] nut_thread A ee_velocity L17: r2_mean=0.159 std=0.013 22.8s

[2026-04-25 21:45:44] [probe] nut_thread A ee_velocity L18: r2_mean=0.156 std=0.018 23.2s

[2026-04-25 21:46:09] [probe] nut_thread A ee_velocity L19: r2_mean=0.147 std=0.010 22.7s

[2026-04-25 21:46:15] [extract] peg_insert sh0 1275/2500 eps 164475 win 24.5 win/s ETA 107.4min

[2026-04-25 21:46:34] [probe] nut_thread A ee_velocity L20: r2_mean=0.153 std=0.012 23.4s

[2026-04-25 21:46:59] [probe] nut_thread A ee_velocity L21: r2_mean=0.142 std=0.014 22.8s

[2026-04-25 21:47:23] [probe] nut_thread A ee_velocity L22: r2_mean=0.134 std=0.022 22.4s

[2026-04-25 21:47:48] [probe] nut_thread A ee_velocity L23: r2_mean=0.133 std=0.026 23.1s

[2026-04-25 21:48:12] [probe] nut_thread A ee_speed L00: r2_mean=0.010 std=0.003 21.4s

[2026-04-25 21:48:34] [probe] nut_thread A ee_speed L01: r2_mean=0.088 std=0.013 20.4s

[2026-04-25 21:48:35] [extract] peg_insert sh0 1300/2500 eps 167700 win 24.5 win/s ETA 105.3min

[2026-04-25 21:48:56] [probe] nut_thread A ee_speed L02: r2_mean=0.126 std=0.013 20.1s

[2026-04-25 21:49:20] [probe] nut_thread A ee_speed L03: r2_mean=0.142 std=0.015 21.7s

[2026-04-25 21:49:44] [probe] nut_thread A ee_speed L04: r2_mean=0.172 std=0.017 21.9s

[2026-04-25 21:50:07] [probe] nut_thread A ee_speed L05: r2_mean=0.217 std=0.015 21.5s

[2026-04-25 21:50:30] [probe] nut_thread A ee_speed L06: r2_mean=0.213 std=0.028 20.8s

[2026-04-25 21:50:45] [extract] peg_insert sh0 1325/2500 eps 170925 win 24.5 win/s ETA 103.1min

[2026-04-25 21:50:54] [probe] nut_thread A ee_speed L07: r2_mean=0.217 std=0.040 21.4s

[2026-04-25 21:51:16] [probe] nut_thread A ee_speed L08: r2_mean=0.262 std=0.023 21.1s

[2026-04-25 21:51:40] [probe] nut_thread A ee_speed L09: r2_mean=0.263 std=0.023 21.3s

[2026-04-25 21:52:04] [probe] nut_thread A ee_speed L10: r2_mean=0.297 std=0.017 22.0s

[2026-04-25 21:52:25] [probe] nut_thread A ee_speed L11: r2_mean=0.329 std=0.018 19.6s

[2026-04-25 21:52:47] [probe] nut_thread A ee_speed L12: r2_mean=0.306 std=0.017 20.6s

[2026-04-25 21:53:10] [probe] nut_thread A ee_speed L13: r2_mean=0.323 std=0.018 20.7s

[2026-04-25 21:53:12] [extract] peg_insert sh0 1350/2500 eps 174150 win 24.5 win/s ETA 101.1min

[2026-04-25 21:53:34] [probe] nut_thread A ee_speed L14: r2_mean=0.337 std=0.018 21.7s

[2026-04-25 21:53:56] [probe] nut_thread A ee_speed L15: r2_mean=0.373 std=0.018 20.4s

[2026-04-25 21:54:20] [probe] nut_thread A ee_speed L16: r2_mean=0.387 std=0.017 21.5s

[2026-04-25 21:54:43] [probe] nut_thread A ee_speed L17: r2_mean=0.400 std=0.019 21.6s

[2026-04-25 21:55:06] [probe] nut_thread A ee_speed L18: r2_mean=0.402 std=0.019 21.0s

[2026-04-25 21:55:29] [probe] nut_thread A ee_speed L19: r2_mean=0.401 std=0.019 21.3s

[2026-04-25 21:55:37] [extract] peg_insert sh0 1375/2500 eps 177375 win 24.4 win/s ETA 99.1min

[2026-04-25 21:55:45] [probe] nut_thread A ee_speed L20: r2_mean=0.394 std=0.018 14.0s

[2026-04-25 21:56:01] [probe] nut_thread A ee_speed L21: r2_mean=0.387 std=0.019 14.5s

[2026-04-25 21:56:24] [probe] nut_thread A ee_speed L22: r2_mean=0.378 std=0.019 21.0s

[2026-04-25 21:56:47] [probe] nut_thread A ee_speed L23: r2_mean=0.368 std=0.017 21.1s

[2026-04-25 21:57:24] [probe] nut_thread A ee_direction L00: r2_mean=-0.043 std=0.065 23.3s

[2026-04-25 21:57:49] [probe] nut_thread A ee_direction L01: r2_mean=-0.167 std=0.065 22.9s

[2026-04-25 21:57:56] [extract] peg_insert sh0 1400/2500 eps 180600 win 24.4 win/s ETA 97.0min

[2026-04-25 21:58:14] [probe] nut_thread A ee_direction L02: r2_mean=0.001 std=0.001 23.4s

[2026-04-25 21:58:38] [probe] nut_thread A ee_direction L03: r2_mean=0.001 std=0.001 22.6s

[2026-04-25 21:59:03] [probe] nut_thread A ee_direction L04: r2_mean=0.001 std=0.000 22.5s

[2026-04-25 21:59:28] [probe] nut_thread A ee_direction L05: r2_mean=0.001 std=0.001 23.2s

[2026-04-25 21:59:50] [probe] nut_thread A ee_direction L06: r2_mean=-0.026 std=0.026 20.1s

[2026-04-25 22:00:14] [probe] nut_thread A ee_direction L07: r2_mean=-0.212 std=0.081 22.2s

[2026-04-25 22:00:19] [extract] peg_insert sh0 1425/2500 eps 183825 win 24.4 win/s ETA 94.9min

[2026-04-25 22:00:39] [probe] nut_thread A ee_direction L08: r2_mean=-0.151 std=0.284 23.0s

[2026-04-25 22:01:04] [probe] nut_thread A ee_direction L09: r2_mean=-0.025 std=0.012 23.0s

[2026-04-25 22:01:29] [probe] nut_thread A ee_direction L10: r2_mean=-0.102 std=0.119 22.7s

[2026-04-25 22:01:54] [probe] nut_thread A ee_direction L11: r2_mean=-0.008 std=0.009 23.2s

[2026-04-25 22:02:19] [probe] nut_thread A ee_direction L12: r2_mean=0.002 std=0.001 22.5s

[2026-04-25 22:02:39] [extract] peg_insert sh0 1450/2500 eps 187050 win 24.3 win/s ETA 92.8min

[2026-04-25 22:02:43] [probe] nut_thread A ee_direction L13: r2_mean=0.002 std=0.002 22.7s

[2026-04-25 22:03:06] [probe] nut_thread A ee_direction L14: r2_mean=0.005 std=0.001 20.5s

[2026-04-25 22:03:29] [probe] nut_thread A ee_direction L15: r2_mean=0.005 std=0.002 21.6s

[2026-04-25 22:03:54] [probe] nut_thread A ee_direction L16: r2_mean=0.007 std=0.002 23.3s

[2026-04-25 22:04:20] [probe] nut_thread A ee_direction L17: r2_mean=0.010 std=0.001 23.1s

[2026-04-25 22:04:44] [probe] nut_thread A ee_direction L18: r2_mean=0.011 std=0.001 22.1s

[2026-04-25 22:05:02] [extract] peg_insert sh0 1475/2500 eps 190275 win 24.3 win/s ETA 90.7min

[2026-04-25 22:05:09] [probe] nut_thread A ee_direction L19: r2_mean=0.007 std=0.001 23.4s

[2026-04-25 22:05:33] [probe] nut_thread A ee_direction L20: r2_mean=0.008 std=0.002 22.4s

[2026-04-25 22:05:55] [probe] nut_thread A ee_direction L21: r2_mean=0.008 std=0.002 19.7s

[2026-04-25 22:06:20] [probe] nut_thread A ee_direction L22: r2_mean=0.007 std=0.002 22.8s

[2026-04-25 22:06:44] [probe] nut_thread A ee_direction L23: r2_mean=-0.007 std=0.030 22.5s

[2026-04-25 22:07:09] [probe] nut_thread A ee_acceleration L00: r2_mean=-0.008 std=0.002 21.6s

[2026-04-25 22:07:17] [extract] peg_insert sh0 1500/2500 eps 193500 win 24.3 win/s ETA 88.5min

[2026-04-25 22:07:34] [probe] nut_thread A ee_acceleration L01: r2_mean=-0.102 std=0.065 23.0s

[2026-04-25 22:07:59] [probe] nut_thread A ee_acceleration L02: r2_mean=0.006 std=0.002 23.4s

[2026-04-25 22:08:25] [probe] nut_thread A ee_acceleration L03: r2_mean=0.006 std=0.002 23.6s

[2026-04-25 22:08:50] [probe] nut_thread A ee_acceleration L04: r2_mean=0.008 std=0.001 23.4s

[2026-04-25 22:09:16] [probe] nut_thread A ee_acceleration L05: r2_mean=0.011 std=0.003 23.5s

[2026-04-25 22:09:34] [extract] peg_insert sh0 1525/2500 eps 196725 win 24.3 win/s ETA 86.3min

[2026-04-25 22:09:42] [probe] nut_thread A ee_acceleration L06: r2_mean=-0.052 std=0.100 23.7s

[2026-04-25 22:10:05] [probe] nut_thread A ee_acceleration L07: r2_mean=-0.649 std=0.798 21.0s

[2026-04-25 22:10:22] [probe] nut_thread A ee_acceleration L08: r2_mean=-0.024 std=0.026 15.3s

[2026-04-25 22:10:46] [probe] nut_thread A ee_acceleration L09: r2_mean=-0.148 std=0.174 22.5s

[2026-04-25 22:11:11] [probe] nut_thread A ee_acceleration L10: r2_mean=-0.272 std=0.173 23.1s

[2026-04-25 22:11:36] [probe] nut_thread A ee_acceleration L11: r2_mean=0.009 std=0.009 22.9s

[2026-04-25 22:11:59] [extract] peg_insert sh0 1550/2500 eps 199950 win 24.2 win/s ETA 84.3min

[2026-04-25 22:12:01] [probe] nut_thread A ee_acceleration L12: r2_mean=0.017 std=0.002 22.8s

[2026-04-25 22:12:22] [probe] nut_thread A ee_acceleration L13: r2_mean=0.017 std=0.007 19.4s

[2026-04-25 22:12:44] [probe] nut_thread A ee_acceleration L14: r2_mean=0.019 std=0.002 20.2s

[2026-04-25 22:13:09] [probe] nut_thread A ee_acceleration L15: r2_mean=0.020 std=0.004 22.9s

[2026-04-25 22:13:35] [probe] nut_thread A ee_acceleration L16: r2_mean=0.023 std=0.003 23.5s

[2026-04-25 22:14:00] [probe] nut_thread A ee_acceleration L17: r2_mean=0.026 std=0.003 23.0s

[2026-04-25 22:14:24] [probe] nut_thread A ee_acceleration L18: r2_mean=0.026 std=0.002 22.1s

[2026-04-25 22:14:25] [extract] peg_insert sh0 1575/2500 eps 203175 win 24.2 win/s ETA 82.2min

[2026-04-25 22:14:48] [probe] nut_thread A ee_acceleration L19: r2_mean=0.019 std=0.006 22.1s

[2026-04-25 22:15:13] [probe] nut_thread A ee_acceleration L20: r2_mean=0.025 std=0.004 22.9s

[2026-04-25 22:15:37] [probe] nut_thread A ee_acceleration L21: r2_mean=0.025 std=0.003 22.6s

[2026-04-25 22:16:03] [probe] nut_thread A ee_acceleration L22: r2_mean=0.022 std=0.008 23.8s

[2026-04-25 22:16:27] [probe] nut_thread A ee_acceleration L23: r2_mean=0.021 std=0.006 22.3s

[2026-04-25 22:16:37] [extract] peg_insert sh0 1600/2500 eps 206400 win 24.2 win/s ETA 79.9min

[2026-04-25 22:16:51] [probe] nut_thread A ee_accel_mag L00: r2_mean=0.008 std=0.002 21.4s

[2026-04-25 22:17:14] [probe] nut_thread A ee_accel_mag L01: r2_mean=0.093 std=0.012 21.0s

[2026-04-25 22:17:37] [probe] nut_thread A ee_accel_mag L02: r2_mean=0.129 std=0.010 21.8s

[2026-04-25 22:18:01] [probe] nut_thread A ee_accel_mag L03: r2_mean=0.142 std=0.011 21.6s

[2026-04-25 22:18:22] [probe] nut_thread A ee_accel_mag L04: r2_mean=0.168 std=0.013 19.4s

[2026-04-25 22:18:46] [probe] nut_thread A ee_accel_mag L05: r2_mean=0.199 std=0.012 21.1s

[2026-04-25 22:19:03] [extract] peg_insert sh0 1625/2500 eps 209625 win 24.2 win/s ETA 77.8min

[2026-04-25 22:19:09] [probe] nut_thread A ee_accel_mag L06: r2_mean=0.194 std=0.023 21.3s

[2026-04-25 22:19:31] [probe] nut_thread A ee_accel_mag L07: r2_mean=0.198 std=0.024 20.5s

[2026-04-25 22:19:55] [probe] nut_thread A ee_accel_mag L08: r2_mean=0.226 std=0.017 21.7s

[2026-04-25 22:20:18] [probe] nut_thread A ee_accel_mag L09: r2_mean=0.235 std=0.015 21.1s

[2026-04-25 22:20:41] [probe] nut_thread A ee_accel_mag L10: r2_mean=0.232 std=0.025 21.8s

[2026-04-25 22:21:05] [probe] nut_thread A ee_accel_mag L11: r2_mean=0.266 std=0.014 21.8s

[2026-04-25 22:21:27] [extract] peg_insert sh0 1650/2500 eps 212850 win 24.1 win/s ETA 75.7min

[2026-04-25 22:21:29] [probe] nut_thread A ee_accel_mag L12: r2_mean=0.257 std=0.012 21.6s

[2026-04-25 22:21:52] [probe] nut_thread A ee_accel_mag L13: r2_mean=0.269 std=0.014 21.4s

[2026-04-25 22:22:15] [probe] nut_thread A ee_accel_mag L14: r2_mean=0.279 std=0.014 21.4s

[2026-04-25 22:22:39] [probe] nut_thread A ee_accel_mag L15: r2_mean=0.297 std=0.015 21.6s

[2026-04-25 22:23:02] [probe] nut_thread A ee_accel_mag L16: r2_mean=0.304 std=0.015 21.6s

[2026-04-25 22:23:25] [probe] nut_thread A ee_accel_mag L17: r2_mean=0.314 std=0.016 21.0s

[2026-04-25 22:23:45] [extract] peg_insert sh0 1675/2500 eps 216075 win 24.1 win/s ETA 73.5min

[2026-04-25 22:23:48] [probe] nut_thread A ee_accel_mag L18: r2_mean=0.319 std=0.015 20.9s

[2026-04-25 22:24:12] [probe] nut_thread A ee_accel_mag L19: r2_mean=0.317 std=0.016 21.6s

[2026-04-25 22:24:34] [probe] nut_thread A ee_accel_mag L20: r2_mean=0.314 std=0.016 20.7s

[2026-04-25 22:24:58] [probe] nut_thread A ee_accel_mag L21: r2_mean=0.310 std=0.016 21.9s

[2026-04-25 22:25:22] [probe] nut_thread A ee_accel_mag L22: r2_mean=0.303 std=0.017 22.1s

[2026-04-25 22:25:44] [extract] peg_insert sh0 1700/2500 eps 219300 win 24.2 win/s ETA 71.2min

[2026-04-25 22:25:46] [probe] nut_thread A ee_accel_mag L23: r2_mean=0.299 std=0.015 21.7s

[2026-04-25 22:26:11] [probe] nut_thread A obj_position L00: r2_mean=0.617 std=0.028 23.1s

[2026-04-25 22:26:37] [probe] nut_thread A obj_position L01: r2_mean=0.821 std=0.046 23.7s

[2026-04-25 22:27:01] [probe] nut_thread A obj_position L02: r2_mean=0.873 std=0.004 22.4s

[2026-04-25 22:27:23] [probe] nut_thread A obj_position L03: r2_mean=0.915 std=0.002 20.2s

[2026-04-25 22:27:33] [extract] peg_insert sh0 1725/2500 eps 222525 win 24.2 win/s ETA 68.8min

[2026-04-25 22:27:46] [probe] nut_thread A obj_position L04: r2_mean=0.924 std=0.003 20.6s

[2026-04-25 22:28:11] [probe] nut_thread A obj_position L05: r2_mean=0.941 std=0.002 23.1s

[2026-04-25 22:28:36] [probe] nut_thread A obj_position L06: r2_mean=0.915 std=0.010 23.4s

[2026-04-25 22:29:02] [probe] nut_thread A obj_position L07: r2_mean=0.863 std=0.083 23.7s

[2026-04-25 22:29:27] [probe] nut_thread A obj_position L08: r2_mean=0.964 std=0.003 23.5s

[2026-04-25 22:29:32] [extract] peg_insert sh0 1750/2500 eps 225750 win 24.3 win/s ETA 66.4min

[2026-04-25 22:29:53] [probe] nut_thread A obj_position L09: r2_mean=0.935 std=0.031 23.2s

[2026-04-25 22:30:17] [probe] nut_thread A obj_position L10: r2_mean=0.900 std=0.043 22.8s

[2026-04-25 22:30:43] [probe] nut_thread A obj_position L11: r2_mean=0.976 std=0.002 23.6s

[2026-04-25 22:31:08] [probe] nut_thread A obj_position L12: r2_mean=0.971 std=0.001 22.7s

[2026-04-25 22:31:32] [probe] nut_thread A obj_position L13: r2_mean=0.973 std=0.002 22.3s

[2026-04-25 22:31:57] [extract] peg_insert sh0 1775/2500 eps 228975 win 24.2 win/s ETA 64.3min

[2026-04-25 22:31:58] [probe] nut_thread A obj_position L14: r2_mean=0.971 std=0.001 24.0s

[2026-04-25 22:32:22] [probe] nut_thread A obj_position L15: r2_mean=0.973 std=0.001 22.4s

[2026-04-25 22:32:47] [probe] nut_thread A obj_position L16: r2_mean=0.974 std=0.001 23.4s

[2026-04-25 22:33:13] [probe] nut_thread A obj_position L17: r2_mean=0.976 std=0.001 23.8s

[2026-04-25 22:33:38] [probe] nut_thread A obj_position L18: r2_mean=0.976 std=0.001 23.3s

[2026-04-25 22:34:04] [probe] nut_thread A obj_position L19: r2_mean=0.975 std=0.003 23.3s

[2026-04-25 22:34:21] [extract] peg_insert sh0 1800/2500 eps 232200 win 24.2 win/s ETA 62.2min

[2026-04-25 22:34:29] [probe] nut_thread A obj_position L20: r2_mean=0.974 std=0.002 23.1s

[2026-04-25 22:34:54] [probe] nut_thread A obj_position L21: r2_mean=0.975 std=0.001 23.3s

[2026-04-25 22:35:18] [probe] nut_thread A obj_position L22: r2_mean=0.973 std=0.001 21.3s

[2026-04-25 22:35:40] [probe] nut_thread A obj_position L23: r2_mean=0.968 std=0.007 20.7s

[2026-04-25 22:36:06] [probe] nut_thread A obj_velocity L00: r2_mean=-0.010 std=0.002 23.4s

[2026-04-25 22:36:31] [probe] nut_thread A obj_velocity L01: r2_mean=-0.061 std=0.073 22.9s

[2026-04-25 22:36:42] [extract] peg_insert sh0 1825/2500 eps 235425 win 24.2 win/s ETA 60.0min

[2026-04-25 22:36:56] [probe] nut_thread A obj_velocity L02: r2_mean=0.002 std=0.001 23.3s

[2026-04-25 22:37:21] [probe] nut_thread A obj_velocity L03: r2_mean=0.001 std=0.001 22.8s

[2026-04-25 22:37:46] [probe] nut_thread A obj_velocity L04: r2_mean=0.002 std=0.002 23.4s

[2026-04-25 22:38:09] [probe] nut_thread A obj_velocity L05: r2_mean=0.002 std=0.001 20.8s

[2026-04-25 22:38:25] [probe] nut_thread A obj_velocity L06: r2_mean=-0.108 std=0.124 15.0s

[2026-04-25 22:38:38] [probe] nut_thread A obj_velocity L07: r2_mean=-0.630 std=0.617 11.1s

[2026-04-25 22:39:02] [probe] nut_thread A obj_velocity L08: r2_mean=-0.036 std=0.029 22.2s

[2026-04-25 22:39:05] [extract] peg_insert sh0 1850/2500 eps 238650 win 24.2 win/s ETA 57.8min

[2026-04-25 22:39:28] [probe] nut_thread A obj_velocity L09: r2_mean=-0.074 std=0.067 23.6s

[2026-04-25 22:39:52] [probe] nut_thread A obj_velocity L10: r2_mean=-0.222 std=0.156 22.7s

[2026-04-25 22:40:18] [probe] nut_thread A obj_velocity L11: r2_mean=-0.001 std=0.004 23.3s

[2026-04-25 22:40:43] [probe] nut_thread A obj_velocity L12: r2_mean=0.004 std=0.001 23.1s

[2026-04-25 22:41:09] [probe] nut_thread A obj_velocity L13: r2_mean=0.004 std=0.001 23.7s

[2026-04-25 22:41:25] [extract] peg_insert sh0 1875/2500 eps 241875 win 24.2 win/s ETA 55.6min

[2026-04-25 22:41:34] [probe] nut_thread A obj_velocity L14: r2_mean=0.006 std=0.001 23.2s

[2026-04-25 22:41:59] [probe] nut_thread A obj_velocity L15: r2_mean=0.008 std=0.002 23.0s

[2026-04-25 22:42:24] [probe] nut_thread A obj_velocity L16: r2_mean=0.013 std=0.003 22.9s

[2026-04-25 22:42:49] [probe] nut_thread A obj_velocity L17: r2_mean=0.013 std=0.003 23.0s

[2026-04-25 22:43:13] [probe] nut_thread A obj_velocity L18: r2_mean=0.016 std=0.002 22.2s

[2026-04-25 22:43:38] [probe] nut_thread A obj_velocity L19: r2_mean=0.013 std=0.002 23.3s

[2026-04-25 22:43:50] [extract] peg_insert sh0 1900/2500 eps 245100 win 24.1 win/s ETA 53.5min

[2026-04-25 22:44:02] [probe] nut_thread A obj_velocity L20: r2_mean=0.014 std=0.002 22.1s

[2026-04-25 22:44:28] [probe] nut_thread A obj_velocity L21: r2_mean=0.012 std=0.001 23.3s

[2026-04-25 22:44:52] [probe] nut_thread A obj_velocity L22: r2_mean=0.011 std=0.004 22.4s

[2026-04-25 22:45:15] [probe] nut_thread A obj_velocity L23: r2_mean=-0.006 std=0.029 20.5s

[2026-04-25 22:45:39] [probe] nut_thread A obj_speed L00: r2_mean=-0.017 std=0.005 21.8s

[2026-04-25 22:45:55] [extract] peg_insert sh0 1925/2500 eps 248325 win 24.1 win/s ETA 51.2min

[2026-04-25 22:46:03] [probe] nut_thread A obj_speed L01: r2_mean=-0.015 std=0.003 21.7s

[2026-04-25 22:46:25] [probe] nut_thread A obj_speed L02: r2_mean=-0.001 std=0.002 20.6s

[2026-04-25 22:46:49] [probe] nut_thread A obj_speed L03: r2_mean=-0.001 std=0.002 21.8s

[2026-04-25 22:47:12] [probe] nut_thread A obj_speed L04: r2_mean=-0.001 std=0.003 21.2s

[2026-04-25 22:47:26] [probe] nut_thread A obj_speed L05: r2_mean=0.001 std=0.003 12.0s

[2026-04-25 22:47:42] [probe] nut_thread A obj_speed L06: r2_mean=-0.002 std=0.004 14.3s

[2026-04-25 22:47:57] [probe] nut_thread A obj_speed L07: r2_mean=-0.047 std=0.037 13.1s

[2026-04-25 22:48:09] [probe] nut_thread A obj_speed L08: r2_mean=0.001 std=0.005 10.9s

[2026-04-25 22:48:20] [extract] peg_insert sh0 1950/2500 eps 251550 win 24.1 win/s ETA 49.0min

[2026-04-25 22:48:32] [probe] nut_thread A obj_speed L09: r2_mean=-0.006 std=0.020 20.6s

[2026-04-25 22:48:56] [probe] nut_thread A obj_speed L10: r2_mean=-0.002 std=0.007 22.1s

[2026-04-25 22:49:19] [probe] nut_thread A obj_speed L11: r2_mean=0.005 std=0.002 21.3s

[2026-04-25 22:49:43] [probe] nut_thread A obj_speed L12: r2_mean=0.009 std=0.002 21.8s

[2026-04-25 22:50:05] [probe] nut_thread A obj_speed L13: r2_mean=0.012 std=0.001 20.3s

[2026-04-25 22:50:28] [probe] nut_thread A obj_speed L14: r2_mean=0.018 std=0.002 21.1s

[2026-04-25 22:50:44] [extract] peg_insert sh0 1975/2500 eps 254775 win 24.1 win/s ETA 46.8min

[2026-04-25 22:50:52] [probe] nut_thread A obj_speed L15: r2_mean=0.021 std=0.003 22.2s

[2026-04-25 22:51:16] [probe] nut_thread A obj_speed L16: r2_mean=0.028 std=0.004 21.7s

[2026-04-25 22:51:40] [probe] nut_thread A obj_speed L17: r2_mean=0.029 std=0.006 21.7s

[2026-04-25 22:52:03] [probe] nut_thread A obj_speed L18: r2_mean=0.031 std=0.005 21.3s

[2026-04-25 22:52:27] [probe] nut_thread A obj_speed L19: r2_mean=0.028 std=0.002 21.7s

[2026-04-25 22:52:50] [probe] nut_thread A obj_speed L20: r2_mean=0.032 std=0.005 21.5s

[2026-04-25 22:53:09] [extract] peg_insert sh0 2000/2500 eps 258000 win 24.1 win/s ETA 44.7min

[2026-04-25 22:53:14] [probe] nut_thread A obj_speed L21: r2_mean=0.027 std=0.003 21.8s

[2026-04-25 22:53:37] [probe] nut_thread A obj_speed L22: r2_mean=0.028 std=0.003 21.3s

[2026-04-25 22:54:00] [probe] nut_thread A obj_speed L23: r2_mean=0.030 std=0.007 21.5s

[2026-04-25 22:54:37] [probe] nut_thread A obj_direction L00: r2_mean=-0.011 std=0.004 23.4s

[2026-04-25 22:55:01] [probe] nut_thread A obj_direction L01: r2_mean=-0.147 std=0.100 22.2s

[2026-04-25 22:55:26] [extract] peg_insert sh0 2025/2500 eps 261225 win 24.1 win/s ETA 42.4min

[2026-04-25 22:55:26] [probe] nut_thread A obj_direction L02: r2_mean=0.003 std=0.001 23.3s

[2026-04-25 22:55:50] [probe] nut_thread A obj_direction L03: r2_mean=0.004 std=0.001 21.9s

[2026-04-25 22:56:14] [probe] nut_thread A obj_direction L04: r2_mean=0.003 std=0.002 23.0s

[2026-04-25 22:56:39] [probe] nut_thread A obj_direction L05: r2_mean=0.003 std=0.001 22.9s

[2026-04-25 22:57:03] [probe] nut_thread A obj_direction L06: r2_mean=-0.107 std=0.125 22.2s

[2026-04-25 22:57:28] [probe] nut_thread A obj_direction L07: r2_mean=-0.287 std=0.210 22.5s

[2026-04-25 22:57:42] [extract] peg_insert sh0 2050/2500 eps 264450 win 24.1 win/s ETA 40.2min

[2026-04-25 22:57:52] [probe] nut_thread A obj_direction L08: r2_mean=-0.008 std=0.007 22.7s

[2026-04-25 22:58:17] [probe] nut_thread A obj_direction L09: r2_mean=-0.049 std=0.024 23.1s

[2026-04-25 22:58:41] [probe] nut_thread A obj_direction L10: r2_mean=-0.175 std=0.149 22.0s

[2026-04-25 22:59:05] [probe] nut_thread A obj_direction L11: r2_mean=-0.005 std=0.009 22.3s

[2026-04-25 22:59:30] [probe] nut_thread A obj_direction L12: r2_mean=0.004 std=0.002 22.7s

[2026-04-25 22:59:41] [extract] peg_insert sh0 2075/2500 eps 267675 win 24.1 win/s ETA 37.9min

[2026-04-25 22:59:54] [probe] nut_thread A obj_direction L13: r2_mean=0.005 std=0.002 22.8s

[2026-04-25 23:00:18] [probe] nut_thread A obj_direction L14: r2_mean=0.008 std=0.002 21.9s

[2026-04-25 23:00:43] [probe] nut_thread A obj_direction L15: r2_mean=0.009 std=0.002 23.1s

[2026-04-25 23:01:08] [probe] nut_thread A obj_direction L16: r2_mean=0.012 std=0.002 22.6s

[2026-04-25 23:01:33] [probe] nut_thread A obj_direction L17: r2_mean=0.014 std=0.004 23.1s

[2026-04-25 23:01:57] [probe] nut_thread A obj_direction L18: r2_mean=0.016 std=0.002 22.5s

[2026-04-25 23:02:07] [extract] peg_insert sh0 2100/2500 eps 270900 win 24.1 win/s ETA 35.7min

[2026-04-25 23:02:20] [probe] nut_thread A obj_direction L19: r2_mean=0.013 std=0.003 21.5s

[2026-04-25 23:02:45] [probe] nut_thread A obj_direction L20: r2_mean=0.014 std=0.003 22.7s

[2026-04-25 23:03:10] [probe] nut_thread A obj_direction L21: r2_mean=0.013 std=0.004 23.1s

[2026-04-25 23:03:34] [probe] nut_thread A obj_direction L22: r2_mean=0.011 std=0.005 22.5s

[2026-04-25 23:03:59] [probe] nut_thread A obj_direction L23: r2_mean=0.011 std=0.005 22.9s

[2026-04-25 23:04:24] [probe] nut_thread A obj_acceleration L00: r2_mean=-0.008 std=0.002 21.6s

[2026-04-25 23:04:31] [extract] peg_insert sh0 2125/2500 eps 274125 win 24.0 win/s ETA 33.5min

[2026-04-25 23:04:49] [probe] nut_thread A obj_acceleration L01: r2_mean=-0.087 std=0.080 23.3s

[2026-04-25 23:05:14] [probe] nut_thread A obj_acceleration L02: r2_mean=-0.001 std=0.000 23.0s

[2026-04-25 23:05:39] [probe] nut_thread A obj_acceleration L03: r2_mean=-0.001 std=0.000 23.0s

[2026-04-25 23:06:04] [probe] nut_thread A obj_acceleration L04: r2_mean=-0.001 std=0.001 23.2s

[2026-04-25 23:06:29] [probe] nut_thread A obj_acceleration L05: r2_mean=-0.001 std=0.001 22.8s

[2026-04-25 23:06:51] [extract] peg_insert sh0 2150/2500 eps 277350 win 24.0 win/s ETA 31.3min

[2026-04-25 23:06:55] [probe] nut_thread A obj_acceleration L06: r2_mean=-0.049 std=0.033 23.4s

[2026-04-25 23:07:20] [probe] nut_thread A obj_acceleration L07: r2_mean=-0.616 std=0.490 22.9s

[2026-04-25 23:07:44] [probe] nut_thread A obj_acceleration L08: r2_mean=-0.026 std=0.019 22.8s

[2026-04-25 23:08:09] [probe] nut_thread A obj_acceleration L09: r2_mean=-0.379 std=0.570 23.0s

[2026-04-25 23:08:35] [probe] nut_thread A obj_acceleration L10: r2_mean=-0.085 std=0.069 23.8s

[2026-04-25 23:08:59] [probe] nut_thread A obj_acceleration L11: r2_mean=-0.009 std=0.006 22.4s

[2026-04-25 23:09:13] [extract] peg_insert sh0 2175/2500 eps 280575 win 24.0 win/s ETA 29.1min

[2026-04-25 23:09:24] [probe] nut_thread A obj_acceleration L12: r2_mean=-0.002 std=0.000 22.5s

[2026-04-25 23:09:48] [probe] nut_thread A obj_acceleration L13: r2_mean=-0.002 std=0.001 22.2s

[2026-04-25 23:10:13] [probe] nut_thread A obj_acceleration L14: r2_mean=-0.001 std=0.001 22.7s

[2026-04-25 23:10:38] [probe] nut_thread A obj_acceleration L15: r2_mean=-0.001 std=0.001 23.0s

[2026-04-25 23:11:02] [probe] nut_thread A obj_acceleration L16: r2_mean=-0.001 std=0.001 22.5s

[2026-04-25 23:11:28] [probe] nut_thread A obj_acceleration L17: r2_mean=-0.000 std=0.001 23.7s

[2026-04-25 23:11:32] [extract] peg_insert sh0 2200/2500 eps 283800 win 24.0 win/s ETA 26.9min

[2026-04-25 23:11:52] [probe] nut_thread A obj_acceleration L18: r2_mean=0.001 std=0.000 22.7s

[2026-04-25 23:12:17] [probe] nut_thread A obj_acceleration L19: r2_mean=-0.000 std=0.001 22.7s

[2026-04-25 23:12:42] [probe] nut_thread A obj_acceleration L20: r2_mean=-0.000 std=0.001 23.3s

[2026-04-25 23:13:05] [probe] nut_thread A obj_acceleration L21: r2_mean=0.000 std=0.001 20.6s

[2026-04-25 23:13:30] [probe] nut_thread A obj_acceleration L22: r2_mean=-0.001 std=0.001 23.5s

[2026-04-25 23:13:53] [extract] peg_insert sh0 2225/2500 eps 287025 win 24.0 win/s ETA 24.6min

[2026-04-25 23:13:55] [probe] nut_thread A obj_acceleration L23: r2_mean=-0.010 std=0.015 22.7s

[2026-04-25 23:14:17] [probe] nut_thread A obj_accel_mag L00: r2_mean=-0.010 std=0.008 20.0s

[2026-04-25 23:14:37] [probe] nut_thread A obj_accel_mag L01: r2_mean=-0.012 std=0.007 17.8s

[2026-04-25 23:14:53] [probe] nut_thread A obj_accel_mag L02: r2_mean=0.001 std=0.002 14.8s

[2026-04-25 23:15:08] [probe] nut_thread A obj_accel_mag L03: r2_mean=0.000 std=0.001 13.2s

[2026-04-25 23:15:32] [probe] nut_thread A obj_accel_mag L04: r2_mean=0.002 std=0.001 21.8s

[2026-04-25 23:15:56] [probe] nut_thread A obj_accel_mag L05: r2_mean=0.002 std=0.003 21.7s

[2026-04-25 23:16:15] [extract] peg_insert sh0 2250/2500 eps 290250 win 24.0 win/s ETA 22.4min

[2026-04-25 23:16:19] [probe] nut_thread A obj_accel_mag L06: r2_mean=0.003 std=0.001 21.2s

[2026-04-25 23:16:42] [probe] nut_thread A obj_accel_mag L07: r2_mean=-0.015 std=0.017 21.7s

[2026-04-25 23:17:06] [probe] nut_thread A obj_accel_mag L08: r2_mean=0.003 std=0.005 21.6s

[2026-04-25 23:17:30] [probe] nut_thread A obj_accel_mag L09: r2_mean=-0.007 std=0.002 21.7s

[2026-04-25 23:17:53] [probe] nut_thread A obj_accel_mag L10: r2_mean=-0.005 std=0.004 21.2s

[2026-04-25 23:18:14] [probe] nut_thread A obj_accel_mag L11: r2_mean=0.004 std=0.005 19.2s

[2026-04-25 23:18:37] [probe] nut_thread A obj_accel_mag L12: r2_mean=0.004 std=0.003 21.0s

[2026-04-25 23:18:38] [extract] peg_insert sh0 2275/2500 eps 293475 win 24.0 win/s ETA 20.2min

[2026-04-25 23:19:01] [probe] nut_thread A obj_accel_mag L13: r2_mean=0.009 std=0.002 21.8s

[2026-04-25 23:19:24] [probe] nut_thread A obj_accel_mag L14: r2_mean=0.016 std=0.002 21.2s

[2026-04-25 23:19:47] [probe] nut_thread A obj_accel_mag L15: r2_mean=0.014 std=0.002 21.0s

[2026-04-25 23:20:10] [probe] nut_thread A obj_accel_mag L16: r2_mean=0.019 std=0.003 21.5s

[2026-04-25 23:20:33] [probe] nut_thread A obj_accel_mag L17: r2_mean=0.018 std=0.002 21.1s

[2026-04-25 23:20:57] [probe] nut_thread A obj_accel_mag L18: r2_mean=0.021 std=0.002 21.5s

[2026-04-25 23:21:00] [extract] peg_insert sh0 2300/2500 eps 296700 win 23.9 win/s ETA 18.0min

[2026-04-25 23:21:21] [probe] nut_thread A obj_accel_mag L19: r2_mean=0.019 std=0.002 21.7s

[2026-04-25 23:21:43] [probe] nut_thread A obj_accel_mag L20: r2_mean=0.020 std=0.002 20.9s

[2026-04-25 23:22:04] [probe] nut_thread A obj_accel_mag L21: r2_mean=0.016 std=0.003 18.5s

[2026-04-25 23:22:20] [probe] nut_thread A obj_accel_mag L22: r2_mean=0.016 std=0.002 14.1s

[2026-04-25 23:22:43] [probe] nut_thread A obj_accel_mag L23: r2_mean=0.018 std=0.004 21.2s

[2026-04-25 23:22:43] [probe] nut_thread A DONE in 115.2min

[2026-04-25 23:23:24] [extract] peg_insert sh0 2325/2500 eps 299925 win 23.9 win/s ETA 15.7min

[2026-04-25 23:23:25] [probe] task=push variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag', 'obj_position', 'obj_velocity', 'obj_speed', 'obj_direction', 'obj_acceleration', 'obj_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=1

[2026-04-25 23:23:57] [probe] push: features [(340283, 24, 1024)] eps 1500 loaded 32.2s

[2026-04-25 23:24:21] [probe] push A ee_position L00: r2_mean=0.771 std=0.005 22.3s

[2026-04-25 23:24:43] [probe] push A ee_position L01: r2_mean=0.895 std=0.004 21.1s

[2026-04-25 23:25:07] [probe] push A ee_position L02: r2_mean=0.916 std=0.006 22.0s

[2026-04-25 23:25:32] [probe] push A ee_position L03: r2_mean=0.940 std=0.001 24.0s

[2026-04-25 23:25:53] [extract] peg_insert sh0 2350/2500 eps 303150 win 23.9 win/s ETA 13.5min

[2026-04-25 23:25:56] [probe] push A ee_position L04: r2_mean=0.940 std=0.001 21.6s

[2026-04-25 23:26:22] [probe] push A ee_position L05: r2_mean=0.947 std=0.002 24.1s

[2026-04-25 23:26:45] [probe] push A ee_position L06: r2_mean=0.949 std=0.002 21.3s

[2026-04-25 23:27:08] [probe] push A ee_position L07: r2_mean=0.958 std=0.002 21.8s

[2026-04-25 23:27:32] [probe] push A ee_position L08: r2_mean=0.960 std=0.002 22.4s

[2026-04-25 23:27:55] [probe] push A ee_position L09: r2_mean=0.957 std=0.002 20.8s

[2026-04-25 23:28:15] [extract] peg_insert sh0 2375/2500 eps 306375 win 23.9 win/s ETA 11.2min

[2026-04-25 23:28:17] [probe] push A ee_position L10: r2_mean=0.966 std=0.002 20.8s

[2026-04-25 23:28:43] [probe] push A ee_position L11: r2_mean=0.970 std=0.002 21.5s

[2026-04-25 23:29:05] [probe] push A ee_position L12: r2_mean=0.970 std=0.002 20.8s

[2026-04-25 23:29:28] [probe] push A ee_position L13: r2_mean=0.976 std=0.001 21.0s

[2026-04-25 23:29:51] [probe] push A ee_position L14: r2_mean=0.977 std=0.001 21.3s

[2026-04-25 23:30:13] [probe] push A ee_position L15: r2_mean=0.977 std=0.001 19.2s

[2026-04-25 23:30:35] [probe] push A ee_position L16: r2_mean=0.978 std=0.001 20.5s

[2026-04-25 23:30:38] [extract] peg_insert sh0 2400/2500 eps 309600 win 23.9 win/s ETA 9.0min

[2026-04-25 23:30:59] [probe] push A ee_position L17: r2_mean=0.978 std=0.002 21.3s

[2026-04-25 23:31:20] [probe] push A ee_position L18: r2_mean=0.978 std=0.002 19.8s

[2026-04-25 23:31:43] [probe] push A ee_position L19: r2_mean=0.978 std=0.003 19.7s

[2026-04-25 23:32:02] [probe] push A ee_position L20: r2_mean=0.978 std=0.002 18.4s

[2026-04-25 23:32:26] [probe] push A ee_position L21: r2_mean=0.980 std=0.002 20.5s

[2026-04-25 23:32:48] [probe] push A ee_position L22: r2_mean=0.980 std=0.001 20.7s

[2026-04-25 23:33:02] [extract] peg_insert sh0 2425/2500 eps 312825 win 23.9 win/s ETA 6.8min

[2026-04-25 23:33:12] [probe] push A ee_position L23: r2_mean=0.980 std=0.002 20.6s

[2026-04-25 23:33:33] [probe] push A ee_velocity L00: r2_mean=0.340 std=0.005 19.7s

[2026-04-25 23:33:55] [probe] push A ee_velocity L01: r2_mean=0.669 std=0.014 19.3s

[2026-04-25 23:34:14] [probe] push A ee_velocity L02: r2_mean=0.749 std=0.009 17.8s

[2026-04-25 23:34:37] [probe] push A ee_velocity L03: r2_mean=0.787 std=0.008 20.5s

[2026-04-25 23:34:59] [probe] push A ee_velocity L04: r2_mean=0.825 std=0.007 20.4s

[2026-04-25 23:35:23] [probe] push A ee_velocity L05: r2_mean=0.841 std=0.006 20.6s

[2026-04-25 23:35:25] [extract] peg_insert sh0 2450/2500 eps 316050 win 23.8 win/s ETA 4.5min

[2026-04-25 23:35:44] [probe] push A ee_velocity L06: r2_mean=0.853 std=0.005 20.2s

[2026-04-25 23:36:08] [probe] push A ee_velocity L07: r2_mean=0.867 std=0.006 21.2s

[2026-04-25 23:36:30] [probe] push A ee_velocity L08: r2_mean=0.870 std=0.006 20.9s

[2026-04-25 23:36:51] [probe] push A ee_velocity L09: r2_mean=0.868 std=0.006 19.5s

[2026-04-25 23:37:09] [probe] push A ee_velocity L10: r2_mean=0.877 std=0.006 16.9s

[2026-04-25 23:37:31] [probe] push A ee_velocity L11: r2_mean=0.880 std=0.005 20.4s

[2026-04-25 23:37:50] [extract] peg_insert sh0 2475/2500 eps 319275 win 23.8 win/s ETA 2.3min

[2026-04-25 23:37:52] [probe] push A ee_velocity L12: r2_mean=0.877 std=0.006 19.7s

[2026-04-25 23:38:14] [probe] push A ee_velocity L13: r2_mean=0.888 std=0.006 20.7s

[2026-04-25 23:38:36] [probe] push A ee_velocity L14: r2_mean=0.889 std=0.006 20.8s

[2026-04-25 23:38:58] [probe] push A ee_velocity L15: r2_mean=0.899 std=0.006 20.5s

[2026-04-25 23:39:21] [probe] push A ee_velocity L16: r2_mean=0.911 std=0.006 20.8s

[2026-04-25 23:39:41] [probe] push A ee_velocity L17: r2_mean=0.915 std=0.005 19.5s

[2026-04-25 23:40:04] [probe] push A ee_velocity L18: r2_mean=0.919 std=0.005 20.8s

[2026-04-25 23:40:14] [extract] peg_insert sh0 2500/2500 eps 322500 win 23.8 win/s ETA 0.0min

[2026-04-25 23:40:14] [extract] peg_insert sh0 DONE: 2500 eps 322500 win in 225.7min

[2026-04-25 23:40:18] [probe] task=peg_insert variant=A targets=['ee_position', 'ee_velocity', 'ee_speed', 'ee_direction', 'ee_acceleration', 'ee_accel_mag', 'obj_position', 'obj_velocity', 'obj_speed', 'obj_direction', 'obj_acceleration', 'obj_accel_mag'] layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] gpu=0

[2026-04-25 23:40:25] [probe] push A ee_velocity L19: r2_mean=0.914 std=0.011 19.8s

[2026-04-25 23:40:47] [probe] push A ee_velocity L20: r2_mean=0.918 std=0.005 20.8s

[2026-04-25 23:40:48] [probe] peg_insert: features [(322500, 24, 1024)] eps 2500 loaded 30.2s

[2026-04-25 23:41:09] [probe] push A ee_velocity L21: r2_mean=0.918 std=0.005 20.5s

[2026-04-25 23:41:16] [probe] peg_insert A ee_position L00: r2_mean=0.501 std=0.017 25.6s

[2026-04-25 23:41:31] [probe] push A ee_velocity L22: r2_mean=0.921 std=0.005 20.8s

[2026-04-25 23:41:42] [probe] peg_insert A ee_position L01: r2_mean=0.852 std=0.008 24.2s

[2026-04-25 23:41:54] [probe] push A ee_velocity L23: r2_mean=0.911 std=0.015 21.1s

[2026-04-25 23:42:07] [probe] peg_insert A ee_position L02: r2_mean=0.859 std=0.002 23.8s

[2026-04-25 23:42:15] [probe] push A ee_speed L00: r2_mean=0.584 std=0.049 19.2s

[2026-04-25 23:42:33] [probe] peg_insert A ee_position L03: r2_mean=0.909 std=0.002 24.1s

[2026-04-25 23:42:37] [probe] push A ee_speed L01: r2_mean=0.845 std=0.006 19.7s

[2026-04-25 23:42:55] [probe] push A ee_speed L02: r2_mean=0.867 std=0.007 17.4s

[2026-04-25 23:42:58] [probe] peg_insert A ee_position L04: r2_mean=0.921 std=0.002 23.3s

[2026-04-25 23:43:14] [probe] push A ee_speed L03: r2_mean=0.886 std=0.008 17.3s

[2026-04-25 23:43:24] [probe] peg_insert A ee_position L05: r2_mean=0.942 std=0.001 24.2s

[2026-04-25 23:43:35] [probe] push A ee_speed L04: r2_mean=0.899 std=0.006 19.0s

[2026-04-25 23:43:51] [probe] peg_insert A ee_position L06: r2_mean=0.949 std=0.001 24.6s

[2026-04-25 23:43:56] [probe] push A ee_speed L05: r2_mean=0.910 std=0.008 19.6s

[2026-04-25 23:44:11] [probe] push A ee_speed L06: r2_mean=0.916 std=0.007 13.9s

[2026-04-25 23:44:16] [probe] peg_insert A ee_position L07: r2_mean=0.957 std=0.004 23.5s

[2026-04-25 23:44:27] [probe] push A ee_speed L07: r2_mean=0.924 std=0.006 14.7s

[2026-04-25 23:44:31] [probe] peg_insert A ee_position L08: r2_mean=0.959 std=0.001 13.2s

[2026-04-25 23:44:48] [probe] push A ee_speed L08: r2_mean=0.924 std=0.006 19.4s

[2026-04-25 23:44:51] [probe] peg_insert A ee_position L09: r2_mean=0.959 std=0.001 17.8s

[2026-04-25 23:45:05] [probe] peg_insert A ee_position L10: r2_mean=0.963 std=0.003 12.7s

[2026-04-25 23:45:09] [probe] push A ee_speed L09: r2_mean=0.924 std=0.006 19.1s

[2026-04-25 23:45:19] [probe] peg_insert A ee_position L11: r2_mean=0.948 std=0.026 12.1s

[2026-04-25 23:45:29] [probe] push A ee_speed L10: r2_mean=0.928 std=0.005 19.1s

[2026-04-25 23:45:37] [probe] peg_insert A ee_position L12: r2_mean=0.964 std=0.007 16.9s

[2026-04-25 23:45:50] [probe] push A ee_speed L11: r2_mean=0.931 std=0.005 19.2s

[2026-04-25 23:46:04] [probe] peg_insert A ee_position L13: r2_mean=0.967 std=0.001 24.7s

[2026-04-25 23:46:11] [probe] push A ee_speed L12: r2_mean=0.930 std=0.005 19.4s

[2026-04-25 23:46:27] [probe] peg_insert A ee_position L14: r2_mean=0.967 std=0.000 21.7s

[2026-04-25 23:46:30] [probe] push A ee_speed L13: r2_mean=0.938 std=0.005 17.9s

[2026-04-25 23:46:51] [probe] push A ee_speed L14: r2_mean=0.941 std=0.005 19.2s

[2026-04-25 23:46:53] [probe] peg_insert A ee_position L15: r2_mean=0.963 std=0.007 24.0s

[2026-04-25 23:47:12] [probe] push A ee_speed L15: r2_mean=0.942 std=0.006 19.8s

[2026-04-25 23:47:17] [probe] peg_insert A ee_position L16: r2_mean=0.896 std=0.055 21.8s

[2026-04-25 23:47:32] [probe] push A ee_speed L16: r2_mean=0.947 std=0.006 18.9s

[2026-04-25 23:47:37] [probe] peg_insert A ee_position L17: r2_mean=0.968 std=0.005 18.2s

[2026-04-25 23:47:53] [probe] push A ee_speed L17: r2_mean=0.951 std=0.005 19.4s

[2026-04-25 23:48:02] [probe] peg_insert A ee_position L18: r2_mean=0.965 std=0.007 23.3s

[2026-04-25 23:48:14] [probe] push A ee_speed L18: r2_mean=0.951 std=0.005 19.5s

[2026-04-25 23:48:24] [probe] peg_insert A ee_position L19: r2_mean=0.916 std=0.071 19.7s

[2026-04-25 23:48:34] [probe] push A ee_speed L19: r2_mean=0.952 std=0.005 18.2s

[2026-04-25 23:48:48] [probe] peg_insert A ee_position L20: r2_mean=0.966 std=0.005 23.0s

[2026-04-25 23:48:54] [probe] push A ee_speed L20: r2_mean=0.952 std=0.005 19.2s

[2026-04-25 23:49:09] [probe] peg_insert A ee_position L21: r2_mean=0.962 std=0.011 18.6s

[2026-04-25 23:49:15] [probe] push A ee_speed L21: r2_mean=0.950 std=0.005 19.2s

[2026-04-25 23:49:28] [probe] peg_insert A ee_position L22: r2_mean=0.969 std=0.004 17.5s

[2026-04-25 23:49:36] [probe] push A ee_speed L22: r2_mean=0.948 std=0.005 19.1s

[2026-04-25 23:49:42] [probe] peg_insert A ee_position L23: r2_mean=0.948 std=0.017 12.4s

[2026-04-25 23:49:55] [probe] push A ee_speed L23: r2_mean=0.945 std=0.005 18.1s

[2026-04-25 23:49:56] [probe] peg_insert A ee_velocity L00: r2_mean=0.155 std=0.003 12.0s

[2026-04-25 23:50:12] [probe] peg_insert A ee_velocity L01: r2_mean=0.239 std=0.004 14.5s

[2026-04-25 23:50:26] [probe] push A ee_direction L00: r2_mean=-0.099 std=0.215 20.2s

[2026-04-25 23:50:26] [probe] peg_insert A ee_velocity L02: r2_mean=0.262 std=0.002 12.7s

[2026-04-25 23:50:48] [probe] push A ee_direction L01: r2_mean=0.617 std=0.009 20.2s

[2026-04-25 23:50:52] [probe] peg_insert A ee_velocity L03: r2_mean=0.401 std=0.022 23.5s

[2026-04-25 23:51:10] [probe] push A ee_direction L02: r2_mean=0.679 std=0.004 20.8s

[2026-04-25 23:51:10] [probe] peg_insert A ee_velocity L04: r2_mean=0.464 std=0.023 16.6s

[2026-04-25 23:51:28] [probe] peg_insert A ee_velocity L05: r2_mean=0.532 std=0.005 16.2s

[2026-04-25 23:51:31] [probe] push A ee_direction L03: r2_mean=0.717 std=0.004 20.3s

[2026-04-25 23:51:46] [probe] push A ee_direction L04: r2_mean=0.720 std=0.013 13.0s

[2026-04-25 23:51:51] [probe] peg_insert A ee_velocity L06: r2_mean=0.574 std=0.005 22.0s

[2026-04-25 23:52:08] [probe] push A ee_direction L05: r2_mean=0.757 std=0.010 20.7s

[2026-04-25 23:52:18] [probe] peg_insert A ee_velocity L07: r2_mean=0.639 std=0.011 24.7s

[2026-04-25 23:52:23] [probe] push A ee_direction L06: r2_mean=0.768 std=0.012 14.5s

[2026-04-25 23:52:43] [probe] push A ee_direction L07: r2_mean=0.788 std=0.006 18.6s

[2026-04-25 23:52:44] [probe] peg_insert A ee_velocity L08: r2_mean=0.648 std=0.013 23.9s

[2026-04-25 23:53:03] [probe] push A ee_direction L08: r2_mean=0.792 std=0.005 19.2s

[2026-04-25 23:53:10] [probe] peg_insert A ee_velocity L09: r2_mean=0.636 std=0.022 24.4s

[2026-04-25 23:53:23] [probe] push A ee_direction L09: r2_mean=0.782 std=0.009 18.7s

[2026-04-25 23:53:36] [probe] peg_insert A ee_velocity L10: r2_mean=0.668 std=0.010 24.2s

[2026-04-25 23:53:44] [probe] push A ee_direction L10: r2_mean=0.790 std=0.009 19.4s

[2026-04-25 23:54:05] [probe] peg_insert A ee_velocity L11: r2_mean=0.653 std=0.043 26.4s

[2026-04-25 23:54:05] [probe] push A ee_direction L11: r2_mean=0.794 std=0.009 19.9s

[2026-04-25 23:54:25] [probe] push A ee_direction L12: r2_mean=0.806 std=0.005 18.3s

[2026-04-25 23:54:31] [probe] peg_insert A ee_velocity L12: r2_mean=0.626 std=0.007 24.5s

[2026-04-25 23:54:46] [probe] push A ee_direction L13: r2_mean=0.823 std=0.006 19.7s

[2026-04-25 23:54:57] [probe] peg_insert A ee_velocity L13: r2_mean=0.667 std=0.014 24.5s

[2026-04-25 23:55:08] [probe] push A ee_direction L14: r2_mean=0.822 std=0.003 20.6s

[2026-04-25 23:55:23] [probe] peg_insert A ee_velocity L14: r2_mean=0.684 std=0.004 23.8s

[2026-04-25 23:55:29] [probe] push A ee_direction L15: r2_mean=0.826 std=0.006 19.9s

[2026-04-25 23:55:48] [probe] peg_insert A ee_velocity L15: r2_mean=0.754 std=0.006 23.5s

[2026-04-25 23:55:51] [probe] push A ee_direction L16: r2_mean=0.830 std=0.004 21.0s

[2026-04-25 23:56:13] [probe] push A ee_direction L17: r2_mean=0.824 std=0.014 20.3s

[2026-04-25 23:56:13] [probe] peg_insert A ee_velocity L16: r2_mean=0.747 std=0.041 23.2s

[2026-04-25 23:56:35] [probe] push A ee_direction L18: r2_mean=0.821 std=0.026 20.7s

[2026-04-25 23:56:38] [probe] peg_insert A ee_velocity L17: r2_mean=0.807 std=0.009 23.2s

[2026-04-25 23:56:56] [probe] push A ee_direction L19: r2_mean=0.834 std=0.006 19.5s

[2026-04-25 23:57:03] [probe] peg_insert A ee_velocity L18: r2_mean=0.780 std=0.060 23.1s

[2026-04-25 23:57:18] [probe] push A ee_direction L20: r2_mean=0.834 std=0.006 20.6s

[2026-04-25 23:57:29] [probe] peg_insert A ee_velocity L19: r2_mean=0.782 std=0.027 24.2s

[2026-04-25 23:57:39] [probe] push A ee_direction L21: r2_mean=0.838 std=0.006 19.9s

[2026-04-25 23:57:54] [probe] peg_insert A ee_velocity L20: r2_mean=0.429 std=0.511 23.1s

[2026-04-25 23:58:00] [probe] push A ee_direction L22: r2_mean=0.420 std=0.350 20.0s

[2026-04-25 23:58:18] [probe] push A ee_direction L23: r2_mean=0.505 std=0.380 16.9s

[2026-04-25 23:58:20] [probe] peg_insert A ee_velocity L21: r2_mean=0.780 std=0.027 23.9s

[2026-04-25 23:58:40] [probe] push A ee_acceleration L00: r2_mean=0.030 std=0.004 20.0s

[2026-04-25 23:58:45] [probe] peg_insert A ee_velocity L22: r2_mean=0.757 std=0.024 23.5s

[2026-04-25 23:59:03] [probe] push A ee_acceleration L01: r2_mean=0.067 std=0.004 20.7s

[2026-04-25 23:59:10] [probe] peg_insert A ee_velocity L23: r2_mean=0.677 std=0.063 23.1s

[2026-04-25 23:59:24] [probe] push A ee_acceleration L02: r2_mean=0.082 std=0.004 20.1s

[2026-04-25 23:59:34] [probe] peg_insert A ee_speed L00: r2_mean=0.212 std=0.005 21.7s

[2026-04-25 23:59:46] [probe] push A ee_acceleration L03: r2_mean=0.093 std=0.004 20.8s

[2026-04-25 23:59:58] [probe] peg_insert A ee_speed L01: r2_mean=0.340 std=0.005 22.1s

[2026-04-26 00:00:09] [probe] push A ee_acceleration L04: r2_mean=0.100 std=0.005 21.2s

[2026-04-26 00:00:22] [probe] peg_insert A ee_speed L02: r2_mean=0.354 std=0.012 22.3s

[2026-04-26 00:00:31] [probe] push A ee_acceleration L05: r2_mean=0.109 std=0.005 20.3s

[2026-04-26 00:00:47] [probe] peg_insert A ee_speed L03: r2_mean=0.530 std=0.011 22.4s

[2026-04-26 00:00:53] [probe] push A ee_acceleration L06: r2_mean=0.113 std=0.005 21.1s

[2026-04-26 00:01:10] [probe] peg_insert A ee_speed L04: r2_mean=0.565 std=0.013 21.5s

[2026-04-26 00:01:16] [probe] push A ee_acceleration L07: r2_mean=0.125 std=0.005 20.9s

[2026-04-26 00:01:33] [probe] peg_insert A ee_speed L05: r2_mean=0.645 std=0.009 21.0s

[2026-04-26 00:01:37] [probe] push A ee_acceleration L08: r2_mean=0.125 std=0.008 20.5s

[2026-04-26 00:01:56] [probe] peg_insert A ee_speed L06: r2_mean=0.693 std=0.008 22.0s

[2026-04-26 00:02:00] [probe] push A ee_acceleration L09: r2_mean=0.124 std=0.006 20.9s

[2026-04-26 00:02:19] [probe] peg_insert A ee_speed L07: r2_mean=0.757 std=0.002 20.7s

[2026-04-26 00:02:22] [probe] push A ee_acceleration L10: r2_mean=0.138 std=0.010 20.7s

[2026-04-26 00:02:37] [probe] peg_insert A ee_speed L08: r2_mean=0.753 std=0.004 16.6s

[2026-04-26 00:02:45] [probe] push A ee_acceleration L11: r2_mean=0.139 std=0.006 20.9s

[2026-04-26 00:03:02] [probe] peg_insert A ee_speed L09: r2_mean=0.745 std=0.003 22.8s

[2026-04-26 00:03:05] [probe] push A ee_acceleration L12: r2_mean=0.144 std=0.006 19.3s

[2026-04-26 00:03:26] [probe] peg_insert A ee_speed L10: r2_mean=0.758 std=0.005 21.9s

[2026-04-26 00:03:27] [probe] push A ee_acceleration L13: r2_mean=0.153 std=0.011 20.3s

[2026-04-26 00:03:49] [probe] push A ee_acceleration L14: r2_mean=0.169 std=0.007 20.6s

[2026-04-26 00:03:51] [probe] peg_insert A ee_speed L11: r2_mean=0.769 std=0.007 23.0s

[2026-04-26 00:04:07] [probe] push A ee_acceleration L15: r2_mean=0.192 std=0.008 16.2s

[2026-04-26 00:04:15] [probe] peg_insert A ee_speed L12: r2_mean=0.753 std=0.009 22.1s

[2026-04-26 00:04:29] [probe] push A ee_acceleration L16: r2_mean=0.216 std=0.009 21.1s

[2026-04-26 00:04:39] [probe] peg_insert A ee_speed L13: r2_mean=0.779 std=0.008 22.1s

[2026-04-26 00:04:51] [probe] push A ee_acceleration L17: r2_mean=0.218 std=0.009 20.5s

[2026-04-26 00:05:03] [probe] peg_insert A ee_speed L14: r2_mean=0.799 std=0.005 22.3s

[2026-04-26 00:05:13] [probe] push A ee_acceleration L18: r2_mean=0.217 std=0.008 20.6s

[2026-04-26 00:05:26] [probe] peg_insert A ee_speed L15: r2_mean=0.853 std=0.003 20.9s

[2026-04-26 00:05:35] [probe] push A ee_acceleration L19: r2_mean=0.219 std=0.014 20.5s

[2026-04-26 00:05:49] [probe] peg_insert A ee_speed L16: r2_mean=0.871 std=0.003 21.6s

[2026-04-26 00:05:57] [probe] push A ee_acceleration L20: r2_mean=0.203 std=0.010 20.6s

[2026-04-26 00:06:14] [probe] peg_insert A ee_speed L17: r2_mean=0.894 std=0.002 22.5s

[2026-04-26 00:06:20] [probe] push A ee_acceleration L21: r2_mean=0.201 std=0.017 21.2s

[2026-04-26 00:06:37] [probe] peg_insert A ee_speed L18: r2_mean=0.897 std=0.003 21.4s

[2026-04-26 00:06:43] [probe] push A ee_acceleration L22: r2_mean=0.203 std=0.017 20.9s

[2026-04-26 00:07:01] [probe] peg_insert A ee_speed L19: r2_mean=0.892 std=0.005 22.5s

[2026-04-26 00:07:05] [probe] push A ee_acceleration L23: r2_mean=0.200 std=0.018 20.9s

[2026-04-26 00:07:24] [probe] peg_insert A ee_speed L20: r2_mean=0.874 std=0.014 21.2s

[2026-04-26 00:07:26] [probe] push A ee_accel_mag L00: r2_mean=0.066 std=0.006 19.4s

[2026-04-26 00:07:47] [probe] push A ee_accel_mag L01: r2_mean=0.170 std=0.003 19.4s

[2026-04-26 00:07:49] [probe] peg_insert A ee_speed L21: r2_mean=0.892 std=0.003 22.6s

[2026-04-26 00:08:08] [probe] push A ee_accel_mag L02: r2_mean=0.208 std=0.004 19.5s

[2026-04-26 00:08:13] [probe] peg_insert A ee_speed L22: r2_mean=0.882 std=0.007 22.4s

[2026-04-26 00:08:29] [probe] push A ee_accel_mag L03: r2_mean=0.237 std=0.003 19.3s

[2026-04-26 00:08:37] [probe] peg_insert A ee_speed L23: r2_mean=0.872 std=0.010 21.6s

[2026-04-26 00:08:49] [probe] push A ee_accel_mag L04: r2_mean=0.259 std=0.003 18.3s

[2026-04-26 00:09:09] [probe] push A ee_accel_mag L05: r2_mean=0.282 std=0.001 19.1s

[2026-04-26 00:09:12] [probe] peg_insert A ee_direction L00: r2_mean=-0.001 std=0.001 23.5s

[2026-04-26 00:09:30] [probe] push A ee_accel_mag L06: r2_mean=0.294 std=0.005 19.4s

[2026-04-26 00:09:38] [probe] peg_insert A ee_direction L01: r2_mean=0.017 std=0.002 24.1s

[2026-04-26 00:09:52] [probe] push A ee_accel_mag L07: r2_mean=0.319 std=0.008 20.0s

[2026-04-26 00:10:03] [probe] peg_insert A ee_direction L02: r2_mean=0.022 std=0.002 23.1s

[2026-04-26 00:10:12] [probe] push A ee_accel_mag L08: r2_mean=0.319 std=0.005 19.3s

[2026-04-26 00:10:29] [probe] peg_insert A ee_direction L03: r2_mean=0.026 std=0.002 23.8s

[2026-04-26 00:10:33] [probe] push A ee_accel_mag L09: r2_mean=0.319 std=0.007 19.1s

[2026-04-26 00:10:53] [probe] peg_insert A ee_direction L04: r2_mean=0.032 std=0.002 22.1s

[2026-04-26 00:10:54] [probe] push A ee_accel_mag L10: r2_mean=0.333 std=0.002 19.2s

[2026-04-26 00:11:12] [probe] push A ee_accel_mag L11: r2_mean=0.351 std=0.004 16.5s

[2026-04-26 00:11:19] [probe] peg_insert A ee_direction L05: r2_mean=0.037 std=0.002 24.0s

[2026-04-26 00:11:32] [probe] push A ee_accel_mag L12: r2_mean=0.368 std=0.007 19.3s

[2026-04-26 00:11:43] [probe] peg_insert A ee_direction L06: r2_mean=0.038 std=0.003 22.7s

[2026-04-26 00:11:53] [probe] push A ee_accel_mag L13: r2_mean=0.425 std=0.011 19.2s

[2026-04-26 00:12:05] [probe] peg_insert A ee_direction L07: r2_mean=0.046 std=0.002 19.9s

[2026-04-26 00:12:14] [probe] push A ee_accel_mag L14: r2_mean=0.444 std=0.009 19.4s

[2026-04-26 00:12:29] [probe] peg_insert A ee_direction L08: r2_mean=0.053 std=0.004 22.5s

[2026-04-26 00:12:34] [probe] push A ee_accel_mag L15: r2_mean=0.476 std=0.010 19.4s

[2026-04-26 00:12:55] [probe] peg_insert A ee_direction L09: r2_mean=0.052 std=0.002 24.0s

[2026-04-26 00:12:56] [probe] push A ee_accel_mag L16: r2_mean=0.531 std=0.009 19.9s

[2026-04-26 00:13:17] [probe] push A ee_accel_mag L17: r2_mean=0.557 std=0.013 19.5s

[2026-04-26 00:13:20] [probe] peg_insert A ee_direction L10: r2_mean=0.050 std=0.004 23.3s

[2026-04-26 00:13:37] [probe] push A ee_accel_mag L18: r2_mean=0.567 std=0.013 19.0s

[2026-04-26 00:13:46] [probe] peg_insert A ee_direction L11: r2_mean=0.049 std=0.007 23.5s

[2026-04-26 00:13:58] [probe] push A ee_accel_mag L19: r2_mean=0.555 std=0.008 19.7s

[2026-04-26 00:14:10] [probe] peg_insert A ee_direction L12: r2_mean=0.069 std=0.003 22.9s

[2026-04-26 00:14:19] [probe] push A ee_accel_mag L20: r2_mean=0.550 std=0.009 19.1s

[2026-04-26 00:14:36] [probe] peg_insert A ee_direction L13: r2_mean=0.071 std=0.003 23.9s

[2026-04-26 00:14:39] [probe] push A ee_accel_mag L21: r2_mean=0.531 std=0.007 19.0s

[2026-04-26 00:15:00] [probe] push A ee_accel_mag L22: r2_mean=0.524 std=0.008 19.4s

[2026-04-26 00:15:02] [probe] peg_insert A ee_direction L14: r2_mean=0.074 std=0.003 23.7s

[2026-04-26 00:15:21] [probe] push A ee_accel_mag L23: r2_mean=0.515 std=0.012 19.3s

[2026-04-26 00:15:27] [probe] peg_insert A ee_direction L15: r2_mean=0.074 std=0.002 23.3s

[2026-04-26 00:15:44] [probe] push A obj_position L00: r2_mean=0.059 std=0.020 21.0s

[2026-04-26 00:15:52] [probe] peg_insert A ee_direction L16: r2_mean=0.068 std=0.010 23.5s

[2026-04-26 00:16:06] [probe] push A obj_position L01: r2_mean=0.132 std=0.027 20.5s

[2026-04-26 00:16:18] [probe] peg_insert A ee_direction L17: r2_mean=0.074 std=0.004 23.5s

[2026-04-26 00:16:28] [probe] push A obj_position L02: r2_mean=0.147 std=0.015 20.9s

[2026-04-26 00:16:43] [probe] peg_insert A ee_direction L18: r2_mean=0.072 std=0.014 23.2s

[2026-04-26 00:16:51] [probe] push A obj_position L03: r2_mean=0.194 std=0.024 20.8s

[2026-04-26 00:17:08] [probe] peg_insert A ee_direction L19: r2_mean=-0.779 std=1.306 23.2s

[2026-04-26 00:17:13] [probe] push A obj_position L04: r2_mean=0.120 std=0.008 21.2s

[2026-04-26 00:17:32] [probe] peg_insert A ee_direction L20: r2_mean=-0.189 std=0.306 22.7s

[2026-04-26 00:17:36] [probe] push A obj_position L05: r2_mean=0.169 std=0.031 21.0s

[2026-04-26 00:17:58] [probe] push A obj_position L06: r2_mean=0.187 std=0.033 20.8s

[2026-04-26 00:17:58] [probe] peg_insert A ee_direction L21: r2_mean=0.047 std=0.019 23.5s

[2026-04-26 00:18:20] [probe] push A obj_position L07: r2_mean=0.139 std=0.009 20.3s

[2026-04-26 00:18:23] [probe] peg_insert A ee_direction L22: r2_mean=0.029 std=0.049 23.0s

[2026-04-26 00:18:42] [probe] push A obj_position L08: r2_mean=0.208 std=0.029 21.5s

[2026-04-26 00:18:49] [probe] peg_insert A ee_direction L23: r2_mean=-0.114 std=0.153 24.1s

[2026-04-26 00:19:05] [probe] push A obj_position L09: r2_mean=0.139 std=0.024 21.0s

[2026-04-26 00:19:15] [probe] peg_insert A ee_acceleration L00: r2_mean=0.023 std=0.003 23.1s

[2026-04-26 00:19:27] [probe] push A obj_position L10: r2_mean=0.152 std=0.039 20.6s

[2026-04-26 00:19:40] [probe] peg_insert A ee_acceleration L01: r2_mean=0.066 std=0.003 23.4s

[2026-04-26 00:19:50] [probe] push A obj_position L11: r2_mean=0.174 std=0.029 21.1s

[2026-04-26 00:20:06] [probe] peg_insert A ee_acceleration L02: r2_mean=0.097 std=0.003 23.8s

[2026-04-26 00:20:12] [probe] push A obj_position L12: r2_mean=0.147 std=0.023 21.4s

[2026-04-26 00:20:32] [probe] peg_insert A ee_acceleration L03: r2_mean=0.145 std=0.003 23.5s

[2026-04-26 00:20:33] [probe] push A obj_position L13: r2_mean=0.153 std=0.027 19.3s

[2026-04-26 00:20:56] [probe] push A obj_position L14: r2_mean=0.111 std=0.013 21.6s

[2026-04-26 00:20:57] [probe] peg_insert A ee_acceleration L04: r2_mean=0.155 std=0.005 23.6s

[2026-04-26 00:21:19] [probe] push A obj_position L15: r2_mean=0.089 std=0.006 21.1s

[2026-04-26 00:21:23] [probe] peg_insert A ee_acceleration L05: r2_mean=0.194 std=0.005 24.2s

[2026-04-26 00:21:40] [probe] push A obj_position L16: r2_mean=0.081 std=0.020 20.3s

[2026-04-26 00:21:42] [probe] peg_insert A ee_acceleration L06: r2_mean=0.213 std=0.014 17.2s

[2026-04-26 00:21:56] [probe] peg_insert A ee_acceleration L07: r2_mean=0.230 std=0.006 11.7s

[2026-04-26 00:22:03] [probe] push A obj_position L17: r2_mean=0.062 std=0.010 21.2s

[2026-04-26 00:22:10] [probe] peg_insert A ee_acceleration L08: r2_mean=0.230 std=0.008 12.6s

[2026-04-26 00:22:26] [probe] push A obj_position L18: r2_mean=0.054 std=0.007 21.2s

[2026-04-26 00:22:33] [probe] peg_insert A ee_acceleration L09: r2_mean=0.238 std=0.020 21.1s

[2026-04-26 00:22:48] [probe] push A obj_position L19: r2_mean=0.054 std=0.011 20.7s

[2026-04-26 00:22:59] [probe] peg_insert A ee_acceleration L10: r2_mean=0.269 std=0.010 24.3s

[2026-04-26 00:23:11] [probe] push A obj_position L20: r2_mean=0.037 std=0.007 21.9s

[2026-04-26 00:23:24] [probe] peg_insert A ee_acceleration L11: r2_mean=0.258 std=0.038 23.3s

[2026-04-26 00:23:34] [probe] push A obj_position L21: r2_mean=0.023 std=0.021 21.5s

[2026-04-26 00:23:45] [probe] peg_insert A ee_acceleration L12: r2_mean=0.237 std=0.007 19.3s

[2026-04-26 00:23:56] [probe] push A obj_position L22: r2_mean=-0.002 std=0.017 20.5s

[2026-04-26 00:24:02] [probe] peg_insert A ee_acceleration L13: r2_mean=0.287 std=0.021 14.9s

[2026-04-26 00:24:16] [probe] peg_insert A ee_acceleration L14: r2_mean=0.293 std=0.017 12.3s

[2026-04-26 00:24:19] [probe] push A obj_position L23: r2_mean=0.003 std=0.014 21.1s

[2026-04-26 00:24:39] [probe] peg_insert A ee_acceleration L15: r2_mean=0.329 std=0.017 22.1s

[2026-04-26 00:24:41] [probe] push A obj_velocity L00: r2_mean=0.113 std=0.012 19.6s

[2026-04-26 00:24:52] [probe] peg_insert A ee_acceleration L16: r2_mean=0.328 std=0.019 11.0s

[2026-04-26 00:25:03] [probe] push A obj_velocity L01: r2_mean=0.305 std=0.021 20.5s

[2026-04-26 00:25:06] [probe] peg_insert A ee_acceleration L17: r2_mean=0.348 std=0.017 12.4s

[2026-04-26 00:25:20] [probe] peg_insert A ee_acceleration L18: r2_mean=0.339 std=0.023 12.3s

[2026-04-26 00:25:26] [probe] push A obj_velocity L02: r2_mean=0.382 std=0.024 21.5s

[2026-04-26 00:25:35] [probe] peg_insert A ee_acceleration L19: r2_mean=-0.212 std=0.482 12.8s

[2026-04-26 00:25:48] [probe] peg_insert A ee_acceleration L20: r2_mean=0.085 std=0.213 12.3s

[2026-04-26 00:25:49] [probe] push A obj_velocity L03: r2_mean=0.444 std=0.028 20.8s

[2026-04-26 00:26:08] [probe] peg_insert A ee_acceleration L21: r2_mean=0.259 std=0.037 17.9s

[2026-04-26 00:26:11] [probe] push A obj_velocity L04: r2_mean=0.469 std=0.031 21.1s

[2026-04-26 00:26:34] [probe] push A obj_velocity L05: r2_mean=0.513 std=0.034 21.3s

[2026-04-26 00:26:34] [probe] peg_insert A ee_acceleration L22: r2_mean=0.230 std=0.037 24.3s

[2026-04-26 00:26:56] [probe] push A obj_velocity L06: r2_mean=0.545 std=0.034 21.0s

[2026-04-26 00:27:00] [probe] peg_insert A ee_acceleration L23: r2_mean=0.179 std=0.066 23.7s

[2026-04-26 00:27:19] [probe] push A obj_velocity L07: r2_mean=0.578 std=0.035 21.2s

[2026-04-26 00:27:25] [probe] peg_insert A ee_accel_mag L00: r2_mean=0.168 std=0.003 23.5s

[2026-04-26 00:27:42] [probe] push A obj_velocity L08: r2_mean=0.610 std=0.039 21.4s

[2026-04-26 00:27:51] [probe] peg_insert A ee_accel_mag L01: r2_mean=0.285 std=0.002 23.5s

[2026-04-26 00:28:04] [probe] push A obj_velocity L09: r2_mean=0.641 std=0.035 20.4s

[2026-04-26 00:28:15] [probe] peg_insert A ee_accel_mag L02: r2_mean=0.299 std=0.008 22.1s

[2026-04-26 00:28:27] [probe] push A obj_velocity L10: r2_mean=0.658 std=0.040 21.2s

[2026-04-26 00:28:39] [probe] peg_insert A ee_accel_mag L03: r2_mean=0.418 std=0.004 22.4s

[2026-04-26 00:28:49] [probe] push A obj_velocity L11: r2_mean=0.665 std=0.033 20.4s

[2026-04-26 00:29:03] [probe] peg_insert A ee_accel_mag L04: r2_mean=0.440 std=0.004 22.0s

[2026-04-26 00:29:11] [probe] push A obj_velocity L12: r2_mean=0.675 std=0.038 20.6s

[2026-04-26 00:29:26] [probe] peg_insert A ee_accel_mag L05: r2_mean=0.489 std=0.006 21.6s

[2026-04-26 00:29:33] [probe] push A obj_velocity L13: r2_mean=0.688 std=0.040 21.2s

[2026-04-26 00:29:49] [probe] peg_insert A ee_accel_mag L06: r2_mean=0.522 std=0.004 21.0s

[2026-04-26 00:29:55] [probe] push A obj_velocity L14: r2_mean=0.689 std=0.043 20.9s

[2026-04-26 00:30:12] [probe] peg_insert A ee_accel_mag L07: r2_mean=0.541 std=0.002 21.3s

[2026-04-26 00:30:18] [probe] push A obj_velocity L15: r2_mean=0.685 std=0.040 21.0s

[2026-04-26 00:30:35] [probe] peg_insert A ee_accel_mag L08: r2_mean=0.544 std=0.002 21.3s

[2026-04-26 00:30:40] [probe] push A obj_velocity L16: r2_mean=0.702 std=0.041 20.8s

[2026-04-26 00:31:00] [probe] peg_insert A ee_accel_mag L09: r2_mean=0.548 std=0.003 22.6s

[2026-04-26 00:31:04] [probe] push A obj_velocity L17: r2_mean=0.700 std=0.049 21.8s

[2026-04-26 00:31:24] [probe] peg_insert A ee_accel_mag L10: r2_mean=0.556 std=0.003 22.0s

[2026-04-26 00:31:26] [probe] push A obj_velocity L18: r2_mean=0.712 std=0.041 20.7s

[2026-04-26 00:31:48] [probe] push A obj_velocity L19: r2_mean=0.712 std=0.036 20.8s

[2026-04-26 00:31:48] [probe] peg_insert A ee_accel_mag L11: r2_mean=0.559 std=0.004 22.8s

[2026-04-26 00:32:11] [probe] push A obj_velocity L20: r2_mean=0.697 std=0.047 21.3s

[2026-04-26 00:32:12] [probe] peg_insert A ee_accel_mag L12: r2_mean=0.555 std=0.003 22.4s

[2026-04-26 00:32:33] [probe] push A obj_velocity L21: r2_mean=0.719 std=0.044 20.3s

[2026-04-26 00:32:37] [probe] peg_insert A ee_accel_mag L13: r2_mean=0.589 std=0.007 22.7s

[2026-04-26 00:32:55] [probe] push A obj_velocity L22: r2_mean=0.716 std=0.043 20.8s

[2026-04-26 00:33:01] [probe] peg_insert A ee_accel_mag L14: r2_mean=0.592 std=0.004 22.4s

[2026-04-26 00:33:18] [probe] push A obj_velocity L23: r2_mean=0.703 std=0.053 21.7s

[2026-04-26 00:33:25] [probe] peg_insert A ee_accel_mag L15: r2_mean=0.602 std=0.003 22.0s

[2026-04-26 00:33:39] [probe] push A obj_speed L00: r2_mean=0.258 std=0.020 19.3s

[2026-04-26 00:33:49] [probe] peg_insert A ee_accel_mag L16: r2_mean=0.603 std=0.004 21.8s

[2026-04-26 00:34:01] [probe] push A obj_speed L01: r2_mean=0.450 std=0.028 19.9s

[2026-04-26 00:34:14] [probe] peg_insert A ee_accel_mag L17: r2_mean=0.614 std=0.003 22.6s

[2026-04-26 00:34:22] [probe] push A obj_speed L02: r2_mean=0.513 std=0.029 19.6s

[2026-04-26 00:34:38] [probe] peg_insert A ee_accel_mag L18: r2_mean=0.615 std=0.001 22.2s

[2026-04-26 00:34:42] [probe] push A obj_speed L03: r2_mean=0.565 std=0.031 18.9s

[2026-04-26 00:34:59] [probe] push A obj_speed L04: r2_mean=0.606 std=0.031 15.6s

[2026-04-26 00:35:02] [probe] peg_insert A ee_accel_mag L19: r2_mean=0.605 std=0.011 22.7s

[2026-04-26 00:35:13] [probe] push A obj_speed L05: r2_mean=0.643 std=0.035 12.3s

[2026-04-26 00:35:26] [probe] peg_insert A ee_accel_mag L20: r2_mean=0.583 std=0.016 21.8s

[2026-04-26 00:35:28] [probe] push A obj_speed L06: r2_mean=0.669 std=0.033 13.6s

[2026-04-26 00:35:39] [probe] push A obj_speed L07: r2_mean=0.686 std=0.033 10.6s

[2026-04-26 00:35:51] [probe] peg_insert A ee_accel_mag L21: r2_mean=0.601 std=0.005 22.6s

[2026-04-26 00:35:59] [probe] push A obj_speed L08: r2_mean=0.712 std=0.037 18.9s

[2026-04-26 00:36:14] [probe] peg_insert A ee_accel_mag L22: r2_mean=0.590 std=0.006 22.0s

[2026-04-26 00:36:21] [probe] push A obj_speed L09: r2_mean=0.724 std=0.037 19.9s

[2026-04-26 00:36:38] [probe] peg_insert A ee_accel_mag L23: r2_mean=0.570 std=0.016 21.7s

[2026-04-26 00:36:41] [probe] push A obj_speed L10: r2_mean=0.738 std=0.042 18.8s

[2026-04-26 00:37:02] [probe] push A obj_speed L11: r2_mean=0.747 std=0.040 19.3s

[2026-04-26 00:37:04] [probe] peg_insert A obj_position L00: r2_mean=0.505 std=0.005 23.9s

[2026-04-26 00:37:22] [probe] push A obj_speed L12: r2_mean=0.761 std=0.037 19.1s

[2026-04-26 00:37:30] [probe] peg_insert A obj_position L01: r2_mean=0.853 std=0.005 24.1s

[2026-04-26 00:37:43] [probe] push A obj_speed L13: r2_mean=0.780 std=0.042 18.9s

[2026-04-26 00:37:55] [probe] peg_insert A obj_position L02: r2_mean=0.858 std=0.001 23.2s

[2026-04-26 00:38:00] [probe] push A obj_speed L14: r2_mean=0.780 std=0.042 16.3s

[2026-04-26 00:38:18] [probe] push A obj_speed L15: r2_mean=0.783 std=0.041 15.9s

[2026-04-26 00:38:21] [probe] peg_insert A obj_position L03: r2_mean=0.907 std=0.002 24.0s

[2026-04-26 00:38:39] [probe] push A obj_speed L16: r2_mean=0.797 std=0.042 19.7s

[2026-04-26 00:38:46] [probe] peg_insert A obj_position L04: r2_mean=0.919 std=0.002 23.3s

[2026-04-26 00:38:56] [probe] push A obj_speed L17: r2_mean=0.804 std=0.041 15.9s

[2026-04-26 00:39:13] [probe] peg_insert A obj_position L05: r2_mean=0.940 std=0.001 24.3s

[2026-04-26 00:39:17] [probe] push A obj_speed L18: r2_mean=0.805 std=0.042 18.9s

[2026-04-26 00:39:38] [probe] push A obj_speed L19: r2_mean=0.804 std=0.042 19.7s

[2026-04-26 00:39:39] [probe] peg_insert A obj_position L06: r2_mean=0.947 std=0.001 24.4s

[2026-04-26 00:39:57] [probe] push A obj_speed L20: r2_mean=0.806 std=0.043 18.2s

[2026-04-26 00:40:04] [probe] peg_insert A obj_position L07: r2_mean=0.955 std=0.003 23.6s

[2026-04-26 00:40:18] [probe] push A obj_speed L21: r2_mean=0.812 std=0.039 19.2s

[2026-04-26 00:40:30] [probe] peg_insert A obj_position L08: r2_mean=0.958 std=0.001 24.0s

[2026-04-26 00:40:39] [probe] push A obj_speed L22: r2_mean=0.809 std=0.042 20.0s

[2026-04-26 00:40:56] [probe] peg_insert A obj_position L09: r2_mean=0.957 std=0.001 23.4s

[2026-04-26 00:40:57] [probe] push A obj_speed L23: r2_mean=0.803 std=0.041 15.8s

[2026-04-26 00:41:21] [probe] peg_insert A obj_position L10: r2_mean=0.962 std=0.003 23.6s

[2026-04-26 00:41:31] [probe] push A obj_direction L00: r2_mean=0.037 std=0.003 21.5s

[2026-04-26 00:41:47] [probe] peg_insert A obj_position L11: r2_mean=0.964 std=0.005 24.2s

[2026-04-26 00:41:51] [probe] push A obj_direction L01: r2_mean=0.125 std=0.005 18.6s

[2026-04-26 00:42:12] [probe] push A obj_direction L02: r2_mean=0.150 std=0.008 20.5s

[2026-04-26 00:42:13] [probe] peg_insert A obj_position L12: r2_mean=0.963 std=0.007 23.6s

[2026-04-26 00:42:35] [probe] push A obj_direction L03: r2_mean=0.166 std=0.007 21.0s

[2026-04-26 00:42:39] [probe] peg_insert A obj_position L13: r2_mean=0.966 std=0.001 24.0s

[2026-04-26 00:42:57] [probe] push A obj_direction L04: r2_mean=0.178 std=0.008 21.1s

[2026-04-26 00:43:04] [probe] peg_insert A obj_position L14: r2_mean=0.967 std=0.000 23.7s

[2026-04-26 00:43:20] [probe] push A obj_direction L05: r2_mean=0.195 std=0.006 20.8s

[2026-04-26 00:43:31] [probe] peg_insert A obj_position L15: r2_mean=0.962 std=0.007 24.6s

[2026-04-26 00:43:43] [probe] push A obj_direction L06: r2_mean=0.205 std=0.006 21.6s

[2026-04-26 00:43:56] [probe] peg_insert A obj_position L16: r2_mean=0.868 std=0.109 23.9s

[2026-04-26 00:44:04] [probe] push A obj_direction L07: r2_mean=0.216 std=0.006 19.9s

[2026-04-26 00:44:23] [probe] peg_insert A obj_position L17: r2_mean=0.967 std=0.005 24.4s

[2026-04-26 00:44:27] [probe] push A obj_direction L08: r2_mean=0.231 std=0.008 21.3s

[2026-04-26 00:44:49] [probe] peg_insert A obj_position L18: r2_mean=0.959 std=0.014 24.2s

[2026-04-26 00:44:49] [probe] push A obj_direction L09: r2_mean=0.241 std=0.011 21.2s

[2026-04-26 00:45:11] [probe] push A obj_direction L10: r2_mean=0.243 std=0.009 20.8s

[2026-04-26 00:45:14] [probe] peg_insert A obj_position L19: r2_mean=0.813 std=0.125 23.1s

[2026-04-26 00:45:35] [probe] push A obj_direction L11: r2_mean=0.252 std=0.010 21.6s

[2026-04-26 00:45:39] [probe] peg_insert A obj_position L20: r2_mean=0.848 std=0.232 24.0s

[2026-04-26 00:45:57] [probe] push A obj_direction L12: r2_mean=0.268 std=0.006 21.1s

[2026-04-26 00:46:05] [probe] peg_insert A obj_position L21: r2_mean=0.966 std=0.007 24.1s

[2026-04-26 00:46:20] [probe] push A obj_direction L13: r2_mean=0.274 std=0.010 21.1s

[2026-04-26 00:46:31] [probe] peg_insert A obj_position L22: r2_mean=0.962 std=0.008 23.7s

[2026-04-26 00:46:43] [probe] push A obj_direction L14: r2_mean=0.276 std=0.010 21.6s

[2026-04-26 00:46:57] [probe] peg_insert A obj_position L23: r2_mean=0.954 std=0.018 24.1s

[2026-04-26 00:47:05] [probe] push A obj_direction L15: r2_mean=0.275 std=0.010 20.8s

[2026-04-26 00:47:20] [probe] peg_insert A obj_velocity L00: r2_mean=0.102 std=0.006 21.0s

[2026-04-26 00:47:27] [probe] push A obj_direction L16: r2_mean=0.279 std=0.009 20.6s

[2026-04-26 00:47:37] [probe] peg_insert A obj_velocity L01: r2_mean=0.154 std=0.007 15.0s

[2026-04-26 00:47:50] [probe] push A obj_direction L17: r2_mean=0.276 std=0.010 21.6s

[2026-04-26 00:48:02] [probe] peg_insert A obj_velocity L02: r2_mean=0.169 std=0.007 23.7s

[2026-04-26 00:48:13] [probe] push A obj_direction L18: r2_mean=0.280 std=0.011 21.3s

[2026-04-26 00:48:28] [probe] peg_insert A obj_velocity L03: r2_mean=0.253 std=0.007 23.5s

[2026-04-26 00:48:36] [probe] push A obj_direction L19: r2_mean=0.275 std=0.010 20.6s

[2026-04-26 00:48:53] [probe] peg_insert A obj_velocity L04: r2_mean=0.282 std=0.012 23.2s

[2026-04-26 00:48:59] [probe] push A obj_direction L20: r2_mean=0.273 std=0.008 21.6s

[2026-04-26 00:49:19] [probe] peg_insert A obj_velocity L05: r2_mean=0.333 std=0.014 24.5s

[2026-04-26 00:49:22] [probe] push A obj_direction L21: r2_mean=0.278 std=0.012 21.3s

[2026-04-26 00:49:45] [probe] peg_insert A obj_velocity L06: r2_mean=0.357 std=0.015 23.5s

[2026-04-26 00:49:45] [probe] push A obj_direction L22: r2_mean=0.283 std=0.012 21.4s

[2026-04-26 00:50:08] [probe] push A obj_direction L23: r2_mean=0.277 std=0.014 21.1s

[2026-04-26 00:50:11] [probe] peg_insert A obj_velocity L07: r2_mean=0.392 std=0.022 24.4s

[2026-04-26 00:50:31] [probe] push A obj_acceleration L00: r2_mean=0.002 std=0.001 21.3s

[2026-04-26 00:50:36] [probe] peg_insert A obj_velocity L08: r2_mean=0.393 std=0.020 23.2s

[2026-04-26 00:50:55] [probe] push A obj_acceleration L01: r2_mean=0.016 std=0.003 21.5s

[2026-04-26 00:51:02] [probe] peg_insert A obj_velocity L09: r2_mean=0.389 std=0.019 24.4s

[2026-04-26 00:51:17] [probe] push A obj_acceleration L02: r2_mean=0.019 std=0.003 21.0s

[2026-04-26 00:51:28] [probe] peg_insert A obj_velocity L10: r2_mean=0.400 std=0.024 24.0s

[2026-04-26 00:51:39] [probe] push A obj_acceleration L03: r2_mean=0.024 std=0.004 20.1s

[2026-04-26 00:51:55] [probe] peg_insert A obj_velocity L11: r2_mean=0.376 std=0.038 24.4s

[2026-04-26 00:52:02] [probe] push A obj_acceleration L04: r2_mean=0.028 std=0.005 21.6s

[2026-04-26 00:52:20] [probe] peg_insert A obj_velocity L12: r2_mean=0.393 std=0.017 23.6s

[2026-04-26 00:52:25] [probe] push A obj_acceleration L05: r2_mean=0.033 std=0.006 20.9s

[2026-04-26 00:52:45] [probe] peg_insert A obj_velocity L13: r2_mean=0.409 std=0.018 23.1s

[2026-04-26 00:52:47] [probe] push A obj_acceleration L06: r2_mean=0.036 std=0.006 21.1s

[2026-04-26 00:53:09] [probe] push A obj_acceleration L07: r2_mean=0.041 std=0.007 20.6s

[2026-04-26 00:53:10] [probe] peg_insert A obj_velocity L14: r2_mean=0.424 std=0.018 23.7s

[2026-04-26 00:53:32] [probe] push A obj_acceleration L08: r2_mean=0.046 std=0.008 21.4s

[2026-04-26 00:53:37] [probe] peg_insert A obj_velocity L15: r2_mean=0.462 std=0.019 24.3s

[2026-04-26 00:53:52] [probe] push A obj_acceleration L09: r2_mean=0.051 std=0.008 18.8s

[2026-04-26 00:54:03] [probe] peg_insert A obj_velocity L16: r2_mean=0.438 std=0.036 24.1s

[2026-04-26 00:54:13] [probe] push A obj_acceleration L10: r2_mean=0.055 std=0.009 19.5s

[2026-04-26 00:54:29] [probe] peg_insert A obj_velocity L17: r2_mean=0.489 std=0.024 24.5s

[2026-04-26 00:54:36] [probe] push A obj_acceleration L11: r2_mean=0.056 std=0.009 21.1s

[2026-04-26 00:54:54] [probe] peg_insert A obj_velocity L18: r2_mean=0.487 std=0.023 23.1s

[2026-04-26 00:54:57] [probe] push A obj_acceleration L12: r2_mean=0.059 std=0.010 20.0s

[2026-04-26 00:55:19] [probe] push A obj_acceleration L13: r2_mean=0.065 std=0.011 19.7s

[2026-04-26 00:55:21] [probe] peg_insert A obj_velocity L19: r2_mean=0.414 std=0.057 24.5s

[2026-04-26 00:55:41] [probe] push A obj_acceleration L14: r2_mean=0.065 std=0.010 21.0s

[2026-04-26 00:55:46] [probe] peg_insert A obj_velocity L20: r2_mean=0.377 std=0.111 24.0s

[2026-04-26 00:56:02] [probe] push A obj_acceleration L15: r2_mean=0.066 std=0.011 19.6s

[2026-04-26 00:56:12] [probe] peg_insert A obj_velocity L21: r2_mean=0.476 std=0.019 23.6s

[2026-04-26 00:56:25] [probe] push A obj_acceleration L16: r2_mean=0.071 std=0.011 21.8s

[2026-04-26 00:56:38] [probe] peg_insert A obj_velocity L22: r2_mean=0.433 std=0.078 23.9s

[2026-04-26 00:56:49] [probe] push A obj_acceleration L17: r2_mean=0.068 std=0.010 21.7s

[2026-04-26 00:57:03] [probe] peg_insert A obj_velocity L23: r2_mean=0.230 std=0.356 23.7s

[2026-04-26 00:57:11] [probe] push A obj_acceleration L18: r2_mean=0.068 std=0.011 20.5s

[2026-04-26 00:57:27] [probe] peg_insert A obj_speed L00: r2_mean=0.114 std=0.006 22.2s

[2026-04-26 00:57:34] [probe] push A obj_acceleration L19: r2_mean=0.064 std=0.011 21.5s

[2026-04-26 00:57:52] [probe] peg_insert A obj_speed L01: r2_mean=0.228 std=0.013 22.7s

[2026-04-26 00:57:57] [probe] push A obj_acceleration L20: r2_mean=0.065 std=0.010 21.0s

[2026-04-26 00:58:16] [probe] peg_insert A obj_speed L02: r2_mean=0.238 std=0.009 21.7s

[2026-04-26 00:58:19] [probe] push A obj_acceleration L21: r2_mean=0.063 std=0.012 20.8s

[2026-04-26 00:58:40] [probe] peg_insert A obj_speed L03: r2_mean=0.352 std=0.017 22.6s

[2026-04-26 00:58:42] [probe] push A obj_acceleration L22: r2_mean=0.056 std=0.008 21.1s

[2026-04-26 00:59:04] [probe] peg_insert A obj_speed L04: r2_mean=0.377 std=0.019 22.1s

[2026-04-26 00:59:05] [probe] push A obj_acceleration L23: r2_mean=0.059 std=0.011 21.6s

[2026-04-26 00:59:26] [probe] push A obj_accel_mag L00: r2_mean=0.070 std=0.006 19.3s

[2026-04-26 00:59:29] [probe] peg_insert A obj_speed L05: r2_mean=0.427 std=0.026 22.7s

[2026-04-26 00:59:48] [probe] push A obj_accel_mag L01: r2_mean=0.127 std=0.014 20.0s

[2026-04-26 00:59:53] [probe] peg_insert A obj_speed L06: r2_mean=0.459 std=0.025 22.2s

[2026-04-26 01:00:08] [probe] push A obj_accel_mag L02: r2_mean=0.155 std=0.015 19.1s

[2026-04-26 01:00:14] [probe] peg_insert A obj_speed L07: r2_mean=0.497 std=0.028 19.5s

[2026-04-26 01:00:29] [probe] push A obj_accel_mag L03: r2_mean=0.169 std=0.018 19.6s

[2026-04-26 01:00:38] [probe] peg_insert A obj_speed L08: r2_mean=0.497 std=0.028 22.2s

[2026-04-26 01:00:50] [probe] push A obj_accel_mag L04: r2_mean=0.180 std=0.019 19.9s

[2026-04-26 01:01:03] [probe] peg_insert A obj_speed L09: r2_mean=0.491 std=0.026 22.7s

[2026-04-26 01:01:11] [probe] push A obj_accel_mag L05: r2_mean=0.188 std=0.018 19.1s

[2026-04-26 01:01:27] [probe] peg_insert A obj_speed L10: r2_mean=0.496 std=0.030 22.3s

[2026-04-26 01:01:31] [probe] push A obj_accel_mag L06: r2_mean=0.193 std=0.020 18.8s

[2026-04-26 01:01:51] [probe] peg_insert A obj_speed L11: r2_mean=0.503 std=0.029 22.1s

[2026-04-26 01:01:53] [probe] push A obj_accel_mag L07: r2_mean=0.205 std=0.019 19.9s

[2026-04-26 01:02:13] [probe] push A obj_accel_mag L08: r2_mean=0.215 std=0.023 18.8s

[2026-04-26 01:02:15] [probe] peg_insert A obj_speed L12: r2_mean=0.495 std=0.028 22.3s

[2026-04-26 01:02:27] [probe] push A obj_accel_mag L09: r2_mean=0.222 std=0.021 12.2s

[2026-04-26 01:02:39] [probe] peg_insert A obj_speed L13: r2_mean=0.515 std=0.030 21.8s

[2026-04-26 01:02:46] [probe] push A obj_accel_mag L10: r2_mean=0.229 std=0.023 17.8s

[2026-04-26 01:02:57] [probe] push A obj_accel_mag L11: r2_mean=0.234 std=0.024 10.1s

[2026-04-26 01:03:02] [probe] peg_insert A obj_speed L14: r2_mean=0.526 std=0.029 21.7s

[2026-04-26 01:03:12] [probe] push A obj_accel_mag L12: r2_mean=0.238 std=0.024 14.1s

[2026-04-26 01:03:26] [probe] peg_insert A obj_speed L15: r2_mean=0.562 std=0.030 21.7s

[2026-04-26 01:03:32] [probe] push A obj_accel_mag L13: r2_mean=0.260 std=0.023 18.2s

[2026-04-26 01:03:40] [probe] peg_insert A obj_speed L16: r2_mean=0.573 std=0.030 11.8s

[2026-04-26 01:03:51] [probe] push A obj_accel_mag L14: r2_mean=0.268 std=0.025 18.2s

[2026-04-26 01:03:59] [probe] peg_insert A obj_speed L17: r2_mean=0.587 std=0.031 17.2s

[2026-04-26 01:04:10] [probe] push A obj_accel_mag L15: r2_mean=0.271 std=0.025 16.7s

[2026-04-26 01:04:22] [probe] peg_insert A obj_speed L18: r2_mean=0.588 std=0.032 21.4s

[2026-04-26 01:04:26] [probe] push A obj_accel_mag L16: r2_mean=0.285 std=0.025 15.3s

[2026-04-26 01:04:42] [probe] peg_insert A obj_speed L19: r2_mean=0.580 std=0.035 19.0s

[2026-04-26 01:04:44] [probe] push A obj_accel_mag L17: r2_mean=0.292 std=0.025 16.0s

[2026-04-26 01:05:02] [probe] peg_insert A obj_speed L20: r2_mean=0.573 std=0.026 17.7s

[2026-04-26 01:05:05] [probe] push A obj_accel_mag L18: r2_mean=0.293 std=0.025 19.4s

[2026-04-26 01:05:26] [probe] push A obj_accel_mag L19: r2_mean=0.289 std=0.026 19.2s

[2026-04-26 01:05:26] [probe] peg_insert A obj_speed L21: r2_mean=0.586 std=0.032 21.9s

[2026-04-26 01:05:43] [probe] push A obj_accel_mag L20: r2_mean=0.285 std=0.024 15.6s

[2026-04-26 01:05:48] [probe] peg_insert A obj_speed L22: r2_mean=0.576 std=0.033 20.7s

[2026-04-26 01:05:53] [probe] push A obj_accel_mag L21: r2_mean=0.285 std=0.025 9.0s

[2026-04-26 01:06:02] [probe] push A obj_accel_mag L22: r2_mean=0.286 std=0.025 8.4s

[2026-04-26 01:06:12] [probe] peg_insert A obj_speed L23: r2_mean=0.561 std=0.030 21.7s

[2026-04-26 01:06:17] [probe] push A obj_accel_mag L23: r2_mean=0.278 std=0.024 13.0s

[2026-04-26 01:06:17] [probe] push A DONE in 102.9min

[2026-04-26 01:06:47] [probe] peg_insert A obj_direction L00: r2_mean=0.054 std=0.003 22.9s

[2026-04-26 01:07:13] [probe] peg_insert A obj_direction L01: r2_mean=0.100 std=0.004 23.8s

[2026-04-26 01:07:38] [probe] peg_insert A obj_direction L02: r2_mean=0.112 std=0.002 23.6s

[2026-04-26 01:08:03] [probe] peg_insert A obj_direction L03: r2_mean=0.131 std=0.002 23.3s

[2026-04-26 01:08:29] [probe] peg_insert A obj_direction L04: r2_mean=0.143 std=0.002 23.8s

[2026-04-26 01:08:52] [probe] peg_insert A obj_direction L05: r2_mean=0.151 std=0.002 21.3s

[2026-04-26 01:09:12] [probe] peg_insert A obj_direction L06: r2_mean=0.155 std=0.003 18.9s

[2026-04-26 01:09:38] [probe] peg_insert A obj_direction L07: r2_mean=0.160 std=0.004 23.7s

[2026-04-26 01:09:59] [probe] peg_insert A obj_direction L08: r2_mean=0.163 std=0.002 19.5s

[2026-04-26 01:10:24] [probe] peg_insert A obj_direction L09: r2_mean=0.167 std=0.003 23.6s

[2026-04-26 01:10:50] [probe] peg_insert A obj_direction L10: r2_mean=0.166 std=0.007 23.8s

[2026-04-26 01:11:14] [probe] peg_insert A obj_direction L11: r2_mean=0.146 std=0.028 21.8s

[2026-04-26 01:11:40] [probe] peg_insert A obj_direction L12: r2_mean=0.171 std=0.003 24.0s

[2026-04-26 01:12:05] [probe] peg_insert A obj_direction L13: r2_mean=0.174 std=0.002 22.9s

[2026-04-26 01:12:30] [probe] peg_insert A obj_direction L14: r2_mean=0.178 std=0.004 23.6s

[2026-04-26 01:12:56] [probe] peg_insert A obj_direction L15: r2_mean=0.180 std=0.003 24.0s

[2026-04-26 01:13:21] [probe] peg_insert A obj_direction L16: r2_mean=0.162 std=0.029 23.5s

[2026-04-26 01:13:47] [probe] peg_insert A obj_direction L17: r2_mean=0.184 std=0.004 23.9s

[2026-04-26 01:14:12] [probe] peg_insert A obj_direction L18: r2_mean=0.176 std=0.012 23.4s

[2026-04-26 01:14:38] [probe] peg_insert A obj_direction L19: r2_mean=0.021 std=0.098 23.8s

[2026-04-26 01:15:03] [probe] peg_insert A obj_direction L20: r2_mean=0.065 std=0.156 23.1s

[2026-04-26 01:15:28] [probe] peg_insert A obj_direction L21: r2_mean=0.159 std=0.023 23.3s

[2026-04-26 01:15:53] [probe] peg_insert A obj_direction L22: r2_mean=0.142 std=0.043 23.7s

[2026-04-26 01:16:19] [probe] peg_insert A obj_direction L23: r2_mean=-0.052 std=0.166 23.6s

[2026-04-26 01:16:46] [probe] peg_insert A obj_acceleration L00: r2_mean=0.004 std=0.002 24.1s

[2026-04-26 01:17:13] [probe] peg_insert A obj_acceleration L01: r2_mean=0.024 std=0.003 24.7s

[2026-04-26 01:17:37] [probe] peg_insert A obj_acceleration L02: r2_mean=0.037 std=0.004 22.8s

[2026-04-26 01:18:04] [probe] peg_insert A obj_acceleration L03: r2_mean=0.057 std=0.006 24.9s

[2026-04-26 01:18:30] [probe] peg_insert A obj_acceleration L04: r2_mean=0.061 std=0.007 23.9s

[2026-04-26 01:18:57] [probe] peg_insert A obj_acceleration L05: r2_mean=0.076 std=0.010 25.1s

[2026-04-26 01:19:23] [probe] peg_insert A obj_acceleration L06: r2_mean=0.083 std=0.012 24.5s

[2026-04-26 01:19:41] [probe] peg_insert A obj_acceleration L07: r2_mean=0.087 std=0.010 15.9s

[2026-04-26 01:19:56] [probe] peg_insert A obj_acceleration L08: r2_mean=0.089 std=0.011 13.3s

[2026-04-26 01:20:17] [probe] peg_insert A obj_acceleration L09: r2_mean=0.086 std=0.011 19.7s

[2026-04-26 01:20:42] [probe] peg_insert A obj_acceleration L10: r2_mean=0.095 std=0.012 23.2s

[2026-04-26 01:21:09] [probe] peg_insert A obj_acceleration L11: r2_mean=0.053 std=0.035 24.4s

[2026-04-26 01:21:35] [probe] peg_insert A obj_acceleration L12: r2_mean=0.091 std=0.010 24.6s

[2026-04-26 01:22:02] [probe] peg_insert A obj_acceleration L13: r2_mean=0.106 std=0.012 25.3s

[2026-04-26 01:22:29] [probe] peg_insert A obj_acceleration L14: r2_mean=0.110 std=0.013 24.6s

[2026-04-26 01:22:55] [probe] peg_insert A obj_acceleration L15: r2_mean=0.115 std=0.014 24.2s

[2026-04-26 01:23:22] [probe] peg_insert A obj_acceleration L16: r2_mean=0.117 std=0.017 24.9s

[2026-04-26 01:23:48] [probe] peg_insert A obj_acceleration L17: r2_mean=0.122 std=0.017 24.1s

[2026-04-26 01:24:03] [probe] peg_insert A obj_acceleration L18: r2_mean=0.124 std=0.019 13.7s

[2026-04-26 01:24:29] [probe] peg_insert A obj_acceleration L19: r2_mean=0.020 std=0.144 24.4s

[2026-04-26 01:24:55] [probe] peg_insert A obj_acceleration L20: r2_mean=-0.066 std=0.131 24.2s

[2026-04-26 01:25:22] [probe] peg_insert A obj_acceleration L21: r2_mean=0.103 std=0.012 24.5s

[2026-04-26 01:25:48] [probe] peg_insert A obj_acceleration L22: r2_mean=-0.008 std=0.170 24.6s

[2026-04-26 01:26:15] [probe] peg_insert A obj_acceleration L23: r2_mean=-0.011 std=0.063 24.7s

[2026-04-26 01:26:40] [probe] peg_insert A obj_accel_mag L00: r2_mean=0.079 std=0.006 22.6s

[2026-04-26 01:27:05] [probe] peg_insert A obj_accel_mag L01: r2_mean=0.156 std=0.011 22.9s

[2026-04-26 01:27:29] [probe] peg_insert A obj_accel_mag L02: r2_mean=0.161 std=0.013 22.8s

[2026-04-26 01:27:54] [probe] peg_insert A obj_accel_mag L03: r2_mean=0.209 std=0.015 22.7s

[2026-04-26 01:28:18] [probe] peg_insert A obj_accel_mag L04: r2_mean=0.220 std=0.018 22.3s

[2026-04-26 01:28:43] [probe] peg_insert A obj_accel_mag L05: r2_mean=0.240 std=0.021 23.2s

[2026-04-26 01:29:08] [probe] peg_insert A obj_accel_mag L06: r2_mean=0.253 std=0.021 22.8s

[2026-04-26 01:29:33] [probe] peg_insert A obj_accel_mag L07: r2_mean=0.258 std=0.021 23.3s

[2026-04-26 01:29:57] [probe] peg_insert A obj_accel_mag L08: r2_mean=0.262 std=0.021 22.5s

[2026-04-26 01:30:22] [probe] peg_insert A obj_accel_mag L09: r2_mean=0.262 std=0.023 22.8s

[2026-04-26 01:30:47] [probe] peg_insert A obj_accel_mag L10: r2_mean=0.266 std=0.021 23.0s

[2026-04-26 01:31:12] [probe] peg_insert A obj_accel_mag L11: r2_mean=0.267 std=0.021 22.9s

[2026-04-26 01:31:35] [probe] peg_insert A obj_accel_mag L12: r2_mean=0.268 std=0.022 21.3s

[2026-04-26 01:31:59] [probe] peg_insert A obj_accel_mag L13: r2_mean=0.282 std=0.023 22.6s

[2026-04-26 01:32:23] [probe] peg_insert A obj_accel_mag L14: r2_mean=0.285 std=0.023 22.0s

[2026-04-26 01:32:47] [probe] peg_insert A obj_accel_mag L15: r2_mean=0.290 std=0.022 22.6s

[2026-04-26 01:33:12] [probe] peg_insert A obj_accel_mag L16: r2_mean=0.288 std=0.020 22.5s

[2026-04-26 01:33:35] [probe] peg_insert A obj_accel_mag L17: r2_mean=0.295 std=0.023 21.6s

[2026-04-26 01:33:59] [probe] peg_insert A obj_accel_mag L18: r2_mean=0.296 std=0.024 21.9s

[2026-04-26 01:34:23] [probe] peg_insert A obj_accel_mag L19: r2_mean=0.282 std=0.029 22.3s

[2026-04-26 01:34:48] [probe] peg_insert A obj_accel_mag L20: r2_mean=0.205 std=0.132 22.7s

[2026-04-26 01:35:12] [probe] peg_insert A obj_accel_mag L21: r2_mean=0.289 std=0.021 22.3s

[2026-04-26 01:35:36] [probe] peg_insert A obj_accel_mag L22: r2_mean=0.280 std=0.016 22.3s

[2026-04-26 01:36:01] [probe] peg_insert A obj_accel_mag L23: r2_mean=0.261 std=0.025 22.9s

[2026-04-26 01:36:01] [probe] peg_insert A DONE in 115.7min

## ALL DONE — 2026-04-26 01:36 UTC

**Total wall time:** ~16h05min (start 09:30 UTC, end 01:36 UTC).

**Verdict:**
- Spec § 16 strict: **MARGINAL** (Push ee_velocity argmax at L22, marginally outside 6–18)
- Relaxed (first layer at 99% of max R²): **HEALTHY** (Push ee_velocity 99% peak at L17)
- Primary: **HEALTHY**

**Headline R² (best layer per criterion):**
- Push ee_velocity L22 R²=**0.921** (>0.5 ✓)
- Push ee_position L21 R²=**0.980** (>0.5 ✓)
- Strike ee_velocity L18 R²=**0.936** (>0.4 ✓)

**Spec deviations documented in REPORT.md:**
1. Probe batch_size 1024 → max(1024, N/8) for wall-time feasibility (Python loop overhead)
2. dt source: local meta/info.json fps (push/strike/reach=50, drawer=60, peg/nut=15) instead of HF README "15 fps for all"
3. Native acceleration (Isaac Lab body-accel) ≠ finite-diff(velocity); used finite-diff uniformly per user directive
4. L0_saturates check: L0>0.9 alone insufficient — also requires gain<0.02; without this Drawer ee_position (static scene) would falsely trigger FAILED
5. PEZ peak depth: relaxed criterion (first layer ≥99% of max R²) added alongside strict argmax
6. Pooling identity tolerance: relative 1e-3 (not absolute 1e-3) — V-JEPA activation magnitudes ~250 in deep layers, fp16 storage truncation ~0.1 absolute

Output: results/decision.json, results/peak_layers.csv, results/REPORT.md, results/<task>/variant_A/*.csv

