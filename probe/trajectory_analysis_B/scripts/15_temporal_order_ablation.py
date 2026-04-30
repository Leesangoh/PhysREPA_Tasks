#!/usr/bin/env python3
"""Temporal-order ablation on Variant B (Codex top recommendation).

For each (task, layer, target) we run a probe under SEVEN slot-structure
transformations applied to the same 8192-d Variant B feature
[N, 24, 8192] = [N, 24, 8 temporal × 1024 spatial-mean]:

  1. baseline_full     : all 8 slots concat (= original Variant B)
  2. mean_over_slots   : per-window average over 8 temporal slots → 1024-d
                         (i.e., recomputes Variant A from Variant B; sanity baseline)
  3. single_slot_k     : only slot k (1024-d), reported per k ∈ {0..7}
  4. prefix_k          : keep first k+1 slots, zero-fill the rest
                         (k ∈ {0..7}; reveals "how many time-steps needed")
  5. reverse           : reorder slots to [7,6,5,4,3,2,1,0] (concat 8192-d)
  6. random_global_perm: a single random permutation of 8 slots applied to all samples
  7. random_per_sample : fresh random permutation per sample (max order destruction)

Probe: Ridge regression with 5-fold GroupKFold by episode_id, within-fold
z-score normalization. (Ridge instead of full Adam sweep for tractable
runtime — this is a comparison study, not the headline metric.)

Targets: ee_acceleration, obj_acceleration (where applicable),
contact_flag, contact_force_log1p_mag.

Tasks: push, strike, drawer (the contact-rich tasks where the dichotomy holds).

Outputs:
- results/stats/temporal_order_ablation.csv:
    task, layer, target, scheme, scheme_param (k or perm), r2_mean, r2_std
- results/plots/temporal_order_<task>_<target>.png:
    layer (x) vs R² (y), one line per scheme. PEZ band shaded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from ta_utils.loader import all_trajectories
from utils.io import load_targets


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


N_TEMPORAL_SLOTS = 8
SLOT_DIM = 1024
TASKS = ["push", "strike", "drawer"]
TARGETS = ["ee_acceleration", "obj_acceleration", "contact_flag", "contact_force_log1p_mag"]
# Subsample windows for tractable Ridge fitting on 8192-d features
N_SUBSAMPLE = 50000
# Restrict layers to a representative subset for the heavy ablation (still 12 layers)
LAYERS_TO_ABLATE = [0, 2, 5, 8, 11, 13, 15, 17, 19, 20, 22, 23]
# Limit schemes (per Codex's "very clean panel" rec)
# Single_slot only for k in {0,3,7}; prefix only for k in {0,3,7}
SINGLE_SLOTS_TO_TEST = [0, 3, 7]
PREFIX_K_TO_TEST = [0, 3, 7]


def fold_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """5-fold GroupKFold Ridge; returns (R²_mean, R²_std)."""
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if not ok.all():
        X = X[ok]; y = y[ok]; groups = groups[ok]
    if y.shape[0] < 100:
        return float("nan"), float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    r2s = []
    for tr, te in gkf.split(X, y, groups=groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        mu_x = Xtr.mean(0); sd_x = Xtr.std(0) + 1e-9
        Xtr_n = (Xtr - mu_x) / sd_x
        Xte_n = (Xte - mu_x) / sd_x
        if y.ndim == 1:
            mu_y, sd_y = ytr.mean(), ytr.std() + 1e-9
        else:
            mu_y, sd_y = ytr.mean(0), ytr.std(0) + 1e-9
        ytr_n = (ytr - mu_y) / sd_y
        m = Ridge(alpha=1.0)
        m.fit(Xtr_n, ytr_n)
        pred = m.predict(Xte_n) * sd_y + mu_y
        r2s.append(r2_score(yte, pred, multioutput="variance_weighted") if yte.ndim > 1 else r2_score(yte, pred))
    return float(np.mean(r2s)), float(np.std(r2s))


def transform_feature(X8192: np.ndarray, scheme: str, param: int = 0,
                      seed: int = 42) -> np.ndarray:
    """Apply slot-structure transformation. Input shape [N, 8192] for one layer.
    Output shape varies by scheme.
    """
    N = X8192.shape[0]
    X = X8192.reshape(N, N_TEMPORAL_SLOTS, SLOT_DIM)   # [N, 8, 1024]

    if scheme == "baseline_full":
        return X.reshape(N, -1)                         # 8192

    if scheme == "mean_over_slots":
        return X.mean(axis=1)                           # 1024

    if scheme == "single_slot":
        return X[:, param, :]                           # 1024

    if scheme == "prefix":
        # Keep slots 0..k, zero-fill k+1..7; concat to 8192 to keep dim consistent
        keep = param + 1
        Y = np.zeros_like(X)
        Y[:, :keep, :] = X[:, :keep, :]
        return Y.reshape(N, -1)

    if scheme == "reverse":
        return X[:, ::-1, :].reshape(N, -1)             # 8192

    if scheme == "random_global_perm":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N_TEMPORAL_SLOTS)
        return X[:, perm, :].reshape(N, -1)             # 8192

    if scheme == "random_per_sample":
        rng = np.random.default_rng(seed)
        # Different permutation per sample
        Y = X.copy()
        # Vectorize: for each row, get a permutation
        perms = np.stack([rng.permutation(N_TEMPORAL_SLOTS) for _ in range(N)])
        rows = np.arange(N)[:, None]
        Y = Y[rows, perms, :]
        return Y.reshape(N, -1)                         # 8192

    raise ValueError(f"unknown scheme: {scheme}")


def main():
    rows = []
    for task in TASKS:
        print(f"[15_temporal_order] {task} loading ...", flush=True)
        trajs = all_trajectories(task)
        tgt = load_targets(task)
        keys_tgt = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
        lut = {k: i for i, k in enumerate(keys_tgt)}

        all_feats = []
        all_eps = []
        all_t_last = []
        for tj in trajs:
            T = tj["t_last"].size
            all_feats.append(tj["feats"])  # [T, 24, 8192]
            all_eps.append(np.full(T, int(tj["episode_id"]), dtype=np.int64))
            all_t_last.append(tj["t_last"])
        feats_full = np.concatenate(all_feats, axis=0)
        eps_arr = np.concatenate(all_eps, axis=0)
        t_last = np.concatenate(all_t_last, axis=0)
        rows_idx = np.array([lut[(int(g), int(t))] for g, t in zip(eps_arr, t_last)], dtype=np.int64)

        # Stratified-by-episode subsample for tractable Ridge runtime on 8192-d feats
        if feats_full.shape[0] > N_SUBSAMPLE:
            rng = np.random.default_rng(42)
            sub = np.sort(rng.choice(feats_full.shape[0], N_SUBSAMPLE, replace=False))
            feats_full = feats_full[sub]
            eps_arr = eps_arr[sub]
            rows_idx = rows_idx[sub]
            print(f"[15_temporal_order] {task}: subsampled to {feats_full.shape[0]} windows", flush=True)

        for L in LAYERS_TO_ABLATE:
            tL = time.time()
            X_layer = feats_full[:, L, :]   # [N_total, 8192]
            for tk in TARGETS:
                if tk not in tgt.files:
                    continue
                y = tgt[tk][rows_idx]

                # Run schemes
                # 1. baseline_full
                Xt = transform_feature(X_layer, "baseline_full")
                r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                rows.append({"task": task, "layer": L, "target": tk, "scheme": "baseline_full",
                             "scheme_param": -1, "r2_mean": r2_m, "r2_std": r2_s})
                # 2. mean_over_slots
                Xt = transform_feature(X_layer, "mean_over_slots")
                r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                rows.append({"task": task, "layer": L, "target": tk, "scheme": "mean_over_slots",
                             "scheme_param": -1, "r2_mean": r2_m, "r2_std": r2_s})
                # 3. single_slot_k (only representative slots)
                for k in SINGLE_SLOTS_TO_TEST:
                    Xt = transform_feature(X_layer, "single_slot", param=k)
                    r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                    rows.append({"task": task, "layer": L, "target": tk, "scheme": "single_slot",
                                 "scheme_param": k, "r2_mean": r2_m, "r2_std": r2_s})
                # 4. prefix_k (only representative)
                for k in PREFIX_K_TO_TEST:
                    Xt = transform_feature(X_layer, "prefix", param=k)
                    r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                    rows.append({"task": task, "layer": L, "target": tk, "scheme": "prefix",
                                 "scheme_param": k, "r2_mean": r2_m, "r2_std": r2_s})
                # 5. reverse
                Xt = transform_feature(X_layer, "reverse")
                r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                rows.append({"task": task, "layer": L, "target": tk, "scheme": "reverse",
                             "scheme_param": -1, "r2_mean": r2_m, "r2_std": r2_s})
                # 6. random_global_perm
                Xt = transform_feature(X_layer, "random_global_perm", seed=42)
                r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                rows.append({"task": task, "layer": L, "target": tk, "scheme": "random_global_perm",
                             "scheme_param": -1, "r2_mean": r2_m, "r2_std": r2_s})
                # 7. random_per_sample
                Xt = transform_feature(X_layer, "random_per_sample", seed=42)
                r2_m, r2_s = fold_r2(Xt, y, eps_arr)
                rows.append({"task": task, "layer": L, "target": tk, "scheme": "random_per_sample",
                             "scheme_param": -1, "r2_mean": r2_m, "r2_std": r2_s})
            print(f"[15_temporal_order] {task} L{L:02d}: {time.time()-tL:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "temporal_order_ablation.csv", index=False)
    print(f"[15_temporal_order] wrote stats ({len(df)} rows)", flush=True)

    # Plots: layer (x) vs R² (y), one line per scheme; per (task, target)
    for task in TASKS:
        for tk in TARGETS:
            sub = df[(df.task == task) & (df.target == tk)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            schemes_plot = ["baseline_full", "mean_over_slots", "reverse",
                            "random_global_perm", "random_per_sample"]
            colors = {"baseline_full": "C0", "mean_over_slots": "C1", "reverse": "C2",
                      "random_global_perm": "C3", "random_per_sample": "C4"}
            for sch in schemes_plot:
                s = sub[sub.scheme == sch].sort_values("layer")
                if s.empty:
                    continue
                ax.plot(s.layer, s.r2_mean, "-o", color=colors.get(sch, "gray"),
                        markersize=3, linewidth=1.3, label=sch)
            ax.set_xlabel("Layer")
            ax.set_ylabel("R²")
            ax.set_title(f"Temporal-order ablation — {task} / {tk}")
            ax.set_xlim(-0.5, 23.5)
            ax.axvspan(6, 18, alpha=0.05, color="green")
            ax.axhline(0, ls=":", color="gray", lw=0.7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(PLOTS / f"temporal_order_{task}_{tk}.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
