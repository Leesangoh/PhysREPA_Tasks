#!/usr/bin/env python3
"""Per-temporal-slot decomposition (Variant B exclusive).

Variant B feature [N_win, 24, 8192] = [N_win, 24, 8 temporal × 1024 spatial-mean].
For each (task, layer, slot t in 0..7) we slice out a 1024-d sub-feature and
analyze it independently:

1. Trajectory geometry stats (path length, mean speed, tortuosity, curvature).
2. Linear probe R² for selected key targets:
   - ee_position, ee_velocity, ee_acceleration  (kinematic; what does EACH temporal slot encode?)
   - contact_flag, contact_force_log1p_mag       (contact)
3. Per-slot intrinsic dim (n_pcs for 90% variance).
4. Compare slot-0 (early frames in window) vs slot-7 (late frames in window).

Outputs:
- results/stats/per_slot_stats.csv:        task, layer, slot, target, r2_mean, r2_std (probe)
- results/stats/per_slot_geometry.csv:     task, layer, slot, path_length, mean_speed, tortuosity
- results/stats/per_slot_intrinsic_dim.csv: task, layer, slot, n_pcs_90, n_pcs_95
- results/plots/per_slot_<target>_<task>.png: layer×slot heatmap of R² per target
- results/plots/per_slot_geometry_<task>.png: layer×slot heatmap of path_length / tortuosity
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
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from ta_utils.loader import all_trajectories, ALL_TASKS
from utils.io import load_targets


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


N_TEMPORAL_SLOTS = 8
SLOT_DIM = 1024
KEY_TARGETS = ["ee_position", "ee_velocity", "ee_acceleration",
               "contact_flag", "contact_force_log1p_mag"]


def slot_view(feats: np.ndarray, slot: int) -> np.ndarray:
    """feats [N, 24, 8192] → [N, 24, 1024] for the given slot t in 0..7."""
    return feats[:, :, slot * SLOT_DIM:(slot + 1) * SLOT_DIM]


def trajectory_geom(X: np.ndarray) -> dict:
    """X: [T, D] one episode at one (layer, slot)."""
    T = X.shape[0]
    if T < 2:
        return {"path_length": 0.0, "mean_speed": 0.0, "tortuosity": 1.0}
    d = np.diff(X, axis=0)
    speeds = np.linalg.norm(d, axis=1)
    direct = float(np.linalg.norm(X[-1] - X[0]))
    return {
        "path_length": float(speeds.sum()),
        "mean_speed": float(speeds.mean()),
        "tortuosity": float(speeds.sum()) / max(direct, 1e-9),
    }


def fit_probe_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray, device: str = "cuda:0") -> tuple[float, float]:
    """Quick GroupKFold(5) Ridge probe — returns (R²_mean, R²_std).
    Uses sklearn Ridge for speed (no need for full Adam sweep here)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if not ok.all():
        X = X[ok]; y = y[ok]; groups = groups[ok]
    if y.shape[0] < 100:
        return float("nan"), float("nan")
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        # Standardize within fold
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        mu_x = X_tr.mean(0); sd_x = X_tr.std(0) + 1e-9
        X_tr_n = (X_tr - mu_x) / sd_x
        X_te_n = (X_te - mu_x) / sd_x
        if y_tr.ndim == 1:
            mu_y, sd_y = y_tr.mean(), y_tr.std() + 1e-9
        else:
            mu_y, sd_y = y_tr.mean(0), y_tr.std(0) + 1e-9
        y_tr_n = (y_tr - mu_y) / sd_y
        model = Ridge(alpha=1.0)
        model.fit(X_tr_n, y_tr_n)
        pred = model.predict(X_te_n) * sd_y + mu_y
        r2s.append(r2_score(y_te, pred, multioutput="variance_weighted") if y_te.ndim > 1 else r2_score(y_te, pred))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    geom_rows = []
    probe_rows = []
    dim_rows = []

    for task in ALL_TASKS:
        print(f"[14_per_slot] {task} loading ...", flush=True)
        trajs = all_trajectories(task)
        tgt = load_targets(task)
        keys_tgt = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
        lut = {k: i for i, k in enumerate(keys_tgt)}

        # Build aligned arrays
        all_feats = []
        all_eps = []
        all_t_last = []
        for tj in trajs:
            ep = int(tj["episode_id"])
            T = tj["t_last"].size
            all_feats.append(tj["feats"])    # [T, 24, 8192]
            all_eps.append(np.full(T, ep, dtype=np.int64))
            all_t_last.append(tj["t_last"])

        feats_full = np.concatenate(all_feats, axis=0)  # [N_total, 24, 8192]
        eps_arr = np.concatenate(all_eps, axis=0)
        t_last = np.concatenate(all_t_last, axis=0)

        # Build target index map
        rows_idx = np.array([lut[(int(g), int(t))] for g, t in zip(eps_arr, t_last)], dtype=np.int64)
        target_data = {}
        for tk in KEY_TARGETS:
            if tk in tgt.files:
                target_data[tk] = tgt[tk][rows_idx]

        # === Per (layer, slot) loop ===
        for L in range(24):
            t_layer = time.time()
            for s in range(N_TEMPORAL_SLOTS):
                X_slot = feats_full[:, L, s * SLOT_DIM:(s + 1) * SLOT_DIM]   # [N_total, 1024]

                # 1. Geometry: per-episode stats, then average
                pl_list = []; ms_list = []; tort_list = []
                for tj in trajs:
                    Tj = tj["t_last"].size
                    Xj = tj["feats"][:, L, s * SLOT_DIM:(s + 1) * SLOT_DIM]
                    g = trajectory_geom(Xj)
                    pl_list.append(g["path_length"]); ms_list.append(g["mean_speed"]); tort_list.append(g["tortuosity"])
                geom_rows.append({"task": task, "layer": L, "slot": s,
                                  "path_length_mean": float(np.mean(pl_list)),
                                  "mean_speed_mean": float(np.mean(ms_list)),
                                  "tortuosity_mean": float(np.mean(tort_list))})

                # 2. Probe R² for key targets
                for tk in KEY_TARGETS:
                    if tk not in target_data:
                        continue
                    r2_mean, r2_std = fit_probe_r2(X_slot, target_data[tk], eps_arr)
                    probe_rows.append({"task": task, "layer": L, "slot": s, "target": tk,
                                       "r2_mean": r2_mean, "r2_std": r2_std})

                # 3. Intrinsic dim
                if X_slot.shape[0] > 64:
                    nc = min(64, X_slot.shape[1], X_slot.shape[0] - 1)
                    pca = PCA(n_components=nc)
                    pca.fit(X_slot)
                    cum = np.cumsum(pca.explained_variance_ratio_)
                    n90 = int(np.searchsorted(cum, 0.90) + 1) if (cum >= 0.90).any() else nc
                    n95 = int(np.searchsorted(cum, 0.95) + 1) if (cum >= 0.95).any() else nc
                    dim_rows.append({"task": task, "layer": L, "slot": s,
                                     "n_pcs_90": n90, "n_pcs_95": n95})
            print(f"[14_per_slot] {task} L{L:02d}: 8 slots in {time.time()-t_layer:.1f}s", flush=True)

    geom_df = pd.DataFrame(geom_rows); geom_df.to_csv(STATS / "per_slot_geometry.csv", index=False)
    probe_df = pd.DataFrame(probe_rows); probe_df.to_csv(STATS / "per_slot_stats.csv", index=False)
    dim_df = pd.DataFrame(dim_rows); dim_df.to_csv(STATS / "per_slot_intrinsic_dim.csv", index=False)
    print(f"[14_per_slot] wrote stats CSVs", flush=True)

    # Plots: layer×slot heatmaps
    for task in ALL_TASKS:
        # Probe target heatmaps
        for tk in KEY_TARGETS:
            sub = probe_df[(probe_df.task == task) & (probe_df.target == tk)]
            if sub.empty:
                continue
            M = np.full((24, N_TEMPORAL_SLOTS), np.nan)
            for _, r in sub.iterrows():
                M[int(r.layer), int(r.slot)] = r.r2_mean
            fig, ax = plt.subplots(figsize=(6, 7))
            im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
            ax.set_xlabel("Temporal slot (0=earliest, 7=latest in window)")
            ax.set_ylabel("Layer")
            ax.set_title(f"Per-slot R² — {task} / {tk}")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            fig.savefig(PLOTS / f"per_slot_{task}_{tk}.png", dpi=130)
            plt.close(fig)

        # Geometry heatmaps (path_length and tortuosity)
        for col in ("path_length_mean", "tortuosity_mean"):
            sub = geom_df[geom_df.task == task]
            M = np.full((24, N_TEMPORAL_SLOTS), np.nan)
            for _, r in sub.iterrows():
                M[int(r.layer), int(r.slot)] = r[col]
            fig, ax = plt.subplots(figsize=(6, 7))
            im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
            ax.set_xlabel("Temporal slot")
            ax.set_ylabel("Layer")
            ax.set_title(f"{col} — {task}")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            fig.savefig(PLOTS / f"per_slot_geometry_{col}_{task}.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
