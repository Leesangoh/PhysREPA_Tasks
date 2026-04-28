#!/usr/bin/env python3
"""Koopman-style linear dynamics scores per layer (Codex bonus).

Two metrics per (task, layer):
A. Latent self-predictability — fit z_{t+1} ≈ A_l z_t (Ridge), report R²_self.
   Compare to PERSISTENCE baseline z_{t+1} ≈ z_t (zero-prediction).
   Headline: ΔR²_self = R²_self − R²_persist.
B. Next-physics predictability from latent — fit x_{t+1} ≈ B_l z_t for several
   physics groups (next-step velocity, next-step acceleration, next-step contact).
   Report R² per group.

Implementation notes:
- Whitened latent (PCA-whiten per layer, var_keep=0.99, K capped 128).
- Episode-aware: drop boundary samples where t+1 crosses episode end.
- Train/test split by episode (80/20), seed=42.
- Ridge alpha = 1.0 baseline.

Outputs:
- results/stats/koopman.csv  (task × layer × metric → R²)
- results/plots/koopman_self_<metric>.png  (task × layer heatmap)
- results/plots/koopman_phys_<group>.png   (task × layer heatmap per group)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from ta_utils.loader import all_trajectories, ALL_TASKS
from utils.io import load_targets


N_EPS_FOR_KOOPMAN = 60


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


WHITEN_VARIANCE = 0.99
K_CAP = 128


def whiten(X: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=min(X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = max(1, int(np.searchsorted(cum, WHITEN_VARIANCE) + 1))
    k = min(k, K_CAP)
    return (X - pca.mean_) @ pca.components_[:k].T / np.sqrt(np.clip(pca.explained_variance_[:k], 1e-9, None))


def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0)) ** 2).sum() + 1e-12)
    return 1 - ss_res / ss_tot


def fit_eval_ridge(X_train, X_test, y_train, y_test, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return r2_score(y_test, model.predict(X_test))


def main():
    rows = []
    for task in ALL_TASKS:
        print(f"[13_koopman] {task} loading ...", flush=True)
        trajs = all_trajectories(task, n_eps=N_EPS_FOR_KOOPMAN)
        tgt = load_targets(task)
        keys_tgt = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
        lut = {k: i for i, k in enumerate(keys_tgt)}

        # Build per-window arrays + episode boundaries
        all_feats = []
        eps_list = []
        ee_pos, ee_vel, ee_acc, obj_pos, obj_vel, obj_acc = [], [], [], [], [], []
        offsets = [0]
        for tj in trajs:
            ep = int(tj["episode_id"]); t_last = tj["t_last"]
            T = t_last.size
            rows_idx = np.array([lut[(ep, int(t))] for t in t_last], dtype=np.int64)
            all_feats.append(tj["feats"])
            eps_list.append(np.full(T, ep, dtype=np.int64))
            ee_pos.append(tgt["ee_position"][rows_idx])
            ee_vel.append(tgt["ee_velocity"][rows_idx])
            ee_acc.append(tgt["ee_acceleration"][rows_idx])
            for src, dst in [("obj_position", obj_pos), ("obj_velocity", obj_vel), ("obj_acceleration", obj_acc)]:
                a = tgt.get(src)
                dst.append(a[rows_idx] if a is not None else np.zeros((T, 0)))
            offsets.append(offsets[-1] + T)

        feats = np.concatenate(all_feats, axis=0)
        eps_arr = np.concatenate(eps_list)
        # Build "valid t→t+1" mask (drop episode-last)
        valid_next = np.ones(feats.shape[0], dtype=bool)
        for o in offsets[1:]:
            if o > 0:
                valid_next[o - 1] = False

        # Episode-level train/test split (80/20)
        unique_eps = np.unique(eps_arr)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_eps)
        ntr = max(int(unique_eps.size * 0.8), 1)
        train_eps = set(unique_eps[:ntr].tolist())
        train_mask = np.isin(eps_arr, list(train_eps))
        test_mask = ~train_mask
        train_pred = train_mask & valid_next   # rows where we use z_t and z_{t+1} exists
        test_pred = test_mask & valid_next

        ee_pos_arr = np.concatenate(ee_pos, axis=0)
        ee_vel_arr = np.concatenate(ee_vel, axis=0)
        ee_acc_arr = np.concatenate(ee_acc, axis=0)
        obj_vel_arr = np.concatenate(obj_vel, axis=0) if obj_vel and obj_vel[0].shape[1] > 0 else np.zeros((feats.shape[0], 0))

        for L in range(24):
            Z_full = whiten(feats[:, L, :].astype(np.float32))
            # Indices of "current" rows where t+1 exists (train/test split as above)
            idx_tr = np.where(train_pred)[0]
            idx_te = np.where(test_pred)[0]

            # A. Self-prediction z_{t+1} from z_t
            X_tr = Z_full[idx_tr]
            Y_tr = Z_full[idx_tr + 1]
            X_te = Z_full[idx_te]
            Y_te = Z_full[idx_te + 1]
            r2_self = fit_eval_ridge(X_tr, X_te, Y_tr, Y_te)
            # Persistence baseline: predict z_{t+1} = z_t
            r2_pers = r2_score(Y_te, X_te)
            r2_self_delta = r2_self - r2_pers

            # B. Next-physics predictability
            r2_phys = {}
            for name, arr in [("ee_pos_next", ee_pos_arr), ("ee_vel_next", ee_vel_arr),
                              ("ee_acc_next", ee_acc_arr), ("obj_vel_next", obj_vel_arr)]:
                if arr.size == 0:
                    continue
                Y_tr_p = arr[idx_tr + 1]
                Y_te_p = arr[idx_te + 1]
                r2_phys[name] = fit_eval_ridge(X_tr, X_te, Y_tr_p, Y_te_p)

            row = {"task": task, "layer": L, "r2_self": r2_self,
                   "r2_persist": r2_pers, "r2_self_delta": r2_self_delta}
            row.update(r2_phys)
            rows.append(row)
            print(f"[13_koopman] {task} L{L:02d} self={r2_self:.3f} delta={r2_self_delta:+.3f} "
                  + " ".join(f"{k}={v:.3f}" for k, v in r2_phys.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "koopman.csv", index=False)
    print(f"[13_koopman] wrote {STATS / 'koopman.csv'} ({len(df)} rows)", flush=True)

    # Heatmaps
    for col, label in [("r2_self_delta", "ΔR²_self (vs persistence)"),
                       ("ee_pos_next", "Next-step ee_pos R²"),
                       ("ee_vel_next", "Next-step ee_vel R²"),
                       ("ee_acc_next", "Next-step ee_acc R²"),
                       ("obj_vel_next", "Next-step obj_vel R²")]:
        if col not in df.columns:
            continue
        M = np.full((6, 24), np.nan)
        for i, t in enumerate(ALL_TASKS):
            sub = df[df.task == t].sort_values("layer")
            if col in sub.columns:
                M[i, :len(sub)] = sub[col].values
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                       extent=[-0.5, 23.5, -0.5, 5.5])
        ax.set_yticks(range(6)); ax.set_yticklabels(ALL_TASKS)
        ax.set_xlabel("Layer"); ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"koopman_{col}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
