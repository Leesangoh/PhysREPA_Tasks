#!/usr/bin/env python3
"""Tangent-space RSA: latent dynamics vs physical dynamics (per Codex spec).

For each (task, layer):
  U_l[t] = Z_white[t+1] - Z_white[t]   first-order latent tangent
  A_l[t] = U_l[t+1] - U_l[t]           second-order latent tangent
  V[t]   = X_vel[t]                    physical velocity (already a tangent)
  G[t]   = X_acc[t]                    physical acceleration
  C[t]   = X_ct[t]                     contact

  RSA_vel_tan(l) = corr(rank(pdist(U_l)), rank(pdist(V)))
  RSA_acc_tan(l) = corr(rank(pdist(A_l)), rank(pdist(G)))
  RSA_ct_tan(l)  = partial corr(A_l, C | pos+vel)

Outputs:
- results/stats/rsa_tangent.csv  (task × layer × rsa_vel_tan, rsa_acc_tan, rsa_ct_tan)
- results/plots/rsa_tangent_<metric>.png  task × layer heatmaps
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA

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


N_SUBSAMPLE = 1500
WHITEN_VARIANCE = 0.99
ALL_TASKS = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]


def whiten_pca(X, var_keep=WHITEN_VARIANCE):
    pca = PCA(n_components=min(X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = max(1, int(np.searchsorted(cum, var_keep) + 1))
    Z = pca.transform(X)[:, :k] / np.sqrt(np.clip(pca.explained_variance_[:k], 1e-9, None))
    return Z


def residualize(y, X_nuisance):
    X = np.concatenate([np.ones((X_nuisance.shape[0], 1)), X_nuisance], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())


def main():
    rows = []
    for task in ALL_TASKS:
        print(f"[09_tangent_rsa] {task} loading ...", flush=True)
        trajs = all_trajectories(task)
        tgt = load_targets(task)
        keys_tgt = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
        lut = {k: i for i, k in enumerate(keys_tgt)}

        # Build per-window arrays + episode boundaries (so we can drop t→t+1 spanning two episodes)
        all_feats = []      # [N, 24, D]
        eps_list = []
        ee_pos = []
        ee_vel = []
        ee_acc = []
        obj_pos = []; obj_vel = []; obj_acc = []
        offsets = [0]
        for tj in trajs:
            ep = int(tj["episode_id"])
            t_last = tj["t_last"]
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
        X_pos = np.concatenate([np.concatenate([p, q], axis=1) for p, q in zip(ee_pos, obj_pos)], axis=0)
        X_vel = np.concatenate([np.concatenate([p, q], axis=1) for p, q in zip(ee_vel, obj_vel)], axis=0)
        X_acc = np.concatenate([np.concatenate([p, q], axis=1) for p, q in zip(ee_acc, obj_acc)], axis=0)

        # Build "valid first-tangent" mask (drop last index of each episode)
        valid_first = np.ones(feats.shape[0], dtype=bool)
        for o in offsets[1:]:
            if o > 0:
                valid_first[o - 1] = False
        # For second-order tangent, drop last 2 of each episode
        valid_second = valid_first.copy()
        for o in offsets[1:]:
            if o >= 2:
                valid_second[o - 2] = False

        # Subsample indices among valid_second (so all metrics share the same row set)
        rng = np.random.default_rng(42)
        valid_idx = np.where(valid_second)[0]
        if valid_idx.size > N_SUBSAMPLE:
            sub_idx = np.sort(rng.choice(valid_idx, N_SUBSAMPLE, replace=False))
        else:
            sub_idx = valid_idx

        # Standardize physics columns
        def stdize(M):
            mu = M.mean(axis=0, keepdims=True); sd = M.std(axis=0, keepdims=True) + 1e-9
            return (M - mu) / sd

        # Velocity targets at the subsample positions
        V_sub = stdize(X_vel[sub_idx])
        # Acceleration: physical second-tangent target (use stored acc directly)
        G_sub = stdize(X_acc[sub_idx])
        # Pos for nuisance
        P_sub = stdize(X_pos[sub_idx])

        d_V = pdist(V_sub, metric="euclidean")
        d_G = pdist(G_sub, metric="euclidean")
        d_P = pdist(P_sub, metric="euclidean")
        r_V = rankdata(d_V); r_G = rankdata(d_G); r_P = rankdata(d_P)
        nuisance = np.column_stack([r_P, r_V])
        e_G = residualize(r_G, nuisance)

        for L in range(24):
            Z = feats[:, L, :].astype(np.float32)
            Zw = whiten_pca(Z)
            # First-order latent tangent: U_l[t] = Z_white[t+1] - Z_white[t]
            U = Zw[1:] - Zw[:-1]                                  # length N-1, but episode boundaries spurious
            # Pad U to align with original indices: U_full[t] = U[t] for t < N-1 else NaN
            U_full = np.zeros_like(Zw)
            U_full[:-1] = U
            # Second-order: A_l[t] = U[t+1] - U[t]
            A_full = np.zeros_like(Zw)
            A_full[:-2] = U[1:] - U[:-1]
            U_sub = U_full[sub_idx]
            A_sub = A_full[sub_idx]

            d_U = pdist(U_sub, metric="euclidean")
            d_A = pdist(A_sub, metric="euclidean")
            r_U = rankdata(d_U); r_A = rankdata(d_A)
            rsa_vel_tan = _corr(r_U, r_V)
            rsa_acc_tan = _corr(r_A, r_G)
            # Partial: A vs (something else if available); use contact below
            e_A = residualize(r_A, nuisance)
            rsa_acc_partial = _corr(e_A, e_G)
            rows.append({"task": task, "layer": L,
                         "rsa_vel_tan": rsa_vel_tan,
                         "rsa_acc_tan": rsa_acc_tan,
                         "rsa_acc_tan_partial": rsa_acc_partial})
            print(f"[09_tangent] {task} L{L:02d} vel_tan={rsa_vel_tan:.3f} acc_tan={rsa_acc_tan:.3f} acc|p+v={rsa_acc_partial:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "rsa_tangent.csv", index=False)
    print(f"[09_tangent] wrote {STATS / 'rsa_tangent.csv'} ({len(df)} rows)", flush=True)

    # Heatmaps
    for col, label in [("rsa_vel_tan", "Tangent RSA: U vs V"),
                       ("rsa_acc_tan", "Tangent RSA: A vs G"),
                       ("rsa_acc_tan_partial", "Tangent RSA: A vs G | pos,vel")]:
        M = np.full((6, 24), np.nan)
        for i, t in enumerate(ALL_TASKS):
            sub = df[df.task == t].sort_values("layer")
            M[i, :len(sub)] = sub[col].values
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                       extent=[-0.5, 23.5, -0.5, 5.5])
        ax.set_yticks(range(6)); ax.set_yticklabels(ALL_TASKS)
        ax.set_xlabel("Layer"); ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"{col}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
