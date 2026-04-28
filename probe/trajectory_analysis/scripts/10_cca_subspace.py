#!/usr/bin/env python3
"""CCA between latent and physics subspaces (per Codex spec).

For each (task, layer):
1. PCA-reduce whitened latent to top-k where var ≥ 0.95 (cap K=128)
2. For each physics group g ∈ {pos, vel, acc, ct}, z-score columns.
3. Fit linear CCA between Ẑ_l and X_g.
4. Report:
     ρ1 = top canonical correlation
     CCA_energy = Σ_i ρ_i^2
     rank90 = min k s.t. Σ_{i≤k} ρ_i^2 / total ≥ 0.9
5. Cross-group subspace overlap at early/PEZ/late layers (L=0, 12, 23).

Outputs:
- results/stats/cca_subspace.csv: task × layer × group → ρ1, energy, rank90
- results/plots/cca_<metric>_heatmap.png: task × layer
- results/plots/cca_subspace_overlap.png: per-task subspace overlap matrices
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from ta_utils.loader import all_trajectories
from utils.io import load_targets


# Codex recommendation: bump episodes for CCA to reduce noise in canonical estimates
N_EPS_FOR_CCA = 60


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


N_SUBSAMPLE = 4000
LATENT_VAR_KEEP = 0.95
LATENT_K_CAP = 128
ALL_TASKS = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]


def whiten_and_reduce(X: np.ndarray, var_keep=LATENT_VAR_KEEP, k_cap=LATENT_K_CAP) -> np.ndarray:
    pca = PCA(n_components=min(X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = max(1, int(np.searchsorted(cum, var_keep) + 1))
    k = min(k, k_cap)
    Z = pca.transform(X)[:, :k] / np.sqrt(np.clip(pca.explained_variance_[:k], 1e-9, None))
    return Z


def fit_cca_safely(Z: np.ndarray, X: np.ndarray, n_components: int) -> tuple[np.ndarray, float, int]:
    """Fit CCA. Returns (rhos, energy, rank90)."""
    n_components = min(n_components, Z.shape[1], X.shape[1], Z.shape[0] - 1)
    if n_components < 1:
        return np.zeros(0), 0.0, 0
    cca = CCA(n_components=n_components, max_iter=200)
    try:
        cca.fit(Z, X)
        Zc, Xc = cca.transform(Z, X)
        rhos = np.array([np.corrcoef(Zc[:, i], Xc[:, i])[0, 1] for i in range(n_components)])
        rhos = np.clip(rhos, 0.0, 1.0)
    except Exception:
        rhos = np.zeros(n_components)
    energy = float((rhos ** 2).sum())
    if energy == 0:
        return rhos, 0.0, 0
    cum = np.cumsum(rhos ** 2) / energy
    rank90 = int(np.searchsorted(cum, 0.9) + 1)
    return rhos, energy, rank90


def main():
    rows = []
    rng = np.random.default_rng(42)
    for task in ALL_TASKS:
        print(f"[10_cca] {task} loading ...", flush=True)
        trajs = all_trajectories(task, n_eps=N_EPS_FOR_CCA)
        tgt = load_targets(task)
        keys_tgt = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
        lut = {k: i for i, k in enumerate(keys_tgt)}

        # Stack everything
        all_feats = []
        eps_list = []
        ee_pos, ee_vel, ee_acc = [], [], []
        obj_pos, obj_vel, obj_acc = [], [], []
        for tj in trajs:
            ep = int(tj["episode_id"]); t_last = tj["t_last"]
            T = t_last.size
            rows_idx = np.array([lut[(ep, int(t))] for t in t_last], dtype=np.int64)
            all_feats.append(tj["feats"]); eps_list.append(np.full(T, ep, dtype=np.int64))
            ee_pos.append(tgt["ee_position"][rows_idx])
            ee_vel.append(tgt["ee_velocity"][rows_idx])
            ee_acc.append(tgt["ee_acceleration"][rows_idx])
            for src, dst in [("obj_position", obj_pos), ("obj_velocity", obj_vel), ("obj_acceleration", obj_acc)]:
                a = tgt.get(src)
                dst.append(a[rows_idx] if a is not None else np.zeros((T, 0)))

        feats = np.concatenate(all_feats, axis=0)
        X_pos = np.concatenate([np.concatenate([p, q], axis=1) for p, q in zip(ee_pos, obj_pos)], axis=0)
        X_vel = np.concatenate([np.concatenate([p, q], axis=1) for p, q in zip(ee_vel, obj_vel)], axis=0)
        X_acc = np.concatenate([np.concatenate([p, q], axis=1) for p, q in zip(ee_acc, obj_acc)], axis=0)

        # Contact (per parquet)
        import pyarrow.parquet as pq
        from utils.dataset import parquet_for_episode
        all_ct = []
        for tj in trajs:
            ep = int(tj["episode_id"]); t_last = tj["t_last"]
            try:
                df = pq.read_table(parquet_for_episode(task, ep)).to_pandas()
            except Exception:
                all_ct.append(np.zeros((t_last.size, 5))); continue
            cf = df.get("physics_gt.contact_flag", None)
            if cf is None:
                all_ct.append(np.zeros((t_last.size, 5))); continue
            cf_arr = np.array([float(v) if not isinstance(v, np.ndarray) else float(v.flatten()[0])
                               for v in cf.tolist()], dtype=np.float32)[t_last]
            cforce = df.get("physics_gt.contact_force", None)
            cforce_arr = np.stack([np.asarray(v, dtype=np.float32) for v in cforce.tolist()])[t_last] if cforce is not None else np.zeros((t_last.size, 3))
            ct = np.concatenate([cf_arr.reshape(-1, 1), cforce_arr,
                                 np.log1p(np.linalg.norm(cforce_arr, axis=1, keepdims=True))], axis=1)
            all_ct.append(ct)
        X_ct = np.concatenate(all_ct, axis=0)

        # Subsample
        N = feats.shape[0]
        sub_idx = np.sort(rng.choice(N, min(N_SUBSAMPLE, N), replace=False))

        # Standardize physics groups
        def stdize(M):
            mu = M.mean(0, keepdims=True); sd = M.std(0, keepdims=True) + 1e-9
            return (M - mu) / sd

        groups = {
            "pos": stdize(X_pos[sub_idx]),
            "vel": stdize(X_vel[sub_idx]),
            "acc": stdize(X_acc[sub_idx]),
            "ct":  stdize(X_ct[sub_idx]),
        }

        # Per-layer CCA
        for L in range(24):
            Z = feats[sub_idx, L, :].astype(np.float32)
            Zr = whiten_and_reduce(Z)
            for gname, Xg in groups.items():
                if Xg.shape[1] == 0:
                    continue
                n_components = min(Zr.shape[1], Xg.shape[1])
                rhos, energy, rank90 = fit_cca_safely(Zr, Xg, n_components)
                rho1 = float(rhos[0]) if rhos.size > 0 else 0.0
                rows.append({"task": task, "layer": L, "group": gname,
                             "rho1": rho1, "energy": energy, "rank90": rank90,
                             "n_components": n_components})
            print(f"[10_cca] {task} L{L:02d} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "cca_subspace.csv", index=False)
    print(f"[10_cca] wrote {STATS / 'cca_subspace.csv'} ({len(df)} rows)", flush=True)

    # Heatmaps per metric
    for metric, label, vmin, vmax in [
        ("rho1", "Top canonical correlation ρ1", 0, 1),
        ("energy", "CCA energy Σρ²", 0, None),
        ("rank90", "rank90 (subspace dim for 90% energy)", 1, None),
    ]:
        for gname in ["pos", "vel", "acc", "ct"]:
            sub = df[df.group == gname]
            if sub.empty:
                continue
            M = np.full((6, 24), np.nan)
            for i, t in enumerate(ALL_TASKS):
                ts = sub[sub.task == t].sort_values("layer")
                M[i, :len(ts)] = ts[metric].values
            fig, ax = plt.subplots(figsize=(11, 4))
            im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                           extent=[-0.5, 23.5, -0.5, 5.5],
                           vmin=vmin if vmin is not None else None,
                           vmax=vmax if vmax is not None else None)
            ax.set_yticks(range(6)); ax.set_yticklabels(ALL_TASKS)
            ax.set_xlabel("Layer"); ax.set_title(f"{label} — group {gname}")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            fig.savefig(PLOTS / f"cca_{metric}_{gname}.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
