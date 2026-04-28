#!/usr/bin/env python3
"""Shared-PCA basis per task (per Codex spec).

For each task:
1. Whiten each layer separately (PCA-whiten on inner-pool with var_keep=0.99).
2. Concatenate whitened features across all 24 layers (rows: window×layer; cols: dim).
   We only need a 2D shared basis, so we randomly subsample to keep PCA tractable.
3. Fit shared PCA(2) on concatenated whitened pool.
4. Project each layer's whitened features into the shared 2D basis.
5. Optional Procrustes alignment of adjacent layers (skipped — shared basis already comparable).
6. Plot 4×6 grid with the SAME PC1-PC2 axes for all 24 panels per task; overlay
   sample episode trajectories.

This produces directly-comparable layer panels — unlike per-layer PCA where each
panel has its own basis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from ta_utils.loader import all_trajectories, ALL_TASKS

PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


N_PCA_FIT = 8000      # subsample windows for shared PCA fit
N_SHOW_EPS = 6        # number of episodes to overlay per panel


def whiten_layer(X: np.ndarray, var_keep: float = 0.99, k_cap: int = 256) -> tuple[np.ndarray, PCA, int]:
    pca = PCA(n_components=min(X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = max(1, int(np.searchsorted(cum, var_keep) + 1))
    k = min(k, k_cap)
    Z = (X - pca.mean_) @ pca.components_[:k].T / np.sqrt(np.clip(pca.explained_variance_[:k], 1e-9, None))
    return Z, pca, k


def main():
    rng = np.random.default_rng(42)
    for task in ALL_TASKS:
        print(f"[11_shared_pca] {task} loading ...", flush=True)
        trajs = all_trajectories(task)

        # Whiten each layer; remember ordering of windows
        whitened: list[np.ndarray] = []      # per-layer [N, k_l] whitened features
        for L in range(24):
            X = np.concatenate([t["feats"][:, L, :] for t in trajs], axis=0).astype(np.float32)
            Zw, _, _ = whiten_layer(X)
            whitened.append(Zw)

        # Determine min k across layers — pad each to common k_min so shared PCA works
        k_min = min(z.shape[1] for z in whitened)
        whitened_trim = [z[:, :k_min] for z in whitened]      # [N, k_min] each

        # Subsample windows for shared PCA fit
        N = whitened_trim[0].shape[0]
        sub = rng.choice(N, min(N_PCA_FIT, N), replace=False)

        # Stack per-layer subsamples into a big pool [24*N_sub, k_min]
        pool = np.concatenate([w[sub] for w in whitened_trim], axis=0)
        shared_pca = PCA(n_components=2)
        shared_pca.fit(pool)

        # Project each layer's full-window set into shared 2D basis
        proj_per_layer = [shared_pca.transform(w) for w in whitened_trim]   # list of [N, 2]

        # Plot per-task grid 4×6
        fig, axes = plt.subplots(4, 6, figsize=(20, 13), squeeze=False)
        fig.suptitle(f"Shared-PCA(2) trajectory — {task}  (basis fit on concatenated whitened pool across all 24 layers)",
                     fontsize=12)

        # Identify episode boundaries to draw individual trajectories
        offsets = [0]
        for tj in trajs:
            offsets.append(offsets[-1] + tj["t_last"].size)
        n_eps = len(trajs)
        show_idx = list(range(min(N_SHOW_EPS, n_eps)))
        cmap = plt.cm.tab10

        # Compute global axes limits to keep panels comparable
        all_pts = np.concatenate([p for p in proj_per_layer], axis=0)
        xlo, xhi = np.percentile(all_pts[:, 0], [0.5, 99.5])
        ylo, yhi = np.percentile(all_pts[:, 1], [0.5, 99.5])

        for L in range(24):
            ax = axes[L // 6, L % 6]
            P = proj_per_layer[L]
            for i in show_idx:
                P_ep = P[offsets[i]:offsets[i + 1]]
                color = cmap(i % 10)
                ax.plot(P_ep[:, 0], P_ep[:, 1], "-", color=color, alpha=0.7, linewidth=0.9)
                ax.scatter(P_ep[0, 0], P_ep[0, 1], color=color, s=18, marker="o", edgecolor="black", linewidth=0.4)
                ax.scatter(P_ep[-1, 0], P_ep[-1, 1], color=color, s=18, marker="s", edgecolor="black", linewidth=0.4)
            ax.set_title(f"L{L:02d}", fontsize=9)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)

        fig.tight_layout()
        out = PLOTS / f"shared_pca_{task}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[11_shared_pca] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
