#!/usr/bin/env python3
"""Per-task PCA(2) visualization across all 24 layers.

For each task: one large figure with 4x6 grid of panels (24 layers).
In each panel:
- PCA fit on stacked features from all sampled episodes for that layer
- Plot up to 8 individual episode trajectories on PC1-PC2 plane (line + start/end markers)
- Color: a fixed palette across episodes for visual distinguishability

Saves: results/plots/pca_<task>.png
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

from ta_utils.loader import ALL_TASKS, all_trajectories


PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_one_task(task: str, n_show_eps: int = 8) -> Path:
    trajs = all_trajectories(task)
    cmap = plt.cm.tab10
    n_eps_total = len(trajs)
    show_idx = list(range(min(n_show_eps, n_eps_total)))

    fig, axes = plt.subplots(4, 6, figsize=(20, 13), squeeze=False)
    fig.suptitle(f"V-JEPA 2 ViT-L Variant A — Per-layer PCA(2) trajectory ({n_eps_total} sampled episodes)\n"
                 f"task = {task}", fontsize=12)
    for L in range(24):
        ax = axes[L // 6, L % 6]
        # Stack all episodes' features at this layer for fitting
        stacked = np.concatenate([t["feats"][:, L, :] for t in trajs], axis=0)  # [sum_T, 1024]
        try:
            pca = PCA(n_components=2)
            pca.fit(stacked)
            evr = pca.explained_variance_ratio_
        except Exception as e:
            ax.set_title(f"L{L:02d} — PCA failed ({e!s})", fontsize=8)
            continue

        for i in show_idx:
            X = trajs[i]["feats"][:, L, :]
            P = pca.transform(X)                              # [T, 2]
            color = cmap(i % 10)
            ax.plot(P[:, 0], P[:, 1], "-", color=color, alpha=0.7, linewidth=0.9)
            ax.scatter(P[0, 0], P[0, 1], color=color, s=18, marker="o", edgecolor="black", linewidth=0.4)
            ax.scatter(P[-1, 0], P[-1, 1], color=color, s=18, marker="s", edgecolor="black", linewidth=0.4)
        ax.set_title(f"L{L:02d}  evr={evr[0]*100:.0f}%/{evr[1]*100:.0f}%", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out = PLOTS / f"pca_{task}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[02_pca] wrote {out}", flush=True)
    return out


def main():
    for task in ALL_TASKS:
        print(f"[02_pca] {task} ...", flush=True)
        plot_one_task(task)


if __name__ == "__main__":
    main()
