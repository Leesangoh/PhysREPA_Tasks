#!/usr/bin/env python3
"""Per-(task, layer) intrinsic dimensionality.

We define intrinsic dim as the smallest k such that PCA explained variance
reaches a target threshold. Three thresholds reported: 90%, 95%, 99%.

Outputs:
- results/stats/intrinsic_dim.csv: task, layer, n_pcs_90, n_pcs_95, n_pcs_99, evr_top1, evr_top10
- results/plots/intrinsic_dim_per_layer.png: layer (x) vs n_pcs_90 (y), one line per task
- results/plots/evr_per_layer_<task>.png: spectrum (cumulative explained variance) per layer
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_utils.loader import ALL_TASKS, all_trajectories


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    cum_evr_per_task: dict[str, np.ndarray] = {}     # task -> [24, k] truncated cumulative spectrum

    K_keep = 64                                       # store top 64 components for plotting

    for task in ALL_TASKS:
        print(f"[03_dim] {task} ...", flush=True)
        trajs = all_trajectories(task)
        cum_mat = np.zeros((24, K_keep))
        for L in range(24):
            stacked = np.concatenate([t["feats"][:, L, :] for t in trajs], axis=0).astype(np.float32)
            n_components = min(K_keep, stacked.shape[0], stacked.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(stacked)
            evr = pca.explained_variance_ratio_
            cum = np.cumsum(evr)
            # Pad to K_keep for storage
            cum_padded = np.full(K_keep, cum[-1])
            cum_padded[: cum.size] = cum
            cum_mat[L] = cum_padded

            def first_above(thr: float) -> int:
                idx = np.argmax(cum_padded >= thr)
                if cum_padded[idx] < thr:
                    return -1
                return int(idx + 1)   # 1-indexed count of components

            rows.append({
                "task": task, "layer": L,
                "n_pcs_90": first_above(0.90),
                "n_pcs_95": first_above(0.95),
                "n_pcs_99": first_above(0.99),
                "evr_top1": float(evr[0]),
                "evr_top10": float(evr[: min(10, evr.size)].sum()),
                "n_total": int(stacked.shape[0]),
            })
        cum_evr_per_task[task] = cum_mat

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "intrinsic_dim.csv", index=False)
    print(f"[03_dim] wrote {STATS / 'intrinsic_dim.csv'} ({len(df)} rows)", flush=True)

    # Plot 1: layer vs n_pcs_90 across tasks
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = {"push": "C0", "strike": "C1", "reach": "C2", "drawer": "C3", "peg_insert": "C4", "nut_thread": "C5"}
    for task in ALL_TASKS:
        sub = df[df.task == task].sort_values("layer")
        ax.plot(sub.layer, sub.n_pcs_90, "-o", label=task, color=cmap[task], markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("# PCA components for 90% variance")
    ax.set_title("Intrinsic dimensionality per layer (V-JEPA 2 ViT-L Variant A)")
    ax.set_xlim(-0.5, 23.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.axvspan(6, 18, alpha=0.05, color="green")
    fig.tight_layout()
    fig.savefig(PLOTS / "intrinsic_dim_per_layer.png", dpi=130)
    plt.close(fig)
    print(f"[03_dim] wrote intrinsic_dim_per_layer.png", flush=True)

    # Plot 2: per-task explained variance spectrum (cumulative) for selected layers
    selected_layers = [0, 6, 12, 18, 23]
    for task in ALL_TASKS:
        cum_mat = cum_evr_per_task[task]
        fig, ax = plt.subplots(figsize=(8, 5))
        for L in selected_layers:
            ax.plot(np.arange(1, K_keep + 1), cum_mat[L], label=f"L{L:02d}", linewidth=1.3)
        ax.axhline(0.9, ls=":", color="gray")
        ax.axhline(0.95, ls=":", color="gray")
        ax.set_xlabel("# PCA components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title(f"PCA spectrum — {task}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(1, K_keep)
        fig.tight_layout()
        fig.savefig(PLOTS / f"evr_spectrum_{task}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
