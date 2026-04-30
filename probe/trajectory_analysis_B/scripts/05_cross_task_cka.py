#!/usr/bin/env python3
"""Cross-task CKA per layer: which layers represent tasks similarly vs differently.

For each layer L: compute CKA between every pair of (task_i, task_j) using
their feature pools (subsampled). Note: tasks have different N and unrelated
samples — strict CKA assumes paired samples. Here we use a subsampled-pool
CKA which measures whether the FEATURE GRAM STRUCTURE is similar between
tasks, treating each task's features as a separate sample of the same
underlying manifold.

Outputs:
- results/stats/cross_task_cka_layerwise.csv: layer × pair → cka
- results/plots/cross_task_cka_layer<L>.png: 6x6 heatmap per selected layer
- results/plots/cross_task_cka_evolution.png: avg pairwise CKA per layer (line plot)
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_utils.loader import ALL_TASKS, all_trajectories


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


def linear_cka_unpaired(X: np.ndarray, Y: np.ndarray) -> float:
    """When N_X != N_Y, we compare via the feature Gram structure (d×d)
    rather than sample Gram (N×N). This measures whether two layers (across
    different tasks) have the same eigenstructure / covariance. For Gaussian
    or linear models this matches paired CKA in expectation.
    Equivalent to: CKA(C_X, C_Y) where C_X = XᵀX (d×d centered)."""
    d = X.shape[1]
    if Y.shape[1] != d:
        raise ValueError("CKA-unpaired requires same feature dim")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Cx = Xc.T @ Xc / max(X.shape[0] - 1, 1)
    Cy = Yc.T @ Yc / max(Y.shape[0] - 1, 1)
    num = float(np.sum(Cx * Cy))
    den = np.sqrt(float(np.sum(Cx * Cx)) * float(np.sum(Cy * Cy)))
    return num / max(den, 1e-12)


def stack_layer(trajs: list[dict], L: int, sub: int) -> np.ndarray:
    X = np.concatenate([t["feats"][:, L, :] for t in trajs], axis=0).astype(np.float32)
    if X.shape[0] > sub:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], sub, replace=False)
        X = X[idx]
    return X


def main():
    print("[05_cross_task] loading per-task trajectories...", flush=True)
    trajs_per_task = {t: all_trajectories(t) for t in ALL_TASKS}

    rows = []
    avg_per_layer = []
    pair_keys = list(combinations(ALL_TASKS, 2))

    for L in range(24):
        layer_feats = {t: stack_layer(trajs_per_task[t], L, sub=3000) for t in ALL_TASKS}
        cka_matrix = np.eye(len(ALL_TASKS), dtype=np.float32)
        for i, ti in enumerate(ALL_TASKS):
            for j in range(i + 1, len(ALL_TASKS)):
                tj = ALL_TASKS[j]
                v = linear_cka_unpaired(layer_feats[ti], layer_feats[tj])
                cka_matrix[i, j] = cka_matrix[j, i] = v
                rows.append({"layer": L, "task_i": ti, "task_j": tj, "cka": v})
        # Average pairwise CKA (excluding diagonal)
        n = len(ALL_TASKS)
        avg = (cka_matrix.sum() - n) / (n * (n - 1))
        avg_per_layer.append({"layer": L, "avg_cka": float(avg)})

        if L in (0, 6, 12, 18, 23):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(ALL_TASKS, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(ALL_TASKS, fontsize=8)
            ax.set_title(f"Cross-task feature CKA — Layer {L:02d}", fontsize=11)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{cka_matrix[i, j]:.2f}", ha="center", va="center",
                            color="white" if cka_matrix[i, j] < 0.6 else "black", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
            fig.savefig(PLOTS / f"cross_task_cka_layer{L:02d}.png", dpi=130)
            plt.close(fig)
            print(f"[05_cross_task] wrote heatmap for L{L:02d}", flush=True)

    pd.DataFrame(rows).to_csv(STATS / "cross_task_cka_layerwise.csv", index=False)
    avg_df = pd.DataFrame(avg_per_layer)
    avg_df.to_csv(STATS / "cross_task_cka_avg_per_layer.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(avg_df.layer, avg_df.avg_cka, "-o", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pairwise CKA across 6 tasks")
    ax.set_title("Cross-task representational similarity per layer")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, 1)
    ax.axvspan(6, 18, alpha=0.05, color="green")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cross_task_cka_evolution.png", dpi=130)
    plt.close(fig)
    print(f"[05_cross_task] wrote evolution plot", flush=True)


if __name__ == "__main__":
    main()
