#!/usr/bin/env python3
"""Cross-layer CKA per task: 24x24 similarity matrix.

CKA (linear, centered): given X1 [N, d1], X2 [N, d2] (same N samples),
HSIC(X1, X2) = ||X1ᵀX2||_F² / (N-1)²
CKA = HSIC(X1, X2) / sqrt(HSIC(X1, X1) · HSIC(X2, X2))

We use centered features and compute CKA via Gram matrices for fairness when
layer dims differ (here all 1024 — but pattern is reusable).

Outputs:
- results/stats/cross_layer_cka_<task>.csv: 24x24
- results/plots/cross_layer_cka_<task>.png: heatmap
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """X: [N, d1], Y: [N, d2]. Returns CKA scalar in [0, 1]."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.sum((Xc.T @ Yc) ** 2))
    denom_x = float(np.sum((Xc.T @ Xc) ** 2))
    denom_y = float(np.sum((Yc.T @ Yc) ** 2))
    return num / max(np.sqrt(denom_x * denom_y), 1e-12)


def stack_layer(trajs: list[dict], L: int) -> np.ndarray:
    return np.concatenate([t["feats"][:, L, :] for t in trajs], axis=0).astype(np.float32)


def main():
    for task in ALL_TASKS:
        print(f"[04_cka] {task} loading ...", flush=True)
        trajs = all_trajectories(task)
        # Subsample windows for speed (CKA is O(N·d) so bring N down to ~5000)
        layer_feats: list[np.ndarray] = []
        for L in range(24):
            X = stack_layer(trajs, L)
            if X.shape[0] > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(X.shape[0], 5000, replace=False)
                X = X[idx]
            layer_feats.append(X)

        # Compute symmetric 24x24 CKA
        M = np.zeros((24, 24), dtype=np.float32)
        for i in range(24):
            for j in range(i, 24):
                v = linear_cka(layer_feats[i], layer_feats[j])
                M[i, j] = v
                M[j, i] = v

        df = pd.DataFrame(M, index=[f"L{i:02d}" for i in range(24)],
                          columns=[f"L{j:02d}" for j in range(24)])
        df.to_csv(STATS / f"cross_layer_cka_{task}.csv")

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks(range(0, 24, 2))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"Cross-layer CKA — {task}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Linear CKA")
        fig.tight_layout()
        fig.savefig(PLOTS / f"cross_layer_cka_{task}.png", dpi=130)
        plt.close(fig)
        print(f"[04_cka] wrote heatmap for {task}", flush=True)


if __name__ == "__main__":
    main()
