#!/usr/bin/env python3
"""
t-SNE embedding trajectory visualization for V-JEPA 2 ViT-G features.

Usage:
    /isaac-sim/python.sh analysis/plot_trajectory_tsne.py
    /isaac-sim/python.sh analysis/plot_trajectory_tsne.py --num_episodes 200
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from safetensors import safe_open
from openTSNE import TSNE
from sklearn.decomposition import PCA

FEATURE_ROOT = Path("/mnt/md1/solee/features/physprobe_vitg")
DATA_ROOT = Path("/home/solee/data/data/isaac_physrepa_v2/step0")
OUTPUT_DIR = Path("/home/solee/physrepa_tasks/results")

DEFAULT_TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]
DEFAULT_LAYERS = [3, 13, 16, 39]
NUM_EPISODES = 300
SEED = 42

VELOCITY_COLUMNS = {
    "push": "physics_gt.object_velocity",
    "strike": "physics_gt.object_velocity",
    "peg_insert": "physics_gt.peg_velocity",
    "nut_thread": "physics_gt.nut_velocity",
    "drawer": "physics_gt.ee_velocity",
    "reach": "physics_gt.ee_velocity",
}

TASK_LABELS = {
    "push": "Push",
    "strike": "Strike",
    "peg_insert": "Peg Insert",
    "nut_thread": "Nut Thread",
    "drawer": "Drawer",
    "reach": "Reach\n(neg. ctrl)",
}

LAYER_LABELS = {
    3: "Layer 3\n(Early)",
    13: "Layer 13\n(PEZ)",
    16: "Layer 16\n(PEZ)",
    39: "Layer 39\n(Final)",
}


def get_episode_indices(task, n, seed):
    total = len(list((FEATURE_ROOT / task).glob("*.safetensors")))
    if n is None or n >= total:
        return list(range(total))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(total, size=n, replace=False).tolist())


def get_contact_onset_window(task, ep_idx, num_windows):
    if task == "reach":
        return None
    chunk = ep_idx // 1000
    p = DATA_ROOT / task / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    if not p.exists():
        return None
    vel_col = VELOCITY_COLUMNS.get(task)
    if vel_col is None:
        return None
    try:
        df = pd.read_parquet(str(p), columns=[vel_col])
    except Exception:
        return None
    vel = np.array([np.array(x) for x in df[vel_col].values])
    speed = np.linalg.norm(vel, axis=1)

    onset_frame = None
    if task == "push":
        m = speed > 0.01
        if m.any():
            onset_frame = int(np.where(m)[0][0])
    elif task == "strike":
        if speed.max() - speed.min() > 0.05:
            onset_frame = int(np.argmax(speed))
    elif task in ("peg_insert", "nut_thread"):
        if speed.max() > 0.01:
            onset_frame = int(np.argmax(speed))
    elif task == "drawer":
        if speed.max() > 0.5:
            onset_frame = int(np.argmax(speed))

    if onset_frame is None:
        return None
    return min(onset_frame // 4, num_windows - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tasks = args.tasks
    layers = args.layers
    nrows, ncols = len(tasks), len(layers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, task in enumerate(tasks):
        eps = get_episode_indices(task, args.num_episodes, args.seed)
        onsets = [get_contact_onset_window(task, e, 58) for e in eps]
        n_contact = sum(1 for o in onsets if o is not None)
        print(f"[{task}] {len(eps)} eps, contact={n_contact}")

        for col, layer in enumerate(layers):
            ax = axes[row, col]
            print(f"  L{layer}...", end=" ", flush=True)

            feats_list = []
            for ep in eps:
                with safe_open(str(FEATURE_ROOT / task / f"{ep:06d}.safetensors"), framework="pt") as f:
                    ws = f.get_tensor("window_starts").numpy()
                    vecs = [f.get_tensor(f"layer_{layer}_window_{w}").float().numpy() for w in range(len(ws))]
                feats_list.append(np.stack(vecs))

            all_vecs = np.vstack(feats_list)

            # PCA to 50D then t-SNE (openTSNE: FFT-accelerated, multi-threaded)
            pca50 = PCA(n_components=50).fit_transform(all_vecs)
            perp = min(30, max(5, len(all_vecs) // 10))
            projected_all = TSNE(
                n_components=2, perplexity=perp, random_state=42,
                initialization="pca", n_iter=400, n_jobs=-1,
            ).fit(pca50)

            offset = 0
            for ep_i, feats in enumerate(feats_list):
                nw = len(feats)
                proj = projected_all[offset : offset + nw]
                offset += nw
                t_norm = np.linspace(0, 1, nw)
                colors = plt.cm.coolwarm(t_norm)
                points = proj.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=colors[:-1], linewidths=0.4, alpha=0.25)
                ax.add_collection(lc)

                onset = onsets[ep_i]
                if onset is not None and onset < nw:
                    ax.scatter(
                        proj[onset, 0], proj[onset, 1],
                        marker="*", s=40, c="gold", edgecolors="black",
                        linewidths=0.2, zorder=5, alpha=0.5,
                    )

            ax.autoscale()
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(LAYER_LABELS.get(layer, f"Layer {layer}"), fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(TASK_LABELS.get(task, task), fontsize=12, fontweight="bold")
            print("done")
            del feats_list, all_vecs, pca50

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.35, pad=0.03, aspect=30)
    cbar.set_label("Time progression", fontsize=11)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Start", "End"])

    from matplotlib.lines import Line2D

    fig.legend(
        handles=[Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                        markeredgecolor="black", markersize=11, label="Contact onset")],
        loc="lower center", ncol=1, fontsize=10, frameon=True, bbox_to_anchor=(0.45, -0.01),
    )

    fig.suptitle(
        f"V-JEPA 2 ViT-G Embedding Trajectories — t-SNE 2D ({args.num_episodes} eps/task)",
        fontsize=14, fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.87, top=0.94, bottom=0.03, wspace=0.15, hspace=0.15)

    out = OUTPUT_DIR / f"embedding_trajectory_vitg_tsne_{args.num_episodes}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
