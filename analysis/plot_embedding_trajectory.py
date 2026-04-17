#!/usr/bin/env python3
"""
Embedding trajectory visualization for V-JEPA 2 ViT-G features.
DIGAN Figure 11 style: PCA 2D trajectories colored by time, contact onset marked.

Usage:
    # All episodes (default)
    /isaac-sim/python.sh analysis/plot_embedding_trajectory.py

    # Subset
    /isaac-sim/python.sh analysis/plot_embedding_trajectory.py --num_episodes 50
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
from sklearn.decomposition import PCA

FEATURE_ROOT = Path("/mnt/md1/solee/features/physprobe_vitg")
DATA_ROOT = Path("/home/solee/data/data/isaac_physrepa_v2/step0")
OUTPUT_DIR = Path("/home/solee/physrepa_tasks/results")

DEFAULT_TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]
DEFAULT_LAYERS = [3, 13, 16, 39]
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


def get_episode_indices(task: str, num_episodes: int | None, seed: int) -> list:
    """Get episode indices — all if num_episodes is None, else sample."""
    feature_dir = FEATURE_ROOT / task
    total = len(list(feature_dir.glob("*.safetensors")))
    if num_episodes is None or num_episodes >= total:
        return list(range(total))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(total, size=num_episodes, replace=False).tolist())


def load_layer_features_batch(task: str, ep_indices: list, layer: int):
    """Load one layer's features for all episodes. Memory-efficient batch loading.

    Returns list of [num_windows, 1408] float32 arrays.
    """
    features = []
    for ep_idx in ep_indices:
        path = FEATURE_ROOT / task / f"{ep_idx:06d}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            ws = f.get_tensor("window_starts").numpy()
            vecs = [f.get_tensor(f"layer_{layer}_window_{w}").float().numpy() for w in range(len(ws))]
        features.append(np.stack(vecs))
    return features


def get_contact_onset_window(task: str, ep_idx: int, num_windows: int) -> int | None:
    """Find contact onset window index."""
    if task == "reach":
        return None

    chunk_idx = ep_idx // 1000
    parquet_path = (
        DATA_ROOT / task / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
    )
    if not parquet_path.exists():
        return None

    vel_col = VELOCITY_COLUMNS.get(task)
    if vel_col is None:
        return None

    try:
        df = pd.read_parquet(parquet_path, columns=[vel_col])
    except Exception:
        return None

    vel = np.array([np.array(x) for x in df[vel_col].values])
    speed = np.linalg.norm(vel, axis=1)
    num_frames = len(speed)

    onset_frame = None
    if task == "push":
        moving = speed > 0.01
        if moving.any():
            onset_frame = int(np.where(moving)[0][0])
    elif task == "strike":
        speed_range = speed.max() - speed.min()
        if speed_range > 0.05 and len(speed) > 1:
            onset_frame = int(np.argmax(speed))
    elif task in ("peg_insert", "nut_thread"):
        if len(speed) > 2 and speed.max() > 0.01:
            onset_frame = int(np.argmax(speed))
    elif task == "drawer":
        # EE velocity peak = grasping/pulling moment
        if len(speed) > 2 and speed.max() > 0.5:
            onset_frame = int(np.argmax(speed))
    else:
        threshold = np.mean(speed) + 2 * np.std(speed)
        above = speed > threshold
        if above.any():
            onset_frame = int(np.where(above)[0][0])

    if onset_frame is None:
        return None

    # Map frame to window index (stride=4)
    window_idx = min(onset_frame // 4, num_windows - 1)
    return window_idx


def plot_trajectories(
    tasks: list, layers: list, num_episodes: int | None, seed: int, output_path: Path
):
    nrows = len(tasks)
    ncols = len(layers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))

    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    layer_labels = {
        3: "Layer 3\n(Early)",
        13: "Layer 13\n(PEZ)",
        16: "Layer 16\n(PEZ)",
        39: "Layer 39\n(Final)",
    }

    for row, task in enumerate(tasks):
        ep_indices = get_episode_indices(task, num_episodes, seed)
        n_eps = len(ep_indices)
        print(f"[{task}] {n_eps} episodes")

        # Precompute contact onsets (need to know num_windows first, use 58 as approx)
        contact_onsets = []
        for ep in ep_indices:
            onset = get_contact_onset_window(task, ep, 58)
            contact_onsets.append(onset)
        n_contact = sum(1 for o in contact_onsets if o is not None)
        print(f"  Contact: {n_contact}/{n_eps}")

        for col, layer in enumerate(layers):
            ax = axes[row, col]
            print(f"  Layer {layer}...", end=" ", flush=True)

            features_list = load_layer_features_batch(task, ep_indices, layer)

            # Fit PCA on all episodes concatenated
            all_vecs = np.vstack(features_list)
            pca = PCA(n_components=2)
            pca.fit(all_vecs)
            var_explained = pca.explained_variance_ratio_.sum()

            # Adaptive styling based on episode count
            if n_eps > 200:
                lw, alpha, ms, star_s = 0.3, 0.15, 0, 60
                show_markers = False
            elif n_eps > 50:
                lw, alpha, ms, star_s = 0.5, 0.3, 10, 100
                show_markers = False
            else:
                lw, alpha, ms, star_s = 1.0, 0.6, 30, 180
                show_markers = True

            for ep_i, feats in enumerate(features_list):
                projected = pca.transform(feats)
                num_windows = len(projected)
                t_norm = np.linspace(0, 1, num_windows)
                colors = plt.cm.coolwarm(t_norm)

                # LineCollection
                points = projected.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=colors[:-1], linewidths=lw, alpha=alpha)
                ax.add_collection(lc)

                # Start/end markers (only for small N)
                if show_markers:
                    ax.scatter(
                        projected[0, 0], projected[0, 1],
                        marker="o", s=ms, c="royalblue", edgecolors="black",
                        linewidths=0.3, zorder=4, alpha=0.7,
                    )
                    ax.scatter(
                        projected[-1, 0], projected[-1, 1],
                        marker="s", s=ms, c="firebrick", edgecolors="black",
                        linewidths=0.3, zorder=4, alpha=0.7,
                    )

                # Contact onset marker
                onset = contact_onsets[ep_i]
                if onset is not None and onset < num_windows:
                    ax.scatter(
                        projected[onset, 0], projected[onset, 1],
                        marker="*", s=star_s, c="gold", edgecolors="black",
                        linewidths=0.3, zorder=5, alpha=min(alpha * 2.5, 0.8),
                    )

            ax.autoscale()
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                label = layer_labels.get(layer, f"Layer {layer}")
                ax.set_title(label, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(TASK_LABELS.get(task, task), fontsize=12, fontweight="bold")

            ax.text(
                0.02, 0.02, f"Var: {var_explained:.1%}",
                transform=ax.transAxes, fontsize=7, color="gray", verticalalignment="bottom",
            )

            # Free memory
            del features_list, all_vecs
            print(f"done ({var_explained:.1%})")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.4, pad=0.03, aspect=30)
    cbar.set_label("Time progression", fontsize=11)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Start", "End"])

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=11, label="Contact onset"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=1, fontsize=10,
        frameon=True, bbox_to_anchor=(0.45, -0.01),
    )

    n_label = "all" if num_episodes is None else str(num_episodes)
    fig.suptitle(
        f"V-JEPA 2 ViT-G Embedding Trajectories — PCA 2D ({n_label} episodes/task)",
        fontsize=14, fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.87, top=0.93, bottom=0.04, wspace=0.15, hspace=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 embedding trajectory visualization")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Number of episodes per task (default: all)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_label = "all" if args.num_episodes is None else str(args.num_episodes)
    if args.output is None:
        output_path = OUTPUT_DIR / f"embedding_trajectory_vitg_pca_{n_label}.png"
    else:
        output_path = Path(args.output)

    print(f"Tasks: {args.tasks}")
    print(f"Layers: {args.layers}")
    print(f"Episodes: {n_label}")
    print()

    plot_trajectories(args.tasks, args.layers, args.num_episodes, args.seed, output_path)


if __name__ == "__main__":
    main()
