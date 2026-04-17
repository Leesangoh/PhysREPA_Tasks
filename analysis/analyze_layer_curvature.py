#!/usr/bin/env python3
"""
Layer-wise trajectory curvature analysis for V-JEPA 2 ViT-G features.
Sweeps all 40 layers and quantifies:
  1. PCA variance explained (2D) per layer
  2. Trajectory curvature (angular change between consecutive segments)
  3. Curvature ratio: post-contact / pre-contact curvature

Usage:
    /isaac-sim/python.sh analysis/analyze_layer_curvature.py
    /isaac-sim/python.sh analysis/analyze_layer_curvature.py --tasks push strike peg_insert reach
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.decomposition import PCA

FEATURE_ROOT = Path("/mnt/md1/solee/features/physprobe_vitg")
DATA_ROOT = Path("/home/solee/data/data/isaac_physrepa_v2/step0")
OUTPUT_DIR = Path("/home/solee/physrepa_tasks/results")

DEFAULT_TASKS = ["push", "strike", "peg_insert", "reach"]
ALL_LAYERS = list(range(40))
NUM_EPISODES = 20
SEED = 42

VELOCITY_COLUMNS = {
    "push": "physics_gt.object_velocity",
    "strike": "physics_gt.object_velocity",
    "peg_insert": "physics_gt.peg_velocity",
    "nut_thread": "physics_gt.peg_velocity",
    "drawer": "physics_gt.ee_velocity",
    "reach": "physics_gt.ee_velocity",
}

TASK_COLORS = {
    "push": "#1f77b4",
    "strike": "#ff7f0e",
    "peg_insert": "#2ca02c",
    "nut_thread": "#d62728",
    "drawer": "#9467bd",
    "reach": "#7f7f7f",
}


def sample_episodes(task: str, n: int, seed: int) -> list:
    feature_dir = FEATURE_ROOT / task
    total = len(list(feature_dir.glob("*.safetensors")))
    rng = np.random.RandomState(seed)
    indices = rng.choice(total, size=min(n, total), replace=False)
    return sorted(indices.tolist())


def load_single_layer_features(task: str, ep_idx: int, layer: int):
    """Load features for a single layer from one episode.

    Returns (window_starts, features) where features is [num_windows, 1408] float32.
    """
    path = FEATURE_ROOT / task / f"{ep_idx:06d}.safetensors"
    with safe_open(str(path), framework="pt") as f:
        window_starts = f.get_tensor("window_starts").numpy()
        num_windows = len(window_starts)
        vecs = []
        for w in range(num_windows):
            vecs.append(f.get_tensor(f"layer_{layer}_window_{w}").float().numpy())
    return window_starts, np.stack(vecs)


def get_contact_onset_frame(task: str, ep_idx: int) -> int | None:
    """Get contact onset as raw frame index."""
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

    if task == "push":
        moving = speed > 0.01
        if moving.any():
            return int(np.where(moving)[0][0])
    elif task == "strike":
        speed_range = speed.max() - speed.min()
        if speed_range > 0.05 and len(speed) > 1:
            return int(np.argmax(speed))
    elif task == "peg_insert":
        if len(speed) > 2:
            accel = np.abs(np.diff(speed))
            median_accel = np.median(accel)
            if median_accel > 0:
                spikes = accel > max(median_accel * 5, 0.005)
                if spikes.any():
                    return int(np.where(spikes)[0][0])
    else:
        threshold = np.mean(speed) + 2 * np.std(speed)
        above = speed > threshold
        if above.any():
            return int(np.where(above)[0][0])

    return None


def frame_to_window(onset_frame: int, window_starts: np.ndarray) -> int:
    return int(np.argmin(np.abs(window_starts - onset_frame)))


def compute_curvature(trajectory: np.ndarray) -> np.ndarray:
    """Compute per-step angular curvature of a trajectory in high-dim space.

    curvature[i] = angle between segment (i-1,i) and (i,i+1) in radians.
    Returns array of length (N-2).
    """
    if len(trajectory) < 3:
        return np.array([])

    # Direction vectors
    d = np.diff(trajectory, axis=0)  # [N-1, D]
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    d_unit = d / norms

    # Cosine of angle between consecutive segments
    cos_angles = np.sum(d_unit[:-1] * d_unit[1:], axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)  # [N-2]

    return angles


def analyze_task(task: str, ep_indices: list, layers: list):
    """Analyze all layers for a task.

    Returns dict with:
        'pca_var': [num_layers] PCA 2D variance explained
        'mean_curvature': [num_layers] mean curvature across episodes
        'curvature_at_contact': [num_layers] mean curvature within ±2 windows of contact
        'curvature_away': [num_layers] mean curvature away from contact (>5 windows)
        'curvature_ratio': [num_layers] at_contact / away ratio
        'curvature_pre': [num_layers] mean curvature before contact
        'curvature_post': [num_layers] mean curvature after contact
    """
    # First, get contact onset for each episode
    contact_frames = []
    for ep_idx in ep_indices:
        contact_frames.append(get_contact_onset_frame(task, ep_idx))

    pca_var = np.zeros(len(layers))
    mean_curvature = np.zeros(len(layers))
    curvature_at_contact = np.zeros(len(layers))
    curvature_away = np.zeros(len(layers))
    curvature_pre = np.zeros(len(layers))
    curvature_post = np.zeros(len(layers))

    for li, layer in enumerate(layers):
        all_curvatures = []
        at_contact_curvatures = []
        away_curvatures = []
        pre_curvatures = []
        post_curvatures = []
        all_features = []

        for ei, ep_idx in enumerate(ep_indices):
            window_starts, features = load_single_layer_features(task, ep_idx, layer)
            all_features.append(features)

            curv = compute_curvature(features)  # [N-2]
            if len(curv) == 0:
                continue

            all_curvatures.extend(curv.tolist())

            onset_frame = contact_frames[ei]
            if onset_frame is not None:
                onset_win = frame_to_window(onset_frame, window_starts)
                num_windows = len(window_starts)

                for ci, c_val in enumerate(curv):
                    win_idx = ci + 1  # curvature[i] corresponds to window i+1
                    dist_to_onset = abs(win_idx - onset_win)

                    if dist_to_onset <= 3:
                        at_contact_curvatures.append(c_val)
                    elif dist_to_onset > 6:
                        away_curvatures.append(c_val)

                    if win_idx < onset_win:
                        pre_curvatures.append(c_val)
                    elif win_idx > onset_win:
                        post_curvatures.append(c_val)

        # PCA variance explained
        all_vecs = np.vstack(all_features)
        pca = PCA(n_components=2)
        pca.fit(all_vecs)
        pca_var[li] = pca.explained_variance_ratio_.sum()

        mean_curvature[li] = np.mean(all_curvatures) if all_curvatures else 0
        curvature_at_contact[li] = np.mean(at_contact_curvatures) if at_contact_curvatures else 0
        curvature_away[li] = np.mean(away_curvatures) if away_curvatures else 0
        curvature_pre[li] = np.mean(pre_curvatures) if pre_curvatures else 0
        curvature_post[li] = np.mean(post_curvatures) if post_curvatures else 0

    # Curvature ratio (at_contact / away), avoid division by zero
    curvature_ratio = np.where(
        curvature_away > 1e-6,
        curvature_at_contact / curvature_away,
        0.0,
    )

    return {
        "pca_var": pca_var,
        "mean_curvature": mean_curvature,
        "curvature_at_contact": curvature_at_contact,
        "curvature_away": curvature_away,
        "curvature_ratio": curvature_ratio,
        "curvature_pre": curvature_pre,
        "curvature_post": curvature_post,
    }


def plot_results(results: dict, tasks: list, layers: list, output_dir: Path):
    """Generate all analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: PCA Variance Explained Sweep ──
    fig, ax = plt.subplots(figsize=(12, 4))
    for task in tasks:
        ax.plot(layers, results[task]["pca_var"], marker=".", markersize=4,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task), linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("PCA 2D Variance Explained", fontsize=12)
    ax.set_title("V-JEPA 2 ViT-G: PCA Variance Explained by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 39)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange", label="PEZ zone")
    fig.tight_layout()
    path = output_dir / "pca_variance_sweep.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 2: Mean Curvature Sweep ──
    fig, ax = plt.subplots(figsize=(12, 4))
    for task in tasks:
        ax.plot(layers, np.degrees(results[task]["mean_curvature"]),
                marker=".", markersize=4, label=task.replace("_", " ").title(),
                color=TASK_COLORS.get(task), linewidth=1.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Mean Curvature (degrees)", fontsize=12)
    ax.set_title("Trajectory Curvature by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    fig.tight_layout()
    path = output_dir / "curvature_sweep.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 3: Curvature at Contact vs Away (bar chart) ──
    contact_tasks = [t for t in tasks if t != "reach"]
    fig, axes = plt.subplots(1, len(contact_tasks), figsize=(6 * len(contact_tasks), 5), sharey=True)
    if len(contact_tasks) == 1:
        axes = [axes]

    for ai, task in enumerate(contact_tasks):
        ax = axes[ai]
        at_c = np.degrees(results[task]["curvature_at_contact"])
        away = np.degrees(results[task]["curvature_away"])
        x = np.array(layers)
        width = 0.35

        # Downsample for readability — show every 2nd layer
        show_idx = list(range(0, len(layers), 2))
        x_show = [layers[i] for i in show_idx]
        at_c_show = [at_c[i] for i in show_idx]
        away_show = [away[i] for i in show_idx]

        x_pos = np.arange(len(x_show))
        ax.bar(x_pos - width / 2, at_c_show, width, label="Near contact (±3 win)",
               color=TASK_COLORS.get(task), alpha=0.8)
        ax.bar(x_pos + width / 2, away_show, width, label="Away (>6 win)",
               color=TASK_COLORS.get(task), alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_show, fontsize=8)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight="bold")
        if ai == 0:
            ax.set_ylabel("Curvature (degrees)", fontsize=11)
        ax.legend(fontsize=9)

        # Highlight PEZ zone
        pez_start = next((i for i, l in enumerate(x_show) if l >= 12), None)
        pez_end = next((i for i, l in enumerate(x_show) if l > 17), None)
        if pez_start is not None and pez_end is not None:
            ax.axvspan(pez_start - 0.5, pez_end - 0.5, alpha=0.08, color="orange")

    fig.suptitle("Curvature Near vs. Away from Contact Onset", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "curvature_contact_vs_away.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 4: Curvature Ratio (at_contact / away) ──
    fig, ax = plt.subplots(figsize=(12, 4))
    for task in contact_tasks:
        ratio = results[task]["curvature_ratio"]
        ax.plot(layers, ratio, marker=".", markersize=4,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task), linewidth=1.5)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Curvature Ratio\n(at contact / away)", fontsize=12)
    ax.set_title("Contact Curvature Amplification by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    fig.tight_layout()
    path = output_dir / "curvature_ratio_sweep.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 5: Pre vs Post Contact Curvature ──
    fig, ax = plt.subplots(figsize=(12, 4))
    for task in contact_tasks:
        pre = np.degrees(results[task]["curvature_pre"])
        post = np.degrees(results[task]["curvature_post"])
        diff = post - pre
        ax.plot(layers, diff, marker=".", markersize=4,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task), linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Post − Pre Curvature (degrees)", fontsize=12)
    ax.set_title("Curvature Change After Contact by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    fig.tight_layout()
    path = output_dir / "curvature_pre_vs_post.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 6: Combined summary (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) PCA Variance
    ax = axes[0, 0]
    for task in tasks:
        ax.plot(layers, results[task]["pca_var"], marker=".", markersize=3,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task), linewidth=1.2)
    ax.set_ylabel("PCA 2D Var. Explained", fontsize=11)
    ax.set_title("(a) PCA Variance Explained", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")

    # (0,1) Mean Curvature
    ax = axes[0, 1]
    for task in tasks:
        ax.plot(layers, np.degrees(results[task]["mean_curvature"]),
                marker=".", markersize=3, label=task.replace("_", " ").title(),
                color=TASK_COLORS.get(task), linewidth=1.2)
    ax.set_ylabel("Mean Curvature (°)", fontsize=11)
    ax.set_title("(b) Overall Trajectory Curvature", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")

    # (1,0) Curvature Ratio
    ax = axes[1, 0]
    for task in contact_tasks:
        ax.plot(layers, results[task]["curvature_ratio"],
                marker=".", markersize=3, label=task.replace("_", " ").title(),
                color=TASK_COLORS.get(task), linewidth=1.2)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Curvature Ratio\n(contact / away)", fontsize=11)
    ax.set_title("(c) Contact Curvature Amplification", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")

    # (1,1) Pre vs Post
    ax = axes[1, 1]
    for task in contact_tasks:
        pre = np.degrees(results[task]["curvature_pre"])
        post = np.degrees(results[task]["curvature_post"])
        ax.plot(layers, post - pre, marker=".", markersize=3,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task), linewidth=1.2)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Post − Pre Curvature (°)", fontsize=11)
    ax.set_title("(d) Curvature Change After Contact", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 39)
    ax.grid(True, alpha=0.3)
    ax.axvspan(12, 17, alpha=0.1, color="orange")

    fig.suptitle(
        "V-JEPA 2 ViT-G: Layer-wise Trajectory Analysis (40 layers, 20 episodes/task)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "layer_analysis_summary.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Layer-wise curvature analysis")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    layers = ALL_LAYERS

    print(f"Tasks: {args.tasks}")
    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Episodes per task: {args.num_episodes}, Seed: {args.seed}")
    print()

    results = {}
    for task in args.tasks:
        ep_indices = sample_episodes(task, args.num_episodes, args.seed)
        print(f"[{task}] Analyzing {len(ep_indices)} episodes across {len(layers)} layers...")
        results[task] = analyze_task(task, ep_indices, layers)
        print(f"  Done. PCA var range: [{results[task]['pca_var'].min():.1%}, {results[task]['pca_var'].max():.1%}]")
        print(f"  Mean curvature range: [{np.degrees(results[task]['mean_curvature'].min()):.1f}°, {np.degrees(results[task]['mean_curvature'].max()):.1f}°]")
        if task != "reach":
            peak_layer = layers[np.argmax(results[task]["curvature_ratio"])]
            peak_ratio = results[task]["curvature_ratio"].max()
            print(f"  Peak curvature ratio: layer {peak_layer} ({peak_ratio:.2f}x)")
        print()

    print("Generating plots...")
    plot_results(results, args.tasks, layers, Path(args.output_dir))
    print("\nDone!")


if __name__ == "__main__":
    main()
