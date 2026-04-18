#!/usr/bin/env python3
"""
Systematic trajectory dispersion analysis for V-JEPA 2 ViT-G features.

For each layer, measures how trajectory spread (inter-episode variance) changes
over time. PEZ layers should show a V-shape: convergence before contact, then
divergence after — indicating physics-dependent trajectory splitting.

Metrics:
  1. Temporal dispersion profile: variance of PCA 2D positions at each window
  2. Dispersion ratio: post-contact / pre-contact variance
  3. Convergence rate: slope of variance decrease before contact
  4. Divergence rate: slope of variance increase after contact

Usage:
    /isaac-sim/python.sh analysis/analyze_trajectory_dispersion.py
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

DEFAULT_TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]
ALL_LAYERS = list(range(40))
SEED = 42

VELOCITY_COLUMNS = {
    "push": "physics_gt.object_velocity",
    "strike": "physics_gt.object_velocity",
    "peg_insert": "physics_gt.peg_velocity",
    "nut_thread": "physics_gt.nut_velocity",
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


def get_episode_indices(task: str, num_episodes: int | None, seed: int) -> list:
    feature_dir = FEATURE_ROOT / task
    total = len(list(feature_dir.glob("*.safetensors")))
    if num_episodes is None or num_episodes >= total:
        return list(range(total))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(total, size=num_episodes, replace=False).tolist())


def get_contact_onset_frame(task: str, ep_idx: int) -> int | None:
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
        if speed.max() - speed.min() > 0.05 and len(speed) > 1:
            return int(np.argmax(speed))
    elif task in ("peg_insert", "nut_thread"):
        if len(speed) > 2 and speed.max() > 0.01:
            return int(np.argmax(speed))
    elif task == "drawer":
        if len(speed) > 2 and speed.max() > 0.5:
            return int(np.argmax(speed))
    return None


def compute_dispersion_profile(task, ep_indices, layer, onset_windows):
    """Compute PCA 2D dispersion at each time window.

    Returns:
        positions: [n_eps, n_windows, 2] PCA-projected positions
        dispersion: [n_windows] variance of positions across episodes
        var_explained: float
    """
    # Load all features for this layer
    all_features = []
    window_counts = []
    for ep_idx in ep_indices:
        path = FEATURE_ROOT / task / f"{ep_idx:06d}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            ws = f.get_tensor("window_starts").numpy()
            vecs = [f.get_tensor(f"layer_{layer}_window_{w}").float().numpy() for w in range(len(ws))]
        all_features.append(np.stack(vecs))
        window_counts.append(len(ws))

    # Use the minimum window count to align all episodes
    min_windows = min(window_counts)

    # Truncate to min_windows and stack
    truncated = np.array([f[:min_windows] for f in all_features])  # [n_eps, min_windows, 1408]
    n_eps = truncated.shape[0]

    # PCA on all windows from all episodes
    flat = truncated.reshape(-1, truncated.shape[-1])
    pca = PCA(n_components=2)
    projected_flat = pca.fit_transform(flat)
    projected = projected_flat.reshape(n_eps, min_windows, 2)

    # Dispersion: mean pairwise distance at each window (or just variance)
    dispersion = np.zeros(min_windows)
    for w in range(min_windows):
        points = projected[:, w, :]  # [n_eps, 2]
        # Use trace of covariance as dispersion metric
        dispersion[w] = np.trace(np.cov(points.T)) if n_eps > 1 else 0

    return projected, dispersion, pca.explained_variance_ratio_.sum(), min_windows


def analyze_all(tasks, layers, num_episodes, seed):
    results = {}

    for task in tasks:
        ep_indices = get_episode_indices(task, num_episodes, seed)
        print(f"[{task}] {len(ep_indices)} episodes")

        # Get contact onsets
        onset_frames = []
        for ep in ep_indices:
            onset_frames.append(get_contact_onset_frame(task, ep))

        # Median onset window for alignment
        onset_windows = []
        for of in onset_frames:
            if of is not None:
                onset_windows.append(of // 4)  # frame to window
        median_onset = int(np.median(onset_windows)) if onset_windows else None
        print(f"  Median contact onset: window {median_onset}")

        task_results = {
            "ep_indices": ep_indices,
            "median_onset": median_onset,
            "layers": {},
        }

        for layer in layers:
            print(f"  Layer {layer}...", end=" ", flush=True)
            _, dispersion, var_exp, n_windows = compute_dispersion_profile(
                task, ep_indices, layer, onset_windows
            )
            task_results["layers"][layer] = {
                "dispersion": dispersion,
                "var_explained": var_exp,
                "n_windows": n_windows,
            }
            print(f"done (Var={var_exp:.1%})")

        results[task] = task_results

    return results


def plot_results(results, tasks, layers, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Dispersion profiles for selected layers (per task) ──
    selected_layers = [3, 8, 13, 16, 20, 30, 39]
    selected_layers = [l for l in selected_layers if l in layers]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(selected_layers)))

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ti, task in enumerate(tasks):
        ax = axes[ti]
        r = results[task]
        onset = r["median_onset"]

        for li, layer in enumerate(selected_layers):
            if layer not in r["layers"]:
                continue
            d = r["layers"][layer]
            disp = d["dispersion"]
            n_w = d["n_windows"]
            x = np.arange(n_w)
            ax.plot(x, disp, color=cmap[li], linewidth=1.2, label=f"L{layer}", alpha=0.8)

        if onset is not None:
            ax.axvline(x=onset, color="red", linestyle="--", alpha=0.6, linewidth=1, label="Contact")

        ax.set_xlabel("Window (time)", fontsize=11)
        if ti == 0:
            ax.set_ylabel("Dispersion (trace of cov)", fontsize=11)
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Temporal Dispersion Profile by Layer", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "dispersion_temporal_profile.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 2: Dispersion ratio (post/pre contact) sweep across ALL layers ──
    contact_tasks = [t for t in tasks if results[t]["median_onset"] is not None]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Dispersion ratio
    ax = axes[0]
    for task in contact_tasks:
        r = results[task]
        onset = r["median_onset"]
        ratios = []
        for layer in layers:
            if layer not in r["layers"]:
                ratios.append(np.nan)
                continue
            disp = r["layers"][layer]["dispersion"]
            pre_window = max(0, onset - 5)
            post_window = min(len(disp), onset + 10)
            pre_mean = disp[pre_window:onset].mean() if onset > pre_window else 1e-10
            post_mean = disp[onset:post_window].mean() if post_window > onset else 0
            ratios.append(post_mean / max(pre_mean, 1e-10))
        ax.plot(layers, ratios, marker=".", markersize=4, linewidth=1.5,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Dispersion Ratio\n(post / pre contact)", fontsize=11)
    ax.set_title("(a) Post/Pre Contact Dispersion", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 39.5)
    ax.grid(True, alpha=0.3)

    # (b) Divergence rate (slope of dispersion increase after contact)
    ax = axes[1]
    for task in contact_tasks:
        r = results[task]
        onset = r["median_onset"]
        div_rates = []
        for layer in layers:
            if layer not in r["layers"]:
                div_rates.append(np.nan)
                continue
            disp = r["layers"][layer]["dispersion"]
            post_end = min(len(disp), onset + 15)
            if post_end - onset < 3:
                div_rates.append(0)
                continue
            post_disp = disp[onset:post_end]
            x = np.arange(len(post_disp))
            slope = np.polyfit(x, post_disp, 1)[0]
            div_rates.append(slope)
        ax.plot(layers, div_rates, marker=".", markersize=4, linewidth=1.5,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Divergence Rate\n(slope post-contact)", fontsize=11)
    ax.set_title("(b) Post-Contact Divergence Rate", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 39.5)
    ax.grid(True, alpha=0.3)

    # (c) Convergence-divergence asymmetry
    ax = axes[2]
    for task in contact_tasks:
        r = results[task]
        onset = r["median_onset"]
        asymmetries = []
        for layer in layers:
            if layer not in r["layers"]:
                asymmetries.append(np.nan)
                continue
            disp = r["layers"][layer]["dispersion"]
            pre_start = max(0, onset - 15)
            post_end = min(len(disp), onset + 15)

            # Pre-contact slope (should be negative = converging)
            if onset - pre_start >= 3:
                pre_disp = disp[pre_start:onset]
                pre_slope = np.polyfit(np.arange(len(pre_disp)), pre_disp, 1)[0]
            else:
                pre_slope = 0

            # Post-contact slope (should be positive = diverging)
            if post_end - onset >= 3:
                post_disp = disp[onset:post_end]
                post_slope = np.polyfit(np.arange(len(post_disp)), post_disp, 1)[0]
            else:
                post_slope = 0

            # Asymmetry: post_slope - pre_slope (large positive = V-shape)
            asymmetries.append(post_slope - pre_slope)

        ax.plot(layers, asymmetries, marker=".", markersize=4, linewidth=1.5,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("V-shape Score\n(div_slope − conv_slope)", fontsize=11)
    ax.set_title("(c) Convergence-Divergence Asymmetry", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 39.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "V-JEPA 2 ViT-G: Trajectory Dispersion Analysis (40 layers)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "dispersion_layer_sweep.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 3: Detailed temporal profiles for PEZ vs non-PEZ (4 layers, per task) ──
    focus_layers = [3, 13, 16, 39]
    focus_layers = [l for l in focus_layers if l in layers]
    layer_colors = {3: "#1f77b4", 13: "#ff7f0e", 16: "#2ca02c", 39: "#d62728"}
    layer_styles = {3: "-", 13: "-", 16: "--", 39: ":"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    task_axes = {t: axes[i // 2, i % 2] for i, t in enumerate(tasks[:4])}

    for task in tasks[:4]:
        ax = task_axes[task]
        r = results[task]
        onset = r["median_onset"]

        for layer in focus_layers:
            if layer not in r["layers"]:
                continue
            d = r["layers"][layer]
            disp = d["dispersion"]
            # Normalize to [0,1] for comparison
            disp_norm = disp / (disp.max() + 1e-10)
            x = np.arange(len(disp_norm))
            label_map = {3: "L3 (Early)", 13: "L13 (PEZ)", 16: "L16 (PEZ)", 39: "L39 (Final)"}
            ax.plot(x, disp_norm, color=layer_colors[layer], linestyle=layer_styles[layer],
                    linewidth=1.8, label=label_map.get(layer, f"L{layer}"), alpha=0.9)

        if onset is not None:
            ax.axvline(x=onset, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
            ax.text(onset + 1, 0.95, "contact", fontsize=8, color="red", alpha=0.7)

        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Window (time)", fontsize=10)
        ax.set_ylabel("Normalized Dispersion", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Trajectory Dispersion Over Time: PEZ vs Non-PEZ Layers",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "dispersion_pez_comparison.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Trajectory dispersion analysis")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Episodes per task (default: all)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    layers = ALL_LAYERS
    print(f"Tasks: {args.tasks}")
    n_label = "all" if args.num_episodes is None else str(args.num_episodes)
    print(f"Episodes: {n_label}")
    print(f"Layers: 0-39 ({len(layers)} layers)")
    print()

    results = analyze_all(args.tasks, layers, args.num_episodes, args.seed)

    print("\nGenerating plots...")
    plot_results(results, args.tasks, layers, Path(args.output_dir))
    print("\nDone!")


if __name__ == "__main__":
    main()
