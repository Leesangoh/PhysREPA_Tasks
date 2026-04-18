#!/usr/bin/env python3
"""
Layer-wise physics parameter correlation analysis for V-JEPA 2 ViT-G features.

For each layer, computes the Spearman correlation between post-contact trajectory
direction (PCA-projected) and ground-truth physics parameters (mass, friction).
Tests whether PEZ layers (12-17) uniquely encode multiple physics parameters.

Usage:
    /isaac-sim/python.sh analysis/analyze_physics_correlation.py
    /isaac-sim/python.sh analysis/analyze_physics_correlation.py --tasks push strike --num_episodes 100
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

FEATURE_ROOT = Path("/mnt/md1/solee/features/physprobe_vitg")
DATA_ROOT = Path("/home/solee/data/data/isaac_physrepa_v2/step0")
OUTPUT_DIR = Path("/home/solee/physrepa_tasks/results")

ALL_LAYERS = list(range(40))
NUM_EPISODES = 50
SEED = 42

# Task-specific physics params and velocity columns
TASK_CONFIG = {
    "push": {
        "vel_col": "physics_gt.object_velocity",
        "onset_method": "velocity_threshold",
        "physics_params": [
            ("object_0_mass", "Mass"),
            ("object_0_static_friction", "Obj. Friction"),
            ("surface_static_friction", "Surface Friction"),
        ],
    },
    "strike": {
        "vel_col": "physics_gt.object_velocity",
        "onset_method": "velocity_peak",
        "physics_params": [
            ("object_0_mass", "Mass"),
            ("object_0_static_friction", "Friction"),
            ("object_0_restitution", "Restitution"),
        ],
    },
    "peg_insert": {
        "vel_col": "physics_gt.peg_velocity",
        "onset_method": "velocity_jerk",
        "physics_params": [
            ("peg_mass", "Peg Mass"),
            ("peg_static_friction", "Peg Friction"),
            ("hole_static_friction", "Hole Friction"),
        ],
    },
}

TASK_COLORS = {
    "push": "#1f77b4",
    "strike": "#ff7f0e",
    "peg_insert": "#2ca02c",
}


def sample_episodes(task: str, n: int, seed: int) -> list:
    feature_dir = FEATURE_ROOT / task
    total = len(list(feature_dir.glob("*.safetensors")))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(total, size=min(n, total), replace=False).tolist())


def load_features(task: str, ep_idx: int, layer: int):
    path = FEATURE_ROOT / task / f"{ep_idx:06d}.safetensors"
    with safe_open(str(path), framework="pt") as f:
        ws = f.get_tensor("window_starts").numpy()
        vecs = [f.get_tensor(f"layer_{layer}_window_{w}").float().numpy() for w in range(len(ws))]
    return ws, np.stack(vecs)


def get_contact_onset_frame(task: str, ep_idx: int) -> int | None:
    config = TASK_CONFIG[task]
    chunk_idx = ep_idx // 1000
    parquet_path = (
        DATA_ROOT / task / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
    )
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path, columns=[config["vel_col"]])
    except Exception:
        return None

    vel = np.array([np.array(x) for x in df[config["vel_col"]].values])
    speed = np.linalg.norm(vel, axis=1)

    if config["onset_method"] == "velocity_threshold":
        moving = speed > 0.01
        if moving.any():
            return int(np.where(moving)[0][0])
    elif config["onset_method"] == "velocity_peak":
        speed_range = speed.max() - speed.min()
        if speed_range > 0.05 and len(speed) > 1:
            return int(np.argmax(speed))
    elif config["onset_method"] == "velocity_jerk":
        # Use speed peak as the interaction event
        if len(speed) > 2 and speed.max() > 0.01:
            return int(np.argmax(speed))
    return None


def load_task_metadata(task: str) -> list:
    meta_path = DATA_ROOT / task / "meta" / "episodes.jsonl"
    with open(meta_path) as f:
        return [json.loads(line) for line in f]


def analyze_task(task: str, ep_indices: list, layers: list):
    """Compute per-layer correlations between post-contact direction and physics params."""
    config = TASK_CONFIG[task]
    all_meta = load_task_metadata(task)

    # Collect physics params and contact onsets
    physics_values = {key: [] for key, _ in config["physics_params"]}
    onset_frames = []
    valid_episodes = []

    for ep in ep_indices:
        meta = all_meta[ep]
        onset = get_contact_onset_frame(task, ep)

        # Check all physics params exist
        all_present = True
        for key, _ in config["physics_params"]:
            if key not in meta:
                all_present = False
                break

        if onset is None or not all_present:
            continue

        valid_episodes.append(ep)
        onset_frames.append(onset)
        for key, _ in config["physics_params"]:
            physics_values[key].append(meta[key])

    for key in physics_values:
        physics_values[key] = np.array(physics_values[key])

    print(f"  Valid episodes with contact: {len(valid_episodes)}/{len(ep_indices)}")

    # Per-layer analysis
    param_names = [label for _, label in config["physics_params"]]
    param_keys = [key for key, _ in config["physics_params"]]
    n_params = len(param_keys)

    correlations = np.zeros((len(layers), n_params))  # Spearman rho
    p_values = np.ones((len(layers), n_params))
    cluster_ratios = np.zeros((len(layers), n_params))

    for li, layer in enumerate(layers):
        post_directions = []
        layer_valid = []

        for i, ep in enumerate(valid_episodes):
            ws, feats = load_features(task, ep, layer)
            onset_win = int(np.argmin(np.abs(ws - onset_frames[i])))

            if onset_win < 2 or onset_win >= len(feats) - 4:
                continue

            post_mean = feats[onset_win + 1 : min(len(feats), onset_win + 8)].mean(axis=0)
            pre_mean = feats[max(0, onset_win - 4) : onset_win].mean(axis=0)
            d = post_mean - pre_mean
            if np.linalg.norm(d) < 1e-8:
                continue

            post_directions.append(d)
            layer_valid.append(i)

        if len(layer_valid) < 15:
            continue

        valid_dirs = np.array(post_directions)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(valid_dirs)

        for pi, key in enumerate(param_keys):
            vals = physics_values[key][layer_valid]

            # Best correlation across PC1 and PC2
            r1, p1 = spearmanr(proj[:, 0], vals)
            r2, p2 = spearmanr(proj[:, 1], vals)
            if abs(r1) >= abs(r2):
                correlations[li, pi] = r1
                p_values[li, pi] = p1
            else:
                correlations[li, pi] = r2
                p_values[li, pi] = p2

            # Cluster ratio (tercile bins)
            bins = np.digitize(vals, np.quantile(vals, [1 / 3, 2 / 3]))
            intra, inter = [], []
            for bi in range(3):
                g = proj[bins == bi]
                if len(g) >= 2:
                    for a in range(len(g)):
                        for b in range(a + 1, len(g)):
                            intra.append(np.linalg.norm(g[a] - g[b]))
            for bi in range(3):
                for bj in range(bi + 1, 3):
                    g1, g2 = proj[bins == bi], proj[bins == bj]
                    if len(g1) > 0 and len(g2) > 0:
                        for a in g1:
                            for b in g2:
                                inter.append(np.linalg.norm(a - b))
            if intra and inter:
                cluster_ratios[li, pi] = np.mean(inter) / np.mean(intra)

    return {
        "correlations": correlations,
        "p_values": p_values,
        "cluster_ratios": cluster_ratios,
        "param_names": param_names,
    }


def plot_results(results: dict, tasks: list, layers: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Correlation sweep per task (individual) ──
    for task in tasks:
        r = results[task]
        fig, ax = plt.subplots(figsize=(13, 4.5))

        for pi, name in enumerate(r["param_names"]):
            abs_corr = np.abs(r["correlations"][:, pi])
            sig_mask = r["p_values"][:, pi] < 0.05
            ax.plot(layers, abs_corr, marker=".", markersize=4, linewidth=1.5, label=name)
            # Mark significant points
            sig_layers = [l for l, s in zip(layers, sig_mask) if s]
            sig_vals = [abs_corr[i] for i, s in enumerate(sig_mask) if s]
            ax.scatter(sig_layers, sig_vals, marker="o", s=20, zorder=5, edgecolors="black",
                       linewidths=0.3)

        ax.axhspan(0, 0.05, color="lightgray", alpha=0.3)
        ax.axvspan(12, 17, alpha=0.1, color="orange")
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("|Spearman ρ|", fontsize=12)
        ax.set_title(
            f"{task.replace('_', ' ').title()}: Post-Contact Direction ↔ Physics Parameter Correlation",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.set_xlim(-0.5, 39.5)
        ax.set_ylim(0, 0.7)
        ax.grid(True, alpha=0.3)

        # PEZ annotation
        ax.text(14.5, 0.65, "PEZ zone", ha="center", fontsize=9, color="darkorange", fontstyle="italic")

        fig.tight_layout()
        path = output_dir / f"physics_correlation_{task}.png"
        fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
        plt.close(fig)

    # ── Plot 2: Multi-task summary — number of significant params per layer ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top: |correlation| heatmap-style line plot
    ax = axes[0]
    for task in tasks:
        r = results[task]
        # Combined: max |correlation| across all params at each layer
        max_corr = np.max(np.abs(r["correlations"]), axis=1)
        ax.plot(layers, max_corr, marker=".", markersize=4, linewidth=1.5,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_ylabel("Max |Spearman ρ|", fontsize=11)
    ax.set_title("Strongest Physics Correlation per Layer", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 39.5)
    ax.grid(True, alpha=0.3)

    # Bottom: number of significant params (p<0.05)
    ax = axes[1]
    for task in tasks:
        r = results[task]
        n_sig = (r["p_values"] < 0.05).sum(axis=1).astype(float)
        ax.plot(layers, n_sig, marker=".", markersize=4, linewidth=1.5,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("# Significant Params\n(p < 0.05)", fontsize=11)
    ax.set_title("Number of Physics Parameters Encoded per Layer", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 39.5)
    ax.set_yticks(range(4))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "physics_correlation_summary.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)

    # ── Plot 3: Detailed heatmap ──
    for task in tasks:
        r = results[task]
        n_params = len(r["param_names"])

        fig, ax = plt.subplots(figsize=(14, max(2, 0.6 * n_params + 1)))

        # Signed correlation with significance overlay
        data = r["correlations"].T  # [n_params, n_layers]
        sig = r["p_values"].T < 0.05

        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                        extent=[-0.5, 39.5, n_params - 0.5, -0.5])

        # Mark significant cells
        for pi in range(n_params):
            for li in range(len(layers)):
                if sig[pi, li]:
                    ax.text(layers[li], pi, "●", ha="center", va="center",
                            fontsize=6, color="black")

        ax.set_yticks(range(n_params))
        ax.set_yticklabels(r["param_names"], fontsize=10)
        ax.set_xlabel("Layer Index", fontsize=11)
        ax.set_title(
            f"{task.replace('_', ' ').title()}: Spearman ρ (● = p < 0.05)",
            fontsize=12, fontweight="bold",
        )

        # PEZ zone
        ax.axvline(x=11.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=17.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Spearman ρ", fontsize=10)

        fig.tight_layout()
        path = output_dir / f"physics_correlation_heatmap_{task}.png"
        fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
        plt.close(fig)

    # ── Plot 4: Combined summary figure for paper ──
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.25)

    # Left column: correlation sweep per task
    for ti, task in enumerate(tasks):
        ax = fig.add_subplot(gs[ti, 0])
        r = results[task]
        for pi, name in enumerate(r["param_names"]):
            abs_corr = np.abs(r["correlations"][:, pi])
            ax.plot(layers, abs_corr, marker=".", markersize=3, linewidth=1.2, label=name)
        ax.axvspan(12, 17, alpha=0.1, color="orange")
        ax.set_xlim(-0.5, 39.5)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("|Spearman ρ|", fontsize=10)
        ax.set_title(f"({chr(97+ti)}) {task.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        if ti == 2:
            ax.set_xlabel("Layer Index", fontsize=10)

    # Right column: summary
    # Top-right: max correlation
    ax = fig.add_subplot(gs[0, 1])
    for task in tasks:
        r = results[task]
        max_corr = np.max(np.abs(r["correlations"]), axis=1)
        ax.plot(layers, max_corr, marker=".", markersize=3, linewidth=1.2,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_xlim(-0.5, 39.5)
    ax.set_ylabel("Max |ρ|", fontsize=10)
    ax.set_title("(d) Peak Correlation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mid-right: n_significant
    ax = fig.add_subplot(gs[1, 1])
    for task in tasks:
        r = results[task]
        n_sig = (r["p_values"] < 0.05).sum(axis=1).astype(float)
        ax.plot(layers, n_sig, marker=".", markersize=3, linewidth=1.2,
                label=task.replace("_", " ").title(), color=TASK_COLORS.get(task))
    ax.axvspan(12, 17, alpha=0.1, color="orange")
    ax.set_xlim(-0.5, 39.5)
    ax.set_yticks(range(4))
    ax.set_ylabel("# Sig. Params", fontsize=10)
    ax.set_title("(e) Multi-Param Encoding", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: heatmap for push
    ax = fig.add_subplot(gs[2, 1])
    r = results["push"]
    data = np.abs(r["correlations"].T)
    sig = r["p_values"].T < 0.05
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6,
                    extent=[-0.5, 39.5, len(r["param_names"]) - 0.5, -0.5])
    for pi in range(len(r["param_names"])):
        for li in range(len(layers)):
            if sig[pi, li]:
                ax.text(layers[li], pi, "●", ha="center", va="center", fontsize=5, color="black")
    ax.set_yticks(range(len(r["param_names"])))
    ax.set_yticklabels(r["param_names"], fontsize=8)
    ax.set_xlabel("Layer Index", fontsize=10)
    ax.set_title("(f) Push |ρ| Heatmap", fontsize=11, fontweight="bold")
    ax.axvline(x=11.5, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(x=17.5, color="orange", linestyle="--", alpha=0.5)

    fig.suptitle(
        "V-JEPA 2 ViT-G: Post-Contact Direction ↔ Physics Parameter Correlation",
        fontsize=14, fontweight="bold", y=0.99,
    )

    path = output_dir / "physics_correlation_paper.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


def print_table(results: dict, tasks: list, layers: list):
    """Print a summary table to stdout."""
    print("\n" + "=" * 80)
    print("SUMMARY: Post-contact trajectory direction ↔ physics parameter correlation")
    print("=" * 80)

    for task in tasks:
        r = results[task]
        print(f"\n--- {task.upper()} ---")
        header = "Layer  " + "  ".join(f"{name:>14s}" for name in r["param_names"])
        print(header)
        print("-" * len(header))

        for li, layer in enumerate(layers):
            parts = [f"  {layer:2d}  "]
            for pi in range(len(r["param_names"])):
                rho = r["correlations"][li, pi]
                p = r["p_values"][li, pi]
                sig = "**" if p < 0.01 else ("* " if p < 0.05 else "  ")
                parts.append(f"  {rho:+.3f}{sig}     ")
            print("".join(parts))

        # Find peak layer for each param
        print()
        for pi, name in enumerate(r["param_names"]):
            abs_corr = np.abs(r["correlations"][:, pi])
            peak_li = np.argmax(abs_corr)
            peak_layer = layers[peak_li]
            peak_r = r["correlations"][peak_li, pi]
            peak_p = r["p_values"][peak_li, pi]
            print(f"  Peak {name}: layer {peak_layer} (ρ={peak_r:+.3f}, p={peak_p:.4f})")

        # PEZ zone summary
        pez_mask = np.array([(12 <= l <= 17) for l in layers])
        n_sig_pez = (r["p_values"][pez_mask] < 0.05).sum()
        n_total_pez = pez_mask.sum() * len(r["param_names"])
        print(f"  PEZ zone (12-17): {n_sig_pez}/{n_total_pez} significant (p<0.05)")


def main():
    parser = argparse.ArgumentParser(description="Physics parameter correlation analysis")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIG.keys()))
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    layers = ALL_LAYERS

    print(f"Tasks: {args.tasks}")
    print(f"Layers: 0-39 ({len(layers)} layers)")
    print(f"Episodes per task: {args.num_episodes}, Seed: {args.seed}")
    print()

    results = {}
    for task in args.tasks:
        if task not in TASK_CONFIG:
            print(f"WARNING: No config for task '{task}', skipping")
            continue
        ep_indices = sample_episodes(task, args.num_episodes, args.seed)
        print(f"[{task}] Analyzing {len(ep_indices)} episodes...")
        results[task] = analyze_task(task, ep_indices, layers)

    valid_tasks = [t for t in args.tasks if t in results]

    print("\nGenerating plots...")
    plot_results(results, valid_tasks, layers, Path(args.output_dir))

    print_table(results, valid_tasks, layers)
    print("\nDone!")


if __name__ == "__main__":
    main()
