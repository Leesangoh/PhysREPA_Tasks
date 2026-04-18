"""
Linear probing sweep: V-JEPA 2 frozen features → physics GT prediction.

For every (task, layer, variable) triple, trains a Ridge regression with
GroupKFold (episode-level splits) and reports R² mean ± std.

Usage:
    /isaac-sim/python.sh analysis/probe_sweep.py \
        --model_size giant --output_dir results/
"""

import argparse
import gc
import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "giant": {"num_layers": 40, "dim": 1408, "tag": "vitg"},
    "large": {"num_layers": 24, "dim": 1024, "tag": "vitl"},
}

FEATURE_BASE = "/mnt/md1/solee/features"
DATA_BASE = "/mnt/md1/solee/data/isaac_physrepa_v2/step0"

TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]

WINDOW_SIZE = 16  # 16-frame clips

# Episode counts per task
TASK_EPISODE_COUNTS = {
    "push": 1500,
    "strike": 3000,
    "peg_insert": 2500,
    "nut_thread": 2500,
    "drawer": 2000,
    "reach": 600,
}

# ── Variable definitions per task ────────────────────────────────────────────

TASK_VARIABLES = {
    "push": {
        "static": [
            "object_0_mass",
            "object_0_static_friction",
            "object_0_dynamic_friction",
            "surface_static_friction",
            "surface_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_angular_velocity": ("physics_gt.ee_angular_velocity", "vector"),
            "object_velocity": ("physics_gt.object_velocity", "vector"),
            "object_acceleration": ("physics_gt.object_acceleration", "vector"),
            "object_angular_velocity": ("physics_gt.object_angular_velocity", "vector"),
            "ee_to_object_distance": ("physics_gt.ee_to_object_distance", "scalar"),
            "object_to_target_distance": ("physics_gt.object_to_target_distance", "scalar"),
            "object_on_surface": ("physics_gt.object_on_surface", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "contact_finger_l_object_flag": ("physics_gt.contact_finger_l_object_flag", "scalar"),
            "contact_finger_l_object_force": ("physics_gt.contact_finger_l_object_force", "vector"),
            "contact_object_surface_flag": ("physics_gt.contact_object_surface_flag", "scalar"),
            "contact_object_surface_force": ("physics_gt.contact_object_surface_force", "vector"),
            "contact_point": ("physics_gt.contact_point", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
            "target_position": ("physics_gt.target_position", "vector"),
        },
    },
    "strike": {
        "static": [
            "object_0_mass",
            "object_0_static_friction",
            "object_0_dynamic_friction",
            "object_0_restitution",
            "surface_static_friction",
            "surface_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_angular_velocity": ("physics_gt.ee_angular_velocity", "vector"),
            "object_velocity": ("physics_gt.object_velocity", "vector"),
            "object_acceleration": ("physics_gt.object_acceleration", "vector"),
            "object_angular_velocity": ("physics_gt.object_angular_velocity", "vector"),
            "ee_to_object_distance": ("physics_gt.ee_to_object_distance", "scalar"),
            "object_to_target_distance": ("physics_gt.object_to_target_distance", "scalar"),
            "object_on_surface": ("physics_gt.object_on_surface", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "contact_finger_l_object_flag": ("physics_gt.contact_finger_l_object_flag", "scalar"),
            "contact_finger_l_object_force": ("physics_gt.contact_finger_l_object_force", "vector"),
            "contact_object_surface_flag": ("physics_gt.contact_object_surface_flag", "scalar"),
            "contact_object_surface_force": ("physics_gt.contact_object_surface_force", "vector"),
            "contact_point": ("physics_gt.contact_point", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
            "target_position": ("physics_gt.target_position", "vector"),
            "ball_planar_travel_distance": ("physics_gt.ball_planar_travel_distance", "scalar"),
        },
    },
    "peg_insert": {
        "static": [
            "peg_mass",
            "peg_static_friction",
            "peg_dynamic_friction",
            "hole_static_friction",
            "hole_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_angular_velocity": ("physics_gt.ee_angular_velocity", "vector"),
            "peg_velocity": ("physics_gt.peg_velocity", "vector"),
            "peg_angular_velocity": ("physics_gt.peg_angular_velocity", "vector"),
            "insertion_depth": ("physics_gt.insertion_depth", "scalar"),
            "peg_hole_lateral_error": ("physics_gt.peg_hole_lateral_error", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "contact_finger_l_peg_flag": ("physics_gt.contact_finger_l_peg_flag", "scalar"),
            "contact_finger_l_peg_force": ("physics_gt.contact_finger_l_peg_force", "vector"),
            "contact_finger_r_peg_flag": ("physics_gt.contact_finger_r_peg_flag", "scalar"),
            "contact_finger_r_peg_force": ("physics_gt.contact_finger_r_peg_force", "vector"),
            "contact_peg_socket_flag": ("physics_gt.contact_peg_socket_flag", "scalar"),
            "contact_peg_socket_force": ("physics_gt.contact_peg_socket_force", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
    "nut_thread": {
        "static": [
            "nut_mass",
            "nut_static_friction",
            "nut_dynamic_friction",
            "bolt_static_friction",
            "bolt_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_angular_velocity": ("physics_gt.ee_angular_velocity", "vector"),
            "nut_velocity": ("physics_gt.nut_velocity", "vector"),
            "nut_angular_velocity": ("physics_gt.nut_angular_velocity", "vector"),
            "axial_progress": ("physics_gt.axial_progress", "scalar"),
            "nut_bolt_relative_angle": ("physics_gt.nut_bolt_relative_angle", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "contact_finger_l_nut_flag": ("physics_gt.contact_finger_l_nut_flag", "scalar"),
            "contact_finger_l_nut_force": ("physics_gt.contact_finger_l_nut_force", "vector"),
            "contact_finger_r_nut_flag": ("physics_gt.contact_finger_r_nut_flag", "scalar"),
            "contact_finger_r_nut_force": ("physics_gt.contact_finger_r_nut_force", "vector"),
            "contact_nut_bolt_flag": ("physics_gt.contact_nut_bolt_flag", "scalar"),
            "contact_nut_bolt_force": ("physics_gt.contact_nut_bolt_force", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
    "drawer": {
        "static": [
            "drawer_handle_mass",
            "drawer_joint_damping",
            "handle_static_friction",
            "handle_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_angular_velocity": ("physics_gt.ee_angular_velocity", "vector"),
            "handle_velocity": ("physics_gt.handle_velocity", "vector"),
            "drawer_joint_pos": ("physics_gt.drawer_joint_pos", "scalar"),
            "drawer_joint_vel": ("physics_gt.drawer_joint_vel", "scalar"),
            "drawer_opening_extent": ("physics_gt.drawer_opening_extent", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "contact_finger_l_handle_flag": ("physics_gt.contact_finger_l_handle_flag", "scalar"),
            "contact_finger_l_handle_force": ("physics_gt.contact_finger_l_handle_force", "vector"),
            "contact_finger_r_handle_flag": ("physics_gt.contact_finger_r_handle_flag", "scalar"),
            "contact_finger_r_handle_force": ("physics_gt.contact_finger_r_handle_force", "vector"),
        },
    },
    "reach": {
        "static": [],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_angular_velocity": ("physics_gt.ee_angular_velocity", "vector"),
            "ee_to_target_distance": ("physics_gt.ee_to_target_distance", "scalar"),
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
}


# ── Data loading (optimized) ────────────────────────────────────────────────


def find_parquet_files(task: str) -> dict[int, str]:
    """Build mapping: episode_index → parquet path (across all chunks)."""
    task_dir = os.path.join(DATA_BASE, task, "data")
    ep_to_path = {}
    for chunk_dir in sorted(glob(os.path.join(task_dir, "chunk-*"))):
        for f in sorted(os.listdir(chunk_dir)):
            if f.endswith(".parquet"):
                ep_idx = int(f.split("_")[1].split(".")[0])
                ep_to_path[ep_idx] = os.path.join(chunk_dir, f)
    return ep_to_path


def load_window_starts_batch(task: str, model_tag: str) -> dict[int, np.ndarray]:
    """Load window_starts for all episodes from feature files."""
    feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
    n_episodes = TASK_EPISODE_COUNTS[task]
    result = {}
    for ep_idx in range(n_episodes):
        feat_path = os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")
        if os.path.exists(feat_path):
            with safe_open(feat_path, framework="numpy") as f:
                result[ep_idx] = f.get_tensor("window_starts").astype(np.int64)
    return result


def aggregate_frames_to_windows(
    col_values: list, window_starts: np.ndarray, var_type: str
) -> np.ndarray:
    """Aggregate per-frame column values into per-window scalars.

    col_values: list of numpy arrays (one per frame)
    Returns: (n_windows,) float64
    """
    # Stack all frames
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in col_values])
    # stacked: (n_frames, D) or (n_frames, 1) or (n_frames,)
    if stacked.ndim == 2 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)

    n_frames = len(stacked)
    window_vals = np.empty(len(window_starts), dtype=np.float64)

    for i, ws in enumerate(window_starts):
        end = min(ws + WINDOW_SIZE, n_frames)
        chunk = stacked[ws:end]
        if len(chunk) == 0:
            window_vals[i] = np.nan
        elif var_type == "vector":
            window_vals[i] = np.nanmean(np.linalg.norm(chunk, axis=1))
        else:
            window_vals[i] = np.nanmean(chunk)
    return window_vals


def load_all_gt_episode_level(task: str, model_tag: str) -> tuple[
    dict[str, np.ndarray],  # all_targets: var → (n_episodes,)
    list[int],              # episode_indices
]:
    """Load all GT data for a task at episode level (1 value per episode).

    Static vars: direct per-episode value.
    Dynamic vars: mean over entire episode (all frames).
    """
    task_vars = TASK_VARIABLES[task]
    static_var_names = task_vars["static"]
    dynamic_var_defs = task_vars["dynamic"]

    # Load static GT from episodes.jsonl
    static_ep_values = {}
    if static_var_names:
        meta_path = os.path.join(DATA_BASE, task, "meta", "episodes.jsonl")
        with open(meta_path) as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = ep["episode_index"]
                for var in static_var_names:
                    if var not in static_ep_values:
                        static_ep_values[var] = {}
                    static_ep_values[var][ep_idx] = float(ep[var])

    # Get episode list from feature files
    feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
    n_episodes = TASK_EPISODE_COUNTS[task]
    episode_indices = []
    for ep_idx in range(n_episodes):
        if os.path.exists(os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")):
            episode_indices.append(ep_idx)
    n_eps = len(episode_indices)
    print(f"  Episodes: {n_eps}", flush=True)

    # Build static targets
    all_targets = {}
    for var in static_var_names:
        arr = np.array([static_ep_values[var].get(ep, np.nan) for ep in episode_indices],
                       dtype=np.float64)
        all_targets[var] = arr

    # Build dynamic targets: mean over entire episode
    for var in dynamic_var_defs:
        all_targets[var] = np.empty(n_eps, dtype=np.float64)

    parquet_map = find_parquet_files(task)
    needed_cols = {col for col, _ in dynamic_var_defs.values()}

    print(f"  Loading dynamic GT from parquets...", flush=True)
    for i, ep_idx in enumerate(tqdm(episode_indices, desc=f"  Parquet [{task}]")):
        pq_path = parquet_map.get(ep_idx)
        if pq_path is None:
            for var in dynamic_var_defs:
                all_targets[var][i] = np.nan
            continue

        df = pd.read_parquet(pq_path, columns=list(needed_cols))

        for var_name, (col_name, var_type) in dynamic_var_defs.items():
            if col_name not in df.columns:
                all_targets[var_name][i] = np.nan
                continue
            raw = df[col_name].values
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in raw])
            if stacked.ndim == 2 and stacked.shape[1] == 1:
                stacked = stacked.squeeze(1)
            if var_type == "vector":
                all_targets[var_name][i] = float(np.nanmean(np.linalg.norm(stacked, axis=1)))
            else:
                all_targets[var_name][i] = float(np.nanmean(stacked))

    # Fill NaN
    for var_name, arr in all_targets.items():
        nan_mask = ~np.isfinite(arr)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            mean_val = np.nanmean(arr)
            arr[nan_mask] = mean_val if np.isfinite(mean_val) else 0.0
            if n_nan > 0:
                print(f"    {var_name}: filled {n_nan}/{len(arr)} NaN", flush=True)

    return all_targets, episode_indices


def load_episode_mean_features(
    task: str, num_layers: int, model_tag: str, episode_indices: list[int],
    dim: int,
) -> dict[int, np.ndarray]:
    """Load features for ALL layers, mean-pooled per episode (1 vector per episode per layer).

    Returns: dict[layer] → np.ndarray of shape (n_episodes, dim), float32
    """
    feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
    n_eps = len(episode_indices)

    X_all = {}
    for layer in range(num_layers):
        X_all[layer] = np.empty((n_eps, dim), dtype=np.float32)

    for i, ep_idx in enumerate(tqdm(episode_indices, desc=f"  Features [{task}]")):
        feat_path = os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")

        with safe_open(feat_path, framework="numpy") as f:
            n_w = len(f.get_tensor("window_starts"))
            for layer in range(num_layers):
                vecs = []
                for w in range(n_w):
                    vecs.append(f.get_tensor(f"layer_{layer}_window_{w}"))
                X_all[layer][i] = np.mean(vecs, axis=0)

    return X_all


# ── Probing ──────────────────────────────────────────────────────────────────


def probe_layer_batch(
    X: np.ndarray, targets: dict[str, np.ndarray], groups: np.ndarray,
    n_splits: int, alpha: float,
) -> dict[str, tuple[float, float, int]]:
    """Probe all variables for one layer using sklearn Ridge + GroupKFold.

    With episode-level data (~1500 samples), sklearn Ridge is fast enough.
    Returns: dict[var_name] → (r2_mean, r2_std, n_valid)
    """
    from sklearn.linear_model import Ridge

    n_samples = X.shape[0]
    n_unique = len(np.unique(groups))
    actual_splits = min(n_splits, n_unique)
    results = {}

    if actual_splits < 2:
        for var_name in targets:
            results[var_name] = (0.0, 0.0, n_samples)
        return results

    cv = GroupKFold(n_splits=actual_splits)
    folds = list(cv.split(X, groups, groups))

    for var_name, y in targets.items():
        if np.std(y) < 1e-10:
            results[var_name] = (0.0, 0.0, n_samples)
            continue

        r2_scores = []
        for train_idx, test_idx in folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            y_mean, y_std = y_tr.mean(), y_tr.std()
            if y_std < 1e-10:
                r2_scores.append(0.0)
                continue
            y_tr_s = (y_tr - y_mean) / y_std
            y_te_s = (y_te - y_mean) / y_std

            probe = Ridge(alpha=alpha)
            probe.fit(X_tr, y_tr_s)
            r2_scores.append(probe.score(X_te, y_te_s))

        results[var_name] = (float(np.mean(r2_scores)), float(np.std(r2_scores)), n_samples)

    return results


# ── Visualization ────────────────────────────────────────────────────────────


def plot_heatmap(csv_path: str, output_path: str):
    """Generate per-task heatmap: layer × variable → R²."""
    df = pd.read_csv(csv_path)
    tasks = df["task"].unique()
    n_tasks = len(tasks)

    fig, axes = plt.subplots(n_tasks, 1, figsize=(16, 5 * n_tasks), squeeze=False)

    for i, task in enumerate(tasks):
        ax = axes[i, 0]
        task_df = df[df["task"] == task]
        pivot = task_df.pivot(index="variable", columns="layer", values="r2_mean")
        pivot = pivot.sort_index()

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=1.0)
        ax.set_title(f"{task} — Linear Probe R²", fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Variable")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=7)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label="R² (mean)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Linear probing sweep on V-JEPA 2 features")
    parser.add_argument(
        "--model_size", type=str, default="giant", choices=["giant", "large"],
    )
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--tasks", type=str, nargs="+", default=TASKS)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    model_tag = cfg["tag"]
    num_layers = cfg["num_layers"]
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, f"probe_sweep_{model_tag}.csv")
    heatmap_path = os.path.join(args.output_dir, f"probe_sweep_{model_tag}_heatmap.png")

    all_results = []

    # Resume support
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        all_results = existing.to_dict("records")
        done_keys = set(zip(existing["task"], existing["layer"], existing["variable"]))
        print(f"Resuming: {len(all_results)} existing results loaded")
    else:
        done_keys = set()

    for task in args.tasks:
        task_vars = TASK_VARIABLES[task]
        all_var_names = [(v, "static") for v in task_vars["static"]]
        all_var_names += [(v, "dynamic") for v in task_vars["dynamic"]]

        # Check if entire task done
        task_total = num_layers * len(all_var_names)
        task_done_count = sum(
            1 for l in range(num_layers) for vn, _ in all_var_names
            if (task, l, vn) in done_keys
        )
        if task_done_count == task_total:
            print(f"[{task}] Already complete ({task_total} results), skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task} | Variables: {len(all_var_names)} | Layers: {num_layers}")
        print(f"{'='*60}", flush=True)

        # Load all GT at episode level (1 value per episode)
        all_targets, episode_indices = load_all_gt_episode_level(task, model_tag)
        groups = np.array(episode_indices, dtype=np.int64)

        # Load features: mean-pooled per episode (1 vector per episode per layer)
        X_all = load_episode_mean_features(
            task, num_layers, model_tag, episode_indices, cfg["dim"]
        )

        # Build var_type lookup
        var_type_map = {}
        for v in task_vars["static"]:
            var_type_map[v] = "static"
        for v in task_vars["dynamic"]:
            var_type_map[v] = "dynamic"

        task_results = []
        for layer in tqdm(range(num_layers), desc=f"  Probing [{task}]"):
            layer_done = all((task, layer, vn) in done_keys for vn, _ in all_var_names)
            if layer_done:
                continue

            # Convert to float32 for faster matrix ops
            X = X_all[layer]

            # Gather targets for this layer (skip already done)
            layer_targets = {}
            for var_name, var_type_label in all_var_names:
                if (task, layer, var_name) not in done_keys:
                    layer_targets[var_name] = all_targets[var_name]

            # Batch probe all variables for this layer
            layer_results = probe_layer_batch(X, layer_targets, groups, args.n_splits, args.alpha)

            for var_name, (r2_mean, r2_std, n_valid) in layer_results.items():
                row = {
                    "task": task,
                    "layer": layer,
                    "variable": var_name,
                    "var_type": var_type_map[var_name],
                    "r2_mean": round(r2_mean, 6),
                    "r2_std": round(r2_std, 6),
                    "n_samples": n_valid,
                }
                task_results.append(row)

        del X_all
        gc.collect()

        all_results.extend(task_results)

        # Save intermediate CSV
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({len(all_results)} total rows)", flush=True)

        # Free GT memory
        del static_targets, dynamic_targets, groups, all_targets
        gc.collect()

    # Final save & heatmap
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_path, index=False)
    print(f"\nFinal CSV: {csv_path} ({len(results_df)} rows)")
    plot_heatmap(csv_path, heatmap_path)

    # Summary
    print("\n" + "=" * 60)
    print("Top R² per task:")
    print("=" * 60)
    for task in args.tasks:
        task_df = results_df[results_df["task"] == task]
        if task_df.empty:
            continue
        best = task_df.loc[task_df["r2_mean"].idxmax()]
        print(
            f"  {task:12s} | L{int(best['layer']):2d} | "
            f"{best['variable']:30s} | R²={best['r2_mean']:.4f}±{best['r2_std']:.4f}"
        )


if __name__ == "__main__":
    main()
