"""
Linear probing sweep v3: window-level + multi-dim vector targets.

Extends probe_sweep_v2.py with PEZ-faithful protocol:
- --probing_level window: each sliding window is a separate sample
  (PEZ paper's "mean-pooled space-time patches" at clip level)
- --vector_mode xyz: keep vector GT as (x, y, z) multi-dim, don't reduce to norm
  (matches PEZ's velocity_xy / accel_xy probes)

Backward-compatible: defaults to v2 behavior (episode-mean + scalar norm)
if you pass --probing_level episode --vector_mode scalar_norm.

Usage:
    # PEZ-faithful (window-level, vector xyz):
    /isaac-sim/python.sh analysis/probe_sweep_v3.py \
        --model_size giant --solver ridge \
        --probing_level window --vector_mode xyz \
        --output_dir results_v3/

    # Legacy v2 behavior (for sanity checks):
    /isaac-sim/python.sh analysis/probe_sweep_v3.py \
        --model_size large --solver ridge \
        --probing_level episode --vector_mode scalar_norm \
        --output_dir results_v3/
"""

import argparse
import gc
import itertools
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# --- Constants (same as v2) ---
APPENDIX_B_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
APPENDIX_B_WD_GRID = [0.01, 0.1, 0.4, 0.8]
CV_N_SPLITS = 5
CV_RANDOM_SEED = 42
TRAINABLE_MAX_EPOCHS = 400
TRAINABLE_PATIENCE = 40

WINDOW_SIZE = 16  # sliding-window length used during feature extraction

MODEL_CONFIGS = {
    "giant": {"num_layers": 40, "dim": 1408, "tag": "vitg"},
    "large": {"num_layers": 24, "dim": 1024, "tag": "vitl"},
}
FEATURE_BASE = "/mnt/md1/solee/features"
DATA_BASE = "/mnt/md1/solee/data/isaac_physrepa_v2/step0"
TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]
TASK_EPISODE_COUNTS = {
    "push": 1500, "strike": 3000, "peg_insert": 2500,
    "nut_thread": 2500, "drawer": 2000, "reach": 600,
}

# Reuse TASK_VARIABLES from v2 via import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_sweep_v2 import TASK_VARIABLES

# --- Import helpers that haven't changed from v2 ---
from probe_sweep_v2 import (
    find_parquet_files,
    compute_r2,
    fit_ridge,
    fit_trainable_batched,
)


# --- Episode-level aggregation (v2 behavior) ---

def load_all_gt_episode_level(task, model_tag, vector_mode):
    """Per-episode GT aggregation. Supports 'scalar_norm' and 'xyz' vector modes.

    Returns:
        all_targets: dict[var_name] -> np.ndarray
            shape (n_eps,) for scalar or scalar_norm,
            shape (n_eps, 3) for xyz vector targets.
        episode_indices: list[int]
    """
    task_vars = TASK_VARIABLES[task]
    static_var_names = task_vars["static"]
    dynamic_var_defs = task_vars["dynamic"]

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
                    val = ep.get(var)
                    static_ep_values[var][ep_idx] = (
                        float(val) if val is not None else np.nan
                    )

    feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
    n_episodes = TASK_EPISODE_COUNTS[task]
    episode_indices = []
    for ep_idx in range(n_episodes):
        if os.path.exists(os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")):
            episode_indices.append(ep_idx)
    n_eps = len(episode_indices)
    print(f"  Episodes: {n_eps}", flush=True)

    all_targets = {}
    for var in static_var_names:
        arr = np.array([static_ep_values[var].get(ep, np.nan) for ep in episode_indices], dtype=np.float64)
        all_targets[var] = arr

    # Peek one parquet to determine vector dim for each dynamic var
    parquet_map = find_parquet_files(task)
    needed_cols = {col for col, _ in dynamic_var_defs.values()}
    first_valid_ep = next((e for e in episode_indices if e in parquet_map), None)
    vec_dims = {}
    if first_valid_ep is not None:
        df_peek = pd.read_parquet(parquet_map[first_valid_ep], columns=list(needed_cols))
        for var_name, (col_name, var_type) in dynamic_var_defs.items():
            if var_type == "vector" and col_name in df_peek.columns:
                first_val = np.asarray(df_peek[col_name].iloc[0])
                if first_val.ndim == 2 and first_val.shape[1] == 1:
                    first_val = first_val.squeeze(1)
                vec_dims[var_name] = int(first_val.shape[-1]) if first_val.ndim >= 1 else 1

    # Allocate target arrays
    for var_name, (col_name, var_type) in dynamic_var_defs.items():
        if var_type == "vector" and vector_mode == "xyz":
            d = vec_dims.get(var_name, 3)
            all_targets[var_name] = np.empty((n_eps, d), dtype=np.float64)
        else:
            all_targets[var_name] = np.empty(n_eps, dtype=np.float64)

    print(f"  Loading dynamic GT from parquets...", flush=True)
    for i, ep_idx in enumerate(tqdm(episode_indices, desc=f"  Parquet [{task}]")):
        pq_path = parquet_map.get(ep_idx)
        if pq_path is None:
            for var_name, arr in all_targets.items():
                if arr.ndim == 1:
                    arr[i] = np.nan
                else:
                    arr[i] = np.nan
            continue
        df = pd.read_parquet(pq_path, columns=list(needed_cols))
        for var_name, (col_name, var_type) in dynamic_var_defs.items():
            arr = all_targets[var_name]
            if col_name not in df.columns:
                if arr.ndim == 1:
                    arr[i] = np.nan
                else:
                    arr[i] = np.nan
                continue
            raw = df[col_name].values
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in raw])
            if stacked.ndim == 2 and stacked.shape[1] == 1:
                stacked = stacked.squeeze(1)
            if var_type == "vector":
                if vector_mode == "xyz":
                    # average vector across frames
                    arr[i] = np.nanmean(stacked, axis=0)
                else:
                    arr[i] = float(np.nanmean(np.linalg.norm(stacked, axis=1)))
            else:
                arr[i] = float(np.nanmean(stacked))

    # Fill NaN with mean
    for var_name, arr in all_targets.items():
        nan_mask = ~np.isfinite(arr)
        if arr.ndim == 2:
            # (n_eps, d)
            for d in range(arr.shape[1]):
                col = arr[:, d]
                col_nan = ~np.isfinite(col)
                n_nan = int(col_nan.sum())
                if n_nan > 0:
                    m = np.nanmean(col)
                    col[col_nan] = m if np.isfinite(m) else 0.0
        else:
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                m = np.nanmean(arr)
                arr[nan_mask] = m if np.isfinite(m) else 0.0
                print(f"    {var_name}: filled {n_nan}/{len(arr)} NaN", flush=True)

    return all_targets, episode_indices


def load_episode_mean_features(task, num_layers, model_tag, episode_indices, dim):
    feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
    n_eps = len(episode_indices)
    X_all = {layer: np.empty((n_eps, dim), dtype=np.float32) for layer in range(num_layers)}
    for i, ep_idx in enumerate(tqdm(episode_indices, desc=f"  Features [{task}]")):
        feat_path = os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")
        with safe_open(feat_path, framework="numpy") as f:
            n_w = len(f.get_tensor("window_starts"))
            for layer in range(num_layers):
                vecs = [f.get_tensor(f"layer_{layer}_window_{w}") for w in range(n_w)]
                X_all[layer][i] = np.mean(vecs, axis=0)
    return X_all


# --- Window-level aggregation (new in v3) ---

def load_window_level_features_and_groups(task, num_layers, model_tag, episode_indices, dim):
    """Load ALL windows across all episodes as individual samples.

    Returns:
        X_all: dict[layer] -> (n_total_windows, D)
        window_episode_ids: np.ndarray of shape (n_total_windows,) — episode_idx per window
        window_starts_all: np.ndarray of shape (n_total_windows,) — frame index where window begins
    """
    feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)

    # First pass: count total windows
    per_episode_ws = {}
    total_windows = 0
    for ep_idx in tqdm(episode_indices, desc=f"  Window counts [{task}]"):
        feat_path = os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")
        with safe_open(feat_path, framework="numpy") as f:
            ws = f.get_tensor("window_starts").astype(np.int64)
        per_episode_ws[ep_idx] = ws
        total_windows += len(ws)

    print(f"  Total windows: {total_windows}", flush=True)

    X_all = {layer: np.empty((total_windows, dim), dtype=np.float32) for layer in range(num_layers)}
    window_episode_ids = np.empty(total_windows, dtype=np.int64)
    window_starts_all = np.empty(total_windows, dtype=np.int64)

    offset = 0
    for ep_idx in tqdm(episode_indices, desc=f"  Features [{task}]"):
        feat_path = os.path.join(feature_dir, f"{ep_idx:06d}.safetensors")
        ws = per_episode_ws[ep_idx]
        n_w = len(ws)
        with safe_open(feat_path, framework="numpy") as f:
            for layer in range(num_layers):
                for w in range(n_w):
                    X_all[layer][offset + w] = f.get_tensor(f"layer_{layer}_window_{w}")
        window_episode_ids[offset:offset + n_w] = ep_idx
        window_starts_all[offset:offset + n_w] = ws
        offset += n_w

    return X_all, window_episode_ids, window_starts_all


def load_all_gt_window_level(task, episode_indices, window_episode_ids, window_starts_all, vector_mode):
    """Per-window GT: static vars broadcast to all windows of an episode,
    dynamic vars averaged over the 16 frames within each window.

    Returns:
        all_targets: dict[var] -> np.ndarray
            shape (n_windows,) for scalar / scalar_norm,
            shape (n_windows, 3) for xyz.
    """
    task_vars = TASK_VARIABLES[task]
    static_var_names = task_vars["static"]
    dynamic_var_defs = task_vars["dynamic"]
    n_windows = len(window_episode_ids)

    # --- Static ---
    static_ep_values = {v: {} for v in static_var_names}
    if static_var_names:
        meta_path = os.path.join(DATA_BASE, task, "meta", "episodes.jsonl")
        with open(meta_path) as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = ep["episode_index"]
                for var in static_var_names:
                    val = ep.get(var)
                    static_ep_values[var][ep_idx] = (
                        float(val) if val is not None else np.nan
                    )

    all_targets = {}
    for var in static_var_names:
        arr = np.empty(n_windows, dtype=np.float64)
        for i, ep_idx in enumerate(window_episode_ids):
            arr[i] = static_ep_values[var].get(int(ep_idx), np.nan)
        all_targets[var] = arr

    # --- Dynamic ---
    parquet_map = find_parquet_files(task)
    needed_cols = {col for col, _ in dynamic_var_defs.values()}

    # Peek vec dims
    first_valid_ep = next((e for e in episode_indices if e in parquet_map), None)
    vec_dims = {}
    if first_valid_ep is not None:
        df_peek = pd.read_parquet(parquet_map[first_valid_ep], columns=list(needed_cols))
        for var_name, (col_name, var_type) in dynamic_var_defs.items():
            if var_type == "vector" and col_name in df_peek.columns:
                first_val = np.asarray(df_peek[col_name].iloc[0])
                if first_val.ndim == 2 and first_val.shape[1] == 1:
                    first_val = first_val.squeeze(1)
                vec_dims[var_name] = int(first_val.shape[-1]) if first_val.ndim >= 1 else 1

    # Allocate
    for var_name, (col_name, var_type) in dynamic_var_defs.items():
        if var_type == "vector" and vector_mode == "xyz":
            d = vec_dims.get(var_name, 3)
            all_targets[var_name] = np.empty((n_windows, d), dtype=np.float64)
        else:
            all_targets[var_name] = np.empty(n_windows, dtype=np.float64)

    # Group windows by episode for efficient parquet loading
    ep_to_win_idx = {}
    for wi, ep_idx in enumerate(window_episode_ids):
        ep_to_win_idx.setdefault(int(ep_idx), []).append(wi)

    print(f"  Loading window-level dynamic GT from parquets...", flush=True)
    for ep_idx, win_idxs in tqdm(ep_to_win_idx.items(), desc=f"  Parquet [{task}]"):
        pq_path = parquet_map.get(ep_idx)
        if pq_path is None:
            for var_name, arr in all_targets.items():
                for wi in win_idxs:
                    if arr.ndim == 1:
                        arr[wi] = np.nan
                    else:
                        arr[wi] = np.nan
            continue
        df = pd.read_parquet(pq_path, columns=list(needed_cols))
        n_frames = len(df)

        for var_name, (col_name, var_type) in dynamic_var_defs.items():
            arr = all_targets[var_name]
            if col_name not in df.columns:
                for wi in win_idxs:
                    if arr.ndim == 1:
                        arr[wi] = np.nan
                    else:
                        arr[wi] = np.nan
                continue
            raw = df[col_name].values
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in raw])
            if stacked.ndim == 2 and stacked.shape[1] == 1:
                stacked = stacked.squeeze(1)

            for wi in win_idxs:
                ws = int(window_starts_all[wi])
                end = min(ws + WINDOW_SIZE, n_frames)
                chunk = stacked[ws:end]
                if len(chunk) == 0:
                    if arr.ndim == 1:
                        arr[wi] = np.nan
                    else:
                        arr[wi] = np.nan
                    continue
                if var_type == "vector":
                    if vector_mode == "xyz":
                        arr[wi] = np.nanmean(chunk, axis=0)
                    else:
                        arr[wi] = float(np.nanmean(np.linalg.norm(chunk, axis=1)))
                else:
                    arr[wi] = float(np.nanmean(chunk))

    # Fill NaN
    for var_name, arr in all_targets.items():
        if arr.ndim == 2:
            for d in range(arr.shape[1]):
                col = arr[:, d]
                col_nan = ~np.isfinite(col)
                n_nan = int(col_nan.sum())
                if n_nan > 0:
                    m = np.nanmean(col)
                    col[col_nan] = m if np.isfinite(m) else 0.0
        else:
            nan_mask = ~np.isfinite(arr)
            n_nan = int(nan_mask.sum())
            if n_nan > 0:
                m = np.nanmean(arr)
                arr[nan_mask] = m if np.isfinite(m) else 0.0
                print(f"    {var_name}: filled {n_nan}/{len(arr)} NaN", flush=True)

    return all_targets


# --- evaluate_layer: supports output_dim > 1 (unchanged from v2) ---

def evaluate_layer(X, y, groups, solver, device, output_dim=1):
    n_unique = len(np.unique(groups))
    n_splits = min(CV_N_SPLITS, n_unique)
    y_for_var = y if y.ndim == 1 else y.reshape(len(y), -1)
    if n_splits < 2 or np.nanstd(y_for_var) < 1e-10:
        return 0.0, 0.0, None, None

    cv = GroupKFold(n_splits=n_splits)
    fold_r2, fold_best_lr, fold_best_wd = [], [], []
    for train_idx, val_idx in cv.split(X, y_for_var, groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        tr_stable = np.nanstd(y_tr.reshape(len(y_tr), -1)) if y.ndim > 1 else np.nanstd(y_tr)
        va_stable = np.nanstd(y_va.reshape(len(y_va), -1)) if y.ndim > 1 else np.nanstd(y_va)
        if tr_stable < 1e-10 or va_stable < 1e-10:
            continue

        if solver == "trainable":
            results = fit_trainable_batched(
                X_tr, y_tr, X_va, y_va,
                output_dim=output_dim,
                lr_grid=APPENDIX_B_LR_GRID,
                wd_grid=APPENDIX_B_WD_GRID,
                device=device,
            )
            best_lr, best_wd, best_r2 = max(results, key=lambda r: r[2])
        elif solver == "ridge":
            best_r2 = -np.inf
            best_lr = 0.0
            best_wd = None
            for wd in APPENDIX_B_WD_GRID:
                r = fit_ridge(X_tr, y_tr, X_va, y_va, alpha=wd)
                if r > best_r2:
                    best_r2 = r
                    best_wd = wd
        else:
            raise ValueError(f"Unknown solver: {solver}")

        fold_r2.append(best_r2)
        fold_best_lr.append(best_lr)
        fold_best_wd.append(best_wd)

    if not fold_r2:
        return 0.0, 0.0, None, None
    return float(np.mean(fold_r2)), float(np.std(fold_r2)), fold_best_lr, fold_best_wd


# --- Visualization ---

def plot_heatmap(csv_path, output_path):
    df = pd.read_csv(csv_path)
    tasks = df["task"].unique()
    n_tasks = len(tasks)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(16, 5 * n_tasks), squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i, 0]
        task_df = df[df["task"] == task]
        pivot = task_df.pivot(index="variable", columns="layer", values="r2_mean").sort_index()
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=1.0)
        ax.set_title(f"{task} — Linear Probe R² (v3)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer"); ax.set_ylabel("Variable")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=7)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label="R² (mean)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {output_path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Linear probing v3 (window-level + multi-dim)")
    parser.add_argument("--model_size", choices=["giant", "large"], default="giant")
    parser.add_argument("--solver", choices=["trainable", "ridge"], default="trainable")
    parser.add_argument("--probing_level", choices=["episode", "window"], default="window",
                        help="episode: one sample per episode (v2 default); window: one sample per sliding window (PEZ-faithful)")
    parser.add_argument("--vector_mode", choices=["scalar_norm", "xyz"], default="xyz",
                        help="scalar_norm: reduce vector GT to ||vec|| (v2 default); xyz: keep multi-dim (PEZ-faithful)")
    parser.add_argument("--output_dir", default="results_v3/")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    model_tag = cfg["tag"]
    num_layers = cfg["num_layers"]
    os.makedirs(args.output_dir, exist_ok=True)

    suffix = f"{model_tag}_{args.solver}_{args.probing_level}_{args.vector_mode}"
    csv_path = os.path.join(args.output_dir, f"probe_sweep_{suffix}.csv")
    heatmap_path = os.path.join(args.output_dir, f"probe_sweep_{suffix}_heatmap.png")
    config_path = os.path.join(args.output_dir, f"probe_sweep_{suffix}_config.json")

    device = args.device if torch.cuda.is_available() else "cpu"

    all_results = []
    done_keys = set()
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        all_results = existing.to_dict("records")
        done_keys = set(zip(existing["task"], existing["layer"], existing["variable"]))
        print(f"Resuming: {len(all_results)} existing results loaded")

    with open(config_path, "w") as f:
        json.dump({
            "model": args.model_size, "solver": args.solver,
            "probing_level": args.probing_level, "vector_mode": args.vector_mode,
            "lr_grid": APPENDIX_B_LR_GRID, "wd_grid": APPENDIX_B_WD_GRID,
            "cv_splits": CV_N_SPLITS, "cv_seed": CV_RANDOM_SEED,
            "max_epochs": TRAINABLE_MAX_EPOCHS, "patience": TRAINABLE_PATIENCE,
            "grouping": "episode_id (window-level) or one-per-episode (episode-level)",
            "note": "v3 extends v2 with PEZ-faithful window-level + multi-dim vector support",
        }, f, indent=2)

    for task in args.tasks:
        task_vars = TASK_VARIABLES[task]
        all_vars = [(v, "static") for v in task_vars["static"]]
        all_vars += [(v, "dynamic") for v in task_vars["dynamic"]]

        task_total = num_layers * len(all_vars)
        task_done = sum(1 for l in range(num_layers) for vn, _ in all_vars if (task, l, vn) in done_keys)
        if task_done == task_total:
            print(f"[{task}] Already complete ({task_total} results), skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task} | Vars: {len(all_vars)} | Layers: {num_layers} | "
              f"Solver: {args.solver} | Level: {args.probing_level} | Vector: {args.vector_mode}")
        print(f"{'='*60}", flush=True)

        # Build episode list once
        feature_dir = os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
        n_episodes = TASK_EPISODE_COUNTS[task]
        episode_indices = [e for e in range(n_episodes)
                           if os.path.exists(os.path.join(feature_dir, f"{e:06d}.safetensors"))]

        if args.probing_level == "episode":
            all_targets, episode_indices = load_all_gt_episode_level(task, model_tag, args.vector_mode)
            groups = np.array(episode_indices, dtype=np.int64)
            X_all = load_episode_mean_features(task, num_layers, model_tag, episode_indices, cfg["dim"])
        else:
            X_all, window_episode_ids, window_starts_all = load_window_level_features_and_groups(
                task, num_layers, model_tag, episode_indices, cfg["dim"]
            )
            all_targets = load_all_gt_window_level(
                task, episode_indices, window_episode_ids, window_starts_all, args.vector_mode
            )
            groups = window_episode_ids  # group by episode so entire ep stays in 1 fold

        var_type_map = {v: "static" for v in task_vars["static"]}
        var_type_map.update({v: "dynamic" for v in task_vars["dynamic"]})

        # Determine output_dim per var
        def _get_output_dim(var_name):
            y = all_targets[var_name]
            if y.ndim == 2:
                return y.shape[1]
            return 1

        task_results = []
        t0 = time.time()
        for layer in tqdm(range(num_layers), desc=f"  Probing [{task}]"):
            X = X_all[layer]
            for var_name, _ in all_vars:
                if (task, layer, var_name) in done_keys:
                    continue
                y = all_targets[var_name]
                output_dim = _get_output_dim(var_name)
                r2_mean, r2_std, _, _ = evaluate_layer(
                    X, y, groups, solver=args.solver, device=device, output_dim=output_dim,
                )
                task_results.append({
                    "task": task, "layer": layer, "variable": var_name,
                    "var_type": var_type_map[var_name],
                    "output_dim": output_dim,
                    "r2_mean": round(r2_mean, 6), "r2_std": round(r2_std, 6),
                    "n_samples": len(y),
                })
        elapsed = time.time() - t0
        print(f"  [{task}] Done in {elapsed:.1f}s", flush=True)

        del X_all; gc.collect()
        all_results.extend(task_results)
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({len(all_results)} total rows)", flush=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_path, index=False)
    print(f"\nFinal CSV: {csv_path} ({len(results_df)} rows)")
    plot_heatmap(csv_path, heatmap_path)

    print("\n" + "=" * 60)
    print("Top R² per task:")
    print("=" * 60)
    for task in args.tasks:
        task_df = results_df[results_df["task"] == task]
        if task_df.empty:
            continue
        best = task_df.loc[task_df["r2_mean"].idxmax()]
        print(f"  {task:12s} | L{int(best['layer']):2d} | {best['variable']:30s} | "
              f"R²={best['r2_mean']:.4f}±{best['r2_std']:.4f}")


if __name__ == "__main__":
    main()
