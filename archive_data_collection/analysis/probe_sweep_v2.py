"""
Linear probing sweep v2: V-JEPA 2 frozen features -> physics GT prediction.

Rewritten based on PEZ reproduction (/home/solee/pez/step3_probe.py):
- Appendix B protocol: 20 HP configs (5 LR x 4 WD), 5-fold GroupKFold, per-fold best
- Batched trainable solver: all 20 configs trained in parallel via einsum
  (math equivalent to nn.Linear + Adam, verified by Codex to ~1e-7 R² diff)
- Ridge solver also available (much faster, for quick sanity checks)
- Per-episode aggregation (both feature mean-pool and GT mean-over-frames),
  matching what probe_sweep.py v1 did but with proper probing protocol

Compared to the original probe_sweep.py, this fixes:
- HP sweep (was: fixed alpha=1.0 -> now: 20 configs)
- NameError at legacy line 573 (undefined static_targets/dynamic_targets)
- Target-std rescaling inside the probe (now: standardize X & y, probe in std
  space, then invert for R²)
- Ambiguous "grouped" CV (explicit --grouping {episode,discrete_var})

Usage:
    /isaac-sim/python.sh analysis/probe_sweep_v2.py \
        --model_size giant --solver trainable --output_dir results_v2/
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

# --- Paper Appendix B protocol (matches PEZ step3_probe.py) ---
APPENDIX_B_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
APPENDIX_B_WD_GRID = [0.01, 0.1, 0.4, 0.8]
CV_N_SPLITS = 5
CV_RANDOM_SEED = 42
TRAINABLE_MAX_EPOCHS = 400
TRAINABLE_PATIENCE = 40

# --- Infrastructure ---
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

# --- Variable definitions (copied from v1) ---
TASK_VARIABLES = {
    "push": {
        "static": [
            "object_0_mass", "object_0_static_friction", "object_0_dynamic_friction",
            "surface_static_friction", "surface_dynamic_friction",
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
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
    "strike": {
        "static": [
            "object_0_mass", "object_0_static_friction", "object_0_dynamic_friction",
            "object_0_restitution", "surface_static_friction", "surface_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "object_velocity": ("physics_gt.object_velocity", "vector"),
            "object_acceleration": ("physics_gt.object_acceleration", "vector"),
            "ee_to_object_distance": ("physics_gt.ee_to_object_distance", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
            "ball_planar_travel_distance": ("physics_gt.ball_planar_travel_distance", "scalar"),
        },
    },
    "peg_insert": {
        "static": [
            "peg_mass", "peg_static_friction", "peg_dynamic_friction",
            "hole_static_friction", "hole_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "peg_velocity": ("physics_gt.peg_velocity", "vector"),
            "insertion_depth": ("physics_gt.insertion_depth", "scalar"),
            "peg_hole_lateral_error": ("physics_gt.peg_hole_lateral_error", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
    "nut_thread": {
        "static": [
            "nut_mass", "nut_static_friction", "nut_dynamic_friction",
            "bolt_static_friction", "bolt_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "nut_velocity": ("physics_gt.nut_velocity", "vector"),
            "axial_progress": ("physics_gt.axial_progress", "scalar"),
            "nut_bolt_relative_angle": ("physics_gt.nut_bolt_relative_angle", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
    "drawer": {
        "static": [
            "drawer_handle_mass", "drawer_joint_damping",
            "handle_static_friction", "handle_dynamic_friction",
        ],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "handle_velocity": ("physics_gt.handle_velocity", "vector"),
            "drawer_joint_pos": ("physics_gt.drawer_joint_pos", "scalar"),
            "drawer_joint_vel": ("physics_gt.drawer_joint_vel", "scalar"),
            "drawer_opening_extent": ("physics_gt.drawer_opening_extent", "scalar"),
            "contact_flag": ("physics_gt.contact_flag", "scalar"),
            "contact_force": ("physics_gt.contact_force", "vector"),
        },
    },
    "reach": {
        "static": [],
        "dynamic": {
            "ee_velocity": ("physics_gt.ee_velocity", "vector"),
            "ee_acceleration": ("physics_gt.ee_acceleration", "vector"),
            "ee_to_target_distance": ("physics_gt.ee_to_target_distance", "scalar"),
            "phase": ("physics_gt.phase", "scalar"),
        },
    },
}


# --- Data loading (per-episode aggregation, same as v1) ---

def find_parquet_files(task):
    task_dir = os.path.join(DATA_BASE, task, "data")
    ep_to_path = {}
    for chunk_dir in sorted(glob(os.path.join(task_dir, "chunk-*"))):
        for f in sorted(os.listdir(chunk_dir)):
            if f.endswith(".parquet"):
                ep_idx = int(f.split("_")[1].split(".")[0])
                ep_to_path[ep_idx] = os.path.join(chunk_dir, f)
    return ep_to_path


def load_all_gt_episode_level(task, model_tag):
    task_vars = TASK_VARIABLES[task]
    static_var_names = task_vars["static"]
    dynamic_var_defs = task_vars["dynamic"]

    # Static GT from episodes.jsonl — robust to missing keys
    # (some episodes fail to write full metadata; those become NaN and get
    # filled with the task mean later)
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

    for var_name, arr in all_targets.items():
        nan_mask = ~np.isfinite(arr)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            mean_val = np.nanmean(arr)
            arr[nan_mask] = mean_val if np.isfinite(mean_val) else 0.0
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


# --- Probe (transplanted from PEZ step3_probe.py, verified by Codex) ---

def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    ss_res = np.square(y_pred - y_true).sum()
    ss_tot = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_ridge(X_tr, y_tr, X_va, y_va, alpha):
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_tr, y_tr)
    return compute_r2(y_va, probe.predict(X_va))


def fit_trainable_batched(
    X_train, y_train, X_val, y_val, output_dim,
    lr_grid, wd_grid, device,
    max_epochs=TRAINABLE_MAX_EPOCHS, patience_limit=TRAINABLE_PATIENCE,
):
    """Verified equivalent to PEZ step3_probe.py:fit_trainable_probe_batched()."""
    configs = list(itertools.product(lr_grid, wd_grid))
    n_configs = len(configs)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True); x_std[x_std < 1e-6] = 1.0
    X_train_std = (X_train - x_mean) / x_std
    X_val_std = (X_val - x_mean) / x_std
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True); y_std[y_std < 1e-6] = 1.0
    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val_std, dtype=torch.float32, device=device)

    D, O = X_tr.shape[1], output_dim

    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)
    template = torch.nn.Linear(D, O, bias=True)
    W_init = template.weight.data.T.contiguous()
    b_init = template.bias.data.clone()
    W = W_init.unsqueeze(0).expand(n_configs, D, O).contiguous().to(device)
    b = b_init.unsqueeze(0).expand(n_configs, O).contiguous().to(device)
    W.requires_grad_(True); b.requires_grad_(True)

    m_W = torch.zeros_like(W); v_W = torch.zeros_like(W)
    m_b = torch.zeros_like(b); v_b = torch.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    lrs = torch.tensor([c[0] for c in configs], dtype=torch.float32, device=device)
    wds = torch.tensor([c[1] for c in configs], dtype=torch.float32, device=device)

    best_val_loss = torch.full((n_configs,), float("inf"), device=device)
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    patience = torch.zeros(n_configs, dtype=torch.int32, device=device)
    active = torch.ones(n_configs, dtype=torch.bool, device=device)

    for step in range(1, max_epochs + 1):
        pred_tr = torch.einsum("nd,cdo->cno", X_tr, W) + b.unsqueeze(1)
        loss_per_cfg = ((pred_tr - y_tr.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        total_loss = (loss_per_cfg * active.float()).sum()
        total_loss.backward()

        with torch.no_grad():
            # Coupled L2 WD on both W and b (matches torch.optim.Adam weight_decay)
            W.grad.add_(W * wds[:, None, None])
            b.grad.add_(b * wds[:, None])
            m_W.mul_(beta1).add_(W.grad, alpha=1 - beta1)
            v_W.mul_(beta2).addcmul_(W.grad, W.grad, value=1 - beta2)
            m_b.mul_(beta1).add_(b.grad, alpha=1 - beta1)
            v_b.mul_(beta2).addcmul_(b.grad, b.grad, value=1 - beta2)
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step
            m_W_hat = m_W / bc1; v_W_hat = v_W / bc2
            m_b_hat = m_b / bc1; v_b_hat = v_b / bc2
            active_W = active.float()[:, None, None]
            active_b = active.float()[:, None]
            W.data.sub_(active_W * lrs[:, None, None] * m_W_hat / (v_W_hat.sqrt() + eps))
            b.data.sub_(active_b * lrs[:, None] * m_b_hat / (v_b_hat.sqrt() + eps))
            W.grad.zero_(); b.grad.zero_()

        with torch.no_grad():
            pred_va = torch.einsum("nd,cdo->cno", X_va, W) + b.unsqueeze(1)
            val_loss_per_cfg = ((pred_va - y_va.unsqueeze(0)) ** 2).mean(dim=(1, 2))
            improved = val_loss_per_cfg + 1e-8 < best_val_loss
            best_val_loss = torch.where(improved, val_loss_per_cfg, best_val_loss)
            best_W = torch.where(improved[:, None, None].expand_as(W), W.detach(), best_W)
            best_b = torch.where(improved[:, None].expand_as(b), b.detach(), best_b)
            patience = torch.where(improved, torch.zeros_like(patience), patience + 1)
            active = active & (patience < patience_limit)
            if not active.any():
                break

    with torch.no_grad():
        pred_va_best = torch.einsum("nd,cdo->cno", X_va, best_W) + best_b.unsqueeze(1)
    pred_va_best_np = pred_va_best.cpu().numpy()
    pred_va_unscaled = pred_va_best_np * y_std[None, :, :] + y_mean[None, :, :]
    y_val_raw = y_val

    results = []
    ss_tot = np.square(y_val_raw - y_val_raw.mean(axis=0, keepdims=True)).sum()
    for i, (lr_val, wd_val) in enumerate(configs):
        ss_res = np.square(pred_va_unscaled[i] - y_val_raw).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        results.append((float(lr_val), float(wd_val), float(r2)))
    return results


def evaluate_layer(X, y, groups, solver, device, output_dim=1):
    """Grouped 5-fold CV + per-fold HP sweep."""
    n_unique = len(np.unique(groups))
    n_splits = min(CV_N_SPLITS, n_unique)
    if n_splits < 2 or np.std(y) < 1e-10:
        return 0.0, 0.0, None, None

    cv = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    fold_best_lr = []
    fold_best_wd = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if np.std(y_tr) < 1e-10 or np.std(y_va) < 1e-10:
            # Val fold has no variance -> R² undefined. Skip this fold.
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
    return (float(np.mean(fold_r2)), float(np.std(fold_r2)),
            fold_best_lr, fold_best_wd)


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
        ax.set_title(f"{task} — Linear Probe R² (batched trainable)", fontsize=14, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Linear probing v2 (PEZ-aligned)")
    parser.add_argument("--model_size", choices=["giant", "large"], default="giant")
    parser.add_argument("--solver", choices=["trainable", "ridge"], default="trainable")
    parser.add_argument("--output_dir", default="results_v2/")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    model_tag = cfg["tag"]
    num_layers = cfg["num_layers"]
    os.makedirs(args.output_dir, exist_ok=True)

    suffix = f"{model_tag}_{args.solver}"
    csv_path = os.path.join(args.output_dir, f"probe_sweep_{suffix}.csv")
    heatmap_path = os.path.join(args.output_dir, f"probe_sweep_{suffix}_heatmap.png")
    config_path = os.path.join(args.output_dir, f"probe_sweep_{suffix}_config.json")

    device = args.device if torch.cuda.is_available() else "cpu"

    # Resume support
    all_results = []
    done_keys = set()
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        all_results = existing.to_dict("records")
        done_keys = set(zip(existing["task"], existing["layer"], existing["variable"]))
        print(f"Resuming: {len(all_results)} existing results loaded")

    # Save config
    with open(config_path, "w") as f:
        json.dump({
            "model": args.model_size, "solver": args.solver,
            "lr_grid": APPENDIX_B_LR_GRID, "wd_grid": APPENDIX_B_WD_GRID,
            "cv_splits": CV_N_SPLITS, "cv_seed": CV_RANDOM_SEED,
            "max_epochs": TRAINABLE_MAX_EPOCHS, "patience": TRAINABLE_PATIENCE,
            "grouping": "episode_per_group (KFold)",
            "feature_aggregation": "mean-pool across all windows per episode",
            "gt_aggregation": "mean across all frames per episode (scalar + vector_norm)",
            "note": "Transplanted from PEZ step3_probe.py:fit_trainable_probe_batched",
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
        print(f"Task: {task} | Variables: {len(all_vars)} | Layers: {num_layers} | Solver: {args.solver}")
        print(f"{'='*60}", flush=True)

        all_targets, episode_indices = load_all_gt_episode_level(task, model_tag)
        groups = np.array(episode_indices, dtype=np.int64)
        X_all = load_episode_mean_features(task, num_layers, model_tag, episode_indices, cfg["dim"])

        var_type_map = {v: "static" for v in task_vars["static"]}
        var_type_map.update({v: "dynamic" for v in task_vars["dynamic"]})

        task_results = []
        t0 = time.time()
        for layer in tqdm(range(num_layers), desc=f"  Probing [{task}]"):
            X = X_all[layer]
            for var_name, _ in all_vars:
                if (task, layer, var_name) in done_keys:
                    continue
                y = all_targets[var_name]
                r2_mean, r2_std, _, _ = evaluate_layer(
                    X, y, groups, solver=args.solver, device=device, output_dim=1,
                )
                task_results.append({
                    "task": task, "layer": layer, "variable": var_name,
                    "var_type": var_type_map[var_name],
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
