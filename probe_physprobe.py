#!/usr/bin/env python3
"""PEZ-style probing on PhysProbe manipulation data.

Phase 1 design:
- existing per-episode safetensors, each containing multiple mean-pooled window vectors
- aggregate windows to one episode-level feature vector per layer
- GroupKFold by episode_id
- trainable 20-HP sweep (Appendix-B-style) as the default solver
- z-score normalization

Outputs:
- artifacts/results/probe_{task}_{target}_{model}[_{feature_type}][_{run_tag}].csv
- artifacts/figures/curve_{task}_{target}_{model}[_{feature_type}][_{run_tag}].png
- artifacts/results/verdict_{run_tag}_{task}.json
- artifacts/results/sanity_{run_tag}_{task}.json
- artifacts/results/EXPERIMENT_RESULTS_{run_tag}_{task}.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]
MODEL_CONFIGS = {
    "large": {"tag": "vitl", "num_layers": 24, "dim": 1024},
    "giant": {"tag": "vitg", "num_layers": 40, "dim": 1408},
}

FEATURE_BASE = "/mnt/md1/solee/features"
DATA_BASE = "/home/solee/data/data/isaac_physrepa_v2/step0"
ARTIFACTS_ROOT = "/home/solee/physrepa_tasks/artifacts"
RESULTS_DIR = os.path.join(ARTIFACTS_ROOT, "results")
FIGURES_DIR = os.path.join(ARTIFACTS_ROOT, "figures")

LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WD_GRID = [0.01, 0.1, 0.4, 0.8]
CV_SPLITS = 5
CV_RANDOM_SEED = 42
MAX_EPOCHS = 400
PATIENCE = 40


TARGET_SPECS = {
    "push": {
        "mass": {"kind": "static", "source": "object_0_mass", "output_dim": 1},
        "obj_friction": {"kind": "static", "source": "object_0_static_friction", "output_dim": 1},
        "obj_dyn_friction": {"kind": "static", "source": "object_0_dynamic_friction", "output_dim": 1},
        "surface_friction": {"kind": "static", "source": "surface_static_friction", "output_dim": 1},
        "surface_dyn_friction": {"kind": "static", "source": "surface_dynamic_friction", "output_dim": 1},
        "ee_pos": {"kind": "dynamic_vector", "source": "physics_gt.ee_position", "output_dim": 3},
        "object_pos": {"kind": "dynamic_vector", "source": "physics_gt.object_position", "output_dim": 3},
        "ee_velocity": {"kind": "dynamic_vector", "source": "physics_gt.ee_velocity", "output_dim": 3},
        "ee_acceleration": {"kind": "dynamic_vector", "source": "physics_gt.ee_acceleration", "output_dim": 3},
        "object_velocity": {"kind": "dynamic_vector", "source": "physics_gt.object_velocity", "output_dim": 3},
        "object_acceleration": {"kind": "dynamic_vector", "source": "physics_gt.object_acceleration", "output_dim": 3},
        "ee_speed": {"kind": "dynamic_vector_norm", "source": "physics_gt.ee_velocity", "output_dim": 1},
        "ee_accel_magnitude": {
            "kind": "dynamic_vector_norm",
            "source": "physics_gt.ee_acceleration",
            "output_dim": 1,
        },
        "ee_direction": {
            "kind": "dynamic_vector_angle_xy",
            "source": "physics_gt.ee_velocity",
            "output_dim": 1,
        },
        "ee_direction_sincos": {
            "kind": "dynamic_vector_angle_xy_sincos",
            "source": "physics_gt.ee_velocity",
            "output_dim": 2,
        },
        "object_speed": {"kind": "dynamic_vector_norm", "source": "physics_gt.object_velocity", "output_dim": 1},
        "object_accel_magnitude": {
            "kind": "dynamic_vector_norm",
            "source": "physics_gt.object_acceleration",
            "output_dim": 1,
        },
        "object_direction": {
            "kind": "dynamic_vector_angle_xy",
            "source": "physics_gt.object_velocity",
            "output_dim": 1,
        },
        "object_direction_sincos": {
            "kind": "dynamic_vector_angle_xy_sincos",
            "source": "physics_gt.object_velocity",
            "output_dim": 2,
        },
        "ee_to_object_distance": {"kind": "dynamic_scalar", "source": "physics_gt.ee_to_object_distance", "output_dim": 1},
        "object_to_target_distance": {"kind": "dynamic_scalar", "source": "physics_gt.object_to_target_distance", "output_dim": 1},
    },
    "strike": {
        "mass": {"kind": "static", "source": "object_0_mass", "output_dim": 1},
        "obj_friction": {"kind": "static", "source": "object_0_static_friction", "output_dim": 1},
        "obj_dyn_friction": {"kind": "static", "source": "object_0_dynamic_friction", "output_dim": 1},
        "surface_friction": {"kind": "static", "source": "surface_static_friction", "output_dim": 1},
        "surface_dyn_friction": {"kind": "static", "source": "surface_dynamic_friction", "output_dim": 1},
        "restitution": {"kind": "static", "source": "object_0_restitution", "output_dim": 1},
        "ee_pos": {"kind": "dynamic_vector", "source": "physics_gt.ee_position", "output_dim": 3},
        "object_pos": {"kind": "dynamic_vector", "source": "physics_gt.object_position", "output_dim": 3},
        "ee_velocity": {"kind": "dynamic_vector", "source": "physics_gt.ee_velocity", "output_dim": 3},
        "ee_acceleration": {"kind": "dynamic_vector", "source": "physics_gt.ee_acceleration", "output_dim": 3},
        "object_velocity": {"kind": "dynamic_vector", "source": "physics_gt.object_velocity", "output_dim": 3},
        "object_acceleration": {"kind": "dynamic_vector", "source": "physics_gt.object_acceleration", "output_dim": 3},
        "ee_speed": {"kind": "dynamic_vector_norm", "source": "physics_gt.ee_velocity", "output_dim": 1},
        "ee_accel_magnitude": {"kind": "dynamic_vector_norm", "source": "physics_gt.ee_acceleration", "output_dim": 1},
        "ee_direction_sincos": {
            "kind": "dynamic_vector_angle_xy_sincos",
            "source": "physics_gt.ee_velocity",
            "output_dim": 2,
        },
        "fake_mod5": {"kind": "synthetic", "source": "episode_index_mod_5", "output_dim": 1},
        "ball_planar_travel_distance": {
            "kind": "dynamic_scalar",
            "source": "physics_gt.ball_planar_travel_distance",
            "output_dim": 1,
        },
    },
    "peg_insert": {
        "mass": {"kind": "static", "source": "peg_mass", "output_dim": 1},
        "peg_friction": {"kind": "static", "source": "peg_static_friction", "output_dim": 1},
        "peg_dyn_friction": {"kind": "static", "source": "peg_dynamic_friction", "output_dim": 1},
        "hole_friction": {"kind": "static", "source": "hole_static_friction", "output_dim": 1},
        "hole_dyn_friction": {"kind": "static", "source": "hole_dynamic_friction", "output_dim": 1},
        "ee_pos": {"kind": "dynamic_vector", "source": "physics_gt.ee_position", "output_dim": 3},
        "peg_pos": {"kind": "dynamic_vector", "source": "physics_gt.peg_position", "output_dim": 3},
        "peg_velocity": {"kind": "dynamic_vector", "source": "physics_gt.peg_velocity", "output_dim": 3},
        "insertion_depth": {"kind": "dynamic_scalar", "source": "physics_gt.insertion_depth", "output_dim": 1},
        "peg_hole_lateral_error": {
            "kind": "dynamic_scalar",
            "source": "physics_gt.peg_hole_lateral_error",
            "output_dim": 1,
        },
    },
    "nut_thread": {
        "mass": {"kind": "static", "source": "nut_mass", "output_dim": 1},
        "nut_friction": {"kind": "static", "source": "nut_static_friction", "output_dim": 1},
        "nut_dyn_friction": {"kind": "static", "source": "nut_dynamic_friction", "output_dim": 1},
        "bolt_friction": {"kind": "static", "source": "bolt_static_friction", "output_dim": 1},
        "bolt_dyn_friction": {"kind": "static", "source": "bolt_dynamic_friction", "output_dim": 1},
        "ee_pos": {"kind": "dynamic_vector", "source": "physics_gt.ee_position", "output_dim": 3},
        "nut_pos": {"kind": "dynamic_vector", "source": "physics_gt.nut_position", "output_dim": 3},
        "nut_velocity": {"kind": "dynamic_vector", "source": "physics_gt.nut_velocity", "output_dim": 3},
        "axial_progress": {"kind": "dynamic_scalar", "source": "physics_gt.axial_progress", "output_dim": 1},
        "nut_bolt_relative_angle": {
            "kind": "dynamic_scalar",
            "source": "physics_gt.nut_bolt_relative_angle",
            "output_dim": 1,
        },
    },
    "drawer": {
        "damping": {"kind": "static", "source": "drawer_joint_damping", "output_dim": 1},
        "drawer_handle_mass": {"kind": "static", "source": "drawer_handle_mass", "output_dim": 1},
        "handle_friction": {"kind": "static", "source": "handle_static_friction", "output_dim": 1},
        "handle_dyn_friction": {"kind": "static", "source": "handle_dynamic_friction", "output_dim": 1},
        "ee_pos": {"kind": "dynamic_vector", "source": "physics_gt.ee_position", "output_dim": 3},
        "handle_pos": {"kind": "dynamic_vector", "source": "physics_gt.handle_position", "output_dim": 3},
        "handle_velocity": {"kind": "dynamic_vector", "source": "physics_gt.handle_velocity", "output_dim": 3},
        "drawer_joint_pos": {"kind": "dynamic_scalar", "source": "physics_gt.drawer_joint_pos", "output_dim": 1},
        "drawer_joint_vel": {"kind": "dynamic_scalar", "source": "physics_gt.drawer_joint_vel", "output_dim": 1},
        "drawer_opening_extent": {
            "kind": "dynamic_scalar",
            "source": "physics_gt.drawer_opening_extent",
            "output_dim": 1,
        },
    },
    "reach": {
        "ee_pos": {"kind": "dynamic_vector", "source": "physics_gt.ee_position", "output_dim": 3},
        "ee_velocity": {"kind": "dynamic_vector", "source": "physics_gt.ee_velocity", "output_dim": 3},
        "ee_acceleration": {"kind": "dynamic_vector", "source": "physics_gt.ee_acceleration", "output_dim": 3},
        "ee_speed": {"kind": "dynamic_vector_norm", "source": "physics_gt.ee_velocity", "output_dim": 1},
        "ee_accel_magnitude": {"kind": "dynamic_vector_norm", "source": "physics_gt.ee_acceleration", "output_dim": 1},
        "ee_direction": {
            "kind": "dynamic_vector_angle_xy",
            "source": "physics_gt.ee_velocity",
            "output_dim": 1,
        },
        "ee_direction_sincos": {
            "kind": "dynamic_vector_angle_xy_sincos",
            "source": "physics_gt.ee_velocity",
            "output_dim": 2,
        },
        "ee_to_target_distance": {
            "kind": "dynamic_scalar",
            "source": "physics_gt.ee_to_target_distance",
            "output_dim": 1,
        },
        "fake_mod5": {"kind": "synthetic", "source": "episode_index_mod_5", "output_dim": 1},
        "fake_shuffled": {"kind": "synthetic", "source": "episode_index_shuffled", "output_dim": 1},
    },
}


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def sanitize_target_name(name: str) -> str:
    return name.replace("/", "_")


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
    return float(1.0 - ss_res / ss_tot)


def normalize_train_val(train, val, mode: str):
    mean = train.mean(axis=0, keepdims=True)
    if mode == "none":
        return train, val, np.zeros_like(mean), np.ones_like(mean)
    if mode == "center":
        return train - mean, val - mean, mean, np.ones_like(mean)
    if mode == "zscore":
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (train - mean) / std, (val - mean) / std, mean, std
    raise ValueError(f"Unknown norm mode: {mode}")


def fit_trainable_batched(
    X_train,
    y_train,
    X_val,
    y_val,
    output_dim,
    lr_grid,
    wd_grid,
    device,
    max_epochs=MAX_EPOCHS,
    patience_limit=PATIENCE,
    norm_mode="zscore",
):
    configs = list(itertools.product(lr_grid, wd_grid))
    n_configs = len(configs)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    X_train_std, X_val_std, _, _ = normalize_train_val(X_train, X_val, norm_mode)
    y_train_std, y_val_std, y_mean, y_std = normalize_train_val(y_train, y_val, norm_mode)

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val_std, dtype=torch.float32, device=device)

    input_dim = X_tr.shape[1]
    output_dim = int(output_dim)

    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)
    template = torch.nn.Linear(input_dim, output_dim, bias=True)
    W = template.weight.data.T.contiguous().unsqueeze(0).expand(n_configs, input_dim, output_dim)
    b = template.bias.data.clone().unsqueeze(0).expand(n_configs, output_dim)
    W = W.contiguous().to(device)
    b = b.contiguous().to(device)
    W.requires_grad_(True)
    b.requires_grad_(True)

    m_W = torch.zeros_like(W)
    v_W = torch.zeros_like(W)
    m_b = torch.zeros_like(b)
    v_b = torch.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    lrs = torch.tensor([cfg[0] for cfg in configs], dtype=torch.float32, device=device)
    wds = torch.tensor([cfg[1] for cfg in configs], dtype=torch.float32, device=device)

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
            W.grad.add_(W * wds[:, None, None])
            b.grad.add_(b * wds[:, None])

            m_W.mul_(beta1).add_(W.grad, alpha=1 - beta1)
            v_W.mul_(beta2).addcmul_(W.grad, W.grad, value=1 - beta2)
            m_b.mul_(beta1).add_(b.grad, alpha=1 - beta1)
            v_b.mul_(beta2).addcmul_(b.grad, b.grad, value=1 - beta2)

            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step
            m_W_hat = m_W / bc1
            v_W_hat = v_W / bc2
            m_b_hat = m_b / bc1
            v_b_hat = v_b / bc2

            active_mask_W = active.float()[:, None, None]
            active_mask_b = active.float()[:, None]

            W.data.sub_(active_mask_W * lrs[:, None, None] * m_W_hat / (v_W_hat.sqrt() + eps))
            b.data.sub_(active_mask_b * lrs[:, None] * m_b_hat / (v_b_hat.sqrt() + eps))
            W.grad.zero_()
            b.grad.zero_()

        with torch.no_grad():
            pred_va = torch.einsum("nd,cdo->cno", X_va, W) + b.unsqueeze(1)
            val_loss = ((pred_va - y_va.unsqueeze(0)) ** 2).mean(dim=(1, 2))
            improved = val_loss + 1e-8 < best_val_loss
            best_val_loss = torch.where(improved, val_loss, best_val_loss)
            best_W = torch.where(improved[:, None, None], W.detach(), best_W)
            best_b = torch.where(improved[:, None], b.detach(), best_b)
            patience = torch.where(improved, torch.zeros_like(patience), patience + 1)
            active = active & (patience < patience_limit)
            if not active.any():
                break

    with torch.no_grad():
        pred_va_best = torch.einsum("nd,cdo->cno", X_va, best_W) + best_b.unsqueeze(1)
    pred_va_unscaled = pred_va_best.cpu().numpy() * y_std[None, :, :] + y_mean[None, :, :]

    results = []
    for cfg_index, (lr, wd) in enumerate(configs):
        results.append((float(lr), float(wd), compute_r2(y_val, pred_va_unscaled[cfg_index])))
    return results


def classify_curve(r2_per_layer):
    r2_per_layer = np.asarray(r2_per_layer, dtype=np.float64)
    l0 = float(r2_per_layer[0])
    peak_r2 = float(np.max(r2_per_layer))
    peak_layer = int(np.argmax(r2_per_layer))
    fractional_depth = peak_layer / max(1, len(r2_per_layer))
    if l0 >= 0.8:
        return "always-linear"
    if peak_r2 < 0.3:
        return "never-linear"
    if peak_r2 >= 0.5 and 0.2 <= fractional_depth <= 0.6:
        return "PEZ-like"
    return "intermediate"


def resolve_feature_root(task: str, model: str, feature_type: str = "mean", feature_root: str | None = None) -> str:
    if feature_root is not None:
        return os.path.join(feature_root, task)
    model_tag = MODEL_CONFIGS[model]["tag"]
    if feature_type == "mean":
        return os.path.join(FEATURE_BASE, f"physprobe_{model_tag}", task)
    if feature_type == "token_patch":
        return os.path.join(FEATURE_BASE, f"physprobe_{model_tag}_tokenpatch", task)
    raise ValueError(f"Unknown feature_type: {feature_type}")


def list_feature_episodes(feature_root: str) -> list[int]:
    episodes = []
    for path in sorted(glob(os.path.join(feature_root, "*.safetensors"))):
        name = os.path.basename(path).split(".")[0]
        episodes.append(int(name))
    return episodes


def load_static_meta(task: str):
    meta_path = os.path.join(DATA_BASE, task, "meta", "episodes.jsonl")
    episode_meta = {}
    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            episode_meta[int(row["episode_index"])] = row
    return episode_meta


def find_parquet_files(task: str):
    task_dir = os.path.join(DATA_BASE, task, "data")
    ep_to_path = {}
    for chunk_dir in sorted(glob(os.path.join(task_dir, "chunk-*"))):
        for f in sorted(os.listdir(chunk_dir)):
            if f.endswith(".parquet"):
                ep_idx = int(f.split("_")[1].split(".")[0])
                ep_to_path[ep_idx] = os.path.join(chunk_dir, f)
    return ep_to_path


def stack_series(values):
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values])
    if stacked.ndim == 2 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)
    return stacked


def load_targets(task: str, episode_indices: list[int], requested_targets: list[str]):
    task_specs = TARGET_SPECS[task]
    missing = [target for target in requested_targets if target not in task_specs]
    if missing:
        raise ValueError(f"Unknown targets for task={task}: {missing}")

    static_meta = load_static_meta(task)
    parquet_map = find_parquet_files(task)

    needed_dynamic_cols = sorted(
        {
            task_specs[target]["source"]
            for target in requested_targets
            if task_specs[target]["kind"].startswith("dynamic")
        }
    )

    targets = {}
    for target in requested_targets:
        spec = task_specs[target]
        if spec["kind"] == "static":
            values = []
            for ep_idx in episode_indices:
                val = static_meta[ep_idx].get(spec["source"], np.nan)
                values.append(float(val) if val is not None else np.nan)
            targets[target] = np.asarray(values, dtype=np.float64)
        elif spec["kind"] == "synthetic":
            if spec["source"] == "episode_index_mod_5":
                targets[target] = (np.asarray(episode_indices, dtype=np.float64) % 5).astype(np.float64)
            elif spec["source"] == "episode_index_shuffled":
                rng = np.random.RandomState(CV_RANDOM_SEED)
                targets[target] = rng.permutation(np.asarray(episode_indices, dtype=np.float64))
            else:
                raise ValueError(f"Unknown synthetic source: {spec['source']}")
        else:
            targets[target] = []

    if needed_dynamic_cols:
        for ep_idx in tqdm(episode_indices, desc=f"Load parquet targets [{task}]"):
            df = pd.read_parquet(parquet_map[ep_idx], columns=needed_dynamic_cols)
            for target in requested_targets:
                spec = task_specs[target]
                if spec["kind"] == "dynamic_scalar":
                    if spec["source"] not in df.columns:
                        targets[target].append(np.nan)
                    else:
                        arr = stack_series(df[spec["source"]].values)
                        targets[target].append(float(np.nanmean(arr)))
                elif spec["kind"] == "dynamic_vector":
                    if spec["source"] not in df.columns:
                        targets[target].append(np.full((spec["output_dim"],), np.nan))
                    else:
                        arr = stack_series(df[spec["source"]].values)
                        vec = np.nanmean(arr, axis=0)
                        targets[target].append(np.asarray(vec, dtype=np.float64))
                elif spec["kind"] == "dynamic_vector_norm":
                    if spec["source"] not in df.columns:
                        targets[target].append(np.nan)
                    else:
                        arr = stack_series(df[spec["source"]].values)
                        norms = np.linalg.norm(arr, axis=1)
                        targets[target].append(float(np.nanmean(norms)))
                elif spec["kind"] == "dynamic_vector_angle_xy":
                    if spec["source"] not in df.columns:
                        targets[target].append(np.nan)
                    else:
                        arr = stack_series(df[spec["source"]].values)
                        angles = np.arctan2(arr[:, 1], arr[:, 0])
                        mean_sin = np.nanmean(np.sin(angles))
                        mean_cos = np.nanmean(np.cos(angles))
                        targets[target].append(float(np.arctan2(mean_sin, mean_cos)))
                elif spec["kind"] == "dynamic_vector_angle_xy_sincos":
                    if spec["source"] not in df.columns:
                        targets[target].append(np.full((2,), np.nan))
                    else:
                        arr = stack_series(df[spec["source"]].values)
                        angles = np.arctan2(arr[:, 1], arr[:, 0])
                        mean_sin = np.nanmean(np.sin(angles))
                        mean_cos = np.nanmean(np.cos(angles))
                        targets[target].append(np.asarray([mean_sin, mean_cos], dtype=np.float64))

    for target in requested_targets:
        spec = task_specs[target]
        arr = np.asarray(targets[target], dtype=np.float64)
        if spec["output_dim"] == 1:
            nan_mask = ~np.isfinite(arr)
            if nan_mask.any():
                fill = np.nanmean(arr)
                arr[nan_mask] = fill if np.isfinite(fill) else 0.0
        else:
            for d in range(arr.shape[1]):
                col = arr[:, d]
                nan_mask = ~np.isfinite(col)
                if nan_mask.any():
                    fill = np.nanmean(col)
                    col[nan_mask] = fill if np.isfinite(fill) else 0.0
                    arr[:, d] = col
        targets[target] = arr

    return targets


def build_output_stem(task: str, target: str, model: str, feature_type: str, run_tag: str) -> str:
    stem = f"{task}_{sanitize_target_name(target)}_{model}"
    if feature_type != "mean":
        stem += f"_{feature_type}"
    if run_tag != "phase1":
        stem += f"_{run_tag}"
    return stem


def load_episode_features(
    task: str,
    model: str,
    episode_indices: list[int],
    feature_type: str = "mean",
    feature_root_override: str | None = None,
):
    cfg = MODEL_CONFIGS[model]
    feature_root = resolve_feature_root(task, model, feature_type=feature_type, feature_root=feature_root_override)
    num_layers = cfg["num_layers"]
    base_dim = cfg["dim"]
    feature_dims = None

    shape_info = {
        "feature_root": feature_root,
        "feature_type": feature_type,
        "num_layers": num_layers,
        "base_dim": base_dim,
        "episodes": len(episode_indices),
        "window_counts": [],
        "missing_keys": [],
        "patch_shape": None,
    }

    X_all = None

    for row_index, ep_idx in enumerate(tqdm(episode_indices, desc=f"Load features [{task}/{model}/{feature_type}]")):
        feat_path = os.path.join(feature_root, f"{ep_idx:06d}.safetensors")
        with safe_open(feat_path, framework="numpy") as f:
            keys = set(f.keys())
            if "window_starts" not in keys:
                raise ValueError(f"window_starts missing in {feat_path}")
            window_starts = f.get_tensor("window_starts").astype(np.int64)
            n_w = len(window_starts)
            shape_info["window_counts"].append(int(n_w))
            for layer in range(num_layers):
                vecs = []
                for w in range(n_w):
                    key = f"layer_{layer}_window_{w}"
                    if key not in keys:
                        shape_info["missing_keys"].append(key)
                        raise ValueError(f"Missing {key} in {feat_path}")
                    vecs.append(f.get_tensor(key))
                episode_feat = np.mean(vecs, axis=0)
                if feature_type == "mean":
                    flat_feat = np.asarray(episode_feat, dtype=np.float32)
                elif feature_type == "token_patch":
                    if episode_feat.ndim != 2:
                        raise ValueError(
                            f"Expected token-patch feature to be rank-2, got {episode_feat.shape} in {feat_path}"
                        )
                    if shape_info["patch_shape"] is None:
                        shape_info["patch_shape"] = list(episode_feat.shape)
                    flat_feat = np.asarray(episode_feat.reshape(-1), dtype=np.float32)
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")

                if X_all is None:
                    feature_dims = {layer_index: int(flat_feat.shape[0]) for layer_index in range(num_layers)}
                    feature_dims[layer] = int(flat_feat.shape[0])
                    X_all = {
                        layer_index: np.empty((len(episode_indices), feature_dims[layer_index]), dtype=np.float32)
                        for layer_index in range(num_layers)
                    }
                X_all[layer][row_index] = flat_feat

    if X_all is None:
        raise ValueError(f"No features loaded from {feature_root}")
    shape_info["feature_dims"] = {int(k): int(v.shape[1]) for k, v in X_all.items()}
    return X_all, shape_info


def evaluate_layer(features, targets, groups, output_dim, device, norm_mode="zscore"):
    n_unique = int(np.unique(groups).size)
    n_splits = min(CV_SPLITS, n_unique)
    if n_splits < 2:
        raise ValueError("Need at least two unique episode groups for GroupKFold")

    splitter = GroupKFold(n_splits=n_splits)
    fold_scores = []
    fold_best_lrs = []
    fold_best_wds = []

    for train_idx, val_idx in splitter.split(features, targets, groups):
        overlap = set(groups[train_idx].tolist()) & set(groups[val_idx].tolist())
        if overlap:
            raise ValueError(f"Group leakage detected: {sorted(list(overlap))[:5]}")

        y_train = np.asarray(targets[train_idx])
        y_val = np.asarray(targets[val_idx])
        if np.asarray(y_train).ndim == 1:
            if float(np.std(y_train)) < 1e-10 or float(np.std(y_val)) < 1e-10:
                fold_scores.append(0.0)
                fold_best_lrs.append(LR_GRID[0])
                fold_best_wds.append(WD_GRID[0])
                continue
        else:
            if float(np.std(y_train)) < 1e-10 or float(np.std(y_val)) < 1e-10:
                fold_scores.append(0.0)
                fold_best_lrs.append(LR_GRID[0])
                fold_best_wds.append(WD_GRID[0])
                continue

        cfg_results = fit_trainable_batched(
            features[train_idx],
            y_train,
            features[val_idx],
            y_val,
            output_dim=output_dim,
            lr_grid=LR_GRID,
            wd_grid=WD_GRID,
            device=device,
            norm_mode=norm_mode,
        )
        best_lr, best_wd, best_r2 = max(cfg_results, key=lambda item: item[2])
        fold_scores.append(float(best_r2))
        fold_best_lrs.append(float(best_lr))
        fold_best_wds.append(float(best_wd))

    return {
        "r2_mean": float(np.mean(fold_scores)),
        "r2_std": float(np.std(fold_scores)),
        "best_lr_mode": float(Counter(fold_best_lrs).most_common(1)[0][0]),
        "best_wd_mode": float(Counter(fold_best_wds).most_common(1)[0][0]),
        "n_folds": int(n_splits),
    }


def save_curve_plot(task, target, model, results_df, feature_type="mean", run_tag="phase1"):
    out_path = os.path.join(FIGURES_DIR, f"curve_{build_output_stem(task, target, model, feature_type, run_tag)}.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    df = results_df.sort_values("layer")
    ax.plot(df["layer"], df["r2_mean"], marker="o", linewidth=2)
    ax.fill_between(df["layer"], df["r2_mean"] - df["r2_std"], df["r2_mean"] + df["r2_std"], alpha=0.2)
    ax.axvline(8 if model == "large" else 13, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{task} / {target} / {model}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R^2")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_gt_distribution_plot(task, target_arrays):
    out_path = os.path.join(FIGURES_DIR, f"gt_distribution_{task}.png")
    n = len(target_arrays)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    idx = 0
    for name, arr in target_arrays.items():
        ax = axes[idx // cols, idx % cols]
        arr = np.asarray(arr)
        if arr.ndim == 1:
            ax.hist(arr, bins=min(20, max(5, len(np.unique(arr)))))
        else:
            flat = arr.reshape(arr.shape[0], -1)
            for d in range(flat.shape[1]):
                ax.hist(flat[:, d], bins=20, alpha=0.5, label=f"d{d}")
            ax.legend(fontsize=8)
        ax.set_title(name)
        idx += 1

    for j in range(idx, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_phase_report(task, model, summary_rows, sanity, run_tag="phase1"):
    path = os.path.join(RESULTS_DIR, f"EXPERIMENT_RESULTS_{run_tag}_{task}.md")
    lines = []
    lines.append(f"# Phase 1 Results: {task} / {model}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            f"| {row['target']} | {row['output_dim']} | {row['L0']:.4f} | {row['L8']:.4f} | "
            f"{row['peak_r2']:.4f} | {row['peak_layer']} | {row['last']:.4f} | {row['classification']} |"
        )
    lines.append("")
    lines.append("## Hypothesis checks")
    lines.append("")
    h1_targets = [r for r in summary_rows if r["target"] in {"ee_pos", "object_pos"}]
    h1_pass = all(r["L0"] >= 0.8 for r in h1_targets) if h1_targets else False
    h2_targets = [r for r in summary_rows if r["target"] in {"mass", "obj_friction", "surface_friction"}]
    h2_pass = any(r["classification"] == "PEZ-like" for r in h2_targets)
    lines.append(f"- H1 (controls high from L0): `{'PASS' if h1_pass else 'FAIL'}`")
    lines.append(f"- H2 (static physics PEZ-like): `{'PASS' if h2_pass else 'FAIL'}`")
    lines.append("")
    lines.append("## Sanity checks")
    lines.append("")
    for key, value in sanity.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Next-step decision")
    lines.append("")
    if not h1_pass:
        lines.append("- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.")
    elif h2_pass:
        lines.append("- Proceed to Strike with the same protocol. Push already yields a candidate PEZ-like static parameter.")
    else:
        lines.append("- Proceed to Strike, but treat Push static parameters as negative evidence under mean-pooled features.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="PEZ-style probing on PhysProbe")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--targets", nargs="+", default=[])
    parser.add_argument("--control-targets", nargs="+", default=[])
    parser.add_argument("--solver", choices=["trainable"], default="trainable")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--norm", choices=["zscore", "center", "none"], default="zscore")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-type", choices=["mean", "token_patch"], default="mean")
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--run-tag", default="phase1")
    args = parser.parse_args()

    ensure_dirs()

    if args.cv_splits != 5:
        raise ValueError("This Phase 1 driver is fixed to 5-fold GroupKFold")
    if args.solver != "trainable":
        raise ValueError("This Phase 1 driver is fixed to the trainable 20-HP sweep")

    requested_targets = list(dict.fromkeys(args.targets + args.control_targets))
    if not requested_targets:
        raise ValueError("Provide at least one target or control target")

    task_specs = TARGET_SPECS[args.task]
    unknown = [t for t in requested_targets if t not in task_specs]
    if unknown:
        raise ValueError(f"Unknown targets for {args.task}: {unknown}")

    device = args.device if torch.cuda.is_available() else "cpu"
    feature_root = resolve_feature_root(
        args.task,
        args.model,
        feature_type=args.feature_type,
        feature_root=args.feature_root,
    )
    episode_indices = list_feature_episodes(feature_root)
    if not episode_indices:
        raise ValueError(f"No safetensors found in {feature_root}")

    features_by_layer, feature_shape_info = load_episode_features(
        args.task,
        args.model,
        episode_indices,
        feature_type=args.feature_type,
        feature_root_override=args.feature_root,
    )
    groups = np.asarray(episode_indices, dtype=np.int64)
    targets = load_targets(args.task, episode_indices, requested_targets)

    sanity = {
        "feature_root": feature_root,
        "feature_type": args.feature_type,
        "run_tag": args.run_tag,
        "n_episodes": len(episode_indices),
        "num_layers": MODEL_CONFIGS[args.model]["num_layers"],
        "feature_dim_layer0": int(features_by_layer[0].shape[1]),
        "window_count_mode": Counter(feature_shape_info["window_counts"]).most_common(1)[0][0],
        "missing_feature_keys": len(feature_shape_info["missing_keys"]),
        "patch_shape": feature_shape_info.get("patch_shape"),
    }

    split_check = GroupKFold(n_splits=CV_SPLITS)
    overlap_zero = True
    for train_idx, val_idx in split_check.split(np.zeros((len(groups), 1)), np.zeros(len(groups)), groups):
        overlap = set(groups[train_idx].tolist()) & set(groups[val_idx].tolist())
        if overlap:
            overlap_zero = False
            break
    sanity["groupkfold_overlap_zero"] = bool(overlap_zero)

    gt_summary = {}
    for target_name, arr in targets.items():
        arr_np = np.asarray(arr)
        gt_summary[target_name] = {
            "shape": list(arr_np.shape),
            "var": float(np.var(arr_np)),
            "finite": bool(np.all(np.isfinite(arr_np))),
        }
        if float(np.var(arr_np)) < 1e-12:
            raise ValueError(f"Target {target_name} has zero variance")
    gt_plot_path = save_gt_distribution_plot(args.task, targets)
    sanity["gt_distribution_plot"] = gt_plot_path

    summary_rows = []
    for target_name in requested_targets:
        output_dim = int(task_specs[target_name]["output_dim"])
        layer_rows = []
        for layer in tqdm(range(MODEL_CONFIGS[args.model]["num_layers"]), desc=f"Probe [{target_name}]"):
            features = features_by_layer[layer]
            if not np.all(np.isfinite(features)):
                raise ValueError(f"NaN/Inf in features at layer {layer}")
            result = evaluate_layer(
                features,
                np.asarray(targets[target_name]),
                groups,
                output_dim=output_dim,
                device=device,
                norm_mode=args.norm,
            )
            row = {
                "task": args.task,
                "model": args.model,
                "target": target_name,
                "layer": layer,
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "best_lr": result["best_lr_mode"],
                "best_wd": result["best_wd_mode"],
                "output_dim": output_dim,
                "n_samples": len(groups),
            }
            layer_rows.append(row)

        results_df = pd.DataFrame(layer_rows)
        csv_path = os.path.join(
            RESULTS_DIR,
            f"probe_{build_output_stem(args.task, target_name, args.model, args.feature_type, args.run_tag)}.csv",
        )
        results_df.to_csv(csv_path, index=False)
        fig_path = save_curve_plot(
            args.task,
            target_name,
            args.model,
            results_df,
            feature_type=args.feature_type,
            run_tag=args.run_tag,
        )

        r2_curve = results_df.sort_values("layer")["r2_mean"].to_numpy(dtype=np.float64)
        peak_layer = int(np.argmax(r2_curve))
        summary_rows.append(
            {
                "task": args.task,
                "model": args.model,
                "target": target_name,
                "output_dim": output_dim,
                "L0": float(r2_curve[0]),
                "L8": float(r2_curve[8]) if len(r2_curve) > 8 else float(r2_curve[-1]),
                "peak_r2": float(np.max(r2_curve)),
                "peak_layer": peak_layer,
                "last": float(r2_curve[-1]),
                "classification": classify_curve(r2_curve),
                "csv": csv_path,
                "figure": fig_path,
            }
        )

    control_rows = [r for r in summary_rows if r["target"] in args.control_targets]
    sanity["control_targets_l0_ge_0_8"] = bool(control_rows and all(r["L0"] >= 0.8 for r in control_rows))
    sanity["gt_summary"] = gt_summary

    verdict_path = os.path.join(RESULTS_DIR, f"verdict_{args.run_tag}_{args.task}.json")
    with open(verdict_path, "w") as f:
        json.dump({"task": args.task, "model": args.model, "targets": summary_rows}, f, indent=2)

    sanity_path = os.path.join(RESULTS_DIR, f"sanity_{args.run_tag}_{args.task}.json")
    with open(sanity_path, "w") as f:
        json.dump(sanity, f, indent=2)

    report_path = write_phase_report(args.task, args.model, summary_rows, sanity, run_tag=args.run_tag)

    print(f"Wrote verdict: {verdict_path}")
    print(f"Wrote sanity: {sanity_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
