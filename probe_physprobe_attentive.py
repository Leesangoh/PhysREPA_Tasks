#!/usr/bin/env python3
"""Attentive probing on PhysProbe token-patch caches.

Uses the same episode-level targets and GroupKFold-by-episode protocol as
`probe_physprobe.py`, but replaces flatten+linear readout with an attentive
pooler plus linear regression head.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys

sys.path.insert(0, "/home/solee/vjepa2")
sys.path.insert(0, "/home/solee/vjepa2/src")
from src.models.attentive_pooler import AttentivePooler

from probe_physprobe import (
    ARTIFACTS_ROOT,
    CV_RANDOM_SEED,
    CV_SPLITS,
    FIGURES_DIR,
    LR_GRID,
    MAX_EPOCHS,
    MODEL_CONFIGS,
    PATIENCE,
    RESULTS_DIR,
    WD_GRID,
    build_output_stem,
    compute_r2,
    ensure_dirs,
    load_targets,
    normalize_train_val,
    sanitize_target_name,
)


class AttentiveRegressor(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_heads: int = 16,
        depth: int = 4,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=False,
        )
        self.linear = torch.nn.Linear(embed_dim, output_dim, bias=True)

    def forward(self, x):
        pooled = self.pooler(x).squeeze(1)
        return self.linear(pooled)


def load_episode_patch_tokens(task: str, model: str, episode_indices: list[int], feature_root: str, layer_indices=None):
    cfg = MODEL_CONFIGS[model]
    num_layers = cfg["num_layers"]
    embed_dim = cfg["dim"]
    layer_indices = list(range(num_layers)) if layer_indices is None else list(layer_indices)

    shape_info = {
        "feature_root": feature_root,
        "feature_type": "token_patch",
        "num_layers": num_layers,
        "loaded_layers": layer_indices,
        "embed_dim": embed_dim,
        "episodes": len(episode_indices),
        "window_counts": [],
        "missing_keys": [],
        "patch_shape": None,
    }

    X_all = None

    for row_index, ep_idx in enumerate(tqdm(episode_indices, desc=f"Load attentive patch tokens [{task}/{model}]")):
        feat_path = os.path.join(feature_root, f"{ep_idx:06d}.safetensors")
        with safe_open(feat_path, framework="numpy") as f:
            keys = set(f.keys())
            window_starts = f.get_tensor("window_starts").astype(np.int64)
            n_w = len(window_starts)
            shape_info["window_counts"].append(int(n_w))

            for layer in layer_indices:
                tokens = []
                for w in range(n_w):
                    key = f"layer_{layer}_window_{w}"
                    if key not in keys:
                        shape_info["missing_keys"].append(key)
                        raise ValueError(f"Missing {key} in {feat_path}")
                    tokens.append(f.get_tensor(key))

                episode_tokens = np.mean(tokens, axis=0).astype(np.float32)
                if episode_tokens.ndim != 2:
                    raise ValueError(f"Expected (n_patches, D), got {episode_tokens.shape} in {feat_path}")
                if shape_info["patch_shape"] is None:
                    shape_info["patch_shape"] = list(episode_tokens.shape)

                if X_all is None:
                    n_patches = int(episode_tokens.shape[0])
                    X_all = {
                        layer_idx: np.empty((len(episode_indices), n_patches, embed_dim), dtype=np.float32)
                        for layer_idx in layer_indices
                    }
                X_all[layer][row_index] = episode_tokens

    if X_all is None:
        raise ValueError(f"No attentive tokens loaded from {feature_root}")
    return X_all, shape_info


def fit_attentive_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dim: int,
    device: str,
    lr: float,
    wd: float,
    num_probe_blocks: int = 4,
    num_heads: int = 16,
    batch_size: int = 16,
    num_epochs: int = 20,
    patience_limit: int = 5,
    norm_mode: str = "zscore",
):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    if norm_mode == "none":
        feat_std = np.ones_like(feat_mean)
    elif norm_mode == "center":
        feat_std = np.ones_like(feat_mean)
    else:
        feat_std = X_train.std(axis=(0, 1), keepdims=True)
        feat_std[feat_std < 1e-6] = 1.0

    if norm_mode == "center":
        X_train = X_train - feat_mean
        X_val = X_val - feat_mean
    elif norm_mode == "zscore":
        X_train = (X_train - feat_mean) / feat_std
        X_val = (X_val - feat_mean) / feat_std

    y_train_std, y_val_std, y_mean, y_std = normalize_train_val(y_train, y_val, norm_mode)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_std)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(X_val).to(device)

    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)

    model = AttentiveRegressor(
        embed_dim=X_train.shape[-1],
        output_dim=output_dim,
        num_heads=num_heads,
        depth=num_probe_blocks,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_r2 = -1e9
    best_pred = None
    best_state = None
    patience = 0

    for _epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = torch.nn.functional.mse_loss(pred, batch_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_std = model(val_x).cpu().numpy()
        pred = pred_std * y_std + y_mean
        val_r2 = compute_r2(y_val, pred)
        if val_r2 > best_r2:
            best_r2 = float(val_r2)
            best_pred = pred.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break

    if best_pred is None:
        raise RuntimeError("Attentive probe failed to produce validation predictions")

    return best_r2


def evaluate_layer_attentive(features, targets, groups, output_dim, device, norm_mode="zscore"):
    n_unique = int(np.unique(groups).size)
    n_splits = min(CV_SPLITS, n_unique)
    splitter = GroupKFold(n_splits=n_splits)

    fold_scores = []
    fold_best_lrs = []
    fold_best_wds = []

    configs = list(itertools.product(LR_GRID, WD_GRID))

    for train_idx, val_idx in splitter.split(features, targets, groups):
        overlap = set(groups[train_idx].tolist()) & set(groups[val_idx].tolist())
        if overlap:
            raise ValueError(f"Group leakage detected: {sorted(list(overlap))[:5]}")

        y_train = np.asarray(targets[train_idx])
        y_val = np.asarray(targets[val_idx])
        if float(np.std(y_train)) < 1e-10 or float(np.std(y_val)) < 1e-10:
            fold_scores.append(0.0)
            fold_best_lrs.append(LR_GRID[0])
            fold_best_wds.append(WD_GRID[0])
            continue

        best = None
        for lr, wd in configs:
            r2 = fit_attentive_probe(
                features[train_idx],
                y_train,
                features[val_idx],
                y_val,
                output_dim=output_dim,
                device=device,
                lr=lr,
                wd=wd,
                norm_mode=norm_mode,
            )
            if best is None or r2 > best[2]:
                best = (lr, wd, r2)

        best_lr, best_wd, best_r2 = best
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


def classify_curve(r2_per_layer):
    r2_per_layer = list(map(float, r2_per_layer))
    L0 = r2_per_layer[0]
    peak_r2 = max(r2_per_layer)
    peak_layer = int(np.argmax(r2_per_layer))
    fractional_depth = peak_layer / len(r2_per_layer)

    if L0 >= 0.8:
        return "always-linear"
    if peak_r2 < 0.3:
        return "never-linear"
    if peak_r2 >= 0.5 and 0.2 <= fractional_depth <= 0.6:
        return "PEZ-like"
    return "intermediate"


def save_curve_plot(task, target, model, results_df, run_tag):
    out_path = os.path.join(FIGURES_DIR, f"curve_{build_output_stem(task, target, model, 'token_patch_attentive', run_tag)}.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    df = results_df.sort_values("layer")
    ax.plot(df["layer"], df["r2_mean"], marker="o", linewidth=2)
    ax.fill_between(df["layer"], df["r2_mean"] - df["r2_std"], df["r2_mean"] + df["r2_std"], alpha=0.2)
    ax.axvline(8 if model == "large" else 13, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{task} / {target} / {model} / attentive")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R^2")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", choices=["large", "giant"], required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-tag", default="attentive")
    parser.add_argument("--norm", choices=["zscore", "center", "none"], default="zscore")
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()

    episode_indices = sorted(
        int(os.path.splitext(os.path.basename(p))[0])
        for p in os.listdir(args.feature_root)
        if p.endswith(".safetensors")
    )
    layer_indices = args.layers if args.layers else list(range(MODEL_CONFIGS[args.model]["num_layers"]))
    X_all, shape_info = load_episode_patch_tokens(
        args.task, args.model, episode_indices, args.feature_root, layer_indices=layer_indices
    )
    targets = load_targets(args.task, episode_indices, args.targets)
    groups = np.asarray(episode_indices)

    summary_rows = []
    for target_name in args.targets:
        output_dim = int(targets[target_name].shape[1] if np.asarray(targets[target_name]).ndim > 1 else 1)
        layer_rows = []
        for layer in tqdm(layer_indices, desc=f"Attentive probe [{target_name}]"):
            metrics = evaluate_layer_attentive(
                X_all[layer],
                np.asarray(targets[target_name]),
                groups,
                output_dim=output_dim,
                device=args.device,
                norm_mode=args.norm,
            )
            layer_rows.append(
                {
                    "task": args.task,
                    "model": args.model,
                    "target": target_name,
                    "layer": layer,
                    "r2_mean": metrics["r2_mean"],
                    "r2_std": metrics["r2_std"],
                    "best_lr": metrics["best_lr_mode"],
                    "best_wd": metrics["best_wd_mode"],
                    "n_folds": metrics["n_folds"],
                }
            )
        df = pd.DataFrame(layer_rows)
        csv_path = os.path.join(
            RESULTS_DIR, f"probe_{build_output_stem(args.task, target_name, args.model, 'token_patch_attentive', args.run_tag)}.csv"
        )
        df.to_csv(csv_path, index=False)
        fig_path = save_curve_plot(args.task, target_name, args.model, df, args.run_tag)
        r2 = df["r2_mean"].tolist()
        summary_rows.append(
            {
                "task": args.task,
                "model": args.model,
                "target": target_name,
                "output_dim": output_dim,
                "L0": float(r2[0]),
                "L8": float(r2[8]),
                "peak_r2": float(max(r2)),
                "peak_layer": int(np.argmax(r2)),
                "last": float(r2[-1]),
                "classification": classify_curve(r2),
                "csv": csv_path,
                "figure": fig_path,
            }
        )

    verdict_path = os.path.join(RESULTS_DIR, f"verdict_{args.run_tag}_{args.task}.json")
    with open(verdict_path, "w") as f:
        json.dump({"task": args.task, "model": args.model, "targets": summary_rows, "shape_info": shape_info}, f, indent=2)
    print(f"Wrote verdict: {verdict_path}")


if __name__ == "__main__":
    main()
