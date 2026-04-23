#!/usr/bin/env python3
"""Window-level event probing with surrogate or native labels.

Current implementation is optimized for Strike:
- existing token-patch cache is reused
- per-window labels are derived from parquet kinematics or native contact-force
- spatial tokens are mean-pooled within each temporal_last_patch window to keep
  the window-level dataset tractable

Outputs:
- artifacts/results/probe_events_{task}_{target}_{model}_{run_tag}.csv
- artifacts/figures/curve_events_{task}_{target}_{model}_{run_tag}.png
- artifacts/results/event_probe_verdict_{task}_{run_tag}.json
- artifacts/results/event_probe_report_{task}_{run_tag}.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from probe_physprobe import (
    ARTIFACTS_ROOT,
    CV_RANDOM_SEED,
    CV_SPLITS,
    DATA_BASE,
    FIGURES_DIR,
    LR_GRID,
    MAX_EPOCHS,
    MODEL_CONFIGS,
    PATIENCE,
    RESULTS_DIR,
    WD_GRID,
    ensure_dirs,
    find_parquet_files,
    fit_trainable_batched,
    list_feature_episodes,
    load_static_meta,
    normalize_train_val,
)


EVENT_CONFIGS = {
    "strike": {
        "mass_key": "object_0_mass",
        "acc_col": "physics_gt.object_acceleration",
        "native_force_col": "physics_gt.contact_force",
        "native_flag_col": "physics_gt.contact_flag",
        "window_len": 16,
    },
}


def compute_auc(y_true: np.ndarray, logits: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, logits))
    except Exception:
        return 0.5


def fit_binary_trainable_batched(
    X_train,
    y_train,
    X_val,
    y_val,
    lr_grid,
    wd_grid,
    device,
    max_epochs=MAX_EPOCHS,
    patience_limit=PATIENCE,
    norm_mode="zscore",
    probe_seed=CV_RANDOM_SEED,
):
    configs = [(lr, wd) for lr in lr_grid for wd in wd_grid]
    n_configs = len(configs)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    X_train_std, X_val_std, _, _ = normalize_train_val(X_train, X_val, norm_mode)
    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val, dtype=torch.float32, device=device)

    input_dim = X_tr.shape[1]
    output_dim = 1

    torch.manual_seed(probe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(probe_seed)
    template = torch.nn.Linear(input_dim, output_dim, bias=True)
    W = template.weight.data.T.contiguous().unsqueeze(0).expand(n_configs, input_dim, output_dim).contiguous().to(device)
    b = template.bias.data.clone().unsqueeze(0).expand(n_configs, output_dim).contiguous().to(device)
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
        logits_tr = torch.einsum("nd,cdo->cno", X_tr, W) + b.unsqueeze(1)
        loss_per_cfg = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tr,
            y_tr.unsqueeze(0).expand_as(logits_tr),
            reduction="none",
        ).mean(dim=(1, 2))
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
            logits_va = torch.einsum("nd,cdo->cno", X_va, W) + b.unsqueeze(1)
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_va,
                y_va.unsqueeze(0).expand_as(logits_va),
                reduction="none",
            ).mean(dim=(1, 2))
            improved = val_loss + 1e-8 < best_val_loss
            best_val_loss = torch.where(improved, val_loss, best_val_loss)
            best_W = torch.where(improved[:, None, None], W.detach(), best_W)
            best_b = torch.where(improved[:, None], b.detach(), best_b)
            patience = torch.where(improved, torch.zeros_like(patience), patience + 1)
            active = active & (patience < patience_limit)
            if not active.any():
                break

    with torch.no_grad():
        logits_tr_best = torch.einsum("nd,cdo->cno", X_tr, best_W) + best_b.unsqueeze(1)
        logits_va_best = torch.einsum("nd,cdo->cno", X_va, best_W) + best_b.unsqueeze(1)
    logits_tr_np = logits_tr_best.squeeze(-1).cpu().numpy()
    logits_va_np = logits_va_best.squeeze(-1).cpu().numpy()

    results = []
    for cfg_index, (lr, wd) in enumerate(configs):
        train_auc = compute_auc(y_train[:, 0], logits_tr_np[cfg_index])
        val_auc = compute_auc(y_val[:, 0], logits_va_np[cfg_index])
        train_pred = (logits_tr_np[cfg_index] > 0.0).astype(np.int64)
        val_pred = (logits_va_np[cfg_index] > 0.0).astype(np.int64)
        results.append(
            (
                float(lr),
                float(wd),
                val_auc,
                train_auc,
                float(balanced_accuracy_score(y_val[:, 0], val_pred)),
                float(balanced_accuracy_score(y_train[:, 0], train_pred)),
            )
        )
    return results


def derive_strike_event_windows(
    episode_indices: list[int],
    feature_root: str,
    data_base: str = DATA_BASE,
    label_mode: str = "surrogate",
):
    cfg = EVENT_CONFIGS["strike"]
    static_meta = load_static_meta("strike", data_base=data_base) if label_mode == "surrogate" else None
    parquet_map = find_parquet_files("strike", data_base=data_base)

    class_samples = []
    reg_samples = []

    for ep_idx in tqdm(episode_indices, desc="Derive strike event windows"):
        feat_path = os.path.join(feature_root, f"{ep_idx:06d}.safetensors")
        if not os.path.exists(feat_path):
            continue
        with safe_open(feat_path, framework="numpy") as f:
            window_starts = f.get_tensor("window_starts").astype(np.int64)

        if label_mode == "surrogate":
            df = pd.read_parquet(parquet_map[ep_idx], columns=[cfg["acc_col"]])
            acc = np.stack(df[cfg["acc_col"]].values).astype(np.float64)
            acc_mag = np.linalg.norm(acc, axis=1)
            med = float(np.median(acc_mag))
            mad = float(np.median(np.abs(acc_mag - med)))
            threshold = med + 5.0 * max(mad, 1e-6)
            pos_frame = acc_mag > threshold
            mass = float(static_meta[ep_idx][cfg["mass_key"]])
            force_signal = mass * acc_mag
        elif label_mode == "native":
            df = pd.read_parquet(parquet_map[ep_idx], columns=[cfg["native_force_col"], cfg["native_flag_col"]])
            native_force = np.linalg.norm(np.stack(df[cfg["native_force_col"]].values).astype(np.float64), axis=1)
            native_flag = np.stack(df[cfg["native_flag_col"]].values).astype(np.float64).reshape(-1) > 0.0
            pos_frame = native_flag | (native_force > 1e-8)
            force_signal = native_force
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        win_event = []
        win_force = []
        T = len(force_signal)
        for start in window_starts:
            s = int(start)
            e = min(s + cfg["window_len"], T)
            event = bool(np.any(pos_frame[s:e]))
            force = float(np.max(force_signal[s:e]))
            win_event.append(event)
            win_force.append(force)

        pos_candidates = [i for i, flag in enumerate(win_event) if flag]
        neg_candidates = [i for i, flag in enumerate(win_event) if not flag]
        if not pos_candidates or not neg_candidates:
            continue

        pos_idx = max(pos_candidates, key=lambda i: win_force[i])
        neg_idx = min(neg_candidates, key=lambda i: win_force[i])

        class_samples.append(
            {
                "episode": ep_idx,
                "window": int(pos_idx),
                "label": 1,
                "force_proxy": float(win_force[pos_idx]),
            }
        )
        class_samples.append(
            {
                "episode": ep_idx,
                "window": int(neg_idx),
                "label": 0,
                "force_proxy": float(win_force[neg_idx]),
            }
        )
        reg_samples.append(
            {
                "episode": ep_idx,
                "window": int(pos_idx),
                "force_proxy": float(win_force[pos_idx]),
            }
        )

    return class_samples, reg_samples


def load_window_features(task: str, model: str, samples: list[dict], feature_root: str):
    num_layers = MODEL_CONFIGS[model]["num_layers"]
    features_by_layer = None
    patch_shape = None
    for row_idx, sample in enumerate(tqdm(samples, desc=f"Load event features [{task}/{model}]")):
        feat_path = os.path.join(feature_root, f"{sample['episode']:06d}.safetensors")
        with safe_open(feat_path, framework="numpy") as f:
            for layer in range(num_layers):
                key = f"layer_{layer}_window_{sample['window']}"
                patch = f.get_tensor(key)
                if patch.ndim == 1:
                    # Some older V-JEPA caches already store a spatially pooled
                    # per-window vector. That is equivalent to the window-level
                    # mean readout used here, so accept it directly.
                    vec = np.asarray(patch, dtype=np.float32)
                    current_shape = [int(patch.shape[0])]
                elif patch.ndim == 2:
                    vec = np.asarray(patch.mean(axis=0), dtype=np.float32)
                    current_shape = list(patch.shape)
                else:
                    raise ValueError(f"Expected rank-1 or rank-2 window feature, got {patch.shape} in {key}")
                if patch_shape is None:
                    patch_shape = current_shape
                if features_by_layer is None:
                    features_by_layer = {
                        layer_idx: np.empty((len(samples), vec.shape[0]), dtype=np.float32)
                        for layer_idx in range(num_layers)
                    }
                features_by_layer[layer][row_idx] = vec
    return features_by_layer, patch_shape


def evaluate_binary_layer(features, labels, groups, device, norm_mode="zscore", probe_seed=CV_RANDOM_SEED):
    splitter = GroupKFold(n_splits=min(CV_SPLITS, int(np.unique(groups).size)))
    fold_aucs = []
    fold_train_aucs = []
    fold_baccs = []
    fold_best_lrs = []
    fold_best_wds = []
    for train_idx, val_idx in splitter.split(features, labels, groups):
        overlap = set(groups[train_idx].tolist()) & set(groups[val_idx].tolist())
        if overlap:
            raise ValueError(f"Group leakage detected: {sorted(list(overlap))[:5]}")
        cfg_results = fit_binary_trainable_batched(
            features[train_idx],
            labels[train_idx],
            features[val_idx],
            labels[val_idx],
            lr_grid=LR_GRID,
            wd_grid=WD_GRID,
            device=device,
            norm_mode=norm_mode,
            probe_seed=probe_seed,
        )
        best_lr, best_wd, best_auc, best_train_auc, best_bacc, _ = max(cfg_results, key=lambda x: x[2])
        fold_aucs.append(float(best_auc))
        fold_train_aucs.append(float(best_train_auc))
        fold_baccs.append(float(best_bacc))
        fold_best_lrs.append(float(best_lr))
        fold_best_wds.append(float(best_wd))
    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "train_auc_mean": float(np.mean(fold_train_aucs)),
        "train_auc_std": float(np.std(fold_train_aucs)),
        "bacc_mean": float(np.mean(fold_baccs)),
        "bacc_std": float(np.std(fold_baccs)),
        "best_lr_mode": float(Counter(fold_best_lrs).most_common(1)[0][0]),
        "best_wd_mode": float(Counter(fold_best_wds).most_common(1)[0][0]),
        "fold_val_aucs": [float(x) for x in fold_aucs],
        "fold_train_aucs": [float(x) for x in fold_train_aucs],
    }


def save_curve(df: pd.DataFrame, metric_col: str, metric_std_col: str, ylabel: str, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sdf = df.sort_values("layer")
    ax.plot(sdf["layer"], sdf[metric_col], marker="o", linewidth=2)
    ax.fill_between(sdf["layer"], sdf[metric_col] - sdf[metric_std_col], sdf[metric_col] + sdf[metric_std_col], alpha=0.2)
    ax.axvline(8, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Window-level surrogate/native contact probing")
    parser.add_argument("--task", choices=["strike"], default="strike")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="large")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--data-base", default=DATA_BASE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-tag", default="phase3_events")
    parser.add_argument("--norm", choices=["zscore", "center", "none"], default="zscore")
    parser.add_argument("--probe-seed", type=int, default=CV_RANDOM_SEED)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--label-mode", choices=["surrogate", "native"], default="surrogate")
    args = parser.parse_args()

    ensure_dirs()
    device = args.device if torch.cuda.is_available() else "cpu"
    feature_root = os.path.join(args.feature_root, args.task)
    episode_indices = list_feature_episodes(feature_root)
    if not episode_indices:
        raise ValueError(f"No safetensors found in {feature_root}")
    if args.episode_limit is not None:
        episode_indices = episode_indices[: args.episode_limit]

    class_samples, reg_samples = derive_strike_event_windows(
        episode_indices,
        feature_root,
        data_base=args.data_base,
        label_mode=args.label_mode,
    )
    if not class_samples or not reg_samples:
        raise ValueError("No event samples derived")

    class_features, patch_shape = load_window_features(args.task, args.model, class_samples, feature_root)
    reg_features, _ = load_window_features(args.task, args.model, reg_samples, feature_root)

    class_labels = np.asarray([row["label"] for row in class_samples], dtype=np.int64)
    class_groups = np.asarray([row["episode"] for row in class_samples], dtype=np.int64)
    reg_targets = np.asarray([row["force_proxy"] for row in reg_samples], dtype=np.float64)
    reg_groups = np.asarray([row["episode"] for row in reg_samples], dtype=np.int64)

    binary_target = "contact_happening" if args.label_mode == "surrogate" else "contact_happening_native"
    reg_target = "contact_force_proxy" if args.label_mode == "surrogate" else "contact_force_native"

    binary_rows = []
    reg_rows = []
    num_layers = MODEL_CONFIGS[args.model]["num_layers"]
    for layer in tqdm(range(num_layers), desc=f"Probe [{binary_target}]"):
        result = evaluate_binary_layer(
            class_features[layer],
            class_labels,
            class_groups,
            device=device,
            norm_mode=args.norm,
            probe_seed=args.probe_seed,
        )
        binary_rows.append(
            {
                "task": args.task,
                "target": binary_target,
                "model": args.model,
                "probe_seed": int(args.probe_seed),
                "layer": layer,
                **result,
            }
        )

    for layer in tqdm(range(num_layers), desc=f"Probe [{reg_target}]"):
        splitter = GroupKFold(n_splits=min(CV_SPLITS, int(np.unique(reg_groups).size)))
        fold_scores = []
        fold_train_scores = []
        fold_best_lrs = []
        fold_best_wds = []
        for train_idx, val_idx in splitter.split(reg_features[layer], reg_targets, reg_groups):
            cfg_results = fit_trainable_batched(
                reg_features[layer][train_idx],
                reg_targets[train_idx],
                reg_features[layer][val_idx],
                reg_targets[val_idx],
                output_dim=1,
                lr_grid=LR_GRID,
                wd_grid=WD_GRID,
                device=device,
                norm_mode=args.norm,
                probe_seed=args.probe_seed,
            )
            best_lr, best_wd, best_r2, best_train_r2 = max(cfg_results, key=lambda item: item[2])
            fold_scores.append(float(best_r2))
            fold_train_scores.append(float(best_train_r2))
            fold_best_lrs.append(float(best_lr))
            fold_best_wds.append(float(best_wd))
        reg_rows.append(
            {
                "task": args.task,
                "target": reg_target,
                "model": args.model,
                "probe_seed": int(args.probe_seed),
                "layer": layer,
                "r2_mean": float(np.mean(fold_scores)),
                "r2_std": float(np.std(fold_scores)),
                "train_r2_mean": float(np.mean(fold_train_scores)),
                "train_r2_std": float(np.std(fold_train_scores)),
                "best_lr": float(Counter(fold_best_lrs).most_common(1)[0][0]),
                "best_wd": float(Counter(fold_best_wds).most_common(1)[0][0]),
                "fold_val_r2s": json.dumps([float(x) for x in fold_scores]),
                "fold_train_r2s": json.dumps([float(x) for x in fold_train_scores]),
            }
        )

    binary_df = pd.DataFrame(binary_rows)
    reg_df = pd.DataFrame(reg_rows)

    binary_csv = os.path.join(RESULTS_DIR, f"probe_events_{args.task}_{binary_target}_{args.model}_{args.run_tag}.csv")
    reg_csv = os.path.join(RESULTS_DIR, f"probe_events_{args.task}_{reg_target}_{args.model}_{args.run_tag}.csv")
    binary_df.to_csv(binary_csv, index=False)
    reg_df.to_csv(reg_csv, index=False)

    binary_fig = os.path.join(FIGURES_DIR, f"curve_events_{args.task}_{binary_target}_{args.model}_{args.run_tag}.png")
    reg_fig = os.path.join(FIGURES_DIR, f"curve_events_{args.task}_{reg_target}_{args.model}_{args.run_tag}.png")
    save_curve(binary_df, "auc_mean", "auc_std", "AUC", binary_fig, f"{args.task} / {binary_target} / {args.model}")
    save_curve(reg_df, "r2_mean", "r2_std", "R^2", reg_fig, f"{args.task} / {reg_target} / {args.model}")

    binary_curve = binary_df.sort_values("layer")["auc_mean"].to_numpy(dtype=np.float64)
    reg_curve = reg_df.sort_values("layer")["r2_mean"].to_numpy(dtype=np.float64)
    summary = {
        "task": args.task,
        "model": args.model,
        "run_tag": args.run_tag,
        "label_mode": args.label_mode,
        "feature_root": feature_root,
        "patch_shape": patch_shape,
        "n_event_class_samples": int(len(class_samples)),
        "n_event_reg_samples": int(len(reg_samples)),
        "positive_rate": float(class_labels.mean()),
        binary_target: {
            "L0": float(binary_curve[0]),
            "L8": float(binary_curve[8]),
            "peak_auc": float(binary_curve.max()),
            "peak_layer": int(binary_curve.argmax()),
            "last": float(binary_curve[-1]),
            "csv": binary_csv,
            "figure": binary_fig,
        },
        reg_target: {
            "L0": float(reg_curve[0]),
            "L8": float(reg_curve[8]),
            "peak_r2": float(reg_curve.max()),
            "peak_layer": int(reg_curve.argmax()),
            "last": float(reg_curve[-1]),
            "csv": reg_csv,
            "figure": reg_fig,
        },
    }

    verdict_path = os.path.join(RESULTS_DIR, f"event_probe_verdict_{args.task}_{args.run_tag}.json")
    with open(verdict_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(RESULTS_DIR, f"event_probe_report_{args.task}_{args.run_tag}.md")
    with open(report_path, "w") as f:
        f.write(f"# Event Probe Report: {args.task} / {args.model}\n\n")
        f.write(f"- label_mode: `{args.label_mode}`\n")
        f.write(f"- feature_root: `{feature_root}`\n")
        f.write(f"- patch_shape: `{patch_shape}`\n")
        f.write(f"- n_event_class_samples: `{len(class_samples)}`\n")
        f.write(f"- positive_rate: `{class_labels.mean():.3f}`\n")
        f.write(f"- {binary_target} peak AUC: `{binary_curve.max():.4f} @ L{int(binary_curve.argmax())}`\n")
        f.write(f"- {reg_target} peak R^2: `{reg_curve.max():.4f} @ L{int(reg_curve.argmax())}`\n")

    print(f"Wrote verdict: {verdict_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
