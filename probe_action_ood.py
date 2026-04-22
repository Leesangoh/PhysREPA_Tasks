#!/usr/bin/env python3
"""Offline physics-OOD action-chunk regression on compact frozen features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
import torch
from torch import nn

from probe_physprobe import (
    ARTIFACTS_ROOT,
    DATA_BASE,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
)


WINDOW_LEN = 16


@dataclass
class RepSpec:
    name: str
    model: str
    layer: int


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def parse_rep(spec: str) -> RepSpec:
    # name=model:layer
    name, rhs = spec.split("=")
    model, layer = rhs.split(":")
    return RepSpec(name=name, model=model, layer=int(layer))


def load_task_meta(task: str, episode_ids: list[int]):
    meta_path = os.path.join(DATA_BASE, task, "meta", "episodes.jsonl")
    meta = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            if ep in episode_ids:
                meta[ep] = rec
    return meta


def build_physics_split(task: str, meta: dict[int, dict], split_seed: int):
    if task == "push":
        required_keys = ["object_0_mass", "object_0_dynamic_friction", "surface_dynamic_friction"]
    elif task == "drawer":
        required_keys = ["drawer_joint_damping"]
    else:
        raise ValueError(f"Unsupported task for OOD split: {task}")

    episode_ids = [ep for ep in sorted(meta) if all(k in meta[ep] for k in required_keys)]
    q_lo, q_hi = 0.15, 0.85
    if task == "push":
        mass = np.array([float(meta[ep]["object_0_mass"]) for ep in episode_ids], dtype=np.float64)
        obj_dyn = np.array([float(meta[ep]["object_0_dynamic_friction"]) for ep in episode_ids], dtype=np.float64)
        surf_dyn = np.array([float(meta[ep]["surface_dynamic_friction"]) for ep in episode_ids], dtype=np.float64)
        thresholds = {
            "mass_lo": float(np.quantile(mass, q_lo)),
            "mass_hi": float(np.quantile(mass, q_hi)),
            "obj_dyn_lo": float(np.quantile(obj_dyn, q_lo)),
            "obj_dyn_hi": float(np.quantile(obj_dyn, q_hi)),
            "surf_dyn_lo": float(np.quantile(surf_dyn, q_lo)),
            "surf_dyn_hi": float(np.quantile(surf_dyn, q_hi)),
        }
    elif task == "drawer":
        damping = np.array([float(meta[ep]["drawer_joint_damping"]) for ep in episode_ids], dtype=np.float64)
        thresholds = {
            "drawer_joint_damping_lo": float(np.quantile(damping, q_lo)),
            "drawer_joint_damping_hi": float(np.quantile(damping, q_hi)),
        }

    central = []
    ood = []
    for ep in episode_ids:
        rec = meta[ep]
        if task == "push":
            in_band = (
                thresholds["mass_lo"] <= float(rec["object_0_mass"]) <= thresholds["mass_hi"]
                and thresholds["obj_dyn_lo"] <= float(rec["object_0_dynamic_friction"]) <= thresholds["obj_dyn_hi"]
                and thresholds["surf_dyn_lo"] <= float(rec["surface_dynamic_friction"]) <= thresholds["surf_dyn_hi"]
            )
        elif task == "drawer":
            in_band = (
                thresholds["drawer_joint_damping_lo"]
                <= float(rec["drawer_joint_damping"])
                <= thresholds["drawer_joint_damping_hi"]
            )
        if in_band:
            central.append(ep)
        else:
            ood.append(ep)

    rng = random.Random(split_seed)
    rng.shuffle(central)
    n_total = len(central)
    n_train = int(round(n_total * 0.70))
    n_val = int(round(n_total * 0.15))
    train_eps = sorted(central[:n_train])
    val_eps = sorted(central[n_train : n_train + n_val])
    iid_eps = sorted(central[n_train + n_val :])
    return {
        "train_eps": train_eps,
        "val_eps": val_eps,
        "iid_eps": iid_eps,
        "ood_eps": sorted(ood),
        "thresholds": thresholds,
        "counts": {
            "eligible_total": len(episode_ids),
            "central_total": len(central),
            "ood_total": len(ood),
            "train": len(train_eps),
            "val": len(val_eps),
            "iid_test": len(iid_eps),
        },
    }


def load_action_cache(task: str, episode_ids: list[int]) -> dict[int, np.ndarray]:
    action_cache = {}
    for ep in episode_ids:
        path = os.path.join(DATA_BASE, task, "data", f"chunk-{ep // 1000:03d}", f"episode_{ep:06d}.parquet")
        df = pd.read_parquet(path, columns=["action"])
        action_cache[ep] = np.stack(df["action"].values).astype(np.float32)
    return action_cache


def collect_samples(feature_root: str, layer: int, episode_ids: list[int], action_cache: dict[int, np.ndarray], chunk_len: int):
    xs = []
    ys = []
    eps = []
    n_missing = 0
    for ep in episode_ids:
        feat_path = os.path.join(feature_root, f"{ep:06d}.safetensors")
        if not os.path.exists(feat_path):
            n_missing += 1
            continue
        with safe_open(feat_path, framework="numpy") as f:
            window_starts = f.get_tensor("window_starts").astype(np.int64)
            feats = f.get_tensor(f"layer_{layer}").astype(np.float32)
        actions = action_cache[ep]
        for w_idx, start in enumerate(window_starts):
            target_start = int(start) + WINDOW_LEN
            target_end = target_start + chunk_len
            if target_end > len(actions):
                continue
            xs.append(feats[w_idx])
            ys.append(actions[target_start:target_end].reshape(-1))
            eps.append(ep)
    if not xs:
        raise ValueError(f"No usable samples for layer {layer} under {feature_root}")
    return np.stack(xs), np.stack(ys), np.asarray(eps, dtype=np.int64), n_missing


class ActionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (other - mean) / std, mean, std


def train_one(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_iid,
    Y_iid,
    X_ood,
    Y_ood,
    *,
    device: str,
    seed: int,
    hidden_dim: int,
    batch_size: int,
):
    set_seed(seed)
    x_tr, x_val, x_mean, x_std = standardize(X_train, X_val)
    _, x_iid, _, _ = standardize(X_train, X_iid)
    _, x_ood, _, _ = standardize(X_train, X_ood)
    y_tr, y_val, y_mean, y_std = standardize(Y_train, Y_val)
    _, y_iid, _, _ = standardize(Y_train, Y_iid)
    _, y_ood, _, _ = standardize(Y_train, Y_ood)

    Xtr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(x_val, dtype=torch.float32, device=device)
    Yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    Xiid = torch.tensor(x_iid, dtype=torch.float32, device=device)
    Xood = torch.tensor(x_ood, dtype=torch.float32, device=device)

    output_dim = Ytr.shape[1]
    best = None
    configs = [(1e-3, 1e-4), (3e-4, 1e-4), (1e-3, 0.0)]
    for lr, wd in configs:
        set_seed(seed)
        model = ActionMLP(Xtr.shape[1], output_dim, hidden_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        best_state = None
        best_val = -float("inf")
        patience = 0
        order = np.arange(len(Xtr))
        for _epoch in range(120):
            np.random.default_rng(seed + _epoch).shuffle(order)
            model.train()
            for idx in range(0, len(order), batch_size):
                batch_idx = order[idx : idx + batch_size]
                xb = Xtr[batch_idx]
                yb = Ytr[batch_idx]
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(Xva).cpu().numpy() * y_std + y_mean
            val_r2 = compute_r2(Y_val, val_pred)
            if val_r2 > best_val + 1e-5:
                best_val = val_r2
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 12:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred = model(Xtr).cpu().numpy() * y_std + y_mean
            val_pred = model(Xva).cpu().numpy() * y_std + y_mean
            iid_pred = model(Xiid).cpu().numpy() * y_std + y_mean
            ood_pred = model(Xood).cpu().numpy() * y_std + y_mean
        metrics = {
            "lr": lr,
            "wd": wd,
            "train_r2": compute_r2(Y_train, train_pred),
            "val_r2": compute_r2(Y_val, val_pred),
            "iid_r2": compute_r2(Y_iid, iid_pred),
            "ood_r2": compute_r2(Y_ood, ood_pred),
        }
        metrics["ood_gap"] = metrics["iid_r2"] - metrics["ood_r2"]
        if best is None or metrics["val_r2"] > best["val_r2"]:
            best = metrics
    return best


def write_csv(path: str, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], key: str):
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(description="Offline physics-OOD action-chunk regression")
    parser.add_argument("--task", default="push")
    parser.add_argument("--feature-root", required=True, help="Base root containing <model>/<task>/<ep>.safetensors")
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-tag", default="functional_significance")
    parser.add_argument("--probe-seeds", nargs="*", type=int, default=[42, 123, 2024])
    parser.add_argument("--reps", nargs="+", required=True, help="name=model:layer")
    args = parser.parse_args()

    ensure_dirs()
    device = args.device if torch.cuda.is_available() else "cpu"
    reps = [parse_rep(x) for x in args.reps]

    available_eps = None
    for rep in reps:
        rep_dir = Path(args.feature_root) / rep.model / args.task
        eps = sorted(int(p.stem) for p in rep_dir.glob("*.safetensors"))
        if available_eps is None:
            available_eps = set(eps)
        else:
            available_eps &= set(eps)
    episode_ids = sorted(available_eps or [])
    if not episode_ids:
        raise ValueError("No common episodes available across requested representations")

    meta = load_task_meta(args.task, episode_ids)
    split = build_physics_split(args.task, meta, split_seed=args.split_seed)
    split_path = os.path.join(RESULTS_DIR, f"{args.run_tag}_split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    action_cache = load_action_cache(args.task, episode_ids)

    summary = {
        "task": args.task,
        "chunk_len": args.chunk_len,
        "feature_root": args.feature_root,
        "split": split,
        "representations": {},
    }

    for rep in reps:
        feature_root = os.path.join(args.feature_root, rep.model, args.task)
        X_train, Y_train, _, missing_train = collect_samples(feature_root, rep.layer, split["train_eps"], action_cache, args.chunk_len)
        X_val, Y_val, _, missing_val = collect_samples(feature_root, rep.layer, split["val_eps"], action_cache, args.chunk_len)
        X_iid, Y_iid, _, missing_iid = collect_samples(feature_root, rep.layer, split["iid_eps"], action_cache, args.chunk_len)
        X_ood, Y_ood, _, missing_ood = collect_samples(feature_root, rep.layer, split["ood_eps"], action_cache, args.chunk_len)

        rows = []
        for seed in args.probe_seeds:
            metrics = train_one(
                X_train,
                Y_train,
                X_val,
                Y_val,
                X_iid,
                Y_iid,
                X_ood,
                Y_ood,
                device=device,
                seed=seed,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size,
            )
            rows.append(
                {
                    "representation": rep.name,
                    "model": rep.model,
                    "layer": rep.layer,
                    "probe_seed": seed,
                    **metrics,
                }
            )
        csv_path = os.path.join(RESULTS_DIR, f"action_ood_{args.task}_{rep.name}_{args.run_tag}.csv")
        write_csv(csv_path, rows)

        summary["representations"][rep.name] = {
            "model": rep.model,
            "layer": rep.layer,
            "n_train_windows": int(len(X_train)),
            "n_val_windows": int(len(X_val)),
            "n_iid_windows": int(len(X_iid)),
            "n_ood_windows": int(len(X_ood)),
            "missing_episodes": {
                "train": missing_train,
                "val": missing_val,
                "iid": missing_iid,
                "ood": missing_ood,
            },
            "csv": csv_path,
            "train_r2_mean": summarize_rows(rows, "train_r2")[0],
            "train_r2_std": summarize_rows(rows, "train_r2")[1],
            "val_r2_mean": summarize_rows(rows, "val_r2")[0],
            "val_r2_std": summarize_rows(rows, "val_r2")[1],
            "iid_r2_mean": summarize_rows(rows, "iid_r2")[0],
            "iid_r2_std": summarize_rows(rows, "iid_r2")[1],
            "ood_r2_mean": summarize_rows(rows, "ood_r2")[0],
            "ood_r2_std": summarize_rows(rows, "ood_r2")[1],
            "ood_gap_mean": summarize_rows(rows, "ood_gap")[0],
            "ood_gap_std": summarize_rows(rows, "ood_gap")[1],
        }

    summary_path = os.path.join(RESULTS_DIR, f"{args.run_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    rep_names = list(summary["representations"].keys())
    iid_vals = [summary["representations"][k]["iid_r2_mean"] for k in rep_names]
    ood_vals = [summary["representations"][k]["ood_r2_mean"] for k in rep_names]
    x = np.arange(len(rep_names))
    plt.figure(figsize=(8, 4.5))
    width = 0.38
    plt.bar(x - width / 2, iid_vals, width=width, label="IID")
    plt.bar(x + width / 2, ood_vals, width=width, label="OOD")
    plt.xticks(x, rep_names, rotation=15, ha="right")
    plt.ylabel("Action-chunk $R^2$")
    plt.title("Push control relevance under hidden-physics shift")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"{args.run_tag}_iid_vs_ood.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    ordered = sorted(
        summary["representations"].items(),
        key=lambda kv: (-kv[1]["ood_r2_mean"], kv[1]["ood_gap_mean"]),
    )
    report_path = os.path.join(RESULTS_DIR, f"functional_significance_verdict_{args.task}_{args.run_tag}.md")
    with open(report_path, "w") as f:
        f.write("# Functional Significance Verdict\n\n")
        f.write(f"- task: `{args.task}`\n")
        f.write(f"- chunk_len: `{args.chunk_len}`\n")
        f.write(f"- split counts: `{json.dumps(split['counts'])}`\n")
        f.write(f"- figure: `{fig_path}`\n\n")
        f.write("| Representation | Model | Layer | IID $R^2$ | OOD $R^2$ | OOD gap |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for name, rec in ordered:
            f.write(
                f"| `{name}` | `{rec['model']}` | `{rec['layer']}` | "
                f"{rec['iid_r2_mean']:.4f} ± {rec['iid_r2_std']:.4f} | "
                f"{rec['ood_r2_mean']:.4f} ± {rec['ood_r2_std']:.4f} | "
                f"{rec['ood_gap_mean']:.4f} ± {rec['ood_gap_std']:.4f} |\n"
            )
        best_name, best_rec = ordered[0]
        f.write("\n## Current ranking\n\n")
        f.write(
            f"- best OOD representation: `{best_name}` "
            f"(`{best_rec['model']}`, layer `{best_rec['layer']}`)\n"
        )
        f.write(
            "- claim under test: `PEZ-aligned representations are more control-relevant "
            "under hidden physics variation`.\n"
        )


if __name__ == "__main__":
    main()
