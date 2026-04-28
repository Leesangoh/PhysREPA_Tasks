#!/usr/bin/env python3
"""Rerun specific (task, variant) × layers with the fixed probe code.

Reads existing per-target CSVs, removes rows for the specified layers, and
appends fresh rows after re-running the probe. Used to repair layers that
diverged due to the chunked-stat numerical instability before the two-pass
fix in utils/probe.py.

Usage:
  /isaac-sim/python.sh scripts/03c_rerun_layers.py \
    --task peg_insert --variant B --gpu 0 --layers 10,11,16,18,19,20,21,22,23
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import (
    cache_paths,
    list_cached_episodes,
    load_common,
    load_episode_features,
    load_targets,
    progress,
)
from utils.probe import run_groupkfold_probe
from utils.targets import task_target_keys


RESULTS = Path("/home/solee/physrepa_tasks/probe/results")


def stack_task_layer(task: str, variant: str, layer: int):
    eps = list_cached_episodes(task, variant)
    rows_X, rows_e, rows_t = [], [], []
    for ep in eps:
        d = load_episode_features(task, variant, ep)
        rows_X.append(d["feats"][:, layer, :].copy())
        rows_e.append(d["episode_id"])
        rows_t.append(d["t_last"])
    return (
        np.concatenate(rows_X, axis=0),
        np.concatenate(rows_e, axis=0),
        np.concatenate(rows_t, axis=0),
    )


def align_targets(tgt: dict, eps: np.ndarray, t_last: np.ndarray, target_key: str) -> np.ndarray:
    keys_t = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
    lut = {k: i for i, k in enumerate(keys_t)}
    sel = np.array([lut[(int(g), int(t))] for g, t in zip(eps, t_last)], dtype=np.int64)
    return tgt[target_key][sel]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--variant", default="B")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--layers", required=True, help="comma-separated layer indices to rerun")
    args = p.parse_args()

    common = load_common()
    targets = task_target_keys(args.task)
    layers = sorted(int(x) for x in args.layers.split(","))

    progress(f"[rerun] task={args.task} variant={args.variant} layers={layers} gpu={args.gpu}")
    t0 = time.time()
    tgt = load_targets(args.task)

    new_rows: dict[str, list[dict]] = {tk: [] for tk in targets}

    for L in layers:
        tL = time.time()
        X_L, eps_L, t_last_L = stack_task_layer(args.task, args.variant, L)
        progress(f"[rerun] {args.task} {args.variant} layer {L:02d}: loaded "
                 f"[{X_L.shape}] in {time.time()-tL:.1f}s")
        for tk in targets:
            y = align_targets(tgt, eps_L, t_last_L, tk)
            is_direction = tk.endswith("direction")
            extra = ("cos_sim",) if is_direction else ()

            if y.ndim == 1:
                ok = np.isfinite(y)
            else:
                ok = np.isfinite(y).all(axis=1)
            if not ok.all():
                X_use = X_L[ok]; e_use = eps_L[ok]; y_use = y[ok]
            else:
                X_use = X_L; e_use = eps_L; y_use = y

            if y_use.shape[0] < 200:
                progress(f"[rerun] {args.task} {args.variant} {tk}: only {y_use.shape[0]} valid windows — skipping")
                continue

            tprobe = time.time()
            results = run_groupkfold_probe(
                X_use.astype(np.float32, copy=False), y_use, e_use,
                lr_grid=common["probe"]["lr_grid"],
                wd_grid=common["probe"]["wd_grid"],
                epochs=common["probe"]["epochs"],
                batch_size=common["probe"]["batch_size"],
                inner_val_frac=common["probe"]["inner_val_episode_frac"],
                n_splits=common["cv"]["n_splits"],
                seed=common["seed"] + L,
                device=torch.device(f"cuda:{args.gpu}"),
                extra_metrics=extra,
            )
            for fr in results:
                row = {
                    "layer": L,
                    "fold": fr.fold,
                    "best_lr": fr.best_lr,
                    "best_wd": fr.best_wd,
                    "r2": fr.r2,
                    "mse": fr.mse,
                    "n_test_windows": fr.n_test_windows,
                }
                if fr.cos_sim_mean is not None:
                    row["cos_sim_mean"] = fr.cos_sim_mean
                new_rows[tk].append(row)
            r2s = [r.r2 for r in results]
            progress(f"[rerun] {args.task} {args.variant} {tk} L{L:02d}: r2_mean={np.mean(r2s):.3f} std={np.std(r2s):.3f} {time.time()-tprobe:.1f}s")

    # Merge into existing CSVs: drop old rows for the rerun layers, append new.
    out_dir = RESULTS / args.task / f"variant_{args.variant}"
    layers_set = set(layers)
    for tk, rows in new_rows.items():
        if not rows:
            continue
        csv_path = out_dir / f"{tk}.csv"
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            keep = existing[~existing["layer"].isin(layers_set)]
        else:
            keep = pd.DataFrame()
        new_df = pd.DataFrame(rows)
        merged = pd.concat([keep, new_df], ignore_index=True).sort_values(["layer", "fold"])
        cols = ["layer", "fold", "best_lr", "best_wd", "r2", "mse", "n_test_windows"]
        if "cos_sim_mean" in merged.columns:
            cols.append("cos_sim_mean")
        merged = merged[cols]
        merged.to_csv(csv_path, index=False)
        progress(f"[rerun] wrote {csv_path} (kept {len(keep)} + new {len(new_df)} = {len(merged)} rows)")

    # Regenerate _summary.csv for the task+variant
    sum_rows = []
    csvs = sorted(out_dir.glob("*.csv"))
    for c in csvs:
        if c.name == "_summary.csv":
            continue
        df = pd.read_csv(c)
        target = c.stem
        agg = df.groupby("layer").agg({"r2": ["mean", "std"], "mse": ["mean", "std"]}).reset_index()
        agg.columns = ["layer", "r2_mean", "r2_std", "mse_mean", "mse_std"]
        agg["target"] = target
        if "cos_sim_mean" in df.columns:
            cos_agg = df.groupby("layer")["cos_sim_mean"].agg(["mean", "std"]).reset_index()
            cos_agg.columns = ["layer", "cos_sim_mean_mean", "cos_sim_mean_std"]
            agg = agg.merge(cos_agg, on="layer", how="left")
        sum_rows.append(agg)
    if sum_rows:
        cos_cols = [c for c in sum_rows[0].columns if c.startswith("cos_sim")]
        cols = ["target", "layer", "r2_mean", "r2_std", "mse_mean", "mse_std"] + cos_cols
        pd.concat(sum_rows, ignore_index=True)[cols].to_csv(out_dir / "_summary.csv", index=False)

    progress(f"[rerun] {args.task} {args.variant} layers {layers} DONE in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
