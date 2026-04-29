#!/usr/bin/env python3
"""Phase 5: Variant A probe sweep — 5-fold GroupKFold Adam batched 20-HP, 100 epochs.

For a given (task, target) and variant in {A}:
- Load all per-episode .npz, stack into arrays (X all 24 layers, y, groups).
- For each layer: run 5-fold GroupKFold Adam batched probe.
- Append per-fold rows to results/<task>/variant_<v>/<target>.csv with columns
  layer, fold, best_lr, best_wd, r2, mse, n_test_windows [, cos_sim_mean].
- Write _summary.csv (groupby layer) at the end of all targets.

Targets per task come from utils.targets.task_target_keys.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
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


def stack_task_features(task: str, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all 24 layers at once. Memory: N × 24 × D × 2 bytes (fp16).
    Variant A (D=1024): ~16 GB for push. Variant B (D=8192): ~134 GB for push."""
    eps = list_cached_episodes(task, variant)
    if not eps:
        raise FileNotFoundError(f"no cached features for task={task} variant={variant}")
    rows_X = []
    rows_e = []
    rows_t = []
    for ep in eps:
        d = load_episode_features(task, variant, ep)
        rows_X.append(d["feats"])
        rows_e.append(d["episode_id"])
        rows_t.append(d["t_last"])
    X = np.concatenate(rows_X, axis=0)
    e = np.concatenate(rows_e, axis=0)
    t = np.concatenate(rows_t, axis=0)
    return X, e, t


def stack_task_layer(task: str, variant: str, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only one layer's features. Memory: N × D × 2 bytes (fp16).
    Variant B push: ~5.6 GB instead of ~134 GB. Disk-cached after first read.
    """
    eps = list_cached_episodes(task, variant)
    if not eps:
        raise FileNotFoundError(f"no cached features for task={task} variant={variant}")
    rows_X, rows_e, rows_t = [], [], []
    for ep in eps:
        d = load_episode_features(task, variant, ep)
        rows_X.append(d["feats"][:, layer, :].copy())
        rows_e.append(d["episode_id"])
        rows_t.append(d["t_last"])
    X = np.concatenate(rows_X, axis=0)
    e = np.concatenate(rows_e, axis=0)
    t = np.concatenate(rows_t, axis=0)
    return X, e, t


def align_targets(tgt: dict, eps: np.ndarray, t_last: np.ndarray, target_key: str) -> np.ndarray:
    """For each (ep, t) in feature index, look up the target row.

    Targets npz also stores (episode_id, t_last) per row; build dict lookup.
    """
    keys_t = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
    lut = {k: i for i, k in enumerate(keys_t)}
    sel = np.array([lut[(int(g), int(t))] for g, t in zip(eps, t_last)], dtype=np.int64)
    return tgt[target_key][sel]


def run_target(task: str, variant: str, target: str, X: np.ndarray, eps: np.ndarray, t_last: np.ndarray,
               tgt: dict, *, gpu: int, common: dict, layers: list[int]) -> list[dict]:
    """Run probe for all `layers` of the given (task, variant, target)."""
    y = align_targets(tgt, eps, t_last, target)
    is_direction = target.endswith("direction")
    extra = ("cos_sim",) if is_direction else ()

    # Drop NaN rows once across both X and y so all layers see the same windows.
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if not ok.all():
        X = X[ok]
        eps_local = eps[ok]
        y = y[ok]
    else:
        eps_local = eps
    if y.shape[0] < 200:
        progress(f"[probe] {task} {variant} {target}: only {y.shape[0]} valid windows — skipping")
        return []

    rows: list[dict] = []
    for L in layers:
        X_L = X[:, L, :].astype(np.float32, copy=False)
        t0 = time.time()
        results = run_groupkfold_probe(
            X_L, y, eps_local,
            lr_grid=common["probe"]["lr_grid"],
            wd_grid=common["probe"]["wd_grid"],
            epochs=common["probe"]["epochs"],
            batch_size=common["probe"]["batch_size"],
            inner_val_frac=common["probe"]["inner_val_episode_frac"],
            n_splits=common["cv"]["n_splits"],
            seed=common["seed"] + L,
            device=torch.device(f"cuda:{gpu}"),
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
            rows.append(row)
        dt = time.time() - t0
        r2s = [fr.r2 for fr in results]
        progress(f"[probe] {task} {variant} {target} L{L:02d}: r2_mean={np.mean(r2s):.3f} std={np.std(r2s):.3f} {dt:.1f}s")
    return rows


def write_target_csv(task: str, variant: str, target: str, rows: list[dict]) -> Path:
    out_dir = RESULTS / task / f"variant_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{target}.csv"
    if not rows:
        return p
    cols = ["layer", "fold", "best_lr", "best_wd", "r2", "mse", "n_test_windows"]
    if any("cos_sim_mean" in r for r in rows):
        cols.append("cos_sim_mean")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def write_task_summary(task: str, variant: str) -> Path:
    """Summary: layer × target → r2_mean/std, mse_mean/std (and cos_sim if applicable)."""
    import pandas as pd
    out_dir = RESULTS / task / f"variant_{variant}"
    csvs = sorted(out_dir.glob("*.csv"))
    if not csvs:
        return out_dir / "_summary.csv"
    summary_rows = []
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
        summary_rows.append(agg)
    sumdf = (
        __import__("pandas").concat(summary_rows, ignore_index=True)
        [["target", "layer", "r2_mean", "r2_std", "mse_mean", "mse_std"]
         + [c for c in summary_rows[0].columns if c.startswith("cos_sim")]]
    )
    p = out_dir / "_summary.csv"
    sumdf.to_csv(p, index=False)
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--variant", default="A")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--targets", default="all", help="comma-separated, or 'all'")
    p.add_argument("--tier", default="kinematic",
                   choices=["kinematic", "rotational", "contact", "progress", "all_extended"],
                   help="which target tier to use (default kinematic = spec § 4)")
    p.add_argument("--layers", default="all", help="comma-separated, or 'all'")
    p.add_argument("--per-layer-load", action="store_true",
                   help="Load one layer at a time (smaller peak memory). Default for variant B.")
    args = p.parse_args()

    common = load_common()
    if args.targets == "all":
        targets = task_target_keys(args.task, tier=args.tier)
    else:
        targets = args.targets.split(",")
    layers = list(range(24)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    per_layer = args.per_layer_load or args.variant == "B"

    progress(f"[probe] task={args.task} variant={args.variant} targets={targets} layers={layers} "
             f"gpu={args.gpu} per_layer_load={per_layer}")
    t0 = time.time()
    tgt = load_targets(args.task)

    if not per_layer:
        # Variant A path: load all 24 layers at once, iterate per target × all layers.
        X, eps, t_last = stack_task_features(args.task, args.variant)
        progress(f"[probe] {args.task}: features [{X.shape}] eps {len(np.unique(eps))} loaded {time.time()-t0:.1f}s")
        for tk in targets:
            rows = run_target(args.task, args.variant, tk, X, eps, t_last, tgt,
                              gpu=args.gpu, common=common, layers=layers)
            write_target_csv(args.task, args.variant, tk, rows)
    else:
        # Variant B path: per-layer outer loop. Each layer's features fit in <10 GB.
        # All 12 targets per layer share the same X, so we run them in inner loop.
        per_target_rows: dict[str, list[dict]] = {tk: [] for tk in targets}
        for L in layers:
            tL = time.time()
            X_L, eps_L, t_last_L = stack_task_layer(args.task, args.variant, L)
            progress(f"[probe] {args.task} {args.variant} layer {L:02d}: loaded "
                     f"[{X_L.shape}] in {time.time()-tL:.1f}s")
            X_L_3d = X_L[:, None, :]   # synthesize a [N, 1, F] tensor so run_target's X[:, L, :] works for layer index 0
            for tk in targets:
                rows = run_target(
                    args.task, args.variant, tk, X_L_3d, eps_L, t_last_L, tgt,
                    gpu=args.gpu, common=common, layers=[0],
                )
                # The probe output's "layer" field will be 0 (because X_L_3d has only one layer).
                # Rewrite to the actual layer index L before appending.
                for r in rows:
                    r["layer"] = L
                    per_target_rows[tk].append(r)
            del X_L, X_L_3d
            # Write CSVs after each layer so partial results are persisted incrementally.
            for tk in targets:
                write_target_csv(args.task, args.variant, tk, per_target_rows[tk])

    write_task_summary(args.task, args.variant)
    progress(f"[probe] {args.task} {args.variant} DONE in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
