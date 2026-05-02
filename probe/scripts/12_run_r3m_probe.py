#!/usr/bin/env python3
"""R3M stage-wise probe runner.

Thin wrapper around the standard Adam 20-HP GroupKFold probe, specialized for
R3M caches stored as one per-episode ``.npz`` with keys:
  - stage_0 .. stage_4
  - t_last
  - episode_id

Unlike V-JEPA caches, R3M stages have heterogeneous dimensionality, so we keep
each stage separate and probe them independently.

Outputs:
  - results/<task>/r3m/<target>.csv
      cols: stage, fold, best_lr, best_wd, r2, mse, n_test_windows
  - results/<task>/r3m/_summary.csv
      cols: target, stage, r2_mean, r2_std, mse_mean, mse_std
  - results/<task>/r3m/_best_stage.csv
      cols: target, best_stage, best_r2_mean, best_r2_std
  - results/r3m_vs_vjepa_summary.csv
      side-by-side best-stage vs best-layer comparison
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

PROBE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from utils.io import load_common, load_targets, progress
from utils.probe import run_groupkfold_probe


RESULTS = PROBE_ROOT / "results"
R3M_CACHE_ROOT = Path("/mnt/md1/solee/physprobe_features")
TASKS_ALLOWED = ["push", "strike", "drawer"]
DEFAULT_TARGETS = ["ee_acceleration", "contact_flag", "contact_force_log1p_mag"]
ALL_STAGES = [f"stage_{i}" for i in range(5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R3M stage-wise probes.")
    parser.add_argument("--task", default="push", choices=TASKS_ALLOWED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--targets", default="all", help="comma-separated target list or 'all'")
    parser.add_argument("--stages", default="all", help="comma-separated stage list or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="5 episodes, one stage, one target, print only")
    return parser.parse_args()


def stage_cache_dir(task: str) -> Path:
    return R3M_CACHE_ROOT / task / "r3m"


def list_r3m_episodes(task: str) -> list[int]:
    cache_dir = stage_cache_dir(task)
    if not cache_dir.exists():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in cache_dir.glob("episode_*.npz"))


def load_r3m_episode(task: str, episode_id: int) -> dict[str, np.ndarray]:
    path = stage_cache_dir(task) / f"episode_{episode_id:06d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def normalize_episode_id_field(episode_id_arr: np.ndarray, n_rows: int, fallback_episode_id: int) -> np.ndarray:
    arr = np.asarray(episode_id_arr)
    if arr.ndim == 0:
        return np.full((n_rows,), int(arr.item()), dtype=np.int32)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full((n_rows,), int(arr[0]), dtype=np.int32)
    if arr.size != n_rows:
        raise ValueError(f"episode_id size mismatch: got {arr.size}, expected {n_rows}")
    return arr.astype(np.int32, copy=False)


def stack_stage(task: str, stage: str, max_eps: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = list_r3m_episodes(task)
    if max_eps is not None and len(eps) > max_eps:
        eps = eps[:max_eps]
    if not eps:
        raise FileNotFoundError(f"no R3M cached episodes for task={task}")
    rows_x = []
    rows_e = []
    rows_t = []
    for ep in eps:
        d = load_r3m_episode(task, ep)
        if stage not in d:
            raise KeyError(f"{stage} missing in episode_{ep:06d}.npz")
        feats = d[stage].astype(np.float32, copy=False)
        t_last = d["t_last"].astype(np.int64, copy=False).reshape(-1)
        if feats.shape[0] != t_last.size:
            raise ValueError(f"{task} {stage} ep {ep}: feats rows {feats.shape[0]} != t_last {t_last.size}")
        ep_arr = normalize_episode_id_field(d["episode_id"], feats.shape[0], ep)
        rows_x.append(feats)
        rows_e.append(ep_arr)
        rows_t.append(t_last)
    X = np.concatenate(rows_x, axis=0)
    eps_arr = np.concatenate(rows_e, axis=0)
    t_last = np.concatenate(rows_t, axis=0)
    return X, eps_arr, t_last


def align_targets(tgt: dict[str, np.ndarray], eps: np.ndarray, t_last: np.ndarray, target_key: str) -> np.ndarray:
    keys_t = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
    lut = {k: i for i, k in enumerate(keys_t)}
    sel = np.array([lut[(int(g), int(t))] for g, t in zip(eps, t_last)], dtype=np.int64)
    return tgt[target_key][sel]


def run_stage_target(
    task: str,
    stage: str,
    target: str,
    X_stage: np.ndarray,
    eps: np.ndarray,
    t_last: np.ndarray,
    tgt: dict[str, np.ndarray],
    *,
    gpu: int,
    common: dict,
) -> list[dict]:
    y = align_targets(tgt, eps, t_last, target)
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if not ok.all():
        X_stage = X_stage[ok]
        eps = eps[ok]
        y = y[ok]
    if y.shape[0] < 200:
        progress(f"[r3m_probe] {task} {stage} {target}: only {y.shape[0]} valid windows — skipping")
        return []

    t0 = time.time()
    results = run_groupkfold_probe(
        X_stage.astype(np.float32, copy=False),
        y,
        eps,
        lr_grid=common["probe"]["lr_grid"],
        wd_grid=common["probe"]["wd_grid"],
        epochs=common["probe"]["epochs"],
        batch_size=common["probe"]["batch_size"],
        inner_val_frac=common["probe"]["inner_val_episode_frac"],
        n_splits=common["cv"]["n_splits"],
        seed=common["seed"] + int(stage.split("_")[1]),
        device=torch.device(f"cuda:{gpu}"),
    )
    rows = [
        {
            "stage": stage,
            "fold": fr.fold,
            "best_lr": fr.best_lr,
            "best_wd": fr.best_wd,
            "r2": fr.r2,
            "mse": fr.mse,
            "n_test_windows": fr.n_test_windows,
        }
        for fr in results
    ]
    r2s = [fr.r2 for fr in results]
    progress(
        f"[r3m_probe] {task} {stage} {target}: "
        f"r2_mean={np.mean(r2s):.3f} std={np.std(r2s):.3f} {time.time()-t0:.1f}s"
    )
    return rows


def write_target_csv(task: str, target: str, rows: list[dict]) -> Path:
    out_dir = RESULTS / task / "r3m"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target}.csv"
    cols = ["stage", "fold", "best_lr", "best_wd", "r2", "mse", "n_test_windows"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        if rows:
            w.writerows(rows)
    return path


def write_task_summary(task: str) -> tuple[Path, Path]:
    out_dir = RESULTS / task / "r3m"
    csvs = sorted(out_dir.glob("*.csv"))
    summary_rows = []
    best_rows = []
    for c in csvs:
        if c.name.startswith("_"):
            continue
        df = pd.read_csv(c)
        if df.empty:
            continue
        target = c.stem
        agg = df.groupby("stage").agg({"r2": ["mean", "std"], "mse": ["mean", "std"]}).reset_index()
        agg.columns = ["stage", "r2_mean", "r2_std", "mse_mean", "mse_std"]
        agg["target"] = target
        summary_rows.append(agg)
        best = agg.sort_values(["r2_mean", "r2_std"], ascending=[False, True]).iloc[0]
        best_rows.append(
            {
                "target": target,
                "best_stage": best["stage"],
                "best_r2_mean": float(best["r2_mean"]),
                "best_r2_std": float(best["r2_std"]),
            }
        )
    sumdf = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(
        columns=["target", "stage", "r2_mean", "r2_std", "mse_mean", "mse_std"]
    )
    sumdf = sumdf[["target", "stage", "r2_mean", "r2_std", "mse_mean", "mse_std"]]
    summary_path = out_dir / "_summary.csv"
    sumdf.to_csv(summary_path, index=False)
    bestdf = pd.DataFrame(best_rows)
    best_path = out_dir / "_best_stage.csv"
    bestdf.to_csv(best_path, index=False)
    return summary_path, best_path


def best_vjepa_row(task: str, target: str) -> dict | None:
    path = RESULTS / task / "variant_A" / f"{target}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    agg = df.groupby("layer").agg({"r2": ["mean", "std"]}).reset_index()
    agg.columns = ["layer", "r2_mean", "r2_std"]
    best = agg.sort_values(["r2_mean", "r2_std"], ascending=[False, True]).iloc[0]
    return {
        "vjepa_best_layer": int(best["layer"]),
        "vjepa_peak_r2": float(best["r2_mean"]),
        "vjepa_peak_r2_std": float(best["r2_std"]),
    }


def write_global_comparison() -> Path:
    rows = []
    for task in TASKS_ALLOWED:
        best_path = RESULTS / task / "r3m" / "_best_stage.csv"
        if not best_path.exists():
            continue
        best_df = pd.read_csv(best_path)
        for _, r in best_df.iterrows():
            target = str(r["target"])
            v = best_vjepa_row(task, target)
            if v is None:
                continue
            rows.append(
                {
                    "task": task,
                    "target": target,
                    "r3m_best_stage": str(r["best_stage"]),
                    "r3m_peak_r2": float(r["best_r2_mean"]),
                    "r3m_peak_r2_std": float(r["best_r2_std"]),
                    **v,
                    "delta_r3m_minus_vjepa": float(r["best_r2_mean"] - v["vjepa_peak_r2"]),
                }
            )
    out = RESULTS / "r3m_vs_vjepa_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    args = parse_args()
    common = load_common()
    targets = DEFAULT_TARGETS if args.targets == "all" else [x.strip() for x in args.targets.split(",") if x.strip()]
    stages = ALL_STAGES if args.stages == "all" else [x.strip() for x in args.stages.split(",") if x.strip()]
    if args.dry_run:
        targets = [targets[0]]
        stages = [stages[0]]

    progress(f"[r3m_probe] task={args.task} targets={targets} stages={stages} gpu={args.gpu} dry_run={args.dry_run}")
    tgt = load_targets(args.task)
    max_eps = 5 if args.dry_run else None

    per_target_rows: dict[str, list[dict]] = {t: [] for t in targets}
    for stage in stages:
        t0 = time.time()
        X_stage, eps, t_last = stack_stage(args.task, stage, max_eps=max_eps)
        progress(f"[r3m_probe] {args.task} {stage}: loaded X={X_stage.shape} eps={len(np.unique(eps))} in {time.time()-t0:.1f}s")
        for target in targets:
            rows = run_stage_target(
                args.task,
                stage,
                target,
                X_stage,
                eps,
                t_last,
                tgt,
                gpu=args.gpu,
                common=common,
            )
            per_target_rows[target].extend(rows)
            if not args.dry_run:
                write_target_csv(args.task, target, per_target_rows[target])

    if args.dry_run:
        preview = []
        for target, rows in per_target_rows.items():
            preview.extend(rows)
        df = pd.DataFrame(preview)
        print(df.head(20).to_string(index=False) if not df.empty else "[r3m_probe] dry-run produced 0 rows", flush=True)
        return

    for target, rows in per_target_rows.items():
        write_target_csv(args.task, target, rows)
    write_task_summary(args.task)
    cmp_path = write_global_comparison()
    progress(f"[r3m_probe] {args.task} DONE; comparison updated at {cmp_path}")


if __name__ == "__main__":
    main()
