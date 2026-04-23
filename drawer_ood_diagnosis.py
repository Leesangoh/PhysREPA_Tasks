#!/usr/bin/env python3
"""Diagnose why Drawer OOD action regression can exceed IID action regression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

from probe_action_ood import WINDOW_LEN, compute_r2
from probe_physprobe import DATA_BASE, RESULTS_DIR, ensure_dirs


def load_actions(task: str, episode_id: int) -> np.ndarray:
    path = Path(DATA_BASE) / task / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
    df = pd.read_parquet(path, columns=["action"])
    return np.stack(df["action"].values).astype(np.float32)


def lag1_autocorr(actions: np.ndarray) -> float:
    if len(actions) < 2:
        return float("nan")
    vals = []
    x = actions[:-1]
    y = actions[1:]
    for dim in range(actions.shape[1]):
        xd = x[:, dim]
        yd = y[:, dim]
        if np.std(xd) < 1e-8 or np.std(yd) < 1e-8:
            continue
        vals.append(np.corrcoef(xd, yd)[0, 1])
    if not vals:
        return float("nan")
    return float(np.nanmean(vals))


def persistence_chunk_r2(actions: np.ndarray, window_starts: np.ndarray, chunk_len: int) -> float:
    y_true = []
    y_pred = []
    for start in window_starts:
        target_start = int(start) + WINDOW_LEN
        target_end = target_start + chunk_len
        if target_end > len(actions) or target_start - 1 < 0:
            continue
        truth = actions[target_start:target_end].reshape(-1)
        pred = np.repeat(actions[target_start - 1][None, :], chunk_len, axis=0).reshape(-1)
        y_true.append(truth)
        y_pred.append(pred)
    if not y_true:
        return float("nan")
    return compute_r2(np.stack(y_true), np.stack(y_pred))


def summarize(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=0)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "q25": float(np.quantile(arr, 0.25)) if len(arr) else float("nan"),
        "q75": float(np.quantile(arr, 0.75)) if len(arr) else float("nan"),
        "n": int(len(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Drawer OOD diagnosis")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--feature-root", required=True, help="Feature root for one representation with window_starts, e.g. .../large/drawer")
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()

    ensure_dirs()
    summary = json.load(open(args.summary_json))
    split = json.load(open(args.split_json))

    groups = {
        "iid": split["iid_eps"],
        "ood": split["ood_eps"],
    }

    rep_episode_stats = {}
    for rep_name, rec in summary["representations"].items():
        df = pd.read_csv(rec["episode_csv"])
        rep_episode_stats[rep_name] = {}
        for split_name in ("iid", "ood"):
            s = (
                df[df["split"] == split_name]
                .groupby("episode_id", as_index=False)["r2"]
                .mean()["r2"]
                .tolist()
            )
            rep_episode_stats[rep_name][split_name] = summarize(s)

    action_stats = {"iid": [], "ood": []}
    for split_name, eps in groups.items():
        for ep in eps:
            actions = load_actions("drawer", ep)
            feat_path = Path(args.feature_root) / f"{ep:06d}.safetensors"
            with safe_open(feat_path, framework="numpy") as f:
                window_starts = f.get_tensor("window_starts").astype(np.int64)
            action_stats[split_name].append(
                {
                    "episode_id": int(ep),
                    "episode_len": int(len(actions)),
                    "action_variance": float(np.var(actions)),
                    "action_autocorr_lag1": lag1_autocorr(actions),
                    "persistence_chunk_r2": persistence_chunk_r2(actions, window_starts, args.chunk_len),
                }
            )

    diag = {
        "summary_json": args.summary_json,
        "split_json": args.split_json,
        "feature_root": args.feature_root,
        "chunk_len": args.chunk_len,
        "rep_episode_r2": rep_episode_stats,
        "action_stats": {
            split_name: {
                metric: summarize([row[metric] for row in rows])
                for metric in ["episode_len", "action_variance", "action_autocorr_lag1", "persistence_chunk_r2"]
            }
            for split_name, rows in action_stats.items()
        },
    }

    json_path = Path(RESULTS_DIR) / f"drawer_ood_diagnosis_{args.run_tag}.json"
    md_path = Path(RESULTS_DIR) / f"drawer_ood_diagnosis_{args.run_tag}.md"
    with open(json_path, "w") as f:
        json.dump(diag, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# Drawer OOD Diagnosis\n\n")
        f.write(f"- summary: `{args.summary_json}`\n")
        f.write(f"- split: `{args.split_json}`\n")
        f.write(f"- chunk_len: `{args.chunk_len}`\n\n")
        f.write("## Per-episode action-regression R²\n\n")
        for rep_name, stats in rep_episode_stats.items():
            f.write(
                f"- `{rep_name}`: IID mean {stats['iid']['mean']:.4f}, "
                f"OOD mean {stats['ood']['mean']:.4f}, "
                f"median shift {stats['ood']['median'] - stats['iid']['median']:.4f}\n"
            )
        f.write("\n## Action statistics\n\n")
        f.write("| Metric | IID mean | OOD mean |\n")
        f.write("| --- | ---: | ---: |\n")
        for metric in ["episode_len", "action_variance", "action_autocorr_lag1", "persistence_chunk_r2"]:
            iid = diag["action_stats"]["iid"][metric]["mean"]
            ood = diag["action_stats"]["ood"][metric]["mean"]
            f.write(f"| `{metric}` | {iid:.4f} | {ood:.4f} |\n")


if __name__ == "__main__":
    main()
