#!/usr/bin/env python3
"""Paired episode bootstrap for physics-OOD action-regression comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from probe_physprobe import RESULTS_DIR, ensure_dirs


DEFAULT_COMPARISONS = {
    "push": [
        ("vjepa_fusion", "vjepa_last"),
        ("vjepa_fusion", "vjepa_pez"),
        ("vjepa_fusion", "videomae_best"),
        ("videomae_best", "dino_mid"),
    ],
    "drawer": [
        ("vjepa_fusion", "vjepa_last"),
        ("vjepa_fusion", "vjepa_pez"),
        ("vjepa_fusion", "videomae_best"),
        ("videomae_best", "dino_best"),
    ],
}


def bootstrap_mean_ci(values: np.ndarray, *, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        draw = rng.integers(0, n, size=n)
        samples[idx] = values[draw].mean()
    return {
        "mean_gap": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def load_episode_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" not in df or "episode_id" not in df or "r2" not in df:
        raise ValueError(f"Episode table missing required columns: {path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap by episode for OOD action regression")
    parser.add_argument("--task", required=True, choices=sorted(DEFAULT_COMPARISONS))
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()

    ensure_dirs()
    summary = json.load(open(args.summary_json))
    reps = summary["representations"]
    results = {
        "task": args.task,
        "summary_json": args.summary_json,
        "run_tag": args.run_tag,
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "comparisons": [],
    }

    per_rep = {}
    for rep_name, rec in reps.items():
        episode_csv = rec.get("episode_csv")
        if not episode_csv:
            raise ValueError(f"Representation {rep_name} missing episode_csv in {args.summary_json}")
        df = load_episode_table(episode_csv)
        ood = (
            df[df["split"] == "ood"]
            .groupby("episode_id", as_index=False)["r2"]
            .mean()
            .rename(columns={"r2": rep_name})
        )
        per_rep[rep_name] = ood

    for left, right in DEFAULT_COMPARISONS[args.task]:
        merged = per_rep[left].merge(per_rep[right], on="episode_id", how="inner")
        gaps = merged[left].to_numpy(dtype=np.float64) - merged[right].to_numpy(dtype=np.float64)
        stats = bootstrap_mean_ci(gaps, n_resamples=args.n_resamples, seed=args.seed)
        stats.update(
            {
                "left": left,
                "right": right,
                "n_episodes": int(len(merged)),
                "left_mean": float(merged[left].mean()),
                "right_mean": float(merged[right].mean()),
            }
        )
        results["comparisons"].append(stats)

    json_path = Path(RESULTS_DIR) / f"ood_pairwise_bootstrap_{args.task}_{args.run_tag}.json"
    md_path = Path(RESULTS_DIR) / f"ood_pairwise_bootstrap_{args.task}_{args.run_tag}.md"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# OOD Pairwise Bootstrap\n\n")
        f.write(f"- task: `{args.task}`\n")
        f.write(f"- summary: `{args.summary_json}`\n")
        f.write(f"- resamples: `{args.n_resamples}`\n")
        f.write(f"- bootstrap seed: `{args.seed}`\n\n")
        f.write("| Comparison | Mean gap | 95% CI | Episode count |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for rec in results["comparisons"]:
            f.write(
                f"| `{rec['left']} - {rec['right']}` | "
                f"{rec['mean_gap']:.4f} | "
                f"[{rec['ci_low']:.4f}, {rec['ci_high']:.4f}] | "
                f"{rec['n_episodes']} |\n"
            )


if __name__ == "__main__":
    main()
