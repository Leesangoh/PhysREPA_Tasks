#!/usr/bin/env python3
"""Summarize Push Huge multiseed scale-law results."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "ee_direction_3d": {
        "glob": "/home/solee/physrepa_tasks/artifacts/results/probe_push_ee_direction_3d_huge_token_patch_scale_huge_seed*.csv",
    },
    "ee_speed": {
        "glob": "/home/solee/physrepa_tasks/artifacts/results/probe_push_ee_speed_huge_token_patch_scale_huge_seed*.csv",
    },
}


def summarize_curve(path: Path) -> dict:
    df = pd.read_csv(path).sort_values("layer")
    vals = df["r2_mean"].to_numpy(dtype=np.float64)
    peak_idx = int(np.argmax(vals))
    n_layers = len(vals)
    return {
        "csv": str(path),
        "probe_seed": int(df["probe_seed"].iloc[0]) if "probe_seed" in df.columns else None,
        "L0": float(vals[0]),
        "L8": float(vals[8]),
        "peak_r2": float(vals[peak_idx]),
        "peak_layer": float(peak_idx),
        "peak_depth": float(peak_idx / n_layers),
        "last": float(vals[-1]),
        "train_peak_r2": float(df["train_r2_mean"].max()) if "train_r2_mean" in df.columns else None,
    }


def summarize_many(rows: list[dict]) -> dict:
    out = {"n_seeds": len(rows), "csvs": [r["csv"] for r in rows], "probe_seeds": [r["probe_seed"] for r in rows]}
    for key in ["L0", "L8", "peak_r2", "peak_layer", "peak_depth", "last", "train_peak_r2"]:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_values"] = [float(v) for v in vals]
    return out


def build_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Huge multiseed scale-law verdict")
    lines.append("")
    lines.append("This pass adds the missing Huge probe seeds so the Push scale-law table can report three-seed statistics.")
    lines.append("")
    lines.append("| Target | Seeds | L0 | L8 | Peak $R^2$ | Peak layer | Peak depth | Last | Train peak |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for target, row in summary["targets"].items():
        lines.append(
            f"| `{target}` | {row['n_seeds']} | {row['L0']:.3f} $\\pm$ {row['L0_std']:.3f} | "
            f"{row['L8']:.3f} $\\pm$ {row['L8_std']:.3f} | {row['peak_r2']:.3f} $\\pm$ {row['peak_r2_std']:.3f} | "
            f"{row['peak_layer']:.1f} $\\pm$ {row['peak_layer_std']:.1f} | {row['peak_depth']:.3f} $\\pm$ {row['peak_depth_std']:.3f} | "
            f"{row['last']:.3f} $\\pm$ {row['last_std']:.3f} | {row['train_peak_r2']:.3f} $\\pm$ {row['train_peak_r2_std']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize huge multiseed scale-law results")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/scale_huge_multiseed_verdict.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/scale_huge_multiseed_summary.json"),
    )
    args = parser.parse_args()

    summary = {"task": "push", "model": "huge", "targets": {}}
    for target, spec in TARGETS.items():
        paths = [Path(p) for p in sorted(glob(spec["glob"]))]
        if len(paths) != 3:
            raise ValueError(f"Expected 3 huge CSVs for {target}, found {len(paths)}")
        rows = [summarize_curve(p) for p in paths]
        summary["targets"][target] = summarize_many(rows)

    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.output_markdown.write_text(build_markdown(summary))


if __name__ == "__main__":
    main()
