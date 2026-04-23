#!/usr/bin/env python3
"""Summarize Strike pretrained-vs-random-init kinematic null results."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGETS = {
    "ee_direction_3d": {
        "pretrained_csv": "/home/solee/physrepa_tasks/artifacts/results/probe_strike_ee_direction_3d_large_token_patch_phase2d_direction3d.csv",
        "random_glob": "/home/solee/physrepa_tasks/artifacts/results/probe_strike_ee_direction_3d_large_token_patch_rev1_randominit_strike_seed*.csv",
    },
    "object_direction_3d": {
        "pretrained_csv": "/home/solee/physrepa_tasks/artifacts/results/probe_strike_object_direction_3d_large_token_patch_phase2d_direction3d.csv",
        "random_glob": "/home/solee/physrepa_tasks/artifacts/results/probe_strike_object_direction_3d_large_token_patch_rev1_randominit_strike_seed*.csv",
    },
    "ee_speed": {
        "pretrained_csv": "/home/solee/physrepa_tasks/artifacts/results/probe_strike_ee_speed_large_token_patch_phase2c_strike.csv",
        "random_glob": "/home/solee/physrepa_tasks/artifacts/results/probe_strike_ee_speed_large_token_patch_rev1_randominit_strike_seed*.csv",
    },
}


def summarize_curve(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path).sort_values("layer")
    val = df["r2_mean"].to_numpy(dtype=np.float64)
    train = df["train_r2_mean"].to_numpy(dtype=np.float64) if "train_r2_mean" in df.columns else None
    peak_idx = int(np.argmax(val))
    return {
        "csv": str(csv_path),
        "L0": float(val[0]),
        "L8": float(val[8]),
        "peak_r2": float(val[peak_idx]),
        "peak_layer": peak_idx,
        "last": float(val[-1]),
        "train_peak_r2": float(np.max(train)) if train is not None else None,
    }


def summarize_many(paths: list[Path]) -> dict:
    rows = [summarize_curve(p) for p in paths]
    out = {
        "csvs": [row["csv"] for row in rows],
        "n_seeds": len(rows),
    }
    for key in ["L0", "L8", "peak_r2", "peak_layer", "last", "train_peak_r2"]:
        vals_raw = [row[key] for row in rows]
        if all(v is None for v in vals_raw):
            out[key] = None
            out[f"{key}_std"] = None
            out[f"{key}_values"] = [None for _ in vals_raw]
            continue
        vals = np.asarray(vals_raw, dtype=np.float64)
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_values"] = [float(v) for v in vals]
    return out


def fmt_scalar(value: float | None, std: float | None = None, decimals: int = 3) -> str:
    if value is None:
        return "NA"
    if std is None:
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def build_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Strike random-init null verdict")
    lines.append("")
    lines.append(
        "This analysis extends the learned-vs-architecture-only null from Push to Strike. "
        "The pretrained Strike rows come from the existing committed paper baseline; "
        "the random-init rows aggregate three new probe seeds over a random-init "
        "V-JEPA 2 Large backbone with `model_seed=0`."
    )
    lines.append("")
    lines.append("| Target | Backbone | L0 | L8 | Peak $R^2$ | Peak layer | Last | Train peak |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for target, row in summary["targets"].items():
        pre = row["pretrained"]
        rnd = row["random_init"]
        lines.append(
            f"| `{target}` | pretrained (existing baseline) | {fmt_scalar(pre['L0'])} | {fmt_scalar(pre['L8'])} | "
            f"{fmt_scalar(pre['peak_r2'])} | {pre['peak_layer']} | {fmt_scalar(pre['last'])} | {fmt_scalar(pre['train_peak_r2'])} |"
        )
        lines.append(
            f"| `{target}` | random-init (3 seeds) | {fmt_scalar(rnd['L0'], rnd['L0_std'])} | "
            f"{fmt_scalar(rnd['L8'], rnd['L8_std'])} | "
            f"{fmt_scalar(rnd['peak_r2'], rnd['peak_r2_std'])} | "
            f"{fmt_scalar(rnd['peak_layer'], rnd['peak_layer_std'], decimals=1)} | "
            f"{fmt_scalar(rnd['last'], rnd['last_std'])} | "
            f"{fmt_scalar(rnd['train_peak_r2'], rnd['train_peak_r2_std'])} |"
        )
    lines.append("")
    lines.append("## Deltas")
    lines.append("")
    lines.append("| Target | Peak delta | Peak delta (%) | Peak-layer shift | Last-layer delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for target, row in summary["targets"].items():
        d = row["delta"]
        lines.append(
            f"| `{target}` | {d['peak_delta']:+.3f} | {d['peak_delta_pct']:+.1f}% | "
            f"{d['peak_layer_shift']:+.1f} | {d['last_delta']:+.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for target, row in summary["targets"].items():
        d = row["delta"]
        direction = "lower and later" if d["peak_delta"] < 0 and d["peak_layer_shift"] > 0 else "mixed"
        lines.append(
            f"- `{target}`: random-init is {direction} than the pretrained baseline "
            f"(peak delta `{d['peak_delta']:+.3f}`, layer shift `{d['peak_layer_shift']:+.1f}`)."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Strike random-init null results")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/random_init_strike_verdict.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/random_init_strike_summary.json"),
    )
    args = parser.parse_args()

    summary = {
        "task": "strike",
        "model": "large",
        "backbone_condition": {
            "pretrained": "existing committed paper baseline",
            "random_init": {"model_seed": 0, "probe_seeds": [42, 123, 2024]},
        },
        "targets": {},
    }

    for target, spec in DEFAULT_TARGETS.items():
        pretrained = summarize_curve(Path(spec["pretrained_csv"]))
        random_paths = [Path(p) for p in sorted(glob(spec["random_glob"]))]
        if len(random_paths) != 3:
            raise ValueError(f"Expected 3 random-init CSVs for {target}, found {len(random_paths)}")
        random_init = summarize_many(random_paths)
        delta = {
            "peak_delta": float(random_init["peak_r2"] - pretrained["peak_r2"]),
            "peak_delta_pct": float(
                100.0 * (random_init["peak_r2"] - pretrained["peak_r2"]) / max(pretrained["peak_r2"], 1e-12)
            ),
            "peak_layer_shift": float(random_init["peak_layer"] - pretrained["peak_layer"]),
            "last_delta": float(random_init["last"] - pretrained["last"]),
        }
        summary["targets"][target] = {
            "pretrained": pretrained,
            "random_init": random_init,
            "delta": delta,
        }

    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.output_markdown.write_text(build_markdown(summary))


if __name__ == "__main__":
    main()
