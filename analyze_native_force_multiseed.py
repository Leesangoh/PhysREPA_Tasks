#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


MODEL_CONFIG = {
    "vjepa_large": {
        "csv_glob": "probe_events_strike_contact_force_native_large_phase3_events_native_force_multiseed_large_seed*.csv",
        "paper_name": "V-JEPA~2 Large",
        "n_layers": 24,
    },
    "videomae_large": {
        "csv_glob": "probe_events_strike_contact_force_native_videomae_large_phase3_events_native_force_multiseed_videomae_seed*.csv",
        "paper_name": "VideoMAE-L",
        "n_layers": 24,
    },
    "dinov2_large": {
        "csv_glob": "probe_events_strike_contact_force_native_dinov2_large_phase3_events_native_force_multiseed_dino_seed*.csv",
        "paper_name": "DINOv2-L",
        "n_layers": 24,
    },
}


@dataclass
class SeedPeak:
    seed: int
    peak_r2: float
    peak_layer: int
    peak_depth: float
    l0: float
    l8: float
    last: float


def bootstrap_mean_ci(xs: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(xs)
    samples = xs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def paired_bootstrap_delta(xs: np.ndarray, ys: np.ndarray, seed: int, n_boot: int = 20000) -> Dict[str, float]:
    assert len(xs) == len(ys)
    rng = np.random.default_rng(seed)
    deltas = xs - ys
    n = len(deltas)
    samples = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975]).tolist()
    p_tail = 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_tail_p": p_tail,
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def load_seed_peaks(result_dir: Path, model_key: str) -> List[SeedPeak]:
    cfg = MODEL_CONFIG[model_key]
    peaks: List[SeedPeak] = []
    for csv_path in sorted(result_dir.glob(cfg["csv_glob"])):
        df = pd.read_csv(csv_path).sort_values("layer").reset_index(drop=True)
        peak_idx = int(df["r2_mean"].idxmax())
        peak_row = df.loc[peak_idx]
        seed = int(peak_row["probe_seed"])
        layer = int(peak_row["layer"])
        peaks.append(
            SeedPeak(
                seed=seed,
                peak_r2=float(peak_row["r2_mean"]),
                peak_layer=layer,
                peak_depth=float(layer / cfg["n_layers"]),
                l0=float(df.loc[df["layer"] == 0, "r2_mean"].iloc[0]),
                l8=float(df.loc[df["layer"] == 8, "r2_mean"].iloc[0]),
                last=float(df.loc[df["layer"] == cfg["n_layers"] - 1, "r2_mean"].iloc[0]),
            )
        )
    if not peaks:
        raise FileNotFoundError(f"No native-force CSVs found for {model_key}")
    return sorted(peaks, key=lambda x: x.seed)


def require_same_seeds(a: List[SeedPeak], b: List[SeedPeak]) -> tuple[np.ndarray, np.ndarray]:
    map_a = {x.seed: x for x in a}
    map_b = {x.seed: x for x in b}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        raise ValueError("No common seeds found between models")
    return (
        np.array([map_a[s].peak_r2 for s in common], dtype=np.float64),
        np.array([map_b[s].peak_r2 for s in common], dtype=np.float64),
    )


def summarize(peaks: List[SeedPeak], seed: int) -> Dict[str, object]:
    peak_r2 = np.array([x.peak_r2 for x in peaks], dtype=np.float64)
    peak_depth = np.array([x.peak_depth for x in peaks], dtype=np.float64)
    peak_layer = np.array([x.peak_layer for x in peaks], dtype=np.float64)
    l0 = np.array([x.l0 for x in peaks], dtype=np.float64)
    l8 = np.array([x.l8 for x in peaks], dtype=np.float64)
    last = np.array([x.last for x in peaks], dtype=np.float64)
    return {
        "seed_peaks": [asdict(x) for x in peaks],
        "peak_r2_mean": float(peak_r2.mean()),
        "peak_r2_std": float(peak_r2.std(ddof=1)) if len(peak_r2) > 1 else 0.0,
        "peak_r2_ci": bootstrap_mean_ci(peak_r2, seed + 1),
        "peak_depth_mean": float(peak_depth.mean()),
        "peak_depth_std": float(peak_depth.std(ddof=1)) if len(peak_depth) > 1 else 0.0,
        "peak_depth_ci": bootstrap_mean_ci(peak_depth, seed + 2),
        "peak_layer_mean": float(peak_layer.mean()),
        "peak_layer_std": float(peak_layer.std(ddof=1)) if len(peak_layer) > 1 else 0.0,
        "l0_mean": float(l0.mean()),
        "l8_mean": float(l8.mean()),
        "last_mean": float(last.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("/home/solee/physrepa_tasks/artifacts/results"))
    parser.add_argument("--output-markdown", type=Path, default=Path("/home/solee/physrepa_tasks/artifacts/results/cross_model_native_force_multiseed_verdict.md"))
    parser.add_argument("--output-json", type=Path, default=Path("/home/solee/physrepa_tasks/artifacts/results/cross_model_native_force_multiseed_summary.json"))
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    peaks = {key: load_seed_peaks(args.result_dir, key) for key in MODEL_CONFIG}
    summaries = {key: summarize(val, args.bootstrap_seed + i * 10) for i, (key, val) in enumerate(peaks.items())}
    vjepa_vs_videomae = paired_bootstrap_delta(*require_same_seeds(peaks["vjepa_large"], peaks["videomae_large"]), seed=args.bootstrap_seed + 101)
    vjepa_vs_dino = paired_bootstrap_delta(*require_same_seeds(peaks["vjepa_large"], peaks["dinov2_large"]), seed=args.bootstrap_seed + 202)
    payload = {
        "summaries": summaries,
        "comparisons": {
            "vjepa_vs_videomae": vjepa_vs_videomae,
            "vjepa_vs_dino": vjepa_vs_dino,
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Cross-Model Native Force Multiseed Verdict",
        "",
        "Matched evaluation on the recollected `Strike` dataset using native `physics_gt.contact_force` rather than the surrogate force proxy.",
        "",
        "## Summary Table",
        "",
        "| Model | Seeds | Peak $R^2$ | 95% bootstrap CI | Peak layer | Peak depth | 95% depth CI | $L0$ | $L8$ | Last |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key in ["vjepa_large", "videomae_large", "dinov2_large"]:
        cfg = MODEL_CONFIG[key]
        s = summaries[key]
        lines.append(
            f"| {cfg['paper_name']} | {len(s['seed_peaks'])} | "
            f"{s['peak_r2_mean']:.3f} ± {s['peak_r2_std']:.3f} | "
            f"[{s['peak_r2_ci'][0]:.3f}, {s['peak_r2_ci'][1]:.3f}] | "
            f"{s['peak_layer_mean']:.1f} ± {s['peak_layer_std']:.1f} | "
            f"{s['peak_depth_mean']:.3f} ± {s['peak_depth_std']:.3f} | "
            f"[{s['peak_depth_ci'][0]:.3f}, {s['peak_depth_ci'][1]:.3f}] | "
            f"{s['l0_mean']:.3f} | {s['l8_mean']:.3f} | {s['last_mean']:.3f} |"
        )
    lines += [
        "",
        "## Paired Bootstrap Comparisons",
        "",
        "| Comparison | Mean peak-$R^2$ delta | 95% bootstrap CI | Tail $p$ | CI excludes 0? |",
        "| --- | ---: | --- | ---: | --- |",
        f"| V-JEPA~2 Large - VideoMAE-L | {vjepa_vs_videomae['delta_mean']:.3f} | [{vjepa_vs_videomae['ci_low']:.3f}, {vjepa_vs_videomae['ci_high']:.3f}] | {vjepa_vs_videomae['bootstrap_tail_p']:.4f} | {'yes' if vjepa_vs_videomae['ci_excludes_zero'] else 'no'} |",
        f"| V-JEPA~2 Large - DINOv2-L | {vjepa_vs_dino['delta_mean']:.3f} | [{vjepa_vs_dino['ci_low']:.3f}, {vjepa_vs_dino['ci_high']:.3f}] | {vjepa_vs_dino['bootstrap_tail_p']:.4f} | {'yes' if vjepa_vs_dino['ci_excludes_zero'] else 'no'} |",
        "",
    ]
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_markdown}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
