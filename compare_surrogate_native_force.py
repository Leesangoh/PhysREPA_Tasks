#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ORDER = [
    ("vjepa_large", "V-JEPA~2 Large"),
    ("videomae_large", "VideoMAE-L"),
    ("dinov2_large", "DINOv2-L"),
]


def row(summary: dict, model_key: str) -> dict:
    s = summary["summaries"][model_key]
    return {
        "peak_r2": s["peak_r2_mean"],
        "peak_r2_std": s["peak_r2_std"],
        "peak_depth": s["peak_depth_mean"],
        "peak_depth_std": s["peak_depth_std"],
        "peak_ci": s["peak_r2_ci"],
        "depth_ci": s["peak_depth_ci"],
    }


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surrogate-json",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/cross_model_force_proxy_multiseed_summary.json"),
    )
    parser.add_argument(
        "--native-json",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/cross_model_native_force_multiseed_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/surrogate_vs_native_force_comparison.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/solee/physrepa_tasks/artifacts/results/surrogate_vs_native_force_comparison.json"),
    )
    args = parser.parse_args()

    surrogate = json.loads(args.surrogate_json.read_text())
    native = json.loads(args.native_json.read_text())

    payload = {"surrogate": surrogate, "native": native}
    args.output_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Surrogate vs Native Force Comparison",
        "",
        "Matched comparison of the earlier surrogate `contact_force_proxy` panel and the recollected native `contact_force` panel on Strike.",
        "",
        "| Model | Surrogate peak $R^2$ | Surrogate depth | Native peak $R^2$ | Native depth |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, paper_name in MODEL_ORDER:
        s = row(surrogate, key)
        n = row(native, key)
        lines.append(
            f"| {paper_name} | {fmt_mean_std(s['peak_r2'], s['peak_r2_std'])} | "
            f"{fmt_mean_std(s['peak_depth'], s['peak_depth_std'])} | "
            f"{fmt_mean_std(n['peak_r2'], n['peak_r2_std'])} | "
            f"{fmt_mean_std(n['peak_depth'], n['peak_depth_std'])} |"
        )
    lines += [
        "",
        "## Bootstrap Comparisons",
        "",
        "| Comparison | Surrogate delta | Surrogate 95% CI | Native delta | Native 95% CI |",
        "| --- | ---: | --- | ---: | --- |",
        f"| V-JEPA~2 Large - VideoMAE-L | {surrogate['comparisons']['vjepa_vs_videomae']['delta_mean']:.3f} | {fmt_ci([surrogate['comparisons']['vjepa_vs_videomae']['ci_low'], surrogate['comparisons']['vjepa_vs_videomae']['ci_high']])} | {native['comparisons']['vjepa_vs_videomae']['delta_mean']:.3f} | {fmt_ci([native['comparisons']['vjepa_vs_videomae']['ci_low'], native['comparisons']['vjepa_vs_videomae']['ci_high']])} |",
        f"| V-JEPA~2 Large - DINOv2-L | {surrogate['comparisons']['vjepa_vs_dino']['delta_mean']:.3f} | {fmt_ci([surrogate['comparisons']['vjepa_vs_dino']['ci_low'], surrogate['comparisons']['vjepa_vs_dino']['ci_high']])} | {native['comparisons']['vjepa_vs_dino']['delta_mean']:.3f} | {fmt_ci([native['comparisons']['vjepa_vs_dino']['ci_low'], native['comparisons']['vjepa_vs_dino']['ci_high']])} |",
        "",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
