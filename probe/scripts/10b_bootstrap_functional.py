#!/usr/bin/env python3
"""H — Bootstrap CIs on new functional cross-model claims.

For each (task, target):
  1. Find V-JEPA A peak layer L_V
  2. Read per-fold R² values at V-JEPA peak (5 folds)
  3. Read per-fold R² values at DINOv2/VideoMAE peak (5 folds, possibly different layer)
  4. Bootstrap mean difference (DINOv2 - V-JEPA) and (VideoMAE - V-JEPA)
     over fold pairs, n_boot=1000, percentile.

Outputs:
  probe/results/bootstrap_cis_functional.csv
    columns: task, target, ref_model, cmp_model, ref_peak_layer, cmp_peak_layer,
             ref_mean, cmp_mean, gap_mean, gap_ci_lo, gap_ci_hi, p_value

Compares each cmp_model to V-JEPA Variant A as reference.

Usage:
  /isaac-sim/python.sh probe/scripts/10b_bootstrap_functional.py [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROBE_ROOT / "results"
OUT_CSV = RESULTS / "bootstrap_cis_functional.csv"

TASKS = ["push", "strike", "drawer", "reach", "peg_insert", "nut_thread"]
REF_VARIANT = "variant_A"  # V-JEPA Variant A baseline
CMP_VARIANTS = [
    ("variant_A_dinov2_large", "DINOv2 A"),
    ("variant_A_videomae_large", "VideoMAE A"),
    ("variant_B_videomae_large", "VideoMAE B"),
    ("variant_B", "V-JEPA B"),
    ("variant_A_shuffled", "V-JEPA A shuffled"),
]
TARGETS = [
    "ee_velocity__mean",
    "ee_acceleration__peak_to_peak",
    "obj_velocity__mean",
    "obj_acceleration__peak_to_peak",
    "contact_flag__positive_fraction",
    "contact_force_log1p_mag__integral",
    "contact_force_mag__integral",
    "ee_velocity__peak_time_frac",
    "contact_flag__first_event_time_frac",
]


def load_per_fold_at_peak(task: str, variant: str, target: str):
    """Load per-fold R² values at the variant's own peak layer.
    Returns (peak_layer:int, fold_r2_array:np.ndarray of shape [5]) or (None, None).
    """
    p = RESULTS / task / variant / f"{target}.csv"
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p)
        per_layer_mean = df.groupby("layer")["r2"].mean()
        peak_layer = int(per_layer_mean.idxmax())
        fold_r2 = df[df["layer"] == peak_layer].sort_values("fold")["r2"].values
        if len(fold_r2) < 3:
            return None, None
        return peak_layer, fold_r2
    except Exception:
        return None, None


def bootstrap_diff(fold_a: np.ndarray, fold_b: np.ndarray, n_boot: int = 1000,
                   seed: int = 42, ci: float = 0.95):
    """Bootstrap of mean(fold_b) - mean(fold_a). Returns (mean, ci_lo, ci_hi, p)."""
    rng = np.random.default_rng(seed)
    n = max(len(fold_a), len(fold_b))
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(fold_a), size=len(fold_a))
        ib = rng.integers(0, len(fold_b), size=len(fold_b))
        diffs.append(np.mean(fold_b[ib]) - np.mean(fold_a[ia]))
    diffs = np.array(diffs)
    mean = float(np.mean(diffs))
    lo = float(np.percentile(diffs, 100 * (1 - ci) / 2))
    hi = float(np.percentile(diffs, 100 * (1 + ci) / 2))
    # two-sided p-value (proportion of bootstrap diffs at-or-beyond zero on opposite side)
    if mean > 0:
        p = float(np.mean(diffs <= 0))
    else:
        p = float(np.mean(diffs >= 0))
    p = max(p, 1.0 / n_boot)
    return mean, lo, hi, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = []
    for task in TASKS:
        for tgt in TARGETS:
            peak_v, fold_v = load_per_fold_at_peak(task, REF_VARIANT, tgt)
            if peak_v is None:
                continue
            for cmp_var, cmp_label in CMP_VARIANTS:
                peak_c, fold_c = load_per_fold_at_peak(task, cmp_var, tgt)
                if peak_c is None:
                    continue
                mean, lo, hi, pv = bootstrap_diff(fold_v, fold_c,
                                                  n_boot=args.n_boot, seed=args.seed)
                rows.append({
                    "task": task,
                    "target": tgt,
                    "ref_model": "V-JEPA A",
                    "cmp_model": cmp_label,
                    "ref_peak_layer": peak_v,
                    "cmp_peak_layer": peak_c,
                    "ref_mean": float(np.mean(fold_v)),
                    "cmp_mean": float(np.mean(fold_c)),
                    "gap_mean": mean,
                    "gap_ci_lo": lo,
                    "gap_ci_hi": hi,
                    "p_value": pv,
                    "n_folds_ref": len(fold_v),
                    "n_folds_cmp": len(fold_c),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}: {len(df)} rows")
    print()
    # Headlines: show -ve gap_mean and ci_hi < 0 (V-JEPA wins with CI)
    sig = df[(df["gap_mean"] < 0) & (df["gap_ci_hi"] < 0)]
    print(f"Significant V-JEPA wins ({len(sig)} of {len(df)}):")
    print(sig[["task", "target", "cmp_model", "gap_mean", "gap_ci_lo", "gap_ci_hi", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
