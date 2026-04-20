#!/usr/bin/env python3
"""Explore kinematic contact surrogates from PhysProbe parquet traces.

This script is intentionally lightweight and read-only. It inspects task-level
parquet logs and estimates whether contact-like events can be recovered from
kinematic spikes even when the exported `contact_*` GT fields are zero-filled.

Outputs:
- artifacts/results/contact_inference_summary.json
- artifacts/results/contact_inference_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob
from typing import Any

import numpy as np
import pandas as pd


DATA_BASE = "/home/solee/data/data/isaac_physrepa_v2/step0"
RESULTS_BASE = "/home/solee/physrepa_tasks/artifacts/results"

TASK_CONFIGS = {
    "push": {
        "entity_pos": "physics_gt.object_position",
        "entity_vel": "physics_gt.object_velocity",
        "entity_acc": "physics_gt.object_acceleration",
        "ee_pos": "physics_gt.ee_position",
        "aux_scalar": "physics_gt.ee_to_object_distance",
        "aux_name": "ee_to_object_distance",
    },
    "strike": {
        "entity_pos": "physics_gt.object_position",
        "entity_vel": "physics_gt.object_velocity",
        "entity_acc": "physics_gt.object_acceleration",
        "ee_pos": "physics_gt.ee_position",
        "aux_scalar": "physics_gt.ee_to_object_distance",
        "aux_name": "ee_to_object_distance",
    },
    "peg_insert": {
        "entity_pos": "physics_gt.peg_position",
        "entity_vel": "physics_gt.peg_velocity",
        "entity_acc": None,
        "ee_pos": "physics_gt.ee_position",
        "aux_scalar": "physics_gt.insertion_depth",
        "aux_name": "insertion_depth",
        "aux_scalar_2": "physics_gt.peg_hole_lateral_error",
        "aux_name_2": "peg_hole_lateral_error",
    },
    "nut_thread": {
        "entity_pos": "physics_gt.nut_position",
        "entity_vel": "physics_gt.nut_velocity",
        "entity_acc": None,
        "ee_pos": "physics_gt.ee_position",
        "aux_scalar": "physics_gt.axial_progress",
        "aux_name": "axial_progress",
        "aux_scalar_2": "physics_gt.nut_bolt_relative_angle",
        "aux_name_2": "nut_bolt_relative_angle",
    },
    "drawer": {
        "entity_pos": "physics_gt.handle_position",
        "entity_vel": "physics_gt.handle_velocity",
        "entity_acc": None,
        "ee_pos": "physics_gt.ee_position",
        "aux_scalar": "physics_gt.drawer_joint_pos",
        "aux_name": "drawer_joint_pos",
        "aux_scalar_2": "physics_gt.drawer_opening_extent",
        "aux_name_2": "drawer_opening_extent",
    },
}


def list_parquet_paths(task: str) -> list[str]:
    data_dir = os.path.join(DATA_BASE, task, "data")
    paths: list[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".parquet"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def load_array_series(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    values = df[col].values
    if len(values) == 0:
        return None
    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v, dtype=np.float64) for v in values])
    return np.asarray(values, dtype=np.float64)


def nonzero_fraction(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(arr) > 1e-8) / arr.size)


def finite_diff(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] <= 1:
        return np.zeros_like(arr)
    return np.diff(arr, axis=0, prepend=arr[:1])


def compute_angle_jumps(vel: np.ndarray) -> np.ndarray:
    if vel is None or vel.shape[0] <= 1:
        return np.zeros((1,), dtype=np.float64)
    speed = np.linalg.norm(vel, axis=1)
    unit = np.zeros_like(vel)
    nz = speed > 1e-8
    unit[nz] = vel[nz] / speed[nz, None]
    dots = np.sum(unit[1:] * unit[:-1], axis=1)
    valid = nz[1:] & nz[:-1]
    dots = np.where(valid, dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def summarize_series(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def proxy_score(maxima: list[float], baselines: list[float]) -> float:
    if not maxima or not baselines:
        return 0.0
    m = float(np.median(maxima))
    b = float(np.median(baselines))
    return m / max(b, 1e-6)


def analyze_task(task: str, episode_limit: int | None) -> dict[str, Any]:
    cfg = TASK_CONFIGS[task]
    paths = list_parquet_paths(task)
    if episode_limit is not None:
        paths = paths[:episode_limit]

    contact_stats: dict[str, list[float]] = {}
    direct_acc_max: list[float] = []
    direct_acc_base: list[float] = []
    fd_acc_max: list[float] = []
    fd_acc_base: list[float] = []
    pos_acc_max: list[float] = []
    pos_acc_base: list[float] = []
    angle_jump_max: list[float] = []
    angle_jump_p95: list[float] = []
    dist_d_min: list[float] = []
    dist_d_p05: list[float] = []
    aux_d_abs_max: dict[str, list[float]] = {}
    aux_d_abs_p95: dict[str, list[float]] = {}
    entity_zero_position = 0
    entity_zero_velocity = 0

    for path in paths:
        df = pd.read_parquet(path)

        for col in [c for c in df.columns if "contact" in c]:
            arr = load_array_series(df, col)
            if arr is None:
                continue
            contact_stats.setdefault(col, []).append(nonzero_fraction(arr))

        entity_pos = load_array_series(df, cfg["entity_pos"])
        entity_vel = load_array_series(df, cfg["entity_vel"])
        ee_pos = load_array_series(df, cfg["ee_pos"])
        direct_acc = load_array_series(df, cfg["entity_acc"]) if cfg.get("entity_acc") else None

        if entity_pos is not None and float(np.max(np.abs(entity_pos))) < 1e-8:
            entity_zero_position += 1
        if entity_vel is not None and float(np.max(np.abs(entity_vel))) < 1e-8:
            entity_zero_velocity += 1

        if direct_acc is not None:
            amag = np.linalg.norm(direct_acc, axis=1)
            direct_acc_max.append(float(amag.max()))
            direct_acc_base.append(float(np.median(amag)))

        if entity_vel is None and entity_pos is not None:
            entity_vel = finite_diff(entity_pos)

        if entity_vel is not None:
            fd_acc = finite_diff(entity_vel)
            fd_mag = np.linalg.norm(fd_acc, axis=1)
            fd_acc_max.append(float(fd_mag.max()))
            fd_acc_base.append(float(np.median(fd_mag)))

            angle_jumps = compute_angle_jumps(entity_vel)
            angle_jump_max.append(float(angle_jumps.max()))
            angle_jump_p95.append(float(np.percentile(angle_jumps, 95)))

        if entity_pos is not None:
            pos_acc = finite_diff(finite_diff(entity_pos))
            pos_mag = np.linalg.norm(pos_acc, axis=1)
            pos_acc_max.append(float(pos_mag.max()))
            pos_acc_base.append(float(np.median(pos_mag)))

        if entity_pos is not None and ee_pos is not None and float(np.max(np.abs(entity_pos))) > 1e-8:
            dist = np.linalg.norm(entity_pos - ee_pos, axis=1)
            d_dist = np.diff(dist)
            if len(d_dist):
                dist_d_min.append(float(d_dist.min()))
                dist_d_p05.append(float(np.percentile(d_dist, 5)))

        for key in ["aux_scalar", "aux_scalar_2"]:
            if not cfg.get(key):
                continue
            aux = load_array_series(df, cfg[key])
            if aux is None:
                continue
            aux = np.asarray(aux, dtype=np.float64).reshape(-1)
            d_aux = np.diff(aux)
            if not len(d_aux):
                continue
            name = cfg[key.replace("scalar", "name")]
            aux_d_abs_max.setdefault(name, []).append(float(np.max(np.abs(d_aux))))
            aux_d_abs_p95.setdefault(name, []).append(float(np.percentile(np.abs(d_aux), 95)))

    recommendations: list[dict[str, Any]] = []
    if direct_acc_max:
        score = proxy_score(direct_acc_max, direct_acc_base)
        if np.median(direct_acc_max) > 1e-8:
            recommendations.append(
                {
                    "proxy": "direct_acceleration_magnitude",
                    "score": score,
                    "evidence": "highest if exported acceleration contains impulse spikes",
                }
            )
    if fd_acc_max:
        score = proxy_score(fd_acc_max, fd_acc_base)
        if np.median(fd_acc_max) > 1e-8:
            recommendations.append(
                {
                    "proxy": "finite_diff_velocity_acceleration_magnitude",
                    "score": score,
                    "evidence": "usable when direct acceleration is absent or zero-filled",
                }
            )
    if pos_acc_max:
        score = proxy_score(pos_acc_max, pos_acc_base)
        if np.median(pos_acc_max) > 1e-8:
            recommendations.append(
                {
                    "proxy": "second_diff_position_acceleration_magnitude",
                    "score": score,
                    "evidence": "fallback when velocity is unreliable",
                }
            )
    if angle_jump_max and entity_zero_velocity < len(paths):
        if np.median(angle_jump_max) > 1e-4:
            recommendations.append(
                {
                    "proxy": "velocity_direction_jump",
                    "score": float(np.median(angle_jump_max)),
                    "evidence": "high values indicate collision-induced reorientation/reversal",
                }
            )
    if dist_d_min and entity_zero_position < len(paths):
        recommendations.append(
            {
                "proxy": "ee_entity_distance_derivative",
                "score": abs(float(np.median(dist_d_min))),
                "evidence": "approach/compression signal, best as support not sole label",
            }
        )
    for aux_name, vals in aux_d_abs_max.items():
        recommendations.append(
            {
                "proxy": f"{aux_name}_delta",
                "score": float(np.median(vals)),
                "evidence": "task-specific progress/contact surrogate",
            }
        )
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return {
        "task": task,
        "episodes_analyzed": len(paths),
        "contact_gt_nonzero_fraction": {k: float(np.mean(v)) for k, v in sorted(contact_stats.items())},
        "entity_zero_position_episodes": int(entity_zero_position),
        "entity_zero_velocity_episodes": int(entity_zero_velocity),
        "direct_acceleration": {
            "max": summarize_series(direct_acc_max),
            "baseline_median": summarize_series(direct_acc_base),
            "proxy_score": proxy_score(direct_acc_max, direct_acc_base),
        },
        "finite_diff_acceleration": {
            "max": summarize_series(fd_acc_max),
            "baseline_median": summarize_series(fd_acc_base),
            "proxy_score": proxy_score(fd_acc_max, fd_acc_base),
        },
        "position_second_diff": {
            "max": summarize_series(pos_acc_max),
            "baseline_median": summarize_series(pos_acc_base),
            "proxy_score": proxy_score(pos_acc_max, pos_acc_base),
        },
        "direction_jump": {
            "max": summarize_series(angle_jump_max),
            "p95": summarize_series(angle_jump_p95),
        },
        "distance_derivative": {
            "min": summarize_series(dist_d_min),
            "p05": summarize_series(dist_d_p05),
        },
        "aux_derivatives": {
            name: {
                "max_abs": summarize_series(vals),
                "p95_abs": summarize_series(aux_d_abs_p95.get(name, [])),
            }
            for name, vals in sorted(aux_d_abs_max.items())
        },
        "recommended_proxies": recommendations[:4],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore contact proxy inference from PhysProbe kinematics")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_CONFIGS),
        choices=sorted(TASK_CONFIGS),
    )
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument(
        "--output-json",
        default=os.path.join(RESULTS_BASE, "contact_inference_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(RESULTS_BASE, "contact_inference_summary.csv"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    summaries = [analyze_task(task, args.episode_limit) for task in args.tasks]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        task = summary["task"]
        for rec in summary["recommended_proxies"]:
            rows.append(
                {
                    "task": task,
                    "proxy": rec["proxy"],
                    "score": rec["score"],
                    "evidence": rec["evidence"],
                    "episodes_analyzed": summary["episodes_analyzed"],
                    "contact_gt_max_nonzero_fraction": max(summary["contact_gt_nonzero_fraction"].values() or [0.0]),
                }
            )
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "proxy",
                "score",
                "evidence",
                "episodes_analyzed",
                "contact_gt_max_nonzero_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
