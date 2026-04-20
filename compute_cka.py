#!/usr/bin/env python3
"""Compute cross-task linear CKA from compact token-patch snapshots.

This implementation uses the feature-space linear CKA equivalent:

    CKA(X, Y) = || X Y^T ||_F^2 / (|| X X^T ||_F * || Y Y^T ||_F)

where X and Y are column-centered feature matrices. This form does not require
rowwise alignment between tasks and is therefore suitable for comparing
balanced episode sets drawn from different task distributions.
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SnapshotMeta:
    task: str
    model: str
    feature_root: str
    num_episodes: int
    num_layers: int
    feature_dim: int


def parse_meta(path: str) -> SnapshotMeta:
    data: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            data[key] = value
    return SnapshotMeta(
        task=data["task"],
        model=data["model"],
        feature_root=data["feature_root"],
        num_episodes=int(data["num_episodes"]),
        num_layers=int(data["num_layers"]),
        feature_dim=int(data["feature_dim"]),
    )


def load_snapshot_meta(snapshot_root: str, task: str) -> SnapshotMeta:
    return parse_meta(os.path.join(snapshot_root, task, "meta.txt"))


def memmap_layer(snapshot_root: str, task: str, layer: int, shape: tuple[int, int]) -> np.memmap:
    path = os.path.join(snapshot_root, task, f"layer_{layer:02d}.dat")
    return np.memmap(path, mode="r", dtype=np.float16, shape=shape)


def compute_feature_means(mm: np.memmap, n_rows: int, chunk_dim: int) -> np.ndarray:
    feat_dim = mm.shape[1]
    means = np.empty(feat_dim, dtype=np.float32)
    for start in range(0, feat_dim, chunk_dim):
        end = min(start + chunk_dim, feat_dim)
        block = np.asarray(mm[:n_rows, start:end], dtype=np.float32)
        means[start:end] = block.mean(axis=0)
    return means


def compute_grams(
    mm_x: np.memmap,
    mm_y: np.memmap,
    n_x: int,
    n_y: int,
    chunk_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_dim = mm_x.shape[1]
    mean_x = compute_feature_means(mm_x, n_x, chunk_dim)
    mean_y = compute_feature_means(mm_y, n_y, chunk_dim)

    k_x = np.zeros((n_x, n_x), dtype=np.float64)
    k_y = np.zeros((n_y, n_y), dtype=np.float64)
    c_xy = np.zeros((n_x, n_y), dtype=np.float64)

    for start in range(0, feat_dim, chunk_dim):
        end = min(start + chunk_dim, feat_dim)
        x_block = np.asarray(mm_x[:n_x, start:end], dtype=np.float32)
        y_block = np.asarray(mm_y[:n_y, start:end], dtype=np.float32)
        x_block -= mean_x[start:end]
        y_block -= mean_y[start:end]
        k_x += x_block @ x_block.T
        k_y += y_block @ y_block.T
        c_xy += x_block @ y_block.T

    return k_x, k_y, c_xy


def cka_from_grams(k_x: np.ndarray, k_y: np.ndarray, c_xy: np.ndarray) -> float:
    num = float(np.sum(c_xy * c_xy))
    den_x = float(np.linalg.norm(k_x, ord="fro"))
    den_y = float(np.linalg.norm(k_y, ord="fro"))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return num / (den_x * den_y)


def bootstrap_cka(
    k_x: np.ndarray,
    k_y: np.ndarray,
    c_xy: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_x, n_y = c_xy.shape
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx_x = rng.integers(0, n_x, size=n_x)
        idx_y = rng.integers(0, n_y, size=n_y)
        vals[b] = cka_from_grams(
            k_x[np.ix_(idx_x, idx_x)],
            k_y[np.ix_(idx_y, idx_y)],
            c_xy[np.ix_(idx_x, idx_y)],
        )
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def save_csv(rows: list[dict[str, object]], path: str) -> None:
    import csv

    if not rows:
        raise ValueError("No rows to save")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_line(rows: list[dict[str, object]], out_path: str) -> None:
    pair_to_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        pair_to_rows.setdefault(str(row["pair"]), []).append(row)

    plt.figure(figsize=(10, 6))
    for pair, prows in sorted(pair_to_rows.items()):
        prows = sorted(prows, key=lambda r: int(r["layer"]))
        layers = [int(r["layer"]) for r in prows]
        means = [float(r["cka_mean"]) for r in prows]
        los = [float(r["cka_ci_lo"]) for r in prows]
        his = [float(r["cka_ci_hi"]) for r in prows]
        plt.plot(layers, means, label=pair, linewidth=2)
        plt.fill_between(layers, los, his, alpha=0.18)

    for marker in [11, 12, 22]:
        plt.axvline(marker, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Linear CKA")
    plt.title("Cross-Task Linear CKA vs Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(rows: list[dict[str, object]], tasks: list[str], out_path: str) -> None:
    pairs = list(itertools.product(tasks, tasks))
    mat = np.full((len(pairs), 24), np.nan, dtype=np.float64)
    pair_names = [f"{a}-{b}" for a, b in pairs]
    index = {name: i for i, name in enumerate(pair_names)}
    for row in rows:
        name = str(row["pair"])
        if name not in index:
            continue
        mat[index[name], int(row["layer"])] = float(row["cka_mean"])

    plt.figure(figsize=(12, max(4, 0.7 * len(pairs))))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="Linear CKA")
    plt.yticks(range(len(pair_names)), pair_names)
    plt.xticks(range(24), range(24))
    plt.xlabel("Layer")
    plt.title("Cross-Task CKA Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute linear CKA from compact snapshots")
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-dim", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    metas = {task: load_snapshot_meta(args.snapshot_root, task) for task in args.tasks}
    num_layers = min(meta.num_layers for meta in metas.values())
    n_common = min(meta.num_episodes for meta in metas.values())
    feat_dim = {meta.feature_dim for meta in metas.values()}
    if len(feat_dim) != 1:
        raise ValueError(f"Feature dims differ across tasks: {sorted(feat_dim)}")

    rows: list[dict[str, object]] = []

    for layer in range(num_layers):
        grams_self: dict[str, np.ndarray] = {}
        for task in args.tasks:
            meta = metas[task]
            mm = memmap_layer(args.snapshot_root, task, layer, (meta.num_episodes, meta.feature_dim))
            k_self, _, _ = compute_grams(mm, mm, n_common, n_common, args.chunk_dim)
            grams_self[task] = k_self
            rows.append(
                {
                    "pair": f"{task}-{task}",
                    "layer": layer,
                    "cka_mean": 1.0,
                    "cka_ci_lo": 1.0,
                    "cka_ci_hi": 1.0,
                    "n_common": n_common,
                }
            )

        for task_a, task_b in itertools.combinations(args.tasks, 2):
            meta_a = metas[task_a]
            meta_b = metas[task_b]
            mm_a = memmap_layer(args.snapshot_root, task_a, layer, (meta_a.num_episodes, meta_a.feature_dim))
            mm_b = memmap_layer(args.snapshot_root, task_b, layer, (meta_b.num_episodes, meta_b.feature_dim))
            k_a, k_b, c_ab = compute_grams(mm_a, mm_b, n_common, n_common, args.chunk_dim)
            mean, lo, hi = bootstrap_cka(k_a, k_b, c_ab, args.bootstrap, args.seed + layer)
            for pair_name in (f"{task_a}-{task_b}", f"{task_b}-{task_a}"):
                rows.append(
                    {
                        "pair": pair_name,
                        "layer": layer,
                        "cka_mean": mean,
                        "cka_ci_lo": lo,
                        "cka_ci_hi": hi,
                        "n_common": n_common,
                    }
                )

    csv_path = os.path.join(args.output_root, "cka_cross_task.csv")
    line_path = os.path.join(args.output_root, "cka_cross_task_lines.png")
    heatmap_path = os.path.join(args.output_root, "cka_cross_task_heatmap.png")
    save_csv(rows, csv_path)
    plot_line(rows, line_path)
    plot_heatmap(rows, args.tasks, heatmap_path)


if __name__ == "__main__":
    main()
