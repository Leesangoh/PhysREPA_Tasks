#!/usr/bin/env python3
"""Prepare compact per-layer episode matrices for later CKA.

This script converts token-patch safetensors into per-episode, per-layer
float16 matrices by averaging windows within each episode and flattening patch
tokens. It preserves the original representations while allowing large raw
token caches to be recycled for later experiments.
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
from safetensors import safe_open
from tqdm import tqdm

from probe_physprobe import MODEL_CONFIGS


def collect_episode_paths(feature_root: str, limit: int | None = None) -> list[str]:
    paths = sorted(glob(os.path.join(feature_root, "*.safetensors")))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No safetensors found in {feature_root}")
    return paths


def snapshot_task(task: str, model: str, feature_root: str, output_root: str, limit: int | None = None):
    cfg = MODEL_CONFIGS[model]
    num_layers = cfg["num_layers"]
    paths = collect_episode_paths(feature_root, limit=limit)

    out_dir = os.path.join(output_root, task)
    os.makedirs(out_dir, exist_ok=True)

    episode_ids = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in paths], dtype=np.int32)
    np.save(os.path.join(out_dir, "episode_ids.npy"), episode_ids)

    matrices = None
    for row, path in enumerate(tqdm(paths, desc=f"Snapshot CKA [{task}/{model}]")):
        with safe_open(path, framework="numpy") as f:
            keys = set(f.keys())
            window_starts = f.get_tensor("window_starts").astype(np.int64)
            n_w = len(window_starts)
            for layer in range(num_layers):
                vecs = []
                for w in range(n_w):
                    key = f"layer_{layer}_window_{w}"
                    if key not in keys:
                        raise ValueError(f"Missing {key} in {path}")
                    vecs.append(f.get_tensor(key))
                episode_feat = np.mean(vecs, axis=0)
                flat_feat = np.asarray(episode_feat.reshape(-1), dtype=np.float16)
                if matrices is None:
                    matrices = {
                        l: np.memmap(
                            os.path.join(out_dir, f"layer_{l:02d}.dat"),
                            mode="w+",
                            dtype=np.float16,
                            shape=(len(paths), flat_feat.shape[0]),
                        )
                        for l in range(num_layers)
                    }
                matrices[layer][row] = flat_feat

    if matrices is None:
        raise ValueError(f"No matrices created for {task}")

    meta = {
        "task": task,
        "model": model,
        "feature_root": feature_root,
        "num_episodes": len(paths),
        "num_layers": num_layers,
        "feature_dim": int(matrices[0].shape[1]),
    }
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        for key, value in meta.items():
            f.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare compact CKA snapshots from token-patch caches")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="large")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    snapshot_task(
        task=args.task,
        model=args.model,
        feature_root=args.feature_root,
        output_root=args.output_root,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
