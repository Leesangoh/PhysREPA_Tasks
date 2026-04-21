#!/usr/bin/env python3
"""Extract compact window-level features for offline action-OOD evaluation.

This extractor avoids re-materializing full token-patch caches. It stores
window-aligned, patch-mean feature vectors for selected layers:

- one `.safetensors` file per episode
- `window_starts`
- `layer_{L}` tensors with shape `(n_windows, D)`

Supported backbones:
- V-JEPA 2 Large
- VideoMAE-L
- DINOv2-L
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from safetensors.torch import save_file
import torch
from tqdm import tqdm

from extract_cross_model_features import (
    MODEL_CONFIGS as CROSS_MODEL_CONFIGS,
    build_transform as build_cross_transform,
    find_video_path,
    load_model as load_cross_model,
    load_video_frames,
)
from extract_token_features import (
    MODEL_CONFIGS as VJEPA_MODEL_CONFIGS,
    N_FRAMES,
    WINDOW_STRIDE,
    build_transform as build_vjepa_transform,
    load_model as load_vjepa_model,
)


TASK_REF_ROOT = "/mnt/md1/solee/features/physprobe_vitl"


def iter_episode_indices(task: str):
    ref_root = Path(TASK_REF_ROOT) / task
    paths = sorted(ref_root.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"No reference episodes found in {ref_root}")
    return [int(path.stem) for path in paths]


def parse_layers(num_layers: int, raw_layers: list[str]) -> list[int]:
    if not raw_layers or raw_layers == ["all"]:
        return list(range(num_layers))
    layers = sorted({int(x) for x in raw_layers})
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range for depth {num_layers}")
    return layers


def extract_vjepa_episode(model, cfg, frames, transform, layer_indices, device: str, batch_size: int):
    if frames.shape[0] < N_FRAMES:
        pad = N_FRAMES - frames.shape[0]
        frames = torch.cat([frames[:1].expand(pad, -1, -1, -1), frames], dim=0)

    window_starts = list(range(0, frames.shape[0] - N_FRAMES + 1, WINDOW_STRIDE))
    if not window_starts:
        window_starts = [0]

    transformed = torch.stack([transform(frame) for frame in frames])
    outputs = {"window_starts": torch.tensor(window_starts, dtype=torch.int32)}
    spatial_grid = cfg["img_size"] // 16
    accum = {layer: [] for layer in layer_indices}

    for batch_start in range(0, len(window_starts), batch_size):
        batch_end = min(batch_start + batch_size, len(window_starts))
        clips = []
        for w_idx in range(batch_start, batch_end):
            start = window_starts[w_idx]
            clip_window = transformed[start : start + N_FRAMES]
            clips.append(clip_window.permute(1, 0, 2, 3))
        batch = torch.stack(clips, dim=0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            layer_outs = model(batch)
        for layer_idx in layer_indices:
            tokens = layer_outs[layer_idx]
            grid = tokens.view(tokens.shape[0], N_FRAMES // 2, spatial_grid, spatial_grid, tokens.shape[-1])
            last_patch = grid[:, -1].reshape(tokens.shape[0], spatial_grid * spatial_grid, tokens.shape[-1])
            pooled = last_patch.mean(dim=1).contiguous().half().cpu()
            accum[layer_idx].append(pooled)

    for layer_idx in layer_indices:
        outputs[f"layer_{layer_idx}"] = torch.cat(accum[layer_idx], dim=0)
    return outputs


def extract_cross_episode(model, processor, cfg, frames, transform, layer_indices, device: str, batch_size: int):
    if frames.shape[0] < N_FRAMES:
        pad = N_FRAMES - frames.shape[0]
        frames = torch.cat([frames[:1].expand(pad, -1, -1, -1), frames], dim=0)

    window_starts = list(range(0, frames.shape[0] - N_FRAMES + 1, WINDOW_STRIDE))
    if not window_starts:
        window_starts = [0]

    transformed = torch.stack([transform(frame) for frame in frames])
    outputs = {"window_starts": torch.tensor(window_starts, dtype=torch.int32)}
    accum = {layer: [] for layer in layer_indices}
    spatial_grid = cfg["img_size"] // cfg["patch_size"]

    for batch_start in range(0, len(window_starts), batch_size):
        batch_end = min(batch_start + batch_size, len(window_starts))
        items = []
        for w_idx in range(batch_start, batch_end):
            start = window_starts[w_idx]
            clip_window = transformed[start : start + N_FRAMES]
            if cfg["mode"] == "video_last_patch":
                items.append(clip_window)
            elif cfg["mode"] == "image_last_frame":
                items.append(clip_window[-1])
            else:
                raise ValueError(f"Unknown cross-model mode: {cfg['mode']}")
        batch = torch.stack(items, dim=0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            outputs_obj = model(batch, output_hidden_states=True)
        hidden_states = outputs_obj.hidden_states[1:]

        for layer_idx in layer_indices:
            tokens = hidden_states[layer_idx]
            if cfg["mode"] == "video_last_patch":
                temporal_slices = cfg["temporal_slices"]
                grid = tokens.view(tokens.shape[0], temporal_slices, spatial_grid, spatial_grid, tokens.shape[-1])
                patch_tokens = grid[:, -1].reshape(tokens.shape[0], spatial_grid * spatial_grid, tokens.shape[-1])
            else:
                patch_tokens = tokens[:, 1:, :]
            pooled = patch_tokens.mean(dim=1).contiguous().half().cpu()
            accum[layer_idx].append(pooled)

    for layer_idx in layer_indices:
        outputs[f"layer_{layer_idx}"] = torch.cat(accum[layer_idx], dim=0)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Extract compact action-probe features")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", choices=["large", "videomae_large", "dinov2_large"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--layers", nargs="*", default=["all"])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    episode_indices = iter_episode_indices(args.task)
    if args.episode_limit is not None:
        episode_indices = episode_indices[: args.episode_limit]

    out_dir = Path(args.output_root) / args.model / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "large":
        model, cfg = load_vjepa_model("large", device=device)
        transform = build_vjepa_transform(cfg["img_size"])
        layer_indices = parse_layers(VJEPA_MODEL_CONFIGS["large"]["depth"], args.layers)
        extractor = lambda frames: extract_vjepa_episode(model, cfg, frames, transform, layer_indices, device, args.batch_size)
    else:
        model, processor, cfg = load_cross_model(args.model, device=device)
        transform = build_cross_transform(processor, cfg["img_size"])
        layer_indices = parse_layers(CROSS_MODEL_CONFIGS[args.model]["num_layers"], args.layers)
        extractor = lambda frames: extract_cross_episode(model, processor, cfg, frames, transform, layer_indices, device, args.batch_size)

    for ep_idx in tqdm(episode_indices, desc=f"Extract action features [{args.task}/{args.model}]"):
        out_path = out_dir / f"{ep_idx:06d}.safetensors"
        if out_path.exists() and not args.overwrite:
            continue
        video_path = find_video_path(args.task, ep_idx)
        if video_path is None:
            raise FileNotFoundError(f"Missing video for task={args.task} ep={ep_idx:06d}")
        frames = load_video_frames(video_path)
        episode_out = extractor(frames)
        save_file(episode_out, str(out_path))


if __name__ == "__main__":
    main()
