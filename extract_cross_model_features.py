#!/usr/bin/env python3
"""Extract token-patch PhysProbe features for non-V-JEPA baselines.

Stage 1 supports VideoMAE-L and DINOv2-L with the same downstream contract used by
`probe_physprobe.py`:

- one `.safetensors` file per episode
- `window_starts`
- `layer_{L}_window_{W}` tensors with shape `(n_patches, D)`

The extraction recipe follows the cross-model fairness plan:
- video backbones receive the full 16-frame window
- image backbones receive only the last frame of each 16-frame window
- per-block hidden states are used
- patch tokens are preserved
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model, VideoMAEImageProcessor, VideoMAEModel


N_FRAMES = 16
WINDOW_STRIDE = 4
DATA_BASE = "/home/solee/data/data/isaac_physrepa_v2/step0"
FEATURE_HINT_BASE = "/mnt/md1/solee/features/physprobe_vitl"

MODEL_CONFIGS = {
    "dinov2_large": {
        "checkpoint": "/mnt/md1/solee/checkpoints/cross_model/dinov2-large",
        "num_layers": 24,
        "dim": 1024,
        "img_size": 224,
        "patch_size": 14,
        "mode": "image_last_frame",
    },
    "videomae_large": {
        "checkpoint": "/mnt/md1/solee/checkpoints/cross_model/videomae-large",
        "num_layers": 24,
        "dim": 1024,
        "img_size": 224,
        "patch_size": 16,
        "tubelet_size": 2,
        "temporal_slices": 8,
        "mode": "video_last_patch",
    },
}


def resolve_resize_and_crop(processor, fallback: int) -> tuple[int, int]:
    resize_size = int(fallback)
    crop_size = int(fallback)
    if hasattr(processor, "size") and isinstance(processor.size, dict):
        for key in ("shortest_edge", "height", "width"):
            if key in processor.size:
                resize_size = int(processor.size[key])
                break
    if hasattr(processor, "crop_size") and isinstance(processor.crop_size, dict):
        for key in ("height", "shortest_edge", "width"):
            if key in processor.crop_size:
                crop_size = int(processor.crop_size[key])
                break
    return resize_size, crop_size


def build_transform(processor, img_size: int):
    resize_size, crop_size = resolve_resize_and_crop(processor, img_size)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


def load_model(model_name: str, device: str):
    cfg = MODEL_CONFIGS[model_name]
    if model_name == "videomae_large":
        processor = VideoMAEImageProcessor.from_pretrained(cfg["checkpoint"])
        model = VideoMAEModel.from_pretrained(cfg["checkpoint"]).to(device).eval()
    elif model_name == "dinov2_large":
        processor = AutoImageProcessor.from_pretrained(cfg["checkpoint"])
        model = Dinov2Model.from_pretrained(cfg["checkpoint"]).to(device).eval()
    else:
        raise ValueError(f"Unsupported cross-model extractor target: {model_name}")
    return model, processor, cfg


def find_video_path(task: str, ep_idx: int, data_base: str = DATA_BASE, video_key: str = "observation.images.image_0") -> Path | None:
    video_dir = Path(data_base) / task / "videos"
    ep_filename = f"episode_{ep_idx:06d}.mp4"
    chunk_idx = ep_idx // 1000
    chunk_dir = video_dir / f"chunk-{chunk_idx:03d}"
    path = chunk_dir / video_key / ep_filename
    if path.exists():
        return path
    fallback = video_dir / video_key / ep_filename
    if fallback.exists():
        return fallback
    return None


def load_video_frames(video_path: Path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {video_path}")
    width, height = map(int, probe.stdout.strip().split(","))

    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-v",
            "error",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")

    raw = result.stdout
    frame_size = width * height * 3
    num_frames = len(raw) // frame_size
    if num_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    frames = np.frombuffer(raw[: num_frames * frame_size], dtype=np.uint8).reshape(num_frames, height, width, 3)
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)


def iter_episode_indices(task: str, data_base: str = DATA_BASE, video_key: str = "observation.images.image_0"):
    video_dir = Path(data_base) / task / "videos"
    patterns = [
        video_dir / "chunk-*" / video_key / "episode_*.mp4",
        video_dir / video_key / "episode_*.mp4",
    ]
    episodes = set()
    for pattern in patterns:
        for path in sorted(video_dir.glob(str(pattern.relative_to(video_dir)))):
            episodes.add(int(path.stem.split("_")[-1]))
    if not episodes:
        raise ValueError(f"No episode videos found in {video_dir}")
    return sorted(episodes)


def extract_episode_features(
    model,
    clip_frames: torch.Tensor,
    transform,
    cfg: dict,
    device: str,
    batch_size: int,
):
    if clip_frames.shape[0] < N_FRAMES:
        pad = N_FRAMES - clip_frames.shape[0]
        clip_frames = torch.cat([clip_frames[:1].expand(pad, -1, -1, -1), clip_frames], dim=0)

    window_starts = list(range(0, clip_frames.shape[0] - N_FRAMES + 1, WINDOW_STRIDE))
    if not window_starts:
        window_starts = [0]

    transformed = torch.stack([transform(frame) for frame in clip_frames])
    outputs = {"window_starts": torch.tensor(window_starts, dtype=torch.int32)}

    num_layers = cfg["num_layers"]
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
                raise ValueError(f"Unknown model mode: {cfg['mode']}")
        batch = torch.stack(items, dim=0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            outputs_obj = model(batch, output_hidden_states=True)

        hidden_states = outputs_obj.hidden_states[1:]
        if len(hidden_states) != num_layers:
            raise ValueError(f"Expected {num_layers} hidden states, got {len(hidden_states)}")

        for layer_idx, tokens in enumerate(hidden_states):
            if cfg["mode"] == "video_last_patch":
                temporal_slices = cfg["temporal_slices"]
                grid = tokens.view(tokens.shape[0], temporal_slices, spatial_grid, spatial_grid, tokens.shape[-1])
                patch_tokens = grid[:, -1].reshape(tokens.shape[0], spatial_grid * spatial_grid, tokens.shape[-1])
            else:
                patch_tokens = tokens[:, 1:, :]
                if patch_tokens.shape[1] != spatial_grid * spatial_grid:
                    raise ValueError(
                        f"Expected {(spatial_grid * spatial_grid)} image patches, got {patch_tokens.shape[1]}"
                    )
            for local_idx in range(patch_tokens.shape[0]):
                outputs[f"layer_{layer_idx}_window_{batch_start + local_idx}"] = (
                    patch_tokens[local_idx].contiguous().half().cpu()
                )
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Extract token-patch PhysProbe features for cross-model baselines")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="videomae_large")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--data-base", default=DATA_BASE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, processor, cfg = load_model(args.model, device=device)
    transform = build_transform(processor, cfg["img_size"])

    episode_indices = iter_episode_indices(args.task, data_base=args.data_base)
    if args.episode_limit is not None:
        episode_indices = episode_indices[: args.episode_limit]

    out_dir = Path(args.output_root) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in tqdm(episode_indices, desc=f"Extract cross-model features [{args.task}/{args.model}]"):
        out_path = out_dir / f"{ep_idx:06d}.safetensors"
        if out_path.exists() and not args.overwrite:
            continue
        video_path = find_video_path(args.task, ep_idx, data_base=args.data_base)
        if video_path is None:
            raise FileNotFoundError(f"Missing video for task={args.task} ep={ep_idx:06d}")
        frames = load_video_frames(video_path)
        episode_features = extract_episode_features(
            model=model,
            clip_frames=frames,
            transform=transform,
            cfg=cfg,
            device=device,
            batch_size=args.batch_size,
        )
        save_file(episode_features, str(out_path))


if __name__ == "__main__":
    main()
