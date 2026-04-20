#!/usr/bin/env python3
"""Extract token-level PhysProbe features with the PEZ best recipe.

Phase 2 target configuration:
- capture: resid_post
- transform: resize
- pooling: temporal_last_patch
- output per episode/window/layer as (n_patches, D) float16 tensors

This intentionally mirrors the PEZ extraction choices while using PhysProbe's
episode MP4 layout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
import numpy as np
from safetensors.torch import save_file
import torch
from torchvision import transforms
from tqdm import tqdm


N_FRAMES = 16
WINDOW_STRIDE = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DATA_BASE = "/home/solee/data/data/isaac_physrepa_v2/step0"
CHECKPOINT_DIR = "/mnt/md1/solee/checkpoints/vjepa2"
VJEPA2_ROOT = "/home/solee/vjepa2"
VJEPA2_SRC = "/home/solee/vjepa2/src"

MODEL_CONFIGS = {
    "large": {
        "factory": "vit_large",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "vitl.pt"),
        "embed_dim": 1024,
        "depth": 24,
        "img_size": 256,
    },
    "huge": {
        "factory": "vit_huge",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "vith.pt"),
        "embed_dim": 1280,
        "depth": 32,
        "img_size": 256,
    },
    "giant": {
        "factory": "vit_giant_xformers",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "vitg.pt"),
        "embed_dim": 1408,
        "depth": 40,
        "img_size": 256,
    },
}


def forward_resid_post(self, x, masks=None):
    """Capture raw residual stream after every transformer block."""
    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    if x.ndim == 4:
        _, _, height, width = x.shape
        tubelets = 1
    else:
        _, _, tubelets, height, width = x.shape
        tubelets = tubelets // self.tubelet_size

    h_patches = height // self.patch_size
    w_patches = width // self.patch_size
    if not self.handle_nonsquare_inputs:
        tubelets = h_patches = w_patches = None

    if not self.use_rope:
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        x = x + pos_embed
    else:
        x = self.patch_embed(x)

    if masks is not None:
        from src.masks.utils import apply_masks

        x = apply_masks(x, masks)
        masks = torch.cat(masks, dim=0)

    outs = []
    for block in self.blocks:
        x = block(
            x,
            mask=masks,
            attn_mask=None,
            T=tubelets,
            H_patches=h_patches,
            W_patches=w_patches,
        )
        outs.append(x.clone())
    return outs


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def derive_shuffle_seed(global_seed: int, episode_idx: int, window_start: int) -> int:
    """Deterministic per-window seed without relying on Python's randomized hash."""
    return (
        (global_seed * 1000003)
        + (episode_idx * 9176)
        + (window_start * 1315423911)
    ) & 0xFFFFFFFF


def load_model(model_name: str, device: str, random_init: bool = False, model_seed: int = 0):
    sys.path.insert(0, VJEPA2_ROOT)
    sys.path.insert(0, VJEPA2_SRC)
    import models.vision_transformer as vit_module

    cfg = MODEL_CONFIGS[model_name]
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    vit_factory = getattr(vit_module, cfg["factory"])
    model = vit_factory(
        patch_size=16,
        img_size=(cfg["img_size"], cfg["img_size"]),
        num_frames=64,
        tubelet_size=2,
        out_layers=list(range(cfg["depth"])),
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )
    if not random_init:
        checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        state = checkpoint.get("target_encoder", checkpoint)
        cleaned = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
        }
        model.load_state_dict(cleaned, strict=True)
    model.__class__.forward = forward_resid_post
    return model.to(device).eval(), cfg


def find_video_path(task: str, ep_idx: int, video_key: str = "observation.images.image_0") -> Path | None:
    video_dir = Path(DATA_BASE) / task / "videos"
    ep_filename = f"episode_{ep_idx:06d}.mp4"
    chunk_idx = ep_idx // 1000
    chunk_dir = video_dir / f"chunk-{chunk_idx:03d}"
    path = chunk_dir / video_key / ep_filename
    if path.exists():
        return path
    fallback = (video_dir / video_key / ep_filename)
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


def iter_episode_indices(task: str):
    feature_hint = Path("/mnt/md1/solee/features/physprobe_vitl") / task
    paths = sorted(feature_hint.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"No reference episodes found in {feature_hint}")
    return [int(path.stem) for path in paths]


def extract_episode_features(
    model,
    clip_frames: torch.Tensor,
    transform,
    num_layers: int,
    img_size: int,
    device: str,
    batch_size: int,
    episode_idx: int,
    shuffle_frames: bool = False,
    shuffle_seed: int = 42,
    debug_print_permutation: bool = False,
):
    if clip_frames.shape[0] < N_FRAMES:
        pad = N_FRAMES - clip_frames.shape[0]
        clip_frames = torch.cat([clip_frames[:1].expand(pad, -1, -1, -1), clip_frames], dim=0)

    window_starts = list(range(0, clip_frames.shape[0] - N_FRAMES + 1, WINDOW_STRIDE))
    if not window_starts:
        window_starts = [0]

    transformed = [transform(frame) for frame in clip_frames]
    transformed = torch.stack(transformed)

    outputs = {"window_starts": torch.tensor(window_starts, dtype=torch.int32)}
    spatial_grid = img_size // 16
    first_perm_logged = False

    for batch_start in range(0, len(window_starts), batch_size):
        batch_end = min(batch_start + batch_size, len(window_starts))
        clips = []
        for w_idx in range(batch_start, batch_end):
            start = window_starts[w_idx]
            clip_window = transformed[start : start + N_FRAMES]
            if shuffle_frames:
                local_seed = derive_shuffle_seed(shuffle_seed, episode_idx, start)
                permutation = np.random.default_rng(local_seed).permutation(N_FRAMES)
                if debug_print_permutation and not first_perm_logged:
                    print(
                        f"[shuffle-debug] episode={episode_idx:06d} window_start={start} "
                        f"seed={local_seed} permutation={permutation.tolist()}",
                        flush=True,
                    )
                    first_perm_logged = True
                clip_window = clip_window[torch.as_tensor(permutation, dtype=torch.long)]
            clip = clip_window.permute(1, 0, 2, 3)
            clips.append(clip)
        batch = torch.stack(clips, dim=0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            layer_outs = model(batch)

        for layer_idx in range(num_layers):
            tokens = layer_outs[layer_idx]
            grid = tokens.view(tokens.shape[0], N_FRAMES // 2, spatial_grid, spatial_grid, tokens.shape[-1])
            last_patch = grid[:, -1].reshape(tokens.shape[0], spatial_grid * spatial_grid, tokens.shape[-1])
            for local_idx in range(last_patch.shape[0]):
                outputs[f"layer_{layer_idx}_window_{batch_start + local_idx}"] = (
                    last_patch[local_idx].contiguous().half().cpu()
                )

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Extract token-patch PhysProbe features with PEZ recipe")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="large")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shuffle-frames", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--debug-print-permutation", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(
        args.model,
        device=device,
        random_init=args.random_init,
        model_seed=args.model_seed,
    )
    transform = build_transform(cfg["img_size"])

    episode_indices = iter_episode_indices(args.task)
    if args.episode_limit is not None:
        episode_indices = episode_indices[: args.episode_limit]

    out_dir = Path(args.output_root) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in tqdm(episode_indices, desc=f"Extract token features [{args.task}/{args.model}]"):
        out_path = out_dir / f"{ep_idx:06d}.safetensors"
        if out_path.exists() and not args.overwrite:
            continue
        video_path = find_video_path(args.task, ep_idx)
        if video_path is None:
            raise FileNotFoundError(f"Missing video for task={args.task} ep={ep_idx:06d}")
        frames = load_video_frames(video_path)
        episode_features = extract_episode_features(
            model=model,
            clip_frames=frames,
            transform=transform,
            num_layers=cfg["depth"],
            img_size=cfg["img_size"],
            device=device,
            batch_size=args.batch_size,
            episode_idx=ep_idx,
            shuffle_frames=args.shuffle_frames,
            shuffle_seed=args.shuffle_seed,
            debug_print_permutation=args.debug_print_permutation,
        )
        save_file(episode_features, str(out_path))


if __name__ == "__main__":
    main()
