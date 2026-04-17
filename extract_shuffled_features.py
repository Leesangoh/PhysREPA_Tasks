"""Frame-shuffled V-JEPA 2 feature extraction for PhysProbe.

Extracts V-JEPA 2 (ViT-L or ViT-G) features from PhysProbe episodes with
randomly permuted frame order within each 16-frame clip. Used to test whether
V-JEPA 2 encodes temporal causality (PEZ hypothesis):
  - If R^2 between shuffled vs original features is similar -> static visual
    correlation dominates (PEZ meaning weak)
  - If R^2 drops significantly -> model encodes temporal dynamics (PEZ meaning strong)

For each task, samples 300 episodes (reproducible via fixed seed) and saves
shuffled features + episode ID manifest.

Usage:
    # ViT-G on GPU 0
    /isaac-sim/python.sh extract_shuffled_features.py \
        --model_size giant --gpu_id 0 --tasks push strike peg_insert

    # ViT-L on GPU 2
    /isaac-sim/python.sh extract_shuffled_features.py \
        --model_size large --gpu_id 2 --tasks push strike peg_insert
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch
from torchvision import transforms
from tqdm import tqdm


# ── Model configurations ───────────────────────────────────────────────────

MODEL_CONFIGS = {
    "large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "img_size": 256,
        "default_out_layers": [8, 10, 12, 23],
        "extra_kwargs": {
            "use_sdpa": True,
            "use_SiLU": False,
            "wide_SiLU": True,
            "uniform_power": False,
            "use_rope": True,
        },
        "checkpoint_key": "target_encoder",
        "arch": "vit_large",
        "default_checkpoint": "/mnt/md1/solee/checkpoints/vjepa2/vitl.pt",
        "output_suffix": "vitl",
        "default_batch_size": 8,
    },
    "giant": {
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 22,
        "img_size": 384,
        "default_out_layers": [13, 16, 20, 39],
        "extra_kwargs": {
            "use_sdpa": True,
            "use_SiLU": False,
            "wide_SiLU": True,
            "uniform_power": False,
            "use_rope": True,
        },
        "checkpoint_key": "target_encoder",
        "arch": "vit_giant_xformers",
        "default_checkpoint": "/mnt/md1/solee/checkpoints/vjepa2/vitg-384.pt",
        "output_suffix": "vitg",
        "default_batch_size": 4,
    },
}

ALL_TASKS = ["push", "strike", "peg_insert", "nut_thread", "drawer", "reach"]


# ── Model loading ──────────────────────────────────────────────────────────

def load_vjepa2_model(checkpoint_path, model_size, out_layers, device):
    """Load V-JEPA 2 model (Large or Giant)."""
    import sys

    sys.path.insert(0, "/home/solee/vjepa2")
    sys.path.insert(0, "/home/solee/vjepa2/src")

    import models.vision_transformer as vit_module

    cfg = MODEL_CONFIGS[model_size]
    factory_fn = getattr(vit_module, cfg["arch"])
    model = factory_fn(
        patch_size=16,
        img_size=(cfg["img_size"], cfg["img_size"]),
        num_frames=64,
        tubelet_size=2,
        out_layers=out_layers,
        **cfg["extra_kwargs"],
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_key = cfg["checkpoint_key"]
    if ckpt_key in state_dict:
        state_dict = state_dict[ckpt_key]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    elif "encoder" in state_dict:
        state_dict = state_dict["encoder"]

    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()

    print(
        f"Loaded V-JEPA 2 {model_size}: embed_dim={cfg['embed_dim']}, "
        f"depth={cfg['depth']}, img_size={cfg['img_size']}, out_layers={out_layers}"
    )
    return model, cfg["img_size"]


# ── Video I/O ──────────────────────────────────────────────────────────────

def find_video_path(video_dir, ep_idx, video_key="observation.images.image_0"):
    """Find video file for an episode, supporting LeRobot chunked layout."""
    ep_filename = f"episode_{ep_idx:06d}.mp4"

    chunk_idx = ep_idx // 1000
    chunk_dir = video_dir / f"chunk-{chunk_idx:03d}"
    if chunk_dir.exists():
        video_path = chunk_dir / video_key / ep_filename
        if video_path.exists():
            return video_path

    if chunk_idx != 0:
        chunk0_dir = video_dir / "chunk-000"
        if chunk0_dir.exists():
            video_path = chunk0_dir / video_key / ep_filename
            if video_path.exists():
                return video_path

    video_path = video_dir / video_key / ep_filename
    if video_path.exists():
        return video_path

    video_path = video_dir / ep_filename
    if video_path.exists():
        return video_path

    return None


def load_video_frames(video_path):
    """Load all video frames via ffmpeg."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return None

    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])

    result = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return None

    raw = result.stdout
    frame_size = w * h * 3
    num_frames = len(raw) // frame_size
    if num_frames == 0:
        return None

    frames = np.frombuffer(raw[: num_frames * frame_size], dtype=np.uint8).reshape(
        num_frames, h, w, 3
    )
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)


# ── Feature extraction with frame shuffle ─────────────────────────────────

def extract_shuffled_features(
    model, frames, out_layers, shuffle_rng,
    window_size=16, window_stride=4, device="cuda", batch_size=1,
):
    """Extract V-JEPA 2 features with frame-shuffled clips.

    For each sliding window clip of `window_size` frames, the frame order is
    randomly permuted before feeding to the model. The shuffle permutation is
    drawn from `shuffle_rng` for reproducibility.
    """
    T = frames.shape[0]
    if T < window_size:
        pad = window_size - T
        frames = torch.cat([frames[:1].expand(pad, -1, -1, -1), frames], dim=0)
        T = window_size

    window_starts = list(range(0, T - window_size + 1, window_stride))
    if not window_starts:
        window_starts = [0]

    results = {}
    results["window_starts"] = torch.tensor(window_starts, dtype=torch.int32)

    # Pre-generate all shuffle permutations for reproducibility
    perms = []
    for _ in window_starts:
        perm = torch.randperm(window_size, generator=shuffle_rng)
        perms.append(perm)

    for batch_start in range(0, len(window_starts), batch_size):
        batch_end = min(batch_start + batch_size, len(window_starts))
        batch_clips = []
        for w_idx in range(batch_start, batch_end):
            start = window_starts[w_idx]
            clip = frames[start : start + window_size]  # (T, C, H, W)
            # Shuffle frame order
            clip = clip[perms[w_idx]]
            clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
            batch_clips.append(clip)

        batch = torch.stack(batch_clips).to(device)  # (B, C, T, H, W)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(batch)

        for b_offset in range(batch_end - batch_start):
            w_idx = batch_start + b_offset
            for layer_idx, layer_out in zip(out_layers, outputs):
                pooled = layer_out[b_offset].mean(dim=0).half().cpu()
                results[f"layer_{layer_idx}_window_{w_idx}"] = pooled

    # Store permutations for auditability
    perm_tensor = torch.stack(perms)  # (num_windows, window_size)
    results["shuffle_perms"] = perm_tensor.to(torch.int32)

    return results


# ── Episode sampling ───────────────────────────────────────────────────────

def get_num_episodes(dataset_path):
    """Get total episode count from dataset metadata."""
    meta_path = dataset_path / "meta"
    episodes_path = meta_path / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            return sum(1 for _ in f)
    else:
        import pandas as pd

        return len(pd.read_parquet(meta_path / "episodes.parquet"))


def sample_episode_ids(num_total, num_sample, seed):
    """Sample episode indices reproducibly."""
    rng = np.random.RandomState(seed)
    if num_sample >= num_total:
        return list(range(num_total))
    return sorted(rng.choice(num_total, size=num_sample, replace=False).tolist())


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract frame-shuffled V-JEPA 2 features for PhysProbe"
    )
    parser.add_argument(
        "--model_size", type=str, choices=["large", "giant"], default="giant",
        help="V-JEPA 2 model size: large (ViT-L, 1024d) or giant (ViT-G, 1408d)",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/home/solee/data/data/isaac_physrepa_v2/step0",
        help="Root of PhysProbe dataset (contains per-task subdirs)",
    )
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Root output dir (default: /mnt/md1/solee/features/physprobe_{vitl,vitg}_shuffled)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help=f"Tasks to process (default: all). Choices: {ALL_TASKS}",
    )
    parser.add_argument("--num_episodes", type=int, default=300, help="Episodes per task to sample")
    parser.add_argument("--out_layers", type=int, nargs="+", default=None)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None, help="Clips per GPU batch (default: 8 for L, 4 for G)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility")
    parser.add_argument("--video_key", type=str, default="observation.images.image_0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    tasks = args.tasks or ALL_TASKS
    out_layers = args.out_layers or cfg["default_out_layers"]
    checkpoint = args.checkpoint or cfg["default_checkpoint"]
    batch_size = args.batch_size or cfg["default_batch_size"]
    data_root = Path(args.data_root)
    device = f"cuda:{args.gpu_id}"

    if args.output_root is None:
        output_root = Path(f"/mnt/md1/solee/features/physprobe_{cfg['output_suffix']}_shuffled")
    else:
        output_root = Path(args.output_root)

    # Load model once
    print(f"Loading V-JEPA 2 {args.model_size} from {checkpoint}...")
    model, img_size = load_vjepa2_model(checkpoint, args.model_size, out_layers, device)

    normalize = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Deterministic RNG for frame shuffling (separate from episode sampling)
    shuffle_rng = torch.Generator()
    shuffle_rng.manual_seed(args.seed)

    model_tag = f"vjepa2_{cfg['output_suffix']}"

    for task in tasks:
        dataset_path = data_root / task
        if not dataset_path.exists():
            print(f"[SKIP] Task '{task}' not found at {dataset_path}")
            continue

        video_dir = dataset_path / "videos"
        num_total = get_num_episodes(dataset_path)

        # Sample episodes
        episode_ids = sample_episode_ids(num_total, args.num_episodes, seed=args.seed)
        print(f"\n{'='*60}")
        print(f"Task: {task} | Total episodes: {num_total} | Sampled: {len(episode_ids)}")
        print(f"Model: {model_tag} | GPU: {args.gpu_id} | Batch: {batch_size}")
        print(f"{'='*60}")

        # Create output dir and save manifest
        task_output_dir = output_root / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "task": task,
            "num_total_episodes": num_total,
            "num_sampled": len(episode_ids),
            "episode_ids": episode_ids,
            "seed": args.seed,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "out_layers": out_layers,
            "model": model_tag,
            "model_size": args.model_size,
            "checkpoint": str(checkpoint),
            "batch_size": batch_size,
        }
        manifest_path = task_output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to {manifest_path}")

        skipped = 0
        for ep_idx in tqdm(episode_ids, desc=f"[{task}] Extracting shuffled features"):
            out_path = task_output_dir / f"{ep_idx:06d}.safetensors"
            if out_path.exists() and not args.overwrite:
                continue

            video_path = find_video_path(video_dir, ep_idx, args.video_key)
            if video_path is None:
                print(f"  Video not found for episode {ep_idx}, skipping")
                skipped += 1
                continue

            frames = load_video_frames(video_path)
            if frames is None:
                print(f"  Failed to load episode {ep_idx}, skipping")
                skipped += 1
                continue

            frames = torch.stack([normalize(f) for f in frames])

            results = extract_shuffled_features(
                model, frames, out_layers, shuffle_rng,
                args.window_size, args.window_stride, device,
                batch_size=batch_size,
            )

            save_file(results, str(out_path))

        print(f"[{task}] Done. Extracted: {len(episode_ids) - skipped}, Skipped: {skipped}")

    print(f"\nAll done! Shuffled features saved under {output_root}/")


if __name__ == "__main__":
    main()
