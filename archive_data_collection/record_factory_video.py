"""Record video from Factory+Camera env."""
from __future__ import annotations

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=["peg_insert", "nut_thread"])
parser.add_argument("--output_dir", type=str, default="/mnt/md1/solee/data/isaac_physrepa_v2/env_test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import imageio

from physrepa_tasks.envs.factory_camera_env import (
    FactoryCameraEnv,
    PegInsertCameraCfg,
    NutThreadCameraCfg,
)

FFMPEG_PATH = "/isaac-sim/kit/python/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"


def encode_video(frames, output_path, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", ffmpeg_params=["-preset", "fast"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main():
    if args.task == "peg_insert":
        cfg = PegInsertCameraCfg()
    else:
        cfg = NutThreadCameraCfg()

    cfg.scene.num_envs = 1
    env = FactoryCameraEnv(cfg=cfg)

    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)
    max_steps = int(cfg.episode_length_s / dt)

    print(f"\n{'='*60}")
    print(f"Recording {args.task} — {max_steps} steps at {fps} FPS")
    print(f"{'='*60}")

    obs, info = env.reset()

    frames_table = []
    frames_wrist = []

    for step in range(max_steps):
        action = torch.zeros(1, 6, device=env.device)
        obs, rew, term, trunc, info = env.step(action)

        table_rgb, wrist_rgb = env.get_camera_data()
        frames_table.append(table_rgb[0].cpu().numpy().astype(np.uint8))
        frames_wrist.append(wrist_rgb[0].cpu().numpy().astype(np.uint8))

        if step % 100 == 0:
            phys = env.get_physics_data()
            ee_held = torch.norm(phys["ee_position"][0] - phys["held_position"][0]).item()
            print(f"  Step {step}/{max_steps}, ee-held={ee_held:.4f}m")

    out_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)

    table_path = os.path.join(out_dir, "factory_table_cam.mp4")
    wrist_path = os.path.join(out_dir, "factory_wrist_cam.mp4")
    encode_video(frames_table, table_path, fps)
    encode_video(frames_wrist, wrist_path, fps)
    print(f"Saved table_cam: {table_path} ({os.path.getsize(table_path)/1024:.1f} KB)")
    print(f"Saved wrist_cam: {wrist_path} ({os.path.getsize(wrist_path)/1024:.1f} KB)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
