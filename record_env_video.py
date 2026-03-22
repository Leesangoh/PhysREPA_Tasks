"""Record sample videos for environment verification.

Usage:
    cd /home/solee/IsaacLab
    PYTHONPATH=/home/solee:$PYTHONPATH ./isaaclab.sh -p \
        /home/solee/physrepa_tasks/record_env_video.py --task strike --num_episodes 1
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True,
                    choices=["push", "strike", "drawer", "reach"])
parser.add_argument("--num_episodes", type=int, default=1)
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

from isaaclab.envs import ManagerBasedRLEnv

FFMPEG_PATH = "/isaac-sim/kit/python/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"


def load_cfg(task_name):
    if task_name == "push":
        from physrepa_tasks.envs.push_env_cfg import PhysREPAPushEnvCfg
        return PhysREPAPushEnvCfg()
    elif task_name == "strike":
        from physrepa_tasks.envs.strike_env_cfg import PhysREPAStrikeEnvCfg
        return PhysREPAStrikeEnvCfg()
    elif task_name == "drawer":
        from physrepa_tasks.envs.drawer_env_cfg import PhysREPADrawerEnvCfg
        return PhysREPADrawerEnvCfg()
    elif task_name == "reach":
        from physrepa_tasks.envs.reach_env_cfg import PhysREPAReachEnvCfg
        return PhysREPAReachEnvCfg()
    else:
        raise ValueError(f"Unknown task: {task_name}")


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
    task_name = args.task
    cfg = load_cfg(task_name)
    cfg.scene.num_envs = 1

    env = ManagerBasedRLEnv(cfg=cfg)
    print(f"\n{'='*60}")
    print(f"Recording {args.num_episodes} episode(s) for: {task_name}")
    print(f"{'='*60}")

    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)
    action_dim = 6 if task_name == "reach" else 7

    out_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep} ---")
        obs, info = env.reset()

        # Warmup
        warmup_steps = int(0.5 / dt)
        for _ in range(warmup_steps):
            obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))

        max_steps = int(cfg.episode_length_s / dt) - warmup_steps
        frames_table = []
        frames_wrist = []

        for step in range(max_steps):
            # Zero action (just observe)
            action = torch.zeros(1, action_dim, device=env.device)
            obs, rew, term, trunc, info = env.step(action)

            # Collect camera frames
            policy_obs = obs["policy"]
            if isinstance(policy_obs, dict):
                if "table_cam" in policy_obs:
                    img = policy_obs["table_cam"][0].cpu().numpy()
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    frames_table.append(img)
                if "wrist_cam" in policy_obs:
                    img = policy_obs["wrist_cam"][0].cpu().numpy()
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    frames_wrist.append(img)

            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, reward={rew[0].item():.4f}")

        # Save videos
        if frames_table:
            path = os.path.join(out_dir, f"table_cam_ep{ep:02d}.mp4")
            encode_video(frames_table, path, fps)
            print(f"  Saved table_cam: {path} ({os.path.getsize(path)/1024:.1f} KB, {len(frames_table)} frames)")

        if frames_wrist:
            path = os.path.join(out_dir, f"wrist_cam_ep{ep:02d}.mp4")
            encode_video(frames_wrist, path, fps)
            print(f"  Saved wrist_cam: {path} ({os.path.getsize(path)/1024:.1f} KB, {len(frames_wrist)} frames)")

        # Save first frame as PNG for quick check
        if frames_table:
            imageio.imwrite(os.path.join(out_dir, f"table_cam_ep{ep:02d}.png"), frames_table[0])
        if frames_wrist:
            imageio.imwrite(os.path.join(out_dir, f"wrist_cam_ep{ep:02d}.png"), frames_wrist[0])

    print(f"\n{'='*60}")
    print(f"Done. Output: {out_dir}")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
