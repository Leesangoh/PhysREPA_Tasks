"""Evaluate RL policy and save video frames for visual inspection.

Usage:
    cd /home/solee/IsaacLab
    PYTHONPATH=/home/solee:$PYTHONPATH ./isaaclab.sh -p /home/solee/physrepa_tasks/eval_rl_policy.py \
        --task PhysREPA-Push-Franka-v0 \
        --checkpoint /path/to/model.pt \
        --num_episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
print = functools.partial(print, flush=True)

import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--output_dir", type=str, default="/mnt/md1/solee/data/isaac_physrepa/eval")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio
from rsl_rl.runners import OnPolicyRunner

import physrepa_tasks.rl_envs  # register tasks

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    # Parse env config
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env_cfg.episode_length_s = 10.0

    # Add camera to the RL env for visualization
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.6, 0.0, 0.90), rot=(0.33900, -0.62054, -0.62054, 0.33900), convention="ros"
        ),
    )
    env_cfg.scene.env_spacing = 5.0
    env_cfg.rerender_on_reset = True

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Load policy
    print(f"Loading checkpoint: {args_cli.checkpoint}")
    policy = torch.jit.load(args_cli.checkpoint, map_location="cuda:0") if args_cli.checkpoint.endswith(".jit") else None

    if policy is None:
        # Load RSL-RL model
        checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
        # Extract actor network
        from rsl_rl.modules import ActorCritic
        # Get obs/action dims
        obs_dim = env.observation_space["policy"].shape[1] if hasattr(env.observation_space["policy"], "shape") else sum(
            v.shape[-1] for v in env.observation_space["policy"].values()
        ) if isinstance(env.observation_space["policy"], dict) else env.observation_space["policy"].shape[0]

        act_dim = env.action_space.shape[-1]
        print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

        actor_critic = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=act_dim,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ).to("cuda:0")
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        actor_critic.eval()

    task_name = args_cli.task.split("-")[1].lower()  # e.g., "push"
    output_dir = os.path.join(args_cli.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(args_cli.num_episodes):
        print(f"\n--- Episode {ep} ---")
        obs, _ = env.reset()
        frames = []

        # Warmup
        for _ in range(25):
            obs, _, _, _, _ = env.step(torch.zeros(1, act_dim, device="cuda:0"))

        max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        for step in range(max_steps):
            # Get observation tensor
            if isinstance(obs["policy"], dict):
                obs_tensor = torch.cat([v.float().flatten(1) for k, v in sorted(obs["policy"].items()) if not k.endswith("_cam")], dim=1)
            else:
                obs_tensor = obs["policy"].float()

            # Get action from policy
            with torch.no_grad():
                action = actor_critic.act_inference(obs_tensor)

            obs, reward, terminated, truncated, info = env.step(action)

            # Save frame every 5 steps
            if step % 5 == 0:
                cam_data = env.scene.sensors.get("table_cam")
                if cam_data is not None:
                    img = cam_data.data.output["rgb"][0].cpu().numpy()
                    frames.append(img)

            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, reward={reward[0].item():.4f}")

        # Save video
        if frames:
            video_path = os.path.join(output_dir, f"episode_{ep:03d}.mp4")
            writer = imageio.get_writer(video_path, fps=10, codec="libx264", quality=8, pixelformat="yuv420p")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Saved: {video_path} ({len(frames)} frames)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
