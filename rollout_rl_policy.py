"""Roll out trained RL policy with cameras and save video.

Supports:
- RSL-RL policies (Push, Strike, Drawer) — adds CameraCfg to scene
- RL-Games policies (PegInsert, NutThread) — uses Factory+Camera env
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
                    choices=["push", "strike", "drawer", "peg_insert", "nut_thread"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--output_dir", type=str, default="/mnt/md1/solee/data/isaac_physrepa_v2/rollout")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import imageio
import gymnasium as gym
import importlib

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

RSL_RL_TASKS = {
    "push": "PhysREPA-Push-Franka-v0",
    "strike": "PhysREPA-Strike-Franka-v0",
    "drawer": "Isaac-Open-Drawer-Franka-v0",
}

FACTORY_TASKS = {"peg_insert", "nut_thread"}


def encode_video(frames, output_path, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", ffmpeg_params=["-preset", "fast"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def add_cameras_to_cfg(env_cfg):
    """Add table_cam and wrist_cam to scene config."""
    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0, height=384, width=384, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.6, 0.0, 0.90), rot=(0.33900, -0.62054, -0.62054, 0.33900), convention="ros"
        ),
    )
    env_cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0, height=384, width=384, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )


def rollout_rsl_rl(task_name, gym_id, checkpoint, num_episodes, out_dir):
    """Rollout RSL-RL policy (Push, Strike, Drawer)."""
    import physrepa_tasks.rl_envs  # register envs
    import isaaclab_tasks  # register official envs (drawer)
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    spec = gym.spec(gym_id)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        mod_path, cls_name = env_cfg_entry.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        env_cfg = getattr(mod, cls_name)()
    else:
        env_cfg = env_cfg_entry()

    agent_cfg_entry = spec.kwargs["rsl_rl_cfg_entry_point"]
    mod_path, cls_name = agent_cfg_entry.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    agent_cfg = getattr(mod, cls_name)()

    env_cfg.scene.num_envs = 1
    add_cameras_to_cfg(env_cfg)
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(gym_id, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"Loading RSL-RL checkpoint: {checkpoint}")
    ppo_runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    ppo_runner.load(checkpoint)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env_cfg.sim.dt * env_cfg.decimation
    fps = int(1.0 / dt)
    max_steps = int(env_cfg.episode_length_s / dt)

    table_cam = env.unwrapped.scene.sensors["table_cam"]
    wrist_cam = env.unwrapped.scene.sensors["wrist_cam"]

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep} ---")
        obs, _ = env_wrapped.get_observations()
        frames_table = []
        total_reward = 0.0

        for step in range(max_steps):
            with torch.inference_mode():
                action = policy(obs)
                obs, rew, dones, infos = env_wrapped.step(action)
                total_reward += rew[0].item()

            table_rgb = table_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            frames_table.append(table_rgb)

            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, rew={rew[0].item():.3f}")
            if dones[0]:
                break

        print(f"  Total reward: {total_reward:.2f}, steps: {len(frames_table)}")

        path = os.path.join(out_dir, f"table_cam_ep{ep:02d}.mp4")
        encode_video(frames_table, path, fps)
        print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")

        for f_idx, label in [(0, "start"), (len(frames_table)//2, "mid"), (len(frames_table)-1, "end")]:
            if f_idx < len(frames_table):
                imageio.imwrite(os.path.join(out_dir, f"ep{ep:02d}_{label}.png"), frames_table[f_idx])

    env.close()


def rollout_factory(task_name, checkpoint, num_episodes, out_dir):
    """Rollout RL-Games policy in Factory+Camera env (PegInsert, NutThread)."""
    from physrepa_tasks.envs.factory_camera_env import (
        FactoryCameraEnv, PegInsertCameraCfg, NutThreadCameraCfg,
    )
    from rl_games.algos_torch import players
    from rl_games.torch_runner import Runner
    import yaml

    cfg = PegInsertCameraCfg() if task_name == "peg_insert" else NutThreadCameraCfg()
    cfg.scene.num_envs = 1
    env = FactoryCameraEnv(cfg=cfg)

    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)
    max_steps = int(cfg.episode_length_s / dt)

    print(f"Factory env created. Running with zero actions (checkpoint visualization TBD)")
    # Note: Loading rl_games checkpoint into Factory env requires the full rl_games config
    # For now, run with zero actions to verify camera works

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep} ---")
        obs, info = env.reset()
        frames_table = []

        for step in range(max_steps):
            action = torch.zeros(1, 6, device=env.device)
            obs, rew, term, trunc, info = env.step(action)

            table_rgb, wrist_rgb = env.get_camera_data()
            frames_table.append(table_rgb[0].cpu().numpy().astype(np.uint8))

            if step % 100 == 0:
                phys = env.get_physics_data()
                ee_held = torch.norm(phys["ee_position"][0] - phys["held_position"][0]).item()
                print(f"  Step {step}/{max_steps}, rew={rew[0].item():.3f}, ee-held={ee_held:.4f}")

        path = os.path.join(out_dir, f"table_cam_ep{ep:02d}.mp4")
        encode_video(frames_table, path, fps)
        print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")

        for f_idx, label in [(0, "start"), (len(frames_table)//2, "mid"), (len(frames_table)-1, "end")]:
            if f_idx < len(frames_table):
                imageio.imwrite(os.path.join(out_dir, f"ep{ep:02d}_{label}.png"), frames_table[f_idx])

    env.close()


def main():
    task_name = args.task
    out_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Rolling out {task_name} — {args.num_episodes} episodes")
    print(f"{'='*60}")

    if task_name in RSL_RL_TASKS:
        rollout_rsl_rl(task_name, RSL_RL_TASKS[task_name], args.checkpoint, args.num_episodes, out_dir)
    elif task_name in FACTORY_TASKS:
        rollout_factory(task_name, args.checkpoint, args.num_episodes, out_dir)

    print(f"\nDone. Output: {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
