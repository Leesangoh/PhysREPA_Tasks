"""Verify Factory PegInsert and NutThread environments."""
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401  — registers gym envs
import gymnasium as gym
import torch

GYM_IDS = {
    "peg_insert": "Isaac-Factory-PegInsert-Direct-v0",
    "nut_thread": "Isaac-Factory-NutThread-Direct-v0",
}


def main():
    gym_id = GYM_IDS[args.task]

    # Get env config from registry
    env_cfg = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    if callable(env_cfg):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = 1
    env = gym.make(gym_id, cfg=env_cfg)
    print(f"\n{'='*60}")
    print(f"Factory environment '{args.task}' ({gym_id}) loaded!")
    print(f"  obs_space: {env.observation_space.shape}")
    print(f"  act_space: {env.action_space.shape}")
    print(f"{'='*60}")

    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    for i in range(10):
        action = torch.zeros(1, env.action_space.shape[-1], device="cuda:0")
        obs, rew, term, trunc, info = env.step(action)
        if i % 5 == 0:
            print(f"  Step {i}: reward={rew[0].item():.4f}")

    print(f"\nFactory '{args.task}' test PASSED!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
