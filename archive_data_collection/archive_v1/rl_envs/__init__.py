# Copyright (c) 2024, PhysREPA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""PhysREPA RL training environments: Pick-Place, Push, and Stack.

Registration of Gymnasium environments for use with Isaac Lab's RL training scripts.
"""

import gymnasium as gym

from . import agents  # noqa: F401

##
# Pick-Place
##

gym.register(
    id="PhysREPA-PickPlace-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "physrepa_tasks.rl_envs.pick_place_rl_cfg:PickPlaceRLEnvCfg",
        "rsl_rl_cfg_entry_point": "physrepa_tasks.rl_envs.agents.rsl_rl_pick_place_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="PhysREPA-PickPlace-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "physrepa_tasks.rl_envs.pick_place_rl_cfg:PickPlaceRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "physrepa_tasks.rl_envs.agents.rsl_rl_pick_place_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Push
##

gym.register(
    id="PhysREPA-Push-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "physrepa_tasks.rl_envs.push_rl_cfg:PushRLEnvCfg",
        "rsl_rl_cfg_entry_point": "physrepa_tasks.rl_envs.agents.rsl_rl_push_cfg:PushPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="PhysREPA-Push-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "physrepa_tasks.rl_envs.push_rl_cfg:PushRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "physrepa_tasks.rl_envs.agents.rsl_rl_push_cfg:PushPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Stack
##

gym.register(
    id="PhysREPA-Stack-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "physrepa_tasks.rl_envs.stack_rl_cfg:StackRLEnvCfg",
        "rsl_rl_cfg_entry_point": "physrepa_tasks.rl_envs.agents.rsl_rl_stack_cfg:StackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="PhysREPA-Stack-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "physrepa_tasks.rl_envs.stack_rl_cfg:StackRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "physrepa_tasks.rl_envs.agents.rsl_rl_stack_cfg:StackPPORunnerCfg",
    },
    disable_env_checker=True,
)
