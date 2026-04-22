"""Shim package exposing archived PhysREPA RL env registrations."""

from __future__ import annotations

from importlib import import_module
import sys

_RL_MODULES = [
    "push_rl_cfg",
    "pick_place_rl_cfg",
    "stack_rl_cfg",
    "strike_rl_cfg",
]

for _name in _RL_MODULES:
    _module = import_module(f"..archive_data_collection.rl_envs.{_name}", __package__)
    sys.modules[f"{__name__}.{_name}"] = _module

_agents = import_module("..archive_data_collection.rl_envs.agents", __package__)
sys.modules[f"{__name__}.agents"] = _agents

from ..archive_data_collection.rl_envs import *  # noqa: F401,F403,E402

