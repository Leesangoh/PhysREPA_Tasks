"""Shim package exposing archived PhysREPA RL agent configs."""

from __future__ import annotations

from importlib import import_module
import sys

for _name in (
    "rsl_rl_pick_place_cfg",
    "rsl_rl_push_cfg",
    "rsl_rl_stack_cfg",
    "rsl_rl_strike_cfg",
):
    _module = import_module(f"...archive_data_collection.rl_envs.agents.{_name}", __package__)
    sys.modules[f"{__name__}.{_name}"] = _module

