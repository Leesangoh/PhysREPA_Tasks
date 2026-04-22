"""Shim package exposing archived PhysREPA environment configs."""

from __future__ import annotations

from importlib import import_module
import sys

_ENV_MODULES = [
    "lift_env_cfg",
    "pick_place_env_cfg",
    "push_env_cfg",
    "stack_env_cfg",
    "strike_env_cfg",
    "drawer_env_cfg",
    "reach_env_cfg",
    "factory_camera_env",
    "peg_insert_env_cfg",
    "nut_thread_env_cfg",
]

for _name in _ENV_MODULES:
    _module = import_module(f"..archive_data_collection.envs.{_name}", __package__)
    sys.modules[f"{__name__}.{_name}"] = _module

