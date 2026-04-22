"""Shim package exposing archived PhysREPA utility modules."""

from __future__ import annotations

from importlib import import_module
import sys

_rl_games_policy = import_module("..archive_data_collection.utils.rl_games_policy", __package__)
sys.modules[f"{__name__}.rl_games_policy"] = _rl_games_policy

