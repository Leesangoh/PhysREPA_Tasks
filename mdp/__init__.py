"""Shim package exposing archived PhysREPA MDP modules."""

from __future__ import annotations

from importlib import import_module
import sys

for _name in ("events", "observations", "sync_marker"):
    _module = import_module(f"..archive_data_collection.mdp.{_name}", __package__)
    sys.modules[f"{__name__}.{_name}"] = _module

from ..archive_data_collection.mdp import *  # noqa: F401,F403,E402

