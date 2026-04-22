"""Shim package exposing archived PhysREPA scripted policies."""

from __future__ import annotations

from importlib import import_module
import sys

_scripted = import_module("..archive_data_collection.policies.scripted_policy", __package__)
sys.modules[f"{__name__}.scripted_policy"] = _scripted

from ..archive_data_collection.policies.scripted_policy import *  # noqa: F401,F403,E402

