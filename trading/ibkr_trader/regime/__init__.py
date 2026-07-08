"""MR-AWQ Meta Allocator v2 package."""

from __future__ import annotations

from .config import MetaConfig
from .context import RegimeContext

__all__ = ["MetaConfig", "RegimeContext", "run_signal_engine"]


def __getattr__(name: str):
    if name == "run_signal_engine":
        from .engine import run_signal_engine

        return run_signal_engine
    raise AttributeError(name)
