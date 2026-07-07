"""Shared enums for the backtesting framework.

All strategy engines import Direction from here,
keeping sim_broker strategy-agnostic.
"""
from __future__ import annotations

from enum import IntEnum


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1
