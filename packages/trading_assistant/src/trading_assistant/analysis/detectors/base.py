"""Shared detector metadata types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectorContext:
    """Minimal context a detector needs outside the full strategy engine."""

    bot_id: str = ""
    strategy_id: str = ""
    week_start: str = ""
    week_end: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorMetadata:
    name: str
    category: str
    threshold_defaults: dict[str, float] = field(default_factory=dict)
    archetype_defaults: dict[str, dict[str, float]] = field(default_factory=dict)
