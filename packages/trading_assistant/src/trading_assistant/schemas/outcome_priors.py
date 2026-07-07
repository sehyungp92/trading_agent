"""Operational priors derived from authoritative monthly outcomes."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class GateStrictness(str, Enum):
    RELAXED_EXPLORATION = "relaxed_exploration"
    NORMAL = "normal"
    STRICTER = "stricter"


class RollbackPriority(str, Enum):
    NONE = "none"
    WATCH = "watch"
    HIGH = "high"
    CRITICAL = "critical"


class OutcomePrior(BaseModel):
    """Aggregated prior for a mutation family/category in one strategy context."""

    prior_id: str
    bot_id: str
    strategy_id: str = ""
    mutation_family: str = ""
    category: str = ""
    positive_count: int = 0
    confirmed_positive_count: int = 0
    negative_count: int = 0
    inconclusive_count: int = 0
    latest_verdict: str = ""
    latest_outcome_id: str = ""
    allocation_multiplier: float = 1.0
    required_confirmation_count: int = 1
    gate_strictness: GateStrictness = GateStrictness.NORMAL
    rollback_priority: RollbackPriority = RollbackPriority.NONE
    evidence_paths: list[str] = Field(default_factory=list)
    source_outcome_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def key_tuple(self) -> tuple[str, str, str, str]:
        return (self.bot_id, self.strategy_id, self.mutation_family, self.category)


def make_outcome_prior_id(
    *,
    bot_id: str,
    strategy_id: str = "",
    mutation_family: str = "",
    category: str = "",
) -> str:
    raw = "|".join([bot_id, strategy_id, mutation_family, category])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
