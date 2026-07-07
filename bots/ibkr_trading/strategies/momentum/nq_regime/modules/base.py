from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strategies.momentum.nq_regime.config import Grade, ModuleId, TradeSide


@dataclass(frozen=True, slots=True)
class NewsEvent:
    name: str
    start_ts: datetime
    end_ts: datetime
    tier: int = 1
    description: str = ""


@dataclass(frozen=True, slots=True)
class SetupCandidate:
    candidate_id: str
    module: ModuleId
    side: TradeSide
    setup_type: str
    timestamp: datetime
    level: float | None
    score: int
    grade: Grade
    entry_price: float
    stop_price: float
    targets: tuple[float, ...]
    entry_model: str
    risk_pct: float
    invalidation_price: float
    target_room_r: float
    vetoes: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.grade not in {Grade.C, Grade.INVALID} and not self.vetoes


@dataclass(frozen=True, slots=True)
class BlockedCandidate:
    candidate: SetupCandidate
    block_reason: str


@dataclass(frozen=True, slots=True)
class RoutingDecisionEvent:
    ts: datetime
    regime: Any
    regime_scores: Any
    selected_module: ModuleId
    selected_candidate_id: str | None
    blocked_candidates: tuple[BlockedCandidate, ...]
    reason_code: str
    confidence: float

