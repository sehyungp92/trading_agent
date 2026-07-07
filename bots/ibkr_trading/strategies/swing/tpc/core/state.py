from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from strategies.swing._shared.etf_core import ETFBarInput, ETFCoreState, ETFFill, ETFOrderUpdate


@dataclass(slots=True)
class TPCSecondEntrySeed:
    symbol: str
    source_setup_id: str
    direction: int
    pullback_low: float
    pullback_high: float
    stop_time: datetime
    source_grade: str
    source_score: float


@dataclass(slots=True)
class TPCCoreState(ETFCoreState):
    second_entry_seeds: dict[str, TPCSecondEntrySeed] = field(default_factory=dict)


@dataclass(slots=True)
class TPCBarInput(ETFBarInput):
    pass


@dataclass(slots=True)
class TPCFill(ETFFill):
    pass


@dataclass(slots=True)
class TPCOrderUpdate(ETFOrderUpdate):
    pass
