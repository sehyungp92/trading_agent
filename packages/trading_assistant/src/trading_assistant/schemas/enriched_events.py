"""Enriched event schemas — additional event types for deeper bot instrumentation.

These define new event types that bots can emit for richer analysis:
indicator snapshots, order book context, filter decisions, and parameter changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field


class IndicatorSnapshot(BaseModel):
    """Snapshot of indicator values at a trading decision point."""

    bot_id: str
    pair: str
    timestamp: datetime
    indicators: dict[str, float] = {}  # name -> value, e.g. {"rsi_14": 45.2}
    signal_name: str = ""
    signal_strength: float = 0.0
    decision: Literal["enter", "skip", "exit"] = "skip"
    context: dict = {}


class OrderBookContext(BaseModel):
    """Order book state at entry/exit for microstructure analysis."""

    bot_id: str
    pair: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    spread_bps: float = 0.0
    bid_depth_10bps: float = 0.0  # volume within 10bps of best bid
    ask_depth_10bps: float = 0.0
    trade_context: Optional[Literal["entry", "exit"]] = None
    related_trade_id: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def imbalance_ratio(self) -> float:
        """Bid/ask depth imbalance. >1 = bid-heavy, <1 = ask-heavy."""
        if self.ask_depth_10bps <= 0:
            return 0.0
        return self.bid_depth_10bps / self.ask_depth_10bps


class FilterDecisionEvent(BaseModel):
    """Per-filter pass/block decision with threshold proximity."""

    bot_id: str
    pair: str
    timestamp: datetime
    filter_name: str
    passed: bool
    threshold: float
    actual_value: float
    signal_name: str = ""
    signal_strength: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin_pct(self) -> float:
        """How close to the boundary, as a percentage of threshold.

        Positive = passed with margin, negative = blocked (below threshold).
        """
        if self.threshold == 0:
            return 0.0
        return ((self.actual_value - self.threshold) / abs(self.threshold)) * 100


class ChangeSource(str, Enum):
    PR_MERGE = "pr_merge"
    MANUAL = "manual"
    HOT_RELOAD = "hot_reload"
    EXPERIMENT = "experiment"


class ParameterChangeEvent(BaseModel):
    """Records when a bot parameter changes, for tracking cause and effect."""

    bot_id: str
    param_name: str
    old_value: Any = None
    new_value: Any = None
    change_source: ChangeSource = ChangeSource.MANUAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    commit_sha: Optional[str] = None
    pr_url: Optional[str] = None
