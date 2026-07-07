# schemas/transfer_proposals.py
"""Transfer proposal schemas — cross-bot pattern application candidates."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransferProposal(BaseModel):
    """A proposal to transfer a validated pattern from one bot to another."""

    pattern_id: str
    source_bot: str
    source_strategy_id: Optional[str] = None  # specific source strategy; None = bot-wide pattern
    target_bot: str
    target_strategy_id: Optional[str] = None  # specific target strategy; None = bot-wide application
    pattern_title: str
    category: str = ""
    compatibility_score: float = 0.0  # 0–1
    rationale: str = ""


class TransferOutcome(BaseModel):
    """Measured result of a pattern transfer to a target bot."""

    pattern_id: str
    source_bot: str
    source_strategy_id: Optional[str] = None
    target_bot: str
    target_strategy_id: Optional[str] = None
    transferred_at: str = ""
    measured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    pnl_delta_7d: float = 0.0
    win_rate_delta_7d: float = 0.0
    verdict: Literal["positive", "neutral", "negative", "inconclusive"] = "neutral"
    regime_matched: bool = True
    measurement_quality: str = "medium"
    before_trade_count: int = 0
    after_trade_count: int = 0
