# schemas/drawdown_analysis.py
"""Drawdown analysis schemas — episode segmentation and attribution.

Segments equity curve into drawdown episodes and attributes each to root causes.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DrawdownEpisode(BaseModel):
    """A single drawdown period from peak to trough."""

    bot_id: str
    start_date: str
    end_date: str
    peak_pnl: float = 0.0
    trough_pnl: float = 0.0
    drawdown_pct: float = 0.0
    trade_count: int = 0
    duration_days: int = 0
    recovered: bool = False
    recovery_date: Optional[str] = None
    contributing_trades: list[str] = []  # trade_ids
    dominant_regime: str = ""
    root_cause_distribution: dict[str, int] = {}


class DrawdownAttribution(BaseModel):
    """Drawdown attribution report for one bot."""

    bot_id: str
    date: str
    episodes: list[DrawdownEpisode] = []
    top_contributing_root_causes: dict[str, int] = {}
    largest_single_loss_pct: float = 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.episodes:
            return 0.0
        return max(ep.drawdown_pct for ep in self.episodes)
