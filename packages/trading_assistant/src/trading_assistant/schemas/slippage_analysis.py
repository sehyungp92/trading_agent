# schemas/slippage_analysis.py
"""Slippage analysis schemas — per-symbol, per-hour slippage distributions.

Used by SlippageAnalyzer to feed empirical data into cost models.
"""
from __future__ import annotations

from pydantic import BaseModel


class SlippageBucket(BaseModel):
    """Slippage statistics for one grouping key (symbol or hour)."""

    key: str
    sample_count: int = 0
    mean_bps: float = 0.0
    median_bps: float = 0.0
    p75_bps: float = 0.0
    p95_bps: float = 0.0


class SlippageDistribution(BaseModel):
    """Slippage breakdown for one bot on one date."""

    bot_id: str
    date: str
    by_symbol: dict[str, SlippageBucket] = {}
    by_hour: dict[str, SlippageBucket] = {}


class SlippageTrend(BaseModel):
    """Slippage trend over multiple weeks for one symbol."""

    bot_id: str
    symbol: str
    weekly_mean_bps: list[float] = []
    trend_direction: str = "stable"  # increasing | decreasing | stable
