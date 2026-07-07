# schemas/hourly_performance.py
"""Hourly performance schemas — time-of-day analysis buckets.

Captures PnL, win rate, and process quality by hour of day.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HourlyBucket(BaseModel):
    """Performance stats for a single hour of the day (0-23 in bot's local tz)."""

    hour: int  # 0-23
    trade_count: int = 0
    pnl: float = 0.0
    win_rate: float = 0.0
    avg_process_quality: float = 0.0


class HourlyPerformance(BaseModel):
    """Time-of-day performance breakdown for one bot on one date."""

    bot_id: str
    date: str
    timezone: str = "UTC"
    buckets: list[HourlyBucket] = []

    @property
    def best_hour(self) -> Optional[int]:
        if not self.buckets:
            return None
        return max(self.buckets, key=lambda b: b.pnl).hour

    @property
    def worst_hour(self) -> Optional[int]:
        if not self.buckets:
            return None
        return min(self.buckets, key=lambda b: b.pnl).hour
