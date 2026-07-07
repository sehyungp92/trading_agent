"""Signal factor rolling history schemas — persistent factor tracking over time."""
from __future__ import annotations

from pydantic import BaseModel


class FactorDayStats(BaseModel):
    """Stats for a single factor on a single day."""

    factor_name: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    avg_contribution: float = 0.0


class DailyFactorSnapshot(BaseModel):
    """One day's factor stats for a bot — the JSONL persistence format."""

    date: str
    bot_id: str
    factors: list[FactorDayStats] = []


class FactorRollingResult(BaseModel):
    """Rolling analysis result for a single factor."""

    factor_name: str
    bot_id: str
    current_win_rate: float = 0.0
    rolling_30d_win_rate: float = 0.0
    win_rate_trend: str = "stable"  # improving / stable / degrading
    current_avg_pnl: float = 0.0
    rolling_30d_avg_pnl: float = 0.0
    days_of_data: int = 0
    below_threshold: bool = False


class SignalFactorRollingReport(BaseModel):
    """Rolling analysis for all factors of one bot."""

    bot_id: str
    date: str
    window_days: int = 30
    factors: list[FactorRollingResult] = []
    alerts: list[str] = []
