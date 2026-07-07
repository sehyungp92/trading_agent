"""Signal health schemas — per-component signal quality metrics.

Used by SignalHealthAnalyzer to assess momentum_nq_01 signal_evolution data.
"""
from __future__ import annotations

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    """Health metrics for a single signal component."""

    component_name: str
    trade_count: int = 0
    avg_entry_value: float = 0.0
    avg_exit_value: float = 0.0
    avg_range: float = 0.0
    stability: float = 0.0  # 1 - normalized_std; stable signals ≈ 1.0
    win_correlation: float = 0.0  # Pearson vs PnL
    trend_during_trade: float = 0.0  # avg (exit_value - entry_value)


class SignalHealthReport(BaseModel):
    """Aggregated signal health report for one bot on one date."""

    bot_id: str
    date: str
    components: list[ComponentHealth] = []
    total_trades_with_data: int = 0
    coverage_pct: float = 0.0
