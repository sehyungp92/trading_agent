# schemas/weekly_metrics.py
"""Weekly metrics schemas — Pydantic models for aggregated weekly data.

These define the output format of the weekly aggregation pipeline (skills/build_weekly_metrics.py).
Built from 7 days of daily curated data (schemas/daily_metrics.py).
"""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class StrategyWeeklySummary(BaseModel):
    """Per-strategy weekly aggregation across daily breakdowns."""

    strategy_id: str
    bot_id: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    daily_pnl: dict[str, float] = {}  # date → net_pnl for time series


class BotWeeklySummary(BaseModel):
    """Aggregated weekly stats for a single bot."""

    week_start: str  # YYYY-MM-DD (Monday)
    week_end: str  # YYYY-MM-DD (Sunday)
    bot_id: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_process_quality: float = 100.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    error_count: int = 0
    avg_uptime_pct: float = 100.0
    daily_pnl: dict[str, float] = {}  # date → PnL for sparkline
    per_strategy_summary: dict[str, StrategyWeeklySummary] = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @computed_field  # type: ignore[prop-decorator]
    @property
    def profit_factor(self) -> float:
        total_wins = self.avg_win * self.win_count
        total_losses = abs(self.avg_loss) * self.loss_count
        if total_losses == 0:
            return float("inf") if total_wins > 0 else 0.0
        return total_wins / total_losses


class WeeklySummary(BaseModel):
    """Portfolio-level weekly summary across all bots."""

    week_start: str
    week_end: str
    bot_summaries: dict[str, BotWeeklySummary] = {}
    total_net_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    total_trades: int = 0
    portfolio_max_drawdown_pct: float = 0.0


class WeekOverWeekComparison(BaseModel):
    """Deltas between current and previous week."""

    current_week: str  # week_start date
    previous_week: str
    pnl_delta: float = 0.0
    pnl_delta_pct: float = 0.0
    win_rate_delta: float = 0.0
    trade_count_delta: int = 0
    avg_process_quality_delta: float = 0.0


class ProcessQualityTrend(BaseModel):
    """Process quality over the last N weeks for one bot."""

    bot_id: str
    weekly_avg_scores: list[float] = []  # last 4 weeks, oldest first
    current_avg: float = 0.0
    trend_direction: str = "stable"  # improving | degrading | stable
    most_frequent_root_causes: dict[str, int] = {}


class RegimePerformanceTrend(BaseModel):
    """Performance trend by regime for one bot over multiple weeks."""

    bot_id: str
    regime: str
    weekly_pnl: list[float] = []
    weekly_win_rate: list[float] = []
    weekly_trade_count: list[int] = []


class FilterWeeklySummary(BaseModel):
    """Aggregated filter impact across the week for one bot."""

    bot_id: str
    filter_name: str
    total_blocks: int = 0
    blocks_that_would_have_won: int = 0
    blocks_that_would_have_lost: int = 0
    net_impact_pnl: float = 0.0  # negative = filter cost more than it saved
    confidence: float = 0.0  # avg confidence of missed opp simulations


class CorrelationSummary(BaseModel):
    """Pairwise bot correlation metrics."""

    bot_a: str
    bot_b: str
    rolling_30d_correlation: float = 0.0
    weekly_pnl_correlation: float = 0.0
    same_direction_pct: float = 0.0  # % of time both bots on same side
