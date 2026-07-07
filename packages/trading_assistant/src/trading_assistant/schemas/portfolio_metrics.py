# schemas/portfolio_metrics.py
"""Portfolio-level metric schemas — family snapshots, rolling metrics, drawdown correlation."""
from __future__ import annotations

from pydantic import BaseModel


class FamilyDailySnapshot(BaseModel):
    """Aggregated daily snapshot for a strategy family."""

    family: str
    date: str
    strategy_ids: list[str] = []
    total_net_pnl: float = 0.0
    total_fees: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    max_drawdown_pct: float = 0.0
    avg_exposure_pct: float = 0.0
    active_strategies: int = 0


class PortfolioRollingMetrics(BaseModel):
    """Rolling portfolio-level risk-adjusted metrics."""

    date: str
    sharpe_7d: float = 0.0
    sharpe_30d: float = 0.0
    sharpe_90d: float = 0.0
    sortino_7d: float = 0.0
    sortino_30d: float = 0.0
    sortino_90d: float = 0.0
    calmar_30d: float = 0.0
    calmar_90d: float = 0.0
    total_pnl_7d: float = 0.0
    total_pnl_30d: float = 0.0
    max_drawdown_30d: float = 0.0
    max_drawdown_90d: float = 0.0
    family_metrics: dict[str, dict] = {}  # family → {sharpe_30d, pnl_30d, ...}


class DrawdownCorrelation(BaseModel):
    """Cross-family drawdown correlation analysis."""

    week_start: str
    simultaneous_drawdown_days: int = 0
    worst_portfolio_drawdown_pct: float = 0.0
    systemic_risk_score: float = 0.0  # 0-100
    family_drawdown_overlap: dict[str, dict] = {}  # "fam1_fam2" → {overlap_days, correlation}
    recovery_divergence: dict = {}  # {fastest_family, slowest_family, divergence_days}
