"""Regime-conditional analysis schemas — per-regime strategy metrics and allocations.

Used by analysis/strategy_engine.py to produce regime-aware allocation suggestions.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RegimeStrategyMetrics(BaseModel):
    """Performance metrics for a strategy in a specific regime."""

    bot_id: str
    strategy_id: str
    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0  # avg PnL per trade
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_exit_efficiency: float = 0.0
    macro_regime: str = ""  # G/R/S/D if this is a macro regime metric


class RegimeAllocation(BaseModel):
    """Suggested allocation for a specific regime."""

    regime: str
    allocations: dict[str, float] = {}  # "bot:strategy" → suggested_pct
    rationale: str = ""


class RegimeDistribution(BaseModel):
    """How much time/trades are spent in each regime."""

    regime: str
    pct_of_time: float = 0.0  # what % of trading time in this regime
    trend: str = ""  # "increasing" | "stable" | "decreasing"
    trade_count: int = 0


class RegimeConditionalReport(BaseModel):
    """Complete regime-conditional analysis for a week."""

    week_start: str
    week_end: str
    metrics: list[RegimeStrategyMetrics] = []
    optimal_allocations: list[RegimeAllocation] = []
    regime_distribution: list[RegimeDistribution] = []
    suggestions: list[dict] = []  # {regime, strategy, current_alloc, suggested_alloc, reason}


class MacroRegimeConditionalReport(BaseModel):
    """Portfolio-level macro regime (G/R/S/D) performance breakdown."""

    week_start: str
    week_end: str
    current_macro_regime: str = ""
    regime_confidence: float = 0.0
    stress_level: float = 0.0
    metrics_by_regime: list[RegimeStrategyMetrics] = []  # macro_regime field populated
    config_effectiveness: list[dict] = []  # {regime, config_key, config_value, pnl, win_rate}
    transition_costs: list[dict] = []  # {from, to, date, pnl_5d_window}
    suggestions: list[dict] = []


# ---------------------------------------------------------------------------
# Regime-conditional parameter analysis (Phase 4)
# ---------------------------------------------------------------------------

class RegimeParameterStats(BaseModel):
    """Performance stats for a parameter value within a single regime."""

    regime: str
    trade_count: int = 0
    optimal_value: Any = None
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0


class RegimeParameterAnalysis(BaseModel):
    """Analysis of how a parameter's optimal value varies across regimes."""

    param_name: str
    bot_id: str = ""
    strategy_id: str = ""
    regimes_analyzed: list[str] = Field(default_factory=list)
    optimal_per_regime: dict[str, Any] = Field(default_factory=dict)
    current_value: Any = None
    regime_sensitivity: float = 0.0  # 0-1, normalized variance of optimal values
    regime_stats: list[RegimeParameterStats] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
