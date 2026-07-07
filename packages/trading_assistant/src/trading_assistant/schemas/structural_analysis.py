"""Structural analysis schemas — strategy lifecycle, architecture mismatch, filter ROI.

Used by skills/structural_analyzer.py to report on structural health of strategies.
"""
from __future__ import annotations

from pydantic import BaseModel


class StrategyLifecycleStatus(BaseModel):
    """Lifecycle phase classification for a single strategy."""

    bot_id: str
    strategy_id: str
    phase: str  # "growing" | "mature" | "decaying"
    sharpe_30d: float = 0.0
    sharpe_60d: float = 0.0
    sharpe_90d: float = 0.0
    sharpe_trend: float = 0.0  # slope of rolling Sharpe (positive = growing)
    edge_half_life_days: float | None = None  # estimated days until Sharpe halves
    trade_count_90d: int = 0


class ArchitectureMismatch(BaseModel):
    """Detected mismatch between signal type and exit mechanism."""

    bot_id: str
    strategy_id: str
    mismatch_type: str  # "momentum_fixed_tp" | "mean_reversion_trailing_stop" | etc.
    current_setup: str  # e.g. "momentum signal + fixed take profit"
    recommended_setup: str  # e.g. "momentum signal + trailing stop"
    evidence: str
    estimated_impact_pnl: float = 0.0
    confidence: float = 0.0


class FilterROI(BaseModel):
    """Return on investment for a single filter on a strategy."""

    bot_id: str
    strategy_id: str
    filter_name: str
    blocks_saved_count: int = 0  # blocks that prevented losses
    blocks_cost_count: int = 0  # blocks that prevented wins
    net_pnl_impact: float = 0.0
    roi: float = 0.0  # net_pnl_impact / missed_opportunity_value


class StructuralReport(BaseModel):
    """Complete structural analysis output for a week."""

    week_start: str
    week_end: str
    lifecycle_statuses: list[StrategyLifecycleStatus] = []
    architecture_mismatches: list[ArchitectureMismatch] = []
    filter_roi: list[FilterROI] = []
    growing_strategies: list[str] = []  # "bot:strategy" keys
    decaying_strategies: list[str] = []
    proposed_changes: list[dict] = []  # {strategy, category, proposal, effort, impact, reversibility}
