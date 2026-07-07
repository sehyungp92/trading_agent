# schemas/proportion_optimization.py
"""Proportion optimization schemas — intra-bot strategy allocation recommendations."""
from __future__ import annotations

from pydantic import BaseModel


class StrategyAllocationRecommendation(BaseModel):
    """Allocation recommendation for a single strategy within a bot."""

    bot_id: str
    strategy_id: str
    current_unit_risk_pct: float = 0.0
    suggested_unit_risk_pct: float = 0.0
    change_pct: float = 0.0
    capital_efficiency: float = 0.0
    marginal_sharpe: float = 0.0
    rationale: str = ""
    evidence_period_days: int = 7


class IntraBotAllocationReport(BaseModel):
    """Allocation report for a single bot's strategies."""

    bot_id: str
    week_start: str
    week_end: str
    recommendations: list[StrategyAllocationRecommendation] = []
    current_bot_sharpe: float = 0.0
    suggested_bot_sharpe: float = 0.0
    sharpe_change: float = 0.0
    rebalance_needed: bool = False
    special_notes: list[str] = []


class ProportionOptimizationReport(BaseModel):
    """Full proportion optimization report across all bots."""

    week_start: str
    week_end: str
    bot_reports: list[IntraBotAllocationReport] = []
