# schemas/portfolio_allocation.py
"""Portfolio allocation schemas — cross-bot capital allocation recommendations."""
from __future__ import annotations

from pydantic import BaseModel


class AllocationConstraints(BaseModel):
    """Constraints for portfolio allocation optimization."""

    min_allocation_pct: float = 5.0  # no bot below 5%
    max_allocation_pct: float = 60.0  # no bot above 60%
    max_single_rebalance_pct: float = 15.0  # max change per rebalance
    min_observation_days: int = 30  # minimum data before suggesting


class BotAllocationRecommendation(BaseModel):
    """Allocation recommendation for a single bot."""

    bot_id: str
    current_allocation_pct: float = 0.0
    suggested_allocation_pct: float = 0.0
    change_pct: float = 0.0  # suggested - current
    capital_efficiency: float = 0.0  # net_pnl / allocation_pct
    calmar_contribution: float = 0.0  # bot's contribution to portfolio Calmar
    rationale: str = ""


class PortfolioAllocationReport(BaseModel):
    """Full portfolio allocation report."""

    week_start: str
    week_end: str
    recommendations: list[BotAllocationRecommendation] = []
    current_portfolio_calmar: float = 0.0
    suggested_portfolio_calmar: float = 0.0
    calmar_change_pct: float = 0.0
    rebalance_needed: bool = False
    method: str = "risk_parity_calmar_tilt"
