"""Filter sensitivity analysis schemas."""
from __future__ import annotations

from pydantic import BaseModel


class SensitivityPoint(BaseModel):
    """Performance estimate at a specific threshold adjustment."""
    threshold_adjustment_pct: float
    estimated_additional_trades: int = 0
    estimated_pnl_impact: float = 0.0


class FilterSensitivityCurve(BaseModel):
    """Sensitivity analysis for a single filter."""
    filter_name: str
    bot_id: str
    current_block_count: int = 0
    current_net_impact: float = 0.0
    blocked_winners: int = 0
    blocked_losers: int = 0
    recommendation: str | None = None
    sensitivity_points: list[SensitivityPoint] = []


class FilterSensitivityReport(BaseModel):
    """Sensitivity analysis across all filters for a bot."""
    bot_id: str
    date: str
    curves: list[FilterSensitivityCurve] = []
