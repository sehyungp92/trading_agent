# schemas/portfolio_risk.py
"""Portfolio risk schemas — cross-bot exposure and crowding detection.

Computed daily by skills/compute_portfolio_risk.py. No LLM calls.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CrowdingAlert(BaseModel):
    """A specific crowding/concentration risk detected."""

    alert_type: str  # high_correlation | same_side | total_exposure | single_symbol_concentration
    description: str
    severity: str = "medium"  # low | medium | high | critical
    bots_involved: list[str] = []
    symbol: Optional[str] = None
    exposure_pct: Optional[float] = None


class PortfolioRiskCard(BaseModel):
    """Daily cross-bot portfolio risk snapshot."""

    date: str  # YYYY-MM-DD
    total_exposure_pct: float = 0.0
    exposure_by_symbol: dict[str, float] = {}
    exposure_by_direction: dict[str, float] = {}
    correlation_matrix: dict[str, float] = {}  # "bot1_bot2" → correlation
    max_simultaneous_leverage: float = 0.0
    concentration_score: float = 0.0  # 0–100, higher = more concentrated
    crowding_alerts: list[CrowdingAlert] = []
