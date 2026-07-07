# schemas/parameter_search.py
"""Parameter search schemas — iterative neighborhood exploration and routing."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from trading_assistant.schemas.regime_conditional import RegimeParameterAnalysis
from trading_assistant.schemas.simulation_metrics import SimulationMetrics


class SearchRouting(str, Enum):
    APPROVE = "approve"       # strong backtest + robust → approval card
    EXPERIMENT = "experiment"  # marginal backtest → A/B experiment first
    DISCARD = "discard"       # no value passes safety → kill suggestion


class CandidateResult(BaseModel):
    """Result of evaluating a single candidate parameter value."""

    value: Any
    metrics: SimulationMetrics = Field(default_factory=SimulationMetrics)
    robustness_score: float = 0.0
    neighborhood_stable: bool = False
    passes_safety: bool = False
    safety_notes: list[str] = Field(default_factory=list)
    composite_score: float = 0.0
    cost_sensitivity_sharpe: float = 0.0


class ParameterSearchReport(BaseModel):
    """Report from a parameter neighborhood search."""

    suggestion_id: str
    bot_id: str
    param_name: str
    original_proposed: Any
    current_value: Any
    baseline_composite: float = 0.0
    candidates_tested: int = 0
    candidates_passing: int = 0
    best_value: Any = None
    best_composite: float = 0.0
    best_robustness_score: float = 0.0
    best_cost_sensitivity_sharpe: float = 0.0
    routing: SearchRouting = SearchRouting.DISCARD
    discard_reason: str = ""
    exploration_summary: str = ""
    regime_analysis: RegimeParameterAnalysis | None = None
    searched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class BacktestCalibrationRecord(BaseModel):
    """Tracks backtest prediction accuracy for meta-learning."""

    suggestion_id: str
    bot_id: str
    param_category: str
    predicted_improvement: float
    predicted_routing: SearchRouting
    actual_composite_delta: Optional[float] = None
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    measured_at: Optional[datetime] = None
    prediction_correct: Optional[bool] = None
