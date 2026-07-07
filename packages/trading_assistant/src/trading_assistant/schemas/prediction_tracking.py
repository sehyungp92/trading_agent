# schemas/prediction_tracking.py
"""Prediction tracking schemas — record and evaluate structured predictions."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictionRecord(BaseModel):
    """A single recorded prediction from Claude's structured output."""

    bot_id: str
    strategy_id: Optional[str] = None  # specific strategy; None = bot-wide
    metric: str
    direction: str
    confidence: float
    timeframe_days: int = 7
    reasoning: str = ""
    week: str = ""
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    predicted_magnitude: Optional[float] = None
    actual_magnitude: Optional[float] = None
    regime_tag: Optional[str] = None
    prediction_family: Optional[str] = None  # performance/reliability/transfer/structural


class PredictionVerdict(BaseModel):
    """Verdict on a single prediction after evaluation."""

    bot_id: str
    strategy_id: Optional[str] = None
    metric: str
    predicted_direction: str
    actual_direction: str = ""
    correct: bool = False
    confidence: float = 0.0
    status: Literal["correct", "incorrect", "insufficient_data"] = "insufficient_data"
    magnitude_score: float = 0.0


class PredictionEvaluation(BaseModel):
    """Evaluation of a batch of predictions for a given week."""

    week: str
    verdicts: list[PredictionVerdict] = []
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    confidence_weighted_accuracy: float = 0.0
    accuracy_by_metric: dict[str, float] = {}
    accuracy_by_strategy: dict[str, float] = {}  # strategy_id → accuracy
    magnitude_weighted_accuracy: float = 0.0
