# schemas/forecast_tracking.py
"""Forecast tracking schemas — rolling meta-analysis of prediction accuracy.

Tracks weekly accuracy snapshots and computes rolling calibration adjustments.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field



class AccuracyTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


class ForecastRecord(BaseModel):
    """A single week's forecast accuracy snapshot."""

    week_start: str
    week_end: str
    predictions_reviewed: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    by_bot: dict[str, float] = {}  # bot_id → accuracy
    by_type: dict[str, float] = {}  # prediction_type → accuracy
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class CalibrationBucket(BaseModel):
    """A single confidence calibration bucket."""

    bucket_lower: float
    bucket_upper: float
    prediction_count: int
    correct_count: int
    mean_confidence: float
    observed_accuracy: float
    gap: float  # mean_confidence - observed_accuracy; positive = overconfident

    @property
    def is_reliable(self) -> bool:
        return self.prediction_count >= 5


class ForecastMetaAnalysis(BaseModel):
    """Rolling meta-analysis of forecast accuracy over time."""

    rolling_accuracy_4w: float = 0.0
    rolling_accuracy_12w: float = 0.0
    trend: AccuracyTrend = AccuracyTrend.STABLE
    accuracy_by_bot: dict[str, float] = {}
    accuracy_by_metric: dict[str, float] = {}  # pnl, win_rate, drawdown, sharpe → accuracy
    accuracy_by_strategy: dict[str, float] = {}  # strategy_id → accuracy (when sufficient data)
    calibration_adjustment: float = 0.0  # -1 to +1; negative = over-confident
    weeks_analyzed: int = 0
    calibration_buckets: list[CalibrationBucket] = Field(default_factory=list)
    expected_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None
    calibration_sample_size: int = 0
    directional_bias: dict[str, dict] = {}
