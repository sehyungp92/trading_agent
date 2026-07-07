# schemas/convergence.py
"""Convergence tracking schemas for learning loop health monitoring."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class DimensionStatus(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    OSCILLATING = "oscillating"
    INSUFFICIENT_DATA = "insufficient_data"


class ConvergenceDimension(BaseModel):
    """Status of a single convergence dimension."""

    name: str  # e.g. "composite_scores", "prediction_accuracy"
    status: DimensionStatus
    trend_value: float  # slope or delta
    window_weeks: int
    detail: str  # human-readable explanation


class LoopHealthMetrics(BaseModel):
    """Quantitative health metrics for the learning loop."""

    proposal_to_measurement_days: float | None = None  # avg days from proposal → outcome measurement
    oscillation_severity: float = 0.0  # 0.0 = none, 1.0 = maximum; quantifies oscillation depth
    transfer_success_rate: float | None = None  # cross-bot transfer positive outcome ratio
    recalibration_effectiveness: float | None = None  # whether recalibrations improve subsequent outcomes
    suggestions_per_cycle: float = 0.0  # avg suggestions proposed per learning cycle
    measurement_coverage: float = 0.0  # fraction of implemented suggestions that got measured


class ConvergenceReport(BaseModel):
    """Multi-dimensional convergence report for the learning system."""

    overall_status: DimensionStatus
    dimensions: list[ConvergenceDimension]
    oscillation_detected: bool
    weeks_analyzed: int
    recommendation: str  # e.g. "System converging — maintain current approach"
    health_metrics: LoopHealthMetrics = LoopHealthMetrics()
