# schemas/experiments.py
"""A/B testing schemas — experiment configuration, variant tracking, and results.

Supports orchestrator-managed experiment lifecycle: DRAFT → ACTIVE → CONCLUDED.
Bots handle variant assignment and data collection; orchestrator manages analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    CONCLUDED = "concluded"
    CANCELLED = "cancelled"


class ExperimentType(str, Enum):
    PARAMETER_AB = "parameter_ab"
    FILTER_AB = "filter_ab"
    ABLATION = "ablation"


class ExperimentVariant(BaseModel):
    """A single experimental variant (control or treatment)."""

    name: str  # e.g. "control", "treatment"
    params: dict[str, Any] = {}  # parameter overrides
    allocation_pct: float  # percentage of trades assigned


class ExperimentConfig(BaseModel):
    """Full configuration for an A/B experiment."""

    experiment_id: str
    bot_id: str
    strategy_id: Optional[str] = None
    experiment_type: ExperimentType = ExperimentType.PARAMETER_AB
    title: str
    description: str = ""
    variants: list[ExperimentVariant]  # at least 2
    success_metric: Literal["pnl", "sharpe", "win_rate", "profit_factor"] = "pnl"
    min_trades_per_variant: int = 30
    max_duration_days: int = 30
    significance_level: float = 0.05
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: Optional[datetime] = None
    concluded_at: Optional[datetime] = None
    source_suggestion_id: Optional[str] = None
    hypothesis_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_variants(self) -> ExperimentConfig:
        if len(self.variants) < 2:
            raise ValueError("At least 2 variants required (control + treatment)")
        total_alloc = sum(v.allocation_pct for v in self.variants)
        if abs(total_alloc - 100.0) > 0.1:
            raise ValueError(
                f"Variant allocation must sum to 100%, got {total_alloc}%"
            )
        return self

    @model_validator(mode="after")
    def _validate_significance(self) -> ExperimentConfig:
        if not (0.01 <= self.significance_level <= 0.10):
            raise ValueError(
                f"significance_level must be 0.01-0.10, got {self.significance_level}"
            )
        return self


class VariantMetrics(BaseModel):
    """Accumulated metrics for a single experiment variant."""

    variant_name: str
    trade_count: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0


class ExperimentResult(BaseModel):
    """Statistical analysis result for a concluded experiment."""

    experiment_id: str
    variant_metrics: list[VariantMetrics]
    p_value: Optional[float] = None
    effect_size: Optional[float] = None  # Cohen's d
    confidence_interval_95: Optional[tuple[float, float]] = None
    winner: Optional[str] = None  # variant name, or None if inconclusive
    recommendation: Literal[
        "adopt_treatment", "keep_control", "inconclusive", "extend"
    ] = "inconclusive"
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
