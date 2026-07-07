"""Automated suggestion outcome measurement schemas.

Regime-controlled outcome measurement with statistical quality assessment.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, computed_field

from trading_assistant.schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


class Verdict(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"


class MeasurementQuality(str, Enum):
    HIGH = "high"          # regime matched, sufficient trades, no concurrent changes
    MEDIUM = "medium"      # minor issues (low volatility ratio, few concurrent changes)
    LOW = "low"            # regime mismatch or high concurrent changes
    INSUFFICIENT = "insufficient"  # not enough trades or data


class OutcomeMeasurement(BaseModel):
    """Before/after comparison for an implemented suggestion with regime controls."""
    suggestion_id: str
    bot_id: str = ""
    category: str = ""
    implemented_date: str
    measurement_date: str
    window_days: int
    pnl_before: float = 0.0
    pnl_after: float = 0.0
    win_rate_before: float = 0.0
    win_rate_after: float = 0.0
    drawdown_before: float = 0.0
    drawdown_after: float = 0.0

    # Regime controls
    before_regime: str = ""
    after_regime: str = ""
    regime_matched: bool = True
    before_volatility: float = 0.0
    after_volatility: float = 0.0
    volatility_ratio: float = 1.0

    # Sample size
    before_trade_count: int = 0
    after_trade_count: int = 0

    # Confounders
    concurrent_changes: list[str] = []

    # Macro regime context
    macro_regime_at_implementation: str = ""  # G/R/S/D at implementation time
    macro_regime_stable: bool = True  # False if macro regime changed during measurement

    # Quality assessment
    measurement_quality: MeasurementQuality = MeasurementQuality.HIGH

    # Targeted metric evaluation
    target_metric: Optional[str] = None  # "pnl" | "win_rate" | "drawdown"
    target_metric_improved: Optional[bool] = None
    target_metric_delta: float = 0.0

    # Provider attribution
    source_provider: str = ""
    source_model: str = ""
    source_run_id: str = ""

    # Effect significance
    effect_size: float = 0.0
    noise_estimate: float = 0.0
    significance_score: float = 0.0  # effect_size / (noise * sqrt(1/n_before + 1/n_after))

    # Cross-links into proposal/hypothesis graph (optional)
    proposal_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    outcome_source: str = "early_warning"
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pnl_delta(self) -> float:
        return self.pnl_after - self.pnl_before

    @computed_field  # type: ignore[prop-decorator]
    @property
    def verdict(self) -> Verdict:
        # Insufficient data: fewer than 3 trades in either window
        if self.before_trade_count < 3 or self.after_trade_count < 3:
            return Verdict.INSUFFICIENT_DATA

        # Only assign directional verdict for high/medium quality
        if self.measurement_quality in (
            MeasurementQuality.LOW,
            MeasurementQuality.INSUFFICIENT,
        ):
            return Verdict.INCONCLUSIVE

        if self.pnl_before == 0:
            return Verdict.INCONCLUSIVE

        pnl_change = self.pnl_delta / abs(self.pnl_before)
        wr_change = self.win_rate_after - self.win_rate_before

        if pnl_change > 0.1 and wr_change >= -0.05:
            return Verdict.POSITIVE
        elif pnl_change < -0.1 or wr_change < -0.1:
            return Verdict.NEGATIVE
        return Verdict.NEUTRAL


def compute_measurement_quality(
    regime_matched: bool,
    before_trade_count: int,
    after_trade_count: int,
    volatility_ratio: float,
    concurrent_changes: list[str],
    window_days: int = 7,
    macro_regime_stable: bool = True,
) -> MeasurementQuality:
    """Determine measurement quality from regime/sample/confounder data.

    Args:
        window_days: Measurement window size. Larger windows require more
            trades for INSUFFICIENT threshold: 7d→3, 14d→5, 30d→10.
        macro_regime_stable: False if macro regime (G/R/S/D) changed during
            measurement window — lowers quality due to regime drift.
    """
    # Scale trade count minimums with window size
    if window_days >= 30:
        min_trades = 10
    elif window_days >= 14:
        min_trades = 5
    else:
        min_trades = 3

    if before_trade_count < min_trades or after_trade_count < min_trades:
        return MeasurementQuality.INSUFFICIENT

    if not regime_matched:
        return MeasurementQuality.LOW

    if len(concurrent_changes) >= 3:
        return MeasurementQuality.LOW

    issues = 0
    if volatility_ratio < 0.5 or volatility_ratio > 2.0:
        issues += 1
    if len(concurrent_changes) >= 1:
        issues += 1
    if before_trade_count < 10 or after_trade_count < 10:
        issues += 1
    if not macro_regime_stable:
        issues += 1

    if issues == 0:
        return MeasurementQuality.HIGH
    elif issues <= 1:
        return MeasurementQuality.MEDIUM
    return MeasurementQuality.LOW


def compute_significance(
    effect_size: float,
    noise: float,
    n_before: int,
    n_after: int,
) -> float:
    """Compute basic significance score: effect / (noise * sqrt(1/n_before + 1/n_after)).

    Returns 0.0 if inputs are insufficient.
    """
    if noise <= 0 or n_before <= 0 or n_after <= 0:
        return 0.0
    denominator = noise * math.sqrt(1.0 / n_before + 1.0 / n_after)
    if denominator <= 0:
        return 0.0
    return abs(effect_size) / denominator
