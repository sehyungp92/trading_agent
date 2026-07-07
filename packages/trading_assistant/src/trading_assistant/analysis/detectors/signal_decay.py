"""Signal-decay detector behavior."""

from __future__ import annotations

from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.strategy_suggestions import (
    StrategySuggestion,
    SuggestionTier,
)


def evaluate_alpha_decay(
    *,
    bot_id: str,
    rolling_sharpe_30d: float,
    rolling_sharpe_90d: float,
    decay_threshold: float,
    strategy_id: str = "",
    strategy_archetype: str = "",
) -> list[StrategySuggestion]:
    """Detect materially declining Sharpe over 30/90 day windows."""
    if rolling_sharpe_90d <= 0:
        return []
    decay_ratio = (rolling_sharpe_90d - rolling_sharpe_30d) / rolling_sharpe_90d
    if decay_ratio < decay_threshold:
        return []
    return [StrategySuggestion(
        tier=SuggestionTier.HYPOTHESIS,
        bot_id=bot_id,
        strategy_id=strategy_id,
        strategy_archetype=strategy_archetype,
        title=f"Alpha decay detected ??{bot_id}",
        description=(
            f"30d Sharpe ({rolling_sharpe_30d:.2f}) is {decay_ratio:.0%} below "
            f"90d Sharpe ({rolling_sharpe_90d:.2f}). The strategy may be losing edge. "
            "Review signal quality and market regime alignment."
        ),
        evidence_days=90,
        confidence=min(0.9, 0.5 + decay_ratio),
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="alpha_decay",
            bot_id=bot_id,
            threshold_name="decay_threshold",
            threshold_value=decay_threshold,
            observed_value=decay_ratio,
        ),
    )]


def evaluate_signal_decay(
    *,
    bot_id: str,
    signal_outcome_correlation_30d: float,
    signal_outcome_correlation_90d: float,
    decay_threshold: float,
) -> list[StrategySuggestion]:
    """Detect declining signal-to-outcome correlation."""
    drop = signal_outcome_correlation_90d - signal_outcome_correlation_30d
    if drop < decay_threshold:
        return []
    return [StrategySuggestion(
        tier=SuggestionTier.HYPOTHESIS,
        bot_id=bot_id,
        title=f"Signal quality decay ??{bot_id}",
        description=(
            f"Signal->outcome correlation dropped from {signal_outcome_correlation_90d:.2f} "
            f"(90d) to {signal_outcome_correlation_30d:.2f} (30d). "
            "Signal may need recalibration or replacement."
        ),
        evidence_days=90,
        confidence=min(0.9, 0.5 + drop),
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="signal_decay",
            bot_id=bot_id,
            threshold_name="decay_threshold",
            threshold_value=decay_threshold,
            observed_value=drop,
        ),
    )]


def evaluate_component_signal_decay(
    *,
    bot_id: str,
    signal_health_data: dict,
    stability_threshold: float,
    correlation_threshold: float,
    min_trades: int = 5,
) -> list[StrategySuggestion]:
    """Detect degraded signal components from signal-health data."""
    components = signal_health_data.get("components", [])
    degraded: list[str] = []
    for component in components:
        trade_count = component.get("trade_count", 0)
        if trade_count < min_trades:
            continue
        stability = component.get("stability", 1.0)
        win_corr = abs(component.get("win_correlation", 1.0))
        if stability < stability_threshold or win_corr < correlation_threshold:
            degraded.append(component.get("component_name", "unknown"))

    if not degraded:
        return []

    min_stability = min(
        (
            component.get("stability", 1.0)
            for component in components
            if component.get("component_name", "unknown") in degraded
        ),
        default=0.0,
    )
    return [StrategySuggestion(
        tier=SuggestionTier.HYPOTHESIS,
        bot_id=bot_id,
        title=f"Signal component decay -{bot_id}",
        description=(
            f"Degraded signal components detected: {', '.join(degraded)}. "
            f"These components show low stability (<{stability_threshold}) or "
            f"near-zero win correlation (<{correlation_threshold}). "
            "Review whether these signals still carry predictive value."
        ),
        evidence_days=7,
        confidence=0.5,
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="component_signal_decay",
            bot_id=bot_id,
            threshold_name="stability_threshold",
            threshold_value=stability_threshold,
            observed_value=min_stability,
        ),
    )]


def evaluate_factor_correlation_decay(
    *,
    bot_id: str,
    factor_rolling_data: list[dict],
) -> list[StrategySuggestion]:
    """Detect degrading signal factors from rolling 30-day analysis."""
    suggestions: list[StrategySuggestion] = []
    for factor in factor_rolling_data:
        trend = factor.get("win_rate_trend", "stable")
        below = factor.get("below_threshold", False)
        if trend != "degrading" and not below:
            continue

        name = factor.get("factor_name", "unknown")
        win_rate = factor.get("rolling_30d_win_rate", 0)
        days = factor.get("days_of_data", 0)
        effective_threshold = 1.0 if below else 0.0
        parts = [f"Factor '{name}' on {bot_id}"]
        if trend == "degrading":
            parts.append("shows degrading win rate trend over 30d window")
        if below:
            parts.append(f"(rolling win rate {win_rate:.0%} is below threshold)")
        parts.append(f"Based on {days} days of data.")
        parts.append("Consider recalibrating or replacing this signal factor.")
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            title=f"Factor decay -{name} on {bot_id}",
            description=" ".join(parts),
            evidence_days=days,
            confidence=0.5,
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="factor_decay",
                bot_id=bot_id,
                threshold_name="below_threshold",
                threshold_value=effective_threshold,
                observed_value=win_rate,
            ),
        ))
    return suggestions
