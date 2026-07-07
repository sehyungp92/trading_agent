"""Time-of-day detector behavior."""

from __future__ import annotations

from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.strategy_suggestions import (
    StrategySuggestion,
    SuggestionTier,
)


def evaluate_time_of_day_patterns(
    *,
    bot_id: str,
    hourly_buckets: list,
    strategy_id: str,
    strategy_archetype: str,
    loss_threshold: float,
    min_trades: int = 10,
) -> list[StrategySuggestion]:
    """Detect hours with consistently poor performance."""
    suggestions: list[StrategySuggestion] = []
    archetype_note = _archetype_note(strategy_archetype)
    for bucket in hourly_buckets:
        if bucket.trade_count < min_trades:
            continue
        if bucket.pnl >= 0 or bucket.win_rate >= loss_threshold:
            continue
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.FILTER,
            bot_id=bot_id,
            strategy_id=strategy_id,
            strategy_archetype=strategy_archetype,
            archetype_note=archetype_note,
            title=f"Poor hour {bucket.hour:02d}:00 — {bot_id}",
            description=(
                f"Hour {bucket.hour:02d}:00 UTC: {bucket.trade_count} trades, "
                f"PnL ${bucket.pnl:.0f}, win rate {bucket.win_rate:.0%}. "
                f"Consider adding a time-of-day gate to avoid this hour."
            ),
            evidence_days=7,
            confidence=0.6,
            detection_context=DetectionContext(
                detector_name="time_of_day",
                bot_id=bot_id,
                threshold_name="loss_threshold",
                threshold_value=loss_threshold,
                observed_value=bucket.win_rate,
            ),
        ))
    return suggestions


def _archetype_note(strategy_archetype: str) -> str:
    if strategy_archetype in {
        "intraday_momentum",
        "opening_range_breakout",
        "vwap_pullback",
        "flow_following",
        "multi_engine_bear",
    }:
        return "HIGH RELEVANCE — intraday strategy, time-of-day is a primary performance lever"
    if strategy_archetype in {
        "trend_follow",
        "divergence_swing",
        "pullback",
        "bear_regime_swing",
    }:
        return "LOW RELEVANCE — multi-day/swing strategy, time-of-day impact is secondary"
    return ""
