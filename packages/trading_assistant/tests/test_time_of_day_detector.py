from __future__ import annotations

from dataclasses import dataclass

from trading_assistant.analysis.detectors.signal_decay import (
    evaluate_component_signal_decay,
    evaluate_factor_correlation_decay,
)
from trading_assistant.analysis.detectors.catalog import DEFAULT_DETECTOR_CATALOG
from trading_assistant.analysis.detectors.time_of_day import evaluate_time_of_day_patterns


@dataclass
class _Bucket:
    hour: int
    trade_count: int
    pnl: float
    win_rate: float


def test_time_of_day_detector_evaluates_without_strategy_engine() -> None:
    suggestions = evaluate_time_of_day_patterns(
        bot_id="bot1",
        hourly_buckets=[_Bucket(hour=3, trade_count=15, pnl=-200.0, win_rate=0.2)],
        strategy_id="IARIC_v1",
        strategy_archetype="intraday_momentum",
        loss_threshold=0.35,
    )

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.detection_context.detector_name == "time_of_day"
    assert suggestion.detection_context.threshold_value == 0.35
    assert "HIGH RELEVANCE" in suggestion.archetype_note


def test_detector_catalog_owns_threshold_metadata() -> None:
    metadata = DEFAULT_DETECTOR_CATALOG.metadata

    assert metadata["exit_timing"].threshold_defaults["efficiency_threshold"] == 0.5
    assert metadata["funding_impact"].threshold_defaults["cost_threshold"] == 0.15
    assert DEFAULT_DETECTOR_CATALOG.threshold_default("time_of_day", "loss_threshold") == 0.35


def test_component_signal_decay_evaluates_without_strategy_engine() -> None:
    suggestions = evaluate_component_signal_decay(
        bot_id="bot1",
        signal_health_data={
            "components": [
                {
                    "component_name": "momentum",
                    "trade_count": 12,
                    "stability": 0.2,
                    "win_correlation": 0.01,
                }
            ]
        },
        stability_threshold=0.3,
        correlation_threshold=0.05,
    )

    assert len(suggestions) == 1
    assert suggestions[0].detection_context.detector_name == "component_signal_decay"


def test_factor_correlation_decay_evaluates_without_strategy_engine() -> None:
    suggestions = evaluate_factor_correlation_decay(
        bot_id="bot1",
        factor_rolling_data=[{
            "factor_name": "breakout_strength",
            "win_rate_trend": "degrading",
            "below_threshold": True,
            "rolling_30d_win_rate": 0.32,
            "days_of_data": 30,
        }],
    )

    assert len(suggestions) == 1
    assert suggestions[0].detection_context.detector_name == "factor_decay"
