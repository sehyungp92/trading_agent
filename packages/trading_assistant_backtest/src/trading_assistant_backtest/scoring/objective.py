"""Objective component vocabulary."""

from __future__ import annotations

from trading_assistant_backtest.scoring.immutable import (
    component_names_for_profile,
    resolve_score_profile,
)

DEFAULT_OBJECTIVE_COMPONENTS = [
    "expected_return",
    "trade_frequency",
    "edge_quality",
    "drawdown_resilience",
    "capture_quality",
    "robustness",
    "balance_or_rule_health",
]


def capped_components(
    cap: int,
    *,
    family: str = "",
    plugin_id: str = "",
    strategy_id: str = "",
) -> list[str]:
    profile = resolve_score_profile(
        family=family,
        plugin_id=plugin_id,
        strategy_id=strategy_id,
    )
    return component_names_for_profile(profile, cap)
