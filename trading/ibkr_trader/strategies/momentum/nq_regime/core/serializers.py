from __future__ import annotations

from dataclasses import replace
from typing import Any

from strategies.core.serialization import restore_dataclass, snapshot_dataclass

from .indicators import IndicatorSnapshot
from .regime import Regime, RegimeScores
from .state import RegimeCoreState
from ..modules.base import RoutingDecisionEvent


def snapshot_state(state: RegimeCoreState) -> dict[str, Any]:
    return snapshot_dataclass(state)


def hydrate_state(payload: dict[str, Any]) -> RegimeCoreState:
    state = restore_dataclass(RegimeCoreState, payload)
    state.regime = _restore_regime(state.regime)
    if isinstance(state.regime_scores, dict):
        state.regime_scores = restore_dataclass(RegimeScores, state.regime_scores)
    if isinstance(state.indicators, dict):
        state.indicators = restore_dataclass(IndicatorSnapshot, state.indicators)
    state.routing_log = [_restore_routing_event(item) for item in state.routing_log]
    return state


def _restore_regime(value: Any) -> Regime | None:
    if value is None or isinstance(value, Regime):
        return value
    return Regime(value)


def _restore_routing_event(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    event = restore_dataclass(RoutingDecisionEvent, value)
    regime = _restore_regime(event.regime)
    scores = restore_dataclass(RegimeScores, event.regime_scores) if isinstance(event.regime_scores, dict) else event.regime_scores
    return replace(event, regime=regime, regime_scores=scores)
