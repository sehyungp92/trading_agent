from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

from strategies.core.events import DecisionEvent


def normalize_decision_event(event: DecisionEvent) -> dict[str, Any]:
    return {
        "schema_version": event.schema_version,
        "event_type": event.event_type,
        "code": event.code,
        "ts": event.ts.isoformat(),
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "bot_id": event.bot_id,
        "strategy_id": event.strategy_id,
        "family_id": event.family_id,
        "portfolio_id": event.portfolio_id,
        "strategy_version": event.strategy_version,
        "config_version": event.config_version,
        "portfolio_config_version": event.portfolio_config_version,
        "risk_config_version": event.risk_config_version,
        "allocation_version": event.allocation_version,
        "strategy_registry_version": event.strategy_registry_version,
        "deployment_id": event.deployment_id,
        "parameter_set_id": event.parameter_set_id,
        "code_sha": event.code_sha,
        "trace_id": event.trace_id,
        "bar_id": event.bar_id,
        "decision_kind": event.decision_kind,
        "sequence": event.sequence,
        "state_ref": event.state_ref,
        "emitted_actions": list(event.emitted_actions),
        "details": _normalize_value(event.details),
    }


def normalize_decision_stream(events: Iterable[DecisionEvent]) -> list[dict[str, Any]]:
    return [normalize_decision_event(event) for event in events]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _normalize_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value
