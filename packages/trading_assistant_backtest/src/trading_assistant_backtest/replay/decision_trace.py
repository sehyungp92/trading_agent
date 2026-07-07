"""Normalized decision trace types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DecisionTraceEvent:
    ts: datetime | None
    dimension: str
    key: str
    payload: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> DecisionTraceEvent:
        return DecisionTraceEvent(
            ts=self.ts,
            dimension=self.dimension.strip().lower(),
            key=self.key.strip(),
            payload=_canonical_payload(self.payload),
        )


def normalize_trace(events: Iterable[DecisionTraceEvent]) -> list[DecisionTraceEvent]:
    return [event.normalized() for event in events]


def event_signature(event: DecisionTraceEvent) -> tuple[str, str, str, tuple[tuple[str, Any], ...]]:
    normalized = event.normalized()
    ts = normalized.ts.isoformat() if normalized.ts is not None else ""
    return (
        normalized.dimension,
        normalized.key,
        ts,
        tuple(sorted(normalized.payload.items())),
    )


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _canonical_value(value) for key, value in sorted(payload.items())}


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return tuple((str(key), _canonical_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list | tuple):
        return tuple(_canonical_value(item) for item in value)
    return value
