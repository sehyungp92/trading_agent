"""Paper shadow replay comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ParityDrift:
    index: int
    field: str
    expected: Any
    actual: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class ShadowReplayReport:
    expected_count: int
    actual_count: int
    drifts: list[ParityDrift] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.drifts and self.expected_count == self.actual_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "drifts": [drift.to_dict() for drift in self.drifts],
        }


def compare_event_streams(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    keys: tuple[str, ...] = ("decision_id", "intent_id", "kind", "symbol", "side"),
) -> ShadowReplayReport:
    """Compare canonical replay/live event dictionaries by stable keys."""
    drifts: list[ParityDrift] = []
    for idx, (left, right) in enumerate(zip(expected, actual)):
        for key in keys:
            if left.get(key) != right.get(key):
                drifts.append(ParityDrift(
                    index=idx,
                    field=key,
                    expected=left.get(key),
                    actual=right.get(key),
                ))
    if len(expected) != len(actual):
        drifts.append(ParityDrift(
            index=min(len(expected), len(actual)),
            field="event_count",
            expected=len(expected),
            actual=len(actual),
        ))
    return ShadowReplayReport(
        expected_count=len(expected),
        actual_count=len(actual),
        drifts=drifts,
    )
