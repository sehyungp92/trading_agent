"""Slice coverage adapters over validation-owned coverage logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from trading_assistant_data.validation import coverage_counts


@dataclass(frozen=True)
class SliceCoverageReport:
    expected: int
    actual: int
    missing_ranges: list


def coverage_report(
    *,
    timestamps: Iterable,
    timeframe: str,
    calendar,
    exchange_timestamps: Iterable | None = None,
) -> SliceCoverageReport:
    expected, actual, missing = coverage_counts(
        timestamps,
        timeframe,
        calendar,
        exchange_timestamps=exchange_timestamps,
    )
    return SliceCoverageReport(expected=expected, actual=actual, missing_ranges=missing)
