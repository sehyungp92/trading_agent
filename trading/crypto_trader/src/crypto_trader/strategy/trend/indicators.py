"""Trend strategy indicators — reuses IncrementalIndicators from momentum."""

from __future__ import annotations

from crypto_trader.core.models import Bar
from crypto_trader.strategy.momentum.indicators import (
    IncrementalIndicators,
    IndicatorSnapshot,
)


class WeeklyTracker:
    """Track prior week high/low from D1 bars."""

    def __init__(self) -> None:
        self._current_week: int | None = None
        self._prior_week_high: float | None = None
        self._prior_week_low: float | None = None
        self._running_high: float = 0.0
        self._running_low: float = float("inf")

    @property
    def prior_week_high(self) -> float | None:
        return self._prior_week_high

    @property
    def prior_week_low(self) -> float | None:
        return self._prior_week_low

    def update(self, d1_bar: Bar) -> None:
        week = d1_bar.timestamp.isocalendar()[1]
        if self._current_week is not None and week != self._current_week:
            self._prior_week_high = self._running_high
            self._prior_week_low = self._running_low
            self._running_high = d1_bar.high
            self._running_low = d1_bar.low
        else:
            self._running_high = max(self._running_high, d1_bar.high)
            self._running_low = min(self._running_low, d1_bar.low)
        self._current_week = week


__all__ = ["IncrementalIndicators", "IndicatorSnapshot", "WeeklyTracker"]
