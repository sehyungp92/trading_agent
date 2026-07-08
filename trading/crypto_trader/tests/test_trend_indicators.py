"""Tests for trend indicators module."""

import pytest
from datetime import datetime, timezone

from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.strategy.trend.config import IndicatorParams
from crypto_trader.strategy.trend.indicators import (
    IncrementalIndicators,
    IndicatorSnapshot,
    WeeklyTracker,
)


class TestWeeklyTracker:
    def _make_d1_bar(self, ts, high, low):
        return Bar(
            timestamp=ts, symbol="BTC",
            open=(high + low) / 2, high=high, low=low,
            close=(high + low) / 2, volume=1000.0,
            timeframe=TimeFrame.D1,
        )

    def test_initial_state(self):
        wt = WeeklyTracker()
        assert wt.prior_week_high is None
        assert wt.prior_week_low is None

    def test_week_change_sets_prior(self):
        wt = WeeklyTracker()
        # Week 11 (Mon Mar 9 - Sun Mar 15)
        wt.update(self._make_d1_bar(datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc), 50200, 49800))
        wt.update(self._make_d1_bar(datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc), 50500, 49700))

        # Week 12 starts (Mon Mar 16)
        wt.update(self._make_d1_bar(datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc), 50100, 49900))

        assert wt.prior_week_high == 50500
        assert wt.prior_week_low == 49700

    def test_running_high_low_updated(self):
        wt = WeeklyTracker()
        wt.update(self._make_d1_bar(datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc), 50000, 49000))
        wt.update(self._make_d1_bar(datetime(2026, 3, 10, 0, 0, tzinfo=timezone.utc), 51000, 48000))

        # Still same week — running values should track extremes
        assert wt.prior_week_high is None  # No week change yet

    def test_multiple_week_transitions(self):
        wt = WeeklyTracker()
        wt.update(self._make_d1_bar(datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc), 50000, 49000))
        wt.update(self._make_d1_bar(datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc), 51000, 48000))
        wt.update(self._make_d1_bar(datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc), 52000, 47000))

        # Prior week should be from week of Mar 16
        assert wt.prior_week_high == 51000
        assert wt.prior_week_low == 48000


class TestIncrementalIndicatorsCompatibility:
    def test_local_indicator_params_works(self):
        """Local IndicatorParams duck-types with IncrementalIndicators."""
        params = IndicatorParams(ema_fast=20, ema_mid=50, ema_slow=200)
        inc = IncrementalIndicators(params)
        assert inc is not None

    def test_update_returns_snapshot(self):
        """After enough bars, update returns IndicatorSnapshot."""
        params = IndicatorParams()
        inc = IncrementalIndicators(params)

        for i in range(300):
            bar = Bar(
                timestamp=datetime(2026, 3, 1, i % 24, 0, tzinfo=timezone.utc),
                symbol="BTC", open=50000 + i, high=50100 + i,
                low=49900 + i, close=50050 + i, volume=100.0,
                timeframe=TimeFrame.H1,
            )
            snap = inc.update(bar)

        # After 300 bars, should have a valid snapshot
        assert snap is not None
        assert isinstance(snap, IndicatorSnapshot)
        assert snap.atr > 0
