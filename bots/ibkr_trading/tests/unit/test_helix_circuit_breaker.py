from __future__ import annotations

from datetime import datetime, timezone

import pytest

from strategies.swing.akc_helix.circuit import roll_circuit_breaker_window
from strategies.swing.akc_helix.models import CircuitBreakerState


def test_circuit_breaker_daily_bucket_resets_without_clearing_week() -> None:
    cb = CircuitBreakerState()

    roll_circuit_breaker_window(cb, datetime(2026, 1, 5, 20, tzinfo=timezone.utc))
    cb.daily_realized_r = -2.0
    cb.weekly_realized_r = -2.0

    roll_circuit_breaker_window(cb, datetime(2026, 1, 6, 14, tzinfo=timezone.utc))

    assert cb.daily_bucket == "2026-01-06"
    assert cb.daily_realized_r == pytest.approx(0.0)
    assert cb.weekly_bucket == "2026-W02"
    assert cb.weekly_realized_r == pytest.approx(-2.0)


def test_circuit_breaker_weekly_bucket_resets_at_new_iso_week() -> None:
    cb = CircuitBreakerState()

    roll_circuit_breaker_window(cb, datetime(2026, 1, 9, 20, tzinfo=timezone.utc))
    cb.daily_realized_r = -1.0
    cb.weekly_realized_r = -4.0

    roll_circuit_breaker_window(cb, datetime(2026, 1, 12, 14, tzinfo=timezone.utc))

    assert cb.daily_bucket == "2026-01-12"
    assert cb.daily_realized_r == pytest.approx(0.0)
    assert cb.weekly_bucket == "2026-W03"
    assert cb.weekly_realized_r == pytest.approx(0.0)
