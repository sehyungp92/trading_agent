from datetime import date, datetime, time, timezone

from libs.config.market_calendar import AssetClass, MarketCalendar


def test_us_equity_holiday_is_blocked() -> None:
    calendar = MarketCalendar()

    assert calendar.is_market_holiday(date(2025, 12, 25), AssetClass.EQUITY)


def test_half_day_afternoon_blocks_new_entries() -> None:
    calendar = MarketCalendar()

    blocked = calendar.is_entry_blocked(
        datetime(2025, 11, 28, 17, 30, tzinfo=timezone.utc),
        AssetClass.EQUITY,
    )

    assert blocked is not None
    assert "Half-day session" in blocked


# ---------------------------------------------------------------------------
# Pre-market / post-market entry guard (equity only)
# ---------------------------------------------------------------------------


def test_equity_premarket_blocked() -> None:
    """Equity entries before 09:30 ET should be blocked."""
    calendar = MarketCalendar()
    # 2025-03-17 is a Monday (not a holiday), EDT = UTC-4
    # 13:29 UTC = 09:29 ET
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 13, 29, tzinfo=timezone.utc),
        AssetClass.EQUITY,
    )
    assert blocked is not None
    assert "Pre-market" in blocked


def test_equity_at_open_not_blocked() -> None:
    """Equity entries at exactly 09:30 ET should NOT be blocked."""
    calendar = MarketCalendar()
    # 13:30 UTC = 09:30 ET (EDT)
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 13, 30, tzinfo=timezone.utc),
        AssetClass.EQUITY,
    )
    assert blocked is None


def test_equity_midday_not_blocked() -> None:
    """Normal trading hours should not block entries."""
    calendar = MarketCalendar()
    # 18:00 UTC = 14:00 ET (EDT)
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 18, 0, tzinfo=timezone.utc),
        AssetClass.EQUITY,
    )
    assert blocked is None


def test_equity_postmarket_blocked() -> None:
    """Equity entries at or after 16:00 ET (normal close) should be blocked."""
    calendar = MarketCalendar()
    # 20:00 UTC = 16:00 ET (EDT)
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 20, 0, tzinfo=timezone.utc),
        AssetClass.EQUITY,
    )
    assert blocked is not None
    assert "Post-market" in blocked


def test_equity_just_before_close_not_blocked() -> None:
    """15:59 ET should still be allowed."""
    calendar = MarketCalendar()
    # 19:59 UTC = 15:59 ET (EDT)
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 19, 59, tzinfo=timezone.utc),
        AssetClass.EQUITY,
    )
    assert blocked is None


def test_cme_futures_no_premarket_block() -> None:
    """CME futures should NOT have pre-market blocking (nearly 24h sessions)."""
    calendar = MarketCalendar()
    # 2025-03-17 05:00 UTC — very early, but CME trades
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 5, 0, tzinfo=timezone.utc),
        AssetClass.CME_FUTURES,
    )
    assert blocked is None


def test_cme_futures_no_postmarket_block() -> None:
    """CME futures should NOT have post-market blocking."""
    calendar = MarketCalendar()
    # 2025-03-17 22:00 UTC — after equity close, but CME trades
    blocked = calendar.is_entry_blocked(
        datetime(2025, 3, 17, 22, 0, tzinfo=timezone.utc),
        AssetClass.CME_FUTURES,
    )
    assert blocked is None


def test_weekend_blocked_for_both_asset_classes() -> None:
    """Weekends should block both equity and CME."""
    calendar = MarketCalendar()
    # 2025-03-15 is a Saturday
    saturday = datetime(2025, 3, 15, 15, 0, tzinfo=timezone.utc)
    assert calendar.is_entry_blocked(saturday, AssetClass.EQUITY) is not None
    assert calendar.is_entry_blocked(saturday, AssetClass.CME_FUTURES) is not None

