"""Tests for momentum session risk management."""

from datetime import datetime, timedelta, timezone

from crypto_trader.strategy.momentum.config import DailyLimitParams
from crypto_trader.strategy.momentum.risk import SessionRiskManager


class TestSessionRiskManager:
    def test_overnight_close_counts_in_new_day(self):
        rm = SessionRiskManager(DailyLimitParams(max_daily_loss_pct=0.02))
        day1 = datetime(2026, 3, 15, 23, 45, tzinfo=timezone.utc)
        day2 = day1 + timedelta(minutes=30)
        rm.is_session_stopped(10000, day1)
        rm.record_trade(-250, day2)
        stopped, reason = rm.is_session_stopped(10000, day2)
        assert stopped is True
        assert reason == "daily_loss_limit"

    def test_new_day_resets_before_recording_losses(self):
        rm = SessionRiskManager(
            DailyLimitParams(max_consecutive_losses=2, max_daily_loss_pct=1.0),
        )
        day1 = datetime(2026, 3, 15, 22, 0, tzinfo=timezone.utc)
        day2 = day1 + timedelta(days=1)

        rm.is_session_stopped(10000, day1)
        rm.record_trade(-10, day1)
        stopped, _ = rm.is_session_stopped(10000, day1)
        assert stopped is False

        rm.record_trade(-10, day2)
        stopped, _ = rm.is_session_stopped(10000, day2)
        assert stopped is False

        rm.record_trade(-10, day2)
        stopped, reason = rm.is_session_stopped(10000, day2)
        assert stopped is True
        assert reason == "consecutive_losses"
