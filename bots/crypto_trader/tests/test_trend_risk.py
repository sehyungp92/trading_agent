"""Tests for trend session risk management."""

import pytest
from datetime import datetime, timedelta, timezone

from crypto_trader.strategy.trend.config import TrendLimitParams
from crypto_trader.strategy.trend.risk import RiskManager


class TestRiskManager:
    def test_not_stopped_initially(self):
        rm = RiskManager(TrendLimitParams())
        t = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        stopped, reason = rm.is_session_stopped(10000, t)
        assert stopped is False
        assert reason == ""

    def test_daily_loss_cap(self):
        rm = RiskManager(TrendLimitParams(max_daily_loss_pct=0.02))
        t = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        rm.is_session_stopped(10000, t)  # Init date
        rm.record_trade(-250, t)  # 2.5% loss > 2% cap
        stopped, reason = rm.is_session_stopped(9750, t)
        assert stopped is True
        assert reason == "daily_loss_limit"

    def test_consecutive_loss_limit(self):
        rm = RiskManager(TrendLimitParams(max_consecutive_losses=2))
        t = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        rm.is_session_stopped(10000, t)  # Init date
        rm.record_trade(-50, t)
        rm.record_trade(-50, t)
        stopped, reason = rm.is_session_stopped(9900, t)
        assert stopped is True
        assert reason == "consecutive_losses"

    def test_consecutive_losses_reset_on_win(self):
        rm = RiskManager(TrendLimitParams(max_consecutive_losses=2))
        t = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        rm.record_trade(-50, t)
        rm.record_trade(100, t)  # Win resets count
        rm.record_trade(-50, t)
        stopped, _ = rm.is_session_stopped(10000, t)
        assert stopped is False

    def test_date_reset(self):
        rm = RiskManager(TrendLimitParams(max_daily_loss_pct=0.02))
        day1 = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        rm.record_trade(-150, day1)
        # Next day — daily PnL resets
        day2 = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)
        stopped, _ = rm.is_session_stopped(9850, day2)
        assert stopped is False

    def test_trade_count_limit(self):
        rm = RiskManager(TrendLimitParams(max_trades_per_day=3))
        t = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        rm.is_session_stopped(10000, t)  # Init date
        for _ in range(3):
            rm.record_trade(10, t)
        stopped, reason = rm.is_session_stopped(10030, t)
        assert stopped is True
        assert reason == "daily_trade_limit"

    def test_overnight_close_counts_in_new_day(self):
        rm = RiskManager(TrendLimitParams(max_daily_loss_pct=0.02))
        day1 = datetime(2026, 3, 15, 23, 45, tzinfo=timezone.utc)
        day2 = day1 + timedelta(minutes=30)
        rm.is_session_stopped(10000, day1)
        rm.record_trade(-250, day2)
        stopped, reason = rm.is_session_stopped(10000, day2)
        assert stopped is True
        assert reason == "daily_loss_limit"
