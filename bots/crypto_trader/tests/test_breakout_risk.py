"""Tests for breakout session risk management."""

from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.strategy.breakout.config import BreakoutLimitParams
from crypto_trader.strategy.breakout.risk import RiskManager

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestRiskManager:
    def test_daily_loss_limit(self):
        """Daily PnL exceeding limit stops session."""
        cfg = BreakoutLimitParams(max_daily_loss_pct=0.02)
        rm = RiskManager(cfg)
        equity = 10000.0
        # Record a loss of $250 (2.5% of equity)
        rm.record_trade_exit(-250.0, TS)
        stopped, reason = rm.is_session_stopped(equity, TS)
        assert stopped is True
        assert reason == "daily_loss_limit"

    def test_consecutive_losses(self):
        """Consecutive losses reaching limit stops session."""
        cfg = BreakoutLimitParams(max_consecutive_losses=3, max_daily_loss_pct=1.0)
        rm = RiskManager(cfg)
        equity = 10000.0
        rm.record_trade_exit(-10.0, TS)
        rm.record_trade_exit(-10.0, TS)
        stopped, _ = rm.is_session_stopped(equity, TS)
        assert stopped is False
        rm.record_trade_exit(-10.0, TS)
        stopped, reason = rm.is_session_stopped(equity, TS)
        assert stopped is True
        assert reason == "consecutive_losses"

    def test_trade_count_limit(self):
        """trades_today >= max stops session."""
        cfg = BreakoutLimitParams(
            max_trades_per_day=2, max_daily_loss_pct=1.0, max_consecutive_losses=999,
        )
        rm = RiskManager(cfg)
        equity = 10000.0
        rm.record_trade_exit(50.0, TS)
        rm.record_trade_exit(50.0, TS)
        stopped, reason = rm.is_session_stopped(equity, TS)
        assert stopped is True
        assert reason == "daily_trade_limit"

    def test_day_rollover_resets(self):
        """New day resets daily counters and consecutive losses."""
        cfg = BreakoutLimitParams(
            max_daily_loss_pct=0.02,
            max_trades_per_day=5,
            max_consecutive_losses=999,
        )
        rm = RiskManager(cfg)
        equity = 10000.0
        # Day 1: lose $250 -> stopped
        rm.record_trade_exit(-250.0, TS)
        stopped, _ = rm.is_session_stopped(equity, TS)
        assert stopped is True
        # Day 2: all counters reset (including consecutive losses)
        day2 = TS + timedelta(days=1)
        stopped, _ = rm.is_session_stopped(equity, day2)
        assert stopped is False
        assert rm._consecutive_losses == 0
