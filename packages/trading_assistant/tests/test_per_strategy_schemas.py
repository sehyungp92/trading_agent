# tests/test_per_strategy_schemas.py
"""Tests for per-strategy schema extensions."""
from trading_assistant.schemas.daily_metrics import PerStrategySummary, BotDailySummary
from trading_assistant.schemas.events import DailySnapshot
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary, BotWeeklySummary


class TestPerStrategySummary:
    def test_creates_with_defaults(self):
        s = PerStrategySummary(strategy_id="momentum_long")
        assert s.strategy_id == "momentum_long"
        assert s.trades == 0
        assert s.net_pnl == 0.0
        assert s.symbols_traded == []

    def test_full_fields(self):
        s = PerStrategySummary(
            strategy_id="mean_revert",
            trades=15,
            win_count=10,
            loss_count=5,
            gross_pnl=800.0,
            net_pnl=720.0,
            win_rate=0.667,
            avg_win=100.0,
            avg_loss=-40.0,
            best_trade_pnl=250.0,
            worst_trade_pnl=-80.0,
            avg_entry_slippage_bps=2.5,
            avg_mfe_pct=3.2,
            avg_mae_pct=1.1,
            avg_exit_efficiency=0.72,
            symbols_traded=["NQ", "ES"],
        )
        assert s.trades == 15
        assert s.avg_exit_efficiency == 0.72
        assert "NQ" in s.symbols_traded


class TestBotDailySummaryPerStrategy:
    def test_empty_per_strategy_by_default(self):
        b = BotDailySummary(date="2026-03-01", bot_id="bot1")
        assert b.per_strategy_summary == {}

    def test_with_strategy_data(self):
        b = BotDailySummary(
            date="2026-03-01",
            bot_id="momentum_nq_01",
            per_strategy_summary={
                "momentum_long": PerStrategySummary(
                    strategy_id="momentum_long", trades=5, net_pnl=200.0,
                ),
                "momentum_short": PerStrategySummary(
                    strategy_id="momentum_short", trades=3, net_pnl=-50.0,
                ),
            },
        )
        assert len(b.per_strategy_summary) == 2
        assert b.per_strategy_summary["momentum_long"].net_pnl == 200.0


class TestDailySnapshotPerStrategy:
    def test_empty_per_strategy_by_default(self):
        snap = DailySnapshot(date="2026-03-01", bot_id="bot1")
        assert snap.per_strategy_summary == {}
        assert snap.overlay_state_summary is None

    def test_with_per_strategy_and_overlay(self):
        snap = DailySnapshot(
            date="2026-03-01",
            bot_id="swing_multi_01",
            per_strategy_summary={
                "ATRSS": {"trades": 3, "net_pnl": 150.0},
                "Helix": {"trades": 2, "net_pnl": -30.0},
            },
            overlay_state_summary={"state": "bullish", "confidence": 0.8},
        )
        assert len(snap.per_strategy_summary) == 2
        assert snap.overlay_state_summary["state"] == "bullish"


class TestStrategyWeeklySummary:
    def test_creates_with_defaults(self):
        s = StrategyWeeklySummary(strategy_id="trend_follow", bot_id="bot1")
        assert s.total_trades == 0
        assert s.daily_pnl == {}

    def test_full_weekly_strategy(self):
        s = StrategyWeeklySummary(
            strategy_id="mean_revert",
            bot_id="swing_multi_01",
            total_trades=28,
            win_count=18,
            loss_count=10,
            gross_pnl=1400.0,
            net_pnl=1200.0,
            win_rate=0.643,
            avg_win=120.0,
            avg_loss=-50.0,
            daily_pnl={
                "2026-02-24": 200.0,
                "2026-02-25": -50.0,
                "2026-02-26": 300.0,
            },
        )
        assert s.total_trades == 28
        assert len(s.daily_pnl) == 3


class TestBotWeeklySummaryPerStrategy:
    def test_empty_per_strategy_by_default(self):
        b = BotWeeklySummary(
            week_start="2026-02-24", week_end="2026-03-02", bot_id="bot1",
        )
        assert b.per_strategy_summary == {}

    def test_with_strategy_data(self):
        b = BotWeeklySummary(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bot_id="momentum_nq_01",
            per_strategy_summary={
                "strat_a": StrategyWeeklySummary(
                    strategy_id="strat_a", bot_id="momentum_nq_01",
                    total_trades=10, net_pnl=500.0,
                ),
            },
        )
        assert "strat_a" in b.per_strategy_summary
        assert b.per_strategy_summary["strat_a"].net_pnl == 500.0
