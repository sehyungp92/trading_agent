# tests/test_weekly_metrics.py
"""Tests for weekly metrics schemas."""
from trading_assistant.schemas.weekly_metrics import (
    BotWeeklySummary,
    WeeklySummary,
    WeekOverWeekComparison,
    ProcessQualityTrend,
    RegimePerformanceTrend,
    FilterWeeklySummary,
    CorrelationSummary,
)


class TestBotWeeklySummary:
    def test_creates_from_minimal_data(self):
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=30,
            loss_count=20,
            gross_pnl=1200.0,
            net_pnl=1100.0,
        )
        assert summary.bot_id == "bot1"
        assert summary.win_rate == 0.6

    def test_win_rate_zero_trades(self):
        summary = BotWeeklySummary(
            week_start="2026-02-23", week_end="2026-03-01", bot_id="bot1"
        )
        assert summary.win_rate == 0.0

    def test_profit_factor_zero_losses(self):
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=5,
            win_count=5,
            loss_count=0,
            avg_win=100.0,
            avg_loss=0.0,
        )
        assert summary.profit_factor == float("inf")

    def test_daily_pnl_series(self):
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            daily_pnl={"2026-02-23": 50.0, "2026-02-24": -20.0, "2026-02-25": 80.0},
        )
        assert len(summary.daily_pnl) == 3


class TestWeeklySummary:
    def test_creates_with_bots(self):
        ws = WeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_summaries={
                "bot1": BotWeeklySummary(
                    week_start="2026-02-23",
                    week_end="2026-03-01",
                    bot_id="bot1",
                    total_trades=30,
                    net_pnl=500.0,
                ),
            },
            total_net_pnl=500.0,
            total_trades=30,
        )
        assert ws.total_net_pnl == 500.0
        assert "bot1" in ws.bot_summaries


class TestWeekOverWeekComparison:
    def test_creates_comparison(self):
        wow = WeekOverWeekComparison(
            current_week="2026-02-23",
            previous_week="2026-02-16",
            pnl_delta=150.0,
            pnl_delta_pct=12.5,
            win_rate_delta=0.05,
            trade_count_delta=-3,
            avg_process_quality_delta=2.0,
        )
        assert wow.pnl_delta == 150.0
        assert wow.pnl_delta_pct == 12.5

    def test_negative_deltas(self):
        wow = WeekOverWeekComparison(
            current_week="2026-02-23",
            previous_week="2026-02-16",
            pnl_delta=-200.0,
            pnl_delta_pct=-15.0,
            win_rate_delta=-0.1,
        )
        assert wow.pnl_delta < 0


class TestProcessQualityTrend:
    def test_creates_trend(self):
        pqt = ProcessQualityTrend(
            bot_id="bot1",
            weekly_avg_scores=[72.0, 75.0, 78.0, 80.0],
            current_avg=80.0,
            trend_direction="improving",
            most_frequent_root_causes={"regime_mismatch": 12, "normal_loss": 25},
        )
        assert pqt.trend_direction == "improving"
        assert pqt.current_avg == 80.0


class TestRegimePerformanceTrend:
    def test_creates_trend(self):
        rpt = RegimePerformanceTrend(
            bot_id="bot1",
            regime="trending_up",
            weekly_pnl=[200.0, 180.0, 250.0, 300.0],
            weekly_win_rate=[0.7, 0.65, 0.75, 0.8],
            weekly_trade_count=[10, 8, 12, 11],
        )
        assert rpt.regime == "trending_up"
        assert len(rpt.weekly_pnl) == 4


class TestFilterWeeklySummary:
    def test_creates_summary(self):
        fws = FilterWeeklySummary(
            bot_id="bot1",
            filter_name="volatility_filter",
            total_blocks=47,
            blocks_that_would_have_won=31,
            blocks_that_would_have_lost=16,
            net_impact_pnl=-180.0,
            confidence=0.7,
        )
        assert fws.net_impact_pnl == -180.0
        assert fws.total_blocks == 47


class TestCorrelationSummary:
    def test_creates_summary(self):
        cs = CorrelationSummary(
            bot_a="bot1",
            bot_b="bot2",
            rolling_30d_correlation=0.45,
            weekly_pnl_correlation=0.52,
            same_direction_pct=0.6,
        )
        assert cs.rolling_30d_correlation == 0.45
