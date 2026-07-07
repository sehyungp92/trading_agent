# tests/test_daily_metrics.py
"""Tests for daily metrics schemas."""

from trading_assistant.schemas.daily_metrics import (
    BotDailySummary,
    WinnerLoserRecord,
    ProcessFailureRecord,
    NotableMissedRecord,
    RegimeAnalysis,
    FilterAnalysis,
    AnomalyRecord,
    RootCauseSummary,
)


class TestBotDailySummary:
    def test_creates_from_minimal_data(self):
        summary = BotDailySummary(
            date="2026-03-01",
            bot_id="bot1",
            total_trades=20,
            win_count=12,
            loss_count=8,
            gross_pnl=150.0,
            net_pnl=140.0,
        )
        assert summary.bot_id == "bot1"
        assert summary.win_rate == 0.6

    def test_win_rate_zero_trades(self):
        summary = BotDailySummary(date="2026-03-01", bot_id="bot1")
        assert summary.win_rate == 0.0

    def test_profit_factor_zero_losses(self):
        summary = BotDailySummary(
            date="2026-03-01",
            bot_id="bot1",
            total_trades=5,
            win_count=5,
            loss_count=0,
            avg_win=100.0,
            avg_loss=0.0,
        )
        assert summary.profit_factor == float("inf")


class TestWinnerLoserRecord:
    def test_creates_with_required_fields(self):
        rec = WinnerLoserRecord(
            trade_id="t1",
            bot_id="bot1",
            pair="BTCUSDT",
            side="LONG",
            pnl=250.0,
            pnl_pct=2.5,
            entry_signal="EMA cross",
            exit_reason="TAKE_PROFIT",
            market_regime="trending_up",
            process_quality_score=85,
            root_causes=["normal_win"],
        )
        assert rec.pnl == 250.0
        assert rec.root_causes == ["normal_win"]


class TestProcessFailureRecord:
    def test_creates_with_low_score(self):
        rec = ProcessFailureRecord(
            trade_id="t2",
            bot_id="bot1",
            pair="ETHUSDT",
            process_quality_score=45,
            root_causes=["regime_mismatch", "weak_signal"],
            pnl=-80.0,
        )
        assert rec.process_quality_score < 60


class TestNotableMissedRecord:
    def test_creates_with_outcome(self):
        rec = NotableMissedRecord(
            bot_id="bot1",
            pair="BTCUSDT",
            signal="RSI divergence",
            blocked_by="volatility_filter",
            hypothetical_entry=50000.0,
            outcome_24h=500.0,
            confidence=0.7,
            assumption_tags=["mid_fill", "zero_slippage"],
        )
        assert rec.blocked_by == "volatility_filter"


class TestRegimeAnalysis:
    def test_creates_with_breakdown(self):
        ra = RegimeAnalysis(
            bot_id="bot1",
            date="2026-03-01",
            regime_pnl={"trending_up": 200.0, "ranging": -50.0},
            regime_trade_count={"trending_up": 8, "ranging": 5},
            regime_win_rate={"trending_up": 0.75, "ranging": 0.4},
        )
        assert ra.regime_pnl["trending_up"] == 200.0


class TestFilterAnalysis:
    def test_creates_with_filter_impact(self):
        fa = FilterAnalysis(
            bot_id="bot1",
            date="2026-03-01",
            filter_block_counts={"volatility_filter": 5, "spread_filter": 2},
            filter_saved_pnl={"volatility_filter": 300.0, "spread_filter": 100.0},
            filter_missed_pnl={"volatility_filter": -50.0, "spread_filter": 0.0},
        )
        assert fa.filter_block_counts["volatility_filter"] == 5


class TestAnomalyRecord:
    def test_creates_anomaly(self):
        a = AnomalyRecord(
            bot_id="bot1",
            date="2026-03-01",
            anomaly_type="volume_spike",
            description="Volume 3× above 30-day average on ETHUSDT",
            severity="medium",
            related_trades=["t5", "t6"],
        )
        assert a.anomaly_type == "volume_spike"


class TestRootCauseSummary:
    def test_creates_distribution(self):
        rcs = RootCauseSummary(
            bot_id="bot1",
            date="2026-03-01",
            distribution={"normal_win": 8, "regime_mismatch": 3, "weak_signal": 2},
            total_trades=13,
        )
        assert sum(rcs.distribution.values()) == rcs.total_trades
