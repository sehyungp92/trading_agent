"""Tests for HourlyAnalyzer — time-of-day performance buckets."""
from datetime import datetime

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.hourly_performance import HourlyBucket, HourlyPerformance
from trading_assistant.skills.hourly_analyzer import HourlyAnalyzer
from tests.factories import make_trade


def _make_trade(hour: int, pnl: float, quality: int = 80) -> TradeEvent:
    return make_trade(
        trade_id=f"t_{hour}_{pnl}",
        pair="BTCUSDT",
        entry_price=50000,
        entry_time=datetime(2026, 3, 1, hour, 30, 0),
        exit_time=datetime(2026, 3, 1, hour + 1, 0, 0),
        pnl=pnl,
        pnl_pct=pnl / 500,
        process_quality_score=quality,
    )


class TestHourlyAnalyzer:
    def test_basic_bucketing(self):
        trades = [
            _make_trade(9, 100.0),
            _make_trade(9, 50.0),
            _make_trade(15, -80.0),
        ]
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute(trades)

        assert isinstance(perf, HourlyPerformance)
        buckets_by_hour = {b.hour: b for b in perf.buckets}
        assert buckets_by_hour[9].trade_count == 2
        assert buckets_by_hour[9].pnl == 150.0
        assert buckets_by_hour[9].win_rate == 1.0
        assert buckets_by_hour[15].trade_count == 1
        assert buckets_by_hour[15].pnl == -80.0
        assert buckets_by_hour[15].win_rate == 0.0

    def test_avg_process_quality(self):
        trades = [
            _make_trade(14, 100.0, quality=90),
            _make_trade(14, -50.0, quality=60),
        ]
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute(trades)

        buckets_by_hour = {b.hour: b for b in perf.buckets}
        assert buckets_by_hour[14].avg_process_quality == 75.0

    def test_empty_trades(self):
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute([])
        assert perf.buckets == []

    def test_best_worst_hours(self):
        trades = [
            _make_trade(9, 200.0),
            _make_trade(12, -100.0),
            _make_trade(18, 50.0),
        ]
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute(trades)
        assert perf.best_hour == 9
        assert perf.worst_hour == 12


class TestHourlyBucket:
    def test_creates_with_stats(self):
        bucket = HourlyBucket(
            hour=14,
            trade_count=8,
            pnl=250.0,
            win_rate=0.75,
            avg_process_quality=82.0,
        )
        assert bucket.hour == 14
        assert bucket.win_rate == 0.75

    def test_defaults(self):
        bucket = HourlyBucket(hour=0)
        assert bucket.trade_count == 0
        assert bucket.pnl == 0.0


class TestHourlyPerformance:
    def test_creates_with_buckets(self):
        perf = HourlyPerformance(
            bot_id="bot1",
            date="2026-03-01",
            buckets=[
                HourlyBucket(hour=9, trade_count=5, pnl=100.0, win_rate=0.8),
                HourlyBucket(hour=15, trade_count=3, pnl=-50.0, win_rate=0.33),
            ],
        )
        assert len(perf.buckets) == 2

    def test_best_hour_property(self):
        perf = HourlyPerformance(
            bot_id="bot1",
            date="2026-03-01",
            buckets=[
                HourlyBucket(hour=9, trade_count=5, pnl=100.0, win_rate=0.8),
                HourlyBucket(hour=15, trade_count=3, pnl=-50.0, win_rate=0.33),
                HourlyBucket(hour=20, trade_count=4, pnl=200.0, win_rate=0.75),
            ],
        )
        assert perf.best_hour == 20

    def test_worst_hour_property(self):
        perf = HourlyPerformance(
            bot_id="bot1",
            date="2026-03-01",
            buckets=[
                HourlyBucket(hour=9, pnl=100.0),
                HourlyBucket(hour=15, pnl=-50.0),
            ],
        )
        assert perf.worst_hour == 15

    def test_empty_buckets_returns_none(self):
        perf = HourlyPerformance(bot_id="bot1", date="2026-03-01")
        assert perf.best_hour is None
        assert perf.worst_hour is None
