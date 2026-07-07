"""Tests for SlippageAnalyzer — computes per-symbol, per-hour slippage distributions."""
from datetime import datetime

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.slippage_analysis import (
    SlippageBucket,
    SlippageDistribution,
    SlippageTrend,
)
from trading_assistant.skills.slippage_analyzer import SlippageAnalyzer
from tests.factories import make_trade


def _make_trade(pair: str, entry_price: float, exit_price: float,
                spread_at_entry: float, entry_hour: int = 14) -> TradeEvent:
    pnl = exit_price - entry_price
    return make_trade(
        trade_id=f"t_{pair}_{entry_hour}",
        pair=pair,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=datetime(2026, 3, 1, entry_hour, 0, 0),
        exit_time=datetime(2026, 3, 1, entry_hour + 1, 0, 0),
        pnl=pnl,
        pnl_pct=(pnl / entry_price) * 100,
        spread_at_entry=spread_at_entry,
    )


class TestSlippageAnalyzer:
    def test_compute_by_symbol(self):
        trades = [
            _make_trade("BTCUSDT", 50000, 50100, 5.0),
            _make_trade("BTCUSDT", 50200, 50300, 4.0),
            _make_trade("ETHUSDT", 3000, 3050, 8.0),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute(trades)

        assert isinstance(dist, SlippageDistribution)
        assert "BTCUSDT" in dist.by_symbol
        assert dist.by_symbol["BTCUSDT"].sample_count == 2
        assert "ETHUSDT" in dist.by_symbol
        assert dist.by_symbol["ETHUSDT"].sample_count == 1

    def test_compute_by_hour(self):
        trades = [
            _make_trade("BTCUSDT", 50000, 50100, 5.0, entry_hour=9),
            _make_trade("BTCUSDT", 50200, 50300, 3.0, entry_hour=9),
            _make_trade("BTCUSDT", 50400, 50500, 7.0, entry_hour=15),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute(trades)

        assert "09" in dist.by_hour
        assert dist.by_hour["09"].sample_count == 2
        assert "15" in dist.by_hour
        assert dist.by_hour["15"].sample_count == 1

    def test_slippage_bps_from_spread(self):
        """Spread at entry is the primary slippage signal."""
        trades = [
            _make_trade("BTCUSDT", 50000, 50100, 10.0),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute(trades)

        # Slippage = spread_at_entry (already in bps)
        assert dist.by_symbol["BTCUSDT"].mean_bps == 10.0

    def test_empty_trades(self):
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute([])
        assert dist.by_symbol == {}
        assert dist.by_hour == {}

    def test_export_for_cost_model(self):
        """Exports regime->bps mapping for empirical cost models."""
        trades = [
            make_trade(
                trade_id="t1", pair="BTCUSDT",
                entry_time=datetime(2026, 3, 1, 14, 0),
                exit_time=datetime(2026, 3, 1, 15, 0),
                entry_price=50000, exit_price=50100,
                pnl=100, pnl_pct=0.2,
                spread_at_entry=5.0, market_regime="trending_up",
            ),
            make_trade(
                trade_id="t2", pair="BTCUSDT",
                entry_time=datetime(2026, 3, 1, 16, 0),
                exit_time=datetime(2026, 3, 1, 17, 0),
                entry_price=50200, exit_price=50300,
                pnl=100, pnl_pct=0.2,
                spread_at_entry=8.0, market_regime="ranging",
            ),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        regime_bps = analyzer.export_regime_bps(trades)
        assert "trending_up" in regime_bps
        assert regime_bps["trending_up"] == 5.0
        assert regime_bps["ranging"] == 8.0


class TestSlippageBucket:
    def test_creates_with_stats(self):
        bucket = SlippageBucket(
            key="BTCUSDT",
            sample_count=50,
            mean_bps=4.2,
            median_bps=3.8,
            p75_bps=5.5,
            p95_bps=8.1,
        )
        assert bucket.mean_bps == 4.2
        assert bucket.sample_count == 50


class TestSlippageDistribution:
    def test_creates_per_symbol(self):
        dist = SlippageDistribution(
            bot_id="bot1",
            date="2026-03-01",
            by_symbol={
                "BTCUSDT": SlippageBucket(key="BTCUSDT", sample_count=50, mean_bps=4.2),
                "ETHUSDT": SlippageBucket(key="ETHUSDT", sample_count=30, mean_bps=6.1),
            },
        )
        assert len(dist.by_symbol) == 2

    def test_creates_per_hour(self):
        dist = SlippageDistribution(
            bot_id="bot1",
            date="2026-03-01",
            by_hour={
                "09": SlippageBucket(key="09", sample_count=10, mean_bps=3.0),
                "15": SlippageBucket(key="15", sample_count=15, mean_bps=5.0),
            },
        )
        assert dist.by_hour["15"].mean_bps == 5.0


class TestSlippageTrend:
    def test_creates_trend(self):
        trend = SlippageTrend(
            bot_id="bot1",
            symbol="BTCUSDT",
            weekly_mean_bps=[3.5, 4.0, 4.2, 4.8],
            trend_direction="increasing",
        )
        assert trend.trend_direction == "increasing"
