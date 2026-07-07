# tests/test_factor_attribution.py
"""Tests for signal factor attribution schemas and pipeline."""
from datetime import datetime, timezone

from trading_assistant.schemas.events import TradeEvent


class TestTradeEventExtensions:
    def test_signal_factors_optional_default_none(self):
        trade = TradeEvent(
            trade_id="t1", bot_id="bot1", pair="BTC/USDT",
            side="LONG", entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
            entry_price=100, exit_price=105, position_size=1, pnl=5, pnl_pct=5.0,
        )
        assert trade.signal_factors is None

    def test_signal_factors_with_data(self):
        factors = [
            {"factor_name": "rsi", "factor_value": 72.0, "threshold": 70.0, "contribution": 0.6},
            {"factor_name": "macd", "factor_value": 1.2, "threshold": 0.0, "contribution": 0.4},
        ]
        trade = TradeEvent(
            trade_id="t1", bot_id="bot1", pair="BTC/USDT",
            side="LONG", entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
            entry_price=100, exit_price=105, position_size=1, pnl=5, pnl_pct=5.0,
            signal_factors=factors,
        )
        assert len(trade.signal_factors) == 2
        assert trade.signal_factors[0]["factor_name"] == "rsi"

    def test_post_exit_prices_optional(self):
        trade = TradeEvent(
            trade_id="t1", bot_id="bot1", pair="BTC/USDT",
            side="LONG", entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
            entry_price=100, exit_price=105, position_size=1, pnl=5, pnl_pct=5.0,
            post_exit_1h_price=106.0, post_exit_4h_price=107.5,
        )
        assert trade.post_exit_1h_price == 106.0
        assert trade.post_exit_4h_price == 107.5


from trading_assistant.schemas.factor_attribution import FactorStats, FactorAttribution


class TestFactorAttributionSchema:
    def test_factor_stats_win_rate(self):
        fs = FactorStats(factor_name="rsi", trade_count=10, win_count=7, total_pnl=500.0, avg_contribution=0.6)
        assert fs.win_rate == 0.7

    def test_factor_attribution_model(self):
        fa = FactorAttribution(
            bot_id="bot1", date="2026-03-01",
            factors=[FactorStats(factor_name="rsi", trade_count=5, win_count=3, total_pnl=200.0, avg_contribution=0.5)],
        )
        assert len(fa.factors) == 1
        assert fa.factors[0].factor_name == "rsi"


class TestFactorAttributionPipeline:
    def _make_trade(self, trade_id, pnl, factors=None):
        from trading_assistant.schemas.events import TradeEvent
        return TradeEvent(
            trade_id=trade_id, bot_id="bot1", pair="BTC/USDT",
            side="LONG", entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
            entry_price=100, exit_price=100 + pnl, position_size=1,
            pnl=pnl, pnl_pct=pnl, signal_factors=factors,
        )

    def test_factor_attribution_aggregates_across_trades(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            self._make_trade("t1", 10.0, [
                {"factor_name": "rsi", "factor_value": 72, "threshold": 70, "contribution": 0.6},
                {"factor_name": "macd", "factor_value": 1.2, "threshold": 0, "contribution": 0.4},
            ]),
            self._make_trade("t2", -5.0, [
                {"factor_name": "rsi", "factor_value": 68, "threshold": 70, "contribution": 0.3},
            ]),
            self._make_trade("t3", 8.0, [
                {"factor_name": "rsi", "factor_value": 75, "threshold": 70, "contribution": 0.7},
                {"factor_name": "macd", "factor_value": 0.8, "threshold": 0, "contribution": 0.3},
            ]),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.factor_attribution(trades)
        assert isinstance(result, FactorAttribution)

        rsi = next(f for f in result.factors if f.factor_name == "rsi")
        assert rsi.trade_count == 3
        assert rsi.win_count == 2

        macd = next(f for f in result.factors if f.factor_name == "macd")
        assert macd.trade_count == 2
        assert macd.win_count == 2

    def test_factor_attribution_skips_trades_without_factors(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            self._make_trade("t1", 10.0, None),
            self._make_trade("t2", 5.0, []),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.factor_attribution(trades)
        assert result.factors == []

    def test_factor_attribution_written_to_curated(self, tmp_path):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            self._make_trade("t1", 10.0, [
                {"factor_name": "rsi", "factor_value": 72, "threshold": 70, "contribution": 0.6},
            ]),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, [], tmp_path)
        assert (output_dir / "factor_attribution.json").exists()
