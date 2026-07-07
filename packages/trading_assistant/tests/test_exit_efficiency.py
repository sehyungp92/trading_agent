"""Tests for exit efficiency schemas and pipeline."""
from trading_assistant.schemas.exit_efficiency import ExitEfficiencyRecord, ExitEfficiencyStats
from tests.factories import make_trade


def _make_trade(trade_id, pnl, exit_reason="SIGNAL", regime="trending",
                post_1h=None, post_4h=None, entry_price=100.0):
    return make_trade(trade_id=trade_id, pnl=pnl, entry_price=entry_price,
                      exit_reason=exit_reason, market_regime=regime,
                      post_exit_1h_price=post_1h, post_exit_4h_price=post_4h)


class TestExitEfficiencySchema:
    def test_record_creation(self):
        r = ExitEfficiencyRecord(
            trade_id="t1", bot_id="bot1", pair="BTC/USDT",
            pnl=10.0, exit_reason="SIGNAL", market_regime="trending",
            exit_efficiency=0.75, continuation_1h=2.0, continuation_4h=5.0,
        )
        assert r.exit_efficiency == 0.75

    def test_stats_model(self):
        s = ExitEfficiencyStats(
            bot_id="bot1", date="2026-03-01",
            avg_efficiency=0.65, premature_exit_pct=0.3,
            by_exit_reason={"SIGNAL": 0.7, "STOP_LOSS": 0.4},
            by_regime={"trending": 0.8, "ranging": 0.5},
            total_trades_with_data=20,
        )
        assert s.avg_efficiency == 0.65


class TestExitEfficiencyPipeline:
    def test_computes_from_post_exit_prices(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            _make_trade("t1", 5.0, post_1h=107.0, post_4h=108.0),
            _make_trade("t2", 3.0, post_1h=101.0, post_4h=99.0),
            _make_trade("t3", -2.0),  # no post-exit data
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.exit_efficiency(trades)
        assert isinstance(result, dict)
        assert result["total_trades_with_data"] == 2
        assert result["avg_efficiency"] > 0

    def test_premature_exit_detection(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            _make_trade("t1", 5.0, post_1h=110.0, post_4h=112.0),
            _make_trade("t2", 3.0, post_1h=108.0, post_4h=110.0),
            _make_trade("t3", 4.0, post_1h=99.0, post_4h=97.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.exit_efficiency(trades)
        assert result["premature_exit_pct"] > 0

    def test_grouping_by_exit_reason(self):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            _make_trade("t1", 5.0, exit_reason="SIGNAL", post_1h=107.0),
            _make_trade("t2", 3.0, exit_reason="STOP_LOSS", post_1h=101.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.exit_efficiency(trades)
        assert "SIGNAL" in result["by_exit_reason"]
        assert "STOP_LOSS" in result["by_exit_reason"]

    def test_exit_efficiency_written_to_curated(self, tmp_path):
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [_make_trade("t1", 5.0, post_1h=107.0, post_4h=108.0)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, [], tmp_path)
        assert (output_dir / "exit_efficiency.json").exists()
