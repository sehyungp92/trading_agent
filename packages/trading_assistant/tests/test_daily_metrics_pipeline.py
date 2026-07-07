"""Tests for the daily metrics data reduction pipeline."""
import json
from pathlib import Path

import pytest

from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
from tests.factories import make_trade as _factory_trade, make_missed as _factory_missed


def _make_trade(trade_id: str, bot_id: str, pnl: float, **kwargs) -> TradeEvent:
    """Helper to create a TradeEvent with sensible defaults."""
    defaults = dict(
        trade_id=trade_id,
        bot_id=bot_id,
        pair="BTCUSDT",
        entry_price=50000.0,
        exit_price=50000.0 + pnl,
        pnl=pnl,
        pnl_pct=pnl / 50000.0 * 100,
        entry_signal="EMA cross",
        exit_reason="TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
        market_regime="trending_up",
        process_quality_score=85,
        root_causes=["normal_win"] if pnl > 0 else ["normal_loss"],
    )
    defaults.update(kwargs)
    return _factory_trade(**defaults)


def _make_missed(bot_id: str, pair: str, blocked_by: str, outcome_24h: float) -> MissedOpportunityEvent:
    return _factory_missed(
        bot_id=bot_id,
        pair=pair,
        signal="RSI divergence",
        blocked_by=blocked_by,
        hypothetical_entry=50000.0,
        outcome_24h=outcome_24h,
        confidence=0.7,
        assumption_tags=["mid_fill"],
    )


class TestDailyMetricsBuilder:
    def test_build_summary_basic(self):
        trades = [
            _make_trade("t1", "bot1", 100.0),
            _make_trade("t2", "bot1", -50.0),
            _make_trade("t3", "bot1", 200.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        summary = builder.build_summary(trades)

        assert summary.total_trades == 3
        assert summary.win_count == 2
        assert summary.loss_count == 1
        assert summary.gross_pnl == 250.0

    def test_top_winners(self):
        trades = [_make_trade(f"t{i}", "bot1", (i + 1) * 10.0) for i in range(10)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        winners = builder.top_winners(trades, n=5)

        assert len(winners) == 5
        assert winners[0].pnl >= winners[1].pnl  # sorted descending

    def test_top_losers(self):
        trades = [_make_trade(f"t{i}", "bot1", -(i + 1) * 10.0) for i in range(10)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        losers = builder.top_losers(trades, n=5)

        assert len(losers) == 5
        assert losers[0].pnl <= losers[1].pnl  # sorted ascending (most negative first)

    def test_process_failures(self):
        trades = [
            _make_trade("t1", "bot1", -80.0, process_quality_score=45, root_causes=["regime_mismatch"]),
            _make_trade("t2", "bot1", 100.0, process_quality_score=90),
            _make_trade("t3", "bot1", -20.0, process_quality_score=55, root_causes=["weak_signal"]),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        failures = builder.process_failures(trades, threshold=60)

        assert len(failures) == 2
        assert all(f.process_quality_score < 60 for f in failures)

    def test_notable_missed(self):
        missed = [
            _make_missed("bot1", "BTCUSDT", "vol_filter", 500.0),
            _make_missed("bot1", "ETHUSDT", "spread_filter", 50.0),
            _make_missed("bot1", "SOLUSDT", "vol_filter", 800.0),
        ]
        trades = [_make_trade("t1", "bot1", 100.0)]  # avg_win = 100
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        notable = builder.notable_missed(missed, trades)

        # Only those with outcome > 2x avg_win (200)
        assert len(notable) == 2
        assert all(n.outcome_24h > 200 for n in notable)

    def test_regime_analysis(self):
        trades = [
            _make_trade("t1", "bot1", 100.0, market_regime="trending_up"),
            _make_trade("t2", "bot1", -50.0, market_regime="ranging"),
            _make_trade("t3", "bot1", 200.0, market_regime="trending_up"),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        ra = builder.regime_analysis(trades)

        assert ra.regime_pnl["trending_up"] == 300.0
        assert ra.regime_pnl["ranging"] == -50.0
        assert ra.regime_trade_count["trending_up"] == 2

    def test_filter_analysis(self):
        missed = [
            _make_missed("bot1", "BTCUSDT", "vol_filter", 500.0),
            _make_missed("bot1", "ETHUSDT", "vol_filter", -100.0),
            _make_missed("bot1", "SOLUSDT", "spread_filter", 200.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        fa = builder.filter_analysis(missed)

        assert fa.filter_block_counts["vol_filter"] == 2
        assert fa.filter_block_counts["spread_filter"] == 1

    def test_root_cause_summary(self):
        trades = [
            _make_trade("t1", "bot1", 100.0, root_causes=["normal_win"]),
            _make_trade("t2", "bot1", -50.0, root_causes=["regime_mismatch", "weak_signal"]),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        rcs = builder.root_cause_summary(trades)

        assert rcs.distribution["normal_win"] == 1
        assert rcs.distribution["regime_mismatch"] == 1
        assert rcs.distribution["weak_signal"] == 1

    def test_write_curated_files(self, tmp_path: Path):
        trades = [
            _make_trade("t1", "bot1", 100.0),
            _make_trade("t2", "bot1", -50.0),
        ]
        missed = [_make_missed("bot1", "BTCUSDT", "vol_filter", 500.0)]

        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        assert (output_dir / "summary.json").exists()
        assert (output_dir / "winners.json").exists()
        assert (output_dir / "losers.json").exists()
        assert (output_dir / "process_failures.json").exists()
        assert (output_dir / "notable_missed.json").exists()
        assert (output_dir / "regime_analysis.json").exists()
        assert (output_dir / "filter_analysis.json").exists()
        assert (output_dir / "root_cause_summary.json").exists()

        # Verify JSON is valid and round-trips
        summary_data = json.loads((output_dir / "summary.json").read_text())
        assert summary_data["bot_id"] == "bot1"


class TestDailyMetricsPipelineEnriched:
    """Tests for hourly_performance and slippage_stats integration."""

    def test_hourly_performance_method(self):
        from datetime import datetime, timezone

        trades = [
            _make_trade("t1", "bot1", 100.0, entry_time=datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)),
            _make_trade("t2", "bot1", -50.0, entry_time=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.hourly_performance(trades)

        assert result["bot_id"] == "bot1"
        assert result["date"] == "2026-03-01"
        assert len(result["buckets"]) == 2
        hours = {b["hour"] for b in result["buckets"]}
        assert hours == {9, 15}

    def test_slippage_stats_method(self):
        trades = [
            _make_trade("t1", "bot1", 100.0, spread_at_entry=5.0),
            _make_trade("t2", "bot1", -50.0, spread_at_entry=3.0, pair="ETHUSDT"),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.slippage_stats(trades)

        assert result["bot_id"] == "bot1"
        assert result["date"] == "2026-03-01"
        assert "BTCUSDT" in result["by_symbol"]
        assert "ETHUSDT" in result["by_symbol"]

    def test_writes_hourly_performance(self, tmp_path):
        from datetime import datetime, timezone

        trades = [
            _make_trade("t1", "bot1", 100.0, entry_time=datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)),
            _make_trade("t2", "bot1", -50.0, entry_time=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        hourly_path = output_dir / "hourly_performance.json"
        assert hourly_path.exists()
        data = json.loads(hourly_path.read_text())
        assert len(data["buckets"]) == 2
        assert data["bot_id"] == "bot1"

    def test_writes_slippage_stats(self, tmp_path):
        trades = [
            _make_trade("t1", "bot1", 100.0, spread_at_entry=5.0),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        slippage_path = output_dir / "slippage_stats.json"
        assert slippage_path.exists()
        data = json.loads(slippage_path.read_text())
        assert data["bot_id"] == "bot1"
        assert "BTCUSDT" in data["by_symbol"]
        assert data["by_symbol"]["BTCUSDT"]["mean_bps"] == 5.0

    def test_write_curated_includes_all_enriched_files(self, tmp_path):
        """Verify write_curated produces both new files alongside existing ones."""
        from datetime import datetime, timezone

        trades = [
            _make_trade("t1", "bot1", 100.0,
                        entry_time=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
                        spread_at_entry=4.0),
            _make_trade("t2", "bot1", -50.0,
                        entry_time=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
                        spread_at_entry=6.0),
        ]
        missed = [_make_missed("bot1", "BTCUSDT", "vol_filter", 500.0)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        # Original files still present
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "winners.json").exists()
        assert (output_dir / "regime_analysis.json").exists()

        # New enriched files present
        assert (output_dir / "hourly_performance.json").exists()
        assert (output_dir / "slippage_stats.json").exists()

    def test_hourly_performance_empty_trades(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.hourly_performance([])
        assert result["buckets"] == []

    def test_slippage_stats_no_spread(self):
        """Trades with spread_at_entry=0 should produce empty distributions."""
        trades = [_make_trade("t1", "bot1", 100.0, spread_at_entry=0.0)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.slippage_stats(trades)
        assert result["by_symbol"] == {}
        assert result["by_hour"] == {}


class TestPerStrategyPipeline:
    """Tests for per-strategy summary mapping from DailySnapshot data."""

    def test_momentum_nq_01_format(self):
        """Multi-strategy bot with multiple keys in per_strategy_summary."""
        snapshot = {
            "per_strategy_summary": {
                "Helix": {
                    "trades": 5, "win_count": 3, "loss_count": 2,
                    "gross_pnl": 380.0, "net_pnl": 340.0,
                    "win_rate": 0.6, "avg_win": 140.0, "avg_loss": -20.0,
                    "best_trade_pnl": 200.0, "worst_trade_pnl": -30.0,
                    "avg_entry_slippage_bps": 1.2,
                },
                "NQDTC": {
                    "trades": 4, "win_count": 2, "loss_count": 2,
                    "gross_pnl": 150.0, "net_pnl": 110.0,
                    "win_rate": 0.5, "avg_win": 100.0, "avg_loss": -25.0,
                    "best_trade_pnl": 120.0, "worst_trade_pnl": -40.0,
                    "avg_entry_slippage_bps": 0.8,
                },
            }
        }
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="momentum_nq_01")
        result = builder.build_per_strategy_from_snapshot(snapshot)

        assert len(result) == 2
        assert "Helix" in result
        assert "NQDTC" in result

        helix = result["Helix"]
        assert helix.strategy_id == "Helix"
        assert helix.trades == 5
        assert helix.win_count == 3
        assert helix.net_pnl == 340.0
        assert helix.win_rate == 0.6
        assert helix.avg_entry_slippage_bps == 1.2

    def test_k_stock_trader_format(self):
        """Mono-strategy bot with single key in per_strategy_summary."""
        snapshot = {
            "per_strategy_summary": {
                "KMP": {
                    "trades": 8, "win_count": 5, "loss_count": 3,
                    "gross_pnl": 500.0, "net_pnl": 480.0,
                    "win_rate": 62.5, "avg_win": 120.0, "avg_loss": -40.0,
                    "best_trade_pnl": 200.0, "worst_trade_pnl": -60.0,
                    "avg_entry_slippage_bps": 2.0,
                }
            }
        }
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="k_stock_trader")
        result = builder.build_per_strategy_from_snapshot(snapshot)

        assert len(result) == 1
        kmp = result["KMP"]
        assert kmp.strategy_id == "KMP"
        assert kmp.trades == 8
        # win_rate 62.5 should be normalized to 0.625
        assert kmp.win_rate == 0.625

    def test_empty_snapshot(self):
        """Empty per_strategy_summary returns empty dict."""
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        assert builder.build_per_strategy_from_snapshot({}) == {}
        assert builder.build_per_strategy_from_snapshot({"per_strategy_summary": {}}) == {}

    def test_missing_fields_use_defaults(self):
        """Missing fields in strategy data should use defaults."""
        snapshot = {
            "per_strategy_summary": {
                "Sparse": {"trades": 3, "net_pnl": 100.0}
            }
        }
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.build_per_strategy_from_snapshot(snapshot)

        sparse = result["Sparse"]
        assert sparse.trades == 3
        assert sparse.net_pnl == 100.0
        assert sparse.win_count == 0
        assert sparse.loss_count == 0
        assert sparse.gross_pnl == 0.0
        assert sparse.avg_win == 0.0
        assert sparse.avg_loss == 0.0
        assert sparse.avg_entry_slippage_bps is None

    def test_non_dict_strategy_data_skipped(self):
        """Non-dict values in per_strategy_summary are skipped."""
        snapshot = {
            "per_strategy_summary": {
                "Good": {"trades": 1, "net_pnl": 50.0},
                "Bad": "not a dict",
                "AlsoBad": 42,
            }
        }
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.build_per_strategy_from_snapshot(snapshot)
        assert len(result) == 1
        assert "Good" in result

    def test_win_rate_normalization_fraction(self):
        """win_rate already in 0-1 range is unchanged."""
        snapshot = {"per_strategy_summary": {"S1": {"win_rate": 0.75}}}
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        assert builder.build_per_strategy_from_snapshot(snapshot)["S1"].win_rate == 0.75

    def test_win_rate_normalization_percentage(self):
        """win_rate in 0-100 range is normalized to 0-1."""
        snapshot = {"per_strategy_summary": {"S1": {"win_rate": 60.0}}}
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        assert builder.build_per_strategy_from_snapshot(snapshot)["S1"].win_rate == 0.6

    def test_win_rate_normalization_zero(self):
        """win_rate of 0 stays 0."""
        snapshot = {"per_strategy_summary": {"S1": {"win_rate": 0}}}
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        assert builder.build_per_strategy_from_snapshot(snapshot)["S1"].win_rate == 0.0

    def test_write_curated_with_snapshot(self, tmp_path: Path):
        """write_curated with daily_snapshot populates per_strategy_summary in summary.json."""
        trades = [
            _make_trade("t1", "bot1", 100.0),
            _make_trade("t2", "bot1", -50.0),
        ]
        missed: list = []
        snapshot = {
            "per_strategy_summary": {
                "Alpha": {
                    "trades": 2, "win_count": 1, "loss_count": 1,
                    "gross_pnl": 50.0, "net_pnl": 50.0,
                    "win_rate": 0.5, "avg_win": 100.0, "avg_loss": -50.0,
                    "best_trade_pnl": 100.0, "worst_trade_pnl": -50.0,
                }
            }
        }

        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path, daily_snapshot=snapshot)

        summary_data = json.loads((output_dir / "summary.json").read_text())
        pss = summary_data["per_strategy_summary"]
        assert "Alpha" in pss
        assert pss["Alpha"]["strategy_id"] == "Alpha"
        assert pss["Alpha"]["trades"] == 2
        assert pss["Alpha"]["win_rate"] == 0.5

    def test_write_curated_without_snapshot(self, tmp_path: Path):
        """write_curated without daily_snapshot leaves per_strategy_summary empty."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []

        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        summary_data = json.loads((output_dir / "summary.json").read_text())
        assert summary_data["per_strategy_summary"] == {}

    def test_backward_compat_existing_callers(self, tmp_path: Path):
        """Existing callers without daily_snapshot param still work."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed = [_make_missed("bot1", "BTCUSDT", "vol_filter", 500.0)]

        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        # Call without daily_snapshot — should work identically to before
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "winners.json").exists()


class TestExcursionStatsPipeline:
    """Tests for MFE/MAE excursion stats pipeline."""

    def test_schema_round_trip(self):
        """TradeEvent with MFE/MAE fields populated validates and round-trips."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        t = TradeEvent(
            trade_id="exc1", bot_id="swing_multi_01", pair="BTCUSDT",
            side="LONG", entry_time=now, exit_time=now,
            entry_price=50000, exit_price=50500, position_size=1.0,
            pnl=500, pnl_pct=1.0,
            mfe_price=50800, mae_price=49700,
            mfe_pct=1.6, mae_pct=0.6,
            mfe_r=2.1, mae_r=0.7,
            exit_efficiency=62.5,
        )
        assert t.mfe_pct == 1.6
        assert t.mae_r == 0.7
        assert t.exit_efficiency == 62.5

        # Round-trip via dict
        d = t.model_dump()
        t2 = TradeEvent(**d)
        assert t2.mfe_price == 50800
        assert t2.exit_efficiency == 62.5

    def test_schema_backward_compat(self):
        """TradeEvent without MFE/MAE fields defaults all to None."""
        t = _make_trade("t1", "bot1", 100.0)
        assert t.mfe_price is None
        assert t.mae_price is None
        assert t.mfe_pct is None
        assert t.mae_pct is None
        assert t.mfe_r is None
        assert t.mae_r is None
        assert t.exit_efficiency is None

    def test_build_excursion_stats_mixed_coverage(self):
        """Mixed trades — some with MFE/MAE, some without."""
        trades = [
            _make_trade("t1", "bot1", 100.0, mfe_pct=2.0, mae_pct=0.5, mfe_r=2.5, mae_r=0.6, exit_efficiency=55.0),
            _make_trade("t2", "bot1", -50.0, mfe_pct=0.8, mae_pct=1.2, mfe_r=1.0, mae_r=1.5, exit_efficiency=30.0),
            _make_trade("t3", "bot1", 200.0),  # no MFE/MAE data
            _make_trade("t4", "bot1", 150.0, mfe_pct=3.0, mae_pct=0.3, mfe_r=3.5, mae_r=0.4, exit_efficiency=70.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.build_excursion_stats(trades)

        assert result["coverage"] == 3
        assert result["total_trades"] == 4
        assert result["coverage_pct"] == 75.0
        assert result["stats"]["mfe_pct"]["mean"] == pytest.approx((2.0 + 0.8 + 3.0) / 3)
        assert result["stats"]["mae_pct"]["min"] == 0.3
        assert result["stats"]["exit_efficiency"]["max"] == 70.0
        assert result["winners"]["count"] == 2
        assert result["losers"]["count"] == 1
        assert result["winners"]["avg_mfe_pct"] == pytest.approx((2.0 + 3.0) / 2)
        assert result["losers"]["avg_mae_pct"] == 1.2

    def test_build_excursion_stats_no_coverage(self):
        """No trades with MFE/MAE data returns zero coverage."""
        trades = [_make_trade("t1", "bot1", 100.0), _make_trade("t2", "bot1", -50.0)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.build_excursion_stats(trades)

        assert result["coverage"] == 0
        assert result["total_trades"] == 2
        assert result["stats"] == {}

    def test_build_excursion_stats_empty_trades(self):
        """Empty trade list returns zero coverage."""
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.build_excursion_stats([])

        assert result["coverage"] == 0
        assert result["total_trades"] == 0
        assert result["stats"] == {}

    def test_build_excursion_stats_partial_fields(self):
        """Trade with mfe_pct/mae_pct but no mfe_r/exit_efficiency."""
        trades = [
            _make_trade("t1", "bot1", 100.0, mfe_pct=1.5, mae_pct=0.4),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.build_excursion_stats(trades)

        assert result["coverage"] == 1
        assert result["stats"]["mfe_pct"]["mean"] == 1.5
        assert result["stats"]["mae_pct"]["mean"] == 0.4
        assert result["stats"]["mfe_r"] == {}  # no mfe_r data
        assert result["stats"]["mae_r"] == {}
        assert result["stats"]["exit_efficiency"] == {}
        assert result["winners"]["avg_exit_efficiency"] == 0

    def test_write_curated_writes_excursion_stats(self, tmp_path: Path):
        """write_curated writes excursion_stats.json when coverage > 0."""
        trades = [
            _make_trade("t1", "bot1", 100.0, mfe_pct=2.0, mae_pct=0.5, exit_efficiency=60.0),
            _make_trade("t2", "bot1", -50.0, mfe_pct=0.8, mae_pct=1.0, exit_efficiency=25.0),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        excursion_path = output_dir / "excursion_stats.json"
        assert excursion_path.exists()
        data = json.loads(excursion_path.read_text())
        assert data["coverage"] == 2
        assert data["total_trades"] == 2
        assert "mfe_pct" in data["stats"]

    def test_write_curated_no_excursion_file_without_data(self, tmp_path: Path):
        """write_curated does NOT write excursion_stats.json when no MFE/MAE data."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        assert not (output_dir / "excursion_stats.json").exists()


class TestOverlayStatePipeline:
    """Tests for overlay state summary pipeline from DailySnapshot."""

    def test_write_curated_with_overlay_state(self, tmp_path: Path):
        """write_curated writes overlay_state_summary.json when present in snapshot."""
        trades = [_make_trade("t1", "swing_multi_01", 100.0)]
        missed: list = []
        snapshot = {
            "per_strategy_summary": {},
            "overlay_state_summary": {
                "date": "2026-03-06",
                "symbols": {
                    "QQQ": {
                        "ema_fast": 10, "ema_slow": 21,
                        "signal": "bullish", "shares": 50,
                        "entry_price": 480.25, "current_price": 485.10,
                        "unrealized_pnl": 242.50, "unrealized_pnl_pct": 1.01,
                    },
                    "GLD": {
                        "ema_fast": 13, "ema_slow": 21,
                        "signal": "bearish", "shares": 0,
                        "entry_price": None, "current_price": 198.40,
                        "unrealized_pnl": 0, "unrealized_pnl_pct": 0,
                    },
                },
                "total_unrealized_pnl": 242.50,
                "capital_deployed_pct": 42.0,
                "transitions_today": 1,
            },
        }

        builder = DailyMetricsBuilder(date="2026-03-06", bot_id="swing_multi_01")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path, daily_snapshot=snapshot)

        overlay_path = output_dir / "overlay_state_summary.json"
        assert overlay_path.exists()
        data = json.loads(overlay_path.read_text())
        assert data["total_unrealized_pnl"] == 242.50
        assert data["capital_deployed_pct"] == 42.0
        assert "QQQ" in data["symbols"]
        assert data["symbols"]["QQQ"]["signal"] == "bullish"

    def test_write_curated_without_overlay_state(self, tmp_path: Path):
        """write_curated without overlay data does not write overlay file."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []
        snapshot = {"per_strategy_summary": {}}

        builder = DailyMetricsBuilder(date="2026-03-06", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path, daily_snapshot=snapshot)

        assert not (output_dir / "overlay_state_summary.json").exists()

    def test_write_curated_no_snapshot(self, tmp_path: Path):
        """write_curated without daily_snapshot param does not write overlay file."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []

        builder = DailyMetricsBuilder(date="2026-03-06", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        assert not (output_dir / "overlay_state_summary.json").exists()

    def test_prompt_assembler_includes_overlay_file(self):
        """Verify overlay_state_summary.json is in _CURATED_FILES list."""
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "overlay_state_summary.json" in _CURATED_FILES

    def test_prompt_assembler_loads_overlay_when_present(self, tmp_path: Path):
        """DailyPromptAssembler loads overlay data when the file exists."""
        from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler

        # Set up curated dir with overlay file
        bot_dir = tmp_path / "curated" / "2026-03-06" / "swing_multi_01"
        bot_dir.mkdir(parents=True)
        overlay_data = {"total_unrealized_pnl": 242.50, "capital_deployed_pct": 42.0}
        (bot_dir / "overlay_state_summary.json").write_text(json.dumps(overlay_data))

        # Set up minimal memory dir
        memory_dir = tmp_path / "memory"
        policies_dir = memory_dir / "policies" / "v1"
        policies_dir.mkdir(parents=True)
        for f in ["soul.md", "trading_rules.md", "agent.md"]:
            (policies_dir / f).write_text("# test")
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        assembler = DailyPromptAssembler(
            date="2026-03-06",
            bots=["swing_multi_01"],
            curated_dir=tmp_path / "curated",
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert "overlay_state_summary" in pkg.data["swing_multi_01"]
        assert pkg.data["swing_multi_01"]["overlay_state_summary"]["total_unrealized_pnl"] == 242.50


class TestCoordinatorImpactRawEvents:
    def test_coordinator_impact_includes_raw_events(self):
        """Verify coordinator_impact() output includes 'events' key with original event dicts."""
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="swing_multi_01")
        events = [
            {"action": "tighten_stop_be", "rule": "rule_1", "symbol": "BTCUSDT"},
            {"action": "size_boost", "rule": "rule_2", "symbol": "ETHUSDT"},
        ]
        result = builder.coordinator_impact(events)
        assert "events" in result
        assert result["events"] == events
        assert result["total_events"] == 2
        assert result["by_action"]["tighten_stop_be"] == 1
        assert result["by_action"]["size_boost"] == 1

    def test_coordinator_impact_empty_includes_events_key(self):
        """Empty coordination events should still have an 'events' key."""
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="swing_multi_01")
        result = builder.coordinator_impact([])
        assert "events" in result
        assert result["events"] == []
        assert result["total_events"] == 0


class TestExperimentBreakdownPipeline:
    """Tests for experiment_breakdown (1.4) pipeline from DailySnapshot."""

    def test_snapshot_schema_accepts_experiment_breakdown(self):
        """DailySnapshot schema accepts experiment_breakdown field."""
        from trading_assistant.schemas.events import DailySnapshot
        snap = DailySnapshot(
            date="2026-03-01", bot_id="swing_multi_01",
            experiment_breakdown={"v1": {"trades": 10, "pnl": 500}},
        )
        assert snap.experiment_breakdown == {"v1": {"trades": 10, "pnl": 500}}

    def test_snapshot_schema_defaults_to_none(self):
        """DailySnapshot defaults experiment_breakdown to None."""
        from trading_assistant.schemas.events import DailySnapshot
        snap = DailySnapshot(date="2026-03-01", bot_id="swing_multi_01")
        assert snap.experiment_breakdown is None

    def test_write_curated_writes_experiment_breakdown(self, tmp_path: Path):
        """write_curated writes experiment_data.json when snapshot has data."""
        trades = [_make_trade("t1", "swing_multi_01", 100.0)]
        missed: list = []
        snapshot = {
            "per_strategy_summary": {},
            "experiment_breakdown": {
                "variant_a": {"trades": 5, "pnl": 200, "win_rate": 0.6},
                "variant_b": {"trades": 5, "pnl": -50, "win_rate": 0.4},
            },
        }
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="swing_multi_01")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path, daily_snapshot=snapshot)

        exp_path = output_dir / "experiment_data.json"
        assert exp_path.exists()
        data = json.loads(exp_path.read_text())
        assert "variant_a" in data
        assert data["variant_a"]["pnl"] == 200

    def test_write_curated_skips_without_experiment_breakdown(self, tmp_path: Path):
        """write_curated skips file when snapshot lacks experiment_breakdown."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []
        snapshot = {"per_strategy_summary": {}}
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path, daily_snapshot=snapshot)
        assert not (output_dir / "experiment_data.json").exists()

    def test_write_curated_skips_without_snapshot(self, tmp_path: Path):
        """write_curated without snapshot does not write experiment_breakdown."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)
        assert not (output_dir / "experiment_data.json").exists()

    def test_prompt_assembler_includes_experiment_breakdown(self):
        """Verify experiment_data.json is in _CURATED_FILES list."""
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "experiment_data.json" in _CURATED_FILES

    def test_prompt_assembler_loads_experiment_breakdown(self, tmp_path: Path):
        """DailyPromptAssembler loads experiment_data when file exists."""
        from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler

        bot_dir = tmp_path / "curated" / "2026-03-01" / "swing_multi_01"
        bot_dir.mkdir(parents=True)
        exp_data = {"variant_a": {"trades": 5, "pnl": 200}}
        (bot_dir / "experiment_data.json").write_text(json.dumps(exp_data))

        memory_dir = tmp_path / "memory"
        policies_dir = memory_dir / "policies" / "v1"
        policies_dir.mkdir(parents=True)
        for f in ["soul.md", "trading_rules.md", "agent.md"]:
            (policies_dir / f).write_text("# test")
        (memory_dir / "findings").mkdir(parents=True)

        assembler = DailyPromptAssembler(
            date="2026-03-01", bots=["swing_multi_01"],
            curated_dir=tmp_path / "curated", memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert "experiment_data" in pkg.data["swing_multi_01"]
        assert pkg.data["swing_multi_01"]["experiment_data"]["variant_a"]["pnl"] == 200


class TestSignalHealthPipeline:
    """Tests for signal_evolution (1.5) pipeline."""

    def test_trade_event_accepts_signal_evolution(self):
        """TradeEvent accepts signal_evolution field."""
        t = _make_trade("t1", "momentum_nq_01", 100.0, signal_evolution=[
            {"bar": 1, "rsi": 0.7, "macd": 0.3},
            {"bar": 2, "rsi": 0.8, "macd": 0.4},
        ])
        assert len(t.signal_evolution) == 2

    def test_trade_event_signal_evolution_defaults_none(self):
        """TradeEvent defaults signal_evolution to None."""
        t = _make_trade("t1", "bot1", 100.0)
        assert t.signal_evolution is None

    def test_signal_health_analyzer_basic(self):
        """SignalHealthAnalyzer computes health metrics from 3 trades with data."""
        from trading_assistant.skills.signal_health_analyzer import SignalHealthAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0, signal_evolution=[
                {"bar": 1, "rsi": 0.6, "macd": 0.2},
                {"bar": 2, "rsi": 0.7, "macd": 0.3},
                {"bar": 3, "rsi": 0.8, "macd": 0.4},
            ]),
            _make_trade("t2", "bot1", -50.0, signal_evolution=[
                {"bar": 1, "rsi": 0.5, "macd": 0.1},
                {"bar": 2, "rsi": 0.4, "macd": 0.0},
            ]),
            _make_trade("t3", "bot1", 200.0, signal_evolution=[
                {"bar": 1, "rsi": 0.9, "macd": 0.5},
                {"bar": 2, "rsi": 0.85, "macd": 0.6},
            ]),
        ]
        analyzer = SignalHealthAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)

        assert report.total_trades_with_data == 3
        assert report.coverage_pct == 100.0
        assert len(report.components) == 2  # rsi, macd
        comp_names = {c.component_name for c in report.components}
        assert comp_names == {"rsi", "macd"}

    def test_signal_health_analyzer_empty_when_no_data(self):
        """SignalHealthAnalyzer returns empty report when no trades have data."""
        from trading_assistant.skills.signal_health_analyzer import SignalHealthAnalyzer
        trades = [_make_trade("t1", "bot1", 100.0)]
        analyzer = SignalHealthAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.total_trades_with_data == 0
        assert report.components == []

    def test_stability_for_constant_values(self):
        """Stability should be 1.0 for constant signal values."""
        from trading_assistant.skills.signal_health_analyzer import SignalHealthAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0, signal_evolution=[
                {"bar": 1, "rsi": 0.5},
                {"bar": 2, "rsi": 0.5},
                {"bar": 3, "rsi": 0.5},
            ]),
        ]
        analyzer = SignalHealthAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        # Constant values → stdev=0, range=0 → stability=1.0
        assert report.components[0].stability == 1.0

    def test_single_bar_trades_skipped(self):
        """Trades with only 1 bar in signal_evolution are skipped (need ≥2)."""
        from trading_assistant.skills.signal_health_analyzer import SignalHealthAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0, signal_evolution=[
                {"bar": 1, "rsi": 0.5},
            ]),
        ]
        analyzer = SignalHealthAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.total_trades_with_data == 0
        assert report.components == []

    def test_write_curated_writes_signal_health(self, tmp_path: Path):
        """write_curated writes signal_health.json when trades have signal_evolution."""
        trades = [
            _make_trade("t1", "bot1", 100.0, signal_evolution=[
                {"bar": 1, "rsi": 0.6}, {"bar": 2, "rsi": 0.7},
            ]),
            _make_trade("t2", "bot1", -50.0, signal_evolution=[
                {"bar": 1, "rsi": 0.5}, {"bar": 2, "rsi": 0.4},
            ]),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        sh_path = output_dir / "signal_health.json"
        assert sh_path.exists()
        data = json.loads(sh_path.read_text())
        assert data["total_trades_with_data"] == 2
        assert data["bot_id"] == "bot1"

    def test_write_curated_skips_signal_health_without_data(self, tmp_path: Path):
        """write_curated skips signal_health.json when no trades have signal_evolution."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)
        assert not (output_dir / "signal_health.json").exists()

    def test_prompt_assembler_includes_signal_health(self):
        """Verify signal_health.json is in _CURATED_FILES list."""
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "signal_health.json" in _CURATED_FILES

    def test_prompt_assembler_loads_signal_health(self, tmp_path: Path):
        """DailyPromptAssembler loads signal_health data when file exists."""
        from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler

        bot_dir = tmp_path / "curated" / "2026-03-01" / "momentum_nq_01"
        bot_dir.mkdir(parents=True)
        sh_data = {"bot_id": "momentum_nq_01", "components": [], "total_trades_with_data": 5}
        (bot_dir / "signal_health.json").write_text(json.dumps(sh_data))

        memory_dir = tmp_path / "memory"
        policies_dir = memory_dir / "policies" / "v1"
        policies_dir.mkdir(parents=True)
        for f in ["soul.md", "trading_rules.md", "agent.md"]:
            (policies_dir / f).write_text("# test")
        (memory_dir / "findings").mkdir(parents=True)

        assembler = DailyPromptAssembler(
            date="2026-03-01", bots=["momentum_nq_01"],
            curated_dir=tmp_path / "curated", memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert "signal_health" in pkg.data["momentum_nq_01"]

    def test_strategy_engine_detects_degraded_components(self):
        """detect_component_signal_decay flags components with low stability/correlation."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        sh_data = {
            "components": [
                {"component_name": "rsi", "trade_count": 10, "stability": 0.2, "win_correlation": 0.01},
                {"component_name": "macd", "trade_count": 10, "stability": 0.8, "win_correlation": 0.5},
            ],
        }
        suggestions = engine.detect_component_signal_decay("bot1", sh_data)
        assert len(suggestions) == 1
        assert "rsi" in suggestions[0].description

    def test_strategy_engine_healthy_components(self):
        """detect_component_signal_decay returns empty for healthy components."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        sh_data = {
            "components": [
                {"component_name": "rsi", "trade_count": 10, "stability": 0.8, "win_correlation": 0.5},
            ],
        }
        suggestions = engine.detect_component_signal_decay("bot1", sh_data)
        assert suggestions == []

    def test_strategy_engine_empty_components(self):
        """detect_component_signal_decay returns empty for empty components list."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        suggestions = engine.detect_component_signal_decay("bot1", {"components": []})
        assert suggestions == []

    def test_strategy_engine_skips_low_trade_count(self):
        """detect_component_signal_decay skips components with <5 trades."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        sh_data = {
            "components": [
                {"component_name": "rsi", "trade_count": 3, "stability": 0.1, "win_correlation": 0.01},
            ],
        }
        suggestions = engine.detect_component_signal_decay("bot1", sh_data)
        assert suggestions == []

    def test_build_report_accepts_signal_health(self):
        """build_report() accepts and uses signal_health parameter."""
        from trading_assistant.analysis.strategy_engine import StrategyEngine
        from trading_assistant.schemas.weekly_metrics import BotWeeklySummary
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        summaries = {
            "bot1": BotWeeklySummary(bot_id="bot1", week_start="2026-02-24", week_end="2026-03-02"),
        }
        sh = {
            "bot1": {
                "components": [
                    {"component_name": "rsi", "trade_count": 10, "stability": 0.1, "win_correlation": 0.01},
                ],
            },
        }
        report = engine.build_report(summaries, signal_health=sh)
        # Should include the signal decay suggestion
        assert any("Signal component decay" in s.title for s in report.suggestions)


class TestFillQualityPipeline:
    """Tests for fill quality (2.6) pipeline."""

    def test_trade_event_accepts_fill_details(self):
        """TradeEvent accepts entry/exit fill detail fields."""
        t = _make_trade("t1", "momentum_nq_01", 100.0,
            entry_fill_details={"slippage_bps": 0.5, "fill_latency_ms": 12, "fill_type": "limit"},
            exit_fill_details={"slippage_bps": 1.2, "fill_latency_ms": 8, "fill_type": "market"},
        )
        assert t.entry_fill_details["slippage_bps"] == 0.5
        assert t.exit_fill_details["fill_type"] == "market"

    def test_trade_event_fill_details_default_none(self):
        """TradeEvent defaults fill detail fields to None."""
        t = _make_trade("t1", "bot1", 100.0)
        assert t.entry_fill_details is None
        assert t.exit_fill_details is None

    def test_fill_quality_analyzer_basic(self):
        """FillQualityAnalyzer computes metrics from trades with fill data."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0,
                entry_fill_details={"slippage_bps": 0.5, "fill_latency_ms": 10, "fill_type": "limit"},
                exit_fill_details={"slippage_bps": 1.0, "fill_latency_ms": 8, "fill_type": "market"},
            ),
            _make_trade("t2", "bot1", -50.0,
                entry_fill_details={"slippage_bps": 0.8, "fill_latency_ms": 15, "fill_type": "market"},
                exit_fill_details={"slippage_bps": 0.3, "fill_latency_ms": 5, "fill_type": "limit"},
            ),
        ]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.overall_entry.sample_count == 2
        assert report.overall_exit.sample_count == 2
        assert report.coverage_pct == 100.0

    def test_fill_quality_analyzer_empty_when_no_data(self):
        """FillQualityAnalyzer returns empty report when no trades have data."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        trades = [_make_trade("t1", "bot1", 100.0)]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.overall_entry.sample_count == 0
        assert report.overall_exit.sample_count == 0

    def test_entry_only_data(self):
        """FillQualityAnalyzer handles trades with only entry fill details."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0,
                entry_fill_details={"slippage_bps": 0.5, "fill_latency_ms": 10, "fill_type": "limit"},
            ),
        ]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.overall_entry.sample_count == 1
        assert report.overall_exit.sample_count == 0
        assert report.coverage_pct == 100.0

    def test_adverse_selection_detected(self):
        """Adverse selection flag when >60% adverse AND avg slippage > 1 bps."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        # All fills have positive slippage (adverse) > 1 bps
        trades = [
            _make_trade(f"t{i}", "bot1", 100.0,
                entry_fill_details={"slippage_bps": 2.0, "fill_latency_ms": 10, "fill_type": "market"},
                exit_fill_details={"slippage_bps": 1.5, "fill_latency_ms": 8, "fill_type": "market"},
            )
            for i in range(5)
        ]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.adverse_selection_detected is True

    def test_no_adverse_below_threshold(self):
        """No adverse selection when conditions not met."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        # Mixed slippages, some negative (favorable)
        trades = [
            _make_trade("t1", "bot1", 100.0,
                entry_fill_details={"slippage_bps": -0.5, "fill_latency_ms": 10, "fill_type": "limit"},
                exit_fill_details={"slippage_bps": -0.2, "fill_latency_ms": 8, "fill_type": "limit"},
            ),
            _make_trade("t2", "bot1", -50.0,
                entry_fill_details={"slippage_bps": 0.3, "fill_latency_ms": 15, "fill_type": "limit"},
                exit_fill_details={"slippage_bps": 0.1, "fill_latency_ms": 5, "fill_type": "limit"},
            ),
        ]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.adverse_selection_detected is False

    def test_per_symbol_breakdown(self):
        """FillQualityAnalyzer produces per-symbol breakdown."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0, pair="BTCUSDT",
                entry_fill_details={"slippage_bps": 0.5, "fill_latency_ms": 10, "fill_type": "limit"},
            ),
            _make_trade("t2", "bot1", -50.0, pair="ETHUSDT",
                entry_fill_details={"slippage_bps": 1.0, "fill_latency_ms": 20, "fill_type": "market"},
            ),
        ]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert "BTCUSDT" in report.by_symbol
        assert "ETHUSDT" in report.by_symbol
        assert report.by_symbol["BTCUSDT"].entry_stats.sample_count == 1
        assert report.by_symbol["ETHUSDT"].entry_stats.sample_count == 1

    def test_per_fill_type_breakdown(self):
        """FillQualityAnalyzer tracks fill type counts."""
        from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
        trades = [
            _make_trade("t1", "bot1", 100.0,
                entry_fill_details={"slippage_bps": 0.5, "fill_latency_ms": 10, "fill_type": "limit"},
            ),
            _make_trade("t2", "bot1", -50.0,
                entry_fill_details={"slippage_bps": 0.8, "fill_latency_ms": 12, "fill_type": "market"},
            ),
            _make_trade("t3", "bot1", 200.0,
                entry_fill_details={"slippage_bps": 0.3, "fill_latency_ms": 5, "fill_type": "limit"},
            ),
        ]
        analyzer = FillQualityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.compute(trades)
        assert report.overall_entry.by_fill_type == {"limit": 2, "market": 1}

    def test_write_curated_writes_fill_quality(self, tmp_path: Path):
        """write_curated writes fill_quality.json when trades have fill data."""
        trades = [
            _make_trade("t1", "bot1", 100.0,
                entry_fill_details={"slippage_bps": 0.5, "fill_latency_ms": 10, "fill_type": "limit"},
            ),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        fq_path = output_dir / "fill_quality.json"
        assert fq_path.exists()
        data = json.loads(fq_path.read_text())
        assert data["bot_id"] == "bot1"
        assert data["overall_entry"]["sample_count"] == 1

    def test_write_curated_skips_fill_quality_without_data(self, tmp_path: Path):
        """write_curated skips fill_quality.json when no trades have fill data."""
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)
        assert not (output_dir / "fill_quality.json").exists()

    def test_prompt_assembler_includes_fill_quality(self):
        """Verify fill_quality.json is in _CURATED_FILES list."""
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "fill_quality.json" in _CURATED_FILES


class TestParameterChangeLog:
    def test_builds_parameter_change_log(self):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        events = [
            {"payload": {"strategy_id": "ATRSS", "param_name": "stop_atr_mult", "old_value": 1.5, "new_value": 2.0, "reason": "monthly validation"}},
            {"payload": {"strategy_id": "ATRSS", "param_name": "tp_ratio", "old_value": 2.0, "new_value": 2.5, "reason": "Manual"}},
            {"payload": {"strategy_id": "AKC_HELIX", "param_name": "entry_threshold", "old_value": 0.7, "new_value": 0.8, "reason": "Backtest"}},
        ]
        result = builder.build_parameter_change_log(events)
        assert result["total_changes"] == 3
        assert result["bot_id"] == "bot1"
        assert len(result["changes"]) == 3
        assert result["by_strategy"]["ATRSS"]["count"] == 2
        assert "stop_atr_mult" in result["by_strategy"]["ATRSS"]["params"]

    def test_empty_events(self):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        result = builder.build_parameter_change_log([])
        assert result["total_changes"] == 0
        assert result["changes"] == []

    def test_prompt_assembler_includes_parameter_changes(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "parameter_changes.json" in _CURATED_FILES


class TestExecutionEvidenceArtifacts:
    def test_builds_order_lifecycle_summary(self):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        events = [
            {"payload": {"order_id": "o1", "pair": "BTCUSDT", "side": "LONG", "order_type": "LIMIT", "status": "submitted", "requested_qty": 1, "strategy_type": "alpha"}},
            {"payload": {"order_id": "o1", "pair": "BTCUSDT", "side": "LONG", "order_type": "LIMIT", "status": "fill", "requested_qty": 1, "filled_qty": 1, "requested_price": 50000.0, "fill_price": 50010.0, "latency_ms": 25, "strategy_type": "alpha"}},
            {"payload": {"order_id": "o2", "pair": "ETHUSDT", "side": "SHORT", "order_type": "STOP", "status": "REJECTED", "requested_qty": 2, "reject_reason": "risk_check", "latency_ms": 10, "strategy_type": "beta"}},
            {"payload": {"order_id": "o3", "pair": "SOLUSDT", "side": "LONG", "order_type": "LIMIT", "status": "CANCELLED", "requested_qty": 3, "strategy_type": "alpha"}},
            {"payload": json.dumps({"order_id": "o4", "pair": "ADAUSDT", "side": "LONG", "order_type": "LIMIT", "status": "partially_filled", "requested_qty": 4, "filled_qty": 2, "slippage_bps": 1.5, "latency_ms": 40, "strategy_type": "alpha"})},
        ]

        result = builder.build_order_lifecycle_summary(events)

        assert result["total_events"] == 5
        assert result["distinct_orders"] == 4
        assert result["status_counts"]["FILLED"] == 1
        assert result["status_counts"]["PARTIAL_FILL"] == 1
        assert result["reject_count"] == 1
        assert result["cancel_count"] == 1
        assert result["partial_fill_count"] == 1
        assert result["fill_count"] == 1
        assert result["avg_fill_slippage_bps"] == pytest.approx(1.75)
        assert result["max_fill_slippage_bps"] == pytest.approx(2.0)
        assert result["avg_latency_ms"] == pytest.approx(25.0)
        assert result["top_reject_reasons"] == [{"reason": "risk_check", "count": 1}]
        assert result["by_strategy"]["alpha"]["event_count"] == 4
        assert result["by_strategy"]["alpha"]["partial_fill_count"] == 1
        assert result["by_strategy"]["beta"]["reject_count"] == 1
        assert result["rejects"][0]["status"] == "REJECTED"

    def test_builds_order_lifecycle_summary_with_common_fallback_fields(self):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        events = [
            {
                "order_id": "o5",
                "pair": "BTCUSDT",
                "status": "partial fill",
                "requested_qty": 4,
                "qty": 2,
                "requested_price": 100.0,
                "price": 100.5,
                "strategy": "gamma",
                "event_metadata": {"exchange_timestamp": "2026-03-01T09:31:00+00:00"},
            },
        ]

        result = builder.build_order_lifecycle_summary(events)

        assert result["partial_fill_count"] == 1
        assert result["avg_fill_slippage_bps"] == pytest.approx(50.0)
        assert result["by_strategy"]["gamma"]["partial_fill_count"] == 1
        assert result["partial_fills"][0]["filled_qty"] == 2.0
        assert result["partial_fills"][0]["fill_price"] == 100.5
        assert result["partial_fills"][0]["timestamp"] == "2026-03-01T09:31:00+00:00"

    @pytest.mark.parametrize("event_type", ["fill", "inferred_fill"])
    def test_fill_events_without_status_count_as_filled(self, event_type):
        builder = DailyMetricsBuilder("2026-06-04", "k_stock_trader")
        events = [
            {
                "payload": {
                    "event_type": event_type,
                    "fill_id": "fill-1",
                    "kis_order_id": "kis-1",
                    "symbol": "005930",
                    "side": "BUY",
                    "qty": 10,
                    "price": 70000.0,
                    "assistant_strategy_id": "KALCB",
                    "exchange_timestamp": "2026-06-04T09:45:00+09:00",
                },
            },
        ]

        result = builder.build_order_lifecycle_summary(events)

        assert result["total_events"] == 1
        assert result["distinct_orders"] == 1
        assert result["status_counts"] == {"FILLED": 1}
        assert result["fill_count"] == 1
        assert result["by_strategy"]["KALCB"]["fill_count"] == 1
        assert result["by_strategy"]["KALCB"]["event_count"] == 1

    def test_builds_process_quality_summary(self):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        events = [
            {"payload": {"trade_id": "t1", "process_quality_score": 52, "classification": "poor", "root_causes": ["order_reject", "latency_spike"], "negative_factors": ["latency"], "evidence_refs": ["run-1"]}},
            {"payload": json.dumps({"trade_id": "t2", "score": 88, "classification": "good", "root_causes": ["good_execution"], "negative_factors": [], "evidence_refs": ["run-2"]})},
            {"payload": {"trade_id": "t3", "process_quality_score": 41, "classification": "poor", "root_causes": ["order_reject"], "negative_factors": ["reject"], "evidence_refs": ["run-3"]}},
        ]

        result = builder.build_process_quality_summary(events)

        assert result["total_scores"] == 3
        assert result["avg_score"] == pytest.approx(60.333)
        assert result["min_score"] == 41.0
        assert result["max_score"] == 88.0
        assert result["low_score_count"] == 2
        assert result["classification_counts"] == {"poor": 2, "good": 1}
        assert result["root_cause_counts"]["order_reject"] == 2
        assert [row["trade_id"] for row in result["worst_scores"][:2]] == ["t3", "t1"]

    def test_builds_process_quality_summary_with_scalar_fields(self):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        events = [
            {
                "payload": {
                    "trade_id": "t4",
                    "process_quality_score": 59,
                    "classification": " Poor ",
                    "root_causes": "order_reject",
                    "negative_factors": "latency",
                    "evidence_refs": "run-4",
                },
            },
        ]

        result = builder.build_process_quality_summary(events)

        assert result["classification_counts"] == {"poor": 1}
        assert result["root_cause_counts"] == {"order_reject": 1}
        assert result["worst_scores"][0]["negative_factors"] == ["latency"]
        assert result["worst_scores"][0]["evidence_refs"] == ["run-4"]

    def test_write_curated_writes_execution_artifacts(self, tmp_path: Path):
        builder = DailyMetricsBuilder("2026-03-01", "bot1")
        output_dir = builder.write_curated(
            trades=[],
            missed=[],
            base_dir=tmp_path,
            order_events=[{"payload": {"order_id": "o1", "status": "FILLED", "requested_price": 100.0, "fill_price": 100.2}}],
            process_quality_events=[{"payload": {"trade_id": "t1", "process_quality_score": 55, "classification": "poor"}}],
        )

        assert (output_dir / "order_lifecycle.json").exists()
        assert (output_dir / "process_quality.json").exists()

    def test_prompt_assembler_includes_order_lifecycle(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "order_lifecycle.json" in _CURATED_FILES

    def test_prompt_assembler_includes_process_quality(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "process_quality.json" in _CURATED_FILES


def test_portfolio_rule_alias_with_allowed_false_counts_as_block() -> None:
    from trading_assistant.skills.build_daily_metrics import build_portfolio_rules_summary

    result = build_portfolio_rules_summary([
        {
            "event_type": "portfolio_rule",
            "payload": {
                "rule_name": "max_exposure",
                "allowed": False,
                "details": {
                    "reason": "exposure_cap",
                    "blocked_symbol": "BTC",
                },
            },
        },
    ])

    assert result["total_evaluations"] == 1
    assert result["total_blocks"] == 1
    assert result["by_rule"]["max_exposure"]["blocks"] == 1
    assert result["blocked_symbols"] == ["BTC"]


def test_portfolio_rule_summary_unwraps_crypto_canonical_payload() -> None:
    from trading_assistant.skills.build_daily_metrics import build_portfolio_rules_summary

    result = build_portfolio_rules_summary([
        {
            "payload": json.dumps({
                "event_type": "portfolio_rule",
                "approved": False,
                "action": "block",
                "blocking_rule": "max_total_positions",
                "denial_reason": "max_total_positions reached",
                "symbol": "ETH",
            }),
        },
    ])

    assert result["total_evaluations"] == 1
    assert result["total_blocks"] == 1
    assert result["by_rule"]["max_total_positions"]["block_reasons"] == [
        "max_total_positions reached",
    ]
    assert result["blocked_symbols"] == ["ETH"]


def test_portfolio_rule_summary_counts_risk_decision_denial() -> None:
    from trading_assistant.skills.build_daily_metrics import build_portfolio_rules_summary

    result = build_portfolio_rules_summary([{
        "event_type": "risk_decision",
        "strategy_id": "strat1",
        "risk_decision_id": "risk-1",
        "decision": "deny",
        "deny_reason": "gross_exposure_limit",
    }])

    assert result["total_evaluations"] == 1
    assert result["total_blocks"] == 1
    assert result["records"][0]["risk_decision_id"] == "risk-1"
