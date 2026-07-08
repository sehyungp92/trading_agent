import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from strategies.stock.instrumentation.src.daily_snapshot import DailySnapshotBuilder, DailySnapshot


def _write_jsonl(filepath: Path, events: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class TestDailySnapshotBuilder:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "strategy_type": "helix",
            "data_dir": self.tmpdir,
        }
        self.date_str = "2026-03-01"

    def _write_trades(self, trades):
        _write_jsonl(Path(self.tmpdir) / "trades" / f"trades_{self.date_str}.jsonl", trades)

    def _write_missed(self, missed):
        _write_jsonl(Path(self.tmpdir) / "missed" / f"missed_{self.date_str}.jsonl", missed)

    def _write_scores(self, scores):
        _write_jsonl(Path(self.tmpdir) / "scores" / f"scores_{self.date_str}.jsonl", scores)

    def _write_errors(self, errors):
        _write_jsonl(Path(self.tmpdir) / "errors" / f"instrumentation_errors_{self.date_str}.jsonl", errors)

    def _write_heartbeats(self, heartbeats):
        _write_jsonl(Path(self.tmpdir) / "heartbeats" / f"heartbeat_{self.date_str}.jsonl", heartbeats)

    def test_empty_day(self):
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.total_trades == 0
        assert snapshot.missed_count == 0
        assert snapshot.error_count == 0

    def test_trade_aggregates(self):
        self._write_trades([
            {"stage": "entry", "trade_id": "t1"},
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 10,
             "market_regime": "trending_up", "entry_slippage_bps": 2.0},
            {"stage": "entry", "trade_id": "t2"},
            {"stage": "exit", "trade_id": "t2", "pnl": -200, "fees_paid": 10,
             "market_regime": "trending_up"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.total_trades == 2
        assert snapshot.win_count == 1
        assert snapshot.loss_count == 1
        assert snapshot.net_pnl == 300.0  # 500 + (-200)
        assert snapshot.win_rate == 0.5

    def test_profit_factor(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 600, "fees_paid": 0},
            {"stage": "exit", "trade_id": "t2", "pnl": -200, "fees_paid": 0},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.profit_factor == 3.0  # 600 / 200

    def test_missed_count(self):
        self._write_missed([
            {"signal": "test1", "blocked_by": "volume_filter", "first_hit": "TP"},
            {"signal": "test2", "blocked_by": "risk_cap", "first_hit": "SL"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.missed_count == 2
        assert snapshot.missed_would_have_won == 1
        assert snapshot.top_missed_filter == "volume_filter"

    def test_process_quality_aggregation(self):
        self._write_scores([
            {"process_quality_score": 90, "classification": "good_process",
             "root_causes": ["regime_aligned", "strong_signal"]},
            {"process_quality_score": 40, "classification": "neutral",
             "root_causes": ["regime_mismatch"]},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.avg_process_quality == 65.0  # (90 + 40) / 2
        assert snapshot.process_scores_distribution["good_process"] == 1
        assert snapshot.process_scores_distribution["neutral"] == 1
        assert snapshot.root_cause_distribution["regime_aligned"] == 1

    def test_error_count(self):
        self._write_errors([
            {"error": "test error 1"},
            {"error": "test error 2"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.error_count == 2

    def test_regime_breakdown(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100, "market_regime": "trending_up"},
            {"stage": "exit", "trade_id": "t2", "pnl": 50, "market_regime": "trending_up"},
            {"stage": "exit", "trade_id": "t3", "pnl": -80, "market_regime": "ranging"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert "trending_up" in snapshot.regime_breakdown
        assert snapshot.regime_breakdown["trending_up"]["trades"] == 2
        assert snapshot.regime_breakdown["trending_up"]["wins"] == 2
        assert snapshot.regime_breakdown["ranging"]["trades"] == 1

    def test_save_creates_json_file(self):
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        builder.save(snapshot)
        filepath = Path(self.tmpdir) / "daily" / f"daily_{self.date_str}.json"
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["date"] == self.date_str
        assert data["bot_id"] == "test_bot"

    def test_to_dict(self):
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        d = snapshot.to_dict()
        assert isinstance(d, dict)
        assert d["bot_id"] == "test_bot"
        assert d["strategy_type"] == "helix"

    def test_per_strategy_summary_single_strategy(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 10,
             "strategy_type": "helix", "entry_slippage_bps": 1.5},
            {"stage": "exit", "trade_id": "t2", "pnl": -200, "fees_paid": 10,
             "strategy_type": "helix"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert "helix" in snapshot.per_strategy_summary
        s = snapshot.per_strategy_summary["helix"]
        assert s["trades"] == 2
        assert s["win_count"] == 1
        assert s["loss_count"] == 1
        assert s["net_pnl"] == 300.0
        assert s["win_rate"] == 0.5
        assert s["avg_win"] == 500.0
        assert s["avg_loss"] == -200.0
        assert s["best_trade_pnl"] == 500.0
        assert s["worst_trade_pnl"] == -200.0
        assert s["avg_entry_slippage_bps"] == 1.5

    def test_per_strategy_summary_multi_strategy(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 0,
             "strategy_type": "helix"},
            {"stage": "exit", "trade_id": "t2", "pnl": -100, "fees_paid": 0,
             "strategy_type": "nqdtc"},
            {"stage": "exit", "trade_id": "t3", "pnl": 300, "fees_paid": 0,
             "strategy_type": "nqdtc"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert len(snapshot.per_strategy_summary) == 2
        assert snapshot.per_strategy_summary["helix"]["trades"] == 1
        assert snapshot.per_strategy_summary["helix"]["net_pnl"] == 500.0
        assert snapshot.per_strategy_summary["nqdtc"]["trades"] == 2
        assert snapshot.per_strategy_summary["nqdtc"]["net_pnl"] == 200.0
        assert snapshot.per_strategy_summary["nqdtc"]["win_count"] == 1
        assert snapshot.per_strategy_summary["nqdtc"]["loss_count"] == 1

    def test_per_strategy_summary_empty(self):
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.per_strategy_summary == {}

    def test_per_strategy_summary_in_saved_json(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 400, "fees_paid": 5,
             "strategy_type": "vdubus"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        builder.save(snapshot)
        filepath = Path(self.tmpdir) / "daily" / f"daily_{self.date_str}.json"
        data = json.loads(filepath.read_text())
        assert "per_strategy_summary" in data
        assert "vdubus" in data["per_strategy_summary"]
        assert data["per_strategy_summary"]["vdubus"]["trades"] == 1
        assert data["per_strategy_summary"]["vdubus"]["net_pnl"] == 400.0

    def test_heartbeat_metrics_are_derived_into_snapshot(self):
        self._write_heartbeats([
            {
                "timestamp": "2026-03-01T14:00:00+00:00",
                "portfolio_exposure": {
                    "gross_notional": 10000,
                    "allocated_nav": 100000,
                    "exposure_pct": 10.0,
                },
            },
            {
                "timestamp": "2026-03-01T14:00:30+00:00",
                "portfolio_exposure": {
                    "gross_notional": 15000,
                    "allocated_nav": 100000,
                    "exposure_pct": 15.0,
                },
            },
            {
                "timestamp": "2026-03-01T14:01:00+00:00",
                "portfolio_exposure": {
                    "gross_notional": 12000,
                    "allocated_nav": 100000,
                    "exposure_pct": 12.0,
                },
            },
            {
                "timestamp": "2026-03-01T14:02:30+00:00",
                "portfolio_exposure": {
                    "gross_notional": 9000,
                    "allocated_nav": 100000,
                    "exposure_pct": 9.0,
                },
            },
        ])

        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)

        assert snapshot.heartbeat_count == 4
        assert snapshot.heartbeat_gap_count == 1
        assert snapshot.data_gaps == 1
        assert snapshot.uptime_pct == 66.67
        assert snapshot.max_exposure == 15000.0
        assert snapshot.exposure_pct == 15.0
        assert snapshot.allocated_nav == 100000.0

    def test_checkpoint_snapshot_includes_timestamp_and_alias_fields(self):
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 250, "fees_paid": 0},
            {"stage": "exit", "trade_id": "t2", "pnl": -50, "fees_paid": 0},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str, snapshot_kind="checkpoint")

        assert snapshot.timestamp
        assert snapshot.snapshot_kind == "checkpoint"
        assert snapshot.trade_count == 2
        assert snapshot.avg_pnl == 100.0

    def test_rolling_performance_metrics_use_prior_daily_history(self):
        prior_dir = Path(self.tmpdir) / "daily"
        prior_dir.mkdir(parents=True, exist_ok=True)
        for date_str, pnl in (
            ("2026-02-26", 1000.0),
            ("2026-02-27", -500.0),
            ("2026-02-28", 750.0),
        ):
            payload = DailySnapshot(
                date=date_str,
                bot_id="test_bot",
                strategy_type="helix",
                net_pnl=pnl,
                allocated_nav=100000.0,
            ).to_dict()
            (prior_dir / f"daily_{date_str}.json").write_text(json.dumps(payload), encoding="utf-8")

        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500.0, "fees_paid": 0},
        ])
        self._write_heartbeats([
            {
                "timestamp": "2026-03-01T15:00:00+00:00",
                "portfolio_exposure": {
                    "gross_notional": 5000,
                    "allocated_nav": 100000,
                    "exposure_pct": 5.0,
                },
            }
        ])

        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)

        assert snapshot.sharpe_rolling_30d is not None
        assert snapshot.sharpe_ratio == snapshot.sharpe_rolling_30d
        assert snapshot.sharpe == snapshot.sharpe_ratio
        assert snapshot.sortino_rolling_30d is not None
        assert snapshot.max_drawdown_pct > 0
        assert snapshot.max_drawdown == snapshot.max_drawdown_pct
        assert snapshot.drawdown == snapshot.max_drawdown_pct
