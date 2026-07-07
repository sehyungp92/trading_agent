"""Tests for experiment breakdown in DailySnapshot and ExperimentRegistry.export_active()."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from strategies.stock.instrumentation.src.daily_snapshot import DailySnapshotBuilder, DailySnapshot
from strategies.stock.instrumentation.src.experiment import ExperimentRegistry


def _write_jsonl(filepath: Path, events: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class TestExperimentBreakdown:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "strategy_type": "helix",
            "data_dir": self.tmpdir,
        }
        self.date_str = "2026-03-01"

    def _write_trades(self, trades):
        _write_jsonl(
            Path(self.tmpdir) / "trades" / f"trades_{self.date_str}.jsonl", trades
        )

    def test_empty_experiments(self):
        """No trades have experiment_id → experiment_breakdown is {}."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 10},
            {"stage": "exit", "trade_id": "t2", "pnl": -200, "fees_paid": 5},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.experiment_breakdown == {}

    def test_multi_strategy_experiment(self):
        """Helix + nqdtc trades in same experiment → grouped correctly."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 10,
             "experiment_id": "exp_001", "experiment_variant": "control",
             "strategy_type": "helix"},
            {"stage": "exit", "trade_id": "t2", "pnl": 300, "fees_paid": 5,
             "experiment_id": "exp_001", "experiment_variant": "control",
             "strategy_type": "helix"},
            {"stage": "exit", "trade_id": "t3", "pnl": -100, "fees_paid": 5,
             "experiment_id": "exp_001", "experiment_variant": "control",
             "strategy_type": "nqdtc"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert "exp_001:control" in snapshot.experiment_breakdown
        bd = snapshot.experiment_breakdown["exp_001:control"]
        assert bd["trades"] == 3
        assert bd["win_count"] == 2
        assert bd["loss_count"] == 1
        # Dominant strategy_type is helix (2 vs 1)
        assert bd["strategy_type"] == "helix"

    def test_param_set_id_uniform(self):
        """All trades in variant share same param_set_id → populated."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a",
             "param_set_id": "abc123", "strategy_type": "helix"},
            {"stage": "exit", "trade_id": "t2", "pnl": 200, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a",
             "param_set_id": "abc123", "strategy_type": "helix"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.experiment_breakdown["exp_001:a"]["param_set_id"] == "abc123"

    def test_param_set_id_mixed(self):
        """Mixed param_set_ids → empty string."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a",
             "param_set_id": "abc123", "strategy_type": "helix"},
            {"stage": "exit", "trade_id": "t2", "pnl": 200, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a",
             "param_set_id": "def456", "strategy_type": "helix"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.experiment_breakdown["exp_001:a"]["param_set_id"] == ""

    def test_avg_slippage_bps(self):
        """Only non-None slippage values are averaged."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a",
             "entry_slippage_bps": 2.0},
            {"stage": "exit", "trade_id": "t2", "pnl": 200, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a",
             "entry_slippage_bps": 4.0},
            {"stage": "exit", "trade_id": "t3", "pnl": -100, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "a"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.experiment_breakdown["exp_001:a"]["avg_slippage_bps"] == 3.0

    def test_multiple_variants(self):
        """Two variants of same experiment → two keys in breakdown."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 500, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "control"},
            {"stage": "exit", "trade_id": "t2", "pnl": -200, "fees_paid": 0,
             "experiment_id": "exp_001", "experiment_variant": "treatment"},
        ])
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert "exp_001:control" in snapshot.experiment_breakdown
        assert "exp_001:treatment" in snapshot.experiment_breakdown
        assert snapshot.experiment_breakdown["exp_001:control"]["net_pnl"] == 500.0
        assert snapshot.experiment_breakdown["exp_001:treatment"]["net_pnl"] == -200.0


class TestExportActive:
    def _make_config(self, tmpdir, experiments: dict) -> Path:
        path = Path(tmpdir) / "experiments.yaml"
        path.write_text(yaml.dump({"experiments": experiments}))
        return path

    def test_export_active_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_config(tmpdir, {
                "exp_001": {
                    "hypothesis": "Tighter trail improves Sharpe",
                    "variants": ["control", "tight"],
                    "start_date": "2026-01-01",
                    "strategy_type": "helix",
                    "primary_metric": "sharpe",
                    "secondary_metrics": ["win_rate", "avg_pnl"],
                    "min_trades_per_variant": 30,
                },
            })
            reg = ExperimentRegistry(config_path=path)
            result = reg.export_active(as_of="2026-03-01")
            assert "exp_001" in result
            exp = result["exp_001"]
            assert exp["hypothesis"] == "Tighter trail improves Sharpe"
            assert exp["variants"] == ["control", "tight"]
            assert exp["primary_metric"] == "sharpe"
            assert exp["strategy_type"] == "helix"
            assert exp["status"] == "active"
            assert exp["min_trades_per_variant"] == 30

    def test_export_active_excludes_inactive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_config(tmpdir, {
                "exp_ended": {
                    "hypothesis": "test",
                    "variants": ["a", "b"],
                    "start_date": "2026-01-01",
                    "end_date": "2026-02-01",
                    "strategy_type": "helix",
                },
                "exp_future": {
                    "hypothesis": "test",
                    "variants": ["a", "b"],
                    "start_date": "2027-01-01",
                    "strategy_type": "helix",
                },
            })
            reg = ExperimentRegistry(config_path=path)
            result = reg.export_active(as_of="2026-03-01")
            assert result == {}

    def test_export_active_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_config(tmpdir, {})
            reg = ExperimentRegistry(config_path=path)
            result = reg.export_active()
            assert result == {}


class TestActiveExperimentsInDailySnapshot:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "strategy_type": "helix",
            "data_dir": self.tmpdir,
        }
        self.date_str = "2026-03-01"

    def test_active_experiments_populated_with_registry(self):
        with tempfile.TemporaryDirectory() as exp_dir:
            exp_path = Path(exp_dir) / "experiments.yaml"
            exp_path.write_text(yaml.dump({"experiments": {
                "exp_001": {
                    "hypothesis": "Test hypothesis",
                    "variants": ["control", "treatment"],
                    "start_date": "2026-01-01",
                    "strategy_type": "helix",
                },
            }}))
            reg = ExperimentRegistry(config_path=exp_path)
            builder = DailySnapshotBuilder(self.config, experiment_registry=reg)
            snapshot = builder.build(self.date_str)
            assert "exp_001" in snapshot.active_experiments
            assert snapshot.active_experiments["exp_001"]["status"] == "active"

    def test_active_experiments_empty_without_registry(self):
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build(self.date_str)
        assert snapshot.active_experiments == {}
