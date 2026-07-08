"""Tests for experiment infrastructure (#20)."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from strategies.stock.instrumentation.src.experiment import ExperimentMetadata, ExperimentRegistry
from strategies.stock.instrumentation.src.experiment_analysis import (
    VariantStats, ExperimentResult, analyze_experiment, _welch_t_test, _compute_variant_stats,
)


class TestExperimentMetadata:
    def test_defaults(self):
        exp = ExperimentMetadata(
            experiment_id="exp_001",
            hypothesis="Tighter trail improves Sharpe",
            variants=["control", "tight"],
            start_date="2026-03-15",
            strategy_type="helix",
        )
        assert exp.primary_metric == "sharpe"
        assert exp.min_trades_per_variant == 30
        assert exp.end_date is None


class TestExperimentRegistry:
    def _make_config(self, tmpdir, experiments: dict) -> Path:
        path = Path(tmpdir) / "experiments.yaml"
        path.write_text(yaml.dump({"experiments": experiments}))
        return path

    def test_load_empty_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            path.write_text("experiments: {}")
            reg = ExperimentRegistry(config_path=path)
            assert reg.all_experiments() == []

    def test_load_with_experiments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_config(tmpdir, {
                "exp_001": {
                    "hypothesis": "Tighter trail",
                    "variants": ["control", "tight"],
                    "start_date": "2026-03-01",
                    "strategy_type": "helix",
                },
            })
            reg = ExperimentRegistry(config_path=path)
            assert len(reg.all_experiments()) == 1
            exp = reg.get("exp_001")
            assert exp is not None
            assert exp.variants == ["control", "tight"]

    def test_get_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_config(tmpdir, {})
            reg = ExperimentRegistry(config_path=path)
            assert reg.get("nonexistent") is None

    def test_active_experiments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_config(tmpdir, {
                "exp_active": {
                    "hypothesis": "test",
                    "variants": ["a", "b"],
                    "start_date": "2026-01-01",
                    "strategy_type": "helix",
                },
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
            active = reg.active_experiments(as_of="2026-03-01")
            assert len(active) == 1
            assert active[0].experiment_id == "exp_active"

    def test_missing_file(self):
        reg = ExperimentRegistry(config_path=Path("/nonexistent/path.yaml"))
        assert reg.all_experiments() == []


class TestWelchTTest:
    def test_identical_means(self):
        p = _welch_t_test(100.0, 100.0, 10.0, 10.0, 50, 50)
        assert p >= 0.99  # same means → p=1

    def test_very_different_means(self):
        p = _welch_t_test(100.0, 0.0, 1.0, 1.0, 100, 100)
        assert p < 0.01  # clearly different

    def test_small_sample(self):
        p = _welch_t_test(100.0, 0.0, 1.0, 1.0, 1, 1)
        assert p == 1.0  # can't test with n=1


class TestComputeVariantStats:
    def test_empty(self):
        vs = _compute_variant_stats("control", [])
        assert vs.trade_count == 0
        assert vs.win_rate == 0.0

    def test_basic_stats(self):
        pnls = [100.0, -50.0, 200.0, -25.0, 150.0]
        vs = _compute_variant_stats("test", pnls)
        assert vs.trade_count == 5
        assert vs.win_rate == 0.6  # 3 wins / 5 trades
        assert vs.total_pnl == 375.0
        assert vs.avg_pnl == 75.0

    def test_single_trade(self):
        vs = _compute_variant_stats("test", [100.0])
        assert vs.trade_count == 1
        assert vs.sharpe == 0.0  # can't compute std with n=1


class TestAnalyzeExperiment:
    def _setup_trades(self, tmpdir, trades: list[dict]) -> Path:
        trades_dir = Path(tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        filepath = trades_dir / "trades_2026-03-01.jsonl"
        with open(filepath, "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")
        return trades_dir

    def test_no_matching_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trades_dir = self._setup_trades(tmpdir, [
                {"experiment_id": "other", "experiment_variant": "a", "stage": "exit", "pnl": 100},
            ])
            result = analyze_experiment("exp_001", trades_dir)
            assert len(result.variants) == 0
            assert not result.sufficient_sample

    def test_basic_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trades = []
            for i in range(40):
                trades.append({
                    "experiment_id": "exp_001",
                    "experiment_variant": "control",
                    "stage": "exit",
                    "pnl": 100.0 if i % 2 == 0 else -50.0,
                })
                trades.append({
                    "experiment_id": "exp_001",
                    "experiment_variant": "treatment",
                    "stage": "exit",
                    "pnl": 150.0 if i % 2 == 0 else -30.0,
                })
            trades_dir = self._setup_trades(tmpdir, trades)
            result = analyze_experiment("exp_001", trades_dir, min_trades=30)
            assert len(result.variants) == 2
            assert result.sufficient_sample is True
            assert result.p_value is not None
            assert "control" in result.variants
            assert "treatment" in result.variants

    def test_insufficient_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trades = [
                {"experiment_id": "exp_001", "experiment_variant": "control", "stage": "exit", "pnl": 100},
                {"experiment_id": "exp_001", "experiment_variant": "treatment", "stage": "exit", "pnl": 200},
            ]
            trades_dir = self._setup_trades(tmpdir, trades)
            result = analyze_experiment("exp_001", trades_dir, min_trades=30)
            assert not result.sufficient_sample
            assert result.p_value is None

    def test_skips_entry_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trades = [
                {"experiment_id": "exp_001", "experiment_variant": "control", "stage": "entry", "pnl": None},
            ]
            trades_dir = self._setup_trades(tmpdir, trades)
            result = analyze_experiment("exp_001", trades_dir)
            assert len(result.variants) == 0
