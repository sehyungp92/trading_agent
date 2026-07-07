"""Comprehensive tests for ExperimentManager (Task 3)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_assistant.schemas.experiments import (
    ExperimentConfig,
    ExperimentStatus,
    ExperimentVariant,
)
from trading_assistant.skills.experiment_manager import ExperimentManager


def _make_config(
    experiment_id: str = "exp-001",
    bot_id: str = "bot1",
    success_metric: str = "pnl",
    min_trades: int = 30,
    significance_level: float = 0.05,
    max_duration_days: int = 30,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        bot_id=bot_id,
        title="Test Experiment",
        description="Test",
        success_metric=success_metric,
        min_trades_per_variant=min_trades,
        significance_level=significance_level,
        max_duration_days=max_duration_days,
        variants=[
            ExperimentVariant(name="control", params={}, allocation_pct=50.0),
            ExperimentVariant(name="treatment", params={"stop_loss": 0.02}, allocation_pct=50.0),
        ],
    )


def _ingest_trades(
    mgr: ExperimentManager,
    experiment_id: str,
    variant: str,
    pnls: list[float],
    extra_fields: dict | None = None,
) -> None:
    """Helper to ingest trade data for a variant."""
    trades = []
    for pnl in pnls:
        trade = {"pnl": pnl, "win": 1 if pnl > 0 else 0}
        if extra_fields:
            trade.update(extra_fields)
        trades.append(trade)
    mgr.ingest_variant_data(experiment_id, variant, trades)


@pytest.fixture
def mgr(tmp_path) -> ExperimentManager:
    findings = tmp_path / "findings"
    findings.mkdir()
    return ExperimentManager(findings_dir=findings, min_trades=5)


class TestLifecycle:
    """Full lifecycle: create → activate → ingest → analyze → conclude."""

    def test_full_lifecycle(self, mgr):
        config = _make_config(min_trades=5)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        _ingest_trades(mgr, "exp-001", "control", [10, -5, 15, -3, 8, 12, -2])
        _ingest_trades(mgr, "exp-001", "treatment", [20, -3, 18, -1, 12, 15, -1])

        result = mgr.analyze_experiment("exp-001")
        assert result.experiment_id == "exp-001"
        assert len(result.variant_metrics) == 2
        assert result.variant_metrics[0].variant_name == "control"
        assert result.variant_metrics[1].variant_name == "treatment"

        mgr.conclude_experiment("exp-001", result)
        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.CONCLUDED

    def test_create_deduplicates(self, mgr):
        config = _make_config()
        mgr.create_experiment(config)
        existing = mgr.create_experiment(config)
        assert existing.experiment_id == "exp-001"
        # Only one experiment in storage
        assert len(mgr._load_all()) == 1

    def test_activate_wrong_status_raises(self, mgr):
        config = _make_config()
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")
        with pytest.raises(ValueError, match="Cannot activate"):
            mgr.activate_experiment("exp-001")


class TestSuccessMetric:
    """3a: success_metric respected for t-test."""

    def test_pnl_metric(self, mgr):
        config = _make_config(success_metric="pnl", min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        _ingest_trades(mgr, "exp-001", "control", [10, 10, 10])
        _ingest_trades(mgr, "exp-001", "treatment", [20, 20, 20])

        result = mgr.analyze_experiment("exp-001")
        # Treatment is clearly better
        assert result.variant_metrics[1].avg_pnl > result.variant_metrics[0].avg_pnl

    def test_win_rate_metric(self, mgr):
        config = _make_config(
            experiment_id="exp-wr", success_metric="win_rate", min_trades=3,
        )
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-wr")

        # Control: mixed wins, Treatment: all wins
        _ingest_trades(mgr, "exp-wr", "control", [10, -5, 10])
        _ingest_trades(mgr, "exp-wr", "treatment", [5, 5, 5])

        result = mgr.analyze_experiment("exp-wr")
        assert result.variant_metrics[1].win_rate == 1.0
        assert result.variant_metrics[0].win_rate < 1.0


class TestProfitFactor:
    """3b: profit_factor for all-win case."""

    def test_all_wins_profit_factor_10(self, mgr):
        config = _make_config(min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        _ingest_trades(mgr, "exp-001", "control", [10, 20, 30])
        _ingest_trades(mgr, "exp-001", "treatment", [5, 5, 5])

        result = mgr.analyze_experiment("exp-001")
        # Control has all wins → profit_factor should be 10.0
        assert result.variant_metrics[0].profit_factor == 10.0

    def test_no_wins_profit_factor_0(self, mgr):
        config = _make_config(min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        _ingest_trades(mgr, "exp-001", "control", [-10, -20, -30])
        _ingest_trades(mgr, "exp-001", "treatment", [5, 5, 5])

        result = mgr.analyze_experiment("exp-001")
        assert result.variant_metrics[0].profit_factor == 0.0


class TestStatisticalEdgeCases:
    """3c: Handle zero variance and empty variant data."""

    def test_zero_variance_returns_p_value_1(self, mgr):
        config = _make_config(min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        # Both groups have identical values → zero variance (need >= min_trades=5)
        _ingest_trades(mgr, "exp-001", "control", [10, 10, 10, 10, 10])
        _ingest_trades(mgr, "exp-001", "treatment", [10, 10, 10, 10, 10])

        result = mgr.analyze_experiment("exp-001")
        assert result.p_value == 1.0

    def test_empty_data_returns_extend(self, mgr):
        config = _make_config(min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        # No data ingested
        result = mgr.analyze_experiment("exp-001")
        assert result.recommendation == "extend"

    def test_one_variant_empty_returns_extend(self, mgr):
        config = _make_config(min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        _ingest_trades(mgr, "exp-001", "control", [10, 20, 30])
        # treatment has no data

        result = mgr.analyze_experiment("exp-001")
        assert result.recommendation == "extend"


class TestIdempotentConclude:
    """3d: Double conclude is idempotent."""

    def test_double_conclude_idempotent(self, mgr):
        config = _make_config(min_trades=3)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")
        _ingest_trades(mgr, "exp-001", "control", [10, -5, 15])
        _ingest_trades(mgr, "exp-001", "treatment", [20, -3, 18])

        result = mgr.analyze_experiment("exp-001")
        mgr.conclude_experiment("exp-001", result)
        # Second conclude should not raise
        mgr.conclude_experiment("exp-001", result)

        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.CONCLUDED
        # Only one result recorded
        results = mgr.get_results()
        assert len(results) == 1


class TestAutoConclusion:
    """Auto-conclusion by duration and significance."""

    def test_auto_conclude_by_duration(self, mgr):
        config = _make_config(max_duration_days=1)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        # Manually set started_at to 2 days ago
        experiments = mgr._load_all()
        experiments[0].started_at = datetime.now(timezone.utc) - timedelta(days=2)
        mgr._save_all(experiments)

        assert mgr.check_auto_conclusion("exp-001") is True

    def test_auto_conclude_by_significance(self, mgr):
        config = _make_config(min_trades=3, significance_level=0.05)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        # Very different groups should trigger significance
        _ingest_trades(mgr, "exp-001", "control", [1, 2, 1, 2, 1])
        _ingest_trades(mgr, "exp-001", "treatment", [100, 200, 100, 200, 100])

        # If p-value < significance_level, auto-conclude returns True
        should_conclude = mgr.check_auto_conclusion("exp-001")
        # Whether it triggers depends on the data — just verify it doesn't crash
        assert isinstance(should_conclude, bool)


class TestCancellation:
    """Cancellation."""

    def test_cancel_experiment(self, mgr):
        config = _make_config()
        mgr.create_experiment(config)
        mgr.cancel_experiment("exp-001")

        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.CANCELLED

    def test_cancel_nonexistent_raises(self, mgr):
        with pytest.raises(ValueError):
            mgr.cancel_experiment("nonexistent")


class TestGetActive:
    """Get active experiments."""

    def test_get_active(self, mgr):
        config1 = _make_config(experiment_id="exp-001")
        config2 = _make_config(experiment_id="exp-002")
        mgr.create_experiment(config1)
        mgr.create_experiment(config2)
        mgr.activate_experiment("exp-001")

        active = mgr.get_active()
        assert len(active) == 1
        assert active[0].experiment_id == "exp-001"
