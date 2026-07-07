# tests/test_backtest_calibration.py
"""Tests for BacktestCalibrationTracker — meta-learning on backtest reliability."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_assistant.schemas.parameter_search import (
    ParameterSearchReport,
    SearchRouting,
)
from trading_assistant.skills.backtest_calibration_tracker import BacktestCalibrationTracker


class TestRecordPrediction:
    def test_record_persists_to_jsonl(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        tracker.record_prediction(
            "sug1", "bot1", "signal", 1.15, SearchRouting.APPROVE,
        )
        path = tmp_path / "backtest_calibration.jsonl"
        assert path.exists()
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["suggestion_id"] == "sug1"
        assert record["predicted_improvement"] == 1.15
        assert record["predicted_routing"] == "approve"
        assert record["actual_composite_delta"] is None

    def test_multiple_predictions_appended(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        tracker.record_prediction("s1", "bot1", "signal", 1.1, SearchRouting.APPROVE)
        tracker.record_prediction("s2", "bot1", "exit", 1.2, SearchRouting.APPROVE)
        lines = (tmp_path / "backtest_calibration.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


class TestRecordOutcome:
    def test_outcome_updates_record(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        tracker.record_prediction("sug1", "bot1", "signal", 1.15, SearchRouting.APPROVE)
        tracker.record_outcome("sug1", 0.05)

        records = tracker._load_all()
        assert len(records) == 1
        assert records[0].actual_composite_delta == 0.05
        assert records[0].prediction_correct is True  # predicted >1.0, actual >0
        assert records[0].measured_at is not None

    def test_negative_outcome_marks_incorrect(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        tracker.record_prediction("sug1", "bot1", "signal", 1.15, SearchRouting.APPROVE)
        tracker.record_outcome("sug1", -0.03)  # Predicted improvement but got worse

        records = tracker._load_all()
        assert records[0].prediction_correct is False

    def test_missing_suggestion_is_noop(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        tracker.record_prediction("sug1", "bot1", "signal", 1.15, SearchRouting.APPROVE)
        tracker.record_outcome("nonexistent", 0.05)  # No-op

        records = tracker._load_all()
        assert records[0].actual_composite_delta is None  # Not updated


class TestGetReliability:
    def test_empty_returns_zero(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        reliability, n = tracker.get_reliability("bot1", "signal")
        assert reliability == 0.0
        assert n == 0

    def test_all_correct_returns_one(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        for i in range(5):
            tracker.record_prediction(f"s{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            tracker.record_outcome(f"s{i}", 0.05)

        reliability, n = tracker.get_reliability("bot1", "signal")
        assert reliability == 1.0
        assert n == 5

    def test_mixed_results(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        for i in range(10):
            tracker.record_prediction(f"s{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            tracker.record_outcome(f"s{i}", 0.05 if i < 7 else -0.02)

        reliability, n = tracker.get_reliability("bot1", "signal")
        assert reliability == 0.7
        assert n == 10

    def test_filters_by_bot_and_category(self, tmp_path: Path):
        tracker = BacktestCalibrationTracker(tmp_path)
        # bot1/signal: 2 correct
        for i in range(2):
            tracker.record_prediction(f"s{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            tracker.record_outcome(f"s{i}", 0.05)
        # bot2/exit: 1 incorrect
        tracker.record_prediction("s10", "bot2", "exit", 1.1, SearchRouting.APPROVE)
        tracker.record_outcome("s10", -0.03)

        r1, n1 = tracker.get_reliability("bot1", "signal")
        assert r1 == 1.0 and n1 == 2
        r2, n2 = tracker.get_reliability("bot2", "exit")
        assert r2 == 0.0 and n2 == 1


class TestApprovalModifier:
    def test_cold_start_returns_normal(self, tmp_path: Path):
        """No data → always 'normal'."""
        tracker = BacktestCalibrationTracker(tmp_path)
        assert tracker.get_approval_modifier("bot1", "signal") == "normal"

    def test_insufficient_samples_returns_normal(self, tmp_path: Path):
        """n < 5 → 'normal' even if reliability is bad."""
        tracker = BacktestCalibrationTracker(tmp_path)
        for i in range(4):
            tracker.record_prediction(f"s{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            tracker.record_outcome(f"s{i}", -0.05)  # All wrong

        assert tracker.get_approval_modifier("bot1", "signal") == "normal"

    def test_high_reliability_returns_fast_track(self, tmp_path: Path):
        """reliability >= 0.70 + n >= 5 → 'fast_track'."""
        tracker = BacktestCalibrationTracker(tmp_path)
        for i in range(5):
            tracker.record_prediction(f"s{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            tracker.record_outcome(f"s{i}", 0.05)

        assert tracker.get_approval_modifier("bot1", "signal") == "fast_track"

    def test_low_reliability_returns_require_experiment(self, tmp_path: Path):
        """reliability < 0.50 + n >= 5 → 'require_experiment'."""
        tracker = BacktestCalibrationTracker(tmp_path)
        for i in range(6):
            tracker.record_prediction(f"s{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            # Only 2/6 correct = 33% reliability
            tracker.record_outcome(f"s{i}", 0.05 if i < 2 else -0.03)

        assert tracker.get_approval_modifier("bot1", "signal") == "require_experiment"


class TestPipelineOverride:
    @pytest.mark.asyncio
    async def test_approve_overridden_to_experiment(self, tmp_path: Path):
        """When calibration says 'require_experiment', APPROVE→EXPERIMENT."""
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline
        from trading_assistant.skills.config_registry import ConfigRegistry
        from trading_assistant.skills.suggestion_backtester import SuggestionBacktester
        from trading_assistant.skills.approval_tracker import ApprovalTracker
        import yaml

        # Setup config
        config_dir = tmp_path / "bot_configs"
        config_dir.mkdir()
        (config_dir / "bot1.yaml").write_text(yaml.dump({
            "bot_id": "bot1",
            "allowed_edit_paths": ["config/*"],
            "parameters": [{
                "param_name": "signal_strength_min",
                "param_type": "YAML_FIELD",
                "file_path": "config/params.yaml",
                "yaml_key": "params.signal_strength_min",
                "current_value": 0.5,
                "valid_range": [0.1, 1.0],
                "value_type": "float",
                "category": "signal",
            }],
        }), encoding="utf-8")

        registry = ConfigRegistry(config_dir)
        backtester = SuggestionBacktester(registry, tmp_path)
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        suggestion_tracker = MagicMock()
        suggestion_tracker.load_all.return_value = [{
            "suggestion_id": "sug1",
            "bot_id": "bot1",
            "title": "Increase signal_strength_min to 0.7",
            "tier": "parameter",
            "category": "signal",
            "confidence": 0.8,
            "proposed_value": 0.7,
        }]

        # Build calibration tracker with low reliability
        cal_tracker = BacktestCalibrationTracker(tmp_path / "cal")
        for i in range(6):
            cal_tracker.record_prediction(f"old{i}", "bot1", "signal", 1.1, SearchRouting.APPROVE)
            cal_tracker.record_outcome(f"old{i}", -0.03)  # All wrong → require_experiment

        searcher = MagicMock()
        searcher.search.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            baseline_composite=0.7,
            best_value=0.75,
            best_composite=0.85,
            routing=SearchRouting.APPROVE,  # Searcher says APPROVE
        )

        # No experiment generator → experiment route returns None
        pipeline = AutonomousPipeline(
            config_registry=registry,
            backtester=backtester,
            approval_tracker=approval_tracker,
            suggestion_tracker=suggestion_tracker,
            parameter_searcher=searcher,
            calibration_tracker=cal_tracker,
            search_log_dir=tmp_path / "findings",
        )

        results = await pipeline.process_new_suggestions(["sug1"])
        # Override APPROVE→EXPERIMENT, but no experiment_config_generator → returns None
        assert len(results) == 0


class TestAutoOutcomeMeasurerIntegration:
    def test_measurer_does_not_write_calibration_tracker(self, tmp_path: Path):
        """AutoOutcomeMeasurer is early-warning only and does not update calibration."""
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        cal_tracker = BacktestCalibrationTracker(tmp_path / "cal")
        cal_tracker.record_prediction("sug1", "bot1", "signal", 1.15, SearchRouting.APPROVE)

        measurer = AutoOutcomeMeasurer(
            curated_dir=tmp_path / "curated",
            calibration_tracker=cal_tracker,
        )

        # Create minimal curated data for before/after
        for offset in range(-7, 7):
            from datetime import datetime, timedelta
            date = datetime(2026, 3, 1) + timedelta(days=offset)
            date_str = date.strftime("%Y-%m-%d")
            bot_dir = tmp_path / "curated" / date_str / "bot1"
            bot_dir.mkdir(parents=True, exist_ok=True)
            summary = {"net_pnl": 100 + offset * 5, "total_trades": 10, "win_count": 6}
            (bot_dir / "summary.json").write_text(json.dumps(summary))

        result = measurer.measure("sug1", "bot1", "2026-03-01")
        if result:
            # Calibration tracker keeps the prediction only; no outcome fan-out.
            records = cal_tracker._load_all()
            assert len(records) == 1
            assert records[0].actual_composite_delta is None
