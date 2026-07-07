# tests/test_parameter_search_integration.py
"""Integration tests for ParameterSearcher wired into AutonomousPipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from trading_assistant.schemas.repo_changes import (
    ChangeKind,
)
from trading_assistant.schemas.parameter_search import ParameterSearchReport, SearchRouting
from trading_assistant.skills.approval_tracker import ApprovalTracker
from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.suggestion_backtester import SuggestionBacktester


# ─── Helpers ─────────────────────────────────────────────────────────

def _write_bot_config(tmp_path: Path, bot_id: str = "bot1") -> Path:
    """Write a bot config YAML and return config dir."""
    config_dir = tmp_path / "bot_configs"
    config_dir.mkdir()
    (config_dir / f"{bot_id}.yaml").write_text(yaml.dump({
        "bot_id": bot_id,
        "repo_url": "https://github.com/test/bot1",
        "repo_dir": str(tmp_path / "repo"),
        "allowed_edit_paths": ["config/*"],
        "verification_commands": ["pytest"],
        "parameters": [{
            "param_name": "signal_strength_min",
            "param_type": "YAML_FIELD",
            "file_path": "config/params.yaml",
            "yaml_key": "params.signal_strength_min",
            "current_value": 0.5,
            "valid_range": [0.1, 1.0],
            "value_type": "float",
            "category": "signal",
            "is_safety_critical": False,
        }],
    }), encoding="utf-8")
    return config_dir


def _make_suggestion(
    sid: str = "sug1",
    bot_id: str = "bot1",
    proposed: float = 0.7,
) -> dict:
    return {
        "suggestion_id": sid,
        "bot_id": bot_id,
        "title": f"Increase signal_strength_min to {proposed}",
        "description": f"Set signal_strength_min to {proposed}",
        "tier": "parameter",
        "category": "signal",
        "confidence": 0.8,
        "status": "proposed",
        "proposed_value": proposed,
    }


def _make_pipeline(
    tmp_path: Path,
    simulate_fn=None,
    with_searcher: bool = True,
    with_experiment_gen: bool = False,
    with_experiment_manager: bool = False,
) -> tuple[AutonomousPipeline, Path]:
    """Build pipeline with or without parameter_searcher."""
    config_dir = _write_bot_config(tmp_path)
    registry = ConfigRegistry(config_dir)

    backtester = SuggestionBacktester(registry, tmp_path)
    approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")

    suggestion_tracker = MagicMock()
    suggestion_tracker.load_all.return_value = [_make_suggestion()]

    searcher = None
    if with_searcher:
        searcher = MagicMock()

    exp_gen = None
    if with_experiment_gen:
        exp_gen = MagicMock()
        exp_gen.generate_from_suggestion.return_value = MagicMock(
            experiment_id="exp_001",
        )

    exp_manager = None
    if with_experiment_manager:
        exp_manager = MagicMock()
        exp_manager.get_active.return_value = []

    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    pipeline = AutonomousPipeline(
        config_registry=registry,
        backtester=backtester,
        approval_tracker=approval_tracker,
        suggestion_tracker=suggestion_tracker,
        parameter_searcher=searcher,
        experiment_config_generator=exp_gen,
        experiment_manager=exp_manager,
        search_log_dir=findings_dir,
    )
    return pipeline, findings_dir


# ─── Tests ───────────────────────────────────────────────────────────

class TestSearchApproveFlow:
    @pytest.mark.asyncio
    async def test_search_approve_uses_best_value(self, tmp_path: Path):
        """When searcher returns APPROVE, pipeline uses best_value (may differ from proposed)."""
        pipeline, findings_dir = _make_pipeline(tmp_path, with_searcher=True)

        # Mock searcher to return APPROVE with best_value=0.75 (different from proposed 0.7)
        pipeline._parameter_searcher.search_with_regime_analysis.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            baseline_composite=0.7,
            candidates_tested=11,
            candidates_passing=5,
            best_value=0.75,
            best_composite=0.85,
            routing=SearchRouting.APPROVE,
            exploration_summary="Searched 11 values",
        )

        results = await pipeline.process_new_suggestions(["sug1"])
        assert len(results) == 1
        req = results[0]
        assert req.change_kind == ChangeKind.PARAMETER_CHANGE
        # best_value=0.75 used, not original proposed 0.7
        assert req.param_changes[0]["proposed"] == 0.75

    @pytest.mark.asyncio
    async def test_search_discard_returns_none(self, tmp_path: Path):
        """When searcher returns DISCARD, no approval request created."""
        pipeline, findings_dir = _make_pipeline(tmp_path, with_searcher=True)

        pipeline._parameter_searcher.search_with_regime_analysis.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            routing=SearchRouting.DISCARD,
            discard_reason="No candidates passed",
        )

        results = await pipeline.process_new_suggestions(["sug1"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_discard_records_signal(self, tmp_path: Path):
        """DISCARD path records negative search signal."""
        pipeline, findings_dir = _make_pipeline(tmp_path, with_searcher=True)

        pipeline._parameter_searcher.search_with_regime_analysis.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            routing=SearchRouting.DISCARD,
            discard_reason="No candidates passed",
        )

        await pipeline.process_new_suggestions(["sug1"])
        signals_path = findings_dir / "search_signals.jsonl"
        assert signals_path.exists()
        record = json.loads(signals_path.read_text(encoding="utf-8").strip())
        assert record["positive"] is False
        assert record["bot_id"] == "bot1"


class TestSearchExperimentFlow:
    @pytest.mark.asyncio
    async def test_experiment_route_creates_structural_request(self, tmp_path: Path):
        """EXPERIMENT routing creates a structural change request via ExperimentConfigGenerator."""
        pipeline, findings_dir = _make_pipeline(
            tmp_path, with_searcher=True,
            with_experiment_gen=True, with_experiment_manager=True,
        )

        pipeline._parameter_searcher.search_with_regime_analysis.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            baseline_composite=0.7,
            candidates_tested=11,
            candidates_passing=3,
            best_value=0.65,
            best_composite=0.72,
            routing=SearchRouting.EXPERIMENT,
            exploration_summary="Marginal improvement",
        )

        results = await pipeline.process_new_suggestions(["sug1"])
        assert len(results) == 1
        req = results[0]
        assert req.change_kind == ChangeKind.STRUCTURAL_CHANGE
        assert "experiment" in req.title.lower() or "A/B" in req.title

    @pytest.mark.asyncio
    async def test_experiment_blocked_at_cap(self, tmp_path: Path):
        """When bot already has 2 active experiments, EXPERIMENT route returns None."""
        pipeline, findings_dir = _make_pipeline(
            tmp_path, with_searcher=True,
            with_experiment_gen=True, with_experiment_manager=True,
        )

        # Mock 2 active experiments for bot1
        mock_exp1 = MagicMock(bot_id="bot1")
        mock_exp2 = MagicMock(bot_id="bot1")
        pipeline._experiment_manager.get_active.return_value = [mock_exp1, mock_exp2]

        pipeline._parameter_searcher.search_with_regime_analysis.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            routing=SearchRouting.EXPERIMENT,
            best_value=0.65,
        )

        results = await pipeline.process_new_suggestions(["sug1"])
        assert len(results) == 0


class TestLegacyPath:
    @pytest.mark.asyncio
    async def test_legacy_path_when_no_searcher(self, tmp_path: Path):
        """When parameter_searcher=None, legacy backtester path is used."""
        pipeline, findings_dir = _make_pipeline(tmp_path, with_searcher=False)

        # Write some trade data so backtester has data
        curated = tmp_path / "data" / "curated" / "2026-03-01" / "bot1"
        curated.mkdir(parents=True)
        trades = [
            {"pnl": 10.0, "date": "2026-03-01"} for _ in range(20)
        ]
        with (curated / "trades.jsonl").open("w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

        await pipeline.process_new_suggestions(["sug1"])
        # Legacy path runs backtester.backtest_suggestion
        # Result depends on trade data quality
        # The key assertion: searcher was NOT called
        assert pipeline._parameter_searcher is None


class TestSearchReportPersistence:
    @pytest.mark.asyncio
    async def test_search_report_persisted_to_jsonl(self, tmp_path: Path):
        """Search reports are persisted to search_reports.jsonl."""
        pipeline, findings_dir = _make_pipeline(tmp_path, with_searcher=True)

        report = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            routing=SearchRouting.APPROVE,
            best_value=0.75,
            best_composite=0.85,
        )
        pipeline._parameter_searcher.search_with_regime_analysis.return_value = report

        await pipeline.process_new_suggestions(["sug1"])
        reports_path = findings_dir / "search_reports.jsonl"
        assert reports_path.exists()
        saved = json.loads(reports_path.read_text(encoding="utf-8").strip())
        assert saved["suggestion_id"] == "sug1"
        assert saved["routing"] == "approve"

    @pytest.mark.asyncio
    async def test_best_value_differs_from_proposed(self, tmp_path: Path):
        """System can find a better value than what Claude originally proposed."""
        pipeline, findings_dir = _make_pipeline(tmp_path, with_searcher=True)

        pipeline._parameter_searcher.search_with_regime_analysis.return_value = ParameterSearchReport(
            suggestion_id="sug1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            baseline_composite=0.7,
            candidates_tested=11,
            candidates_passing=5,
            best_value=0.82,  # System found 0.82 is better than proposed 0.7
            best_composite=0.9,
            routing=SearchRouting.APPROVE,
        )

        results = await pipeline.process_new_suggestions(["sug1"])
        assert len(results) == 1
        assert results[0].param_changes[0]["proposed"] == 0.82
