# tests/test_structural_experiments.py
"""Tests for structural experiment basics (Section C)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_assistant.schemas.structural_experiment import (
    AcceptanceCriteria,
    ExperimentRecord,
    ExperimentStatus,
)
from trading_assistant.skills.structural_experiment_tracker import StructuralExperimentTracker


class TestAcceptanceCriteria:
    def test_defaults(self):
        c = AcceptanceCriteria(metric="pnl")
        assert c.direction == "improve"
        assert c.minimum_change == 0.0
        assert c.observation_window_days == 14
        assert c.minimum_trade_count == 20
        assert c.baseline_value is None


class TestExperimentRecord:
    def test_all_criteria_met_true(self):
        e = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            criteria_met=[True, True],
        )
        assert e.all_criteria_met is True

    def test_all_criteria_met_false(self):
        e = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            criteria_met=[True, False],
        )
        assert e.all_criteria_met is False

    def test_all_criteria_met_empty(self):
        e = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
        )
        assert e.all_criteria_met is False

    def test_is_evaluable_not_active(self):
        e = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            status=ExperimentStatus.PROPOSED,
        )
        assert e.is_evaluable is False

    def test_is_evaluable_within_window(self):
        e = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            status=ExperimentStatus.ACTIVE,
            activated_at=datetime.now(timezone.utc),
            acceptance_criteria=[AcceptanceCriteria(metric="pnl", observation_window_days=14)],
        )
        assert e.is_evaluable is False

    def test_is_evaluable_past_window(self):
        e = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            status=ExperimentStatus.ACTIVE,
            activated_at=datetime.now(timezone.utc) - timedelta(days=15),
            acceptance_criteria=[AcceptanceCriteria(metric="pnl", observation_window_days=14)],
        )
        assert e.is_evaluable is True


class TestStructuralExperimentTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        return StructuralExperimentTracker(store_dir=tmp_path)

    def _make_experiment(self, exp_id="exp_001", **kwargs):
        defaults = dict(
            experiment_id=exp_id, bot_id="bot1", title="Test experiment",
            acceptance_criteria=[AcceptanceCriteria(metric="pnl")],
        )
        defaults.update(kwargs)
        return ExperimentRecord(**defaults)

    def test_record_experiment(self, tracker):
        exp = self._make_experiment()
        assert tracker.record_experiment(exp) is True

    def test_record_experiment_dedup(self, tracker):
        exp = self._make_experiment()
        tracker.record_experiment(exp)
        assert tracker.record_experiment(exp) is False

    def test_activate(self, tracker):
        exp = self._make_experiment()
        tracker.record_experiment(exp)
        assert tracker.activate("exp_001") is True

        records = tracker._load_all()
        assert records[0].status == ExperimentStatus.ACTIVE
        assert records[0].activated_at is not None

    def test_activate_wrong_status(self, tracker):
        exp = self._make_experiment(status=ExperimentStatus.ACTIVE)
        tracker.record_experiment(exp)
        assert tracker.activate("exp_001") is False

    def test_activate_nonexistent(self, tracker):
        assert tracker.activate("nonexistent") is False

    def test_resolve_passed(self, tracker):
        exp = self._make_experiment(status=ExperimentStatus.ACTIVE)
        tracker.record_experiment(exp)
        assert tracker.resolve("exp_001", [True], [0.05], "Looks good") is True

        records = tracker._load_all()
        assert records[0].status == ExperimentStatus.PASSED
        assert records[0].criteria_met == [True]
        assert records[0].actual_values == [0.05]
        assert records[0].resolved_at is not None

    def test_resolve_failed(self, tracker):
        exp = self._make_experiment(status=ExperimentStatus.ACTIVE)
        tracker.record_experiment(exp)
        assert tracker.resolve("exp_001", [False], [-0.02]) is True

        records = tracker._load_all()
        assert records[0].status == ExperimentStatus.FAILED

    def test_resolve_not_active(self, tracker):
        exp = self._make_experiment()
        tracker.record_experiment(exp)
        assert tracker.resolve("exp_001", [True], [0.05]) is False

    def test_abandon(self, tracker):
        exp = self._make_experiment()
        tracker.record_experiment(exp)
        assert tracker.abandon("exp_001", "No longer relevant") is True

        records = tracker._load_all()
        assert records[0].status == ExperimentStatus.ABANDONED
        assert records[0].resolution_notes == "No longer relevant"

    def test_abandon_already_resolved(self, tracker):
        exp = self._make_experiment(status=ExperimentStatus.PASSED)
        tracker.record_experiment(exp)
        assert tracker.abandon("exp_001") is False

    def test_get_active_experiments(self, tracker):
        tracker.record_experiment(self._make_experiment("exp_001"))
        tracker.record_experiment(self._make_experiment(
            "exp_002", status=ExperimentStatus.ACTIVE,
        ))
        tracker.record_experiment(self._make_experiment(
            "exp_003", status=ExperimentStatus.PASSED,
        ))
        active = tracker.get_active_experiments()
        assert len(active) == 1
        assert active[0].experiment_id == "exp_002"

    def test_get_evaluable_experiments(self, tracker):
        tracker.record_experiment(self._make_experiment(
            "exp_001",
            status=ExperimentStatus.ACTIVE,
            activated_at=datetime.now(timezone.utc) - timedelta(days=30),
        ))
        tracker.record_experiment(self._make_experiment(
            "exp_002",
            status=ExperimentStatus.ACTIVE,
            activated_at=datetime.now(timezone.utc),
        ))
        evaluable = tracker.get_evaluable_experiments()
        assert len(evaluable) == 1
        assert evaluable[0].experiment_id == "exp_001"

    def test_compute_track_record_empty(self, tracker):
        record = tracker.compute_track_record()
        assert record["total"] == 0
        assert record["pass_rate"] == 0.0

    def test_compute_track_record(self, tracker):
        tracker.record_experiment(self._make_experiment(
            "exp_001", status=ExperimentStatus.PASSED,
        ))
        tracker.record_experiment(self._make_experiment(
            "exp_002", status=ExperimentStatus.FAILED,
        ))
        tracker.record_experiment(self._make_experiment(
            "exp_003", status=ExperimentStatus.ABANDONED,
        ))
        tracker.record_experiment(self._make_experiment(
            "exp_004", status=ExperimentStatus.ACTIVE,
        ))
        record = tracker.compute_track_record()
        assert record["total"] == 4
        assert record["passed"] == 1
        assert record["failed"] == 1
        assert record["abandoned"] == 1
        assert record["active"] == 1
        assert record["pass_rate"] == 0.5


class TestFindBySuggestionId:
    @pytest.fixture
    def tracker(self, tmp_path):
        return StructuralExperimentTracker(store_dir=tmp_path)

    def test_finds_matching_experiment(self, tracker):
        exp = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            suggestion_id="sug_abc",
        )
        tracker.record_experiment(exp)
        found = tracker.find_by_suggestion_id("sug_abc")
        assert found is not None
        assert found.experiment_id == "exp_001"

    def test_returns_none_when_not_found(self, tracker):
        exp = ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            suggestion_id="sug_abc",
        )
        tracker.record_experiment(exp)
        assert tracker.find_by_suggestion_id("sug_xyz") is None

    def test_returns_none_when_empty(self, tracker):
        assert tracker.find_by_suggestion_id("sug_abc") is None


class TestResponseValidatorStructural:
    def test_blocks_proposals_without_criteria(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        validator = ResponseValidator()
        proposal = StructuralProposal(
            bot_id="bot1", title="No criteria", description="test",
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.blocked_structural_proposals) == 1
        assert len(result.approved_structural_proposals) == 0

    def test_approves_proposals_with_criteria(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        validator = ResponseValidator()
        proposal = StructuralProposal(
            bot_id="bot1", title="With criteria",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.approved_structural_proposals) == 1
        assert len(result.blocked_structural_proposals) == 0

    def test_blocks_criteria_without_metric(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        validator = ResponseValidator()
        proposal = StructuralProposal(
            bot_id="bot1", title="Bad criteria",
            acceptance_criteria=[{"direction": "improve"}],  # No metric
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.blocked_structural_proposals) == 1

    def test_mixed_proposals(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        validator = ResponseValidator()
        good = StructuralProposal(
            bot_id="bot1", title="Good",
            acceptance_criteria=[{"metric": "win_rate", "direction": "improve"}],
        )
        bad = StructuralProposal(
            bot_id="bot1", title="Bad",
        )
        parsed = ParsedAnalysis(structural_proposals=[good, bad])
        result = validator.validate(parsed)
        assert len(result.approved_structural_proposals) == 1
        assert len(result.blocked_structural_proposals) == 1
        assert "Bad" in result.validator_notes


class TestResponseValidatorHypothesisBlocking:
    """Verify Fix 10: validator blocks proposals linked to retired/negative hypotheses."""

    def test_blocks_proposal_with_retired_hypothesis(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        track_record = {
            "h-bad": {"effectiveness": -0.5, "status": "retired", "title": "Bad idea"},
        }
        validator = ResponseValidator(hypothesis_track_record=track_record)
        proposal = StructuralProposal(
            bot_id="bot1", title="Uses bad hypothesis",
            hypothesis_id="h-bad",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.blocked_structural_proposals) == 1
        assert len(result.approved_structural_proposals) == 0

    def test_blocks_proposal_with_zero_effectiveness(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        track_record = {
            "h-neutral": {"effectiveness": 0.0, "status": "active", "title": "Neutral"},
        }
        validator = ResponseValidator(hypothesis_track_record=track_record)
        proposal = StructuralProposal(
            bot_id="bot1", title="Neutral hypothesis",
            hypothesis_id="h-neutral",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.blocked_structural_proposals) == 1

    def test_approves_proposal_with_positive_hypothesis(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        track_record = {
            "h-good": {"effectiveness": 0.8, "status": "active", "title": "Good idea"},
        }
        validator = ResponseValidator(hypothesis_track_record=track_record)
        proposal = StructuralProposal(
            bot_id="bot1", title="Good hypothesis",
            hypothesis_id="h-good",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.approved_structural_proposals) == 1
        assert len(result.blocked_structural_proposals) == 0

    def test_no_track_record_allows_proposal(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis, StructuralProposal

        validator = ResponseValidator()  # No hypothesis_track_record
        proposal = StructuralProposal(
            bot_id="bot1", title="Unknown hypothesis",
            hypothesis_id="h-unknown",
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        parsed = ParsedAnalysis(structural_proposals=[proposal])
        result = validator.validate(parsed)
        assert len(result.approved_structural_proposals) == 1


class TestMetricFieldMapping:
    """Verify the metric name → GroundTruthSnapshot field mapping used in
    structural experiment evaluation (Fix 4)."""

    def test_common_metric_names_map_to_snapshot_fields(self):
        from trading_assistant.schemas.learning_ledger import GroundTruthSnapshot

        # These are the mappings used in app.py _check_experiments
        mapping = {
            "sharpe": "sharpe_ratio_30d",
            "sharpe_ratio": "sharpe_ratio_30d",
            "win_rate": "win_rate",
            "drawdown": "max_drawdown_pct",
            "max_drawdown": "max_drawdown_pct",
            "process_quality": "avg_process_quality",
            "composite": "composite_score",
            "composite_score": "composite_score",
            "pnl": "pnl_total",
        }
        snapshot = GroundTruthSnapshot(
            snapshot_date="2026-03-01", bot_id="bot1",
            pnl_total=100.0, sharpe_ratio_30d=1.5, win_rate=0.6,
            max_drawdown_pct=0.05, avg_process_quality=75.0,
            composite_score=0.7,
        )
        for common_name, field_name in mapping.items():
            val = getattr(snapshot, field_name, None)
            assert val is not None, f"Field {field_name} (for metric '{common_name}') not found on GroundTruthSnapshot"


class TestStructuralProposalSchema:
    def test_acceptance_criteria_field(self):
        from trading_assistant.schemas.agent_response import StructuralProposal

        p = StructuralProposal(
            bot_id="bot1", title="Test",
            acceptance_criteria=[
                {"metric": "pnl", "direction": "improve", "minimum_change": 0.01},
            ],
        )
        assert len(p.acceptance_criteria) == 1
        assert p.acceptance_criteria[0]["metric"] == "pnl"

    def test_acceptance_criteria_default_empty(self):
        from trading_assistant.schemas.agent_response import StructuralProposal

        p = StructuralProposal(bot_id="bot1", title="Test")
        assert p.acceptance_criteria == []


class TestContextBuilderExperiments:
    def test_load_experiment_track_record_empty(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        ctx = ContextBuilder(tmp_path)
        assert ctx.load_experiment_track_record() == {}

    def test_load_active_experiments_empty(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        ctx = ContextBuilder(tmp_path)
        assert ctx.load_active_experiments() == []

    def test_load_experiment_track_record_with_data(self, tmp_path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        findings = tmp_path / "findings"
        findings.mkdir()
        tracker = StructuralExperimentTracker(store_dir=findings)
        tracker.record_experiment(ExperimentRecord(
            experiment_id="exp_001", bot_id="bot1", title="Test",
            status=ExperimentStatus.PASSED,
        ))
        ctx = ContextBuilder(tmp_path)
        record = ctx.load_experiment_track_record()
        assert record["total"] == 1
        assert record["passed"] == 1
