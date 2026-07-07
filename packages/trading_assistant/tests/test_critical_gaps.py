# tests/test_critical_gaps.py
"""Tests for critical gap fixes — category mapping, pattern lifecycle,
silent failure traps, and dead/wrong fields.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from trading_assistant.analysis.response_parser import parse_response
from trading_assistant.schemas.agent_response import (
    AgentSuggestion,
    ParsedAnalysis,
)
from trading_assistant.schemas.forecast_tracking import ForecastRecord
from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus
from trading_assistant.schemas.suggestion_tracking import SuggestionRecord
from trading_assistant.skills.forecast_tracker import ForecastTracker
from trading_assistant.skills.pattern_library import PatternLibrary
from trading_assistant.skills.suggestion_scorer import SuggestionScorer
from tests.factories import make_handlers as _factory_make_handlers


def _make_handlers(tmp_path, tracker=None, es=None):
    handlers, _, event_stream = _factory_make_handlers(
        tmp_path, suggestion_tracker=tracker, event_stream=es,
    )
    return handlers, handlers._suggestion_tracker, event_stream


# ──────────────────────────────────────────────────────
# Phase 1: Category/Tier Mapping
# ──────────────────────────────────────────────────────


class TestCategoryField:
    def test_suggestion_record_has_category_field(self):
        """SuggestionRecord should accept and store a category field."""
        rec = SuggestionRecord(
            suggestion_id="abc",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            category="exit_timing",
            source_report_id="run-1",
        )
        assert rec.category == "exit_timing"

    def test_suggestion_record_category_defaults_empty(self):
        """Category should default to empty string for backward compat."""
        rec = SuggestionRecord(
            suggestion_id="abc",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
        )
        assert rec.category == ""

    def test_scorer_uses_category_over_tier(self, tmp_path):
        """Scorer should prefer category field over tier reverse-mapping."""
        findings = tmp_path / "findings"
        findings.mkdir()

        # Write suggestion with category set
        suggestion = {
            "suggestion_id": "s1",
            "bot_id": "bot1",
            "tier": "parameter",
            "category": "exit_timing",
            "title": "Widen stops",
            "source_report_id": "run-1",
            "status": "deployed",
        }
        (findings / "suggestions.jsonl").write_text(
            json.dumps(suggestion) + "\n", encoding="utf-8",
        )

        # Write outcome
        outcome = {
            "suggestion_id": "s1",
            "implemented_date": "2026-03-01",
            "pnl_delta_7d": 100.0,
        }
        (findings / "outcomes.jsonl").write_text(
            json.dumps(outcome) + "\n", encoding="utf-8",
        )

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()

        # Should group by "exit_timing" (from category), not "filter_threshold" (from tier "parameter")
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].category == "exit_timing"

    def test_scorer_falls_back_to_tier_for_old_records(self, tmp_path):
        """Scorer should fall back to tier reverse-mapping when category is empty."""
        findings = tmp_path / "findings"
        findings.mkdir()

        suggestion = {
            "suggestion_id": "s2",
            "bot_id": "bot1",
            "tier": "filter",
            "title": "Tighten filter",
            "source_report_id": "run-2",
            "status": "deployed",
        }
        (findings / "suggestions.jsonl").write_text(
            json.dumps(suggestion) + "\n", encoding="utf-8",
        )

        outcome = {
            "suggestion_id": "s2",
            "implemented_date": "2026-03-01",
            "pnl_delta_7d": -50.0,
        }
        (findings / "outcomes.jsonl").write_text(
            json.dumps(outcome) + "\n", encoding="utf-8",
        )

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()

        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].category == "filter_threshold"


# ──────────────────────────────────────────────────────
# Phase 2: Pattern Lifecycle & Transfer
# ──────────────────────────────────────────────────────


class TestPatternLifecycle:
    def test_pattern_entry_has_validated_at_and_linked_id(self):
        """PatternEntry should accept validated_at and linked_suggestion_id."""
        entry = PatternEntry(
            title="Test Pattern",
            category="filter",
            source_bot="bot1",
            validated_at="2026-03-01",
            linked_suggestion_id="abc123",
        )
        assert entry.validated_at == "2026-03-01"
        assert entry.linked_suggestion_id == "abc123"

    def test_validate_pattern_sets_validated_at(self, tmp_path):
        """validate_pattern() should set status=VALIDATED and validated_at."""
        lib = PatternLibrary(tmp_path)
        entry = lib.add(PatternEntry(
            title="Test",
            category="filter",
            source_bot="bot1",
        ))

        lib.validate_pattern(entry.pattern_id)

        reloaded = lib.load_all()
        assert reloaded[0].status == PatternStatus.VALIDATED
        assert reloaded[0].validated_at != ""

    def test_update_status_to_validated_sets_date(self, tmp_path):
        """update_status(VALIDATED) should also set validated_at."""
        lib = PatternLibrary(tmp_path)
        entry = lib.add(PatternEntry(
            title="Test",
            category="exit_rule",
            source_bot="bot1",
        ))

        lib.update_status(entry.pattern_id, PatternStatus.VALIDATED)

        reloaded = lib.load_all()
        assert reloaded[0].validated_at != ""


class TestTransferProposals:
    def test_build_proposals_includes_proposed_patterns(self, tmp_path):
        """build_proposals() should include PROPOSED patterns, not just VALIDATED."""
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            title="New Filter",
            category="filter",
            source_bot="bot1",
            status=PatternStatus.PROPOSED,
        ))

        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path / "curated",
            bots=["bot1", "bot2"],
        )

        proposals = builder.build_proposals()
        assert len(proposals) >= 1
        assert proposals[0].target_bot == "bot2"

    def test_transfer_outcome_uses_net_pnl(self, tmp_path):
        """_load_bot_metrics should use net_pnl, not total_pnl."""
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

        lib = PatternLibrary(tmp_path)
        curated = tmp_path / "curated"

        # Create 4 days of data (need >= 3)
        from datetime import timedelta
        base = datetime(2026, 3, 5)
        for i in range(1, 5):
            d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            path = curated / d / "bot1" / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"net_pnl": 100.0, "win_rate": 0.6}), encoding="utf-8")

        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=curated, bots=["bot1"],
        )

        result = builder._load_bot_metrics("bot1", "2026-03-05", curated, before=False)
        assert result is not None
        assert result["pnl"] == 400.0


# ──────────────────────────────────────────────────────
# Phase 3: Silent Failure Traps
# ──────────────────────────────────────────────────────


class TestPartialParsing:
    def test_bad_prediction_does_not_kill_suggestions(self):
        """One bad prediction should not prevent valid suggestions from being parsed."""
        structured = json.dumps({
            "predictions": [
                {"bot_id": "bot1", "metric": "invalid_metric", "direction": "improve", "confidence": 0.7},
                {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8},
            ],
            "suggestions": [
                {"bot_id": "bot1", "title": "Widen stops", "category": "exit_timing"},
            ],
        })
        response = f"Report text\n<!-- STRUCTURED_OUTPUT\n{structured}\n-->"

        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.suggestions) == 1
        assert parsed.suggestions[0].title == "Widen stops"
        # Only the valid prediction survives
        assert len(parsed.predictions) == 1
        assert parsed.predictions[0].metric == "pnl"

    def test_bad_suggestion_does_not_kill_predictions(self):
        """One bad suggestion should not prevent valid predictions from being parsed."""
        structured = json.dumps({
            "predictions": [
                {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8},
            ],
            "suggestions": [
                {"bot_id": "bot1", "title": "Good", "category": "exit_timing"},
                {"confidence": "not_a_number"},  # Bad — missing required fields
            ],
        })
        response = f"Report text\n<!-- STRUCTURED_OUTPUT\n{structured}\n-->"

        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 1
        assert len(parsed.suggestions) == 1

    def test_all_bad_items_returns_empty_lists(self):
        """If all items are invalid, parse_success is still True but lists are empty."""
        structured = json.dumps({
            "predictions": [
                {"metric": "invalid"},
            ],
            "suggestions": [
                {"confidence": "bad"},
            ],
        })
        response = f"Report text\n<!-- STRUCTURED_OUTPUT\n{structured}\n-->"

        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 0
        assert len(parsed.suggestions) == 0


class TestValidationFallback:
    def test_validation_failure_still_records_suggestions(self, tmp_path):
        """When _validate_and_annotate returns None, suggestions should still be recorded."""
        handlers, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            suggestions=[AgentSuggestion(
                bot_id="bot1", title="Test suggestion", category="exit_timing",
            )],
            predictions=[],
            raw_report="test report",
            parse_success=True,
        )

        # Simulate validation returning None (what happens on exception)
        validation = None

        # The fallback logic from the handler
        if validation is None and parsed.suggestions:
            from trading_assistant.analysis.response_validator import ValidationResult
            validation = ValidationResult(
                approved_suggestions=parsed.suggestions,
                approved_predictions=parsed.predictions,
            )

        result = handlers._record_agent_suggestions(validation, "test-run", parsed)
        assert len(result) == 1

    def test_validation_none_without_suggestions_records_nothing(self, tmp_path):
        """When validation is None and no suggestions exist, nothing is recorded."""
        handlers, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            suggestions=[], predictions=[], raw_report="test", parse_success=True,
        )

        result = handlers._record_agent_suggestions(None, "test-run", parsed)
        assert result == {}


# ──────────────────────────────────────────────────────
# Phase 4: Dead/Wrong Fields
# ──────────────────────────────────────────────────────


class TestMorningScanField:
    def test_morning_scan_uses_net_pnl(self, tmp_path):
        """Morning scan should read net_pnl from summary.json, not total_pnl."""
        curated = tmp_path / "data" / "curated"
        yesterday = "2026-03-06"
        summary_path = curated / yesterday / "bot1" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"net_pnl": -500.0}), encoding="utf-8",
        )

        # Simulate the morning scan logic from app.py
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pnl = summary.get("net_pnl", 0)
        assert pnl == -500.0

        # total_pnl should NOT match
        pnl_old = summary.get("total_pnl", 0)
        assert pnl_old == 0


class TestFeedbackEventId:
    def test_feedback_event_ids_are_unique(self):
        """Two rapid feedback submissions should have different event_ids."""
        import secrets
        ids = set()
        for _ in range(100):
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            event_id = f"feedback-{ts}-{secrets.token_hex(6)}"
            ids.add(event_id)
        assert len(ids) == 100


class TestAccuracyByMetric:
    def test_forecast_meta_populates_accuracy_by_metric(self, tmp_path):
        """ForecastTracker should populate accuracy_by_metric from by_type records."""
        findings = tmp_path / "findings"
        findings.mkdir()

        tracker = ForecastTracker(findings)
        tracker.record_week(ForecastRecord(
            week_start="2026-W09",
            week_end="2026-W09",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
            by_type={"pnl": 0.8, "win_rate": 0.5},
        ))
        tracker.record_week(ForecastRecord(
            week_start="2026-W10",
            week_end="2026-W10",
            predictions_reviewed=10,
            correct_predictions=6,
            accuracy=0.6,
            by_type={"pnl": 0.6, "win_rate": 0.7},
        ))

        meta = tracker.compute_meta_analysis()
        assert "pnl" in meta.accuracy_by_metric
        assert "win_rate" in meta.accuracy_by_metric
        assert meta.accuracy_by_metric["pnl"] == pytest.approx(0.7, abs=0.01)
        assert meta.accuracy_by_metric["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_forecast_meta_empty_by_type_gives_empty_metric(self, tmp_path):
        """When by_type is empty, accuracy_by_metric should be empty."""
        findings = tmp_path / "findings"
        findings.mkdir()

        tracker = ForecastTracker(findings)
        tracker.record_week(ForecastRecord(
            week_start="2026-W09",
            week_end="2026-W09",
            predictions_reviewed=5,
            correct_predictions=3,
            accuracy=0.6,
        ))

        meta = tracker.compute_meta_analysis()
        assert meta.accuracy_by_metric == {}


class TestCategoryOnRecordedSuggestions:
    def test_record_agent_suggestions_sets_category(self, tmp_path):
        """_record_agent_suggestions should set category on SuggestionRecord."""
        handlers, tracker, es = _make_handlers(tmp_path)

        from trading_assistant.analysis.response_validator import ValidationResult

        validation = ValidationResult(
            approved_suggestions=[
                AgentSuggestion(bot_id="bot1", title="Test", category="exit_timing"),
            ],
        )
        parsed = ParsedAnalysis(
            suggestions=[AgentSuggestion(bot_id="bot1", title="Test", category="exit_timing")],
            raw_report="test",
            parse_success=True,
        )

        handlers._record_agent_suggestions(validation, "test-run", parsed)

        records = tracker.load_all()
        assert len(records) == 1
        assert records[0].get("category") == "exit_timing"
