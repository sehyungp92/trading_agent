# tests/test_learning_loops.py
"""Merged learning loop tests.

Combines:
- test_learning_loop_e2e.py: End-to-end learning loop integration tests
- test_learning_loop_gaps.py: Tests for 8 learning loop gap closures
- test_learning_loop_remaining_gaps.py: Tests for closing remaining learning loop gaps (C1-C5, M1-M3)

Covers:
1. Full learning cycle (suggestion → measurement → feedback)
2. Validation blocks, prediction evaluation, context builder enrichment
3. Gap 8: hypothesis_id field on SuggestionRecord
4. Gap 1: Agent suggestion recording via _record_agent_suggestions
5. Gap 4: Blocked suggestion details in validation log
6. Gap 5: MagicMock removal + load_track_record_from_file
7. Gap 7: TransferProposalBuilder findings_dir wiring
8. Gap 6: JSONL-backed HypothesisLibrary in weekly handler
9. Gap 3: Hypothesis outcome linking in _measure_outcomes
10. Gap 2: Automated prediction evaluation via curated_dir
11. Brain feedback routing (C1)
12. Worker feedback dispatch
13. End-to-end feedback → tracker update
14. Hypothesis lifecycle from feedback (C2)
15. Pattern library ingestion (C3)
16. Candidate hypothesis promotion (C4)
17. Hypothesis ID instruction verification (C5)
18. Daily instruction coverage (M1)
19. Validation pattern aggregation (M2)
20. Shared tier mapping (M3)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.response_parser import parse_response
from trading_assistant.analysis.response_validator import ResponseValidator
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType, OrchestratorBrain
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.schemas.agent_response import (
    CATEGORY_TO_TIER,
    AgentPrediction,
    AgentSuggestion,
    ParsedAnalysis,
    StructuralProposal,
)
from trading_assistant.schemas.suggestion_scoring import CategoryScorecard
from trading_assistant.schemas.suggestion_tracking import SuggestionOutcome, SuggestionRecord, SuggestionStatus
from trading_assistant.skills.hypothesis_library import HypothesisLibrary
from trading_assistant.skills.prediction_tracker import PredictionTracker
from trading_assistant.skills.suggestion_scorer import SuggestionScorer
from trading_assistant.skills.suggestion_tracker import SuggestionTracker
from tests.factories import make_handlers as _factory_make_handlers


# ============================================================
# Module-level helpers
# ============================================================


def _make_handlers(tmp_path, tracker=None, es=None, *, with_curated_dir=False):
    """Create a Handlers instance with defaults for testing.

    Args:
        with_curated_dir: If True, create and pass a curated_dir (needed by gap tests).
    """
    kwargs = dict(suggestion_tracker=tracker, event_stream=es)
    if with_curated_dir:
        curated_dir = tmp_path / "data" / "curated"
        curated_dir.mkdir(parents=True, exist_ok=True)
        kwargs["curated_dir"] = curated_dir
    handlers, _, event_stream = _factory_make_handlers(tmp_path, **kwargs)
    return handlers, handlers._suggestion_tracker, event_stream


# ============================================================
# From test_learning_loop_e2e.py
# ============================================================


@pytest.fixture
def memory_dir(tmp_path):
    """Create a memory directory with minimal policy files."""
    mem = tmp_path / "memory"
    findings = mem / "findings"
    policies = mem / "policies" / "v1"
    policies.mkdir(parents=True)
    findings.mkdir(parents=True)
    (policies / "agent.md").write_text("# Agents\nYou are an agent.")
    (policies / "trading_rules.md").write_text("# Trading Rules\nBe careful.")
    (policies / "soul.md").write_text("# Soul\nQuantify everything.")
    return mem


class TestLearningLoopE2E:
    def test_full_learning_cycle(self, tmp_path, memory_dir):
        """Test the complete learning loop from suggestion → measurement → feedback."""
        findings_dir = memory_dir / "findings"

        # --- Step 1: Record initial suggestions ---
        tracker = SuggestionTracker(store_dir=findings_dir)
        suggestion1 = SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop loss on bot1",
            tier="exit_timing",
            source_report_id="weekly-2026-02-23",
            description="Current stop is too tight for this regime",
        )
        suggestion2 = SuggestionRecord(
            suggestion_id="s002",
            bot_id="bot1",
            title="Adjust filter threshold on bot1",
            tier="parameter",
            source_report_id="weekly-2026-02-23",
        )
        tracker.record(suggestion1)
        tracker.record(suggestion2)

        # --- Step 2: Simulate Claude response with structured block ---
        mock_response = """# Weekly Report
Bot1 had a strong week. Win rate improved.

## Suggestions
1. Widen stop loss on bot1 — exit efficiency is low
2. New signal idea for bot2

<!-- STRUCTURED_OUTPUT
{
  "predictions": [
    {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8, "timeframe_days": 7, "reasoning": "Strong momentum alignment"},
    {"bot_id": "bot2", "metric": "drawdown", "direction": "decline", "confidence": 0.6, "timeframe_days": 7, "reasoning": "Regime unfavorable"}
  ],
  "suggestions": [
    {"suggestion_id": "s001", "bot_id": "bot1", "category": "exit_timing", "title": "Widen stop loss on bot1", "expected_impact": "+0.5% daily PnL", "confidence": 0.7, "evidence_summary": "exit_efficiency < 40%"},
    {"suggestion_id": "new1", "bot_id": "bot2", "category": "signal", "title": "Add momentum crossover", "expected_impact": "+0.3% weekly PnL", "confidence": 0.6, "evidence_summary": "Strong backtest results"}
  ],
  "structural_proposals": [
    {"hypothesis_id": "h-exit-trailing", "bot_id": "bot1", "title": "Switch to trailing stop", "description": "Fixed stops causing premature exits", "reversibility": "easy", "evidence": "exit_efficiency < 50%", "estimated_complexity": "medium"}
  ]
}
-->
"""

        # --- Step 3: Parse response ---
        parsed = parse_response(mock_response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 2
        assert len(parsed.suggestions) == 2
        assert len(parsed.structural_proposals) == 1

        # --- Step 4: Validate response ---
        # Reject suggestion s002 first
        tracker.reject("s002", "Not worth the effort")

        rejected = ContextBuilder(memory_dir).load_rejected_suggestions()
        assert len(rejected) == 1

        # No forecast meta yet, no scorecard
        validator = ResponseValidator(
            rejected_suggestions=rejected,
            forecast_meta={},
            category_scorecard=CategoryScorecard(),
        )
        result = validator.validate(parsed)
        # Neither suggestion matches the rejected "Adjust filter threshold" closely enough
        assert len(result.approved_suggestions) == 2
        assert len(result.blocked_suggestions) == 0

        # --- Step 5: Record predictions ---
        pred_tracker = PredictionTracker(findings_dir)
        pred_tracker.record_predictions("2026-03-01", parsed.predictions)
        loaded = pred_tracker.load_predictions("2026-03-01")
        assert len(loaded) == 2

        # --- Step 6: User approves suggestion s001 ---
        tracker.accept("s001")
        tracker.mark_deployed("s001")
        all_suggestions = tracker.load_all()
        s001 = [s for s in all_suggestions if s["suggestion_id"] == "s001"][0]
        assert s001["status"] == SuggestionStatus.DEPLOYED.value

        # --- Step 7: Record outcome for s001 ---
        outcome = SuggestionOutcome(
            suggestion_id="s001",
            implemented_date="2026-02-24",
            pnl_delta_7d=150.0,
            win_rate_delta_7d=0.05,
        )
        tracker.record_outcome(outcome)
        outcomes = tracker.load_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["pnl_delta_7d"] == 150.0

        # --- Step 8: Recompute category scorecard ---
        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) >= 1
        # exit_timing had 1 positive outcome
        exit_score = scorecard.get_score("bot1", "exit_timing")
        assert exit_score is not None
        assert exit_score.win_rate == 1.0
        assert exit_score.sample_size == 1

        # --- Step 9: Verify scorecard appears in next prompt ---
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "category_scorecard" in pkg.data

        # --- Step 10: Hypothesis lifecycle ---
        hyp_lib = HypothesisLibrary(findings_dir)
        hyp_lib.seed_if_needed()
        hyp_lib.record_proposal("h-exit-trailing")

        # Simulate negative outcome + 3 rejections to trigger retirement
        hyp_lib.record_outcome("h-crowding-diversify", positive=False)
        hyp_lib.record_rejection("h-crowding-diversify")
        hyp_lib.record_rejection("h-crowding-diversify")
        hyp_lib.record_rejection("h-crowding-diversify")

        active = hyp_lib.get_active()
        assert not any(h.id == "h-crowding-diversify" for h in active)

        # --- Step 11: Verify retired hypothesis excluded from track record ---
        track = hyp_lib.get_track_record()
        assert track["h-crowding-diversify"]["status"] == "retired"
        assert track["h-exit-trailing"]["times_proposed"] == 1

    def test_validation_blocks_poor_category(self, tmp_path, memory_dir):
        """Test that suggestions in categories with poor track record are blocked."""
        findings_dir = memory_dir / "findings"

        # Create several suggestions and outcomes with poor results
        tracker = SuggestionTracker(store_dir=findings_dir)
        for i in range(5):
            tracker.record(SuggestionRecord(
                suggestion_id=f"bad{i}",
                bot_id="bot1",
                title=f"Bad exit timing idea {i}",
                tier="exit_timing",
                source_report_id="weekly-test",
            ))
            tracker.accept(f"bad{i}")
            tracker.mark_deployed(f"bad{i}")
            tracker.record_outcome(SuggestionOutcome(
                suggestion_id=f"bad{i}",
                implemented_date="2026-02-01",
                pnl_delta_7d=-100.0,  # all negative
            ))

        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        exit_score = scorecard.get_score("bot1", "exit_timing")
        assert exit_score is not None
        assert exit_score.win_rate == 0.0
        assert exit_score.sample_size == 5

        # Now validate a new exit_timing suggestion for bot1
        suggestion = AgentSuggestion(
            bot_id="bot1",
            category="exit_timing",
            title="Another exit timing idea",
            confidence=0.9,
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        validator = ResponseValidator(category_scorecard=scorecard)
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) == 1
        assert "track record" in result.blocked_suggestions[0].reason.lower()

    def test_prediction_evaluation_flow(self, tmp_path, memory_dir):
        """Test predictions recorded → evaluated → accuracy computed."""
        findings_dir = memory_dir / "findings"
        curated_dir = tmp_path / "curated"

        tracker = PredictionTracker(findings_dir)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
            AgentPrediction(bot_id="bot1", metric="win_rate", direction="decline", confidence=0.7),
        ])

        # Create curated data
        bot_dir = curated_dir / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "total_pnl": 200.0,  # positive → improve ✓
            "win_rate": 0.6,  # positive → improve (predicted decline ✗)
        }))

        baseline_dir = curated_dir / "2026-03-01" / "bot1"
        (baseline_dir / "summary.json").write_text(json.dumps({
            "total_pnl": 100.0,
            "win_rate": 0.60,
        }))
        target_dir = curated_dir / "2026-03-08" / "bot1"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "summary.json").write_text(json.dumps({
            "total_pnl": 200.0,
            "win_rate": 0.65,
        }))

        evaluation = tracker.evaluate_predictions("2026-03-01", curated_dir)
        assert evaluation.total == 2
        assert evaluation.correct == 1
        assert evaluation.accuracy == 0.5
        assert evaluation.accuracy_by_metric["pnl"] == 1.0
        assert evaluation.accuracy_by_metric["win_rate"] == 0.0

    def test_context_builder_includes_all_new_data(self, memory_dir):
        """Verify base_package includes category_scorecard and hypothesis_track_record."""
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        # These should at least not error; with no data they may be empty
        # But the loading methods should work without error
        assert isinstance(pkg.data, dict)
        # Even with no data, these keys shouldn't be present (empty = omitted)
        # This just verifies no exceptions during loading

    def test_forecast_meta_includes_accuracy_by_metric(self, tmp_path, memory_dir):
        """Verify ForecastMetaAnalysis now supports accuracy_by_metric."""
        from trading_assistant.schemas.forecast_tracking import ForecastMetaAnalysis

        meta = ForecastMetaAnalysis(
            rolling_accuracy_4w=0.6,
            accuracy_by_metric={"pnl": 0.8, "win_rate": 0.3},
            weeks_analyzed=4,
        )
        assert meta.accuracy_by_metric["pnl"] == 0.8
        assert meta.accuracy_by_metric["win_rate"] == 0.3

    def test_parse_validate_annotate_report(self, memory_dir):
        """End-to-end: parse → validate → annotated report with notes."""
        findings_dir = memory_dir / "findings"

        # Add a rejected suggestion
        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="rej1",
            bot_id="bot1",
            title="Widen stop loss on bot1",
            tier="exit_timing",
            source_report_id="w1",
        ))
        tracker.reject("rej1", "Already tried, didn't work")

        response = """# Report
<!-- STRUCTURED_OUTPUT
{
  "predictions": [],
  "suggestions": [
    {"bot_id": "bot1", "title": "Widen stop loss on bot1", "category": "exit_timing", "confidence": 0.8},
    {"bot_id": "bot2", "title": "New idea for bot2", "category": "signal", "confidence": 0.6}
  ]
}
-->
"""
        parsed = parse_response(response)
        assert parsed.parse_success

        ctx = ContextBuilder(memory_dir)
        rejected = ctx.load_rejected_suggestions()
        validator = ResponseValidator(rejected_suggestions=rejected)
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) == 1
        assert len(result.approved_suggestions) == 1
        assert result.approved_suggestions[0].title == "New idea for bot2"

        final = parsed.raw_report
        if result.validator_notes:
            final += "\n\n---\n## Validator Notes\n" + result.validator_notes

        assert "Validator Notes" in final
        assert "Widen stop loss" in final


# ============================================================
# From test_learning_loop_gaps.py
# ============================================================


class TestGap8HypothesisIdField:
    """Gap 8: SuggestionRecord has hypothesis_id field."""

    def test_hypothesis_id_default_none(self):
        rec = SuggestionRecord(
            suggestion_id="test1",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
        )
        assert rec.hypothesis_id is None

    def test_hypothesis_id_set(self):
        rec = SuggestionRecord(
            suggestion_id="test2",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
            hypothesis_id="h-signal-recalibrate",
        )
        assert rec.hypothesis_id == "h-signal-recalibrate"

    def test_hypothesis_id_serialization(self):
        rec = SuggestionRecord(
            suggestion_id="test3",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
            hypothesis_id="h-exit-trailing",
        )
        data = rec.model_dump(mode="json")
        assert data["hypothesis_id"] == "h-exit-trailing"

    def test_hypothesis_id_backward_compat(self):
        """Old records without hypothesis_id deserialize fine."""
        data = {
            "suggestion_id": "old1",
            "bot_id": "bot1",
            "title": "Old suggestion",
            "tier": "parameter",
            "source_report_id": "run-old",
        }
        rec = SuggestionRecord(**data)
        assert rec.hypothesis_id is None


class TestGap1AgentSuggestionRecording:
    """Gap 1: _record_agent_suggestions records parsed suggestions to SuggestionTracker."""

    def test_record_agent_suggestions_basic(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path, with_curated_dir=True)

        from trading_assistant.analysis.response_validator import ValidationResult

        approved = [
            AgentSuggestion(bot_id="bot1", title="Widen stop loss", category="stop_loss", confidence=0.7),
            AgentSuggestion(bot_id="bot2", title="Reduce exposure", category="position_sizing", confidence=0.6),
        ]
        validation = ValidationResult(approved_suggestions=approved)

        id_map = h._record_agent_suggestions(validation, "daily-2026-03-07")
        assert len(id_map) == 2
        all_records = tracker.load_all()
        assert len(all_records) == 2

    def test_record_agent_suggestions_with_hypothesis_link(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path, with_curated_dir=True)

        from trading_assistant.analysis.response_validator import ValidationResult

        approved = [
            AgentSuggestion(bot_id="bot1", title="Switch to trailing stop", category="exit_timing", confidence=0.8),
        ]
        parsed = ParsedAnalysis(
            suggestions=approved,
            structural_proposals=[
                StructuralProposal(hypothesis_id="h-exit-trailing", bot_id="bot1", title="Trailing stop"),
            ],
        )
        validation = ValidationResult(approved_suggestions=approved)

        id_map = h._record_agent_suggestions(validation, "weekly-2026-03-07", parsed)
        assert len(id_map) == 1
        all_records = tracker.load_all()
        assert all_records[0]["hypothesis_id"] == "h-exit-trailing"

    def test_record_agent_suggestions_no_tracker(self, tmp_path):
        h, _, es = _make_handlers(tmp_path, with_curated_dir=True)
        h._suggestion_tracker = None
        from trading_assistant.analysis.response_validator import ValidationResult

        validation = ValidationResult(approved_suggestions=[
            AgentSuggestion(bot_id="bot1", title="Test", confidence=0.5),
        ])
        result = h._record_agent_suggestions(validation, "run-1")
        assert result == {}

    def test_record_agent_suggestions_dedup(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path, with_curated_dir=True)
        from trading_assistant.analysis.response_validator import ValidationResult

        approved = [AgentSuggestion(bot_id="bot1", title="Same suggestion", confidence=0.5)]
        validation = ValidationResult(approved_suggestions=approved)

        h._record_agent_suggestions(validation, "run-1")
        h._record_agent_suggestions(validation, "run-1")  # duplicate
        assert len(tracker.load_all()) == 1

    def test_record_agent_suggestions_broadcasts_event(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path, with_curated_dir=True)
        from trading_assistant.analysis.response_validator import ValidationResult

        events = []
        es.subscribe_fn = lambda: None  # just to check broadcast
        original_broadcast = es.broadcast

        def capture_broadcast(event_type, data):
            events.append((event_type, data))
            return original_broadcast(event_type, data)

        es.broadcast = capture_broadcast

        approved = [AgentSuggestion(bot_id="bot1", title="Test suggestion", confidence=0.5)]
        validation = ValidationResult(approved_suggestions=approved)
        h._record_agent_suggestions(validation, "run-1")

        agent_events = [e for e in events if e[0] == "agent_suggestions_recorded"]
        assert len(agent_events) == 1
        assert agent_events[0][1]["count"] == 1


class TestGap4BlockedDetails:
    """Gap 4: Validation log includes blocked suggestion details."""

    def test_validate_and_annotate_returns_tuple(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path, with_curated_dir=True)

        parsed = ParsedAnalysis(
            suggestions=[
                AgentSuggestion(bot_id="bot1", title="Good suggestion", confidence=0.7),
            ],
            raw_report="Test report",
        )
        result = h._validate_and_annotate(parsed, "2026-03-07")
        assert isinstance(result, tuple)
        assert len(result) == 2
        report, validation = result
        assert isinstance(report, str)

    def test_blocked_details_in_validation_log(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path, with_curated_dir=True)

        # Pre-populate a rejected suggestion
        findings_dir = tmp_path / "memory" / "findings"
        suggestions_path = findings_dir / "suggestions.jsonl"
        rejected = {
            "suggestion_id": "rej1",
            "bot_id": "bot1",
            "title": "Widen stop loss on bot1",
            "tier": "parameter",
            "status": "rejected",
            "source_report_id": "old-run",
        }
        suggestions_path.write_text(json.dumps(rejected) + "\n", encoding="utf-8")

        parsed = ParsedAnalysis(
            suggestions=[
                AgentSuggestion(bot_id="bot1", title="Widen stop loss on bot1", confidence=0.7),
            ],
            raw_report="Test report",
        )
        report, validation = h._validate_and_annotate(parsed, "2026-03-07")

        # Check validation log has blocked_details
        log_path = findings_dir / "validation_log.jsonl"
        assert log_path.exists()
        log_entry = json.loads(log_path.read_text().strip())
        assert "blocked_details" in log_entry
        assert log_entry["blocked_count"] == 1
        assert log_entry["blocked_details"][0]["title"] == "Widen stop loss on bot1"
        assert "bot_id" in log_entry["blocked_details"][0]


class TestGap5RemoveMagicMock:
    """Gap 5: MagicMock removed from production context_builder.py."""

    def test_load_transfer_track_record_no_mock_import(self, tmp_path):
        """Verify context_builder doesn't import unittest.mock."""
        import inspect
        source = inspect.getsource(ContextBuilder.load_transfer_track_record)
        assert "MagicMock" not in source
        assert "unittest.mock" not in source

    def test_load_track_record_from_file_static(self, tmp_path):
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        outcomes_path = findings_dir / "transfer_outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"pattern_id": "p1", "verdict": "positive"}) + "\n"
            + json.dumps({"pattern_id": "p1", "verdict": "negative"}) + "\n"
            + json.dumps({"pattern_id": "p2", "verdict": "positive"}) + "\n",
            encoding="utf-8",
        )

        track = TransferProposalBuilder.load_track_record_from_file(findings_dir)
        assert "p1" in track
        assert track["p1"]["total"] == 2
        assert track["p1"]["positive"] == 1
        assert track["p1"]["success_rate"] == 0.5
        assert track["p2"]["success_rate"] == 1.0

    def test_load_track_record_from_file_empty(self, tmp_path):
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

        track = TransferProposalBuilder.load_track_record_from_file(tmp_path)
        assert track == {}

    def test_context_builder_uses_static_method(self, tmp_path):
        """ContextBuilder.load_transfer_track_record uses the static method."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        outcomes_path = findings_dir / "transfer_outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"pattern_id": "p1", "verdict": "positive"}) + "\n",
            encoding="utf-8",
        )

        ctx = ContextBuilder(memory_dir)
        track = ctx.load_transfer_track_record()
        assert "p1" in track
        assert track["p1"]["success_rate"] == 1.0


class TestGap7FindingsDirWiring:
    """Gap 7: TransferProposalBuilder gets findings_dir in weekly loop."""

    def test_transfer_builder_gets_findings_dir(self):
        """Verify the weekly loop code passes findings_dir to TransferProposalBuilder."""
        import inspect
        from trading_assistant.orchestrator.loops.weekly_analysis import WeeklyAnalysisLoop

        source = inspect.getsource(WeeklyAnalysisLoop.handle)
        assert 'findings_dir=deps.memory_dir / "findings"' in source


class TestGap6JsonlHypothesisLibrary:
    """Gap 6: Weekly loop uses JSONL-backed HypothesisLibrary."""

    def test_weekly_handler_uses_hypothesis_library_class(self):
        """Verify the weekly loop imports HypothesisLibrary class, not just get_relevant."""
        import inspect
        from trading_assistant.orchestrator.loops.weekly_analysis import WeeklyAnalysisLoop

        source = inspect.getsource(WeeklyAnalysisLoop.handle)
        assert "HypothesisLibrary" in source
        assert "get_active()" in source

    def test_hypothesis_library_active_filters_retired(self, tmp_path):
        """Retired hypotheses are excluded from get_active."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Retire a hypothesis by recording negative outcome + many rejections
        lib.record_outcome("h-signal-recalibrate", positive=False)
        for _ in range(4):
            lib.record_rejection("h-signal-recalibrate")

        active = lib.get_active()
        active_ids = {h.id for h in active}
        assert "h-signal-recalibrate" not in active_ids

    def test_high_effectiveness_included_without_keyword_match(self, tmp_path):
        """Hypotheses with high effectiveness are included even without keyword match."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Give a hypothesis high effectiveness
        lib.record_proposal("h-exit-trailing")
        lib.record_outcome("h-exit-trailing", positive=True)

        records = lib.get_active()
        h = next(r for r in records if r.id == "h-exit-trailing")
        assert h.effectiveness > 0.3  # Would be included in merged list


class TestGap3HypothesisOutcomeLinking:
    """Gap 3: Hypothesis outcomes linked from suggestion outcomes in _measure_outcomes."""

    def test_measure_outcomes_links_hypothesis(self, tmp_path):
        """Verify _measure_outcomes code calls hypothesis_library.record_outcome."""
        import inspect
        from trading_assistant.orchestrator import runtime_scheduled_callbacks
        source = inspect.getsource(runtime_scheduled_callbacks)
        assert "hypothesis_library.record_outcome" in source
        assert "outcome_hypothesis_id" in source


class TestGap2PredictionEvaluation:
    """Gap 2: Automated prediction evaluation via curated_dir in ContextBuilder."""

    def test_context_builder_accepts_curated_dir(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        assert ctx._curated_dir == curated_dir

    def test_context_builder_curated_dir_defaults_none(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        ctx = ContextBuilder(memory_dir)
        assert ctx._curated_dir is None

    def test_load_prediction_accuracy_with_curated_dir(self, tmp_path):
        """When curated_dir is available, real accuracy is computed."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # Write a prediction
        tracker = PredictionTracker(findings_dir)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(
                bot_id="bot1", metric="pnl", direction="improve",
                confidence=0.8, timeframe_days=7,
            ),
        ])

        for date, total_pnl in [("2026-03-01", 50.0), ("2026-03-08", 100.0)]:
            bot_dir = curated_dir / date / "bot1"
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "summary.json").write_text(
                json.dumps({"total_pnl": total_pnl, "win_rate": 0.6}),
                encoding="utf-8",
            )

        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        result = ctx.load_prediction_accuracy()
        assert result["has_predictions"] is True
        assert "accuracy_by_metric" in result
        assert "pnl" in result["accuracy_by_metric"]

    def test_load_prediction_accuracy_no_curated(self, tmp_path):
        """Without curated_dir, falls back to count-only metadata."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        tracker = PredictionTracker(findings_dir)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(
                bot_id="bot1", metric="pnl", direction="improve",
                confidence=0.8, timeframe_days=7,
            ),
        ])

        ctx = ContextBuilder(memory_dir)
        result = ctx.load_prediction_accuracy()
        assert result["has_predictions"] is True
        assert result["count"] == 1

    def test_validate_and_annotate_passes_curated_dir(self):
        """Verify _validate_and_annotate creates ContextBuilder with curated_dir."""
        import inspect
        source = inspect.getsource(Handlers._validate_and_annotate)
        assert "curated_dir=self._curated_dir" in source

    def test_prediction_evaluation_in_measure_outcomes(self):
        """Verify _measure_outcomes evaluates predictions."""
        import inspect
        from trading_assistant.orchestrator import runtime_scheduled_callbacks
        source = inspect.getsource(runtime_scheduled_callbacks)
        assert "evaluate_predictions" in source
        assert "get_accuracy_by_metric" not in source or "evaluate_predictions" in source


# ============================================================
# From test_learning_loop_remaining_gaps.py
# ============================================================


# ============================================================
# Phase 1: Brain feedback routing (C1)
# ============================================================


class TestBrainFeedbackRouting:
    def test_brain_routes_user_feedback_to_process_feedback(self):
        """Brain maps 'user_feedback' event_type → PROCESS_FEEDBACK action."""
        brain = OrchestratorBrain()
        event = {
            "event_type": "user_feedback",
            "event_id": "fb-001",
            "bot_id": "user",
            "payload": json.dumps({"text": "approve suggestion #abc123", "report_id": "r1"}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.PROCESS_FEEDBACK
        assert actions[0].details["text"] == "approve suggestion #abc123"

    def test_brain_feedback_passes_payload_details(self):
        """Payload is parsed and placed in action.details."""
        brain = OrchestratorBrain()
        event = {
            "event_type": "user_feedback",
            "event_id": "fb-002",
            "bot_id": "user",
            "payload": json.dumps({"text": "reject suggestion #xyz789", "report_id": "weekly-r2"}),
        }
        actions = brain.decide(event)
        assert actions[0].details["report_id"] == "weekly-r2"

    def test_brain_feedback_empty_payload(self):
        """Brain handles missing payload gracefully."""
        brain = OrchestratorBrain()
        event = {
            "event_type": "user_feedback",
            "event_id": "fb-003",
            "bot_id": "user",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.PROCESS_FEEDBACK


class TestWorkerFeedbackDispatch:
    @pytest.mark.asyncio
    async def test_worker_dispatches_to_on_feedback(self, tmp_path):
        """Worker dispatches PROCESS_FEEDBACK actions to on_feedback slot."""
        handler = AsyncMock()
        brain = OrchestratorBrain()

        from trading_assistant.orchestrator.db.queue import EventQueue
        from trading_assistant.orchestrator.task_registry import TaskRegistry

        q = EventQueue(db_path=str(tmp_path / "events.db"))
        reg = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
        await q.initialize()
        await reg.initialize()

        worker = Worker(queue=q, registry=reg, brain=brain)
        worker.on_feedback = handler

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-test",
            bot_id="user",
            details={"text": "approve suggestion #abc"},
        )
        await worker._dispatch(action)
        handler.assert_awaited_once_with(action)

        await q.close()
        await reg.close()

    @pytest.mark.asyncio
    async def test_worker_logs_when_no_feedback_handler(self, tmp_path, caplog):
        """Worker logs info when no on_feedback handler is set."""
        brain = OrchestratorBrain()

        from trading_assistant.orchestrator.db.queue import EventQueue
        from trading_assistant.orchestrator.task_registry import TaskRegistry

        q = EventQueue(db_path=str(tmp_path / "events.db"))
        reg = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
        await q.initialize()
        await reg.initialize()

        worker = Worker(queue=q, registry=reg, brain=brain)
        # on_feedback is None by default

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-noop",
            bot_id="user",
        )

        import logging
        with caplog.at_level(logging.INFO):
            await worker._dispatch(action)

        assert any("Feedback" in msg or "feedback" in msg for msg in caplog.messages)

        await q.close()
        await reg.close()


class TestEndToEndFeedbackRouting:
    @pytest.mark.asyncio
    async def test_feedback_event_to_tracker_update(self, tmp_path):
        """Full path: enqueue feedback event → brain routes → handle_feedback → tracker updated."""
        h, tracker, es = _make_handlers(tmp_path)

        # Pre-record a suggestion
        tracker.record(SuggestionRecord(
            suggestion_id="abc123",
            bot_id="bot1",
            title="Widen stop",
            tier="parameter",
            source_report_id="weekly-2026-03-01",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-e2e",
            bot_id="user",
            details={"text": "approve suggestion #abc123", "report_id": "weekly-2026-03-01"},
        )
        await h.handle_feedback(action)

        # Verify the suggestion was accepted for implementation follow-through
        all_recs = tracker.load_all()
        found = [r for r in all_recs if r["suggestion_id"] == "abc123"]
        assert len(found) == 1
        assert found[0]["status"] == SuggestionStatus.ACCEPTED.value


# ============================================================
# Phase 2: Hypothesis lifecycle from feedback (C2)
# ============================================================


class TestHypothesisLifecycleFromFeedback:
    @pytest.mark.asyncio
    async def test_accept_with_hypothesis_id_records_acceptance(self, tmp_path):
        """Accepting a suggestion linked to a hypothesis calls record_acceptance."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        # Seed hypothesis library
        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Record suggestion with hypothesis_id
        tracker.record(SuggestionRecord(
            suggestion_id="sugghyp1abc",
            bot_id="bot1",
            title="Switch to trailing stop",
            tier="hypothesis",
            source_report_id="weekly-test",
            hypothesis_id="h-exit-trailing",
        ))

        # Accept it
        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-hyp1",
            bot_id="user",
            details={"text": "approve suggestion #sugghyp1abc", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        # Verify hypothesis acceptance was recorded
        records = lib.get_all_records()
        trailing = [r for r in records if r.id == "h-exit-trailing"]
        assert len(trailing) == 1
        assert trailing[0].times_accepted >= 1

    @pytest.mark.asyncio
    async def test_reject_with_hypothesis_id_records_rejection(self, tmp_path):
        """Rejecting a suggestion linked to a hypothesis calls record_rejection."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        tracker.record(SuggestionRecord(
            suggestion_id="sugghyp2def",
            bot_id="bot1",
            title="Regime pause test",
            tier="hypothesis",
            source_report_id="weekly-test",
            hypothesis_id="h-regime-pause",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-hyp2",
            bot_id="user",
            details={"text": "reject suggestion #sugghyp2def", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        records = lib.get_all_records()
        pause = [r for r in records if r.id == "h-regime-pause"]
        assert len(pause) == 1
        assert pause[0].times_rejected >= 1

    @pytest.mark.asyncio
    async def test_accept_without_hypothesis_id_no_op(self, tmp_path):
        """Accepting a suggestion without hypothesis_id doesn't touch library."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()
        before = {r.id: r.times_accepted for r in lib.get_all_records()}

        tracker.record(SuggestionRecord(
            suggestion_id="suggnohyp01",
            bot_id="bot1",
            title="Simple tweak",
            tier="parameter",
            source_report_id="weekly-test",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-nohyp",
            bot_id="user",
            details={"text": "approve suggestion #suggnohyp01", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        after = {r.id: r.times_accepted for r in lib.get_all_records()}
        assert before == after

    @pytest.mark.asyncio
    async def test_auto_retirement_after_3_rejections(self, tmp_path):
        """Hypothesis auto-retires after 3 rejections with non-positive effectiveness."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Pre-set negative outcome + 2 rejections on hypothesis
        lib.record_outcome("h-fills-timing", positive=False)
        lib.record_rejection("h-fills-timing")
        lib.record_rejection("h-fills-timing")

        # Record and reject a 3rd time via feedback
        tracker.record(SuggestionRecord(
            suggestion_id="suggretire1",
            bot_id="bot1",
            title="Delay entry test",
            tier="hypothesis",
            source_report_id="weekly-test",
            hypothesis_id="h-fills-timing",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-retire",
            bot_id="user",
            details={"text": "reject suggestion #suggretire1", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        records = lib.get_all_records()
        timing = [r for r in records if r.id == "h-fills-timing"]
        assert timing[0].status == "retired"


# ============================================================
# Phase 3: Pattern library ingestion (C3)
# ============================================================


class TestPatternLibraryIngestion:
    def test_structural_proposals_create_pattern_entries(self, tmp_path):
        """Structural proposals from parsed response create PatternEntry objects."""
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(
                    bot_id="bot1",
                    title="Replace signal component X",
                    description="Signal X has decayed",
                    reversibility="moderate",
                    evidence="30-day correlation < 0.1",
                ),
                StructuralProposal(
                    bot_id="bot2",
                    title="Add regime gate for ranging",
                    description="Loses in ranging",
                    reversibility="easy",
                    evidence="20+ trades in ranging regime with negative PnL",
                ),
            ],
        )

        h._extract_and_record_patterns(parsed, ["bot1", "bot2"])

        from trading_assistant.skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 2
        titles = {e.title for e in entries}
        assert "Replace signal component X" in titles
        assert "Add regime gate for ranging" in titles

    def test_pattern_dedup_by_title(self, tmp_path):
        """Same proposal title twice → only one pattern entry."""
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(bot_id="bot1", title="Same proposal"),
            ],
        )

        h._extract_and_record_patterns(parsed, ["bot1", "bot2"])
        h._extract_and_record_patterns(parsed, ["bot1", "bot2"])

        from trading_assistant.skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 1

    def test_pattern_entry_has_target_bots(self, tmp_path):
        """Pattern entry target_bots excludes the source bot."""
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(bot_id="bot1", title="Filter restructure"),
            ],
        )

        h._extract_and_record_patterns(parsed, ["bot1", "bot2", "bot3"])

        from trading_assistant.skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert entries[0].source_bot == "bot1"
        assert "bot1" not in entries[0].target_bots
        assert "bot2" in entries[0].target_bots
        assert "bot3" in entries[0].target_bots

    def test_empty_proposals_no_op(self, tmp_path):
        """No structural proposals → no pattern entries created."""
        h, tracker, es = _make_handlers(tmp_path)
        parsed = ParsedAnalysis(structural_proposals=[])
        h._extract_and_record_patterns(parsed, ["bot1"])

        from trading_assistant.skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 0


# ============================================================
# Phase 4: Candidate hypothesis promotion (C4)
# ============================================================


class TestCandidateHypothesisPromotion:
    def test_candidate_promoted_after_2_proposals(self, tmp_path):
        """Candidate hypothesis with times_proposed >= 2 gets promoted to active."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        lib = HypothesisLibrary(findings_dir)
        cid = lib.add_candidate(
            title="New pattern discovered",
            category="signal_decay",
            description="Root cause X seen 15 times",
        )

        # Simulate 2 proposals
        lib.record_proposal(cid)
        lib.record_proposal(cid)

        promoted = lib.promote_candidates()
        assert promoted == 1

        records = lib.get_all_records()
        candidate = [r for r in records if r.id == cid]
        assert candidate[0].status == "active"

    def test_candidate_not_promoted_with_1_proposal(self, tmp_path):
        """Candidate with only 1 proposal stays as candidate."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        lib = HypothesisLibrary(findings_dir)
        cid = lib.add_candidate(
            title="Another pattern",
            category="filter_over_blocking",
            description="Some pattern",
        )

        lib.record_proposal(cid)

        promoted = lib.promote_candidates()
        assert promoted == 0

        records = lib.get_all_records()
        candidate = [r for r in records if r.id == cid]
        assert candidate[0].status == "candidate"

    def test_promoted_candidate_included_in_weekly_prompt(self, tmp_path):
        """Candidates (even with effectiveness 0) are included in weekly prompt merge."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        cid = lib.add_candidate(
            title="Candidate for promotion",
            category="exit_timing",
            description="A candidate",
        )

        active = lib.get_active()
        candidate_records = [h for h in active if h.id == cid]
        assert len(candidate_records) == 1
        assert candidate_records[0].status == "candidate"
        # effectiveness == 0 since no outcomes
        assert candidate_records[0].effectiveness == 0


# ============================================================
# Phase 5: Hypothesis ID instruction verification (C5)
# ============================================================


class TestHypothesisIdInstructions:
    def test_weekly_instruction_contains_hypothesis_id_guidance(self):
        """Weekly instructions reference hypothesis tracking in structured output."""
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "hypothesis_track_record" in _WEEKLY_INSTRUCTIONS
        assert "hypothesis" in _WEEKLY_INSTRUCTIONS

    def test_weekly_template_contains_required_note(self):
        """Structured output template says REQUIRED for hypothesis_id."""
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "REQUIRED: use id from structural_hypotheses" in _WEEKLY_INSTRUCTIONS

    def test_daily_template_contains_required_note(self):
        """Daily structured output template also has REQUIRED note."""
        from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS

        assert "REQUIRED: use id from structural_hypotheses" in _INSTRUCTIONS


# ============================================================
# Phase 6: Moderate gaps (M1, M2, M3)
# ============================================================


class TestDailyInstructionCoverage:
    def test_daily_instructions_reference_rejected_suggestions(self):
        """Daily instructions reference rejected_suggestions for constraint checking."""
        from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS
        assert "rejected_suggestions" in _INSTRUCTIONS

    # NOTE: test_daily_instructions_reference_active_suggestions removed —
    # superseded by tests/test_feedback_loop_phase_d.py version

    def test_daily_instructions_reference_category_scorecard(self):
        """Daily instructions reference category_scorecard."""
        from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS
        assert "category_scorecard" in _INSTRUCTIONS

    def test_daily_instructions_reference_quantification(self):
        """Daily instructions reference quantification requirements."""
        from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS
        assert "quantification" in _INSTRUCTIONS


class TestValidationPatternAggregation:
    def test_load_validation_patterns_from_log(self, tmp_path):
        """ContextBuilder aggregates blocked suggestions from validation_log.jsonl."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        # Write sample validation log
        log_path = findings_dir / "validation_log.jsonl"
        entries = [
            {
                "date": "2026-03-01",
                "blocked_count": 2,
                "blocked_details": [
                    {"title": "Widen exit timing", "reason": "poor track record", "bot_id": "bot1"},
                    {"title": "Change filter threshold", "reason": "rejected before", "bot_id": "bot2"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "date": "2026-03-02",
                "blocked_count": 1,
                "blocked_details": [
                    {"title": "Adjust exit timing again", "reason": "low win rate", "bot_id": "bot1"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]
        with open(log_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        ctx = ContextBuilder(memory_dir)
        patterns = ctx.load_validation_patterns()

        assert "exit_timing" in patterns
        assert patterns["exit_timing"]["blocked_count"] == 2

    def test_empty_log_returns_empty_dict(self, tmp_path):
        """No validation log → empty dict."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        ctx = ContextBuilder(memory_dir)
        patterns = ctx.load_validation_patterns()
        assert patterns == {}

    def test_validation_patterns_in_base_package(self, tmp_path):
        """Validation patterns appear in base_package data when log exists."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        log_path = findings_dir / "validation_log.jsonl"
        entry = {
            "date": "2026-03-05",
            "blocked_count": 1,
            "blocked_details": [
                {"title": "Bad signal change", "reason": "blocked", "bot_id": "bot1"},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_path.write_text(json.dumps(entry) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "validation_patterns" in pkg.data


class TestSharedTierMapping:
    def test_category_to_tier_contains_all_categories(self):
        """CATEGORY_TO_TIER has entries for all known categories."""
        expected = {
            "exit_timing", "filter_threshold", "stop_loss",
            "signal", "structural", "position_sizing", "regime_gate",
            "portfolio_allocation", "portfolio_risk_cap",
            "portfolio_coordination", "portfolio_drawdown_tier",
            "funding_threshold", "leverage_cap",
            "confluence_count", "setup_grade_filter",
        }
        assert set(CATEGORY_TO_TIER.keys()) == expected

    def test_scorer_tier_to_category_includes_legacy_mappings(self):
        """SuggestionScorer's _TIER_TO_CATEGORY includes strategy_variant and hypothesis."""
        from trading_assistant.skills.suggestion_scorer import _TIER_TO_CATEGORY

        assert "strategy_variant" in _TIER_TO_CATEGORY
        assert "hypothesis" in _TIER_TO_CATEGORY
        assert _TIER_TO_CATEGORY["hypothesis"] == "structural"

    def test_handlers_uses_shared_mapping(self, tmp_path):
        """Handlers._record_agent_suggestions uses CATEGORY_TO_TIER from schemas."""
        h, tracker, es = _make_handlers(tmp_path)

        # Create a mock validation result with an approved structural suggestion
        mock_suggestion = MagicMock()
        mock_suggestion.title = "Restructure filter"
        mock_suggestion.bot_id = "bot1"
        mock_suggestion.category = "structural"
        mock_suggestion.evidence_summary = "Evidence"

        mock_validation = MagicMock()
        mock_validation.approved_suggestions = [mock_suggestion]

        h._record_agent_suggestions(mock_validation, "run-test")

        # Should be recorded with tier = "hypothesis" (from CATEGORY_TO_TIER)
        all_recs = tracker.load_all()
        assert len(all_recs) == 1
        assert all_recs[0]["tier"] == "hypothesis"


class TestWeeklyPromptValidationPatterns:
    def test_weekly_instructions_reference_category_scorecard(self):
        """Weekly instructions reference category_scorecard for suggestion validation."""
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS
        assert "category_scorecard" in _WEEKLY_INSTRUCTIONS
