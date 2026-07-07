# tests/test_feedback_loops.py
"""Tests for Closed Feedback Loops (Phases A–D).

Phase A: Close the Suggestion Lifecycle
  - path fix, suggestion recording, dedup, accept/reject feedback, IDs in prompts
Phase B: Activate Outcome Feedback Loop
  - prompt instructions, ForecastTracker, wiring into handlers/context_builder
Phase C: Structural Improvement Framework
  - prescriptive consolidation, hypothesis library, cross-bot transfer proposals
Phase D: Daily Context Enrichment
  - active suggestions in daily prompt, lifecycle broadcasts
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.feedback_handler import FeedbackHandler
from trading_assistant.schemas.corrections import CorrectionType
from trading_assistant.schemas.forecast_tracking import AccuracyTrend, ForecastRecord
from trading_assistant.schemas.memory import ConsolidationSummary, PatternCount
from trading_assistant.schemas.suggestion_tracking import SuggestionRecord
from trading_assistant.skills.forecast_tracker import ForecastTracker
from trading_assistant.skills.suggestion_tracker import SuggestionTracker


# =============================================================================
# Phase A: Close the Suggestion Lifecycle
# =============================================================================


# --- A0: Path mismatch fix ---


class TestPathAlignment:
    """Verify tracker and context_builder share the same findings path."""

    def test_tracker_and_context_builder_share_path(self, tmp_path):
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        ContextBuilder(memory_dir)

        # Record a suggestion via tracker
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        # Context builder should see it via load_rejected_suggestions path
        # (same directory)
        assert tracker._suggestions_path.parent == (memory_dir / "findings")
        all_suggestions = tracker.load_all()
        assert len(all_suggestions) == 1

    def test_load_rejected_sees_tracker_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))
        tracker.reject("s001", "Bad idea")

        ctx = ContextBuilder(memory_dir)
        rejected = ctx.load_rejected_suggestions()
        assert len(rejected) == 1
        assert rejected[0]["suggestion_id"] == "s001"

    def test_load_outcome_measurements_sees_tracker_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        from trading_assistant.schemas.suggestion_tracking import SuggestionOutcome

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record_outcome(SuggestionOutcome(
            suggestion_id="s001", implemented_date="2026-03-01",
            pnl_delta_7d=100.0,
        ))

        ctx = ContextBuilder(memory_dir)
        outcomes, _low_q = ctx.load_outcome_measurements()
        assert len(outcomes) == 1
        assert outcomes[0]["suggestion_id"] == "s001"


# --- A1: Handlers accepts suggestion_tracker ---


class TestHandlersSuggestionTracker:
    def test_handlers_with_tracker(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )
        assert h._suggestion_tracker is tracker

    def test_handlers_without_tracker(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
        )
        assert h._suggestion_tracker is None


# --- A2: Suggestion recording in weekly handler ---


class TestRecordSuggestions:
    def test_record_suggestions_creates_records(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        # Create mock StrategySuggestion objects
        suggestions = [
            MagicMock(title="Widen stop", bot_id="bot1", tier=MagicMock(value="parameter"), description="Widen the stop loss"),
            MagicMock(title="Remove filter X", bot_id="bot2", tier=MagicMock(value="filter"), description="Filter too aggressive"),
        ]

        id_map = h._record_suggestions(suggestions, "weekly-2026-03-01")
        assert len(id_map) == 2
        # Verify persisted
        all_recs = tracker.load_all()
        assert len(all_recs) == 2

    def test_record_suggestions_deterministic_ids(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        ids1 = h._record_suggestions(suggestions, "run1")
        # Same run_id + suggestions produce same IDs
        tracker2 = SuggestionTracker(store_dir=tmp_path / "dir2")
        h2 = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker2,
        )
        ids2 = h2._record_suggestions(suggestions, "run1")
        assert list(ids1.keys()) == list(ids2.keys())

    def test_record_suggestions_no_duplicates_on_rerun(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        h._record_suggestions(suggestions, "run1")
        h._record_suggestions(suggestions, "run1")  # re-run

        all_recs = tracker.load_all()
        assert len(all_recs) == 1  # dedup prevents duplicate

    def test_record_suggestions_without_tracker(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
        )
        # Should return empty dict and not crash
        result = h._record_suggestions([MagicMock()], "run1")
        assert result == {}


# --- A3: Deduplication ---


class TestSuggestionDedup:
    def test_same_id_twice_no_duplicate(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        rec = SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        )
        assert tracker.record(rec) is True
        assert tracker.record(rec) is False
        assert len(tracker.load_all()) == 1

    def test_different_ids_both_recorded(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        for sid in ["s001", "s002"]:
            result = tracker.record(SuggestionRecord(
                suggestion_id=sid, bot_id="bot1", title=f"Test {sid}",
                tier="parameter", source_report_id="r1",
            ))
            assert result is True
        assert len(tracker.load_all()) == 2


# --- A4: Accept/reject feedback patterns ---


class TestFeedbackPatterns:
    def test_parse_accept_suggestion(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("approve suggestion #abc123")
        assert correction.correction_type == CorrectionType.SUGGESTION_ACCEPT
        assert correction.target_id == "abc123"

    def test_parse_reject_suggestion(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("reject suggestion #def456")
        assert correction.correction_type == CorrectionType.SUGGESTION_REJECT
        assert correction.target_id == "def456"

    def test_parse_accept_recommendation(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("implement recommendation abc123")
        assert correction.correction_type == CorrectionType.SUGGESTION_ACCEPT
        assert correction.target_id == "abc123"

    def test_parse_decline_suggestion(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("skip suggestion xyz")
        assert correction.correction_type == CorrectionType.SUGGESTION_REJECT
        assert correction.target_id == "xyz"

    @pytest.mark.asyncio
    async def test_handle_feedback_accept_routes_to_tracker(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="abc123", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test-ev",
            bot_id="system",
            details={"text": "approve suggestion #abc123", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        suggestions = tracker.load_all()
        match = [s for s in suggestions if s["suggestion_id"] == "abc123"]
        assert match[0]["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_handle_feedback_reject_routes_to_tracker(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="def456", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test-ev",
            bot_id="system",
            details={"text": "reject suggestion #def456", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        suggestions = tracker.load_all()
        match = [s for s in suggestions if s["suggestion_id"] == "def456"]
        assert match[0]["status"] == "rejected"


# --- A5: Suggestion IDs in weekly instructions ---


class TestWeeklyInstructions:
    def test_instructions_contain_suggestion_id(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "suggestion_id" in _WEEKLY_INSTRUCTIONS

    def test_metadata_populated_after_recording(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Widen stop", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        id_map = h._record_suggestions(suggestions, "run1")
        assert len(id_map) == 1
        # All IDs are 12 hex chars
        for sid in id_map:
            assert len(sid) == 12


# =============================================================================
# Phase B: Activate Outcome Feedback Loop
# =============================================================================


# --- B0 + B1: Instructions ---


class TestOutcomeInstructions:
    def test_weekly_instructions_reference_retrospective(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "RETROSPECTIVE QUESTIONS" in _WEEKLY_INSTRUCTIONS

    def test_weekly_instructions_reference_outcome_measurements(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "outcome_measurements" in _WEEKLY_INSTRUCTIONS

    def test_daily_instructions_reference_outcome_measurements(self):
        from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS

        # Daily focused instructions reference quantified impact and evidence base
        assert "evidence base" in _INSTRUCTIONS


# --- B2: ForecastTracker ---


class TestForecastTracker:
    def test_record_and_load(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        rec = ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
            by_bot={"bot1": 0.8, "bot2": 0.6},
        )
        tracker.record_week(rec)
        loaded = tracker.load_all()
        assert len(loaded) == 1
        assert loaded[0].accuracy == 0.7

    def test_rolling_accuracy(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # 4 weeks of data
        for i in range(4):
            tracker.record_week(ForecastRecord(
                week_start=f"2026-02-{i * 7 + 3:02d}",
                week_end=f"2026-02-{i * 7 + 9:02d}",
                predictions_reviewed=10,
                correct_predictions=6 + i,  # 6, 7, 8, 9 correct
                accuracy=(6 + i) / 10,
            ))
        meta = tracker.compute_meta_analysis()
        assert meta.weeks_analyzed == 4
        # Total correct = 6+7+8+9 = 30 out of 40
        assert meta.rolling_accuracy_4w == 0.75
        assert meta.rolling_accuracy_12w == 0.75  # same since < 12 weeks

    def test_trend_detection_improving(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # Older weeks: low accuracy; recent: high
        for i in range(6):
            tracker.record_week(ForecastRecord(
                week_start=f"2026-01-{i * 7 + 6:02d}",
                week_end=f"2026-01-{i * 7 + 12:02d}",
                predictions_reviewed=10,
                correct_predictions=3 if i < 3 else 8,
                accuracy=0.3 if i < 3 else 0.8,
            ))
        meta = tracker.compute_meta_analysis()
        assert meta.trend == AccuracyTrend.IMPROVING

    def test_trend_detection_degrading(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # Older weeks: high; recent: low
        for i in range(6):
            tracker.record_week(ForecastRecord(
                week_start=f"2026-01-{i * 7 + 6:02d}",
                week_end=f"2026-01-{i * 7 + 12:02d}",
                predictions_reviewed=10,
                correct_predictions=8 if i < 3 else 3,
                accuracy=0.8 if i < 3 else 0.3,
            ))
        meta = tracker.compute_meta_analysis()
        assert meta.trend == AccuracyTrend.DEGRADING

    def test_per_bot_breakdown(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
            by_bot={"bot1": 0.9, "bot2": 0.5},
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.accuracy_by_bot["bot1"] == 0.9
        assert meta.accuracy_by_bot["bot2"] == 0.5

    def test_calibration_computation(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # Very high accuracy → positive calibration (under-confident)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=9,
            accuracy=0.9,
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.calibration_adjustment > 0  # accuracy 0.9 > 0.5

    def test_empty_history_defaults(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        meta = tracker.compute_meta_analysis()
        assert meta.weeks_analyzed == 0
        assert meta.rolling_accuracy_4w == 0.0
        assert meta.trend == AccuracyTrend.STABLE

    def test_single_week_is_stable(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.trend == AccuracyTrend.STABLE


# --- B3: Wiring ---


class TestForecastWiring:
    def test_context_builder_loads_forecast_meta(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        tracker = ForecastTracker(findings_dir)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
        ))

        ctx = ContextBuilder(tmp_path)
        meta = ctx.load_forecast_meta()
        assert meta["weeks_analyzed"] == 1
        assert meta["rolling_accuracy_4w"] == 0.7

    def test_base_package_includes_forecast_meta(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        tracker = ForecastTracker(findings_dir)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
        ))

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "forecast_meta_analysis" in pkg.data

    def test_weekly_instructions_reference_calibration(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "forecast_meta_analysis" in _WEEKLY_INSTRUCTIONS

    def test_base_package_includes_active_suggestions(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "active_suggestions" in pkg.data
        assert len(pkg.data["active_suggestions"]) == 1

    def test_context_builder_no_forecast_on_empty(self, tmp_path):
        (tmp_path / "findings").mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "forecast_meta_analysis" not in pkg.data


# =============================================================================
# Phase C: Structural Improvement Framework
# =============================================================================


# --- C0: Prescriptive consolidation ---


class TestPrescriptiveConsolidation:
    def test_concentration_insight(self, tmp_path):
        from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_bots=[PatternCount(category="bot", key="bot1", count=50)],
        )
        insights = consolidator._generate_insights(summary)
        assert any("Concentration alert" in i for i in insights)
        assert any("bot1" in i for i in insights)

    def test_recurring_issue_insight(self, tmp_path):
        from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_root_causes=[PatternCount(category="root_cause", key="regime_mismatch", count=15)],
        )
        insights = consolidator._generate_insights(summary)
        assert any("Systemic pattern" in i for i in insights)
        assert any("regime_mismatch" in i for i in insights)

    def test_error_pattern_insight(self, tmp_path):
        from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_error_types=[PatternCount(category="error_type", key="IndexError", count=8)],
        )
        insights = consolidator._generate_insights(summary)
        assert any("Blind spot" in i for i in insights)

    def test_no_insights_on_sparse_data(self, tmp_path):
        from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_bots=[PatternCount(category="bot", key="bot1", count=10)],
            top_root_causes=[PatternCount(category="root_cause", key="normal_loss", count=3)],
            top_error_types=[PatternCount(category="error_type", key="ValueError", count=2)],
        )
        insights = consolidator._generate_insights(summary)
        assert len(insights) == 0

    def test_markdown_includes_insights_section(self, tmp_path):
        from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path, threshold=1)

        # Write enough entries to trigger consolidation
        path = tmp_path / "corrections.jsonl"
        entries = []
        for _i in range(5):
            entries.append(json.dumps({
                "bot_id": "bot1", "root_cause": "regime_mismatch",
                "correction_type": "trade_reclassify",
            }))
        # Add more for high root cause count
        for _i in range(10):
            entries.append(json.dumps({
                "bot_id": "bot1", "root_cause": "regime_mismatch",
            }))
        path.write_text("\n".join(entries), encoding="utf-8")

        consolidator.consolidate("corrections.jsonl")

        md_path = tmp_path / "patterns_consolidated.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Actionable Insights" in content


# --- C1: Hypothesis library ---


class TestHypothesisLibrary:
    def test_get_all_returns_hypotheses(self):
        from trading_assistant.skills.hypothesis_library import get_all

        all_h = get_all()
        assert len(all_h) >= 10

    def test_get_by_category_signal_decay(self):
        from trading_assistant.skills.hypothesis_library import get_by_category

        results = get_by_category("signal_decay")
        assert len(results) >= 1
        assert all(h.category == "signal_decay" for h in results)

    def test_get_relevant_with_suggestions(self):
        from trading_assistant.skills.hypothesis_library import get_relevant

        suggestions = [
            MagicMock(title="Signal decay on bot1", description="Win rate declining"),
            MagicMock(title="Filter over-blocking", description="Too many missed trades"),
        ]
        results = get_relevant(suggestions)
        categories = {h.category for h in results}
        assert "signal_decay" in categories
        assert "filter_over_blocking" in categories

    def test_get_relevant_empty_on_unknown(self):
        from trading_assistant.skills.hypothesis_library import get_relevant

        suggestions = [MagicMock(title="Unknown thing", description="Something unrelated")]
        results = get_relevant(suggestions)
        assert len(results) == 0

    def test_no_duplicate_hypotheses(self):
        from trading_assistant.skills.hypothesis_library import get_relevant

        # Multiple suggestions pointing to same category
        suggestions = [
            MagicMock(title="Signal decay A", description=""),
            MagicMock(title="Signal alpha B", description=""),
        ]
        results = get_relevant(suggestions)
        ids = [h.id for h in results]
        assert len(ids) == len(set(ids))

    def test_weekly_instructions_reference_hypotheses(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "structural_hypotheses" in _WEEKLY_INSTRUCTIONS
        assert "hypothes" in _WEEKLY_INSTRUCTIONS


# --- C2: Transfer proposals ---


class TestTransferProposals:
    def _make_pattern_library(self, tmp_path, patterns):
        from trading_assistant.skills.pattern_library import PatternLibrary

        lib = PatternLibrary(tmp_path)
        for p in patterns:
            lib.add(p)
        return lib

    def test_proposed_patterns_included_in_proposals(self, tmp_path):
        """PROPOSED patterns are now included in transfer proposals (not just VALIDATED)."""
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.PROPOSED, source_bot="bot1",
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2"])
        proposals = builder.build_proposals()
        assert len(proposals) == 1
        assert proposals[0].target_bot == "bot2"

    def test_source_excluded_from_proposals(self, tmp_path):
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED, source_bot="bot1",
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2"])
        proposals = builder.build_proposals()
        target_bots = [p.target_bot for p in proposals]
        assert "bot1" not in target_bots  # source bot excluded

    def test_already_targeted_excluded(self, tmp_path):
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED, source_bot="bot1",
            target_bots=["bot2"],  # already targeted
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2", "bot3"])
        proposals = builder.build_proposals()
        target_bots = [p.target_bot for p in proposals]
        assert "bot2" not in target_bots
        assert "bot3" in target_bots

    def test_sorted_by_score(self, tmp_path):
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test A", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED, source_bot="bot1",
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2", "bot3"])
        proposals = builder.build_proposals()
        if len(proposals) >= 2:
            assert proposals[0].compatibility_score >= proposals[1].compatibility_score

    def test_weekly_instructions_reference_transfer(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "CROSS-BOT TRANSFER" in _WEEKLY_INSTRUCTIONS
        assert "transfer_proposals" in _WEEKLY_INSTRUCTIONS

    def test_eligible_bots_only(self, tmp_path):
        from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder
        from trading_assistant.skills.pattern_library import PatternLibrary
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.IMPLEMENTED, source_bot="bot1",
        ))

        # Only bot2 and bot3 are eligible
        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2", "bot3"])
        proposals = builder.build_proposals()
        assert len(proposals) == 2
        assert {p.target_bot for p in proposals} == {"bot2", "bot3"}


# =============================================================================
# Phase D: Daily Context Enrichment
# =============================================================================


# --- D0: Active suggestions in daily prompt ---


class TestActiveSuggestionsInDaily:
    def test_loads_only_non_rejected(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        tracker = SuggestionTracker(store_dir=findings_dir)
        # proposed
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Active one",
            tier="parameter", source_report_id="r1",
        ))
        # rejected
        tracker.record(SuggestionRecord(
            suggestion_id="s002", bot_id="bot1", title="Rejected one",
            tier="parameter", source_report_id="r1",
        ))
        tracker.reject("s002", "Bad idea")
        # deployed
        tracker.record(SuggestionRecord(
            suggestion_id="s003", bot_id="bot1", title="Deployed one",
            tier="parameter", source_report_id="r1",
        ))
        tracker.mark_deployed("s003")

        ctx = ContextBuilder(tmp_path)
        active = ctx.load_active_suggestions()
        ids = [s["suggestion_id"] for s in active]
        assert "s001" in ids  # proposed
        assert "s002" not in ids  # rejected
        assert "s003" in ids  # deployed

    def test_applies_temporal_window(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        tracker = SuggestionTracker(store_dir=findings_dir)
        # Create 40 suggestions (cap is 30)
        for i in range(40):
            tracker.record(SuggestionRecord(
                suggestion_id=f"s{i:03d}", bot_id="bot1", title=f"Test {i}",
                tier="parameter", source_report_id="r1",
            ))

        ctx = ContextBuilder(tmp_path)
        active = ctx.load_active_suggestions()
        assert len(active) <= 30

    def test_in_base_package(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "active_suggestions" in pkg.data

    def test_daily_instructions_reference_active_suggestions(self):
        from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS

        assert "active_suggestions" in _INSTRUCTIONS
        assert "DEPLOYED" in _INSTRUCTIONS


# --- D1: Lifecycle broadcasts ---


class TestLifecycleBroadcasts:
    @pytest.mark.asyncio
    async def test_accept_broadcasts_event(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="abc123", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)

        es = EventStream()
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=es,
            dispatcher=AsyncMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test",
            bot_id="system",
            details={"text": "approve suggestion #abc123", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        recent = es.get_recent()
        types = [e.event_type for e in recent]
        assert "suggestion_accepted" in types

    @pytest.mark.asyncio
    async def test_reject_broadcasts_event(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType

        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="def456", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)

        es = EventStream()
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=es,
            dispatcher=AsyncMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test",
            bot_id="system",
            details={"text": "reject suggestion #def456", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        recent = es.get_recent()
        types = [e.event_type for e in recent]
        assert "suggestion_rejected" in types

    def test_record_suggestions_broadcasts(self, tmp_path):
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        es = EventStream()
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=es,
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        h._record_suggestions(suggestions, "run1")
        recent = es.get_recent()
        types = [e.event_type for e in recent]
        assert "suggestions_recorded" in types
