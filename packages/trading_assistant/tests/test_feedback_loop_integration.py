# tests/test_feedback_loop_integration.py
"""End-to-end integration test for the closed feedback loop.

Tests the full lifecycle: generate → record → track → accept/reject → implement →
measure → calibrate.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.schemas.suggestion_tracking import SuggestionOutcome, SuggestionRecord, SuggestionStatus
from trading_assistant.skills.suggestion_tracker import SuggestionTracker
from tests.factories import make_handlers as _factory_make_handlers


def _make_handlers(tmp_path, tracker=None, es=None):
    """Create a Handlers instance with defaults for testing."""
    handlers, _, event_stream = _factory_make_handlers(
        tmp_path, suggestion_tracker=tracker, event_stream=es,
    )
    return handlers, handlers._suggestion_tracker, event_stream


class TestFeedbackLoopIntegration:
    """Full lifecycle integration tests."""

    def test_strategy_engine_to_suggestion_recording(self, tmp_path):
        """1. Strategy engine generates suggestions → 2. Weekly handler records them."""
        h, tracker, es = _make_handlers(tmp_path)

        suggestions = [
            MagicMock(
                title="Widen stop loss on bot1",
                bot_id="bot1",
                tier=MagicMock(value="parameter"),
                description="Stop is too tight, causing premature exits",
            ),
            MagicMock(
                title="Remove filter X on bot2",
                bot_id="bot2",
                tier=MagicMock(value="filter"),
                description="Filter blocking too many good signals",
            ),
        ]

        id_map = h._record_suggestions(suggestions, "weekly-2026-03-01")

        # Verify recorded
        all_recs = tracker.load_all()
        assert len(all_recs) == 2
        assert all(r["status"] == "proposed" for r in all_recs)
        assert len(id_map) == 2

    def test_suggestions_appear_in_daily_prompt(self, tmp_path):
        """3. Recorded suggestions appear in daily prompt via ContextBuilder."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="abc123", bot_id="bot1", title="Widen stop",
            tier="parameter", source_report_id="weekly-2026-03-01",
        ))

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "active_suggestions" in pkg.data
        assert any(s["suggestion_id"] == "abc123" for s in pkg.data["active_suggestions"])

    @pytest.mark.asyncio
    async def test_accept_suggestion_sets_accepted(self, tmp_path):
        """4-5. User accepts suggestion → status transitions to ACCEPTED."""
        h, tracker, es = _make_handlers(tmp_path)

        # Record a suggestion first
        tracker.record(SuggestionRecord(
            suggestion_id="abc123", bot_id="bot1", title="Widen stop",
            tier="parameter", source_report_id="weekly-2026-03-01",
        ))

        # Simulate user feedback
        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test",
            bot_id="system",
            details={"text": "approve suggestion #abc123", "report_id": "weekly-2026-03-01"},
        )
        await h.handle_feedback(action)

        all_recs = tracker.load_all()
        match = [r for r in all_recs if r["suggestion_id"] == "abc123"]
        assert match[0]["status"] == SuggestionStatus.ACCEPTED.value
        assert match[0]["accepted_at"] is not None
        assert match[0]["resolved_at"] is None

    def test_outcome_measurement_on_deployed(self, tmp_path):
        """6. AutoOutcomeMeasurer finds DEPLOYED and measures it."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="abc123", bot_id="bot1", title="Widen stop",
            tier="parameter", source_report_id="weekly-2026-03-01",
        ))
        tracker.accept("abc123")
        tracker.mark_deployed("abc123")

        # Simulate outcome measurement
        outcome = SuggestionOutcome(
            suggestion_id="abc123",
            implemented_date="2026-03-01",
            pnl_delta_7d=150.0,
            win_rate_delta_7d=0.05,
        )
        tracker.record_outcome(outcome)

        # Verify outcome visible via ContextBuilder
        ctx = ContextBuilder(memory_dir)
        outcomes, _low_q = ctx.load_outcome_measurements()
        assert len(outcomes) == 1
        assert outcomes[0]["suggestion_id"] == "abc123"
        assert outcomes[0]["pnl_delta_7d"] == 150.0

    def test_outcome_in_weekly_prompt(self, tmp_path):
        """7. Outcome appears in next weekly prompt."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record_outcome(SuggestionOutcome(
            suggestion_id="abc123",
            implemented_date="2026-03-01",
            pnl_delta_7d=150.0,
        ))

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "outcome_measurements" in pkg.data

    def test_forecast_tracker_meta_analysis(self, tmp_path):
        """8. ForecastTracker computes meta-analysis from retrospective data."""
        from trading_assistant.schemas.forecast_tracking import ForecastRecord
        from trading_assistant.skills.forecast_tracker import ForecastTracker

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        ft = ForecastTracker(findings_dir)
        ft.record_week(ForecastRecord(
            week_start="2026-02-24", week_end="2026-03-02",
            predictions_reviewed=10, correct_predictions=7, accuracy=0.7,
            by_bot={"bot1": 0.8, "bot2": 0.6},
        ))

        meta = ft.compute_meta_analysis()
        assert meta.weeks_analyzed == 1
        assert meta.rolling_accuracy_4w == 0.7
        assert "bot1" in meta.accuracy_by_bot

    @pytest.mark.asyncio
    async def test_rejection_flow(self, tmp_path):
        """9. Generate → reject → appears in rejected_suggestions."""
        h, tracker, es = _make_handlers(tmp_path)
        memory_dir = tmp_path / "memory"

        tracker.record(SuggestionRecord(
            suggestion_id="def456", bot_id="bot2", title="Remove filter X",
            tier="filter", source_report_id="weekly-2026-03-01",
        ))

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test",
            bot_id="system",
            details={"text": "reject suggestion #def456", "report_id": "weekly-2026-03-01"},
        )
        await h.handle_feedback(action)

        ctx = ContextBuilder(memory_dir)
        rejected = ctx.load_rejected_suggestions()
        assert any(r["suggestion_id"] == "def456" for r in rejected)

    def test_dedup_same_week_twice(self, tmp_path):
        """10. Same week twice → no duplicate suggestions."""
        h, tracker, _ = _make_handlers(tmp_path)

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        h._record_suggestions(suggestions, "weekly-2026-03-01")
        h._record_suggestions(suggestions, "weekly-2026-03-01")

        all_recs = tracker.load_all()
        assert len(all_recs) == 1
