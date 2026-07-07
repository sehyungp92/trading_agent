# tests/test_suggestion_tracker.py
"""Tests for SuggestionTracker — records and measures suggestion outcomes."""
import json
import time
from pathlib import Path

from trading_assistant.orchestrator.jsonl_store import write_json_projection
from trading_assistant.schemas.suggestion_tracking import (
    SuggestionRecord,
    SuggestionOutcome,
    SuggestionStatus,
)
from trading_assistant.skills.suggestion_tracker import SuggestionTracker


class TestSuggestionTracker:
    def test_record_suggestion(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        rec = SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop",
            tier="parameter",
            source_report_id="weekly-2026-02-24",
        )
        tracker.record(rec)

        suggestions = tracker.load_all()
        assert len(suggestions) == 1
        assert suggestions[0]["suggestion_id"] == "s001"

    def test_mark_rejected(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        rec = SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop",
            tier="parameter",
            source_report_id="weekly-2026-02-24",
        )
        tracker.record(rec)
        tracker.reject("s001", reason="Not convinced by evidence")

        suggestions = tracker.load_all()
        match = [s for s in suggestions if s["suggestion_id"] == "s001"]
        assert match[0]["status"] == "rejected"
        assert match[0]["rejection_reason"] == "Not convinced by evidence"

    def test_mark_deployed(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        rec = SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop",
            tier="parameter",
            source_report_id="weekly-2026-02-24",
        )
        tracker.record(rec)
        tracker.accept("s001")
        tracker.mark_deployed("s001")

        suggestions = tracker.load_all()
        match = [s for s in suggestions if s["suggestion_id"] == "s001"]
        assert match[0]["status"] == SuggestionStatus.DEPLOYED.value

    def test_index_projection_tracks_lifecycle_updates(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop",
            tier="parameter",
            source_report_id="weekly-2026-02-24",
        ))
        projection_path = Path(str(tmp_path / "suggestions.jsonl") + ".index.json")
        assert projection_path.exists()

        tracker.accept("s001", approval_request_id="req-1")
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        assert projection["s001"]["status"] == SuggestionStatus.ACCEPTED.value
        assert projection["s001"]["approval_request_id"] == "req-1"

    def test_projection_backed_update_large_history_latency(self, tmp_path):
        path = tmp_path / "suggestions.jsonl"
        records = [
            SuggestionRecord(
                suggestion_id=f"s{i}",
                bot_id="bot1",
                title=f"Suggestion {i}",
                tier="parameter",
                source_report_id="daily-2026-06-22",
            ).model_dump(mode="json")
            for i in range(1_500)
        ]
        path.write_text(
            "\n".join(json.dumps(record, default=str) for record in records) + "\n",
            encoding="utf-8",
        )
        write_json_projection(path, key_field="suggestion_id", records=records)

        tracker = SuggestionTracker(store_dir=tmp_path)
        started = time.perf_counter()
        tracker.accept("s1499", approval_request_id="req-1499")
        elapsed = time.perf_counter() - started

        assert elapsed < 3.0
        projection = json.loads((Path(str(path) + ".index.json")).read_text(encoding="utf-8"))
        assert projection["s1499"]["status"] == SuggestionStatus.ACCEPTED.value
        assert projection["s1499"]["approval_request_id"] == "req-1499"

    def test_record_outcome(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        outcome = SuggestionOutcome(
            suggestion_id="s001",
            implemented_date="2026-02-25",
            pnl_delta_7d=120.0,
            win_rate_delta_7d=0.03,
        )
        tracker.record_outcome(outcome)

        outcomes = tracker.load_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["pnl_delta_7d"] == 120.0

    def test_get_rejected_suggestions(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        for i, title in enumerate(["Widen stop", "Remove filter", "Add gate"]):
            tracker.record(SuggestionRecord(
                suggestion_id=f"s{i:03d}",
                bot_id="bot1",
                title=title,
                tier="parameter",
                source_report_id="weekly-2026-02-24",
            ))
        tracker.reject("s000", reason="No evidence")
        tracker.reject("s002", reason="Too risky")

        rejected = tracker.get_rejected(bot_id="bot1")
        assert len(rejected) == 2
        titles = [r["title"] for r in rejected]
        assert "Widen stop" in titles
        assert "Add gate" in titles

    def test_get_rejected_filters_by_bot(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="A",
            tier="parameter", source_report_id="r1",
        ))
        tracker.record(SuggestionRecord(
            suggestion_id="s002", bot_id="bot2", title="B",
            tier="parameter", source_report_id="r1",
        ))
        tracker.reject("s001", reason="x")
        tracker.reject("s002", reason="y")

        assert len(tracker.get_rejected(bot_id="bot1")) == 1
        assert len(tracker.get_rejected(bot_id="bot2")) == 1

    def test_empty_store_returns_empty(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        assert tracker.load_all() == []
        assert tracker.load_outcomes() == []
        assert tracker.get_rejected(bot_id="bot1") == []
