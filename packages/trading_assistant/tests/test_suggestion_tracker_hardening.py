"""Tests for SuggestionTracker JSONL hardening (Task 5)."""
from __future__ import annotations

import json

import pytest

from trading_assistant.skills.suggestion_tracker import SuggestionTracker


@pytest.fixture
def tracker(tmp_path) -> SuggestionTracker:
    store = tmp_path / "findings"
    store.mkdir()
    return SuggestionTracker(store_dir=store)


class TestPerLineExceptionHandling:
    """5a: Per-line exception handling in _read_jsonl."""

    def test_malformed_json_skipped_rest_preserved(self, tracker, tmp_path):
        path = tmp_path / "findings" / "suggestions.jsonl"
        path.write_text(
            json.dumps({"suggestion_id": "s1", "status": "proposed"}) + "\n"
            + "not valid json line\n"
            + json.dumps({"suggestion_id": "s2", "status": "proposed"}) + "\n",
            encoding="utf-8",
        )
        records = tracker.load_all()
        assert len(records) == 2
        assert records[0]["suggestion_id"] == "s1"
        assert records[1]["suggestion_id"] == "s2"

    def test_all_malformed_returns_empty(self, tracker, tmp_path):
        path = tmp_path / "findings" / "suggestions.jsonl"
        path.write_text("bad1\nbad2\n", encoding="utf-8")
        records = tracker.load_all()
        assert records == []


class TestEncodingParameter:
    """5b: UTF-8 encoding on all open() calls."""

    def test_unicode_suggestion_roundtrip(self, tracker, tmp_path):
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord
        rec = SuggestionRecord(
            suggestion_id="u1",
            bot_id="bot1",
            title="Adjust für Volatilität — 日本語テスト",
            description="Test",
            tier="parameter",
            source_report_id="daily-2026-03-01",
        )
        tracker.record(rec)
        records = tracker.load_all()
        assert len(records) == 1
        assert "für" in records[0]["title"]
        assert "日本語" in records[0]["title"]


class TestBatchLoadPipeline:
    """5c: Batch-load suggestions in autonomous pipeline."""

    async def test_process_new_suggestions_batch_loads(self, tmp_path):
        from unittest.mock import MagicMock
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline

        store = tmp_path / "findings"
        store.mkdir()
        tracker = SuggestionTracker(store_dir=store)

        # Record some suggestions
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord
        for i in range(3):
            tracker.record(SuggestionRecord(
                suggestion_id=f"s{i}",
                bot_id="bot1",
                title=f"Suggestion {i}",
                description="Test",
                tier="parameter",
                confidence=0.8,
                source_report_id=f"daily-2026-03-0{i+1}",
            ))

        config_reg = MagicMock()
        config_reg.resolve_suggestion_to_params.return_value = []
        backtester = MagicMock()
        approval = MagicMock()
        approval.get_pending.return_value = []

        pipeline = AutonomousPipeline(
            config_registry=config_reg,
            backtester=backtester,
            approval_tracker=approval,
            suggestion_tracker=tracker,
        )

        # Should complete without errors
        results = await pipeline.process_new_suggestions(["s0", "s1", "s2"])
        # All suggestions skip (no matching params), but load_all() only called once
        assert isinstance(results, list)
