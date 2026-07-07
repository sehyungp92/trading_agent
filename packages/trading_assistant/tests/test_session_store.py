"""Tests for session persistence (H5)."""

from __future__ import annotations

import pytest

from trading_assistant.orchestrator.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(base_dir=tmp_path / "sessions")


class TestRecordInvocation:
    def test_creates_session_file(self, store, tmp_path):
        record = store.record_invocation(
            session_id="sess-1",
            agent_type="daily_analysis",
            prompt_package={"system_prompt": "test"},
            response="Analysis complete.",
            token_usage={"input_tokens": 100, "output_tokens": 50},
            duration_ms=1500,
        )
        assert record.session_id == "sess-1"
        assert record.agent_type == "daily_analysis"
        assert record.duration_ms == 1500
        # File should exist
        date_str = record.timestamp.strftime("%Y-%m-%d")
        path = tmp_path / "sessions" / "daily_analysis" / date_str / "sessions.jsonl"
        assert path.exists()

    def test_prompt_hash_is_deterministic(self, store):
        pkg = {"system_prompt": "hello", "data": {"x": 1}}
        r1 = store.record_invocation("s1", "daily_analysis", pkg, "resp1")
        r2 = store.record_invocation("s2", "daily_analysis", pkg, "resp2")
        assert r1.prompt_hash == r2.prompt_hash

    def test_response_truncated_to_500(self, store):
        long_resp = "x" * 1000
        record = store.record_invocation("s1", "daily_analysis", {}, long_resp)
        assert len(record.response_summary) == 500

    def test_appends_to_existing_file(self, store):
        store.record_invocation("s1", "daily_analysis", {}, "resp1")
        store.record_invocation("s2", "daily_analysis", {}, "resp2")
        sessions = store.list_sessions(agent_type="daily_analysis")
        assert len(sessions) == 2


class TestGetSession:
    def test_returns_matching_records(self, store):
        store.record_invocation("s1", "daily_analysis", {}, "resp1")
        store.record_invocation("s2", "daily_analysis", {}, "resp2")
        r1 = store.record_invocation("s1", "daily_analysis", {"step": 2}, "resp3")

        date_str = r1.timestamp.strftime("%Y-%m-%d")
        records = store.get_session("s1", "daily_analysis", date_str)
        assert len(records) == 2
        assert all(r.session_id == "s1" for r in records)

    def test_returns_empty_for_nonexistent(self, store):
        records = store.get_session("nope", "daily_analysis", "2026-01-01")
        assert records == []


class TestListSessions:
    def test_lists_all_sessions(self, store):
        store.record_invocation("s1", "daily_analysis", {}, "r")
        store.record_invocation("s2", "weekly_analysis", {}, "r")
        sessions = store.list_sessions()
        assert len(sessions) == 2

    def test_filters_by_agent_type(self, store):
        store.record_invocation("s1", "daily_analysis", {}, "r")
        store.record_invocation("s2", "weekly_analysis", {}, "r")
        sessions = store.list_sessions(agent_type="daily_analysis")
        assert len(sessions) == 1
        assert sessions[0]["agent_type"] == "daily_analysis"

    def test_list_sessions_fresh_install(self, tmp_path):
        """list_sessions should return empty list when base_dir doesn't exist."""
        store = SessionStore(base_dir=tmp_path / "nonexistent" / "sessions")
        sessions = store.list_sessions()
        assert sessions == []

    def test_includes_optional_runtime_metadata(self, store):
        store.record_invocation(
            "s1",
            "daily_analysis",
            {},
            "r",
            metadata={
                "provider": "claude_max",
                "effective_model": "sonnet",
                "first_output_ms": 140,
                "tool_call_count": 3,
            },
        )

        sessions = store.list_sessions(agent_type="daily_analysis")

        assert sessions[0]["provider"] == "claude_max"
        assert sessions[0]["effective_model"] == "sonnet"
        assert sessions[0]["first_output_ms"] == 140
        assert sessions[0]["tool_call_count"] == 3
