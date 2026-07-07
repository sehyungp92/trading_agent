"""Tests for stream parsing and output parsing in AgentRunner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.stream_parser import StreamState
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.session_store import SessionStore


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def event_stream() -> EventStream:
    return EventStream()


@pytest.fixture
def runner(tmp_path: Path, session_store: SessionStore, event_stream: EventStream) -> AgentRunner:
    return AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        event_stream=event_stream,
    )


def _make_state() -> StreamState:
    return StreamState()


# ---------------------------------------------------------------------------
# Claude Stream Line Parsing
# ---------------------------------------------------------------------------


class TestParseClaudeStreamLine:
    def test_result_event_captured(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({"type": "result", "result": "Analysis complete."})
        runner._stream_parser.parse_claude_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert state.final_result == "Analysis complete."

    def test_session_id_captured(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({"session_id": "sess-abc123"})
        runner._stream_parser.parse_claude_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert state.resolved_session_id == "sess-abc123"

    def test_thread_id_also_captures_session(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({"thread_id": "thread-xyz"})
        runner._stream_parser.parse_claude_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert state.resolved_session_id == "thread-xyz"

    def test_cost_captured(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({"cost_usd": 0.042, "duration_ms": 5000})
        runner._stream_parser.parse_claude_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert state.cost_usd == pytest.approx(0.042)

    def test_cost_from_usage_block(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({"usage": {"total_cost_usd": 0.08}})
        runner._stream_parser.parse_claude_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert state.cost_usd == pytest.approx(0.08)

    def test_tool_use_increments_count(self, runner: AgentRunner):
        state = _make_state()
        msg = {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read"}]},
        }
        runner._stream_parser.parse_claude_stream_line(
            line=json.dumps(msg), state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert state.tool_call_count == 1

    def test_text_content_appended(self, runner: AgentRunner):
        state = _make_state()
        msg = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Here is the report."}],
            },
        }
        runner._stream_parser.parse_claude_stream_line(
            line=json.dumps(msg), state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        assert "Here is the report." in state.texts

    def test_invalid_json_handled_gracefully(self, runner: AgentRunner):
        state = _make_state()
        runner._stream_parser.parse_claude_stream_line(
            line="not json at all", state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        # Should not raise — non-JSON lines are broadcast as raw

    def test_non_dict_json_ignored(self, runner: AgentRunner):
        state = _make_state()
        runner._stream_parser.parse_claude_stream_line(
            line=json.dumps([1, 2, 3]), state=state,
            run_id="r1", agent_type="daily", provider="claude_max", runtime="claude_cli",
        )
        # Non-dict JSON is silently ignored; final_result stays at default ""
        assert state.final_result == ""
        assert len(state.texts) == 0


# ---------------------------------------------------------------------------
# Codex Stream Line Parsing
# ---------------------------------------------------------------------------


class TestParseCodexStreamLine:
    def test_message_item_text_captured(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({
            "item": {"type": "message", "text": "Codex analysis."},
        })
        runner._stream_parser.parse_codex_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="codex_pro", runtime="codex_cli",
        )
        assert "Codex analysis." in state.texts

    def test_tool_item_increments_count(self, runner: AgentRunner):
        state = _make_state()
        line = json.dumps({
            "item": {"type": "tool_call", "name": "shell"},
        })
        runner._stream_parser.parse_codex_stream_line(
            line=line, state=state,
            run_id="r1", agent_type="daily", provider="codex_pro", runtime="codex_cli",
        )
        assert state.tool_call_count >= 1


# ---------------------------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------------------------


class TestParseOutput:
    def test_json_output_extracts_result(self, runner: AgentRunner, tmp_path: Path):
        stdout = json.dumps({"result": "Report text.", "cost_usd": 0.1, "duration_ms": 500})
        result = runner._stream_parser._parse_json_output(
            stdout=stdout, run_dir=tmp_path, elapsed_ms=600,
            session_id="s1", provider="claude_max", runtime="claude_cli",
            requested_model=None, effective_model="sonnet", auth_mode="claude.ai",
        )
        assert result["response"] == "Report text."
        assert result["cost_usd"] == pytest.approx(0.1)

    def test_json_output_fallback_on_invalid(self, runner: AgentRunner, tmp_path: Path):
        result = runner._stream_parser._parse_json_output(
            stdout="plain text output", run_dir=tmp_path, elapsed_ms=100,
            session_id="s1", provider="claude_max", runtime="claude_cli",
            requested_model=None, effective_model="sonnet", auth_mode="claude.ai",
        )
        assert result["response"] == "plain text output"
        assert result["success"] is True

    def test_jsonl_output_extracts_messages(self, runner: AgentRunner, tmp_path: Path):
        lines = [
            json.dumps({"item": {"type": "message", "text": "First."}}),
            json.dumps({"item": {"type": "message", "text": "Second."}}),
            json.dumps({"usage": {"total_cost_usd": 0.05}}),
        ]
        stdout = "\n".join(lines)
        result = runner._stream_parser._parse_jsonl_output(
            stdout=stdout, run_dir=tmp_path, elapsed_ms=200,
            session_id="s1", provider="codex_pro", runtime="codex_cli",
            requested_model=None, effective_model="gpt-5.4", auth_mode="chatgpt",
        )
        assert "First." in result["response"]
        assert "Second." in result["response"]
        assert result["cost_usd"] == pytest.approx(0.05)

    def test_jsonl_output_captures_thread_id(self, runner: AgentRunner, tmp_path: Path):
        lines = [
            json.dumps({"thread_id": "thread-xyz"}),
            json.dumps({"item": {"type": "message", "text": "done"}}),
        ]
        result = runner._stream_parser._parse_jsonl_output(
            stdout="\n".join(lines), run_dir=tmp_path, elapsed_ms=100,
            session_id="s1", provider="codex_pro", runtime="codex_cli",
            requested_model=None, effective_model="gpt-5.4", auth_mode="chatgpt",
        )
        assert result["session_id"] == "thread-xyz"

    def test_parse_output_routes_to_json(self, runner: AgentRunner, tmp_path: Path):
        stdout = json.dumps({"result": "routed"})
        result = runner._stream_parser.parse_output(
            stdout=stdout, run_dir=tmp_path, elapsed_ms=50,
            session_id="s", parse_mode="stream-json",
            provider="claude_max", runtime="claude_cli",
            requested_model=None, effective_model="sonnet", auth_mode="claude.ai",
        )
        assert result["response"] == "routed"

    def test_parse_output_routes_to_jsonl(self, runner: AgentRunner, tmp_path: Path):
        line = json.dumps({"item": {"type": "message", "text": "jsonl-routed"}})
        result = runner._stream_parser.parse_output(
            stdout=line, run_dir=tmp_path, elapsed_ms=50,
            session_id="s", parse_mode="jsonl",
            provider="codex_pro", runtime="codex_cli",
            requested_model=None, effective_model="gpt-5.4", auth_mode="chatgpt",
        )
        assert "jsonl-routed" in result["response"]

    def test_jsonl_tool_calls_counted(self, runner: AgentRunner, tmp_path: Path):
        lines = [
            json.dumps({"item": {"type": "tool_call", "name": "shell"}}),
            json.dumps({"item": {"type": "tool_call", "name": "read"}}),
            json.dumps({"item": {"type": "message", "text": "done"}}),
        ]
        result = runner._stream_parser._parse_jsonl_output(
            stdout="\n".join(lines), run_dir=tmp_path, elapsed_ms=100,
            session_id="s1", provider="codex_pro", runtime="codex_cli",
            requested_model=None, effective_model="gpt-5.4", auth_mode="chatgpt",
        )
        assert result["tool_call_count"] == 2
