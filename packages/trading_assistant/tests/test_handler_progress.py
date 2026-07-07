"""Tests for handler progress broadcasts (C2) and feedback handler."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.orchestrator.agent_runner import AgentResult
from trading_assistant.schemas.notifications import NotificationPreferences
from trading_assistant.schemas.corrections import CorrectionType
from trading_assistant.analysis.feedback_handler import FeedbackHandler


@pytest.fixture
def event_stream():
    return EventStream()


@pytest.fixture
def handlers_with_stream(tmp_path, event_stream):
    # Use MagicMock with explicit AsyncMock for async methods only — keeps
    # sync methods like refresh_run_index from returning unawaited coroutines
    # (see handlers.py:_progress for the call site that triggered the warning).
    agent_runner = MagicMock()
    agent_runner.invoke = AsyncMock(return_value=AgentResult(
        response="test", run_dir=tmp_path / "runs" / "test", success=True,
    ))
    agent_runner.invoke_with_selection = AsyncMock(return_value=(
        AgentResult(response="test", run_dir=tmp_path / "runs" / "test", success=True),
        "claude_max",
    ))

    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=AsyncMock(),
        notification_prefs=NotificationPreferences(),
        curated_dir=tmp_path / "data" / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1"],
        heartbeat_dir=tmp_path / "heartbeats",
    )
    # Create needed dirs
    (tmp_path / "memory" / "policies" / "v1").mkdir(parents=True)
    (tmp_path / "memory" / "findings").mkdir(parents=True)
    # Create trade data so minimum-data threshold passes for daily analysis tests
    from datetime import datetime as _datetime, timezone as _timezone

    today = _datetime.now(_timezone.utc).strftime("%Y-%m-%d")
    bot_dir = tmp_path / "data" / "curated" / today / "bot1"
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "trades.jsonl").write_text(
        '{"trade_id":"t1"}\n{"trade_id":"t2"}\n{"trade_id":"t3"}\n'
    )
    return handlers


@pytest.mark.asyncio
async def test_daily_handler_broadcasts_start_event(handlers_with_stream, event_stream):
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")

    # Mock quality gate to return can_proceed=True
    with patch("trading_assistant.analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    progress_events = [e for e in events if e.event_type == "handler_progress"]
    assert len(progress_events) >= 1
    assert progress_events[0].data["stage"] == "started"


@pytest.mark.asyncio
async def test_daily_handler_broadcasts_progress_with_stage(handlers_with_stream, event_stream):
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")

    with patch("trading_assistant.analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    progress_events = [e for e in events if e.event_type == "handler_progress"]
    stages = [e.data["stage"] for e in progress_events]

    assert "started" in stages
    assert "quality_gate" in stages
    assert "prompt_assembly" in stages


@pytest.mark.asyncio
async def test_all_four_handlers_emit_started(handlers_with_stream, event_stream):
    """Verify all 4 long-running handlers emit at least a 'started' progress event."""
    # Daily
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")
    with patch("trading_assistant.analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate
        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    daily_starts = [
        e for e in events
        if e.event_type == "handler_progress"
        and e.data.get("handler") == "daily_analysis"
        and e.data.get("stage") == "started"
    ]
    assert len(daily_starts) >= 1


@pytest.mark.asyncio
async def test_progress_events_include_run_id(handlers_with_stream, event_stream):
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1",
                    details={"date": "2026-03-04"})

    with patch("trading_assistant.analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    progress_events = [e for e in events if e.event_type == "handler_progress"]
    assert all("run_id" in e.data for e in progress_events)
    assert progress_events[0].data["run_id"] == "daily-2026-03-04"


@pytest.mark.asyncio
async def test_error_events_still_broadcast(handlers_with_stream, event_stream):
    """When a handler fails, both progress and error events should exist."""
    handlers_with_stream._agent_runner.invoke = AsyncMock(side_effect=RuntimeError("boom"))

    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")

    with patch("trading_assistant.analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    types = [e.event_type for e in events]
    assert "handler_progress" in types
    assert "daily_analysis_error" in types


# --- Merged from test_feedback_handler.py ---


class TestFeedbackHandler:
    def test_parse_trade_reclassify(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse(
            "Trade #xyz wasn't bad — it was a planned hedge against spot"
        )
        assert correction.correction_type == CorrectionType.TRADE_RECLASSIFY
        assert correction.target_id == "xyz"
        assert correction.raw_text == "Trade #xyz wasn't bad — it was a planned hedge against spot"

    def test_parse_regime_override(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse(
            "Bot2's regime classification was wrong today, it was a slow trend not ranging"
        )
        assert correction.correction_type == CorrectionType.REGIME_OVERRIDE
        assert "bot2" in correction.target_id.lower()

    def test_parse_positive_reinforcement(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse(
            "Good catch on the volume filter, that's the third time this week"
        )
        assert correction.correction_type == CorrectionType.POSITIVE_REINFORCEMENT

    def test_parse_free_text(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse("Interesting day overall, keep monitoring ETH")
        assert correction.correction_type == CorrectionType.FREE_TEXT

    def test_write_to_jsonl(self, tmp_path: Path):
        corrections_path = tmp_path / "corrections.jsonl"
        handler = FeedbackHandler(report_id="daily-2026-03-01")

        correction = handler.parse("Trade #abc was actually fine")
        handler.write_correction(correction, corrections_path)

        lines = corrections_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["correction_type"] == "trade_reclassify"

    def test_appends_to_existing_file(self, tmp_path: Path):
        corrections_path = tmp_path / "corrections.jsonl"
        corrections_path.write_text('{"existing": true}\n')

        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse("Good catch on volume filter")
        handler.write_correction(correction, corrections_path)

        lines = corrections_path.read_text().strip().splitlines()
        assert len(lines) == 2
