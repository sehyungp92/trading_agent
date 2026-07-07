# tests/test_event_batching.py
"""Tests for QUEUE_FOR_DAILY/WEEKLY event batching in worker."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.orchestrator.worker import Worker


class TestEventBatching:
    def test_worker_tracks_daily_queue_counts(self):
        queue = AsyncMock()
        registry = AsyncMock()
        brain = MagicMock()
        worker = Worker(queue=queue, registry=registry, brain=brain)
        assert hasattr(worker, "daily_queue_counts")
        assert isinstance(worker.daily_queue_counts, dict)

    def test_queue_for_daily_increments_counter(self):
        queue = AsyncMock()
        registry = AsyncMock()
        brain = MagicMock()
        worker = Worker(queue=queue, registry=registry, brain=brain)
        worker._record_queued_event("bot1", ActionType.QUEUE_FOR_DAILY)
        worker._record_queued_event("bot1", ActionType.QUEUE_FOR_DAILY)
        worker._record_queued_event("bot2", ActionType.QUEUE_FOR_DAILY)
        assert worker.daily_queue_counts["bot1"] == 2
        assert worker.daily_queue_counts["bot2"] == 1

    def test_weekly_queue_counts_tracked_separately(self):
        queue = AsyncMock()
        registry = AsyncMock()
        brain = MagicMock()
        worker = Worker(queue=queue, registry=registry, brain=brain)
        worker._record_queued_event("bot1", ActionType.QUEUE_FOR_WEEKLY)
        assert worker.weekly_queue_counts["bot1"] == 1
        assert worker.daily_queue_counts.get("bot1", 0) == 0

    def test_get_and_reset_daily_counts(self):
        queue = AsyncMock()
        registry = AsyncMock()
        brain = MagicMock()
        worker = Worker(queue=queue, registry=registry, brain=brain)
        worker._record_queued_event("bot1", ActionType.QUEUE_FOR_DAILY)
        worker._record_queued_event("bot1", ActionType.QUEUE_FOR_DAILY)
        counts = worker.get_and_reset_daily_counts()
        assert counts == {"bot1": 2}
        assert worker.daily_queue_counts == {}

    def test_get_and_reset_weekly_counts(self):
        queue = AsyncMock()
        registry = AsyncMock()
        brain = MagicMock()
        worker = Worker(queue=queue, registry=registry, brain=brain)
        worker._record_queued_event("bot1", ActionType.QUEUE_FOR_WEEKLY)
        worker._record_queued_event("bot2", ActionType.QUEUE_FOR_WEEKLY)
        counts = worker.get_and_reset_weekly_counts()
        assert counts == {"bot1": 1, "bot2": 1}
        assert worker.weekly_queue_counts == {}


class TestEventBatchingConsumption:
    """Tests that event batching counts are consumed in the daily analysis handler."""

    @pytest.mark.asyncio
    async def test_daily_handler_consumes_event_counts(self):
        """Handlers.handle_daily_analysis calls worker.get_and_reset_daily_counts."""
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream

        mock_runner = MagicMock()
        mock_runner.invoke = AsyncMock(return_value=MagicMock(
            success=True, response="test report", run_dir="/tmp/run",
        ))
        mock_runner.invoke_with_selection = AsyncMock(return_value=(
            MagicMock(success=True, response="test report", run_dir="/tmp/run"),
            "claude_max",
        ))
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch = AsyncMock()
        mock_stream = MagicMock(spec=EventStream)

        worker = MagicMock()
        worker.get_and_reset_daily_counts = MagicMock(return_value={"bot1": 15})

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path as _P
            curated = _P(tmpdir) / "curated"
            # Create trade data so minimum-data threshold passes
            bot_dir = curated / "2026-03-04" / "bot1"
            bot_dir.mkdir(parents=True)
            (bot_dir / "trades.jsonl").write_text(
                '{"trade_id":"t1"}\n{"trade_id":"t2"}\n{"trade_id":"t3"}\n'
            )

            handlers = Handlers(
                agent_runner=mock_runner,
                event_stream=mock_stream,
                dispatcher=mock_dispatcher,
                notification_prefs=MagicMock(),
                curated_dir=str(curated),
                memory_dir=str(_P(tmpdir) / "memory"),
                runs_dir=str(_P(tmpdir) / "runs"),
                source_root=str(_P(tmpdir) / "src"),
                bots=["bot1"],
                worker=worker,
            )

            action = Action(
                type=ActionType.SPAWN_DAILY_ANALYSIS,
                event_id="e1", bot_id="system",
                details={"date": "2026-03-04"},
            )

            # Quality gate will fail with no bots (returns early), which is fine —
            # what matters is that we test the flow. Let's mock the quality gate.
            with patch("trading_assistant.analysis.quality_gate.QualityGate") as MockGate:
                mock_checklist = MagicMock()
                mock_checklist.overall = "PASS"
                MockGate.return_value.run.return_value = mock_checklist
                await handlers.handle_daily_analysis(action)

        worker.get_and_reset_daily_counts.assert_called_once()
