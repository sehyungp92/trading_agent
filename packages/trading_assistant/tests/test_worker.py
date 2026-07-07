import json
import asyncio
from unittest.mock import AsyncMock

import pytest

from trading_assistant.orchestrator.app import create_app
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
from trading_assistant.orchestrator.db.queue import EventQueue
from trading_assistant.orchestrator.subagent import CapacityExceeded
from trading_assistant.orchestrator.task_registry import TaskRegistry
from trading_assistant.schemas.tasks import TaskStatus


@pytest.fixture
async def queue(tmp_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_path / "queue.db"))
    await q.initialize()
    return q


@pytest.fixture
async def registry(tmp_path) -> TaskRegistry:
    r = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
    await r.initialize()
    return r


@pytest.fixture
def brain() -> OrchestratorBrain:
    return OrchestratorBrain()


@pytest.fixture
def worker(queue, registry, brain) -> Worker:
    return Worker(queue=queue, registry=registry, brain=brain)


class TestWorker:
    async def test_process_pending_events(self, worker: Worker, queue: EventQueue):
        await queue.enqueue({
            "event_id": "e001",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "t001"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1

        # Event should be acked
        pending = await queue.peek(limit=10)
        assert len(pending) == 0

    async def test_critical_error_calls_alert_handler(self, worker: Worker, queue: EventQueue):
        worker.on_alert = AsyncMock()

        await queue.enqueue({
            "event_id": "err001",
            "bot_id": "bot3",
            "event_type": "error",
            "payload": json.dumps({"severity": "CRITICAL", "message": "crash"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        await worker.process_batch(limit=10)
        worker.on_alert.assert_called_once()

    async def test_heartbeat_calls_heartbeat_handler(self, worker: Worker, queue: EventQueue):
        worker.on_heartbeat = AsyncMock()

        await queue.enqueue({
            "event_id": "hb001",
            "bot_id": "bot1",
            "event_type": "heartbeat",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        await worker.process_batch(limit=10)
        worker.on_heartbeat.assert_called_once()

    async def test_empty_queue_returns_zero(self, worker: Worker):
        processed = await worker.process_batch(limit=10)
        assert processed == 0

    async def test_capacity_exceeded_requeues_without_consuming_retry(
        self, worker: Worker, queue: EventQueue,
    ):
        async def raise_capacity(action):
            raise CapacityExceeded("no slot")

        worker.on_daily_analysis = raise_capacity

        await queue.enqueue({
            "event_id": "daily-trigger-001",
            "bot_id": "bot1",
            "event_type": "daily_analysis_trigger",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 0  # not counted as processed

        # Event must be back in pending state, not acked, not dead-lettered
        pending = await queue.peek(limit=10)
        assert len(pending) == 1
        assert pending[0]["event_id"] == "daily-trigger-001"
        assert pending[0]["retry_count"] == 0

    async def test_spawned_handler_failure_is_retryable_by_source_event(self, tmp_path):
        app = create_app(
            db_dir=str(tmp_path),
            config=AppConfig(
                bot_ids=["bot1"],
                allow_unauthenticated_local=True,
                bind_host="127.0.0.1",
            ),
        )
        await app.state.queue.initialize()
        await app.state.registry.initialize()

        async def failing_daily_handler(action):
            await asyncio.sleep(0)
            raise RuntimeError("post-spawn boom")

        app.state.daily_analysis_loop.handle = failing_daily_handler
        await app.state.queue.enqueue({
            "event_id": "daily-trigger-linked",
            "bot_id": "bot1",
            "event_type": "daily_analysis_trigger",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await app.state.worker.process_batch(limit=1)
        assert processed == 1
        assert await app.state.queue.peek(limit=10) == []

        linked = []
        for _ in range(20):
            linked = await app.state.registry.list_by_source_event("daily-trigger-linked")
            if linked and linked[0].status == TaskStatus.PENDING:
                break
            await asyncio.sleep(0.01)

        assert len(linked) == 1
        assert linked[0].status == TaskStatus.PENDING
        assert linked[0].retries == 1
        assert linked[0].source_action_type == "spawn_daily_analysis"
        assert "post-spawn boom" in linked[0].error

        await app.state.queue.close()
        await app.state.registry.close()

    async def test_linked_subagent_reconciler_retries_and_completes(self, tmp_path):
        app = create_app(
            db_dir=str(tmp_path),
            config=AppConfig(
                bot_ids=["bot1"],
                allow_unauthenticated_local=True,
                bind_host="127.0.0.1",
            ),
        )
        await app.state.queue.initialize()
        await app.state.registry.initialize()

        attempts = 0

        async def flaky_daily_handler(action):
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(0)
            if attempts == 1:
                raise RuntimeError("retry me")

        app.state.daily_analysis_loop.handle = flaky_daily_handler
        await app.state.queue.enqueue({
            "event_id": "daily-trigger-retry",
            "bot_id": "bot1",
            "event_type": "daily_analysis_trigger",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await app.state.worker.process_batch(limit=1)
        assert processed == 1
        for _ in range(20):
            linked = await app.state.registry.list_by_source_event("daily-trigger-retry")
            if linked and linked[0].status == TaskStatus.PENDING:
                break
            await asyncio.sleep(0.01)

        result = await app.state.reconcile_linked_subagent_tasks()
        assert result["retried"] == 1

        for _ in range(20):
            linked = await app.state.registry.list_by_source_event("daily-trigger-retry")
            if linked and linked[0].status == TaskStatus.COMPLETED:
                break
            await asyncio.sleep(0.01)

        assert attempts == 2
        assert linked[0].status == TaskStatus.COMPLETED
        assert linked[0].retries == 1

        await app.state.queue.close()
        await app.state.registry.close()

    async def test_linked_subagent_reconciler_marks_terminal_after_retry_budget(self, tmp_path):
        app = create_app(
            db_dir=str(tmp_path),
            config=AppConfig(
                bot_ids=["bot1"],
                allow_unauthenticated_local=True,
                bind_host="127.0.0.1",
            ),
        )
        await app.state.queue.initialize()
        await app.state.registry.initialize()

        async def always_failing_daily_handler(action):
            await asyncio.sleep(0)
            raise RuntimeError("still broken")

        app.state.daily_analysis_loop.handle = always_failing_daily_handler
        await app.state.queue.enqueue({
            "event_id": "daily-trigger-terminal",
            "bot_id": "bot1",
            "event_type": "daily_analysis_trigger",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await app.state.worker.process_batch(limit=1)
        assert processed == 1
        for _ in range(20):
            linked = await app.state.registry.list_by_source_event("daily-trigger-terminal")
            if linked and linked[0].status == TaskStatus.PENDING:
                break
            await asyncio.sleep(0.01)

        result = await app.state.reconcile_linked_subagent_tasks()
        assert result["retried"] == 1
        for _ in range(20):
            linked = await app.state.registry.list_by_source_event("daily-trigger-terminal")
            if linked and linked[0].status == TaskStatus.FAILED:
                break
            await asyncio.sleep(0.01)

        assert linked[0].status == TaskStatus.FAILED
        assert linked[0].retries == linked[0].max_retries

        await app.state.queue.close()
        await app.state.registry.close()


class TestPersistRawEvent:
    """Tests for Worker._persist_raw_event bot_id injection."""

    def _make_worker_with_raw_dir(self, tmp_path):
        w = Worker.__new__(Worker)
        w._raw_data_dir = tmp_path / "raw"
        w._raw_data_dir.mkdir()
        w._bot_configs = {}
        return w

    def _make_action(self, bot_id="bot1", details=None):
        from types import SimpleNamespace
        return SimpleNamespace(bot_id=bot_id, details=details or {})

    def test_persist_raw_event_injects_bot_id(self, tmp_path):
        worker = self._make_worker_with_raw_dir(tmp_path)
        action = self._make_action(
            bot_id="bot1",
            details={
                "event_type": "trade",
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                "symbol": "AAPL",
            },
        )
        worker._persist_raw_event(action)

        jsonl_path = list((tmp_path / "raw").rglob("trade.jsonl"))[0]
        record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
        assert record["bot_id"] == "bot1"

    def test_persist_raw_event_preserves_existing_bot_id(self, tmp_path):
        worker = self._make_worker_with_raw_dir(tmp_path)
        action = self._make_action(
            bot_id="bot1",
            details={
                "event_type": "trade",
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                "bot_id": "original_bot",
            },
        )
        worker._persist_raw_event(action)

        jsonl_path = list((tmp_path / "raw").rglob("trade.jsonl"))[0]
        record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
        assert record["bot_id"] == "original_bot"

    def test_persist_raw_event_handles_string_payload(self, tmp_path):
        worker = self._make_worker_with_raw_dir(tmp_path)
        action = self._make_action(
            bot_id="bot1",
            details={
                "event_type": "trade",
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                "payload": "raw_string_data",
            },
        )
        # Should not crash on string payload
        worker._persist_raw_event(action)

        jsonl_path = list((tmp_path / "raw").rglob("trade.jsonl"))[0]
        content = jsonl_path.read_text(encoding="utf-8").strip()
        assert len(content) > 0

    def test_persist_raw_event_no_op_without_raw_dir(self, tmp_path):
        worker = Worker.__new__(Worker)
        worker._raw_data_dir = None
        action = self._make_action(
            bot_id="bot1",
            details={"event_type": "trade"},
        )
        # Should return immediately without error
        worker._persist_raw_event(action)
