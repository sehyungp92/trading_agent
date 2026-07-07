# tests/test_comms_orchestrator_wiring.py
"""Tests for Phase 6 orchestrator wiring."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


class TestBrainNotificationRouting:
    def test_notification_trigger_event(self):
        brain = OrchestratorBrain()
        event = {
            "event_type": "notification_trigger",
            "event_id": "notif-001",
            "bot_id": "system",
            "payload": '{"type": "daily_report"}',
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.SEND_NOTIFICATION

    def test_send_notification_action_type_exists(self):
        assert hasattr(ActionType, "SEND_NOTIFICATION")
        assert ActionType.SEND_NOTIFICATION == "send_notification"


class TestWorkerNotificationDispatch:
    @pytest.fixture
    def mock_queue(self):
        q = AsyncMock()
        q.claim = AsyncMock(return_value=[])
        q.ack = AsyncMock()
        return q

    @pytest.fixture
    def mock_registry(self):
        return AsyncMock()

    @pytest.fixture
    def worker(self, mock_queue, mock_registry):
        brain = OrchestratorBrain()
        return Worker(queue=mock_queue, registry=mock_registry, brain=brain)

    def test_on_notification_callback_exists(self, worker):
        assert hasattr(worker, "on_notification")

    @pytest.mark.asyncio
    async def test_notification_action_calls_on_notification(self, worker, mock_queue):
        handler = AsyncMock()
        worker.on_notification = handler
        mock_queue.claim = AsyncMock(return_value=[{
            "event_type": "notification_trigger",
            "event_id": "notif-001",
            "bot_id": "system",
            "payload": '{"type": "daily_report"}',
        }])
        await worker.process_batch(limit=1)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_without_handler_logs(self, worker, mock_queue):
        mock_queue.claim = AsyncMock(return_value=[{
            "event_type": "notification_trigger",
            "event_id": "notif-001",
            "bot_id": "system",
            "payload": '{"type": "daily_report"}',
        }])
        processed = await worker.process_batch(limit=1)
        assert processed == 1


class TestSchedulerNotificationJobs:
    def test_morning_scan_config(self):
        config = SchedulerConfig()
        assert hasattr(config, "morning_scan_hour")
        assert hasattr(config, "morning_scan_minute")

    def test_evening_report_config(self):
        config = SchedulerConfig()
        assert hasattr(config, "evening_report_hour")
        assert hasattr(config, "evening_report_minute")

    def test_morning_scan_job_created(self):
        config = SchedulerConfig()
        morning_fn = AsyncMock()
        jobs = create_scheduler_jobs(
            config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            morning_scan_fn=morning_fn,
        )
        job_names = [j["name"] for j in jobs]
        assert "morning_scan" in job_names

    def test_evening_report_job_created(self):
        config = SchedulerConfig()
        evening_fn = AsyncMock()
        jobs = create_scheduler_jobs(
            config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            evening_report_fn=evening_fn,
        )
        job_names = [j["name"] for j in jobs]
        assert "evening_report" in job_names

    def test_default_morning_scan_hour(self):
        config = SchedulerConfig()
        assert config.morning_scan_hour == 7

    def test_default_evening_report_hour(self):
        config = SchedulerConfig()
        assert config.evening_report_hour == 22
