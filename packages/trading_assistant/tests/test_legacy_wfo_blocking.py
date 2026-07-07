"""Tests proving legacy WFO cannot be routed, scheduled, or configured."""
import asyncio
import inspect

from trading_assistant.orchestrator.orchestrator_brain import ActionType, OrchestratorBrain
from trading_assistant.orchestrator.scheduler import (
    SchedulerConfig,
    build_scheduled_job_specs,
    create_scheduler_jobs,
)
from trading_assistant.orchestrator.worker import Worker


async def _noop_tracked(_scheduled_for=None):
    return None


async def _noop_legacy():
    return None


class TestLegacyWFORouting:
    def test_wfo_trigger_is_not_routed(self):
        brain = OrchestratorBrain()
        event = {
            "event_id": "wfo-bot2-2026-03-01",
            "event_type": "wfo_trigger",
            "bot_id": "bot2",
            "payload": '{"data_end":"2026-03-01"}',
        }

        actions = brain.decide(event)

        assert len(actions) == 1
        assert actions[0].type == ActionType.LOG_UNKNOWN
        assert actions[0].bot_id == "bot2"


class TestLegacyWFOWorkerDispatch:
    def test_wfo_trigger_does_not_dispatch_to_handler(self):
        from unittest.mock import AsyncMock, MagicMock

        from trading_assistant.orchestrator.db.queue import EventQueue
        from trading_assistant.orchestrator.task_registry import TaskRegistry

        queue = AsyncMock(spec=EventQueue)
        queue.claim.return_value = [
            {
                "event_id": "wfo-bot2-2026-03-01",
                "event_type": "wfo_trigger",
                "bot_id": "bot2",
                "payload": "{}",
            },
        ]

        registry = MagicMock(spec=TaskRegistry)
        brain = OrchestratorBrain()
        worker = Worker(queue, registry, brain)

        asyncio.run(worker.process_batch(limit=1))

        queue.ack.assert_called_once_with("wfo-bot2-2026-03-01")


class TestLegacyWFOSchedulerBlocking:
    def test_scheduler_config_exposes_no_wfo_fields(self):
        cfg = SchedulerConfig()
        assert not any(name.startswith("wfo") for name in vars(cfg))

    def test_scheduler_builders_expose_no_wfo_parameters(self):
        for fn in (create_scheduler_jobs, build_scheduled_job_specs):
            params = inspect.signature(fn).parameters
            assert "wfo_fn" not in params
            assert "wfo_fns" not in params

    def test_legacy_scheduler_emits_no_wfo_jobs(self):
        jobs = create_scheduler_jobs(
            SchedulerConfig(),
            worker_fn=_noop_legacy,
            monitoring_fn=_noop_legacy,
            relay_fn=_noop_legacy,
        )

        assert not any(job["name"].startswith("wfo") for job in jobs)

    def test_tracked_scheduler_emits_no_wfo_specs(self):
        specs = build_scheduled_job_specs(
            SchedulerConfig(),
            worker_fn=_noop_tracked,
            monitoring_fn=_noop_tracked,
            relay_fn=_noop_tracked,
        )

        assert not any(spec.name.startswith("wfo") or spec.job_key == "wfo" for spec in specs)
