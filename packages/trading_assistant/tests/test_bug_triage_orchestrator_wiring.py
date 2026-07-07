# tests/test_bug_triage_orchestrator_wiring.py
"""Tests for Phase 5 orchestrator wiring — brain + worker + scheduler."""
import json

from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType
from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


class TestBrainErrorRouting:
    """The brain already handles error events, but now we test the full severity spectrum."""

    def test_critical_error_produces_alert(self):
        # P1-1: critical errors emit both ALERT_IMMEDIATE and QUEUE_FOR_DAILY
        # so the failure is also persisted in the raw daily store.
        brain = OrchestratorBrain()
        event = {
            "event_id": "e1",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({
                "severity": "CRITICAL",
                "error_type": "SystemExit",
                "message": "crash",
            }),
        }
        actions = brain.decide(event)
        types = [a.type for a in actions]
        assert ActionType.ALERT_IMMEDIATE in types
        assert ActionType.QUEUE_FOR_DAILY in types

    def test_high_error_spawns_triage(self):
        # P1-1: high errors emit both SPAWN_TRIAGE and QUEUE_FOR_DAILY.
        brain = OrchestratorBrain()
        event = {
            "event_id": "e2",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "HIGH"}),
        }
        actions = brain.decide(event)
        types = [a.type for a in actions]
        assert ActionType.SPAWN_TRIAGE in types
        assert ActionType.QUEUE_FOR_DAILY in types

    def test_medium_error_queues_for_daily(self):
        brain = OrchestratorBrain()
        event = {
            "event_id": "e3",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "MEDIUM"}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_low_error_queues_for_weekly(self):
        brain = OrchestratorBrain()
        event = {
            "event_id": "e4",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "LOW"}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_WEEKLY


class TestSchedulerStaleErrorSweep:
    def test_config_has_stale_error_sweep_fields(self):
        config = SchedulerConfig()
        assert hasattr(config, "stale_error_sweep_interval_seconds")
        assert config.stale_error_sweep_interval_seconds == 600  # 10 min default

    def test_stale_error_sweep_job_created(self):
        config = SchedulerConfig()

        async def noop():
            pass

        jobs = create_scheduler_jobs(
            config,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
            stale_error_sweep_fn=noop,
        )
        job_names = [j["name"] for j in jobs]
        assert "stale_error_sweep" in job_names

    def test_stale_error_sweep_not_created_without_fn(self):
        config = SchedulerConfig()

        async def noop():
            pass

        jobs = create_scheduler_jobs(
            config,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
        )
        job_names = [j["name"] for j in jobs]
        assert "stale_error_sweep" not in job_names
