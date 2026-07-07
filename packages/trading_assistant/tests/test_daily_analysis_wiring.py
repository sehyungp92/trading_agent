# tests/test_daily_analysis_wiring.py
"""Tests for daily analysis wiring — brain trigger, scheduler cron, worker dispatch."""

from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType
from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


class TestBrainDailyTrigger:
    def test_daily_analysis_trigger_event(self):
        brain = OrchestratorBrain()
        event = {
            "event_type": "daily_analysis_trigger",
            "event_id": "cron-daily-2026-03-01",
            "bot_id": "_system",
            "payload": '{"date": "2026-03-01"}',
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.SPAWN_DAILY_ANALYSIS
        assert actions[0].details["date"] == "2026-03-01"


class TestSchedulerDailyCron:
    def test_scheduler_includes_daily_analysis_job(self):
        config = SchedulerConfig()

        async def noop():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
            daily_analysis_fn=noop,
        )
        job_names = [j["name"] for j in jobs]
        assert "daily_analysis" in job_names

    def test_daily_analysis_default_cron(self):
        config = SchedulerConfig()

        async def noop():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
            daily_analysis_fn=noop,
        )
        daily_job = next(j for j in jobs if j["name"] == "daily_analysis")
        assert daily_job["trigger"] == "cron"
        assert daily_job["hour"] == config.daily_analysis_hour
