"""Tests for tracked startup catch-up planning."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_assistant.orchestrator.catchup import StartupCatchup, bootstrap_run_store_from_history
from trading_assistant.orchestrator.scheduler import ScheduledJobClass, ScheduledJobSpec
from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore


async def _noop_job(scheduled_for=None) -> None:
    return None


@pytest.fixture
async def run_store(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def _daily_spec(scope_key: str = "global") -> ScheduledJobSpec:
    return ScheduledJobSpec(
        name=f"daily_analysis_{scope_key}",
        job_key="daily_analysis",
        trigger="cron",
        job_class=ScheduledJobClass.STATEFUL,
        execute=_noop_job,
        scope_key=scope_key,
        hour=6,
        minute=0,
        catchup_limit=7,
    )


def _weekly_spec() -> ScheduledJobSpec:
    return ScheduledJobSpec(
        name="weekly_analysis",
        job_key="weekly_summary",
        trigger="cron",
        job_class=ScheduledJobClass.STATEFUL,
        execute=_noop_job,
        scope_key="global",
        day_of_week="sun",
        hour=8,
        minute=0,
        catchup_limit=4,
    )


def _morning_spec(scope_key: str = "global") -> ScheduledJobSpec:
    return ScheduledJobSpec(
        name=f"morning_scan_{scope_key}",
        job_key="morning_scan",
        trigger="cron",
        job_class=ScheduledJobClass.COALESCED,
        execute=_noop_job,
        scope_key=scope_key,
        hour=7,
        minute=0,
        catchup_limit=1,
    )


def _interval_spec() -> ScheduledJobSpec:
    return ScheduledJobSpec(
        name="relay_poll",
        job_key="relay_poll",
        trigger="interval",
        job_class=ScheduledJobClass.INTERVAL,
        execute=_noop_job,
        seconds=300,
    )


def _legacy_wfo_spec() -> ScheduledJobSpec:
    return ScheduledJobSpec(
        name="wfo_bot_alpha",
        job_key="wfo",
        trigger="cron",
        job_class=ScheduledJobClass.STATEFUL,
        execute=_noop_job,
        scope_key="bot:bot_alpha",
        day_of_week="sat",
        hour=2,
        minute=0,
        catchup_limit=4,
    )


class TestStartupCatchup:
    @pytest.mark.asyncio
    async def test_no_false_daily_catchup_after_completed_run(self, run_store: ScheduledRunStore):
        spec = _daily_spec()
        now = datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc)
        await run_store.set_baseline(datetime(2026, 3, 5, 0, 0, tzinfo=timezone.utc))
        await run_store.mark_completed(spec.job_key, spec.scope_key, datetime(2026, 3, 5, 6, 0, tzinfo=timezone.utc))

        catchup = StartupCatchup([spec], run_store)
        assert await catchup.build_plan(now=now) == []

    @pytest.mark.asyncio
    async def test_no_false_weekly_catchup_after_completed_run(self, run_store: ScheduledRunStore):
        spec = _weekly_spec()
        now = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
        await run_store.set_baseline(datetime(2026, 3, 8, 0, 0, tzinfo=timezone.utc))
        await run_store.mark_completed(spec.job_key, spec.scope_key, datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc))

        catchup = StartupCatchup([spec], run_store)
        assert await catchup.build_plan(now=now) == []

    @pytest.mark.asyncio
    async def test_multi_timezone_stateful_jobs_replay_each_missed_occurrence(self, run_store: ScheduledRunStore):
        specs = [_daily_spec("bots:bot_a"), _daily_spec("bots:bot_b")]
        now = datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
        await run_store.set_baseline(datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc))
        await run_store.mark_completed("daily_analysis", "bots:bot_a", datetime(2026, 3, 3, 6, 0, tzinfo=timezone.utc))
        await run_store.mark_completed("daily_analysis", "bots:bot_b", datetime(2026, 3, 3, 6, 0, tzinfo=timezone.utc))

        catchup = StartupCatchup(specs, run_store)
        plan = await catchup.build_plan(now=now)

        assert len(plan) == 4
        assert {(item.spec.scope_key, item.scheduled_for.isoformat()) for item in plan} == {
            ("bots:bot_a", "2026-03-04T06:00:00+00:00"),
            ("bots:bot_a", "2026-03-05T06:00:00+00:00"),
            ("bots:bot_b", "2026-03-04T06:00:00+00:00"),
            ("bots:bot_b", "2026-03-05T06:00:00+00:00"),
        }

    @pytest.mark.asyncio
    async def test_coalesced_jobs_only_replay_latest_occurrence(self, run_store: ScheduledRunStore):
        spec = _morning_spec()
        now = datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
        await run_store.set_baseline(datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc))

        catchup = StartupCatchup([spec], run_store)
        plan = await catchup.build_plan(now=now)

        assert len(plan) == 1
        assert plan[0].scheduled_for == datetime(2026, 3, 5, 7, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_interval_jobs_are_not_replayed(self, run_store: ScheduledRunStore):
        catchup = StartupCatchup([_interval_spec()], run_store)
        assert await catchup.build_plan(now=datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)) == []


class TestBootstrapRunStoreFromHistory:
    @pytest.mark.asyncio
    async def test_legacy_wfo_history_is_not_seeded(self, run_store: ScheduledRunStore, tmp_path: Path):
        history = tmp_path / "run_history.jsonl"
        history.write_text(
            '{"run_id":"wfo-bot_alpha-2026-03-07","handler":"wfo","status":"completed"}\n',
            encoding="utf-8",
        )

        seeded = await bootstrap_run_store_from_history(
            run_store,
            [_legacy_wfo_spec()],
            history,
        )

        assert seeded is None
        assert await run_store.count_runs() == 0
