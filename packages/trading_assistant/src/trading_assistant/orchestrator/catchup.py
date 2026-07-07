"""Startup catch-up planning for tracked scheduled jobs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.orchestrator.scheduler import (
    ScheduledJobClass,
    ScheduledJobSpec,
    recent_cron_occurrences,
)
from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore

logger = logging.getLogger(__name__)

_DAILY_RUN_RE = re.compile(r"^daily-(\d{4}-\d{2}-\d{2})")
_WEEKLY_RUN_RE = re.compile(r"^weekly-(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class ScheduledOccurrence:
    spec: ScheduledJobSpec
    scheduled_for: datetime


class StartupCatchup:
    """Plans which missed cron executions should be replayed on startup."""

    def __init__(
        self,
        job_specs: list[ScheduledJobSpec],
        run_store: ScheduledRunStore,
    ) -> None:
        self._job_specs = job_specs
        self._run_store = run_store

    async def build_plan(self, *, now: datetime | None = None) -> list[ScheduledOccurrence]:
        now = _ensure_utc(now or datetime.now(timezone.utc))
        baseline = await self._run_store.get_baseline()
        occurrences: list[ScheduledOccurrence] = []

        for spec in self._job_specs:
            if not spec.is_catchup_eligible:
                continue

            candidates = recent_cron_occurrences(spec, now, spec.catchup_limit)
            if not candidates:
                continue

            if baseline is not None:
                candidates = [candidate for candidate in candidates if candidate > baseline]
                if not candidates:
                    continue

            records = await self._run_store.get_records(
                spec.job_key,
                spec.scope_key,
                since=candidates[0],
                until=candidates[-1],
            )
            completed = {
                record.scheduled_for.replace(microsecond=0)
                for record in records
                if record.status == "completed"
            }
            due = [
                candidate.replace(microsecond=0)
                for candidate in candidates
                if candidate.replace(microsecond=0) not in completed
            ]
            if not due:
                continue

            if spec.job_class == ScheduledJobClass.COALESCED:
                occurrences.append(ScheduledOccurrence(spec=spec, scheduled_for=due[-1]))
            else:
                occurrences.extend(
                    ScheduledOccurrence(spec=spec, scheduled_for=scheduled_for)
                    for scheduled_for in due
                )

        occurrences.sort(key=lambda item: (item.scheduled_for, item.spec.name))
        return occurrences


async def bootstrap_run_store_from_history(
    run_store: ScheduledRunStore,
    job_specs: list[ScheduledJobSpec],
    run_history_path: Path,
) -> datetime | None:
    """Seed tracked completions from run_history.jsonl as a one-time migration."""
    path = Path(run_history_path)
    if not path.exists():
        return None

    earliest_seeded: datetime | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Could not read run history at %s", path)
        return None

    daily_specs = [spec for spec in job_specs if spec.job_key == "daily_analysis"]
    weekly_specs = [spec for spec in job_specs if spec.job_key == "weekly_summary"]

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        status = entry.get("status", "")
        if status not in {"completed", "skipped"}:
            continue

        handler_name = entry.get("handler") or entry.get("agent_type") or ""
        run_id = entry.get("run_id", "")

        if handler_name == "daily_analysis":
            match = _DAILY_RUN_RE.match(run_id)
            if not match:
                continue
            run_date = match.group(1)
            for spec in daily_specs:
                scheduled_for = _scheduled_for_daily(run_date, spec)
                await run_store.seed_completion(
                    spec.job_key,
                    spec.scope_key,
                    scheduled_for,
                    started_at=entry.get("started_at", ""),
                    finished_at=entry.get("finished_at", ""),
                )
                earliest_seeded = _min_datetime(earliest_seeded, scheduled_for)
            continue

        if handler_name == "weekly_analysis":
            match = _WEEKLY_RUN_RE.match(run_id)
            if not match or not weekly_specs:
                continue
            week_start = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            week_end = week_start + timedelta(days=6)
            for spec in weekly_specs:
                scheduled_for = datetime(
                    week_end.year,
                    week_end.month,
                    week_end.day,
                    spec.hour or 0,
                    spec.minute or 0,
                    tzinfo=timezone.utc,
                )
                await run_store.seed_completion(
                    spec.job_key,
                    spec.scope_key,
                    scheduled_for,
                    started_at=entry.get("started_at", ""),
                    finished_at=entry.get("finished_at", ""),
                )
                earliest_seeded = _min_datetime(earliest_seeded, scheduled_for)
            continue

    return earliest_seeded


def _scheduled_for_daily(run_date: str, spec: ScheduledJobSpec) -> datetime:
    date = datetime.strptime(run_date, "%Y-%m-%d").date()
    return datetime(
        date.year,
        date.month,
        date.day,
        spec.hour or 0,
        spec.minute or 0,
        tzinfo=timezone.utc,
    )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _min_datetime(current: datetime | None, candidate: datetime) -> datetime:
    if current is None:
        return candidate
    return min(current, candidate)
