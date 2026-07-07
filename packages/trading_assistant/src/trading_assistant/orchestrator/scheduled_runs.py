"""Persistent tracking for scheduled cron job executions."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledRunRecord:
    job_key: str
    scope_key: str
    scheduled_for: datetime
    status: str
    started_at: str
    finished_at: str
    error: str


class ScheduledRunStore:
    """SQLite-backed store for tracked scheduled job runs."""

    def __init__(
        self,
        db_path: str,
        *,
        final_status_observer: Callable[[ScheduledRunRecord], object] | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None
        self._final_status_observer = final_status_observer

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ScheduledRunStore not initialized")
        return self._db

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self.db.execute("PRAGMA journal_mode=WAL")
        # P1-9: wait on writer-lock contention rather than failing fast.
        await self.db.execute("PRAGMA busy_timeout=5000")
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_runs (
                job_key TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                finished_at TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (job_key, scope_key, scheduled_for)
            )
            """
        )
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.db.row_factory = aiosqlite.Row
        await self.db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def count_runs(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM scheduled_runs")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def is_empty(self) -> bool:
        return await self.count_runs() == 0

    async def has_records(self, job_key: str, scope_key: str) -> bool:
        cursor = await self.db.execute(
            """
            SELECT 1
            FROM scheduled_runs
            WHERE job_key = ? AND scope_key = ?
            LIMIT 1
            """,
            (job_key, scope_key),
        )
        return await cursor.fetchone() is not None

    async def get_records(
        self,
        job_key: str,
        scope_key: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[ScheduledRunRecord]:
        clauses = ["job_key = ?", "scope_key = ?"]
        params: list[str] = [job_key, scope_key]
        if since is not None:
            clauses.append("scheduled_for >= ?")
            params.append(_to_iso(since))
        if until is not None:
            clauses.append("scheduled_for <= ?")
            params.append(_to_iso(until))

        cursor = await self.db.execute(
            f"""
            SELECT job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            FROM scheduled_runs
            WHERE {' AND '.join(clauses)}
            ORDER BY scheduled_for ASC
            """,
            params,
        )
        rows = await cursor.fetchall()
        return [
            ScheduledRunRecord(
                job_key=row["job_key"],
                scope_key=row["scope_key"],
                scheduled_for=_from_iso(row["scheduled_for"]),
                status=row["status"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                error=row["error"],
            )
            for row in rows
        ]

    async def list_recent_records(self, *, limit: int = 100) -> list[ScheduledRunRecord]:
        cursor = await self.db.execute(
            """
            SELECT job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            FROM scheduled_runs
            ORDER BY scheduled_for DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        return [
            ScheduledRunRecord(
                job_key=row["job_key"],
                scope_key=row["scope_key"],
                scheduled_for=_from_iso(row["scheduled_for"]),
                status=row["status"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                error=row["error"],
            )
            for row in rows
        ]

    async def is_completed(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
    ) -> bool:
        cursor = await self.db.execute(
            """
            SELECT status
            FROM scheduled_runs
            WHERE job_key = ? AND scope_key = ? AND scheduled_for = ?
            """,
            (job_key, scope_key, _to_iso(scheduled_for)),
        )
        row = await cursor.fetchone()
        return bool(row and row["status"] == "completed")

    async def get_status(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
    ) -> str | None:
        cursor = await self.db.execute(
            """
            SELECT status
            FROM scheduled_runs
            WHERE job_key = ? AND scope_key = ? AND scheduled_for = ?
            """,
            (job_key, scope_key, _to_iso(scheduled_for)),
        )
        row = await cursor.fetchone()
        return row["status"] if row else None

    async def mark_started(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
        *,
        started_at: str | None = None,
    ) -> None:
        started = started_at or _to_iso(datetime.now(timezone.utc))
        await self.db.execute(
            """
            INSERT INTO scheduled_runs (
                job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            )
            VALUES (?, ?, ?, 'running', ?, '', '')
            ON CONFLICT(job_key, scope_key, scheduled_for)
            DO UPDATE SET
                status = 'running',
                started_at = COALESCE(NULLIF(scheduled_runs.started_at, ''), excluded.started_at),
                finished_at = '',
                error = ''
            """,
            (job_key, scope_key, _to_iso(scheduled_for), started),
        )
        await self.db.commit()

    async def mark_enqueued(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
        *,
        started_at: str | None = None,
        enqueued_at: str | None = None,
    ) -> None:
        """Trigger has been enqueued; downstream handler still owes mark_completed.

        Used for daily/weekly/monthly/triage cron jobs where the cron only enqueues
        a trigger event and the actual report/analysis runs asynchronously.
        """
        started = started_at or _to_iso(datetime.now(timezone.utc))
        enqueued = enqueued_at or _to_iso(datetime.now(timezone.utc))
        await self.db.execute(
            """
            INSERT INTO scheduled_runs (
                job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            )
            VALUES (?, ?, ?, 'enqueued', ?, ?, '')
            ON CONFLICT(job_key, scope_key, scheduled_for)
            DO UPDATE SET
                status = 'enqueued',
                started_at = COALESCE(NULLIF(scheduled_runs.started_at, ''), excluded.started_at),
                finished_at = excluded.finished_at,
                error = ''
            """,
            (job_key, scope_key, _to_iso(scheduled_for), started, enqueued),
        )
        await self.db.commit()

    async def mark_completed(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        started = started_at or _to_iso(datetime.now(timezone.utc))
        finished = finished_at or _to_iso(datetime.now(timezone.utc))
        await self.db.execute(
            """
            INSERT INTO scheduled_runs (
                job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            )
            VALUES (?, ?, ?, 'completed', ?, ?, '')
            ON CONFLICT(job_key, scope_key, scheduled_for)
            DO UPDATE SET
                status = 'completed',
                started_at = COALESCE(NULLIF(scheduled_runs.started_at, ''), excluded.started_at),
                finished_at = excluded.finished_at,
                error = ''
            """,
            (job_key, scope_key, _to_iso(scheduled_for), started, finished),
        )
        await self.db.commit()
        await self._notify_final_status(job_key, scope_key, scheduled_for)

    async def mark_failed(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
        *,
        error: str,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        started = started_at or _to_iso(datetime.now(timezone.utc))
        finished = finished_at or _to_iso(datetime.now(timezone.utc))
        await self.db.execute(
            """
            INSERT INTO scheduled_runs (
                job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            )
            VALUES (?, ?, ?, 'failed', ?, ?, ?)
            ON CONFLICT(job_key, scope_key, scheduled_for)
            DO UPDATE SET
                status = 'failed',
                started_at = COALESCE(NULLIF(scheduled_runs.started_at, ''), excluded.started_at),
                finished_at = excluded.finished_at,
                error = excluded.error
            """,
            (job_key, scope_key, _to_iso(scheduled_for), started, finished, error),
        )
        await self.db.commit()
        await self._notify_final_status(job_key, scope_key, scheduled_for)

    async def seed_completion(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
        *,
        started_at: str = "",
        finished_at: str = "",
        error: str = "",
    ) -> None:
        await self.db.execute(
            """
            INSERT OR IGNORE INTO scheduled_runs (
                job_key, scope_key, scheduled_for, status, started_at, finished_at, error
            )
            VALUES (?, ?, ?, 'completed', ?, ?, ?)
            """,
            (
                job_key,
                scope_key,
                _to_iso(scheduled_for),
                started_at,
                finished_at,
                error,
            ),
        )
        await self.db.commit()

    async def get_baseline(self) -> datetime | None:
        cursor = await self.db.execute(
            "SELECT value FROM metadata WHERE key = 'baseline_started_at'"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _from_iso(row["value"])

    async def set_baseline(self, value: datetime) -> None:
        await self.db.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('baseline_started_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_to_iso(value),),
        )
        await self.db.commit()

    async def _notify_final_status(
        self,
        job_key: str,
        scope_key: str,
        scheduled_for: datetime,
    ) -> None:
        if self._final_status_observer is None:
            return
        records = await self.get_records(
            job_key,
            scope_key,
            since=scheduled_for,
            until=scheduled_for,
        )
        if not records:
            return
        try:
            result = self._final_status_observer(records[-1])
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning(
                "Scheduled-run projection observer failed for %s/%s @ %s",
                job_key,
                scope_key,
                scheduled_for.isoformat(),
                exc_info=True,
            )


def _to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
