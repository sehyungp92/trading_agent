"""Task registry — tracks every agent invocation (OpenClaw pattern).

Persisted to SQLite. Supports status transitions, retry logic, and stale task detection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from trading_assistant.orchestrator.db.connection import create_connection
from trading_assistant.schemas.tasks import TaskRecord, TaskStatus

_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    agent           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    context_files   TEXT NOT NULL DEFAULT '[]',
    run_folder      TEXT NOT NULL DEFAULT '',
    retries         INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    result_summary  TEXT NOT NULL DEFAULT '',
    error           TEXT NOT NULL DEFAULT '',
    source_event_id TEXT NOT NULL DEFAULT '',
    source_action_type TEXT NOT NULL DEFAULT '',
    subagent_id     TEXT NOT NULL DEFAULT '',
    notify_on_complete INTEGER NOT NULL DEFAULT 1,
    notify_channels TEXT NOT NULL DEFAULT '["telegram"]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(type);
"""


class TaskRegistry:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await create_connection(self._db_path)
        await self._db.executescript(_TASKS_SCHEMA)
        await self._ensure_columns()
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_source_event ON tasks(source_event_id)"
        )
        await self._db.commit()

    async def _ensure_columns(self) -> None:
        cursor = await self.db.execute("PRAGMA table_info(tasks)")
        rows = await cursor.fetchall()
        existing = {row["name"] for row in rows}
        for name in ("source_event_id", "source_action_type", "subagent_id"):
            if name not in existing:
                await self.db.execute(
                    f"ALTER TABLE tasks ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                )

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Call initialize() first"
        return self._db

    async def create(self, task: TaskRecord) -> None:
        await self.db.execute(
            """INSERT INTO tasks
               (id, type, agent, status, context_files, run_folder, retries, max_retries,
                result_summary, error, source_event_id, source_action_type, subagent_id,
                notify_on_complete, notify_channels, created_at, updated_at, started_at,
                completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id, task.type, task.agent, task.status.value,
                json.dumps(task.context_files), task.run_folder,
                task.retries, task.max_retries,
                task.result_summary, task.error,
                task.source_event_id, task.source_action_type, task.subagent_id,
                int(task.notify_on_complete), json.dumps(task.notify_channels),
                task.created_at.isoformat(), task.updated_at.isoformat(),
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
            ),
        )
        await self.db.commit()

    def _row_to_task(self, row: aiosqlite.Row) -> TaskRecord:
        d = dict(row)
        d["context_files"] = json.loads(d["context_files"])
        d["notify_on_complete"] = bool(d["notify_on_complete"])
        d["notify_channels"] = json.loads(d["notify_channels"])
        return TaskRecord.model_validate(d)

    async def get(self, task_id: str) -> TaskRecord | None:
        cursor = await self.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return self._row_to_task(row) if row else None

    async def update_status(self, task_id: str, status: TaskStatus) -> None:
        now = datetime.now(timezone.utc).isoformat()
        extra = ""
        if status == TaskStatus.RUNNING:
            extra = ", started_at = ?, completed_at = NULL"
            params = (status.value, now, now, task_id)
        else:
            params = (status.value, now, task_id)

        await self.db.execute(
            f"UPDATE tasks SET status = ?, updated_at = ?{extra} WHERE id = ?",
            params,
        )
        await self.db.commit()

    async def mark_running(self, task_id: str, *, subagent_id: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        extra = ", subagent_id = ?" if subagent_id else ""
        params: tuple[object, ...]
        if subagent_id:
            params = (TaskStatus.RUNNING.value, now, now, subagent_id, task_id)
        else:
            params = (TaskStatus.RUNNING.value, now, now, task_id)
        await self.db.execute(
            f"""UPDATE tasks SET status = ?, updated_at = ?, started_at = ?,
                completed_at = NULL{extra} WHERE id = ?""",
            params,
        )
        await self.db.commit()

    async def complete(self, task_id: str, result_summary: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """UPDATE tasks SET status = ?, result_summary = ?,
               updated_at = ?, completed_at = ? WHERE id = ?""",
            (TaskStatus.COMPLETED.value, result_summary, now, now, task_id),
        )
        await self.db.commit()

    async def fail(self, task_id: str, error: str = "") -> None:
        task = await self.get(task_id)
        assert task is not None, f"Task {task_id} not found"

        new_retries = task.retries + 1
        now = datetime.now(timezone.utc).isoformat()

        if new_retries >= task.max_retries:
            new_status = TaskStatus.FAILED.value
        else:
            new_status = TaskStatus.PENDING.value  # retry

        await self.db.execute(
            """UPDATE tasks SET status = ?, retries = ?, error = ?, updated_at = ? WHERE id = ?""",
            (new_status, new_retries, error, now, task_id),
        )
        await self.db.commit()

    async def list_by_status(self, status: TaskStatus) -> list[TaskRecord]:
        cursor = await self.db.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY created_at ASC",
            (status.value,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(row) for row in rows]

    async def list_by_source_event(self, source_event_id: str) -> list[TaskRecord]:
        cursor = await self.db.execute(
            "SELECT * FROM tasks WHERE source_event_id = ? ORDER BY created_at ASC",
            (source_event_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(row) for row in rows]

    async def list_retryable_linked(self, limit: int = 20) -> list[TaskRecord]:
        cursor = await self.db.execute(
            """SELECT * FROM tasks
               WHERE status = 'pending'
                 AND source_event_id != ''
                 AND source_action_type != ''
                 AND retries < max_retries
               ORDER BY updated_at ASC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(row) for row in rows]

    async def find_stale(self, timeout_seconds: int = 3600) -> list[TaskRecord]:
        """Find tasks stuck in RUNNING state beyond timeout."""
        if timeout_seconds <= 0:
            cursor = await self.db.execute(
                """SELECT * FROM tasks
                   WHERE status = 'running'
                     AND started_at IS NOT NULL
                   ORDER BY started_at ASC"""
            )
            rows = await cursor.fetchall()
            return [self._row_to_task(row) for row in rows]

        cursor = await self.db.execute(
            """SELECT * FROM tasks
               WHERE status = 'running'
                 AND started_at IS NOT NULL
                 AND (julianday('now') - julianday(started_at)) * 86400 >= ?
               ORDER BY started_at ASC""",
            (timeout_seconds,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(row) for row in rows]
