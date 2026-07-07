# skills/learning_write_coordinator.py
"""Unified write coordinator for learning system artifacts.

Addresses the distributed-write problem where daily/weekly handlers perform
4-5+ sequential writes in independent try/except blocks.  Partial failures
leave the system in an inconsistent state with no provenance linking.

LearningWriteCoordinator:
  - Groups related writes into a single logical operation
  - Links all writes with a shared ``write_group_id``
  - Tracks success/failure per write for rollback or retry
  - Provides idempotency via dedup keys
  - Emits a single event on completion with full provenance
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class WriteOp:
    """A single write operation within a group."""

    name: str
    target: str  # e.g., "suggestions.jsonl", "outcomes.jsonl"
    data: Any = None
    dedup_key: str = ""
    status: str = "pending"  # pending | success | failed | skipped
    error: str = ""
    written_at: datetime | None = None


@dataclass
class WriteGroup:
    """A group of related writes with shared provenance."""

    group_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_workflow: str = ""
    source_run_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operations: list[WriteOp] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(op.status == "success" for op in self.operations)

    @property
    def has_failures(self) -> bool:
        return any(op.status == "failed" for op in self.operations)

    @property
    def summary(self) -> dict:
        return {
            "group_id": self.group_id,
            "source_workflow": self.source_workflow,
            "source_run_id": self.source_run_id,
            "created_at": self.created_at.isoformat(),
            "operations": [
                {
                    "name": op.name,
                    "target": op.target,
                    "status": op.status,
                    "error": op.error,
                }
                for op in self.operations
            ],
            "all_succeeded": self.all_succeeded,
        }


class LearningWriteCoordinator:
    """Coordinates learning system writes with provenance and consistency.

    Usage::

        coord = LearningWriteCoordinator(findings_dir)
        group = coord.begin("weekly_analysis", run_id="weekly-2026-04-13")

        coord.add_jsonl_append(group, "record_suggestions",
            "suggestions.jsonl", records, dedup_key=suggestion_id)
        coord.add_jsonl_append(group, "record_outcomes",
            "outcomes.jsonl", outcome_records)
        coord.add_callback(group, "update_hypotheses", fn, args)

        result = coord.execute(group)
        # result.all_succeeded -> True/False
    """

    def __init__(
        self,
        findings_dir: Path,
        event_stream=None,
        write_log_path: Path | None = None,
    ) -> None:
        self._findings_dir = findings_dir
        self._event_stream = event_stream
        self._write_log_path = write_log_path or (findings_dir / "write_log.jsonl")
        # Track dedup keys to prevent double-writes within a session
        self._seen_dedup_keys: set[str] = set()

    def begin(
        self,
        source_workflow: str = "",
        source_run_id: str = "",
    ) -> WriteGroup:
        """Start a new write group."""
        return WriteGroup(
            source_workflow=source_workflow,
            source_run_id=source_run_id,
        )

    def add_jsonl_append(
        self,
        group: WriteGroup,
        name: str,
        filename: str,
        records: list[dict],
        dedup_key: str = "",
    ) -> None:
        """Add a JSONL append operation to the group."""
        group.operations.append(WriteOp(
            name=name,
            target=filename,
            data=records,
            dedup_key=dedup_key,
        ))

    def add_json_write(
        self,
        group: WriteGroup,
        name: str,
        filename: str,
        data: Any,
        dedup_key: str = "",
    ) -> None:
        """Add a JSON file write operation to the group."""
        group.operations.append(WriteOp(
            name=name,
            target=filename,
            data=data,
            dedup_key=dedup_key,
        ))

    def add_callback(
        self,
        group: WriteGroup,
        name: str,
        fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict | None = None,
        dedup_key: str = "",
    ) -> None:
        """Add an arbitrary callback to the group."""
        group.operations.append(WriteOp(
            name=name,
            target="callback",
            data=(fn, args, kwargs or {}),
            dedup_key=dedup_key,
        ))

    def execute(self, group: WriteGroup) -> WriteGroup:
        """Execute all operations in the group sequentially.

        Each operation is wrapped in try/except.  The group records
        success/failure per op.  A provenance log entry is written at
        the end regardless of individual outcomes.
        """
        for op in group.operations:
            # Idempotency: skip if dedup_key already seen
            if op.dedup_key and op.dedup_key in self._seen_dedup_keys:
                op.status = "skipped"
                continue

            try:
                if op.target == "callback":
                    fn, args, kwargs = op.data
                    fn(*args, **kwargs)
                elif op.target.endswith(".jsonl"):
                    self._execute_jsonl_append(op, group.group_id)
                else:
                    self._execute_json_write(op, group.group_id)

                op.status = "success"
                op.written_at = datetime.now(timezone.utc)

                if op.dedup_key:
                    self._seen_dedup_keys.add(op.dedup_key)

            except Exception as exc:
                op.status = "failed"
                op.error = str(exc)[:300]
                logger.error(
                    "Write op '%s' failed in group %s: %s",
                    op.name, group.group_id, exc,
                )

        # Log the write group
        self._log_write_group(group)

        # Broadcast completion event
        if self._event_stream:
            self._event_stream.broadcast("learning_write_completed", group.summary)

        if group.has_failures:
            logger.warning(
                "Write group %s (%s) completed with failures: %s",
                group.group_id, group.source_workflow,
                [op.name for op in group.operations if op.status == "failed"],
            )
        else:
            logger.info(
                "Write group %s (%s) completed successfully (%d ops)",
                group.group_id, group.source_workflow,
                len(group.operations),
            )

        return group

    def _execute_jsonl_append(self, op: WriteOp, group_id: str) -> None:
        """Append records to a JSONL file with provenance."""
        path = self._findings_dir / op.target
        path.parent.mkdir(parents=True, exist_ok=True)
        records = op.data if isinstance(op.data, list) else [op.data]
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                if isinstance(record, dict):
                    record = {**record, "_write_group_id": group_id}
                f.write(json.dumps(record, default=str) + "\n")

    def _execute_json_write(self, op: WriteOp, group_id: str) -> None:
        """Write a JSON file with provenance."""
        path = self._findings_dir / op.target
        path.parent.mkdir(parents=True, exist_ok=True)
        data = op.data
        if isinstance(data, dict):
            data = {**data, "_write_group_id": group_id}
        path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8",
        )

    def _log_write_group(self, group: WriteGroup) -> None:
        """Append group summary to the write log."""
        try:
            self._write_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._write_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(group.summary, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write provenance log for group %s", group.group_id)
