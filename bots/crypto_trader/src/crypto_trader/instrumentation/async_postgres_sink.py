"""Bounded async PostgreSQL sink wrapper."""

from __future__ import annotations

import queue
import copy
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from crypto_trader.instrumentation.postgres_sink import PostgresSink

log = structlog.get_logger()

CRITICAL_EVENTS = frozenset({
    "trade",
    "daily_snapshot",
    "heartbeat",
    "health_report",
    "error",
    "reconciliation_event",
    "position_allocation_snapshot",
})
VERBOSE_EVENTS = frozenset({
    "filter_decision",
    "indicator_snapshot",
    "pipeline_funnel",
    "missed_opportunity",
})

EVENT_VALUE_CLASSES = {
    "trade": "learning_authority",
    "missed_opportunity": "learning_authority",
    "pipeline_funnel": "learning_authority",
    "filter_decision": "learning_authority",
    "order": "learning_authority",
    "fill": "learning_authority",
    "portfolio_rule": "learning_authority",
    "risk_decision": "learning_authority",
    "indicator_snapshot": "learning_gap_diagnostic",
    "daily_snapshot": "operational_health",
    "heartbeat": "operational_health",
    "health_report": "operational_health",
    "error": "operational_health",
    "reconciliation_event": "operational_health",
    "position_allocation_snapshot": "operational_health",
    "equity_snapshot": "operational_health",
    "position_snapshot": "operational_health",
    "exchange_position_snapshot": "operational_health",
}


@dataclass(slots=True)
class PostgresWriteJob:
    operation: str
    event_type: str = ""
    payload: Any = None
    payload_schema: str = "assistant_event_v1"
    priority: str = "normal"
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt: int = 0


class AsyncPostgresSink:
    """Async wrapper implementing the full instrumentation sink protocol."""

    def __init__(
        self,
        dsn: str,
        *,
        queue_capacity: int = 5000,
        batch_size: int = 100,
        error_callback: Callable[[dict[str, Any]], None] | None = None,
        writer_factory: Callable[..., Any] = PostgresSink,
    ) -> None:
        self._dsn = dsn
        self._queue: queue.Queue[PostgresWriteJob] = queue.Queue(maxsize=queue_capacity)
        self._batch_size = max(1, int(batch_size))
        self._error_callback = error_callback
        self._writer_factory = writer_factory
        self._writer: Any | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._accepting = True
        self._lock = threading.Lock()
        self._latencies_ms: list[float] = []
        self._metrics: dict[str, Any] = {
            "enabled": True,
            "worker_alive": False,
            "queue_capacity": queue_capacity,
            "jobs_enqueued": 0,
            "jobs_written": 0,
            "jobs_dropped": 0,
            "dropped_by_priority": {"critical": 0, "normal": 0, "verbose": 0},
            "write_failures": 0,
            "last_success_at": None,
            "last_error": "",
            "last_batch_job_count": 0,
            "last_batch_duration_ms": 0.0,
        }

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run,
                name="async-postgres-sink",
                daemon=True,
            )
            self._worker.start()

    def write_trade(self, event) -> None:
        self._enqueue("write_trade", payload=_freeze_event(event), event_type="trade", priority="critical")

    def write_daily(self, event) -> None:
        self._enqueue("write_daily", payload=_freeze_event(event), event_type="daily_snapshot", priority="critical")

    def write_health_report(self, event) -> None:
        self._enqueue("write_health_report", payload=_freeze_event(event), event_type="heartbeat", priority="critical")

    def write_missed(self, event) -> None:
        self._enqueue("write_missed", payload=_freeze_event(event), event_type="missed_opportunity", priority="verbose")

    def write_error(self, event) -> None:
        self._enqueue("write_error", payload=_freeze_event(event), event_type="error", priority="critical")

    def write_funnel(self, event) -> None:
        self._enqueue("write_funnel", payload=_freeze_event(event), event_type="pipeline_funnel", priority="verbose")

    def write_event(self, event_type: str, event) -> None:
        self._enqueue(
            "write_event",
            payload=_freeze_event(event),
            event_type=event_type,
            priority=_priority_for_event(event_type),
        )

    def write_equity(self, equity: float, timestamp: datetime) -> None:
        self._enqueue(
            "write_equity",
            payload={"equity": float(equity), "timestamp": timestamp},
            event_type="equity_snapshot",
            priority="normal",
        )

    def upsert_positions(self, positions: list[dict[str, Any]]) -> None:
        self._enqueue(
            "upsert_positions",
            payload=[dict(position) for position in positions],
            event_type="position_snapshot",
            priority="normal",
        )

    def upsert_strategy_position_allocations(self, allocations: list[dict[str, Any]]) -> None:
        self._enqueue(
            "upsert_strategy_position_allocations",
            payload=[dict(allocation) for allocation in allocations],
            event_type="position_allocation_snapshot",
            priority="critical",
        )

    def upsert_exchange_positions(self, positions: list[dict[str, Any]]) -> None:
        self._enqueue(
            "upsert_exchange_positions",
            payload=[dict(position) for position in positions],
            event_type="exchange_position_snapshot",
            priority="normal",
        )

    def close(self, flush_timeout_sec: float = 5.0) -> None:
        self._accepting = False
        deadline = time.monotonic() + max(0.0, flush_timeout_sec)
        while self._queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        writer = self._writer
        if writer is not None:
            try:
                writer.close()
            except Exception:
                log.exception("async_postgres_sink.writer_close_failed")
        with self._lock:
            self._metrics["worker_alive"] = bool(worker and worker.is_alive())

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            metrics = dict(self._metrics)
            metrics["dropped_by_priority"] = dict(self._metrics["dropped_by_priority"])
            metrics["queue_depth"] = self._queue.qsize()
            metrics["oldest_job_age_sec"] = self._oldest_job_age_sec()
            metrics["worker_alive"] = bool(self._worker and self._worker.is_alive())
            metrics["p50_write_latency_ms"] = _percentile(self._latencies_ms, 50)
            metrics["p95_write_latency_ms"] = _percentile(self._latencies_ms, 95)
            return metrics

    def _enqueue(self, operation: str, *, payload: Any, event_type: str, priority: str) -> None:
        if not self._accepting:
            self._record_drop(priority, reason="sink_closing", event_type=event_type)
            return
        job = PostgresWriteJob(
            operation=operation,
            event_type=event_type,
            payload=payload,
            priority=priority,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            if priority in {"critical", "normal"} and self._drop_one_verbose_job():
                try:
                    self._queue.put_nowait(job)
                except queue.Full:
                    self._record_drop(priority, reason="queue_full_after_preempt", event_type=event_type)
                    return
            else:
                self._record_drop(priority, reason="queue_full", event_type=event_type)
                return
        with self._lock:
            self._metrics["jobs_enqueued"] += 1
        self.start()

    def _run(self) -> None:
        try:
            self._writer = self._writer_factory(
                self._dsn,
                error_callback=self._writer_error_callback,
            )
        except TypeError:
            self._writer = self._writer_factory(self._dsn)
        except Exception as exc:
            self._record_failure(exc, operation="init")
            self._drop_all_queued(reason="writer_init_failed")
            return

        while not self._stop.is_set() or self._queue.qsize() > 0:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            self._drain_batch(batch)
            self._write_batch(batch)
            for _ in batch:
                self._queue.task_done()

    def _drain_batch(self, batch: list[PostgresWriteJob]) -> None:
        if batch[0].operation != "write_event":
            return
        event_type = batch[0].event_type
        with self._queue.mutex:
            while len(batch) < self._batch_size and self._queue.queue:
                candidate = self._queue.queue[0]
                if candidate.operation != "write_event" or candidate.event_type != event_type:
                    return
                batch.append(self._queue.queue.popleft())

    def _drop_all_queued(self, *, reason: str) -> None:
        dropped: list[PostgresWriteJob] = []
        with self._queue.mutex:
            while self._queue.queue:
                dropped.append(self._queue.queue.popleft())
                self._queue.unfinished_tasks = max(0, self._queue.unfinished_tasks - 1)
        for job in dropped:
            self._record_drop(job.priority, reason=reason, event_type=job.event_type)

    def _write_batch(self, batch: list[PostgresWriteJob]) -> None:
        started = time.perf_counter()
        try:
            if len(batch) > 1 and batch[0].operation == "write_event":
                self._writer.write_events_batch(
                    batch[0].event_type,
                    [job.payload for job in batch],
                )
            else:
                for job in batch:
                    self._write_job(job)
            duration_ms = (time.perf_counter() - started) * 1000
            with self._lock:
                self._metrics["jobs_written"] += len(batch)
                self._metrics["last_success_at"] = datetime.now(timezone.utc).isoformat()
                self._metrics["last_batch_job_count"] = len(batch)
                self._metrics["last_batch_duration_ms"] = duration_ms
                self._latencies_ms.append(duration_ms)
                self._latencies_ms[:] = self._latencies_ms[-200:]
        except Exception as exc:
            self._record_failure(exc, operation=batch[0].operation)
            for job in batch:
                self._retry_or_drop(job)

    def _write_job(self, job: PostgresWriteJob) -> None:
        if job.operation == "write_event":
            self._writer.write_event(job.event_type, job.payload)
        elif job.operation == "write_trade":
            self._writer.write_trade(job.payload)
        elif job.operation == "write_daily":
            self._writer.write_daily(job.payload)
        elif job.operation == "write_health_report":
            self._writer.write_health_report(job.payload)
        elif job.operation == "write_missed":
            self._writer.write_missed(job.payload)
        elif job.operation == "write_error":
            self._writer.write_error(job.payload)
        elif job.operation == "write_funnel":
            self._writer.write_funnel(job.payload)
        elif job.operation == "write_equity":
            self._writer.write_equity(job.payload["equity"], job.payload["timestamp"])
        elif job.operation == "upsert_positions":
            self._writer.upsert_positions(job.payload)
        elif job.operation == "upsert_strategy_position_allocations":
            self._writer.upsert_strategy_position_allocations(job.payload)
        elif job.operation == "upsert_exchange_positions":
            self._writer.upsert_exchange_positions(job.payload)
        else:
            raise ValueError(f"unknown postgres write operation: {job.operation}")

    def _retry_or_drop(self, job: PostgresWriteJob) -> None:
        job.attempt += 1
        if job.attempt > 3 or self._stop.is_set():
            self._record_drop(job.priority, reason="retry_exhausted", event_type=job.event_type)
            return
        time.sleep(min(0.25 * (2 ** (job.attempt - 1)), 2.0))
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._record_drop(job.priority, reason="retry_queue_full", event_type=job.event_type)

    def _drop_one_verbose_job(self) -> bool:
        dropped: PostgresWriteJob | None = None
        with self._queue.mutex:
            for index, job in enumerate(list(self._queue.queue)):
                if job.priority != "verbose":
                    continue
                del self._queue.queue[index]
                self._queue.unfinished_tasks = max(0, self._queue.unfinished_tasks - 1)
                dropped = job
                break
        if dropped is None:
            return False
        self._record_drop("verbose", reason="queue_full_preempted", event_type=dropped.event_type)
        return True

    def _record_drop(self, priority: str, *, reason: str, event_type: str) -> None:
        with self._lock:
            self._metrics["jobs_dropped"] += 1
            dropped = self._metrics["dropped_by_priority"]
            dropped[priority] = dropped.get(priority, 0) + 1
            if priority == "critical":
                self._metrics["last_error"] = f"critical job dropped: {event_type} ({reason})"
        if priority == "critical":
            self._safe_error_callback({
                "component": "postgres_sink",
                "event_type": event_type,
                "error_type": "QueueFull",
                "message": f"critical postgres job dropped: {reason}",
                "severity": "critical",
                "recovery_action": "jsonl_backfill_required",
            })

    def _record_failure(self, exc: Exception, *, operation: str) -> None:
        with self._lock:
            self._metrics["write_failures"] += 1
            self._metrics["last_error"] = f"{type(exc).__name__}: {exc}"
        self._safe_error_callback({
            "component": "postgres_sink",
            "event_type": operation,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "severity": "medium",
            "recovery_action": "continue_with_jsonl_and_backfill",
        })

    def _safe_error_callback(self, payload: dict[str, Any]) -> None:
        if self._error_callback is None:
            return
        try:
            self._error_callback(payload)
        except Exception:
            log.exception("async_postgres_sink.error_callback_failed")

    def _writer_error_callback(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._metrics["write_failures"] += 1
            message = str(payload.get("message") or payload.get("error_type") or "writer error")
            self._metrics["last_error"] = message
        self._safe_error_callback(payload)

    def _oldest_job_age_sec(self) -> float:
        with self._queue.mutex:
            if not self._queue.queue:
                return 0.0
            oldest = self._queue.queue[0].enqueued_at
        return max(0.0, (datetime.now(timezone.utc) - oldest).total_seconds())


class _FrozenObject:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = dict(payload)

    def __getattr__(self, name: str) -> Any:
        value = self._payload.get(name)
        if isinstance(value, dict):
            return _FrozenObject(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def _freeze_event(event: Any) -> Any:
    if isinstance(event, (dict, list, tuple)):
        return copy.deepcopy(event)
    if isinstance(event, (str, int, float, bool, type(None))):
        return event
    if hasattr(event, "to_dict"):
        return _FrozenObject(event.to_dict())
    return _FrozenObject(dict(event))


def _priority_for_event(event_type: str) -> str:
    if event_type in CRITICAL_EVENTS:
        return "critical"
    if event_type in VERBOSE_EVENTS:
        return "verbose"
    return "normal"


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return float(ordered[index])
