"""Replay durable JSONL instrumentation events into PostgreSQL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from crypto_trader.instrumentation.postgres_sink import PostgresSink


@dataclass(slots=True)
class PostgresBackfillResult:
    files_read: int = 0
    events_seen: int = 0
    events_replayed: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "files_read": self.files_read,
            "events_seen": self.events_seen,
            "events_replayed": self.events_replayed,
            "failed": self.failed,
        }


def replay_jsonl_to_postgres(
    state_dir: Path | str,
    dsn: str,
    *,
    writer_factory: Callable[..., Any] = PostgresSink,
) -> PostgresBackfillResult:
    """Replay canonical JSONL event files into Postgres idempotently."""
    state_path = Path(state_dir)
    writer = writer_factory(dsn)
    result = PostgresBackfillResult()
    try:
        for path in _event_files(state_path):
            result.files_read += 1
            event_type = _event_type_from_path(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    result.events_seen += 1
                    try:
                        payload = json.loads(line)
                        writer.write_event(str(payload.get("event_type") or event_type), payload)
                        result.events_replayed += 1
                    except Exception:
                        result.failed += 1
        return result
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()


def _event_files(state_dir: Path) -> list[Path]:
    canonical = state_dir / "instrumentation" / "events"
    files = list(canonical.glob("*/*.jsonl")) if canonical.exists() else []
    files.extend(
        path for path in state_dir.glob("*.jsonl")
        if path.name not in {"sidecar_watermarks.jsonl"}
    )
    return sorted(set(files))


def _event_type_from_path(path: Path) -> str:
    if path.parent.parent.name == "events":
        return path.parent.name
    name = path.stem
    return {
        "instrumented_trades": "trade",
        "missed_opportunities": "missed_opportunity",
        "daily_snapshots": "daily_snapshot",
        "errors": "error",
        "pipeline_funnels": "pipeline_funnel",
        "health_reports": "heartbeat",
    }.get(name, name)
