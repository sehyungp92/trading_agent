"""Event sinks for instrumentation output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
    canonical_event_envelope,
)

log = structlog.get_logger()


@runtime_checkable
class Sink(Protocol):
    """Protocol for event sinks."""

    def write_trade(self, event: InstrumentedTradeEvent) -> None: ...
    def write_missed(self, event: MissedOpportunityEvent) -> None: ...
    def write_daily(self, event: DailySnapshot) -> None: ...
    def write_error(self, event: ErrorEvent) -> None: ...
    def write_funnel(self, event: PipelineFunnelSnapshot) -> None: ...
    def write_health_report(self, event: HealthReportSnapshot) -> None: ...
    def write_event(self, event_type: str, event) -> None: ...


_LEGACY_FILES = {
    "trade": "instrumented_trades.jsonl",
    "missed_opportunity": "missed_opportunities.jsonl",
    "daily_snapshot": "daily_snapshots.jsonl",
    "error": "errors.jsonl",
    "pipeline_funnel": "pipeline_funnels.jsonl",
    "heartbeat": "health_reports.jsonl",
}


class JsonlSink:
    """Appends serialized events to per-type JSONL files."""

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _append(self, filename: str, data: dict) -> None:
        path = self._dir / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception:
            log.exception("jsonl_sink.append_failed", path=str(path))

    def _append_canonical(self, event_type: str, data: dict) -> None:
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        timestamp = (
            data.get("exchange_timestamp")
            or metadata.get("exchange_timestamp")
            or data.get("timestamp")
            or ""
        )
        date_part = str(timestamp)[:10] if timestamp else "undated"
        if len(date_part) != 10:
            date_part = "undated"
        path = Path("instrumentation") / "events" / event_type / f"{date_part}.jsonl"
        self._append(str(path), data)

    def _write_typed(self, event_type: str, event, legacy_filename: str | None = None) -> None:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        bot_id = str(data.get("bot_id") or metadata.get("bot_id") or "")
        canonical = canonical_event_envelope(
            event_type,
            data,
            bot_id=bot_id,
            source={"sink": "jsonl", "legacy_file": legacy_filename or ""},
        )
        self._append_canonical(event_type, canonical)
        if legacy_filename is not None:
            self._append(legacy_filename, data)

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        self._write_typed("trade", event, _LEGACY_FILES["trade"])

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        self._write_typed("missed_opportunity", event, _LEGACY_FILES["missed_opportunity"])

    def write_daily(self, event: DailySnapshot) -> None:
        self._write_typed("daily_snapshot", event, _LEGACY_FILES["daily_snapshot"])

    def write_error(self, event: ErrorEvent) -> None:
        self._write_typed("error", event, _LEGACY_FILES["error"])

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        self._write_typed("pipeline_funnel", event, _LEGACY_FILES["pipeline_funnel"])

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        self._write_typed("heartbeat", event, _LEGACY_FILES["heartbeat"])

    def write_event(self, event_type: str, event) -> None:
        self._write_typed(event_type, event)


class InMemorySink:
    """Collects events in lists. Used for backtest analysis and testing."""

    def __init__(self) -> None:
        self.trades: list[InstrumentedTradeEvent] = []
        self.missed: list[MissedOpportunityEvent] = []
        self.daily: list[DailySnapshot] = []
        self.errors: list[ErrorEvent] = []
        self.funnels: list[PipelineFunnelSnapshot] = []
        self.health_reports: list[HealthReportSnapshot] = []
        self.events_by_type: dict[str, list] = {}

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        self.trades.append(event)
        self.events_by_type.setdefault("trade", []).append(event)

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        self.missed.append(event)
        self.events_by_type.setdefault("missed_opportunity", []).append(event)

    def write_daily(self, event: DailySnapshot) -> None:
        self.daily.append(event)
        self.events_by_type.setdefault("daily_snapshot", []).append(event)

    def write_error(self, event: ErrorEvent) -> None:
        self.errors.append(event)
        self.events_by_type.setdefault("error", []).append(event)

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        self.funnels.append(event)
        self.events_by_type.setdefault("pipeline_funnel", []).append(event)

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        self.health_reports.append(event)
        self.events_by_type.setdefault("heartbeat", []).append(event)

    def write_event(self, event_type: str, event) -> None:
        self.events_by_type.setdefault(event_type, []).append(event)
