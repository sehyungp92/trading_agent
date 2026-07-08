"""EventEmitter — dispatches events to registered sinks."""

from __future__ import annotations

import structlog

from crypto_trader.instrumentation.sinks import Sink
from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)

log = structlog.get_logger()


class EventEmitter:
    """Dispatches instrumentation events to all registered sinks."""

    def __init__(self) -> None:
        self._sinks: list[Sink] = []
        self._sink_failures: dict[str, int] = {}

    def add_sink(self, sink: Sink) -> None:
        self._sinks.append(sink)

    @property
    def sink_failures(self) -> dict[str, int]:
        return dict(self._sink_failures)

    def emit(self, event_type: str, event) -> None:
        """Emit a generic assistant instrumentation event."""
        self._dispatch("event", lambda sink: sink.write_event(event_type, event))

    def emit_trade(self, event: InstrumentedTradeEvent) -> None:
        self._dispatch("trade", lambda sink: sink.write_trade(event))

    def emit_missed(self, event: MissedOpportunityEvent) -> None:
        self._dispatch("missed", lambda sink: sink.write_missed(event))

    def emit_daily(self, event: DailySnapshot) -> None:
        self._dispatch("daily", lambda sink: sink.write_daily(event))

    def emit_error(self, event: ErrorEvent) -> None:
        self._dispatch("error", lambda sink: sink.write_error(event))

    def emit_funnel(self, event: PipelineFunnelSnapshot) -> None:
        self._dispatch("funnel", lambda sink: sink.write_funnel(event))

    def emit_health_report(self, event: HealthReportSnapshot) -> None:
        self._dispatch("health_report", lambda sink: sink.write_health_report(event))

    def _dispatch(self, operation: str, write_fn) -> None:
        for sink in self._sinks:
            sink_name = type(sink).__name__
            try:
                write_fn(sink)
            except Exception:
                self._sink_failures[sink_name] = self._sink_failures.get(sink_name, 0) + 1
                log.exception(f"emitter.{operation}_failed", sink=sink_name)
