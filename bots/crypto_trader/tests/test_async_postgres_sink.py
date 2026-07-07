from __future__ import annotations

import json
import queue
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from crypto_trader.instrumentation.async_postgres_sink import AsyncPostgresSink
from crypto_trader.instrumentation.postgres_backfill import replay_jsonl_to_postgres
from crypto_trader.instrumentation.types import EventMetadata, GenericInstrumentationEvent


class FakeWriter:
    instances: list["FakeWriter"] = []

    def __init__(self, dsn: str, error_callback=None) -> None:
        self.dsn = dsn
        self.error_callback = error_callback
        self.events: list[tuple[str, object]] = []
        self.trades: list[object] = []
        self.closed = False
        FakeWriter.instances.append(self)

    def write_event(self, event_type: str, event) -> None:
        self.events.append((event_type, event))

    def write_events_batch(self, event_type: str, events: list[object]) -> None:
        for event in events:
            self.write_event(event_type, event)

    def write_trade(self, event) -> None:
        self.trades.append(event)
        self.write_event("trade", event)

    def write_daily(self, event) -> None:
        self.write_event("daily_snapshot", event)

    def write_health_report(self, event) -> None:
        self.write_event("heartbeat", event)

    def write_missed(self, event) -> None:
        self.write_event("missed_opportunity", event)

    def write_error(self, event) -> None:
        self.write_event("error", event)

    def write_funnel(self, event) -> None:
        self.write_event("pipeline_funnel", event)

    def write_equity(self, equity, timestamp) -> None:
        self.events.append(("equity_snapshot", {"equity": equity, "timestamp": timestamp}))

    def upsert_positions(self, positions) -> None:
        self.events.append(("positions", positions))

    def upsert_strategy_position_allocations(self, allocations) -> None:
        self.events.append(("strategy_position_allocations", allocations))

    def upsert_exchange_positions(self, positions) -> None:
        self.events.append(("exchange_positions", positions))

    def close(self) -> None:
        self.closed = True


class SlowWriter(FakeWriter):
    def write_event(self, event_type: str, event) -> None:
        time.sleep(0.25)
        super().write_event(event_type, event)


class SlowInitWriter(FakeWriter):
    def __init__(self, dsn: str, error_callback=None) -> None:
        time.sleep(0.25)
        super().__init__(dsn, error_callback=error_callback)


class FailingWriter(FakeWriter):
    def __init__(self, dsn: str, error_callback=None) -> None:
        raise RuntimeError("db down")


class CallbackFailWriter(FakeWriter):
    def write_event(self, event_type: str, event) -> None:
        if self.error_callback is not None:
            self.error_callback({
                "event_type": event_type,
                "error_type": "SyntheticWriteFailure",
                "message": "writer swallowed failure",
            })
        super().write_event(event_type, event)


def _event(event_type: str = "order_intent") -> GenericInstrumentationEvent:
    ts = datetime(2026, 6, 4, tzinfo=timezone.utc)
    return GenericInstrumentationEvent(
        metadata=EventMetadata.create(
            bot_id="bot",
            strategy_id="momentum",
            exchange_ts=ts,
            event_type=event_type,
            payload_key=f"{event_type}:1",
        ),
        payload={"symbol": "BTC", "event_type": event_type},
    )


def test_enqueue_returns_without_waiting_for_slow_writer() -> None:
    FakeWriter.instances.clear()
    sink = AsyncPostgresSink("postgres://test", writer_factory=SlowWriter, queue_capacity=10)

    started = time.perf_counter()
    sink.write_event("order_intent", _event())
    elapsed = time.perf_counter() - started
    sink.close(flush_timeout_sec=1.0)

    assert elapsed < 0.1
    assert FakeWriter.instances[0].events


def test_queue_drops_verbose_before_critical() -> None:
    FakeWriter.instances.clear()
    errors: list[dict] = []
    sink = AsyncPostgresSink(
        "postgres://test",
        writer_factory=SlowInitWriter,
        queue_capacity=1,
        error_callback=errors.append,
    )

    sink.write_funnel(_event("pipeline_funnel"))
    sink.write_error(_event("error"))
    sink.close(flush_timeout_sec=1.0)

    metrics = sink.metrics()
    assert metrics["dropped_by_priority"]["verbose"] >= 1
    assert metrics["dropped_by_priority"]["critical"] == 0


@pytest.mark.parametrize("priority", ["critical", "normal"])
def test_enqueue_preemption_race_records_incoming_drop(monkeypatch, priority: str) -> None:
    errors: list[dict] = []
    sink = AsyncPostgresSink(
        "postgres://test",
        writer_factory=FakeWriter,
        queue_capacity=1,
        error_callback=errors.append,
    )
    put_attempts = 0

    def full_put(_job) -> None:
        nonlocal put_attempts
        put_attempts += 1
        raise queue.Full

    monkeypatch.setattr(sink._queue, "put_nowait", full_put)
    monkeypatch.setattr(sink, "_drop_one_verbose_job", lambda: True)

    if priority == "critical":
        sink.write_error(_event("error"))
    else:
        sink.write_event("order_intent", _event("order_intent"))

    metrics = sink.metrics()
    assert put_attempts == 2
    assert metrics["jobs_enqueued"] == 0
    assert metrics["dropped_by_priority"][priority] == 1
    assert metrics["queue_depth"] == 0
    if priority == "critical":
        assert errors
        assert errors[0]["severity"] == "critical"
        assert "queue_full_after_preempt" in errors[0]["message"]
    else:
        assert errors == []


def test_worker_init_failure_is_fail_visible() -> None:
    errors: list[dict] = []
    sink = AsyncPostgresSink(
        "postgres://test",
        writer_factory=FailingWriter,
        queue_capacity=10,
        error_callback=errors.append,
    )

    sink.write_event("order_intent", _event())
    time.sleep(0.1)

    assert errors
    metrics = sink.metrics()
    assert metrics["write_failures"] >= 1
    assert metrics["jobs_dropped"] >= 1
    assert metrics["queue_depth"] == 0
    assert metrics["worker_alive"] is False


def test_enqueue_freezes_mutable_dict_payload() -> None:
    FakeWriter.instances.clear()
    sink = AsyncPostgresSink("postgres://test", writer_factory=SlowWriter, queue_capacity=10)
    payload = {"nested": {"value": "before"}}

    sink.write_event("order_intent", payload)
    payload["nested"]["value"] = "after"
    sink.close(flush_timeout_sec=1.0)

    written = FakeWriter.instances[0].events[0][1]
    assert written["nested"]["value"] == "before"


def test_writer_error_callback_updates_async_metrics() -> None:
    errors: list[dict] = []
    sink = AsyncPostgresSink(
        "postgres://test",
        writer_factory=CallbackFailWriter,
        queue_capacity=10,
        error_callback=errors.append,
    )

    sink.write_event("order_intent", _event())
    sink.close(flush_timeout_sec=1.0)

    metrics = sink.metrics()
    assert metrics["jobs_written"] == 1
    assert metrics["write_failures"] >= 1
    assert "writer swallowed failure" in metrics["last_error"]
    assert errors


def test_shutdown_drains_and_closes_writer() -> None:
    FakeWriter.instances.clear()
    sink = AsyncPostgresSink("postgres://test", writer_factory=FakeWriter, queue_capacity=10)

    sink.write_equity(10_000.0, datetime.now(timezone.utc))
    sink.upsert_positions([{"symbol": "BTC", "direction": "LONG", "qty": 0.1, "avg_entry": 90_000.0}])
    sink.close(flush_timeout_sec=1.0)

    writer = FakeWriter.instances[0]
    assert writer.closed is True
    assert sink.metrics()["jobs_written"] >= 2


def test_jsonl_backfill_replays_events_idempotently(tmp_path: Path) -> None:
    FakeWriter.instances.clear()
    event_dir = tmp_path / "instrumentation" / "events" / "order_intent"
    event_dir.mkdir(parents=True)
    payload = _event("order_intent").to_dict()
    (event_dir / "2026-06-04.jsonl").write_text(
        json.dumps(payload) + "\n" + json.dumps(payload) + "\n",
        encoding="utf-8",
    )

    result = replay_jsonl_to_postgres(tmp_path, "postgres://test", writer_factory=FakeWriter)

    assert result.events_seen == 2
    assert result.events_replayed == 2
    assert FakeWriter.instances[0].events[0][0] == "order_intent"


def test_live_engine_registers_async_postgres_sink_by_default(tmp_path: Path) -> None:
    from crypto_trader.live.config import LiveConfig
    from crypto_trader.live.engine import LiveEngine

    engine = LiveEngine(LiveConfig(
        state_dir=tmp_path,
        data_dir=tmp_path / "data",
        bot_id="bot",
        postgres_dsn="postgres://test",
    ))
    try:
        assert isinstance(engine._pg_sink, AsyncPostgresSink)
        assert [type(sink).__name__ for sink in engine._emitter._sinks].count("AsyncPostgresSink") == 1
        assert "PostgresSink" not in [type(sink).__name__ for sink in engine._emitter._sinks]
    finally:
        engine._oms.close()
