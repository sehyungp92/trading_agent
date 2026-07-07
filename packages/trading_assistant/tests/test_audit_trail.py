"""Tests for EventStream audit trail consumer (A4)."""
from __future__ import annotations

import asyncio
import json

import pytest

from trading_assistant.orchestrator.event_stream import AuditTrailConsumer, EventStream


@pytest.fixture
def event_stream():
    return EventStream()


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


@pytest.mark.asyncio
async def test_consumer_writes_events_to_jsonl(event_stream, log_dir):
    consumer = AuditTrailConsumer(log_dir=log_dir)
    consumer.start(event_stream)

    event_stream.broadcast("test_event", {"key": "value"})
    await asyncio.sleep(0)  # let consumer process

    await consumer.stop()

    files = list(log_dir.glob("events-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "test_event"
    assert record["data"] == {"key": "value"}


@pytest.mark.asyncio
async def test_file_named_by_date(event_stream, log_dir):
    consumer = AuditTrailConsumer(log_dir=log_dir)
    consumer.start(event_stream)

    ev = event_stream.broadcast("date_test", {})
    date_str = ev.timestamp.strftime("%Y-%m-%d")
    await asyncio.sleep(0)

    await consumer.stop()

    expected_file = log_dir / f"events-{date_str}.jsonl"
    assert expected_file.exists()


@pytest.mark.asyncio
async def test_events_include_timestamp_and_sequence(event_stream, log_dir):
    consumer = AuditTrailConsumer(log_dir=log_dir)
    consumer.start(event_stream)

    event_stream.broadcast("ev1", {"n": 1})
    event_stream.broadcast("ev2", {"n": 2})
    await asyncio.sleep(0)

    await consumer.stop()

    files = list(log_dir.glob("events-*.jsonl"))
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 2

    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert "timestamp" in r1
    assert r1["sequence"] == 1
    assert r2["sequence"] == 2


@pytest.mark.asyncio
async def test_consumer_handles_multiple_events(event_stream, log_dir):
    consumer = AuditTrailConsumer(log_dir=log_dir)
    consumer.start(event_stream)

    for i in range(10):
        event_stream.broadcast(f"event_{i}", {"i": i})
    await asyncio.sleep(0)

    await consumer.stop()

    files = list(log_dir.glob("events-*.jsonl"))
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 10


@pytest.mark.asyncio
async def test_stop_cleanly_unsubscribes(event_stream, log_dir):
    consumer = AuditTrailConsumer(log_dir=log_dir)
    consumer.start(event_stream)
    assert len(event_stream._subscribers) == 1

    await consumer.stop()
    assert len(event_stream._subscribers) == 0


@pytest.mark.asyncio
async def test_graceful_on_write_errors(event_stream, tmp_path):
    # Point to a path that can't be a directory (file blocking it)
    blocker = tmp_path / "logs"
    blocker.write_text("not a dir")

    consumer = AuditTrailConsumer(log_dir=blocker / "sub")
    consumer.start(event_stream)

    event_stream.broadcast("fail_test", {})
    await asyncio.sleep(0)

    # Should not crash
    await consumer.stop()
