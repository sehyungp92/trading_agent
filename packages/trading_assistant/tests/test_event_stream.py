"""Tests for SSE event stream (H4)."""

from __future__ import annotations

import asyncio


from trading_assistant.orchestrator.event_stream import EventStream


class TestEventStream:
    def test_broadcast_increments_sequence(self):
        stream = EventStream()
        e1 = stream.broadcast("test", {"key": "val1"})
        e2 = stream.broadcast("test", {"key": "val2"})
        assert e1.sequence == 1
        assert e2.sequence == 2

    def test_ring_buffer_limits_size(self):
        stream = EventStream(buffer_size=3)
        for i in range(5):
            stream.broadcast("test", {"i": i})
        recent = stream.get_recent()
        assert len(recent) == 3
        assert recent[0].data["i"] == 2

    def test_get_recent_since_sequence(self):
        stream = EventStream()
        stream.broadcast("a")
        stream.broadcast("b")
        stream.broadcast("c")
        recent = stream.get_recent(since_sequence=1)
        assert len(recent) == 2
        assert recent[0].event_type == "b"

    async def test_subscriber_receives_events(self):
        stream = EventStream()
        q = stream.subscribe()
        stream.broadcast("test", {"data": 1})
        event = await asyncio.wait_for(q.get(), timeout=1.0)
        assert event.event_type == "test"
        assert event.data["data"] == 1

    async def test_unsubscribe_stops_events(self):
        stream = EventStream()
        q = stream.subscribe()
        stream.unsubscribe(q)
        stream.broadcast("test")
        assert q.empty()

    async def test_multiple_subscribers(self):
        stream = EventStream()
        q1 = stream.subscribe()
        q2 = stream.subscribe()
        stream.broadcast("test")
        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert e1.sequence == e2.sequence
