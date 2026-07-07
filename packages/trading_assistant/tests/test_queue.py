import json
from datetime import datetime, timezone

import pytest

from trading_assistant.orchestrator.db.queue import EventQueue


@pytest.fixture
async def queue(tmp_db_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_db_path))
    await q.initialize()
    return q


def _make_event(event_id: str = "abc123", bot_id: str = "bot1", event_type: str = "trade") -> dict:
    return {
        "event_id": event_id,
        "bot_id": bot_id,
        "event_type": event_type,
        "payload": json.dumps({"trade_id": "t001", "pnl": 50.0}),
        "exchange_timestamp": datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc).isoformat(),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


class TestEventQueue:
    async def test_enqueue_and_dequeue(self, queue: EventQueue):
        event = _make_event()
        inserted = await queue.enqueue(event)
        assert inserted is True

        pending = await queue.peek(limit=10)
        assert len(pending) == 1
        assert pending[0]["event_id"] == "abc123"

    async def test_idempotent_insert(self, queue: EventQueue):
        """Inserting the same event_id twice does not create duplicates."""
        event = _make_event(event_id="dup001")
        first = await queue.enqueue(event)
        second = await queue.enqueue(event)
        assert first is True
        assert second is False  # duplicate silently ignored

        pending = await queue.peek(limit=10)
        assert len(pending) == 1

    async def test_ack_removes_from_pending(self, queue: EventQueue):
        event = _make_event(event_id="ack001")
        await queue.enqueue(event)
        await queue.ack("ack001")

        pending = await queue.peek(limit=10)
        assert len(pending) == 0

    async def test_batch_enqueue(self, queue: EventQueue):
        events = [_make_event(event_id=f"batch{i}") for i in range(5)]
        result = await queue.enqueue_batch(events)
        assert result.inserted == 5
        assert result.duplicates == 0

        pending = await queue.peek(limit=10)
        assert len(pending) == 5

    async def test_batch_enqueue_with_duplicates(self, queue: EventQueue):
        events = [_make_event(event_id="same")] * 3
        result = await queue.enqueue_batch(events)
        assert result.inserted == 1
        assert result.duplicates == 2

    async def test_batch_enqueue_classified_preserves_per_event_outcome(self, queue: EventQueue):
        await queue.enqueue(_make_event(event_id="existing"))
        events = [
            _make_event(event_id="new"),
            _make_event(event_id="existing"),
            _make_event(event_id="new"),
        ]

        result = await queue.enqueue_batch_classified(events)

        assert [(item.event_id, item.classification) for item in result] == [
            ("new", "enqueued"),
            ("existing", "duplicate"),
            ("new", "duplicate"),
        ]

    async def test_schema_has_pending_order_index(self, queue: EventQueue):
        cursor = await queue.db.execute("PRAGMA index_list(events)")
        rows = await cursor.fetchall()
        index_names = {row["name"] for row in rows}
        assert "idx_events_status_created_at" in index_names

    async def test_pending_order_queries_use_composite_index(self, queue: EventQueue):
        for i in range(40):
            await queue.enqueue(_make_event(event_id=f"pending-{i:03d}"))
        for i in range(40):
            event_id = f"acked-{i:03d}"
            await queue.enqueue(_make_event(event_id=event_id))
            await queue.ack(event_id)

        cursor = await queue.db.execute(
            """EXPLAIN QUERY PLAN
               SELECT event_id FROM events
               WHERE status = 'pending'
               ORDER BY created_at ASC
               LIMIT 10"""
        )
        claim_plan = " ".join(row["detail"] for row in await cursor.fetchall())
        cursor = await queue.db.execute(
            """EXPLAIN QUERY PLAN
               SELECT * FROM events
               WHERE status = 'pending'
               ORDER BY created_at ASC
               LIMIT 10"""
        )
        peek_plan = " ".join(row["detail"] for row in await cursor.fetchall())

        assert "idx_events_status_created_at" in claim_plan
        assert "idx_events_status_created_at" in peek_plan

    async def test_relay_quarantine_roundtrip(self, queue: EventQueue):
        await queue.quarantine_relay_event(
            source="relay",
            raw_event_id="bad1",
            reason="missing payload",
            payload={"event_id": "bad1"},
        )

        rows = await queue.get_relay_quarantine()
        assert len(rows) == 1
        assert rows[0]["source"] == "relay"
        assert rows[0]["raw_event_id"] == "bad1"
        assert rows[0]["reason"] == "missing payload"

    async def test_relay_ingest_classification_roundtrip(self, queue: EventQueue):
        await queue.record_relay_ingest_classification(
            source="relay",
            raw_event_id="raw-1",
            event_id="event-1",
            classification="duplicate",
            reason="already present",
            payload={"event_id": "event-1"},
        )

        rows = await queue.get_relay_ingest_classifications()
        assert len(rows) == 1
        assert rows[0]["source"] == "relay"
        assert rows[0]["raw_event_id"] == "raw-1"
        assert rows[0]["event_id"] == "event-1"
        assert rows[0]["classification"] == "duplicate"

    async def test_peek_respects_limit(self, queue: EventQueue):
        for i in range(10):
            await queue.enqueue(_make_event(event_id=f"limit{i}"))

        pending = await queue.peek(limit=3)
        assert len(pending) == 3

    async def test_claim_returns_and_marks_processing(self, queue: EventQueue):
        """claim() should return events and mark them as 'processing'."""
        await queue.enqueue(_make_event(event_id="c1"))
        await queue.enqueue(_make_event(event_id="c2"))
        claimed = await queue.claim(limit=10)
        assert len(claimed) == 2
        assert all(c["status"] == "processing" for c in claimed)
        # No more pending
        assert await queue.count_pending() == 0

    async def test_claim_is_exclusive(self, queue: EventQueue):
        """Two sequential claims should get disjoint sets."""
        for i in range(4):
            await queue.enqueue(_make_event(event_id=f"ex{i}"))
        first = await queue.claim(limit=2)
        second = await queue.claim(limit=2)
        first_ids = {e["event_id"] for e in first}
        second_ids = {e["event_id"] for e in second}
        assert len(first_ids) == 2
        assert len(second_ids) == 2
        assert first_ids.isdisjoint(second_ids)

    async def test_claim_respects_limit(self, queue: EventQueue):
        """claim() should respect the limit parameter."""
        for i in range(5):
            await queue.enqueue(_make_event(event_id=f"lim{i}"))
        claimed = await queue.claim(limit=2)
        assert len(claimed) == 2
        # 3 still pending
        assert await queue.count_pending() == 3

    async def test_claimed_events_invisible_to_peek(self, queue: EventQueue):
        """After claim(), those events should not appear in peek()."""
        await queue.enqueue(_make_event(event_id="vis1"))
        await queue.enqueue(_make_event(event_id="vis2"))
        await queue.claim(limit=1)
        pending = await queue.peek(limit=10)
        assert len(pending) == 1
        assert pending[0]["event_id"] == "vis2"

    async def test_watermark_tracking(self, queue: EventQueue):
        """Watermark tracks the latest acked event for a bot."""
        await queue.enqueue(_make_event(event_id="w1", bot_id="bot1"))
        await queue.enqueue(_make_event(event_id="w2", bot_id="bot1"))
        await queue.ack("w1")
        await queue.update_watermark("bot1", "w1")

        wm = await queue.get_watermark("bot1")
        assert wm == "w1"
