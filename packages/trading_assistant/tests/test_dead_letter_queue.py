"""Tests for dead-letter queue with retry semantics (H1)."""
from __future__ import annotations

import pytest

from trading_assistant.orchestrator.db.queue import EventQueue


def _make_event(event_id: str = "evt-1", bot_id: str = "bot-a", event_type: str = "trade") -> dict:
    return {
        "event_id": event_id,
        "bot_id": bot_id,
        "event_type": event_type,
        "payload": "{}",
        "exchange_timestamp": "2026-01-01T00:00:00Z",
        "received_at": "2026-01-01T00:00:01Z",
    }


@pytest.fixture
async def queue(tmp_path):
    q = EventQueue(db_path=str(tmp_path / "test.db"))
    await q.initialize()
    yield q
    await q.close()


class TestNack:
    async def test_nack_increments_retry_count(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        is_dead = await queue.nack("evt-1", "some error")
        assert is_dead is False

        cursor = await queue.db.execute("SELECT retry_count, last_error, status FROM events WHERE event_id = 'evt-1'")
        row = await cursor.fetchone()
        assert row["retry_count"] == 1
        assert row["last_error"] == "some error"
        assert row["status"] == "pending"

    async def test_nack_moves_to_dead_letter_after_max_retries(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        # Default max_retries=3, so nack 3 times
        await queue.nack("evt-1", "error 1")
        await queue.nack("evt-1", "error 2")
        is_dead = await queue.nack("evt-1", "error 3")
        assert is_dead is True

        cursor = await queue.db.execute("SELECT status, retry_count FROM events WHERE event_id = 'evt-1'")
        row = await cursor.fetchone()
        assert row["status"] == "dead_letter"
        assert row["retry_count"] == 3

    async def test_nack_updates_last_error_each_time(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        await queue.nack("evt-1", "first error")
        await queue.nack("evt-1", "second error")

        cursor = await queue.db.execute("SELECT last_error FROM events WHERE event_id = 'evt-1'")
        row = await cursor.fetchone()
        assert row["last_error"] == "second error"

    async def test_dead_letter_event_not_returned_by_peek(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        for i in range(3):
            await queue.nack("evt-1", f"error {i}")

        events = await queue.peek()
        assert len(events) == 0


class TestGetDeadLetters:
    async def test_returns_dead_letter_events(self, queue: EventQueue):
        await queue.enqueue(_make_event("evt-1"))
        await queue.enqueue(_make_event("evt-2", bot_id="bot-b"))

        # Kill evt-1
        for i in range(3):
            await queue.nack("evt-1", f"error {i}")

        dead = await queue.get_dead_letters()
        assert len(dead) == 1
        assert dead[0]["event_id"] == "evt-1"

    async def test_returns_empty_when_no_dead_letters(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        dead = await queue.get_dead_letters()
        assert dead == []


class TestReprocessDeadLetter:
    async def test_reprocess_moves_back_to_pending(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        for i in range(3):
            await queue.nack("evt-1", f"error {i}")

        success = await queue.reprocess_dead_letter("evt-1")
        assert success is True

        cursor = await queue.db.execute("SELECT status, retry_count, last_error FROM events WHERE event_id = 'evt-1'")
        row = await cursor.fetchone()
        assert row["status"] == "pending"
        assert row["retry_count"] == 0
        assert row["last_error"] is None

    async def test_reprocess_returns_false_for_nonexistent(self, queue: EventQueue):
        success = await queue.reprocess_dead_letter("nonexistent")
        assert success is False

    async def test_reprocess_only_affects_dead_letter_events(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        # Event is pending, not dead_letter
        success = await queue.reprocess_dead_letter("evt-1")
        assert success is False


class TestRecoverStale:
    async def test_recover_stale_resets_processing_events(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        # Manually set to processing with old processed_at
        await queue.db.execute(
            "UPDATE events SET status = 'processing', processed_at = datetime('now', '-2 hours') WHERE event_id = 'evt-1'"
        )
        await queue.db.commit()

        recovered = await queue.recover_stale(timeout_seconds=3600)
        assert recovered == 1

        events = await queue.peek()
        assert len(events) == 1
        assert events[0]["event_id"] == "evt-1"

    async def test_recover_stale_ignores_recent_processing(self, queue: EventQueue):
        await queue.enqueue(_make_event())
        await queue.db.execute(
            "UPDATE events SET status = 'processing', processed_at = datetime('now') WHERE event_id = 'evt-1'"
        )
        await queue.db.commit()

        recovered = await queue.recover_stale(timeout_seconds=3600)
        assert recovered == 0
