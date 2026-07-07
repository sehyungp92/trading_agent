"""SQLite-backed event queue with idempotent deduplication and dead-letter support.

Every event has a deterministic event_id (hash of bot_id + timestamp + type + payload_key).
Duplicate inserts are silently ignored via INSERT OR IGNORE.

Failed events are retried up to max_retries times before being moved to dead_letter status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import aiosqlite

from trading_assistant.orchestrator.db.connection import create_connection, initialize_schema


@dataclass
class BatchResult:
    inserted: int
    duplicates: int


@dataclass(frozen=True)
class EventInsertClassification:
    event_id: str
    classification: str


class EventQueue:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await create_connection(self._db_path)
        await initialize_schema(self._db)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Call initialize() first"
        return self._db

    async def enqueue(self, event: dict) -> bool:
        """Insert a single event. Returns True if inserted, False if duplicate."""
        cursor = await self.db.execute(
            """INSERT OR IGNORE INTO events
               (event_id, bot_id, event_type, payload, exchange_timestamp, received_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event["event_id"],
                event["bot_id"],
                event["event_type"],
                event["payload"],
                event["exchange_timestamp"],
                event["received_at"],
            ),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def enqueue_batch(self, events: list[dict]) -> BatchResult:
        """Insert a batch of events with idempotent dedup. Returns counts.

        P2-5: uses executemany so a 1000-event relay drain incurs one
        round-trip instead of 1000.
        """
        if not events:
            return BatchResult(inserted=0, duplicates=0)

        rows = [
            (
                event["event_id"],
                event["bot_id"],
                event["event_type"],
                event["payload"],
                event["exchange_timestamp"],
                event["received_at"],
            )
            for event in events
        ]
        # changes() reflects total mutations; capture the delta around
        # executemany since aiosqlite doesn't surface per-statement rowcount
        # for executemany reliably across versions.
        before_cursor = await self.db.execute("SELECT changes()")
        before_row = await before_cursor.fetchone()
        before_total = await self._total_changes()

        await self.db.executemany(
            """INSERT OR IGNORE INTO events
               (event_id, bot_id, event_type, payload, exchange_timestamp, received_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self.db.commit()

        after_total = await self._total_changes()
        inserted = max(0, after_total - before_total)
        # Defensive cap: inserted cannot exceed batch size.
        inserted = min(inserted, len(events))
        del before_row  # cursor cleanup
        return BatchResult(inserted=inserted, duplicates=len(events) - inserted)

    async def enqueue_batch_classified(
        self,
        events: list[dict],
    ) -> list[EventInsertClassification]:
        """Insert events and return a per-event enqueue/duplicate outcome."""
        if not events:
            return []

        first_events: list[dict] = []
        seen_first_ids: set[str] = set()
        for event in events:
            if event["event_id"] in seen_first_ids:
                continue
            seen_first_ids.add(event["event_id"])
            first_events.append(event)

        inserted_ids: set[str] = set()
        chunk_size = 100
        for offset in range(0, len(first_events), chunk_size):
            chunk = first_events[offset:offset + chunk_size]
            placeholders = ", ".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
            params: list[object] = []
            for event in chunk:
                params.extend(
                    (
                        event["event_id"],
                        event["bot_id"],
                        event["event_type"],
                        event["payload"],
                        event["exchange_timestamp"],
                        event["received_at"],
                    )
                )
            cursor = await self.db.execute(
                f"""INSERT OR IGNORE INTO events
                    (event_id, bot_id, event_type, payload, exchange_timestamp, received_at)
                    VALUES {placeholders}
                    RETURNING event_id""",
                params,
            )
            rows = await cursor.fetchall()
            inserted_ids.update(str(row["event_id"]) for row in rows)
        await self.db.commit()

        outcomes: list[EventInsertClassification] = []
        seen_in_batch: set[str] = set()
        for event in events:
            event_id = event["event_id"]
            classification = (
                "enqueued"
                if event_id in inserted_ids and event_id not in seen_in_batch
                else "duplicate"
            )
            outcomes.append(
                EventInsertClassification(
                    event_id=event_id,
                    classification=classification,
                )
            )
            seen_in_batch.add(event_id)
        return outcomes

    async def _total_changes(self) -> int:
        cursor = await self.db.execute("SELECT total_changes()")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def claim(self, limit: int = 10) -> list[dict]:
        """Atomically claim pending events for processing.

        Moves events from 'pending' to 'processing' and sets processed_at.
        Claimed events won't be returned by subsequent claim() or peek() calls.
        Use this instead of peek() when processing events to prevent duplicates.
        """
        cursor = await self.db.execute(
            """UPDATE events
               SET status = 'processing', processed_at = datetime('now')
               WHERE event_id IN (
                   SELECT event_id FROM events
                   WHERE status = 'pending'
                   ORDER BY created_at ASC
                   LIMIT ?
               )
               RETURNING *""",
            (limit,),
        )
        rows = await cursor.fetchall()
        await self.db.commit()
        return [dict(row) for row in rows]

    async def peek(self, limit: int = 10) -> list[dict]:
        """Get pending events without changing their status (display-only)."""
        cursor = await self.db.execute(
            "SELECT * FROM events WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def count_pending(self) -> int:
        """Count events with pending status."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM events WHERE status = 'pending'")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def oldest_pending_age_seconds(self) -> float:
        """Return seconds since the oldest pending event was enqueued (P2-8).

        Returns 0.0 when the queue has no pending events. Uses julianday
        differential so the value reflects wall-clock age regardless of
        process restarts.
        """
        cursor = await self.db.execute(
            """SELECT (julianday('now') - julianday(MIN(created_at))) * 86400
               FROM events WHERE status = 'pending'"""
        )
        row = await cursor.fetchone()
        if not row or row[0] is None:
            return 0.0
        return float(row[0])

    async def ack(self, event_id: str) -> None:
        """Mark an event as acknowledged/processed."""
        await self.db.execute(
            "UPDATE events SET status = 'acked', processed_at = datetime('now') WHERE event_id = ?",
            (event_id,),
        )
        await self.db.commit()

    async def requeue(self, event_id: str) -> None:
        """Reset an event to pending without incrementing retry_count.

        Use for transient back-pressure (e.g. subagent capacity) where the
        event should be retried but the failure should not count against the
        dead-letter budget.
        """
        await self.db.execute(
            "UPDATE events SET status = 'pending', processed_at = NULL WHERE event_id = ?",
            (event_id,),
        )
        await self.db.commit()

    async def nack(self, event_id: str, error: str) -> bool:
        """Increment retry count and record error. Returns True if moved to dead_letter."""
        await self.db.execute(
            """UPDATE events
               SET retry_count = retry_count + 1,
                   last_error = ?,
                   status = CASE
                     WHEN retry_count + 1 >= max_retries THEN 'dead_letter'
                     ELSE 'pending'
                   END
               WHERE event_id = ?""",
            (error, event_id),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT status FROM events WHERE event_id = ?", (event_id,),
        )
        row = await cursor.fetchone()
        return row is not None and row["status"] == "dead_letter"

    async def count_dead_letters(self) -> int:
        """Count events in dead-letter status."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM events WHERE status = 'dead_letter'")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_dead_letters(self, limit: int = 50) -> list[dict]:
        """Get events that have exhausted retries."""
        cursor = await self.db.execute(
            "SELECT * FROM events WHERE status = 'dead_letter' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get(self, event_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def quarantine_relay_event(
        self,
        *,
        source: str,
        raw_event_id: str,
        reason: str,
        payload: object,
    ) -> None:
        """Persist a malformed relay event so it does not block later events."""
        await self.db.execute(
            """INSERT INTO relay_quarantine (source, raw_event_id, reason, payload)
               VALUES (?, ?, ?, ?)""",
            (
                source,
                raw_event_id,
                reason,
                json.dumps(payload, default=str),
            ),
        )
        await self.db.commit()

    async def record_relay_ingest_classification(
        self,
        *,
        source: str,
        raw_event_id: str,
        event_id: str,
        classification: str,
        payload: object,
        reason: str = "",
    ) -> None:
        await self.db.execute(
            """INSERT INTO relay_ingest_classifications
               (source, raw_event_id, event_id, classification, reason, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                source,
                raw_event_id,
                event_id,
                classification,
                reason,
                json.dumps(payload, default=str),
            ),
        )
        await self.db.commit()

    async def get_relay_ingest_classifications(self, limit: int = 50) -> list[dict]:
        cursor = await self.db.execute(
            """SELECT * FROM relay_ingest_classifications
               ORDER BY created_at DESC, id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_relay_quarantine(self, limit: int = 50) -> list[dict]:
        cursor = await self.db.execute(
            """SELECT * FROM relay_quarantine
               ORDER BY created_at DESC, id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def reprocess_dead_letter(self, event_id: str) -> bool:
        """Move a dead-letter event back to pending with reset retry count."""
        cursor = await self.db.execute(
            """UPDATE events
               SET status = 'pending', retry_count = 0, last_error = NULL
               WHERE event_id = ? AND status = 'dead_letter'""",
            (event_id,),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def recover_stale(self, timeout_seconds: int = 3600) -> int:
        """Reset events stuck in 'processing' status beyond the timeout back to 'pending'."""
        cursor = await self.db.execute(
            """UPDATE events
               SET status = 'pending'
               WHERE status = 'processing'
                 AND processed_at IS NOT NULL
                 AND (julianday('now') - julianday(processed_at)) * 86400 > ?""",
            (timeout_seconds,),
        )
        await self.db.commit()
        return cursor.rowcount

    async def update_watermark(self, bot_id: str, last_event_id: str) -> None:
        """Update the watermark for relay pull protocol."""
        await self.db.execute(
            """INSERT INTO watermarks (bot_id, last_event_id, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(bot_id) DO UPDATE SET
                 last_event_id = excluded.last_event_id,
                 updated_at = excluded.updated_at""",
            (bot_id, last_event_id),
        )
        await self.db.commit()

    async def get_watermark(self, bot_id: str) -> str | None:
        """Get the last acked event_id for a bot."""
        cursor = await self.db.execute(
            "SELECT last_event_id FROM watermarks WHERE bot_id = ?",
            (bot_id,),
        )
        row = await cursor.fetchone()
        return row["last_event_id"] if row else None
