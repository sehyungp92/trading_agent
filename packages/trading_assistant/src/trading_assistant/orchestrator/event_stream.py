"""In-memory SSE event stream with ring buffer for client catch-up (H4)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from pathlib import Path

from trading_assistant.schemas.stream_events import StreamEvent

logger = logging.getLogger(__name__)


class EventStream:
    """In-memory event stream with ring buffer for client catch-up."""

    def __init__(self, buffer_size: int = 100) -> None:
        self._buffer: deque[StreamEvent] = deque(maxlen=buffer_size)
        self._sequence = 0
        self._subscribers: list[asyncio.Queue[StreamEvent]] = []

    @property
    def buffer_size(self) -> int:
        return self._buffer.maxlen  # type: ignore[return-value]

    @property
    def sequence(self) -> int:
        return self._sequence

    def broadcast(self, event_type: str, data: dict | None = None) -> StreamEvent:
        """Broadcast an event to all subscribers and add to ring buffer."""
        self._sequence += 1
        event = StreamEvent(sequence=self._sequence, event_type=event_type, data=data or {})
        self._buffer.append(event)
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop events for slow consumers
        return event

    def get_recent(self, since_sequence: int = 0) -> list[StreamEvent]:
        """Get buffered events since a sequence number (for client catch-up)."""
        return [e for e in self._buffer if e.sequence > since_sequence]

    def subscribe(self) -> asyncio.Queue[StreamEvent]:
        """Create a new subscriber queue."""
        q: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=50)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[StreamEvent]) -> None:
        """Remove a subscriber queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)


class AuditTrailConsumer:
    """Subscribes to EventStream and writes events to date-stamped JSONL files."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._queue: asyncio.Queue[StreamEvent] | None = None
        self._task: asyncio.Task | None = None
        self._event_stream: EventStream | None = None

    def start(self, event_stream: EventStream) -> None:
        """Subscribe to the event stream and start background writing."""
        self._event_stream = event_stream
        self._queue = event_stream.subscribe()
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create audit log directory: %s", self._log_dir)
        self._task = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        """Unsubscribe and cancel the background task."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._event_stream and self._queue:
            self._event_stream.unsubscribe(self._queue)
        self._queue = None
        self._task = None
        self._event_stream = None

    async def _consume(self) -> None:
        """Background loop: read events from queue and write to JSONL."""
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                self._write_event(event)
        except asyncio.CancelledError:
            raise

    def _write_event(self, event: StreamEvent) -> None:
        """Append a single event to the date-stamped JSONL file."""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            path = self._log_dir / f"events-{date_str}.jsonl"
            record = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "data": event.data,
                "sequence": event.sequence,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.warning("Audit trail write failed for event %d", event.sequence)
