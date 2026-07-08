"""Tests for EventMetadata and event_id generation."""
from datetime import datetime, timezone

from strategies.swing.instrumentation.src.event_metadata import (
    compute_event_id, create_event_metadata, compute_clock_skew,
)


class TestEventMetadata:
    def test_event_id_deterministic(self):
        id1 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        id2 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        assert id1 == id2

    def test_event_id_unique_on_different_input(self):
        id1 = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc123")
        id2 = compute_event_id("bot1", "2026-03-01T10:00:01Z", "trade", "abc123")
        assert id1 != id2

    def test_event_id_length(self):
        eid = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc")
        assert len(eid) == 16

    def test_event_id_hex(self):
        eid = compute_event_id("bot1", "2026-03-01T10:00:00Z", "trade", "abc")
        assert all(c in "0123456789abcdef" for c in eid)

    def test_clock_skew_positive(self):
        exch = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        local = datetime(2026, 3, 1, 9, 59, 59, tzinfo=timezone.utc)
        skew = compute_clock_skew(exch, local)
        assert skew == 1000

    def test_clock_skew_negative(self):
        exch = datetime(2026, 3, 1, 9, 59, 59, tzinfo=timezone.utc)
        local = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        skew = compute_clock_skew(exch, local)
        assert skew == -1000

    def test_clock_skew_zero(self):
        ts = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert compute_clock_skew(ts, ts) == 0

    def test_create_event_metadata_returns_all_fields(self):
        now = datetime.now(timezone.utc)
        meta = create_event_metadata(
            bot_id="bot1",
            event_type="trade",
            payload_key="test123",
            exchange_timestamp=now,
            data_source_id="test_source",
        )
        assert meta.event_id
        assert meta.bot_id == "bot1"
        assert meta.exchange_timestamp
        assert meta.local_timestamp
        assert meta.data_source_id == "test_source"
        assert isinstance(meta.clock_skew_ms, int)

    def test_create_event_metadata_with_bar_id(self):
        now = datetime.now(timezone.utc)
        meta = create_event_metadata(
            bot_id="bot1",
            event_type="snapshot",
            payload_key="snap1",
            exchange_timestamp=now,
            data_source_id="test",
            bar_id="2026-03-01T14:00Z_1h",
        )
        assert meta.bar_id == "2026-03-01T14:00Z_1h"

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        meta = create_event_metadata(
            bot_id="bot1",
            event_type="trade",
            payload_key="t1",
            exchange_timestamp=now,
            data_source_id="test",
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert "event_id" in d
        assert "bot_id" in d
