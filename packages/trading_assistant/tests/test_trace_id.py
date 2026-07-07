"""Tests for trace_id propagation through event pipeline."""

from trading_assistant.schemas.events import EventMetadata


class TestTraceId:
    def test_event_metadata_has_trace_id(self):
        from datetime import datetime, timezone
        meta = EventMetadata(
            bot_id="bot1",
            exchange_timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
            local_timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
            data_source_id="src1",
            event_type="trade",
            payload_key="t1",
            trace_id="abc-123",
        )
        assert meta.trace_id == "abc-123"

    def test_trace_id_auto_generated_if_missing(self):
        from datetime import datetime, timezone
        meta = EventMetadata(
            bot_id="bot1",
            exchange_timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
            local_timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
            data_source_id="src1",
            event_type="trade",
            payload_key="t1",
        )
        assert meta.trace_id is not None
        assert len(meta.trace_id) > 0
