"""EventMetadata and deterministic event_id generation."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from libs.instrumentation.lineage import lineage_to_payload, stable_hash


@dataclass
class EventMetadata:
    """Attached to every event emitted by this bot."""
    event_id: str
    bot_id: str
    exchange_timestamp: str
    local_timestamp: str
    clock_skew_ms: int
    data_source_id: str
    bar_id: Optional[str] = None
    event_type: str = ""
    payload_key: str = ""
    schema_version: str = ""
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_event_id(bot_id: str, timestamp: str, event_type: str, payload_key: str) -> str:
    """Return the deterministic idempotency key for an emitted event."""
    raw = f"{bot_id}|{timestamp}|{event_type}|{payload_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_clock_skew(exchange_ts: datetime, local_ts: datetime) -> int:
    """Returns estimated clock skew in milliseconds."""
    delta = exchange_ts - local_ts
    return int(delta.total_seconds() * 1000)


def create_event_metadata(
    bot_id: str,
    event_type: str,
    payload_key: str,
    exchange_timestamp: datetime,
    data_source_id: str,
    bar_id: Optional[str] = None,
    *,
    schema_version: str = "",
    strategy_id: str = "",
    family_id: str = "",
    portfolio_id: str = "",
    trace_id: str = "",
    lineage=None,
) -> EventMetadata:
    """Factory function. Call this for every event you emit."""
    local_now = datetime.now(timezone.utc)
    exchange_ts_str = exchange_timestamp.isoformat()
    local_ts_str = local_now.isoformat()
    event_id = compute_event_id(bot_id, exchange_ts_str, event_type, payload_key)
    resolved_schema = schema_version or "event_metadata_v2"
    lineage_payload = lineage_to_payload(lineage) if lineage is not None else {}
    resolved_strategy_id = strategy_id or str(lineage_payload.get("strategy_id") or "")
    resolved_family_id = family_id or str(lineage_payload.get("family_id") or "")
    resolved_portfolio_id = portfolio_id or str(lineage_payload.get("portfolio_id") or "")
    resolved_trace_id = trace_id or str(lineage_payload.get("trace_id") or "") or stable_hash(
        "trace_",
        {
            "event_id": event_id,
            "event_type": event_type,
            "payload_key": payload_key,
        },
    )

    return EventMetadata(
        event_id=event_id,
        bot_id=bot_id,
        exchange_timestamp=exchange_ts_str,
        local_timestamp=local_ts_str,
        clock_skew_ms=compute_clock_skew(exchange_timestamp, local_now),
        data_source_id=data_source_id,
        bar_id=bar_id,
        event_type=event_type,
        payload_key=payload_key,
        schema_version=resolved_schema,
        strategy_id=resolved_strategy_id,
        family_id=resolved_family_id,
        portfolio_id=resolved_portfolio_id,
        trace_id=resolved_trace_id,
    )
