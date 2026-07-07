from __future__ import annotations

from apps.relay.db.store import EventStore


def _event(event_id: str, priority: int) -> dict:
    return {
        "event_id": event_id,
        "bot_id": "bot-1",
        "event_type": "test",
        "payload": "{}",
        "exchange_timestamp": "2026-05-10T12:00:00+00:00",
        "priority": priority,
    }


def test_relay_priority_first_requires_exact_ack_and_preserves_default_order(tmp_path):
    store = EventStore(str(tmp_path / "relay.db"))
    store.insert_events([
        _event("low-old", 4),
        _event("high-new", 1),
    ])

    default_events = store.get_events()
    assert [event["event_id"] for event in default_events] == ["low-old", "high-new"]

    priority_events = store.get_events(priority_first=True, max_priority=1)
    assert [event["event_id"] for event in priority_events] == ["high-new"]

    assert store.ack_exact(["high-new"]) == 1
    remaining = store.get_events()
    assert [event["event_id"] for event in remaining] == ["low-old"]

    assert store.ack_up_to("low-old") == 1
    assert store.get_events() == []
