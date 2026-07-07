from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _event(event_id: str, priority: int) -> dict:
    return {
        "event_id": event_id,
        "bot_id": "bot-1",
        "event_type": "trade",
        "payload": json.dumps({"timestamp": "2026-07-06T12:00:00Z"}),
        "exchange_timestamp": "2026-07-06T12:00:00Z",
        "priority": priority,
    }


def _load_receiver_cls():
    sys.modules.setdefault("orchestrator", types.ModuleType("orchestrator"))
    sys.modules.setdefault("orchestrator.db", types.ModuleType("orchestrator.db"))
    queue_mod = types.ModuleType("orchestrator.db.queue")
    queue_mod.EventQueue = object
    sys.modules["orchestrator.db.queue"] = queue_mod

    path = (
        Path(__file__).resolve().parents[4]
        / "packages/trading_assistant/src/"
        "trading_assistant/orchestrator/adapters/vps_receiver.py"
    )
    spec = importlib.util.spec_from_file_location("vps_receiver_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.VPSReceiver


class _Response:
    def __init__(
        self,
        events: list[dict] | None = None,
        *,
        priority_first: bool = False,
    ) -> None:
        self._events = events or []
        self._priority_first = priority_first

    def json(self) -> dict:
        payload = {"events": self._events}
        if self._priority_first:
            payload.update({"delivery_mode": "priority_first", "ack_mode": "exact"})
        return payload

    def raise_for_status(self) -> None:
        pass


class _Client:
    def __init__(self, get_batches: list[list[dict]]) -> None:
        self.get_batches = list(get_batches)
        self.gets: list[dict] = []
        self.posts: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, path: str, params: dict):
        self.gets.append({"path": path, "params": dict(params)})
        return _Response(
            self.get_batches.pop(0),
            priority_first=params.get("priority_first") == "true",
        )

    async def post(self, path: str, json: dict):
        self.posts.append((path, dict(json)))
        return _Response()


class _Queue:
    def __init__(self, watermark: str = "normal-0") -> None:
        self.watermark = watermark
        self.enqueued: list[list[dict]] = []
        self.updated: list[str] = []

    async def get_watermark(self, key: str) -> str:
        return self.watermark

    async def enqueue_batch(self, events: list[dict]):
        self.enqueued.append(events)
        return SimpleNamespace(inserted=len(events), duplicates=0)

    async def update_watermark(self, key: str, event_id: str) -> None:
        self.updated.append(event_id)
        self.watermark = event_id


class _DuplicateQueue(_Queue):
    async def enqueue_batch(self, events: list[dict]):
        self.enqueued.append(events)
        return SimpleNamespace(inserted=0, duplicates=len(events))


@pytest.mark.asyncio
async def test_urgent_events_use_priority_first_and_exact_ack() -> None:
    VPSReceiver = _load_receiver_cls()
    urgent = [_event("urgent-2", 1)]
    normal = [_event("normal-1", 3)]
    client = _Client([urgent, normal])
    queue = _Queue("normal-0")
    receiver = VPSReceiver("https://relay", queue, _client_factory=lambda: client)

    inserted = await receiver.pull_and_store(limit=100)

    assert inserted == 2
    assert client.gets[0]["params"] == {
        "limit": 100,
        "priority_first": "true",
        "max_priority": 1,
    }
    assert "since" not in client.gets[0]["params"]
    assert client.posts[0] == ("/ack-exact", {"event_ids": ["urgent-2"]})
    assert client.gets[1]["params"] == {"limit": 100, "since": "normal-0"}
    assert client.posts[1] == ("/ack", {"watermark": "normal-1"})
    assert queue.updated == ["normal-1"]


@pytest.mark.asyncio
async def test_urgent_backlog_prevents_normal_watermark_drain() -> None:
    VPSReceiver = _load_receiver_cls()
    urgent = [_event("urgent-1", 0)]
    client = _Client([urgent])
    queue = _Queue("normal-0")
    receiver = VPSReceiver("https://relay", queue, _client_factory=lambda: client)

    inserted = await receiver.pull_and_store(limit=1)

    assert inserted == 1
    assert len(client.gets) == 1
    assert client.posts == [("/ack-exact", {"event_ids": ["urgent-1"]})]
    assert queue.updated == []


@pytest.mark.asyncio
async def test_drain_continues_when_full_urgent_page_is_duplicate() -> None:
    VPSReceiver = _load_receiver_cls()
    urgent = [_event("urgent-dup", 1)]
    client = _Client([urgent, [], []])
    queue = _DuplicateQueue("normal-0")
    receiver = VPSReceiver("https://relay", queue, _client_factory=lambda: client)

    inserted = await receiver.drain(batch_size=1, max_batches=2)

    assert inserted == 0
    assert client.posts == [("/ack-exact", {"event_ids": ["urgent-dup"]})]
    assert len(client.gets) == 3


@pytest.mark.asyncio
async def test_priority_can_be_disabled_for_legacy_watermark_only() -> None:
    VPSReceiver = _load_receiver_cls()
    normal = [_event("normal-1", 3)]
    client = _Client([normal])
    queue = _Queue("normal-0")
    receiver = VPSReceiver(
        "https://relay",
        queue,
        priority_first=False,
        _client_factory=lambda: client,
    )

    inserted = await receiver.pull_and_store(limit=50)

    assert inserted == 1
    assert client.gets == [{"path": "/events", "params": {"limit": 50, "since": "normal-0"}}]
    assert client.posts == [("/ack", {"watermark": "normal-1"})]
