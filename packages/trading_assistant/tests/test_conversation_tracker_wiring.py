"""Tests for ConversationTracker wiring into Worker (A1)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from trading_assistant.orchestrator.app import create_app
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.conversation_tracker import ConversationTracker
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
from trading_assistant.orchestrator.worker import Worker


@pytest.fixture
def app_with_tmp(tmp_path):
    config = AppConfig(
        data_dir=str(tmp_path),
        bot_ids=["bot1"],
        allow_unauthenticated_local=True,
    )
    return create_app(db_dir=str(tmp_path), config=config)


def test_tracker_instantiated_in_create_app(app_with_tmp):
    assert hasattr(app_with_tmp.state, "conversation_tracker")
    assert isinstance(app_with_tmp.state.conversation_tracker, ConversationTracker)


def test_worker_accepts_conversation_tracker():
    queue = AsyncMock()
    registry = AsyncMock()
    brain = OrchestratorBrain()
    tracker = ConversationTracker()

    worker = Worker(
        queue=queue, registry=registry, brain=brain,
        conversation_tracker=tracker,
    )
    assert worker._conversation_tracker is tracker


@pytest.mark.asyncio
async def test_new_events_begin_a_chain():
    queue = AsyncMock()
    queue.claim = AsyncMock(return_value=[
        {"event_id": "ev1", "event_type": "trade", "bot_id": "b1", "payload": "{}"},
    ])
    queue.ack = AsyncMock()
    registry = AsyncMock()
    brain = OrchestratorBrain()
    tracker = ConversationTracker()

    worker = Worker(
        queue=queue, registry=registry, brain=brain,
        conversation_tracker=tracker,
    )
    await worker.process_batch(limit=1)

    chains = tracker.get_active_chains()
    assert len(chains) == 1
    assert "ev1" in chains[0].event_ids


@pytest.mark.asyncio
async def test_chain_id_populated_on_action():
    queue = AsyncMock()
    queue.claim = AsyncMock(return_value=[
        {"event_id": "ev1", "event_type": "trade", "bot_id": "b1", "payload": "{}"},
    ])
    queue.ack = AsyncMock()
    registry = AsyncMock()
    brain = OrchestratorBrain()
    tracker = ConversationTracker()

    dispatched_actions = []

    worker = Worker(
        queue=queue, registry=registry, brain=brain,
        conversation_tracker=tracker,
    )


    async def capture_dispatch(action):
        dispatched_actions.append(action)

    worker._dispatch = capture_dispatch
    await worker.process_batch(limit=1)

    assert len(dispatched_actions) == 1
    assert dispatched_actions[0].chain_id.startswith("chain-")


@pytest.mark.asyncio
async def test_spawned_events_extend_the_chain():
    tracker = ConversationTracker()
    chain = tracker.begin_chain("ev1")

    ok = tracker.extend_chain(chain.chain_id, "ev2")
    assert ok is True
    assert chain.depth == 1
    assert "ev2" in chain.event_ids


@pytest.mark.asyncio
async def test_loop_detected_when_depth_exceeds_max():
    AsyncMock()
    tracker = ConversationTracker(max_depth=2)
    chain = tracker.begin_chain("ev0")

    # Extend to max depth
    tracker.extend_chain(chain.chain_id, "ev1")
    tracker.extend_chain(chain.chain_id, "ev2")

    # This should fail (depth = 3 > max_depth = 2)
    ok = tracker.extend_chain(chain.chain_id, "ev3")
    assert ok is False


@pytest.mark.asyncio
async def test_loop_detected_skips_processing():
    tracker = ConversationTracker(max_depth=0)
    chain = tracker.begin_chain("ev0")
    chain_id = chain.chain_id

    queue = AsyncMock()
    queue.claim = AsyncMock(return_value=[
        {"event_id": "ev1", "event_type": "trade", "bot_id": "b1",
         "payload": "{}", "chain_id": chain_id},
    ])
    queue.ack = AsyncMock()
    registry = AsyncMock()
    brain = OrchestratorBrain()

    worker = Worker(
        queue=queue, registry=registry, brain=brain,
        conversation_tracker=tracker,
    )

    dispatched = []
    worker._dispatch = lambda action: dispatched.append(action)

    result = await worker.process_batch(limit=1)
    assert result == 1  # event was ack'd
    assert len(dispatched) == 0  # but no dispatch happened


@pytest.mark.asyncio
async def test_expired_chains_cleaned_up():
    from datetime import timedelta

    tracker = ConversationTracker(timeout_minutes=0)
    chain = tracker.begin_chain("ev0")

    # Force chain to be expired
    from datetime import datetime, timezone
    chain.started_at = datetime.now(timezone.utc) - timedelta(minutes=1)

    removed = tracker.cleanup_expired()
    assert removed == 1
    assert len(tracker.get_active_chains()) == 0
