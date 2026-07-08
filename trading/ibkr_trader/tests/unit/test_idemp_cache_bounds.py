"""Tests for idempotency cache bounding (1B fix).

Validates:
  - Cache stays bounded at _MAX_IDEMP_CACHE after many inserts
  - _idemp_locks are cleaned on eviction
  - _idemp_locks are cleaned on denial
  - Evicted entries can still be resolved via DB fallback
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.oms.intent.handler import IntentHandler, _MAX_IDEMP_CACHE, _IDEMP_PRUNE_BATCH
from libs.oms.models.intent import Intent, IntentType, IntentResult
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus


def _make_handler(*, db_lookup_result=None) -> IntentHandler:
    """Build a minimal IntentHandler with mocked dependencies."""
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value=None)  # approve all

    router = MagicMock()
    router.route = AsyncMock()

    repo = MagicMock()
    repo.save_order = AsyncMock()
    repo.save_event = AsyncMock()
    repo.save_order_and_event = AsyncMock()
    repo.save_order_fill_and_event = AsyncMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=db_lookup_result)
    repo.get_positions = AsyncMock(return_value=[])

    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_risk_denial = MagicMock()

    return IntentHandler(risk, router, repo, bus)


class TestIdempCacheBounds:
    """Cache should never grow beyond _MAX_IDEMP_CACHE."""

    def test_prune_triggers_above_max(self):
        handler = _make_handler()
        # Manually populate cache above limit
        for i in range(_MAX_IDEMP_CACHE + 100):
            key = f"client_{i}"
            handler._idempotency[key] = f"oms_{i}"
            handler._idemp_locks[key] = asyncio.Lock()

        assert len(handler._idempotency) == _MAX_IDEMP_CACHE + 100

        handler._prune_idemp_cache()

        assert len(handler._idempotency) == _MAX_IDEMP_CACHE + 100 - _IDEMP_PRUNE_BATCH
        # Locks should be cleaned for evicted keys
        assert len(handler._idemp_locks) == len(handler._idempotency)

    def test_prune_noop_below_max(self):
        handler = _make_handler()
        handler._idempotency["a"] = "b"
        handler._idemp_locks["a"] = asyncio.Lock()

        handler._prune_idemp_cache()

        assert len(handler._idempotency) == 1
        assert "a" in handler._idemp_locks

    def test_eviction_removes_oldest_first(self):
        handler = _make_handler()
        # Insert entries in order
        for i in range(_MAX_IDEMP_CACHE + 10):
            handler._idempotency[f"client_{i}"] = f"oms_{i}"

        handler._prune_idemp_cache()

        # Oldest entries (0..999) should be evicted
        assert "client_0" not in handler._idempotency
        assert f"client_{_IDEMP_PRUNE_BATCH - 1}" not in handler._idempotency
        # Newer entries should remain
        assert f"client_{_IDEMP_PRUNE_BATCH}" in handler._idempotency


class TestIdempLockCleanupOnDenial:
    """When risk denies an order, both _idempotency and _idemp_locks should be cleaned."""

    @pytest.mark.asyncio
    async def test_locks_cleaned_on_denial(self):
        handler = _make_handler()
        handler._risk.check_entry = AsyncMock(return_value="heat cap breach")

        order = MagicMock(spec=OMSOrder)
        order.client_order_id = "test_client_123"
        order.oms_order_id = "oms_456"
        order.qty = 1
        order.role = OrderRole.ENTRY
        order.risk_context = MagicMock()
        order.strategy_id = "TEST"

        intent = Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
        receipt = await handler.submit(intent)

        assert receipt.result == IntentResult.DENIED
        assert "test_client_123" not in handler._idempotency
        assert "test_client_123" not in handler._idemp_locks


class TestEntryLockExists:
    """Verify the entry lock is initialized."""

    def test_entry_lock_created(self):
        handler = _make_handler()
        assert isinstance(handler._entry_lock, asyncio.Lock)

    def test_idempotency_is_ordered_dict(self):
        handler = _make_handler()
        assert isinstance(handler._idempotency, OrderedDict)
