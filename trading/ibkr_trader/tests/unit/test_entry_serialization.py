"""Tests for concurrent entry serialization (1C fix).

Validates that two concurrent entries sharing one OMS/IntentHandler
are serialized so only one passes when aggregate risk would exceed the cap.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.oms.intent.handler import IntentHandler
from libs.oms.models.intent import Intent, IntentType, IntentResult
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus


def _make_order(strategy_id: str, client_order_id: str) -> OMSOrder:
    order = MagicMock(spec=OMSOrder)
    order.strategy_id = strategy_id
    order.client_order_id = client_order_id
    order.oms_order_id = f"oms_{client_order_id}"
    order.qty = 1
    order.role = OrderRole.ENTRY
    order.risk_context = MagicMock()
    order.risk_context.risk_dollars = 100.0
    order.risk_context.unit_risk_dollars = 100.0
    order.side = OrderSide.BUY
    order.instrument = MagicMock()
    order.instrument.symbol = "NQ"
    return order


@pytest.mark.asyncio
async def test_concurrent_entries_serialized():
    """Two concurrent 1.0R entries with 1.5R cap: only first should pass.

    The mock uses a read-yield-write pattern to simulate a real DB risk check.
    Without the entry lock, both coroutines read open_R=0.0 before either
    writes, so both approve (race condition).  With the lock, the second
    coroutine reads the updated value and correctly denies.
    """
    call_count = 0
    shared_risk = {"open_R": 0.0}

    async def _risk_check(order, **_kwargs):
        nonlocal call_count
        call_count += 1
        # 1. Read current risk (simulate DB query)
        current_open_R = shared_risk["open_R"]
        # 2. Yield — without the entry lock, the other coroutine reads here too
        await asyncio.sleep(0.01)
        # 3. Check cap using the (possibly stale) read
        if current_open_R + 1.0 > 1.5:
            return f"Heat cap breach: {current_open_R:.1f}R + 1.0R > 1.5R cap"
        # 4. Write — simulate order persist updating risk state
        shared_risk["open_R"] += 1.0
        return None  # approved

    risk = MagicMock()
    risk.check_entry = AsyncMock(side_effect=_risk_check)
    risk.check_account_gate = AsyncMock(return_value=None)

    router = MagicMock()
    router.route = AsyncMock()

    repo = MagicMock()
    repo.save_order = AsyncMock()
    repo.save_event = AsyncMock()
    repo.save_order_and_event = AsyncMock()
    repo.save_order_fill_and_event = AsyncMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=None)
    repo.get_positions = AsyncMock(return_value=[])

    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_risk_denial = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)

    order1 = _make_order("ATRSS", "entry_1")
    order2 = _make_order("AKC_HELIX", "entry_2")

    intent1 = Intent(intent_type=IntentType.NEW_ORDER, strategy_id="ATRSS", order=order1)
    intent2 = Intent(intent_type=IntentType.NEW_ORDER, strategy_id="AKC_HELIX", order=order2)

    # Submit both concurrently
    results = await asyncio.gather(
        handler.submit(intent1),
        handler.submit(intent2),
    )

    accepted = [r for r in results if r.result == IntentResult.ACCEPTED]
    denied = [r for r in results if r.result == IntentResult.DENIED]

    # The entry lock ensures serialization: first passes, second denied
    assert len(accepted) == 1, f"Expected 1 accepted, got {len(accepted)}"
    assert len(denied) == 1, f"Expected 1 denied, got {len(denied)}"
    assert call_count == 2


@pytest.mark.asyncio
async def test_entry_lock_does_not_block_exits():
    """Exit/cancel intents should not be blocked by the entry lock."""
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value=None)

    router = MagicMock()
    router.route = AsyncMock()
    router.cancel = AsyncMock()

    repo = MagicMock()
    repo.save_order = AsyncMock()
    repo.save_event = AsyncMock()
    repo.save_order_and_event = AsyncMock()
    repo.save_order_fill_and_event = AsyncMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=None)
    repo.get_positions = AsyncMock(return_value=[])
    repo.get_order = AsyncMock(return_value=MagicMock(
        status=OrderStatus.WORKING,
        oms_order_id="oms_cancel_1",
    ))

    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_risk_denial = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)

    # Cancel intent doesn't go through _handle_new_order, so no lock
    cancel_intent = Intent(
        intent_type=IntentType.CANCEL_ORDER,
        strategy_id="ATRSS",
        target_oms_order_id="oms_cancel_1",
    )
    result = await handler.submit(cancel_intent)
    assert result.result == IntentResult.ACCEPTED
