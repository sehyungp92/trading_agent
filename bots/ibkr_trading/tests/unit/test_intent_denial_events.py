"""Forensic per-denial events from OMSService.submit_intent.

Counters on OMSService (``_intents_denied``, ``_consecutive_denials``) only
tell the watchdog that *many* denials are happening. This callback path
gives operators rule + reason + symbol per denial so a quiet day can be
attributed to a specific gateway block (heat cap, daily stop, session
block, account gate, etc.) without spelunking Postgres.

Mirrors the construction pattern in
``tests/unit/test_liveness_detection.py::test_oms_service_intent_tracking``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.oms.models.intent import IntentReceipt, IntentResult
from libs.oms.services.oms_service import OMSService


def _intent(strategy_id: str = "ATRSS", symbol: str = "QQQ", qty: int = 10) -> MagicMock:
    instrument = SimpleNamespace(symbol=symbol)
    side = SimpleNamespace(value="BUY")
    role = SimpleNamespace(value="ENTRY")
    order = SimpleNamespace(
        strategy_id=strategy_id,
        instrument=instrument,
        side=side,
        role=role,
        qty=qty,
    )
    intent = MagicMock()
    intent.order = order
    return intent


@pytest.mark.asyncio
async def test_denied_intent_fires_callback() -> None:
    handler = AsyncMock()
    bus = MagicMock()
    reconciler = MagicMock()
    sink: list[dict] = []

    svc = OMSService(handler, bus, reconciler, on_intent_denied=sink.append)
    svc._ready.set()

    handler.submit.return_value = IntentReceipt(
        result=IntentResult.DENIED, intent_id="i1", denial_reason="Heat cap breach"
    )
    await svc.submit_intent(_intent(strategy_id="ATRSS", symbol="QQQ", qty=12))

    assert len(sink) == 1
    event = sink[0]
    assert event["strategy_id"] == "ATRSS"
    assert event["symbol"] == "QQQ"
    assert event["qty"] == 12
    assert event["side"] == "BUY"
    assert event["role"] == "ENTRY"
    assert event["denial_reason"] == "Heat cap breach"
    assert event["consecutive_denials"] == 1
    assert event["intent_id"] == "i1"
    assert "ts" in event


@pytest.mark.asyncio
async def test_consecutive_denials_count_climbs_then_resets() -> None:
    handler = AsyncMock()
    bus = MagicMock()
    reconciler = MagicMock()
    sink: list[dict] = []

    svc = OMSService(handler, bus, reconciler, on_intent_denied=sink.append)
    svc._ready.set()

    # 6 sequential denials
    for i in range(6):
        handler.submit.return_value = IntentReceipt(
            result=IntentResult.DENIED, intent_id=f"d{i}", denial_reason="Daily stop"
        )
        await svc.submit_intent(_intent())

    assert len(sink) == 6
    assert [e["consecutive_denials"] for e in sink] == [1, 2, 3, 4, 5, 6]
    assert svc._consecutive_denials == 6
    assert svc._intents_denied == 6

    # Acceptance resets
    handler.submit.return_value = IntentReceipt(
        result=IntentResult.ACCEPTED, intent_id="a1", oms_order_id="o1"
    )
    await svc.submit_intent(_intent())
    assert svc._consecutive_denials == 0
    assert len(sink) == 6  # acceptance does not fire callback


@pytest.mark.asyncio
async def test_no_callback_when_not_provided() -> None:
    """OMSService still works without the callback (backward compatibility)."""
    handler = AsyncMock()
    bus = MagicMock()
    reconciler = MagicMock()

    svc = OMSService(handler, bus, reconciler)
    svc._ready.set()

    handler.submit.return_value = IntentReceipt(
        result=IntentResult.DENIED, intent_id="i1", denial_reason="x"
    )
    receipt = await svc.submit_intent(_intent())
    assert receipt.result == IntentResult.DENIED
    assert svc._consecutive_denials == 1


@pytest.mark.asyncio
async def test_callback_exception_does_not_break_submit() -> None:
    """Bad callback must not crash the submit_intent path — denials are critical."""
    handler = AsyncMock()
    bus = MagicMock()
    reconciler = MagicMock()

    def _broken(_event: dict) -> None:
        raise RuntimeError("boom")

    svc = OMSService(handler, bus, reconciler, on_intent_denied=_broken)
    svc._ready.set()

    handler.submit.return_value = IntentReceipt(
        result=IntentResult.DENIED, intent_id="i1", denial_reason="x"
    )
    receipt = await svc.submit_intent(_intent())
    assert receipt.result == IntentResult.DENIED
    assert svc._consecutive_denials == 1
