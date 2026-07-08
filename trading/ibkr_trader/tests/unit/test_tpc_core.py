from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from libs.oms.models.events import OMSEvent, OMSEventType
from strategies.swing._shared.etf_core import SetupSnapshot
from strategies.swing._shared.models import Direction
from strategies.swing.tpc.core import logic
from strategies.swing.tpc.core.serializers import restore_state, snapshot_state
from strategies.swing.tpc.core.state import TPCFill
from strategies.swing.tpc.engine import TPCEngine


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_tpc_live_wrapper_entry_fill_matches_replay_core_state(tmp_path) -> None:
    engine = TPCEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
        state_dir=tmp_path,
    )
    setup = _setup()
    engine._state.setups[setup.setup_id] = setup
    engine._state.pending_orders["ENTRY-1"] = setup

    initial_state = restore_state(snapshot_state(engine._state))
    fill_time = datetime(2026, 5, 20, 14, 45, tzinfo=timezone.utc)
    fill = TPCFill(
        oms_order_id="ENTRY-1",
        fill_price=101.25,
        fill_qty=25,
        symbol="QQQ",
        fill_time=fill_time,
        commission=1.0,
        order_role="entry",
    )

    wrapper_actions, wrapper_events = engine.process_fill(fill)
    replay = run_replay(
        initial_state,
        steps=[ReplayStep(fills=[fill])],
        on_bar=lambda state, bar_input: logic.on_bar(state, bar_input, None),
        on_order_update=logic.on_order_update,
        on_fill=logic.on_fill,
    )

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert normalize_decision_stream(wrapper_events) == normalize_decision_stream(replay.events)
    assert [type(action).__name__ for action in wrapper_actions] == [type(action).__name__ for action in replay.actions]
    assert snapshot_state(replay.state) == snapshot_state(engine._state)


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_tpc_oms_event_loop_handles_ack_then_fill_and_terminal_statuses(tmp_path, monkeypatch) -> None:
    engine = TPCEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
        state_dir=tmp_path,
    )
    engine._event_queue = asyncio.Queue()
    engine._running = True

    fills: list[object] = []
    updates: list[object] = []
    monkeypatch.setattr(engine, "process_fill", lambda fill: fills.append(fill))
    monkeypatch.setattr(engine, "process_order_update", lambda update: updates.append(update))
    monkeypatch.setattr(engine, "_persist_state", lambda: None)

    task = asyncio.create_task(engine._oms_event_loop())
    for event_type in (
        OMSEventType.ORDER_ACKED,
        OMSEventType.ORDER_CANCELLED,
        OMSEventType.ORDER_REJECTED,
        OMSEventType.ORDER_EXPIRED,
        OMSEventType.ORDER_FILLED,
    ):
        await engine._event_queue.put(
            OMSEvent(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                strategy_id="TPC",
                oms_order_id="OMS-1",
                payload={"symbol": "QQQ", "price": 101.0, "qty": 5, "commission": 0.0},
            )
        )

    await _eventually(lambda: len(updates) == 5)
    assert fills == []

    await engine._event_queue.put(
        OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=datetime.now(timezone.utc),
            strategy_id="TPC",
            oms_order_id="OMS-1",
            payload={"symbol": "QQQ", "price": 101.0, "qty": 5, "commission": 0.0},
        )
    )
    await _eventually(lambda: len(fills) == 1)
    engine._running = False
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert [update.status for update in updates] == [
        OMSEventType.ORDER_ACKED.value,
        OMSEventType.ORDER_CANCELLED.value,
        OMSEventType.ORDER_REJECTED.value,
        OMSEventType.ORDER_EXPIRED.value,
        OMSEventType.ORDER_FILLED.value,
    ]
    assert fills[0].fill_price == 101.0
    assert fills[0].fill_qty == 5


def _setup() -> SetupSnapshot:
    return SetupSnapshot(
        setup_id="TPC-QQQ-test",
        strategy_id="TPC",
        symbol="QQQ",
        direction=Direction.LONG,
        grade="valid",
        setup_type="classic_38_62",
        entry_model="market",
        state="entry_ready",
        created_ts=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
        entry_price=101.0,
        stop_price=98.0,
        qty=25,
        score=7.5,
        risk_pct=0.01,
        t1_r=1.5,
        t1_partial_pct=0.4,
        t2_r=2.5,
        t2_partial_pct=0.3,
        entry_order_type="MARKET",
    )


async def _eventually(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()
