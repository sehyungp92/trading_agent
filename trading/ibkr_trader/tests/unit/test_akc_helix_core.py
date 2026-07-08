from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
import pytest

from strategies.swing.akc_helix.core.logic import build_core_state as build_akc_helix_runtime_state
from strategies.core.actions import (
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitProtectiveStop,
)
from strategies.swing.akc_helix.core import logic as akc_helix_logic
from strategies.swing.akc_helix.core.serializers import restore_state as restore_akc_helix_state
from strategies.swing.akc_helix.core.serializers import snapshot_state as snapshot_akc_helix_state
from strategies.swing.akc_helix.core.state import (
    AKCHelixCoreState,
    AKCHelixEntryRequest,
    AKCHelixFill,
)
from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.akc_helix.models import Direction, SetupClass, SetupInstance, SetupState

UTC = timezone.utc


def _setup(*, setup_id: str = "HELIX-1") -> SetupInstance:
    return SetupInstance(
        setup_id=setup_id,
        symbol="QQQ",
        setup_class=SetupClass.CLASS_A,
        direction=Direction.LONG,
        origin_tf="4H",
        state=SetupState.NEW,
        created_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        bos_level=505.5,
        stop0=499.0,
        current_stop=499.0,
        qty_planned=3,
        oca_group="HELIX-OCA",
    )


def test_akc_helix_on_bar_entry_request_emits_submit_entry() -> None:
    state, actions, events = akc_helix_logic.on_bar(
        AKCHelixCoreState(),
        bar_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        entry_request=AKCHelixEntryRequest(
            client_order_id="ENTRY-1",
            setup=_setup(),
            order_type="STOP_LIMIT",
            limit_price=505.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 3
    assert actions[0].stop_price == 505.5
    assert actions[0].limit_price == 505.75
    assert events[0].code == "ENTRY_REQUESTED"
    assert state.pending_setups["HELIX-1"].state is SetupState.ARMED
    assert state.pending_setups["HELIX-1"].primary_order_id == "ENTRY-1"
    assert state.order_to_setup["ENTRY-1"] == "HELIX-1"


def test_akc_helix_on_bar_add_request_keeps_active_setup_active() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.fill_price = 505.75
    setup.avg_entry_price = 505.75
    setup.fill_qty = 3
    setup.qty_open = 3

    state, actions, events = akc_helix_logic.on_bar(
        AKCHelixCoreState(active_setups={setup.setup_id: setup}),
        bar_ts=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        entry_request=AKCHelixEntryRequest(
            client_order_id="ADD-1",
            setup=setup,
            order_type="MARKET",
            order_role="add",
            qty=1,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitAddOnEntry)
    assert actions[0].qty == 1
    assert events[0].code == "ADD_REQUESTED"
    assert "HELIX-1" in state.active_setups
    assert "HELIX-1" not in state.pending_setups
    assert state.active_setups["HELIX-1"].state is SetupState.ACTIVE
    assert state.active_setups["HELIX-1"].primary_order_id == ""
    assert state.active_setups["HELIX-1"].add_done is True
    assert state.order_to_setup["ADD-1"] == "HELIX-1"


def test_akc_helix_on_fill_entry_creates_stop_action() -> None:
    setup = _setup()
    state = AKCHelixCoreState(
        pending_setups={setup.setup_id: setup},
        order_to_setup={"ENTRY-1": setup.setup_id},
    )

    next_state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="ENTRY-1",
            fill_price=505.75,
            fill_qty=3,
            fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
            order_role="entry",
        ),
    )

    filled_setup = next_state.active_setups[setup.setup_id]
    assert filled_setup.state is SetupState.ACTIVE
    assert filled_setup.qty_open == 3
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].stop_price == 499.0
    assert events[0].code == "ENTRY_FILLED"


def test_akc_helix_partial_fill_resizes_stop() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.fill_price = 505.75
    setup.fill_qty = 3
    setup.qty_open = 3
    setup.stop_order_id = "STOP-1"

    state = AKCHelixCoreState(
        active_setups={setup.setup_id: setup},
        order_to_setup={"PARTIAL-1": setup.setup_id},
    )

    next_state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="PARTIAL-1",
            fill_price=508.5,
            fill_qty=1,
            fill_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
            order_role="partial",
        ),
    )

    assert next_state.active_setups[setup.setup_id].qty_open == 2
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 2
    assert events[0].code == "PARTIAL_EXIT_FILLED"


def test_akc_helix_add_fill_preserves_initial_entry_and_updates_average_basis() -> None:
    setup = _setup()
    setup.state = SetupState.ACTIVE
    setup.fill_price = 100.0
    setup.avg_entry_price = 100.0
    setup.fill_qty = 10
    setup.qty_open = 10
    setup.stop_order_id = "STOP-1"

    state = AKCHelixCoreState(
        active_setups={setup.setup_id: setup},
        order_to_setup={"ADD-1": setup.setup_id},
    )

    next_state, actions, events = akc_helix_logic.on_fill(
        state,
        AKCHelixFill(
            oms_order_id="ADD-1",
            fill_price=110.0,
            fill_qty=10,
            fill_time=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
            order_role="add",
        ),
    )

    filled_setup = next_state.active_setups[setup.setup_id]
    assert filled_setup.fill_price == pytest.approx(100.0)
    assert filled_setup.avg_entry_price == pytest.approx(105.0)
    assert filled_setup.qty_open == 20
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 20
    assert events[0].code == "ADD_FILLED"


def test_akc_helix_snapshot_hydrate_continue_matches_uninterrupted_core() -> None:
    setup = _setup()
    entry_step = ReplayStep(
        bar_input={
            "bar_ts": datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
            "entry_request": AKCHelixEntryRequest(
                client_order_id="ENTRY-1",
                setup=setup,
                order_type="STOP_LIMIT",
                limit_price=505.75,
            ),
        }
    )
    fill_step = ReplayStep(
        fills=[
            AKCHelixFill(
                oms_order_id="ENTRY-1",
                fill_price=505.75,
                fill_qty=3,
                fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
                order_role="entry",
            )
        ]
    )
    replay_kwargs = {
        "on_bar": lambda state, payload: akc_helix_logic.on_bar(state, **payload),
        "on_order_update": akc_helix_logic.on_order_update,
        "on_fill": akc_helix_logic.on_fill,
    }

    first_leg = run_replay(AKCHelixCoreState(), steps=[entry_step], **replay_kwargs)
    restored = restore_akc_helix_state(snapshot_akc_helix_state(first_leg.state))
    resumed = run_replay(restored, steps=[fill_step], **replay_kwargs)
    uninterrupted = run_replay(AKCHelixCoreState(), steps=[entry_step, fill_step], **replay_kwargs)

    assert snapshot_akc_helix_state(resumed.state) == snapshot_akc_helix_state(uninterrupted.state)
    assert [event.code for event in resumed.events] == ["ENTRY_FILLED"]
    assert [event.code for event in first_leg.events + resumed.events] == [
        event.code for event in uninterrupted.events
    ]


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_akc_helix_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch) -> None:
    setup = _setup()
    receipt_log: list[object] = []

    async def _fake_submit_intent(intent) -> SimpleNamespace:
        receipt_log.append(intent)
        return SimpleNamespace(oms_order_id=None)

    engine = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(
            stream_events=lambda *_args, **_kwargs: None,
            submit_intent=_fake_submit_intent,
        ),
        instruments={"QQQ": SimpleNamespace(symbol="QQQ")},
        config={},
    )
    engine.pending_setups[setup.setup_id] = setup
    engine._order_to_setup["ENTRY-1"] = setup.setup_id

    monkeypatch.setattr(engine, "_cancel_setup_timers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_record_akc_entry_instrumentation", lambda *_args, **_kwargs: None)

    initial_state = restore_akc_helix_state(snapshot_akc_helix_state(build_akc_helix_runtime_state(engine)))

    event = SimpleNamespace(
        payload={"price": 505.75, "qty": 3, "commission": 0.0},
        timestamp=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
    )
    await engine._on_fill_core_routed("ENTRY-1", event)

    wrapper_snapshot = snapshot_akc_helix_state(build_akc_helix_runtime_state(engine))
    fill_time = engine.active_setups[setup.setup_id].fill_ts
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    AKCHelixFill(
                        oms_order_id="ENTRY-1",
                        fill_price=505.75,
                        fill_qty=3,
                        fill_time=fill_time,
                        order_role="entry",
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: akc_helix_logic.on_bar(state, **payload),
        on_order_update=akc_helix_logic.on_order_update,
        on_fill=akc_helix_logic.on_fill,
    )

    assert len(receipt_log) == 1
    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_akc_helix_state(replay.state) == wrapper_snapshot
