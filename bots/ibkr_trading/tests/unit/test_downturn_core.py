from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
import pytest

from libs.oms.models.events import OMSEventType
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitExit
from strategies.momentum.downturn.engine import DownturnEngine
from strategies.momentum.downturn.core.logic import on_bar, on_fill, on_order_update
from strategies.momentum.downturn.core.serializers import restore_state, snapshot_state
from strategies.momentum.downturn.core.state import (
    DownturnCoreState,
    DownturnEntryRequest,
    DownturnFill,
    DownturnOrderUpdate,
    DownturnStopUpdateRequest,
)
from strategies.momentum.downturn.models import ActivePosition, CompositeRegime, EngineTag, VolState, WorkingEntry

UTC = timezone.utc


def _entry_request() -> DownturnEntryRequest:
    return DownturnEntryRequest(
        client_order_id="entry-1",
        symbol="MNQ",
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        qty=2,
        entry_price=18990.0,
        stop0=19010.0,
        order_type="STOP_LIMIT",
        price=18990.0,
        limit_price=18990.0,
        stop_price=18992.0,
        submitted_bar_idx=10,
        ttl_bars=72,
        composite_regime=CompositeRegime.EMERGING_BEAR,
        vol_state=VolState.NORMAL,
    )


def test_downturn_core_entry_lifecycle_and_snapshot_roundtrip() -> None:
    state = DownturnCoreState(symbol="MNQ", bar_count_5m=10)

    state, actions, events = on_bar(
        state,
        bar_count_5m=10,
        bar_ts=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        entry_request=_entry_request(),
    )
    assert isinstance(actions[0], SubmitEntry)
    assert events[-1].code == "ENTRY_REQUESTED"

    state, _, events = on_order_update(
        state,
        DownturnOrderUpdate(
            oms_order_id="OMS-1",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 10, 1, tzinfo=UTC),
            order_role="entry",
            accepted_entry=_entry_request(),
        ),
    )
    assert state.working_entries[0].oms_order_id == "OMS-1"
    assert events[-1].code == "ENTRY_SUBMITTED"

    state, actions, events = on_fill(
        state,
        DownturnFill(
            oms_order_id="OMS-1",
            fill_price=18989.0,
            fill_qty=2,
            commission=1.25,
            fill_time=datetime(2026, 4, 25, 10, 2, tzinfo=UTC),
        ),
    )
    assert state.position is not None
    assert isinstance(actions[0], SubmitExit)
    assert actions[0].stop_price == 19010.0
    assert events[-1].code == "ENTRY_FILLED"

    snapshot = snapshot_state(state)
    restored = restore_state(snapshot)

    assert restored.symbol == "MNQ"
    assert restored.position is not None
    assert restored.position.entry_price == 18989.0
    assert restored.position.commission == 1.25


def test_downturn_core_stop_replace_and_flatten_transitions() -> None:
    state = DownturnCoreState(
        symbol="MNQ",
        position=ActivePosition(
            engine_tag=EngineTag.FADE,
            signal_class="vwap_rejection",
            trade_id="trade-1",
            entry_price=18989.0,
            stop0=19010.0,
            qty=2,
            remaining_qty=2,
            entry_oms_order_id="OMS-1",
            stop_oms_order_id="STOP-1",
        ),
    )

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 10, 3, tzinfo=UTC),
        stop_update=DownturnStopUpdateRequest(stop_price=19005.0, qty=2, reason="trail"),
    )
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].symbol == "MNQ"
    assert events[-1].code == "STOP_REPLACEMENT_REQUESTED"

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 10, 4, tzinfo=UTC),
        flatten_reason="risk_off",
    )
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].symbol == "MNQ"
    assert events[-1].code == "FLATTEN_REQUESTED"


def test_downturn_core_ignores_unmatched_fill_without_exit_context() -> None:
    state = DownturnCoreState(
        symbol="MNQ",
        position=ActivePosition(
            engine_tag=EngineTag.FADE,
            signal_class="vwap_rejection",
            trade_id="trade-1",
            entry_price=18989.0,
            stop0=19010.0,
            qty=2,
            remaining_qty=2,
            entry_oms_order_id="OMS-1",
            stop_oms_order_id="STOP-1",
        ),
    )

    next_state, actions, events = on_fill(
        state,
        DownturnFill(
            oms_order_id="UNRELATED",
            fill_price=18995.0,
            fill_qty=1,
            fill_time=datetime(2026, 4, 25, 10, 6, tzinfo=UTC),
        ),
    )

    assert next_state.position is not None
    assert next_state.position.trade_id == "trade-1"
    assert actions == []
    assert events == []


def test_downturn_replay_driver_produces_normalized_decision_stream() -> None:
    steps = [
        ReplayStep(
            bar_input={
                "bar_count_5m": 10,
                "bar_ts": datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
                "entry_request": _entry_request(),
            }
        ),
        ReplayStep(
            order_updates=[
                DownturnOrderUpdate(
                    oms_order_id="OMS-1",
                    status="accepted",
                    timestamp=datetime(2026, 4, 25, 10, 1, tzinfo=UTC),
                    order_role="entry",
                    accepted_entry=_entry_request(),
                )
            ]
        ),
        ReplayStep(
            fills=[
                DownturnFill(
                    oms_order_id="OMS-1",
                    fill_price=18989.0,
                    fill_qty=2,
                    fill_time=datetime(2026, 4, 25, 10, 2, tzinfo=UTC),
                )
            ]
        ),
    ]

    result = run_replay(
        DownturnCoreState(symbol="MNQ"),
        steps=steps,
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    codes = [event["code"] for event in normalize_decision_stream(result.events)]
    assert codes == ["ENTRY_REQUESTED", "ENTRY_SUBMITTED", "ENTRY_FILLED"]


class _DummyIB:
    pass


class _DummyOMS:
    pass


@pytest.mark.asyncio
async def test_downturn_engine_snapshot_and_hydrate_preserve_wrapper_contract(tmp_path) -> None:
    engine = DownturnEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._position = ActivePosition(
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        trade_id="trade-1",
        entry_price=18989.0,
        stop0=19010.0,
        qty=2,
        remaining_qty=2,
        entry_oms_order_id="OMS-1",
        stop_oms_order_id="STOP-1",
    )
    engine._working_entries = [
        WorkingEntry(
            oms_order_id="OMS-2",
            engine_tag=EngineTag.REVERSAL,
            signal_class="reversal",
            entry_price=18980.0,
            stop0=19020.0,
            qty=1,
            submitted_bar_idx=14,
            ttl_bars=24,
            composite_regime=CompositeRegime.EMERGING_BEAR,
            vol_state=VolState.HIGH,
        )
    ]
    engine._bar_count_5m = 15
    engine._bars_since_last_entry = 2
    engine._last_decision_code = "ENTRY_FILLED"
    engine._last_decision_details = {"qty": 2}
    engine._last_bar_ts = datetime(2026, 4, 25, 10, 5, tzinfo=UTC)

    snapshot = engine.snapshot_state()

    restored = DownturnEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    await restored.hydrate(snapshot)

    assert restored.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert restored._position is not None
    assert restored._position.trade_id == "trade-1"
    assert restored._working_entries[0].oms_order_id == "OMS-2"
    assert restored._bar_count_5m == 15
    assert restored._symbol == "MNQ"


@pytest.mark.asyncio
async def test_downturn_engine_entry_fill_routes_through_shared_core(tmp_path) -> None:
    engine = DownturnEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._working_entries = [
        WorkingEntry(
            oms_order_id="OMS-1",
            engine_tag=EngineTag.FADE,
            signal_class="vwap_rejection",
            entry_price=18990.0,
            stop0=19010.0,
            qty=2,
            submitted_bar_idx=10,
            ttl_bars=72,
            composite_regime=CompositeRegime.EMERGING_BEAR,
            vol_state=VolState.NORMAL,
        )
    ]

    await engine._on_fill(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id="OMS-1",
            payload={"price": 18989.0, "qty": 2, "commission": 1.25},
            timestamp=datetime(2026, 4, 25, 10, 2, tzinfo=UTC),
        )
    )

    assert engine._position is not None
    assert engine._position.entry_oms_order_id == "OMS-1"
    assert engine._position.entry_price == 18989.0
    assert engine._working_entries == []
    assert engine.health_status()["last_decision_code"] == "ENTRY_FILLED"


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_downturn_live_wrapper_entry_fill_matches_replay_core_state(tmp_path, monkeypatch) -> None:
    engine = DownturnEngine(
        ib_session=_DummyIB(),
        oms_service=_DummyOMS(),
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    working_entry = WorkingEntry(
        oms_order_id="OMS-1",
        engine_tag=EngineTag.FADE,
        signal_class="vwap_rejection",
        entry_price=18990.0,
        stop0=19010.0,
        qty=2,
        submitted_bar_idx=10,
        ttl_bars=72,
        composite_regime=CompositeRegime.EMERGING_BEAR,
        vol_state=VolState.NORMAL,
    )
    engine._working_entries = [working_entry]

    async def _fake_place_stop(_stop_price: float, _qty: int) -> None:
        return None

    monkeypatch.setattr(engine, "_place_protective_stop", _fake_place_stop)
    monkeypatch.setattr(engine, "_persist_state", lambda: None)

    initial_state = restore_state(snapshot_state(engine._build_core_state()))
    fill_time = datetime(2026, 4, 25, 10, 2, tzinfo=UTC)

    await engine._on_entry_fill(working_entry, 18989.0, 2, 1.25, fill_time)

    wrapper_snapshot = snapshot_state(engine._build_core_state())
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    DownturnFill(
                        oms_order_id="OMS-1",
                        fill_price=18989.0,
                        fill_qty=2,
                        commission=1.25,
                        fill_time=fill_time,
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_state(replay.state) == wrapper_snapshot
