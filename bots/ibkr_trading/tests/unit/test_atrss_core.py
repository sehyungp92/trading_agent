from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from backtests.swing.engine.backtest_engine import TradeRecord
import pytest

from strategies.swing.atrss.core.logic import build_core_state as build_atrss_runtime_state
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProtectiveStop
from strategies.swing.atrss.core import logic as atrss_logic
from strategies.swing.atrss.core.serializers import restore_state as restore_atrss_state
from strategies.swing.atrss.core.serializers import snapshot_state as snapshot_atrss_state
from strategies.swing.atrss.core.state import (
    ATRSSCoreState,
    ATRSSEntryRequest,
    ATRSSFill,
    ATRSSFlattenRequest,
    ATRSSOrderUpdate,
    ATRSSPartialExitRequest,
)
from strategies.swing.atrss.engine import ATRSSEngine
from strategies.swing.atrss.models import Candidate, CandidateType, Direction, HourlyState, PositionBook, PositionLeg

UTC = timezone.utc


def test_atrss_on_bar_entry_request_emits_submit_entry() -> None:
    candidate = Candidate(
        symbol="QQQ",
        type=CandidateType.PULLBACK,
        direction=Direction.LONG,
        trigger_price=510.25,
        initial_stop=503.5,
        qty=3,
        signal_bar=HourlyState(time=datetime(2026, 4, 26, 13, 0, tzinfo=UTC)),
    )
    state, actions, events = atrss_logic.on_bar(
        ATRSSCoreState(),
        bar_ts=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        entry_request=ATRSSEntryRequest(
            client_order_id="ENTRY-1",
            symbol="QQQ",
            candidate=candidate,
            limit_price=510.75,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 3
    assert actions[0].stop_price == 510.25
    assert events[0].code == "ENTRY_REQUESTED"
    assert state.last_decision_code == "ENTRY_REQUESTED"


def test_atrss_trade_record_exposes_gross_and_net_pnl_for_canonical_outcomes() -> None:
    signal_time = datetime(2026, 4, 26, 13, 0, tzinfo=UTC)
    fill_time = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)
    trade = TradeRecord(
        symbol="QQQ",
        direction=1,
        qty=10,
        pnl_dollars=125.0,
        commission=3.5,
        signal_time=signal_time,
        fill_time=fill_time,
        entry_time=fill_time,
        exit_time=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
    )

    outcome = trade_outcomes_from_records([trade])[0]

    assert trade.gross_pnl == pytest.approx(125.0)
    assert trade.net_pnl == pytest.approx(121.5)
    assert outcome["gross_pnl"] == pytest.approx(125.0)
    assert outcome["net_pnl"] == pytest.approx(121.5)
    assert outcome["decision_ts"] == signal_time.isoformat()
    assert outcome["fill_ts"] == fill_time.isoformat()


def test_atrss_on_fill_entry_creates_position_and_protective_stop() -> None:
    state = ATRSSCoreState(
        pending_orders={
            "ENTRY-1": {
                "symbol": "QQQ",
                "type": CandidateType.PULLBACK,
                "direction": Direction.LONG,
                "trigger_price": 510.25,
                "initial_stop": 503.5,
                "qty": 3,
            }
        }
    )

    next_state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="ENTRY-1",
            fill_price=510.5,
            fill_qty=3,
            fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
        ),
    )

    position = next_state.positions["QQQ"]
    assert position.direction is Direction.LONG
    assert position.base_leg is not None
    assert position.base_leg.qty == 3
    assert position.stop_pending is True
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].stop_price == 503.5
    assert events[0].code == "ENTRY_FILLED"


def test_atrss_order_update_registers_submitted_entry_before_fill() -> None:
    signal_ts = datetime(2026, 4, 26, 13, 0, tzinfo=UTC)
    fill_ts = datetime(2026, 4, 26, 14, 0, tzinfo=UTC)
    candidate = Candidate(
        symbol="QQQ",
        type=CandidateType.PULLBACK,
        direction=Direction.LONG,
        trigger_price=510.25,
        initial_stop=503.5,
        qty=3,
        signal_bar=HourlyState(time=signal_ts),
    )

    replay = run_replay(
        ATRSSCoreState(),
        steps=[
            ReplayStep(
                bar_input={
                    "bar_ts": signal_ts,
                    "entry_request": ATRSSEntryRequest(
                        client_order_id="ENTRY-1",
                        symbol="QQQ",
                        candidate=candidate,
                        limit_price=510.75,
                    ),
                },
                order_updates=[
                    ATRSSOrderUpdate(
                        oms_order_id="ENTRY-1",
                        status="submitted",
                        symbol="QQQ",
                        timestamp=signal_ts,
                        order_role="entry",
                        metadata={
                            "symbol": "QQQ",
                            "type": CandidateType.PULLBACK,
                            "direction": Direction.LONG,
                            "trigger_price": 510.25,
                            "initial_stop": 503.5,
                            "qty": 3,
                        },
                    )
                ],
                fills=[
                    ATRSSFill(
                        oms_order_id="ENTRY-1",
                        fill_price=510.5,
                        fill_qty=3,
                        fill_time=fill_ts,
                    )
                ],
            )
        ],
        on_bar=lambda state, payload: atrss_logic.on_bar(state, **payload),
        on_order_update=atrss_logic.on_order_update,
        on_fill=atrss_logic.on_fill,
    )

    assert replay.state.positions["QQQ"].base_leg is not None
    assert replay.state.positions["QQQ"].base_leg.qty == 3
    assert [event.code for event in replay.events] == [
        "ENTRY_REQUESTED",
        "ORDER_SUBMITTED",
        "ENTRY_FILLED",
    ]


def test_atrss_order_update_registers_submitted_stop_and_flatten() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.LONG,
                legs=[PositionLeg(qty=3, entry_price=510.5, initial_stop=503.5)],
                current_stop=503.5,
                stop_pending=True,
            )
        }
    )

    state, actions, events = atrss_logic.on_order_update(
        state,
        ATRSSOrderUpdate(
            oms_order_id="STOP-1",
            status="submitted",
            symbol="QQQ",
            timestamp=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
            order_role="stop",
            metadata={"qty": 3, "stop_price": 503.5},
        ),
    )

    assert actions == []
    assert events[0].code == "STOP_SUBMITTED"
    assert state.positions["QQQ"].stop_oms_order_id == "STOP-1"
    assert state.positions["QQQ"].stop_pending is False

    state, actions, events = atrss_logic.on_order_update(
        state,
        ATRSSOrderUpdate(
            oms_order_id="FLAT-1",
            status="submitted",
            symbol="QQQ",
            timestamp=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
            order_role="flatten",
            metadata={"symbol": "QQQ", "reason": "FLATTEN_TIME_DECAY", "qty": 3},
        ),
    )

    assert actions == []
    assert events[0].code == "FLATTEN_ORDER_SUBMITTED"
    assert state.pending_flattens["QQQ"]["oms_order_id"] == "FLAT-1"


def test_atrss_on_fill_partial_close_resizes_stop() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.LONG,
                legs=[
                    PositionLeg(
                        qty=3,
                        entry_price=510.5,
                        initial_stop=503.5,
                        fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
                    )
                ],
                current_stop=507.0,
                stop_oms_order_id="STOP-1",
            )
        },
        pending_orders={
            "PARTIAL-1": {
                "symbol": "QQQ",
                "type": "PARTIAL_CLOSE",
                "direction": Direction.LONG,
                "partial_qty": 1,
                "reason": "TP1",
            }
        },
    )

    next_state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="PARTIAL-1",
            fill_price=514.0,
            fill_qty=1,
            fill_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        ),
    )

    assert next_state.positions["QQQ"].base_leg is not None
    assert next_state.positions["QQQ"].base_leg.qty == 2
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 2
    assert events[0].code == "PARTIAL_EXIT_FILLED"


def test_atrss_on_fill_stop_grants_voucher_and_removes_position() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.LONG,
                legs=[
                    PositionLeg(
                        qty=2,
                        entry_price=510.5,
                        initial_stop=503.5,
                        fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
                    )
                ],
                current_stop=507.0,
                stop_oms_order_id="STOP-1",
                mfe=1.25,
            )
        }
    )

    next_state, actions, events = atrss_logic.on_fill(
        state,
        ATRSSFill(
            oms_order_id="STOP-1",
            fill_price=507.0,
            fill_qty=2,
            fill_time=datetime(2026, 4, 26, 14, 30, tzinfo=UTC),
            exit_type="STOP",
        ),
    )

    assert "QQQ" not in next_state.positions
    assert next_state.reentry_states["QQQ"].voucher_long is True
    assert actions == []
    assert events[0].code == "STOP_FILLED"


def test_atrss_core_quarantines_unmatched_fill_as_decision_event() -> None:
    last_bar_ts = datetime(2026, 4, 26, 13, 0, tzinfo=UTC)

    next_state, actions, events = atrss_logic.on_fill(
        ATRSSCoreState(last_bar_ts=last_bar_ts),
        ATRSSFill(
            oms_order_id="UNKNOWN-1",
            symbol="QQQ",
            fill_price=510.5,
            fill_qty=3,
            fill_time=datetime(2026, 4, 26, 13, 5, tzinfo=UTC),
        ),
    )

    assert actions == []
    assert events[0].code == "UNMATCHED_FILL"
    assert events[0].details["oms_order_id"] == "UNKNOWN-1"
    assert next_state.last_decision_code == "UNMATCHED_FILL"
    assert next_state.last_bar_ts == last_bar_ts


def test_atrss_on_bar_flatten_emits_flatten_action() -> None:
    state = ATRSSCoreState(
        positions={
            "QQQ": PositionBook(
                symbol="QQQ",
                direction=Direction.SHORT,
                legs=[PositionLeg(qty=2, entry_price=510.5, initial_stop=517.5)],
            )
        }
    )

    _state, actions, events = atrss_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
        flatten_request=ATRSSFlattenRequest(symbol="QQQ", reason="FLATTEN_TIME_DECAY"),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].side == "BUY"
    assert events[0].code == "FLATTEN_REQUESTED"


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_atrss_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch) -> None:
    pending_order = {
        "symbol": "QQQ",
        "type": CandidateType.PULLBACK,
        "direction": Direction.LONG,
        "trigger_price": 510.25,
        "initial_stop": 503.5,
        "qty": 3,
    }
    engine = ATRSSEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.pending_orders["ENTRY-1"] = dict(pending_order)

    placed_stops: list[tuple[str, float, int]] = []

    async def _fake_place_stop(symbol: str, stop_price: float, qty: int) -> str:
        placed_stops.append((symbol, stop_price, qty))
        return ""

    monkeypatch.setattr(engine, "_place_stop", _fake_place_stop)

    initial_state = restore_atrss_state(snapshot_atrss_state(build_atrss_runtime_state(engine)))

    await engine._on_fill("ENTRY-1", {"price": 510.5, "qty": 3, "commission": 0.0})

    wrapper_snapshot = snapshot_atrss_state(build_atrss_runtime_state(engine))
    fill_time = engine.positions["QQQ"].base_leg.fill_time
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    ATRSSFill(
                        oms_order_id="ENTRY-1",
                        fill_price=510.5,
                        fill_qty=3,
                        fill_time=fill_time,
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: atrss_logic.on_bar(state, **payload),
        on_order_update=atrss_logic.on_order_update,
        on_fill=atrss_logic.on_fill,
    )
    replay.state.positions["QQQ"].stop_pending = False

    assert placed_stops == [("QQQ", 503.5, 3)]
    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_atrss_state(replay.state) == wrapper_snapshot
