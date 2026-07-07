from __future__ import annotations

from datetime import date, datetime, time, timezone
from types import SimpleNamespace

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
import pytest

from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitPartialExit
from strategies.stock.alcb.core.logic import build_core_state as build_alcb_runtime_state
from strategies.stock.alcb.core.logic import apply_carry_roll
from strategies.stock.alcb.core.logic import on_bar, on_fill, on_order_update
from strategies.stock.alcb.core.serializers import restore_state, snapshot_state
from strategies.stock.alcb.core.state import (
    ALCBCoreState,
    ALCBEntryFillContext,
    ALCBEntryRequest,
    ALCBFill,
    ALCBFlattenRequest,
    ALCBOrderUpdate,
    ALCBPartialExitRequest,
    ALCBStopUpdateRequest,
)
from strategies.stock.alcb.engine import ALCBT2Engine
from strategies.stock.alcb.models import (
    CandidateArtifact,
    CandidateItem,
    Campaign,
    CampaignState,
    Direction,
    EntryType,
    PositionPlan,
    RegimeSnapshot,
    T2PositionState,
)

UTC = timezone.utc


def _artifact(trade_date: date) -> CandidateArtifact:
    item = CandidateItem(
        symbol="AAA",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        adv20_usd=25_000_000.0,
        median_spread_pct=0.001,
        selection_score=87,
        selection_detail={"compression": 40, "momentum": 47},
        stock_regime="BULL",
        market_regime="BULL",
        sector_regime="BULL",
        daily_trend_sign=1,
        relative_strength_percentile=92.0,
        accumulation_score=1.4,
        ttm_squeeze_bonus=2,
        average_30m_volume=120_000.0,
        median_30m_volume=110_000.0,
        tradable_flag=True,
        direction_bias="LONG",
        price=12.45,
        earnings_risk_flag=False,
        campaign=Campaign(symbol="AAA", state=CampaignState.COMPRESSION, campaign_id=7),
    )
    return CandidateArtifact(
        trade_date=trade_date,
        generated_at=datetime.combine(trade_date, time(0, 0), tzinfo=UTC),
        regime=RegimeSnapshot(
            score=0.9,
            tier="A",
            risk_multiplier=1.0,
            price_ok=True,
            breadth_ok=True,
            vol_ok=True,
            credit_ok=True,
            market_regime="BULL",
        ),
        items=[item],
        tradable=[item],
        overflow=[],
        long_candidates=[item],
        short_candidates=[],
    )


def _position() -> T2PositionState:
    return T2PositionState(
        symbol="AAA",
        direction=Direction.LONG,
        entry_price=25.0,
        stop_price=23.5,
        current_stop=24.0,
        quantity=100,
        qty_original=100,
        risk_per_share=1.5,
        entry_time=datetime(2026, 4, 25, 14, 35, tzinfo=UTC),
        entry_type=EntryType.OR_BREAKOUT.value,
        sector="Technology",
        regime_tier="A",
        momentum_score=7,
        avwap_at_entry=24.8,
        or_high=24.9,
        or_low=24.1,
        max_favorable=25.0,
        max_adverse=25.0,
        carry_days=1,
        trade_id="T2-AAA-1",
        setup_tag="T2_OR_BREAKOUT",
    )


def _plan() -> PositionPlan:
    return PositionPlan(
        symbol="AAA",
        direction=Direction.LONG,
        entry_type=EntryType.OR_BREAKOUT,
        entry_price=25.0,
        stop_price=23.5,
        tp1_price=26.5,
        tp2_price=28.0,
        quantity=100,
        risk_per_share=1.5,
        risk_dollars=150.0,
        quality_mult=1.0,
        regime_mult=1.0,
        corr_mult=1.0,
    )


def _entry_request() -> ALCBEntryRequest:
    return ALCBEntryRequest(
        client_order_id="entry-1",
        symbol="AAA",
        plan=_plan(),
        meta={
            "entry_type": EntryType.OR_BREAKOUT.value,
            "sector": "Technology",
            "regime_tier": "A",
            "momentum_score": 7,
            "avwap": 24.8,
            "or_high": 24.9,
            "or_low": 24.1,
        },
    )


def test_alcb_core_snapshot_roundtrip_preserves_campaign_state_machine_payload() -> None:
    state = ALCBCoreState(
        positions={"AAA": _position()},
        or_data={"AAA": (24.5, 24.0, 100_000.0)},
        or_built={"AAA": True},
        order_index={"OMS-1": ("AAA", "ENTRY")},
        pending_entries={"AAA": "OMS-1"},
        pending_plans={"OMS-1": _plan()},
        entry_meta={"OMS-1": {"entry_type": EntryType.OR_BREAKOUT.value}},
        exit_reasons={"OMS-2": "EOD_FLATTEN"},
        last_decision_code="ENTRY_SUBMITTED",
        last_decision_details={"symbol": "AAA"},
        last_bar_ts=datetime(2026, 4, 25, 15, 0, tzinfo=UTC),
    )

    restored = restore_state(snapshot_state(state))

    assert restored.positions["AAA"].trade_id == "T2-AAA-1"
    assert restored.pending_plans["OMS-1"].entry_type == EntryType.OR_BREAKOUT
    assert restored.or_data["AAA"] == (24.5, 24.0, 100_000.0)
    assert restored.last_decision_code == "ENTRY_SUBMITTED"


def test_alcb_core_entry_partial_and_exit_lifecycle() -> None:
    state = ALCBCoreState()

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 14, 30, tzinfo=UTC),
        entry_request=_entry_request(),
    )
    assert actions[0].symbol == "AAA"
    assert events[-1].code == "ENTRY_REQUESTED"

    state, _, events = on_order_update(
        state,
        ALCBOrderUpdate(
            oms_order_id="OMS-ENTRY",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 31, tzinfo=UTC),
            accepted_entry=_entry_request(),
        ),
    )
    assert state.pending_entries["AAA"] == "OMS-ENTRY"
    assert state.pending_plans["OMS-ENTRY"].quantity == 100
    assert events[-1].code == "ENTRY_SUBMITTED"

    state, actions, events = on_fill(
        state,
        ALCBFill(
            oms_order_id="OMS-ENTRY",
            fill_price=25.1,
            fill_qty=100,
            fill_time=datetime(2026, 4, 25, 14, 32, tzinfo=UTC),
            commission=1.0,
            entry_context=ALCBEntryFillContext(trade_id="T2-AAA-1"),
        ),
    )
    assert state.positions["AAA"].trade_id == "T2-AAA-1"
    assert state.positions["AAA"].entry_commission == pytest.approx(1.0)
    assert actions[0].order_type == "STOP"
    assert actions[0].stop_price == pytest.approx(_plan().stop_price)
    assert events[-1].code == "ENTRY_FILLED"

    state, _, events = on_order_update(
        state,
        ALCBOrderUpdate(
            oms_order_id="OMS-STOP",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 33, tzinfo=UTC),
            symbol="AAA",
            order_role="stop",
        ),
    )
    assert state.positions["AAA"].stop_order_id == "OMS-STOP"
    assert events[-1].code == "PROTECTIVE_STOP_SUBMITTED"

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 14, 34, tzinfo=UTC),
        stop_update=ALCBStopUpdateRequest(symbol="AAA", stop_price=24.4, qty=100, reason="trail"),
        partial_exit_request=ALCBPartialExitRequest(client_order_id="partial-1", symbol="AAA", qty=40),
    )
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert isinstance(actions[1], SubmitPartialExit)
    assert [event.code for event in events] == ["STOP_REPLACEMENT_REQUESTED", "PARTIAL_EXIT_REQUESTED"]

    state, _, _ = on_order_update(
        state,
        ALCBOrderUpdate(
            oms_order_id="OMS-PARTIAL",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 35, tzinfo=UTC),
            symbol="AAA",
            order_role="partial",
            reason="PARTIAL",
        ),
    )
    state, actions, events = on_fill(
        state,
        ALCBFill(
            oms_order_id="OMS-PARTIAL",
            fill_price=26.0,
            fill_qty=40,
            fill_time=datetime(2026, 4, 25, 14, 36, tzinfo=UTC),
            commission=0.4,
        ),
    )
    assert state.positions["AAA"].quantity == 60
    assert state.positions["AAA"].partial_taken is True
    assert state.positions["AAA"].partial_qty_exited == 40
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 60
    assert events[-1].code == "PARTIAL_EXIT_FILLED"

    state, actions, events = on_bar(
        state,
        bar_ts=datetime(2026, 4, 25, 14, 37, tzinfo=UTC),
        flatten_request=ALCBFlattenRequest(symbol="AAA", reason="EOD_FLATTEN"),
    )
    assert isinstance(actions[0], FlattenPosition)
    assert events[-1].code == "FLATTEN_REQUESTED"

    state, _, _ = on_order_update(
        state,
        ALCBOrderUpdate(
            oms_order_id="OMS-EXIT",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 38, tzinfo=UTC),
            symbol="AAA",
            order_role="exit",
            reason="EOD_FLATTEN",
        ),
    )
    state, actions, events = on_fill(
        state,
        ALCBFill(
            oms_order_id="OMS-EXIT",
            fill_price=25.5,
            fill_qty=60,
            fill_time=datetime(2026, 4, 25, 14, 39, tzinfo=UTC),
            commission=0.6,
            exit_type="EOD_FLATTEN",
        ),
    )
    assert "AAA" not in state.positions
    assert actions == []
    assert events[-1].code == "EXIT_FILLED"


def test_alcb_core_terminal_and_unmatched_fill_are_safe_noops() -> None:
    state = ALCBCoreState()
    state, _, _ = on_order_update(
        state,
        ALCBOrderUpdate(
            oms_order_id="OMS-ENTRY",
            status="accepted",
            timestamp=datetime(2026, 4, 25, 14, 31, tzinfo=UTC),
            accepted_entry=_entry_request(),
        ),
    )
    state, _, events = on_order_update(
        state,
        ALCBOrderUpdate(
            oms_order_id="OMS-ENTRY",
            status="cancelled",
            timestamp=datetime(2026, 4, 25, 14, 32, tzinfo=UTC),
        ),
    )
    assert state.pending_entries == {}
    assert "OMS-ENTRY" not in state.pending_plans
    assert events[-1].code == "ORDER_TERMINATED"

    next_state, actions, events = on_fill(
        state,
        ALCBFill(
            oms_order_id="UNRELATED",
            fill_price=25.0,
            fill_qty=10,
            fill_time=datetime(2026, 4, 25, 14, 33, tzinfo=UTC),
        ),
    )
    assert next_state.positions == {}
    assert actions == []
    assert events == []


def test_alcb_replay_driver_produces_normalized_decision_stream() -> None:
    steps = [
        ReplayStep(
            bar_input={
                "bar_ts": datetime(2026, 4, 25, 14, 30, tzinfo=UTC),
                "entry_request": _entry_request(),
            }
        ),
        ReplayStep(
            order_updates=[
                ALCBOrderUpdate(
                    oms_order_id="OMS-ENTRY",
                    status="accepted",
                    timestamp=datetime(2026, 4, 25, 14, 31, tzinfo=UTC),
                    accepted_entry=_entry_request(),
                )
            ]
        ),
        ReplayStep(
            fills=[
                ALCBFill(
                    oms_order_id="OMS-ENTRY",
                    fill_price=25.1,
                    fill_qty=100,
                    fill_time=datetime(2026, 4, 25, 14, 32, tzinfo=UTC),
                    entry_context=ALCBEntryFillContext(trade_id="T2-AAA-1"),
                )
            ]
        ),
    ]

    result = run_replay(
        ALCBCoreState(),
        steps=steps,
        on_bar=lambda state, payload: on_bar(state, **payload),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    codes = [event["code"] for event in normalize_decision_stream(result.events)]
    assert codes == ["ENTRY_REQUESTED", "ENTRY_SUBMITTED", "ENTRY_FILLED"]


def test_alcb_carry_roll_increments_only_held_symbols() -> None:
    state = ALCBCoreState(
        positions={
            "AAA": _position(),
            "BBB": T2PositionState(
                symbol="BBB",
                direction=Direction.LONG,
                entry_price=10.0,
                stop_price=9.5,
                current_stop=9.7,
                quantity=50,
                qty_original=50,
                risk_per_share=0.5,
                entry_time=datetime(2026, 4, 25, 14, 40, tzinfo=UTC),
                entry_type=EntryType.OR_BREAKOUT.value,
                sector="Technology",
                regime_tier="A",
                momentum_score=5,
                avwap_at_entry=9.8,
                or_high=9.9,
                or_low=9.6,
                max_favorable=10.0,
                max_adverse=10.0,
            ),
        }
    )

    next_state = apply_carry_roll(state, ["AAA"])

    assert next_state.positions["AAA"].carry_days == 2
    assert next_state.positions["BBB"].carry_days == 0
    assert state.positions["AAA"].carry_days == 1


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_alcb_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch) -> None:
    artifact = _artifact(date(2026, 4, 25))
    engine = ALCBT2Engine(
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        artifact=artifact,
        account_id="ACCT-1",
        nav=100_000.0,
    )
    engine._order_index["OMS-ENTRY"] = ("AAA", "ENTRY")
    engine._pending_entries["AAA"] = "OMS-ENTRY"
    engine._pending_plans["OMS-ENTRY"] = _plan()
    engine._entry_meta["OMS-ENTRY"] = {
        "entry_type": EntryType.OR_BREAKOUT.value,
        "sector": "Technology",
        "regime_tier": "A",
        "momentum_score": 7,
        "avwap": 24.8,
        "or_high": 24.9,
        "or_low": 24.1,
    }

    async def _fake_submit_stop(_symbol: str) -> None:
        return None

    async def _fake_entry_fill_instrumentation(*_args, **_kwargs) -> None:
        return None

    async def _fake_exit_fill_instrumentation(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(engine, "_submit_stop", _fake_submit_stop)
    monkeypatch.setattr(engine, "_replace_stop", lambda _symbol: None)
    monkeypatch.setattr(engine, "_handle_entry_fill_instrumentation", _fake_entry_fill_instrumentation)
    monkeypatch.setattr(engine, "_handle_exit_fill_instrumentation", _fake_exit_fill_instrumentation)
    monkeypatch.setattr(engine, "_handle_partial_fill_instrumentation", lambda *args, **kwargs: None)

    initial_state = restore_state(snapshot_state(build_alcb_runtime_state(engine)))

    await engine._handle_fill(
        SimpleNamespace(
            oms_order_id="OMS-ENTRY",
            payload={"price": 25.1, "qty": 100, "commission": 1.0},
        )
    )

    wrapper_snapshot = snapshot_state(build_alcb_runtime_state(engine))
    position = engine._positions["AAA"]
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    ALCBFill(
                        oms_order_id="OMS-ENTRY",
                        fill_price=25.1,
                        fill_qty=100,
                        fill_time=position.entry_time,
                        commission=1.0,
                        entry_context=ALCBEntryFillContext(trade_id=position.trade_id),
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
