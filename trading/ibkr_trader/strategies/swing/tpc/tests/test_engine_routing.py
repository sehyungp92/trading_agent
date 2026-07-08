"""Verify TPCEngine routes core events into kit calls in the right order."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import IntentType
from libs.oms.models.order import OrderRole
from strategies.core.actions import (
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitAddOnEntry,
    SubmitMarketExit,
    SubmitPartialExit,
)
from strategies.swing._shared.etf_core import ETFCoreState, ETFFill, ETFPosition, SetupSnapshot
from strategies.swing.tpc.config import SYMBOL_CONFIGS, TPCSymbolConfig
from strategies.swing.tpc.engine import TPCEngine
from strategies.swing.tpc.models import Direction


class _RecordingKit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def log_entry(self, **kw: Any) -> None: self.calls.append(("log_entry", kw))
    def log_exit(self, **kw: Any) -> None: self.calls.append(("log_exit", kw))
    def log_stop_adjustment(self, **kw: Any) -> None: self.calls.append(("log_stop_adjustment", kw))
    def log_missed(self, **kw: Any) -> None: self.calls.append(("log_missed", kw))
    def on_indicator_snapshot(self, **kw: Any) -> None: self.calls.append(("on_indicator_snapshot", kw))
    def on_filter_decision(self, **kw: Any) -> None: self.calls.append(("on_filter_decision", kw))
    def on_order_event(self, **kw: Any) -> None: self.calls.append(("on_order_event", kw))


def _engine(kit: _RecordingKit) -> TPCEngine:
    return TPCEngine(
        ib_session=object(),
        oms_service=object(),
        instruments={},
        config=SYMBOL_CONFIGS,
        kit=kit,
        equity=10_000.0,
    )


def _instrument(symbol: str = "QQQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )


def _setup(symbol: str = "QQQ") -> SetupSnapshot:
    ts = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)
    return SetupSnapshot(
        setup_id=f"TPC-{symbol}-1",
        strategy_id="TPC",
        symbol=symbol,
        direction=Direction.LONG,
        grade="a_plus",
        setup_type="TYPE_A",
        entry_model="market_next_bar",
        state="entry_ready",
        created_ts=ts,
        entry_price=100.0,
        stop_price=99.0,
        qty=10,
        score=18.0,
        risk_pct=0.0085,
        t1_r=1.5, t1_partial_pct=0.45,
        t2_r=2.75, t2_partial_pct=0.275,
        meta={
            "confirmations": ["vwap"], "depth": 0.4, "rr": 3.0, "atr_4h": 1.2,
            "asset_context_score": 1.0, "setup_lane": "primary", "pullback_timeframe": "1h",
            "score": 18.0, "daily_has_room": True, "orderly_pullback": True,
        },
    )


def _position(symbol: str = "QQQ") -> ETFPosition:
    return ETFPosition(
        setup_id=f"TPC-{symbol}-1",
        symbol=symbol,
        direction=Direction.LONG,
        qty_open=10, qty_initial=10,
        entry_price=100.0,
        current_stop=99.0, initial_stop=99.0,
        entry_ts=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc),
        risk_per_share=1.0,
        setup_type="TYPE_A", grade="a_plus",
        entry_model="market_next_bar", score=18.0,
        stop_order_id="QQQ-stop-1",
        mfe_price=102.5, mae_price=99.5,
        bars_held_15m=8,
        meta={"setup_lane": "primary"},
    )


def test_entry_filled_routes_to_log_entry():
    kit = _RecordingKit()
    engine = _engine(kit)
    setup = _setup()
    # Seed engine state and cache as if ENTRY_REQUESTED already fired
    engine._state.setups[setup.setup_id] = setup
    engine._setup_cache[setup.setup_id] = setup

    # Stub super().process_fill to mimic the parent producing an ENTRY_FILLED event
    from types import SimpleNamespace

    pos = _position()
    def _fake_super_process_fill(fill: ETFFill):
        engine._state.positions[fill.symbol] = pos
        ev = SimpleNamespace(
            code="ENTRY_FILLED",
            details={"setup_id": setup.setup_id, "qty": 10, "price": 100.0},
            symbol=fill.symbol, ts=fill.fill_time, timeframe="15m", strategy_id="TPC",
        )
        return [], [ev]

    fill = ETFFill(
        oms_order_id="oms-1", fill_price=100.0, fill_qty=10, symbol="QQQ",
        fill_time=datetime(2026, 5, 9, 14, 15, tzinfo=timezone.utc),
        commission=0.5, order_role="entry",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_fill
    try:
        parent_mod.ETFCoreLiveEngine.process_fill = lambda self, f: _fake_super_process_fill(f)
        engine.process_fill(fill)
    finally:
        parent_mod.ETFCoreLiveEngine.process_fill = original

    log_entries = [c for c in kit.calls if c[0] == "log_entry"]
    assert len(log_entries) == 1
    _, kwargs = log_entries[0]
    assert kwargs["trade_id"] == setup.setup_id
    assert kwargs["entry_price"] == 100.0
    assert kwargs["entry_signal"] == "TYPE_A"


@pytest.mark.asyncio
async def test_cycle_once_routes_core_entry_action_to_oms():
    kit = _RecordingKit()
    oms = SimpleNamespace(
        submit_intent=AsyncMock(return_value=SimpleNamespace(oms_order_id="oms-tpc-entry"))
    )
    engine = TPCEngine(
        ib_session=object(),
        oms_service=oms,
        instruments={"QQQ": _instrument("QQQ")},
        config={"QQQ": SYMBOL_CONFIGS["QQQ"]},
        kit=kit,
        equity=10_000.0,
    )
    action = SubmitEntry(
        client_order_id="TPC-QQQ-entry-1",
        symbol="QQQ",
        side="BUY",
        qty=10,
        order_type="LIMIT",
        limit_price=100.0,
        stop_price=99.0,
        risk_context={"stop_for_risk": 99.0, "planned_entry_price": 100.0},
        metadata={"setup_id": "TPC-QQQ-1"},
    )
    engine._build_bar_input = AsyncMock(return_value=SimpleNamespace(bar_15m=object()))
    engine.process_bar_input = lambda _bar_input: ([action], [])
    engine._persist_state = lambda: None

    await engine._cycle_once(request_kind="test")

    oms.submit_intent.assert_awaited_once()
    intent = oms.submit_intent.await_args.args[0]
    assert intent.order.strategy_id == "TPC"
    assert intent.order.client_order_id == "TPC-QQQ-entry-1"
    assert intent.order.risk_context.risk_dollars == pytest.approx(10.0)


def test_exit_filled_routes_to_log_exit_with_pre_position_mfe():
    kit = _RecordingKit()
    engine = _engine(kit)
    pre_pos = _position()
    engine._state.positions["QQQ"] = pre_pos

    from types import SimpleNamespace

    def _fake_super_process_fill(fill: ETFFill):
        engine._state.positions.pop(fill.symbol, None)
        ev = SimpleNamespace(
            code="EXIT_FILLED",
            details={"qty": 10, "price": 101.5, "reason": "T1"},
            symbol=fill.symbol, ts=fill.fill_time, timeframe="15m", strategy_id="TPC",
        )
        return [], [ev]

    fill = ETFFill(
        oms_order_id="oms-1", fill_price=101.5, fill_qty=10, symbol="QQQ",
        fill_time=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
        commission=0.5, order_role="exit",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_fill
    try:
        parent_mod.ETFCoreLiveEngine.process_fill = lambda self, f: _fake_super_process_fill(f)
        engine.process_fill(fill)
    finally:
        parent_mod.ETFCoreLiveEngine.process_fill = original

    log_exits = [c for c in kit.calls if c[0] == "log_exit"]
    assert len(log_exits) == 1
    _, kwargs = log_exits[0]
    assert kwargs["exit_reason"] == "TAKE_PROFIT_T1"
    assert kwargs["mfe_price"] == 102.5
    assert kwargs["mae_price"] == 99.5


def test_partial_exit_filled_does_not_call_log_exit():
    kit = _RecordingKit()
    engine = _engine(kit)
    engine._state.positions["QQQ"] = _position()

    from types import SimpleNamespace

    def _fake_super_process_fill(fill: ETFFill):
        ev = SimpleNamespace(
            code="PARTIAL_EXIT_FILLED",
            details={"qty": 4, "price": 101.5, "reason": "T1"},
            symbol=fill.symbol, ts=fill.fill_time, timeframe="15m", strategy_id="TPC",
        )
        return [], [ev]

    fill = ETFFill(
        oms_order_id="oms-1", fill_price=101.5, fill_qty=4, symbol="QQQ",
        fill_time=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
        commission=0.2, order_role="partial",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_fill
    try:
        parent_mod.ETFCoreLiveEngine.process_fill = lambda self, f: _fake_super_process_fill(f)
        engine.process_fill(fill)
    finally:
        parent_mod.ETFCoreLiveEngine.process_fill = original

    assert not any(c[0] == "log_exit" for c in kit.calls)


def test_replace_protective_stop_logs_only_when_changed():
    kit = _RecordingKit()
    engine = _engine(kit)
    pos = _position()
    engine._state.positions["QQQ"] = pos
    engine._position_stop_history[pos.setup_id] = 99.0  # prior stop

    from types import SimpleNamespace
    from strategies.swing._shared.etf_core import ETFBarInput

    bar = SimpleNamespace(timestamp=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
                          open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0)
    bar_input = ETFBarInput(symbol="QQQ", bar_15m=bar, indicators={"atr_4h": 1.2}, equity=10_000.0)

    raise_stop = ReplaceProtectiveStop(
        symbol="QQQ", target_order_id="QQQ-stop-1",
        side="SELL", stop_price=100.0, qty=10, reason="t1_profit_lock",
    )
    no_change_stop = ReplaceProtectiveStop(
        symbol="QQQ", target_order_id="QQQ-stop-1",
        side="SELL", stop_price=100.0, qty=10, reason="t1_profit_lock",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_bar_input

    def _fake_super_first(self, bi):
        return [raise_stop], [SimpleNamespace(code="MANAGING_POSITION", details={"qty": 10},
                                              symbol="QQQ", ts=bar.timestamp,
                                              timeframe="15m", strategy_id="TPC")]

    def _fake_super_second(self, bi):
        return [no_change_stop], [SimpleNamespace(code="MANAGING_POSITION", details={"qty": 10},
                                                  symbol="QQQ", ts=bar.timestamp,
                                                  timeframe="15m", strategy_id="TPC")]

    try:
        parent_mod.ETFCoreLiveEngine.process_bar_input = _fake_super_first
        engine.process_bar_input(bar_input)
        parent_mod.ETFCoreLiveEngine.process_bar_input = _fake_super_second
        engine.process_bar_input(bar_input)
    finally:
        parent_mod.ETFCoreLiveEngine.process_bar_input = original

    stop_calls = [c for c in kit.calls if c[0] == "log_stop_adjustment"]
    assert len(stop_calls) == 1
    _, kwargs = stop_calls[0]
    assert kwargs["adjustment_type"] == "BREAKEVEN"
    assert kwargs["trigger"] == "T1_HIT"
    assert kwargs["old_stop"] == 99.0
    assert kwargs["new_stop"] == 100.0


def test_setup_rejected_event_routes_to_log_missed():
    kit = _RecordingKit()
    engine = _engine(kit)

    from types import SimpleNamespace
    from strategies.swing._shared.etf_core import ETFBarInput

    bar = SimpleNamespace(timestamp=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
                          open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0)
    bar_input = ETFBarInput(symbol="QQQ", bar_15m=bar, indicators={"atr_4h": 1.2}, equity=10_000.0)

    rejection_event = SimpleNamespace(
        code="SETUP_REJECTED",
        details={
            "symbol": "QQQ", "lane": "primary",
            "blocked_by": "asset_context_score_low",
            "block_reason": "low cross-asset score",
            "direction": "LONG", "grade": "a",
        },
        symbol="QQQ", ts=bar.timestamp, timeframe="15m", strategy_id="TPC",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_bar_input
    try:
        parent_mod.ETFCoreLiveEngine.process_bar_input = lambda self, bi: ([], [rejection_event])
        engine.process_bar_input(bar_input)
    finally:
        parent_mod.ETFCoreLiveEngine.process_bar_input = original

    missed = [c for c in kit.calls if c[0] == "log_missed"]
    assert len(missed) == 1
    _, kwargs = missed[0]
    assert kwargs["blocked_by"] == "asset_context_score_low"
    assert kwargs["pair"] == "QQQ"
    assert kwargs["signal"] == "TPC_primary"


def test_indicator_snapshot_emitted_each_bar():
    kit = _RecordingKit()
    engine = _engine(kit)

    from types import SimpleNamespace
    from strategies.swing._shared.etf_core import ETFBarInput

    bar = SimpleNamespace(timestamp=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
                          open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0)
    bar_input = ETFBarInput(symbol="QQQ", bar_15m=bar,
                            indicators={"atr_4h": 1.2, "vwap_15m": 100.5}, equity=10_000.0)

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_bar_input
    try:
        no_signal = SimpleNamespace(code="NO_SIGNAL", details={}, symbol="QQQ",
                                    ts=bar.timestamp, timeframe="15m", strategy_id="TPC")
        parent_mod.ETFCoreLiveEngine.process_bar_input = lambda self, bi: ([], [no_signal])
        engine.process_bar_input(bar_input)
    finally:
        parent_mod.ETFCoreLiveEngine.process_bar_input = original

    snaps = [c for c in kit.calls if c[0] == "on_indicator_snapshot"]
    assert len(snaps) == 1
    _, kwargs = snaps[0]
    assert kwargs["pair"] == "QQQ"
    assert kwargs["signal_name"] == "NO_SIGNAL"
    assert kwargs["decision"] == "NO_SIGNAL"


def test_mfe_giveback_exit_routes_with_trailing_giveback_reason():
    """MFE-giveback final close → log_exit(TRAILING_GIVEBACK) with sign-correct pnl_pct."""
    kit = _RecordingKit()
    engine = _engine(kit)
    pre_pos = _position()  # entry=100, mfe=102.5, mae=99.5
    engine._state.positions["QQQ"] = pre_pos

    from types import SimpleNamespace

    def _fake_super_process_fill(fill: ETFFill):
        engine._state.positions.pop(fill.symbol, None)
        ev = SimpleNamespace(
            code="EXIT_FILLED",
            details={"qty": 10, "price": 101.2, "reason": "MFE_GIVEBACK"},
            symbol=fill.symbol, ts=fill.fill_time, timeframe="15m", strategy_id="TPC",
        )
        return [], [ev]

    fill = ETFFill(
        oms_order_id="oms-2", fill_price=101.2, fill_qty=10, symbol="QQQ",
        fill_time=datetime(2026, 5, 9, 14, 45, tzinfo=timezone.utc),
        commission=0.5, order_role="exit",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_fill
    try:
        parent_mod.ETFCoreLiveEngine.process_fill = lambda self, f: _fake_super_process_fill(f)
        engine.process_fill(fill)
    finally:
        parent_mod.ETFCoreLiveEngine.process_fill = original

    log_exits = [c for c in kit.calls if c[0] == "log_exit"]
    assert len(log_exits) == 1
    _, kwargs = log_exits[0]
    assert kwargs["exit_reason"] == "TRAILING_GIVEBACK"
    # LONG, exit > entry → positive pnl_pct = (101.2 - 100.0) / 100.0
    assert kwargs["pnl_pct"] == pytest.approx(0.012)
    # MAE pct must remain signed (LONG: mae below entry → negative)
    assert kwargs["mae_pct"] is not None and kwargs["mae_pct"] < 0


def test_order_terminal_event_routes_to_on_order_event():
    kit = _RecordingKit()
    engine = _engine(kit)

    from types import SimpleNamespace
    from strategies.swing._shared.etf_core import ETFOrderUpdate

    update = ETFOrderUpdate(
        oms_order_id="oms-99", status="rejected", symbol="QQQ",
        timestamp=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
        order_role="stop_replace",
    )
    terminal = SimpleNamespace(
        code="ORDER_TERMINAL",
        details={"setup_id": "TPC-QQQ-1", "status": "rejected"},
        symbol="QQQ", ts=update.timestamp, timeframe="15m", strategy_id="TPC",
    )

    import strategies.swing._shared.etf_live_engine as parent_mod
    original = parent_mod.ETFCoreLiveEngine.process_order_update
    try:
        parent_mod.ETFCoreLiveEngine.process_order_update = lambda self, u: ([], [terminal])
        engine.process_order_update(update)
    finally:
        parent_mod.ETFCoreLiveEngine.process_order_update = original

    order_events = [c for c in kit.calls if c[0] == "on_order_event"]
    assert len(order_events) == 1
    _, kwargs = order_events[0]
    assert kwargs["order_id"] == "oms-99"
    assert kwargs["pair"] == "QQQ"
    assert kwargs["status"] == "rejected"


@pytest.mark.asyncio
async def test_tpc_cycle_persists_after_successful_bar_mutation():
    kit = _RecordingKit()
    engine = _engine(kit)
    persisted = {"count": 0}

    from types import SimpleNamespace
    from strategies.swing._shared.etf_core import ETFBarInput

    bar = SimpleNamespace(
        timestamp=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1.0,
    )
    bar_input = ETFBarInput(symbol="QQQ", bar_15m=bar, indicators={}, equity=10_000.0)

    async def _build_bar_input(symbol: str, request_kind: str):
        return bar_input

    engine._config = {"QQQ": SYMBOL_CONFIGS["QQQ"]}
    engine._build_bar_input = _build_bar_input
    engine.process_bar_input = lambda bi: ([], [])
    engine._persist_state = lambda: persisted.__setitem__("count", persisted["count"] + 1)

    await engine._cycle_once(request_kind="test")

    assert persisted["count"] == 1


def test_tpc_state_persist_uses_atomic_replace(tmp_path):
    kit = _RecordingKit()
    engine = TPCEngine(
        ib_session=object(),
        oms_service=object(),
        instruments={},
        config=SYMBOL_CONFIGS,
        kit=kit,
        equity=10_000.0,
        state_dir=tmp_path,
    )

    engine._persist_state()

    snap_path = tmp_path / "TPC.json"
    assert snap_path.exists()
    assert not (tmp_path / "TPC.json.tmp").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected_role", "expected_cache_role"),
    [
        (
            SubmitAddOnEntry(
                client_order_id="add-1",
                symbol="QQQ",
                side="BUY",
                qty=3,
                order_type="LIMIT",
                limit_price=101.0,
                risk_context={"planned_entry_price": 101.0, "stop_for_risk": 99.0},
            ),
            OrderRole.ENTRY,
            "add_on_entry",
        ),
        (
            SubmitPartialExit(
                client_order_id="px-1",
                symbol="QQQ",
                side="SELL",
                qty=2,
                order_type="MARKET",
            ),
            OrderRole.EXIT,
            "partial_exit",
        ),
        (
            SubmitMarketExit(
                client_order_id="mx-1",
                symbol="QQQ",
                side="SELL",
                qty=5,
            ),
            OrderRole.EXIT,
            "market_exit",
        ),
    ],
)
async def test_tpc_dispatch_handles_add_on_and_exit_actions(
    action,
    expected_role: OrderRole,
    expected_cache_role: str,
):
    kit = _RecordingKit()
    engine = _engine(kit)
    engine._instruments["QQQ"] = _instrument()
    receipt = type("Receipt", (), {"oms_order_id": f"oms-{action.client_order_id}"})()
    oms = type("OMS", (), {})()
    oms.submit_intent = AsyncMock(return_value=receipt)
    engine._oms = oms

    await engine._dispatch_action(action, "QQQ")

    oms.submit_intent.assert_awaited_once()
    intent = oms.submit_intent.await_args.args[0]
    assert intent.intent_type == IntentType.NEW_ORDER
    assert intent.order.role == expected_role
    assert engine._oms_order_role[receipt.oms_order_id] == expected_cache_role
