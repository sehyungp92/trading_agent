from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from backtests.shared.parity.replay_driver import ReplayStep, run_replay
import pytest

from strategies.core.actions import CancelAction, FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProtectiveStop
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.core import logic as iaric_logic
from strategies.stock.iaric.core.logic import build_core_state as build_iaric_runtime_state
from strategies.stock.iaric.core.serializers import restore_state as restore_iaric_state
from strategies.stock.iaric.core.serializers import snapshot_state as snapshot_iaric_state
from strategies.stock.iaric.core.state import (
    IARICCoreState,
    IARICEntryRequest,
    IARICFill,
    IARICFlattenRequest,
    IARICOrderUpdate,
    IARICPartialExitRequest,
)
from strategies.stock.iaric.exits import check_v2_partial
from backtests.stock.engine.iaric_pullback_intraday_hybrid_engine import _PBHybridState
from strategies.stock.iaric.diagnostics import JsonlDiagnostics
from strategies.stock.iaric.engine import IARICEngine
from strategies.stock.iaric.models import Bar, MarketSnapshot, PBSymbolState, PendingOrderState, PositionState, RegimeSnapshot, WatchlistArtifact

UTC = timezone.utc


def test_v2_partial_requires_positive_profit_trigger() -> None:
    assert check_v2_partial(0.5, already_taken=False, trigger_r=0.5)
    assert not check_v2_partial(0.5, already_taken=False, trigger_r=0.0)
    assert not check_v2_partial(0.5, already_taken=False, trigger_r=-0.1)
    assert not check_v2_partial(1.0, already_taken=True, trigger_r=0.5)


def _state(*symbols: PBSymbolState) -> IARICCoreState:
    return IARICCoreState(
        trade_date=date(2026, 4, 26),
        saved_at=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        symbols=list(symbols),
        last_decision_code="IDLE",
        meta={
            "active_symbols": [symbol.symbol for symbol in symbols],
            "pending_entry_risk": {},
            "order_index": {},
        },
    )


def _item():
    return SimpleNamespace(expected_5m_volume=1_000.0, average_30m_volume=6_000.0)


def _bar(base: datetime, *, open_: float, high: float, low: float, close: float, volume: float = 1_200.0) -> Bar:
    return Bar(
        symbol="MSFT",
        start_time=base,
        end_time=base + timedelta(minutes=5),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _market(symbol: str, bars: list[Bar], *, vwap: float) -> MarketSnapshot:
    market = MarketSnapshot(symbol=symbol)
    market.session_vwap = vwap
    market.session_low = min(bar.low for bar in bars)
    market.session_high = max(bar.high for bar in bars)
    market.last_5m_bar = bars[-1]
    for bar in bars:
        market.bars_5m.append(bar)
    return market


def _backtest_route_state(item, *, daily_signal_score: float = 72.0, daily_atr: float = 1.6) -> _PBHybridState:
    return _PBHybridState(
        symbol="MSFT",
        item=item,
        record=None,
        trigger_type="RSI2",
        entry_rsi=12.0,
        entry_gap_pct=-1.0,
        entry_sma_dist_pct=3.0,
        entry_cdd=2,
        entry_rank=1,
        entry_rank_pct=25.0,
        n_candidates=1,
        prev_iloc=0,
        sector="Tech",
        daily_atr=daily_atr,
        daily_signal_score=daily_signal_score,
    )


def test_iaric_on_bar_entry_request_emits_submit_entry() -> None:
    state = _state(PBSymbolState(symbol="MSFT", route_family="OPENING_RECLAIM", stop_level=404.5))

    next_state, actions, events = iaric_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 14, 30, tzinfo=UTC),
        entry_request=IARICEntryRequest(
            client_order_id="ENTRY-1",
            symbol="MSFT",
            route="OPENING_RECLAIM",
            qty=25,
            limit_price=410.25,
            stop_price=404.5,
        ),
    )

    assert len(actions) == 1
    assert isinstance(actions[0], SubmitEntry)
    assert actions[0].qty == 25
    assert actions[0].limit_price == 410.25
    assert actions[0].risk_context["stop_for_risk"] == 404.5
    assert events[0].code == "ENTRY_REQUESTED"
    assert next_state.last_decision_code == "ENTRY_REQUESTED"


def test_iaric_on_fill_entry_creates_position_and_stop_action() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            route_family="VWAP_BOUNCE",
            stop_level=404.5,
            entry_order=PendingOrderState(
                oms_order_id="ENTRY-1",
                submitted_at=datetime(2026, 4, 26, 14, 29, tzinfo=UTC),
                role="ENTRY",
                requested_qty=25,
                limit_price=410.25,
            ),
            active_order_id="ENTRY-1",
        )
    )
    state.meta["pending_entry_risk"] = {"MSFT": 143.75}

    next_state, actions, events = iaric_logic.on_fill(
        state,
        IARICFill(
            oms_order_id="ENTRY-1",
            symbol="MSFT",
            order_role="ENTRY",
            fill_price=410.5,
            fill_qty=25,
            fill_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
            commission=1.25,
        ),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.in_position is True
    assert symbol_state.position is not None
    assert symbol_state.position.qty_open == 25
    assert symbol_state.entry_order is None
    assert next_state.meta["pending_entry_risk"] == {}
    assert len(actions) == 1
    assert isinstance(actions[0], SubmitProtectiveStop)
    assert actions[0].qty == 25
    assert events[0].code == "ENTRY_FILLED"


def test_iaric_on_bar_flatten_with_pending_tp_emits_cancel() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=12,
                final_stop=404.5,
                current_stop=407.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
            ),
            exit_order=PendingOrderState(
                oms_order_id="TP-1",
                submitted_at=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
                role="TP",
                requested_qty=13,
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_bar(
        state,
        bar_ts=datetime(2026, 4, 26, 15, 1, tzinfo=UTC),
        flatten_request=IARICFlattenRequest(symbol="MSFT", reason="FLOW_REVERSAL", qty=12),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.pending_hard_exit is True
    assert symbol_state.exit_order is not None
    assert symbol_state.exit_order.cancel_requested is True
    assert len(actions) == 1
    assert isinstance(actions[0], CancelAction)
    assert actions[0].target_order_id == "TP-1"
    assert events[0].code == "FLATTEN_QUEUED_AFTER_CANCEL"


def test_iaric_on_fill_partial_exit_resizes_stop() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            v2_partial_taken=False,
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=25,
                final_stop=404.5,
                current_stop=408.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
            ),
            exit_order=PendingOrderState(
                oms_order_id="TP-1",
                submitted_at=datetime(2026, 4, 26, 15, 0, tzinfo=UTC),
                role="TP",
                requested_qty=12,
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_fill(
        state,
        IARICFill(
            oms_order_id="TP-1",
            symbol="MSFT",
            order_role="TP",
            fill_price=413.0,
            fill_qty=12,
            fill_time=datetime(2026, 4, 26, 15, 2, tzinfo=UTC),
            commission=0.75,
        ),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.position is not None
    assert symbol_state.position.qty_open == 13
    assert symbol_state.v2_partial_taken is True
    assert len(actions) == 1
    assert isinstance(actions[0], ReplaceProtectiveStop)
    assert actions[0].qty == 13
    assert events[0].code == "PARTIAL_EXIT_FILLED"


def test_iaric_on_order_update_unexpected_stop_terminal_flattens() -> None:
    state = _state(
        PBSymbolState(
            symbol="MSFT",
            in_position=True,
            stage="IN_POSITION",
            position=PositionState(
                entry_price=410.5,
                qty_entry=25,
                qty_open=10,
                final_stop=404.5,
                current_stop=408.0,
                entry_time=datetime(2026, 4, 26, 14, 31, tzinfo=UTC),
                initial_risk_per_share=6.0,
                max_favorable_price=414.0,
                max_adverse_price=409.0,
                stop_order_id="STOP-1",
            ),
        )
    )

    next_state, actions, events = iaric_logic.on_order_update(
        state,
        IARICOrderUpdate(
            oms_order_id="STOP-1",
            symbol="MSFT",
            order_role="STOP",
            status="cancelled",
            timestamp=datetime(2026, 4, 26, 15, 5, tzinfo=UTC),
        ),
    )

    symbol_state = next(symbol_state for symbol_state in next_state.symbols if symbol_state.symbol == "MSFT")
    assert symbol_state.position is not None
    assert symbol_state.position.stop_order_id == ""
    assert len(actions) == 1
    assert isinstance(actions[0], FlattenPosition)
    assert actions[0].qty == 10
    assert events[0].code == "STOP_TERMINAL"


def test_iaric_shared_opening_reclaim_progression_matches_live_and_backtest_state_shapes() -> None:
    settings = StrategySettings(
        pb_opening_reclaim_enabled=True,
        pb_opening_reclaim_min_daily_signal_score=0.0,
        pb_flush_window_bars=3,
        pb_ready_acceptance_bars=1,
        pb_ready_min_volume_ratio=0.5,
        pb_ready_min_cpr=0.5,
    )
    item = _item()
    start = datetime(2026, 4, 26, 14, 30, tzinfo=UTC)
    bars = [
        _bar(start, open_=100.0, high=100.2, low=99.0, close=99.2, volume=1_000.0),
        _bar(start + timedelta(minutes=5), open_=99.2, high=100.4, low=99.1, close=100.1, volume=1_300.0),
        _bar(start + timedelta(minutes=10), open_=100.1, high=100.8, low=100.0, close=100.7, volume=1_500.0),
    ]

    live_state = PBSymbolState(symbol="MSFT", daily_signal_score=72.0, daily_atr=1.6)
    backtest_state = _backtest_route_state(item)

    live_steps = []
    backtest_steps = []
    for idx, bar in enumerate(bars):
        market = _market("MSFT", bars[: idx + 1], vwap=99.8)
        live_state.session_low = market.session_low or 0.0
        live_steps.append(
            iaric_logic.advance_opening_reclaim_route(
                settings, live_state, item, bar, market, idx, 1.0, bars=bars[: idx + 1]
            )
        )
        backtest_steps.append(
            iaric_logic.advance_opening_reclaim_route(
                settings, backtest_state, item, bar, market, idx, 1.0, bars=bars[: idx + 1]
            )
        )

    assert [step.stage if step is not None else None for step in live_steps] == ["FLUSH_LOCKED", "RECLAIMING", "READY"]
    assert [step.stage if step is not None else None for step in backtest_steps] == ["FLUSH_LOCKED", "RECLAIMING", "READY"]
    assert live_state.intraday_score == backtest_state.intraday_score
    assert live_state.target_entry_price == backtest_state.target_entry_price
    assert live_state.ready_bar_idx == backtest_state.ready_bar_idx == 2


def test_iaric_shared_delayed_confirm_and_ready_acceptance_match_live_and_backtest_state_shapes() -> None:
    settings = StrategySettings(
        pb_v2_enabled=False,
        pb_delayed_confirm_after_bar=5,
        pb_delayed_confirm_score_min=40.0,
        pb_entry_score_min=55.0,
        pb_ready_min_volume_ratio=0.5,
    )
    item = _item()
    start = datetime(2026, 4, 26, 14, 30, tzinfo=UTC)
    bars = [
        _bar(start + timedelta(minutes=5 * idx), open_=100.0 + idx * 0.05, high=100.2 + idx * 0.1, low=99.3, close=99.8 + idx * 0.08, volume=1_050.0)
        for idx in range(5)
    ]
    bars.append(_bar(start + timedelta(minutes=25), open_=100.2, high=101.2, low=99.4, close=100.95, volume=1_600.0))
    bars.append(_bar(start + timedelta(minutes=30), open_=100.9, high=101.1, low=100.6, close=100.98, volume=1_250.0))

    live_state = PBSymbolState(symbol="MSFT", daily_signal_score=82.0, daily_atr=1.6)
    backtest_state = _backtest_route_state(item, daily_signal_score=82.0)
    live_market = _market("MSFT", bars[:6], vwap=100.2)
    backtest_market = _market("MSFT", bars[:6], vwap=100.2)
    live_state.session_low = live_market.session_low or 0.0

    live_step = iaric_logic.activate_delayed_confirm_route(
        settings, live_state, item, bars[5], live_market, 5, 1.0, bars=bars[:6]
    )
    backtest_step = iaric_logic.activate_delayed_confirm_route(
        settings, backtest_state, item, bars[5], backtest_market, 5, 1.0, bars=bars[:6]
    )

    assert live_step is not None and live_step.stage == "READY"
    assert backtest_step is not None and backtest_step.stage == "READY"
    assert live_state.ready_bar_idx == backtest_state.ready_bar_idx == 5
    assert live_state.intraday_score == backtest_state.intraday_score

    live_market = _market("MSFT", bars[:7], vwap=100.25)
    backtest_market = _market("MSFT", bars[:7], vwap=100.25)
    live_state.session_low = live_market.session_low or 0.0
    live_accept = iaric_logic.evaluate_ready_entry(
        settings, live_state, item, bars[6], live_market, 6, 1.0, bars=bars[:7]
    )
    backtest_accept = iaric_logic.evaluate_ready_entry(
        settings, backtest_state, item, bars[6], backtest_market, 6, 1.0, bars=bars[:7]
    )

    assert live_accept is not None and live_accept.acceptance is not None
    assert backtest_accept is not None and backtest_accept.acceptance is not None
    assert live_accept.acceptance.accepted_bar_idx == backtest_accept.acceptance.accepted_bar_idx == 6
    assert live_accept.acceptance.accepted_entry_price == backtest_accept.acceptance.accepted_entry_price
    assert live_accept.acceptance.entry_trigger == backtest_accept.acceptance.entry_trigger == "DELAYED_CONFIRM"


def test_iaric_shared_thirty_min_context_bonus_matches_legacy_flat_bar_semantics() -> None:
    market = MarketSnapshot(symbol="MSFT")
    market.last_30m_bar = Bar(
        symbol="MSFT",
        start_time=datetime(2026, 4, 26, 14, 0, tzinfo=UTC),
        end_time=datetime(2026, 4, 26, 14, 30, tzinfo=UTC),
        open=192.0,
        high=192.0,
        low=192.0,
        close=192.0,
        volume=1_000.0,
    )

    assert iaric_logic.thirty_min_context_bonus(market, weight=4.0) == 2.0


def test_iaric_shared_volume_ratio_preserves_legacy_zero_expected_volume_behavior() -> None:
    bar = _bar(datetime(2026, 4, 26, 14, 30, tzinfo=UTC), open_=100.0, high=101.0, low=99.5, close=100.5, volume=480.0)
    item = SimpleNamespace(expected_5m_volume=0.0, average_30m_volume=0.0)

    assert iaric_logic.compute_volume_ratio(bar, item) == 480.0


def test_iaric_shared_reset_route_state_uses_strategy_specific_reset_defaults() -> None:
    state = _backtest_route_state(_item())
    state.stage = "READY"
    state.route_family = "DELAYED_CONFIRM"
    state.ready_bar_idx = 7
    state.accepted_bar_idx = 9
    state.accepted_entry_price = 101.25

    iaric_logic.reset_route_state(state)

    assert state.stage == "WATCHING"
    assert state.route_family == ""
    assert state.ready_bar_idx == 0
    assert state.accepted_bar_idx == -1
    assert state.accepted_entry_price == 0.0


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_iaric_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch, tmp_path) -> None:
    artifact = WatchlistArtifact(
        trade_date=date(2026, 4, 26),
        generated_at=datetime(2026, 4, 26, 13, 0, tzinfo=UTC),
        regime=RegimeSnapshot(
            score=0.75,
            tier="B",
            risk_multiplier=1.0,
            price_ok=True,
            breadth_ok=True,
            vol_ok=True,
            credit_ok=True,
        ),
        items=[],
        tradable=[],
        overflow=[],
    )
    engine = IARICEngine(
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        artifact=artifact,
        account_id="ACCT-1",
        nav=100_000.0,
        settings=StrategySettings(diagnostics_dir=str(tmp_path)),
        diagnostics=JsonlDiagnostics(Path(tmp_path), enabled=False),
    )
    engine._items["MSFT"] = SimpleNamespace(tick_size=0.01)
    engine._markets["MSFT"] = MarketSnapshot(symbol="MSFT")
    engine._symbols["MSFT"] = PBSymbolState(
        symbol="MSFT",
        route_family="VWAP_BOUNCE",
        stop_level=404.5,
        entry_order=PendingOrderState(
            oms_order_id="ENTRY-1",
            submitted_at=datetime(2026, 4, 26, 14, 29, tzinfo=UTC),
            role="ENTRY",
            requested_qty=25,
            limit_price=410.25,
        ),
        active_order_id="ENTRY-1",
    )
    engine._portfolio.pending_entry_risk["MSFT"] = 143.75
    engine._order_index["ENTRY-1"] = ("MSFT", "ENTRY")

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(engine, "_submit_stop", _noop)
    monkeypatch.setattr(engine, "_replace_stop", _noop)
    monkeypatch.setattr(engine, "_cancel_stop", _noop)
    monkeypatch.setattr(engine, "_submit_market_exit", _noop)
    monkeypatch.setattr(engine, "_record_entry_instrumentation", _noop)
    monkeypatch.setattr(engine, "_record_exit_instrumentation", _noop)

    initial_state = restore_iaric_state(snapshot_iaric_state(build_iaric_runtime_state(engine)))
    fill_time = datetime(2026, 4, 26, 14, 31, tzinfo=UTC)

    await engine._handle_fill(
        SimpleNamespace(
            oms_order_id="ENTRY-1",
            payload={"price": 410.5, "qty": 25, "commission": 1.25},
            timestamp=fill_time,
        )
    )

    wrapper_snapshot = snapshot_iaric_state(build_iaric_runtime_state(engine))
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    IARICFill(
                        oms_order_id="ENTRY-1",
                        fill_price=410.5,
                        fill_qty=25,
                        fill_time=fill_time,
                        commission=1.25,
                        symbol="MSFT",
                        order_role="ENTRY",
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: iaric_logic.on_bar(state, **payload),
        on_order_update=iaric_logic.on_order_update,
        on_fill=iaric_logic.on_fill,
    )

    replay_snapshot = snapshot_iaric_state(replay.state)
    replay_snapshot.pop("saved_at", None)
    wrapper_snapshot.pop("saved_at", None)

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert replay_snapshot == wrapper_snapshot
