from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from backtests.momentum.config_regime import NqRegimeBacktestConfig
from backtests.momentum.engine.regime_engine import NqRegimeTradeRecord, _TradeLedger, _extract_metrics, load_nq_regime_data
from backtests.momentum.engine.sim_broker import FillResult, FillStatus, OrderSide, OrderType, SimBroker, SimOrder
from libs.oms.models.events import OMSEventType
from strategies.core.actions import CancelAction, FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.momentum.nq_regime import config as nq_config
from strategies.momentum.nq_regime.config import Grade, ModuleId, StrategyRuntimeSettings, TradeSide
from strategies.momentum.nq_regime.core.data_policy import CompletedBarPolicy
from strategies.momentum.nq_regime.core.indicators import IndicatorSnapshot
from strategies.momentum.nq_regime.core.levels import KeyLevels, build_ib_levels
from strategies.momentum.nq_regime.core.logic import _manage_open_position, on_bar, on_fill, on_order_update
from strategies.momentum.nq_regime.core.regime import Regime, RegimeResult, RegimeScores
from strategies.momentum.nq_regime.core.serializers import hydrate_state, snapshot_state
from strategies.momentum.nq_regime.core.session import SessionPhase
from strategies.momentum.nq_regime.core.state import BarData, BarEvent, FillEvent, OrderUpdateEvent, RegimeCoreState
from strategies.momentum.nq_regime.engine import NQRegimeEngine
from strategies.momentum.nq_regime.modules import second_wind as second_wind_module
from strategies.momentum.nq_regime.modules.base import SetupCandidate
from strategies.momentum.nq_regime.modules.structural_expansion import evaluate as evaluate_structural
from strategies.scalp._shared.time_utils import session_date

ET = ZoneInfo("America/New_York")


def test_completed_bar_policy_attaches_15m_only_on_completed_boundary() -> None:
    policy = CompletedBarPolicy()
    bars = [
        _bar(10, 5, 100, 102, 99, 101, 100),
        _bar(10, 10, 101, 103, 100, 102, 120),
        _bar(10, 15, 102, 104, 101, 103, 140),
    ]

    no_15m = policy.build_event(bar_5m=bars[0], recent_5m=bars[:1])
    with_15m = policy.build_event(bar_5m=bars[-1], recent_5m=bars)

    assert no_15m.is_new_15m is False
    assert no_15m.bar_15m_closed is None
    assert with_15m.is_new_15m is True
    assert with_15m.bar_15m_closed is not None
    assert with_15m.bar_15m_closed.open == 100
    assert with_15m.bar_15m_closed.high == 104
    assert with_15m.bar_15m_closed.low == 99
    assert with_15m.bar_15m_closed.close == 103


def test_nq_regime_loader_shifts_start_labeled_bars_to_close_availability(tmp_path) -> None:
    index = pd.DatetimeIndex(
        [
            "2026-04-27 13:30:00+00:00",
            "2026-04-27 19:55:00+00:00",
            "2026-04-27 20:00:00+00:00",
        ],
        name="time",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.5, 110.5, 120.5],
            "volume": [10, 20, 30],
        },
        index=index,
    )
    frame.to_parquet(tmp_path / "NQ_5m.parquet", engine="pyarrow", index=True)

    data = load_nq_regime_data(NqRegimeBacktestConfig(data_dir=tmp_path))
    times = pd.to_datetime(data.bars_5m.times, utc=True)

    assert list(times) == [
        pd.Timestamp("2026-04-27 13:35:00+00:00"),
        pd.Timestamp("2026-04-27 20:00:00+00:00"),
    ]


def test_nq_regime_loader_preserves_close_labeled_bars_when_configured(tmp_path) -> None:
    index = pd.DatetimeIndex(
        [
            "2026-04-27 13:35:00+00:00",
            "2026-04-27 20:00:00+00:00",
            "2026-04-27 20:05:00+00:00",
        ],
        name="time",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.5, 110.5, 120.5],
            "volume": [10, 20, 30],
        },
        index=index,
    )
    frame.to_parquet(tmp_path / "NQ_5m.parquet", engine="pyarrow", index=True)

    data = load_nq_regime_data(NqRegimeBacktestConfig(data_dir=tmp_path, bar_timestamp_mode="close"))
    times = pd.to_datetime(data.bars_5m.times, utc=True)

    assert list(times) == [
        pd.Timestamp("2026-04-27 13:35:00+00:00"),
        pd.Timestamp("2026-04-27 20:00:00+00:00"),
    ]


def test_structural_expansion_candidate_reflects_ib_acceptance_breakout_edge() -> None:
    breakout = _bar(10, 15, 20020, 20060, 20012, 20055, 3_000)
    state = RegimeCoreState(
        active_session_date=session_date(breakout.ts).isoformat(),
        ib_levels=build_ib_levels(20000, 19900),
        ib_high_working=20000,
        ib_low_working=19900,
        ib_locked=True,
        levels=KeyLevels(pdh=20200),
        bars_15m=[
            _bar(9, 45, 19955, 19975, 19945, 19960, 900),
            _bar(10, 0, 19960, 19995, 19950, 19980, 900),
            breakout,
        ],
    )
    state.ib_type = state.ib_levels.ib_type
    indicators = IndicatorSnapshot(
        vwap=19980,
        atr_15m=18,
        volume_multiple_15m=1.8,
        trend_direction=1,
    )
    event = BarEvent(ts=breakout.ts, bar_5m=breakout, bar_15m_closed=breakout, is_new_15m=True)

    candidate = evaluate_structural(state, event, indicators)

    assert candidate is not None
    assert candidate.module is ModuleId.STRUCTURAL_EXPANSION
    assert candidate.side is TradeSide.LONG
    # Round 6 uses structure_shift entries, so the breakout edge can promote
    # directly instead of being vetoed by the old retest-distance guard.
    assert candidate.grade is Grade.A_PLUS
    assert candidate.valid
    assert candidate.entry_price > state.ib_levels.high
    assert candidate.stop_price < candidate.entry_price
    assert candidate.target_room_r >= 1.5


def test_structural_expansion_pm_requires_existing_ib_expansion() -> None:
    breakout = _bar(13, 45, 20020, 20060, 20012, 20055, 3_000)
    state = RegimeCoreState(
        active_session_date=session_date(breakout.ts).isoformat(),
        phase=SessionPhase.PM_CONTINUATION,
        ib_levels=build_ib_levels(20000, 19900),
        ib_high_working=20000,
        ib_low_working=19900,
        ib_locked=True,
        levels=KeyLevels(pdh=20200),
        bars_15m=[
            _bar(10, 0, 19950, 19990, 19940, 19980, 2_000),
            _bar(13, 30, 19980, 20010, 19970, 20000, 1_400),
            breakout,
        ],
    )
    state.ib_type = state.ib_levels.ib_type
    indicators = IndicatorSnapshot(vwap=19980, atr_15m=18, volume_multiple_15m=1.8, trend_direction=1)
    event = BarEvent(ts=breakout.ts, bar_5m=breakout, bar_15m_closed=breakout, is_new_15m=True)

    assert evaluate_structural(state, event, indicators) is None

    state.expansion_state.active_break_side = TradeSide.LONG
    assert evaluate_structural(state, event, indicators) is not None


def test_core_routes_valid_candidate_to_neutral_entry_and_snapshot_hydrates_candidate(monkeypatch) -> None:
    ts = _dt(10, 15)
    state = _ready_state(ts)
    candidate = _candidate(ts)

    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.STRUCTURAL_EXPANSION, RegimeScores(expansion=1.0), 1.0, 1.0),
    )
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.modules.structural_expansion.evaluate",
        lambda *args, **kwargs: candidate,
    )

    event = BarEvent(ts=ts, bar_5m=_bar(10, 15, 20020, 20040, 20010, 20030, 2_000))
    next_state, actions, events = on_bar(
        state,
        event,
        settings=StrategyRuntimeSettings(initial_equity=100_000, max_contracts=5, enable_liquidity_reversion=False, enable_second_wind=False),
    )

    assert any(isinstance(action, SubmitEntry) for action in actions)
    assert events[-1].code == "ENTRY_REQUESTED"
    assert next_state.working_entry_order_id
    hydrated = hydrate_state(snapshot_state(next_state))
    assert isinstance(next(iter(hydrated.pending_candidates.values())), SetupCandidate)


def test_entry_fill_places_protective_orders_and_partial_r_uses_full_trade_risk(monkeypatch) -> None:
    ts = _dt(10, 15)
    state = _ready_state(ts)
    candidate = _candidate(ts)
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.STRUCTURAL_EXPANSION, RegimeScores(expansion=1.0), 1.0, 1.0),
    )
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.modules.structural_expansion.evaluate",
        lambda *args, **kwargs: candidate,
    )
    state, entry_actions, _ = on_bar(
        state,
        BarEvent(ts=ts, bar_5m=_bar(10, 15, 20020, 20040, 20010, 20030, 2_000)),
        settings=StrategyRuntimeSettings(initial_equity=100_000, max_contracts=10, enable_liquidity_reversion=False, enable_second_wind=False),
    )
    entry = next(action for action in entry_actions if isinstance(action, SubmitEntry))

    state, child_actions, fill_events = on_fill(
        state,
        FillEvent(entry.client_order_id, fill_price=20000, fill_qty=10, fill_time=ts, symbol="MNQ"),
    )

    assert fill_events[-1].code == "ENTRY_FILLED"
    assert any(isinstance(action, SubmitProtectiveStop) for action in child_actions)
    target = next(action for action in child_actions if isinstance(action, SubmitProfitTarget))

    state, partial_actions, partial_events = on_fill(
        state,
        FillEvent(target.client_order_id, fill_price=20010, fill_qty=target.qty, fill_time=_dt(10, 20), symbol="MNQ", order_role="target_1"),
    )

    assert partial_events[-1].code == "PARTIAL_EXIT_FILLED"
    assert any(isinstance(action, ReplaceProtectiveStop) for action in partial_actions)
    assert any(isinstance(action, SubmitProfitTarget) and action.role == "target_2" for action in partial_actions)
    assert state.qty_open == 10 - target.qty
    assert state.daily_realized_r == pytest.approx(target.qty / 10, abs=1e-9)


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_nq_regime_live_wrapper_entry_fill_matches_replay_core_state(tmp_path, monkeypatch) -> None:
    ts = _dt(10, 15)
    engine = NQRegimeEngine(
        ib_session=None,
        oms_service=None,
        instruments={},
        state_dir=tmp_path,
        instrumentation=None,
    )
    engine._settings = StrategyRuntimeSettings(
        initial_equity=100_000,
        max_contracts=10,
        enable_liquidity_reversion=False,
        enable_second_wind=False,
    )
    state = _ready_state(ts)
    candidate = _candidate(ts)
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.STRUCTURAL_EXPANSION, RegimeScores(expansion=1.0), 1.0, 1.0),
    )
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.modules.structural_expansion.evaluate",
        lambda *args, **kwargs: candidate,
    )
    state, entry_actions, _ = on_bar(
        state,
        BarEvent(ts=ts, bar_5m=_bar(10, 15, 20020, 20040, 20010, 20030, 2_000)),
        settings=engine._settings,
    )
    entry = next(action for action in entry_actions if isinstance(action, SubmitEntry))
    engine._state = hydrate_state(snapshot_state(state))
    initial_state = hydrate_state(snapshot_state(engine._state))
    fill_time = _dt(10, 20)
    wrapper_events = []
    record_events = engine._record_events

    def _capture_recorded_events(events):
        wrapper_events.extend(events)
        record_events(events)

    monkeypatch.setattr(engine, "_record_events", _capture_recorded_events)

    await engine._handle_oms_event(
        SimpleNamespace(
            event_type=OMSEventType.FILL,
            oms_order_id=entry.client_order_id,
            payload={"price": 20000, "qty": 10, "commission": 0.0},
            timestamp=fill_time,
        )
    )

    wrapper_snapshot = snapshot_state(engine._state)
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    FillEvent(
                        entry.client_order_id,
                        fill_price=20000,
                        fill_qty=10,
                        fill_time=fill_time,
                        symbol="MNQ",
                    )
                ]
            )
        ],
        on_bar=lambda replay_state, event: on_bar(replay_state, event, settings=engine._settings),
        on_order_update=on_order_update,
        on_fill=on_fill,
    )

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert normalize_decision_stream(wrapper_events) == normalize_decision_stream(replay.events)
    assert snapshot_state(replay.state) == wrapper_snapshot


def test_ib_lock_includes_1000_close_bar(monkeypatch) -> None:
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.TRANSITION, RegimeScores(), 0.0, 0.0),
    )
    state = RegimeCoreState()
    state, _, _ = on_bar(state, BarEvent(ts=_dt(9, 55), bar_5m=_bar(9, 55, 19950, 19980, 19940, 19970)))
    state, _, _ = on_bar(state, BarEvent(ts=_dt(10, 0), bar_5m=_bar(10, 0, 19970, 20010, 19935, 20000)))

    assert state.ib_locked is True
    assert state.ib_levels.high == 20010
    assert state.ib_levels.low == 19935


def test_retest_models_submit_resting_entries(monkeypatch) -> None:
    ts = _dt(13, 45)
    state = _ready_state(ts)
    candidate = _candidate(ts)
    candidate = replace(candidate, module=ModuleId.SECOND_WIND, entry_model="breakout_close_retest")
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.PM_CONTINUATION, RegimeScores(pm_continuation=1.0), 1.0, 1.0),
    )
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.modules.second_wind.evaluate",
        lambda *args, **kwargs: candidate,
    )

    state, actions, _ = on_bar(
        state,
        BarEvent(ts=ts, bar_5m=_bar(13, 45, 20020, 20040, 20010, 20030, 2_000)),
        settings=StrategyRuntimeSettings(initial_equity=100_000, max_contracts=5, enable_structural_expansion=False, enable_liquidity_reversion=False),
    )

    entry = next(action for action in actions if isinstance(action, SubmitEntry))
    assert entry.order_type == "LIMIT"
    assert entry.limit_price == candidate.entry_price
    assert entry.metadata["ttl_minutes"] == 120


def test_final_stop_fill_cancels_remaining_targets(monkeypatch) -> None:
    ts = _dt(10, 15)
    state = _ready_state(ts)
    candidate = _candidate(ts)
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.STRUCTURAL_EXPANSION, RegimeScores(expansion=1.0), 1.0, 1.0),
    )
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.modules.structural_expansion.evaluate",
        lambda *args, **kwargs: candidate,
    )
    state, entry_actions, _ = on_bar(
        state,
        BarEvent(ts=ts, bar_5m=_bar(10, 15, 20020, 20040, 20010, 20030, 2_000)),
        settings=StrategyRuntimeSettings(initial_equity=100_000, max_contracts=10, enable_liquidity_reversion=False, enable_second_wind=False),
    )
    entry = next(action for action in entry_actions if isinstance(action, SubmitEntry))
    state, child_actions, _ = on_fill(state, FillEvent(entry.client_order_id, 20000, 10, ts, symbol="MNQ"))
    stop = next(action for action in child_actions if isinstance(action, SubmitProtectiveStop))

    state, stop_actions, _ = on_fill(
        state,
        FillEvent(stop.client_order_id, fill_price=19990, fill_qty=10, fill_time=_dt(10, 20), symbol="MNQ", order_role="stop"),
    )

    assert state.position_side is TradeSide.FLAT
    assert sum(isinstance(action, CancelAction) for action in stop_actions) == 1


def test_cancelled_entry_order_update_clears_pending_candidate(monkeypatch) -> None:
    ts = _dt(10, 15)
    state = _ready_state(ts)
    candidate = _candidate(ts)
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.core.logic.classify_regime",
        lambda *args, **kwargs: RegimeResult(Regime.STRUCTURAL_EXPANSION, RegimeScores(expansion=1.0), 1.0, 1.0),
    )
    monkeypatch.setattr(
        "strategies.momentum.nq_regime.modules.structural_expansion.evaluate",
        lambda *args, **kwargs: candidate,
    )
    state, actions, _ = on_bar(
        state,
        BarEvent(ts=ts, bar_5m=_bar(10, 15, 20020, 20040, 20010, 20030, 2_000)),
        settings=StrategyRuntimeSettings(initial_equity=100_000, max_contracts=10, enable_liquidity_reversion=False, enable_second_wind=False),
    )
    entry = next(action for action in actions if isinstance(action, SubmitEntry))

    state, _, events = on_order_update(
        state,
        OrderUpdateEvent(entry.client_order_id, "cancelled", _dt(10, 20), symbol="MNQ", order_role="entry"),
    )

    assert events[-1].code == "ORDER_TERMINAL"
    assert state.working_entry_order_id is None
    assert state.pending_candidates == {}
    assert state.last_submitted_signal_id is None


def test_snapshot_hydrate_restores_runtime_types() -> None:
    state = RegimeCoreState(regime=Regime.STRUCTURAL_EXPANSION, regime_scores=RegimeScores(expansion=1.0))
    state.indicators = IndicatorSnapshot(vwap=20000)

    restored = hydrate_state(snapshot_state(state))

    assert restored.regime is Regime.STRUCTURAL_EXPANSION
    assert isinstance(restored.regime_scores, RegimeScores)
    assert isinstance(restored.indicators, IndicatorSnapshot)


def test_sim_broker_oca_prevents_same_bar_stop_and_target_double_fill() -> None:
    broker = SimBroker()
    ts = _dt(10, 20)
    broker.submit_order(
        SimOrder(
            order_id="stop",
            symbol="MNQ",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            qty=1,
            stop_price=19990,
            tick_size=0.25,
            submit_time=_dt(10, 15),
            tag="stop",
            oca_group="trade-stage-1",
        )
    )
    broker.submit_order(
        SimOrder(
            order_id="target",
            symbol="MNQ",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=1,
            limit_price=20010,
            tick_size=0.25,
            submit_time=_dt(10, 15),
            tag="target_1",
            oca_group="trade-stage-1",
        )
    )

    fills = broker.process_bar("MNQ", ts, 20000, 20020, 19980, 20005, 0.25)

    assert [(fill.order.order_id, fill.status) for fill in fills] == [
        ("stop", FillStatus.FILLED),
        ("target", FillStatus.CANCELLED),
    ]


def test_limit_fill_reports_whether_fill_was_at_bar_open() -> None:
    broker = SimBroker()
    ts = _dt(10, 20)
    broker.submit_order(
        SimOrder(
            order_id="gap-limit",
            symbol="MNQ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=1,
            limit_price=100.0,
            tick_size=0.25,
            submit_time=_dt(10, 15),
            tag="entry",
        )
    )
    gap_fill = broker.process_bar("MNQ", ts, 99.5, 101.0, 99.0, 100.5, 0.25)[0]

    broker.submit_order(
        SimOrder(
            order_id="intrabar-limit",
            symbol="MNQ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=1,
            limit_price=100.0,
            tick_size=0.25,
            submit_time=_dt(10, 25),
            tag="entry",
        )
    )
    intrabar_fill = broker.process_bar("MNQ", _dt(10, 30), 101.0, 101.5, 99.0, 100.5, 0.25)[0]

    assert gap_fill.filled_at_open is True
    assert intrabar_fill.filled_at_open is False


def test_trade_ledger_includes_exit_fill_price_in_excursion_bounds() -> None:
    ledger = _TradeLedger(point_value=1.0)
    state = RegimeCoreState()
    entry = _fill_result("entry", OrderSide.BUY, 100.0, _dt(10, 20), tag="entry", stop_for_risk=90.0, filled_at_open=True)

    assert ledger.on_fill(entry, state) is None
    completed = ledger.on_fill(
        _fill_result("target", OrderSide.SELL, 112.0, _dt(10, 25), tag="target_1"),
        state,
    )

    assert completed is not None
    assert completed.r_multiple == pytest.approx(1.2)
    assert completed.mfe_r >= completed.r_multiple
    assert completed.mae_r <= completed.r_multiple


def test_trade_ledger_preserves_flatten_exit_reason_from_metadata() -> None:
    ledger = _TradeLedger(point_value=1.0)
    state = RegimeCoreState()
    entry = _fill_result("entry", OrderSide.BUY, 100.0, _dt(10, 20), tag="entry", stop_for_risk=90.0, filled_at_open=True)

    assert ledger.on_fill(entry, state) is None
    completed = ledger.on_fill(
        _fill_result(
            "flatten",
            OrderSide.SELL,
            104.0,
            _dt(10, 25),
            tag="flatten",
            metadata={"exit_reason": "reversion_vwap_touch"},
        ),
        state,
    )

    assert completed is not None
    assert completed.exit_reason == "reversion_vwap_touch"


def test_reversion_time_stop_is_module_specific_capture_exit(monkeypatch) -> None:
    monkeypatch.setattr(nq_config, "REVERSION_TIME_STOP_ENABLED", True)
    monkeypatch.setattr(nq_config, "REVERSION_TIME_STOP_BARS", 6)
    monkeypatch.setattr(nq_config, "REVERSION_TIME_STOP_MIN_MFE_R", 0.50)
    state = RegimeCoreState(
        position_side=TradeSide.LONG,
        entry_price=100.0,
        entry_bar_index=4,
        stop_price=90.0,
        qty=2,
        qty_open=2,
        entry_module=ModuleId.LIQUIDITY_REVERSION,
        initial_risk_points=10.0,
        planned_targets=(105.0, 110.0),
        active_trade_id="capture-test",
        working_stop_order_id="stop-1",
        working_target_order_ids=("target-1",),
        bar_index=10,
    )
    event = BarEvent(ts=_dt(11, 0), bar_5m=_bar(11, 0, 101.0, 104.0, 100.0, 102.0))

    actions, details = _manage_open_position(state, event, "MNQ")

    assert details["management_action"] == "reversion_time_stop"
    assert any(isinstance(action, CancelAction) and action.reason == "reversion_time_stop" for action in actions)
    flatten = next(action for action in actions if isinstance(action, FlattenPosition))
    assert flatten.reason == "reversion_time_stop"
    assert flatten.qty == 2


def test_trade_ledger_does_not_mark_full_entry_bar_for_intrabar_limit_fill() -> None:
    ledger = _TradeLedger(point_value=1.0)
    state = RegimeCoreState()
    entry = _fill_result("entry", OrderSide.BUY, 100.0, _dt(10, 20), tag="entry", stop_for_risk=90.0, filled_at_open=False)

    ledger.on_fill(entry, state)
    ledger.mark_bar(_dt(10, 20), high=130.0, low=95.0)
    completed = ledger.on_fill(
        _fill_result("target", OrderSide.SELL, 110.0, _dt(10, 25), tag="target_1"),
        state,
    )

    assert completed is not None
    assert completed.mfe_r == pytest.approx(1.0)
    assert completed.mae_r == pytest.approx(0.0)


def test_second_wind_subfamily_controls_veto_weak_vwap_and_second_leg(monkeypatch) -> None:
    ts = _dt(13, 45)
    state = RegimeCoreState(
        regime=Regime.PM_CONTINUATION,
        levels=KeyLevels(pdh=20250),
        bars_5m=[
            _bar(13, 15, 20080, 20100, 20070, 20095, 1_000),
            _bar(13, 20, 20095, 20105, 20085, 20100, 1_000),
            _bar(13, 25, 20100, 20110, 20090, 20105, 1_000),
            _bar(13, 30, 20105, 20115, 20095, 20110, 1_000),
            _bar(13, 35, 20110, 20120, 20100, 20115, 1_000),
            _bar(13, 40, 20115, 20125, 20105, 20120, 1_000),
        ],
        bars_15m=[_bar(11, minute, 20000 + minute, 20020 + minute, 19990 + minute, 20010 + minute, 2_000) for minute in range(0, 50, 5)],
    )
    indicators = IndicatorSnapshot(
        atr_5m=10.0,
        atr_15m=12.0,
        ema9_15m=20120.0,
        ema20_15m=20105.0,
        ema50_15m=20090.0,
        trend_direction=1,
        squeeze_duration=5,
        am_vwap_control=0.5,
    )
    monkeypatch.setattr(nq_config, "SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION", 0.75)
    monkeypatch.setattr(nq_config, "SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR", 0.10)
    monkeypatch.setattr(nq_config, "SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR", 0.10)
    monkeypatch.setattr(nq_config, "SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE", True)

    vwap_candidate = second_wind_module._build_candidate(
        state,
        _bar(13, 45, 20100, 20108, 20098, 20104),
        indicators,
        TradeSide.LONG,
        setup_type="pm_vwap_reclaim",
        level=20100.0,
        score=10,
        details={"close_location": 0.60, "volume_multiple": 1.5, "reclaim_distance": 0.5, "ema_aligned": True},
    )
    second_leg_candidate = second_wind_module._build_candidate(
        state,
        _bar(13, 50, 20110, 20118, 20105, 20116),
        indicators,
        TradeSide.LONG,
        setup_type="pm_second_leg",
        level=20115.5,
        score=10,
        details={"close_location": 0.80, "volume_multiple": 1.5, "breakout_distance": 0.5, "had_impulse": False},
    )

    assert vwap_candidate is not None
    assert vwap_candidate.grade is Grade.INVALID
    assert "weak_setup_close_location" in vwap_candidate.vetoes
    assert "weak_vwap_reclaim_distance" in vwap_candidate.vetoes
    assert second_leg_candidate is not None
    assert second_leg_candidate.grade is Grade.INVALID
    assert "weak_second_leg_breakout" in second_leg_candidate.vetoes
    assert "second_leg_missing_impulse" in second_leg_candidate.vetoes


def test_extract_metrics_exposes_second_wind_subfamily_edges() -> None:
    trades = [
        NqRegimeTradeRecord(
            symbol="MNQ",
            side="BUY",
            qty=1,
            entry_time=_dt(13, 45),
            module="second_wind",
            setup_type="pm_vwap_reclaim",
            pnl_dollars=-50.0,
            r_multiple=-0.50,
            mfe_r=0.25,
            mae_r=-0.70,
        ),
        NqRegimeTradeRecord(
            symbol="MNQ",
            side="BUY",
            qty=1,
            entry_time=_dt(14, 15),
            module="second_wind",
            setup_type="pm_second_leg",
            pnl_dollars=120.0,
            r_multiple=1.20,
            mfe_r=1.50,
            mae_r=-0.20,
        ),
    ]

    metrics = _extract_metrics(trades, [10_000.0, 10_070.0], [_dt(13, 45), _dt(14, 30)], 10_000.0, [])

    assert metrics["module_second_wind_pm_vwap_reclaim_trades"] == 1.0
    assert metrics["module_second_wind_pm_vwap_reclaim_avg_r"] == pytest.approx(-0.50)
    assert metrics["module_second_wind_pm_second_leg_trades"] == 1.0
    assert metrics["module_second_wind_pm_second_leg_avg_r"] == pytest.approx(1.20)


def _ready_state(ts: datetime) -> RegimeCoreState:
    state = RegimeCoreState(
        active_session_date=session_date(ts).isoformat(),
        ib_levels=build_ib_levels(20000, 19900),
        ib_high_working=20000,
        ib_low_working=19900,
        ib_locked=True,
        levels=KeyLevels(pdh=20200),
        bars_5m=[_bar(10, 0, 19950, 19980, 19940, 19970, 1_000), _bar(10, 5, 19970, 19995, 19960, 19990, 1_000)],
        bars_15m=[_bar(10, 0, 19950, 19990, 19940, 19980, 2_000)],
    )
    state.ib_type = state.ib_levels.ib_type
    return state


def _candidate(ts: datetime) -> SetupCandidate:
    return SetupCandidate(
        candidate_id="candidate-1",
        module=ModuleId.STRUCTURAL_EXPANSION,
        side=TradeSide.LONG,
        setup_type="test_breakout",
        timestamp=ts,
        level=20000,
        score=10,
        grade=Grade.A,
        entry_price=20000,
        stop_price=19990,
        targets=(20010, 20020, 20030),
        entry_model="breakout_close",
        risk_pct=0.0,
        invalidation_price=19950,
        target_room_r=3.0,
    )


def _bar(hour: int, minute: int, open_: float, high: float, low: float, close: float, volume: float = 1_000) -> BarData:
    return BarData(ts=_dt(hour, minute), open=open_, high=high, low=low, close=close, volume=volume)


def _dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 4, 27, hour, minute, tzinfo=ET)


def _fill_result(
    order_id: str,
    side: OrderSide,
    price: float,
    ts: datetime,
    *,
    tag: str,
    stop_for_risk: float | None = None,
    filled_at_open: bool = False,
    metadata: dict[str, object] | None = None,
    commission: float = 0.0,
) -> FillResult:
    order = SimOrder(
        order_id=order_id,
        symbol="MNQ",
        side=side,
        order_type=OrderType.LIMIT,
        qty=1,
        limit_price=price,
        tick_size=0.25,
        submit_time=ts,
        tag=tag,
    )
    if stop_for_risk is not None or metadata:
        order.metadata = dict(metadata or {})
        if stop_for_risk is not None:
            order.metadata.update({"module": ModuleId.SECOND_WIND.value, "stop_for_risk": stop_for_risk})
    return FillResult(
        order=order,
        status=FillStatus.FILLED,
        fill_price=price,
        fill_time=ts,
        commission=commission,
        filled_at_open=filled_at_open,
    )
