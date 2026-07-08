from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from backtests.momentum.data.preprocessing import NumpyBars
from backtests.momentum.engine import vdubus_engine as vdub_backtest_engine
from backtests.momentum.config_vdubus import VdubusBacktestConfig
from backtests.momentum.engine.vdubus_engine import (
    VdubusEngine as BacktestVdubusEngine,
    VdubusTradeRecord,
    _ActivePosition,
)
from backtests.shared.parity.replay_driver import ReplayStep, run_replay
from strategies.momentum.vdub import config as vdub_config
from strategies.momentum.vdub import engine as vdub_live_engine
from strategies.momentum.vdub.core.logic import build_core_state as build_vdub_runtime_state
from strategies.momentum.vdub.core.logic import on_bar as on_vdub_bar
from strategies.momentum.vdub.core.logic import on_fill as on_vdub_fill
from strategies.momentum.vdub.core.logic import on_order_update as on_vdub_order_update
from strategies.momentum.vdub.core.serializers import (
    restore_state as restore_vdub_state,
    snapshot_state as snapshot_vdub_state,
)
from strategies.momentum.vdub.core.state import VdubCoreState, VdubEntryFillContext, VdubFill
from strategies.momentum.vdub.engine import VdubNQv4Engine
from strategies.momentum.vdub.models import (
    DayCounters,
    Direction as VdubDirection,
    EntryType as VdubEntryType,
    EventBlockState,
    PositionStage,
    PositionState,
    RegimeState,
    SessionWindow as VdubSessionWindow,
    SubWindow as VdubSubWindow,
    VolState,
    WorkingEntry,
)
from strategies.stock.iaric.core.serializers import (
    restore_state as restore_iaric_state,
    snapshot_state as snapshot_iaric_state,
)
from strategies.stock.iaric.models import IntradayStateSnapshot, PBSymbolState
from strategies.swing.akc_helix.core.serializers import (
    restore_state as restore_akc_helix_state,
    snapshot_state as snapshot_akc_helix_state,
)
from strategies.swing.akc_helix.core.state import AKCHelixCoreState
from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.akc_helix.models import (
    CircuitBreakerState as AKCHelixCircuitBreakerState,
    PivotStore,
    Regime,
    TFState as AKCHelixTFState,
)
from strategies.swing.atrss.core.serializers import (
    restore_state as restore_atrss_state,
    snapshot_state as snapshot_atrss_state,
)
from strategies.swing.atrss.core.state import ATRSSCoreState
from strategies.swing.atrss.engine import ATRSSEngine
from strategies.swing.atrss.models import (
    BreakoutArmState,
    Candidate,
    CandidateType,
    Direction as ATRSSDirection,
    HaltState,
)
UTC = timezone.utc


def test_atrss_core_serializer_roundtrip_preserves_typed_state() -> None:
    state = ATRSSCoreState(
        pending_reverses=[
            Candidate(
                symbol="QQQ",
                type=CandidateType.REVERSE,
                direction=ATRSSDirection.LONG,
                trigger_price=512.5,
                time=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
            )
        ],
        halt_states={
            "QQQ": HaltState(
                is_halted=True,
                halt_detected_at=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
                queued_stop_updates=[("STOP-1", 505.0)],
            )
        },
        breakout_arm_states={
            "QQQ": BreakoutArmState(
                breakout_armed_dir=ATRSSDirection.SHORT,
                breakout_arm_low=503.25,
            )
        },
        last_decision_code="ATRSS",
    )

    restored = restore_atrss_state(snapshot_atrss_state(state))

    assert isinstance(restored.pending_reverses[0], Candidate)
    assert restored.pending_reverses[0].type is CandidateType.REVERSE
    assert restored.pending_reverses[0].direction is ATRSSDirection.LONG
    assert isinstance(restored.halt_states["QQQ"], HaltState)
    assert restored.halt_states["QQQ"].queued_stop_updates == [("STOP-1", 505.0)]
    assert isinstance(restored.breakout_arm_states["QQQ"], BreakoutArmState)
    assert restored.breakout_arm_states["QQQ"].breakout_armed_dir is ATRSSDirection.SHORT


def test_akc_helix_core_serializer_roundtrip_preserves_nested_runtime_models() -> None:
    state = AKCHelixCoreState(
        tf_states={"CL": {"1H": AKCHelixTFState(tf_label="1H", atr=3.2)}},
        pivots={"CL": {"1H": PivotStore(max_size=25)}},
        circuit_breakers={"CL": AKCHelixCircuitBreakerState(consecutive_stops=2)},
        regime_4h={"CL": Regime.BULL},
        prev_regimes={"CL": Regime.CHOP},
        last_decision_code="HELIX",
    )

    restored = restore_akc_helix_state(snapshot_akc_helix_state(state))

    assert isinstance(restored.tf_states["CL"]["1H"], AKCHelixTFState)
    assert restored.tf_states["CL"]["1H"].atr == pytest.approx(3.2)
    assert isinstance(restored.pivots["CL"]["1H"], PivotStore)
    assert restored.pivots["CL"]["1H"].max_size == 25
    assert isinstance(restored.circuit_breakers["CL"], AKCHelixCircuitBreakerState)
    assert restored.regime_4h["CL"] is Regime.BULL
    assert restored.prev_regimes["CL"] is Regime.CHOP


def test_vdub_core_serializer_roundtrip_preserves_typed_runtime_state() -> None:
    state = VdubCoreState(
        regime=RegimeState(daily_trend=1, vol_state=VolState.HIGH),
        counters=DayCounters(long_fills=2, breaker_hit=True),
        working_entries={
            "ENTRY-1": WorkingEntry(
                oms_order_id="ENTRY-1",
                direction=VdubDirection.LONG,
                qty=2,
                filter_decisions=[{"gate": "trend", "passed": True}],
            )
        },
        event_state=EventBlockState(
            blocked=True,
            block_end_ts=datetime(2026, 4, 25, 13, 15, tzinfo=UTC),
            cooldown_remaining=3,
        ),
        recent_wins=[True, False],
        last_decision_code="VDUB",
    )

    restored = restore_vdub_state(snapshot_vdub_state(state))

    assert isinstance(restored.regime, RegimeState)
    assert restored.regime.vol_state is VolState.HIGH
    assert isinstance(restored.counters, DayCounters)
    assert restored.counters.long_fills == 2
    assert isinstance(restored.working_entries["ENTRY-1"], WorkingEntry)
    assert restored.working_entries["ENTRY-1"].direction is VdubDirection.LONG
    assert restored.event_state.blocked is True
    assert restored.recent_wins == [True, False]




@pytest.mark.asyncio
async def test_atrss_engine_snapshot_and_hydrate_preserve_runtime_types() -> None:
    engine = ATRSSEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.breakout_arm_states["QQQ"] = BreakoutArmState(breakout_arm_high=512.0)
    engine.halt_states["QQQ"] = HaltState(
        is_halted=True,
        halt_detected_at=datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
    )
    engine._last_decision_code = "HALT_GUARDED"

    restored = ATRSSEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    await restored.hydrate(engine.snapshot_state())

    assert isinstance(restored.breakout_arm_states["QQQ"], BreakoutArmState)
    assert restored.breakout_arm_states["QQQ"].breakout_arm_high == pytest.approx(512.0)
    assert isinstance(restored.halt_states["QQQ"], HaltState)
    assert restored.halt_states["QQQ"].is_halted is True
    assert restored.health_status()["last_decision_code"] == "HALT_GUARDED"


@pytest.mark.asyncio
async def test_akc_helix_engine_snapshot_and_hydrate_preserve_tf_state() -> None:
    engine = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    engine.tf_states["CL"] = {"1H": AKCHelixTFState(tf_label="1H", atr=3.2)}
    engine.circuit_breakers["CL"] = AKCHelixCircuitBreakerState(consecutive_stops=2)
    engine._last_decision_code = "SETUP_QUEUED"

    restored = HelixEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments={},
        config={},
    )
    await restored.hydrate(engine.snapshot_state())

    assert isinstance(restored.tf_states["CL"]["1H"], AKCHelixTFState)
    assert restored.tf_states["CL"]["1H"].atr == pytest.approx(3.2)
    assert isinstance(restored.circuit_breakers["CL"], AKCHelixCircuitBreakerState)
    assert restored.circuit_breakers["CL"].consecutive_stops == 2
    assert restored.health_status()["last_decision_code"] == "SETUP_QUEUED"


@pytest.mark.asyncio
async def test_vdub_engine_snapshot_and_hydrate_preserve_regime_state() -> None:
    instrument = SimpleNamespace(symbol="NQ")
    engine = VdubNQv4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=[instrument],
    )
    engine.regime.daily_trend = 1
    engine.counters.long_fills = 2
    engine._recent_wins = [True, False]
    engine._last_decision_code = "ENTRY_ARMED"

    restored = VdubNQv4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(stream_events=lambda *_args, **_kwargs: None),
        instruments=[instrument],
    )
    await restored.hydrate(engine.snapshot_state())

    assert type(restored.regime) is type(engine.regime)
    assert restored.regime.daily_trend == 1
    assert type(restored.counters) is type(engine.counters)
    assert restored.counters.long_fills == 2
    assert restored._recent_wins == [True, False]
    assert restored.health_status()["last_decision_code"] == "ENTRY_ARMED"



def test_iaric_core_serializer_roundtrip_preserves_intraday_snapshot_shape() -> None:
    snapshot = IntradayStateSnapshot(
        trade_date=date(2026, 4, 25),
        saved_at=datetime(2026, 4, 25, 15, 30, tzinfo=UTC),
        symbols=[PBSymbolState(symbol="MSFT", stage="SCANNING")],
        last_decision_code="SNAPSHOT",
        meta={"active_symbols": ["MSFT"]},
    )

    restored = restore_iaric_state(snapshot_iaric_state(snapshot))

    assert isinstance(restored, IntradayStateSnapshot)
    assert isinstance(restored.symbols[0], PBSymbolState)
    assert restored.symbols[0].symbol == "MSFT"
    assert restored.meta["active_symbols"] == ["MSFT"]




@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_vdub_live_wrapper_entry_fill_matches_replay_core_state(monkeypatch) -> None:
    instrument = SimpleNamespace(symbol="NQ", point_value=vdub_config.NQ_SPEC["point_value"])

    async def _fake_submit_intent(*_args, **_kwargs):
        return SimpleNamespace(oms_order_id=None)

    engine = VdubNQv4Engine(
        ib_session=object(),
        oms_service=SimpleNamespace(
            stream_events=lambda *_args, **_kwargs: None,
            submit_intent=_fake_submit_intent,
        ),
        instruments=[instrument],
    )
    working_entry = WorkingEntry(
        oms_order_id="OMS-V1",
        entry_type=VdubEntryType.TYPE_A,
        direction=VdubDirection.LONG,
        stop_entry=20010.0,
        qty=2,
        initial_stop=19980.0,
        is_addon=True,
    )
    engine.working_entries["OMS-V1"] = working_entry

    initial_state = restore_vdub_state(snapshot_vdub_state(build_vdub_runtime_state(engine)))

    await engine._on_fill("OMS-V1", {"price": 20010.0, "qty": 2, "commission": 1.25})

    wrapper_snapshot = snapshot_vdub_state(build_vdub_runtime_state(engine))
    fill_time = engine.positions[0].entry_time
    replay = run_replay(
        initial_state,
        steps=[
            ReplayStep(
                fills=[
                    VdubFill(
                        oms_order_id="OMS-V1",
                        fill_price=20010.0,
                        fill_qty=2,
                        fill_time=fill_time,
                        point_value=vdub_config.NQ_SPEC["point_value"],
                        commission=1.25,
                        entry_context=VdubEntryFillContext(working_entry=working_entry),
                    )
                ]
            )
        ],
        on_bar=lambda state, payload: on_vdub_bar(state, **payload),
        on_order_update=on_vdub_order_update,
        on_fill=on_vdub_fill,
    )

    assert replay.events[-1].code == engine.health_status()["last_decision_code"] == "ENTRY_FILLED"
    assert snapshot_vdub_state(replay.state) == wrapper_snapshot


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_vdub_pyramid_entry_submission_matches_backtest_order_risk_and_counters() -> None:
    class _CaptureOMS:
        def __init__(self) -> None:
            self.intents = []

        async def submit_intent(self, intent):
            self.intents.append(intent)
            return SimpleNamespace(oms_order_id="LIVE-ADD-1")

        def stream_events(self, *_args, **_kwargs):
            return None

    oms = _CaptureOMS()
    instrument = SimpleNamespace(
        symbol="NQ",
        point_value=vdub_config.NQ_SPEC["point_value"],
        tick_size=vdub_config.NQ_SPEC["tick"],
    )
    live = VdubNQv4Engine(
        ib_session=object(),
        oms_service=oms,
        instruments=[instrument],
    )
    backtest = BacktestVdubusEngine(
        "NQ",
        VdubusBacktestConfig(symbols=["NQ"], track_shadows=False),
    )
    live._bar_idx = 12
    backtest._bar_idx = 12

    kwargs = dict(
        direction=VdubDirection.LONG,
        qty=3,
        stop_entry=20010.0,
        limit_entry=20011.0,
        initial_stop=19990.0,
        signal_type=VdubEntryType.TYPE_A,
        is_flip=False,
        is_pyramid=True,
        class_mult=1.2,
        vwap_used=20000.0,
        session=VdubSessionWindow.RTH,
    )

    await live._submit_entry(filter_decisions=[], **kwargs)
    backtest._submit_entry(
        sub_window=VdubSubWindow.CORE,
        bar_time=datetime(2026, 5, 20, 14, 0, tzinfo=UTC),
        **kwargs,
    )

    live_order = oms.intents[0].order
    backtest_order = backtest.broker.pending_orders[0]
    live_working = live.working_entries["LIVE-ADD-1"]
    backtest_working = next(iter(backtest._working.values()))
    expected_risk = abs(kwargs["stop_entry"] - kwargs["initial_stop"]) * vdub_config.NQ_SPEC["point_value"] * kwargs["qty"]

    assert live_order.qty == backtest_order.qty == kwargs["qty"]
    assert live_order.stop_price == backtest_order.stop_price == kwargs["stop_entry"]
    assert live_order.limit_price == backtest_order.limit_price == kwargs["limit_entry"]
    assert live_order.risk_context.risk_dollars == pytest.approx(expected_risk)
    assert live_working.is_addon is backtest_working.is_addon is True
    assert live_working.qty == backtest_working.qty == kwargs["qty"]
    assert live_working.initial_stop == backtest_working.initial_stop == kwargs["initial_stop"]
    assert live.counters.addon_used_long is True
    assert backtest.counters.addon_used_long is True


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_vdub_pyramid_signal_path_matches_backtest_addon_intent(monkeypatch) -> None:
    class _CaptureOMS:
        def __init__(self) -> None:
            self.intents = []

        async def submit_intent(self, intent):
            self.intents.append(intent)
            return SimpleNamespace(oms_order_id=f"LIVE-ADD-{len(self.intents)}")

        def stream_events(self, *_args, **_kwargs):
            return None

    def _signal(*_args, **_kwargs):
        return {"type": "A", "vwap_used": 100.0}

    monkeypatch.setattr(vdub_live_engine.reg, "direction_allowed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(vdub_backtest_engine.reg, "direction_allowed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(vdub_live_engine.sig, "slope_ok", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(vdub_backtest_engine.sig, "slope_ok", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(vdub_live_engine.sig, "type_a_check", _signal)
    monkeypatch.setattr(vdub_backtest_engine.sig, "type_a_check", _signal)
    monkeypatch.setattr(vdub_live_engine.sig, "predator_present", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(vdub_backtest_engine.sig, "predator_present", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(vdub_live_engine.risk, "compute_qty", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(vdub_live_engine.risk, "pass_viability", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(vdub_backtest_engine.risk, "pass_viability", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(vdub_live_engine.risk, "pass_risk_gates", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(vdub_backtest_engine.risk, "pass_risk_gates", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(vdub_config, "HOURLY_ALIGNED_MULT", 1.0)
    monkeypatch.setattr(vdub_config, "USE_MICRO_TRIGGER", False)

    bars_15m = _vdub_bars("2026-05-20T14:00:00", minutes=15)
    bars_5m = _vdub_bars("2026-05-20T14:00:00", minutes=5)
    hourly = _vdub_bars("2026-05-20T14:00:00", minutes=60)
    ts = datetime(2026, 5, 20, 14, 30, tzinfo=UTC)

    live = VdubNQv4Engine(
        ib_session=object(),
        oms_service=_CaptureOMS(),
        instruments=[SimpleNamespace(symbol="NQ", point_value=vdub_config.NQ_SPEC["point_value"], tick_size=vdub_config.NQ_SPEC["tick"])],
    )
    backtest = BacktestVdubusEngine(
        "NQ",
        VdubusBacktestConfig(symbols=["NQ"], fixed_qty=3, track_shadows=False),
    )
    for engine in (live, backtest):
        engine.regime.daily_trend = 1
        engine.regime.trend_1h = 1
        engine.regime.choppiness = 10.0
        engine._bar_idx = 2
        engine._mom15 = np.array([1.0, 1.0, 1.0])
        engine._atr15 = np.array([1.0, 1.0, 1.0])
        engine._atr1h = np.array([2.0])
        engine._svwap = np.array([100.0, 100.0, 100.0])
        engine._vwap_a_arr = np.array([100.0, 100.0, 100.0])

    live._c15 = bars_15m.closes
    live._h15 = bars_15m.highs
    live._l15 = bars_15m.lows
    live.positions = [_eligible_vdub_addon_position()]
    backtest._active = _ActivePosition(
        pos=_eligible_vdub_addon_position(),
        record=VdubusTradeRecord(symbol="NQ", direction=int(VdubDirection.LONG)),
    )

    await live._evaluate_direction(VdubDirection.LONG, VdubSessionWindow.RTH, VdubSubWindow.CORE, ts)
    backtest._evaluate_direction(
        VdubDirection.LONG,
        VdubSessionWindow.RTH,
        VdubSubWindow.CORE,
        ts,
        bars_15m,
        hourly,
        h_idx=0,
        t=2,
        bars_5m=bars_5m,
        five_to_15_idx_map=np.array([2, 2, 2]),
    )

    live_order = live._oms.intents[0].order
    backtest_order = backtest.broker.pending_orders[0]
    live_working = next(iter(live.working_entries.values()))
    backtest_working = next(iter(backtest._working.values()))

    assert len(live._oms.intents) == len(backtest.broker.pending_orders) == 1
    assert live_working.is_addon is backtest_working.is_addon is True
    assert live_order.qty == backtest_order.qty == 3
    assert live_order.stop_price == pytest.approx(backtest_order.stop_price)
    assert live_order.limit_price == pytest.approx(backtest_order.limit_price)
    assert live_working.initial_stop == pytest.approx(backtest_working.initial_stop)
    assert live.counters.addon_used_long is True
    assert backtest.counters.addon_used_long is True


def _eligible_vdub_addon_position() -> PositionState:
    return PositionState(
        trade_id="base-long",
        direction=VdubDirection.LONG,
        entry_price=100.0,
        stop_price=99.0,
        qty_entry=1,
        qty_open=1,
        r_points=1.0,
        stage=PositionStage.ACTIVE_FREE,
        highest_since_entry=101.5,
        lowest_since_entry=100.0,
        entry_session=VdubSessionWindow.RTH,
    )


def _vdub_bars(start: str, *, minutes: int) -> NumpyBars:
    times = np.array(
        [np.datetime64(start) + np.timedelta64(minutes * offset, "m") for offset in range(3)]
    )
    return NumpyBars(
        opens=np.array([100.0, 100.5, 101.0]),
        highs=np.array([100.5, 101.0, 101.75]),
        lows=np.array([99.5, 100.0, 100.75]),
        closes=np.array([100.25, 100.75, 101.5]),
        volumes=np.array([1_000.0, 1_100.0, 1_200.0]),
        times=times,
    )
