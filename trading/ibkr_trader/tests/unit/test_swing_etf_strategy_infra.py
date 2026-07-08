from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from backtests.swing.config_etf_base import ETFSlippageConfig
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.auto.etf_common import ETFMetrics, composite_score
from backtests.swing.auto.tpc.plugin import TPC_SCORING_WEIGHTS
from backtests.swing.data.multitimeframe import align_15m_to_1h
from backtests.swing.engine.backtest_engine import SymbolResult, TradeRecord
from backtests.swing.engine.etf_engine_base import ETFStrategyBacktestEngine, _OpenTrade
from backtests.swing.engine.sim_broker import FillResult, FillStatus, OrderSide, OrderType, SimBroker, SimOrder
from backtests.shared.parity.legacy_result_outputs import trade_outcomes_from_records
from backtests.swing.auto.tpc.phase_candidates import get_phase_candidates as tpc_candidates
from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitAddOnEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.swing._shared.etf_core import (
    BarData,
    BarWindow,
    ETFCoreState,
    ETFBarInput,
    ETFFill,
    ETFOrderUpdate,
    ETFPosition,
    SetupSnapshot,
    default_manage_position,
    on_fill_common,
    on_order_update_common,
)
from strategies.swing._shared.models import Direction
from strategies.swing.tpc.core import serializers as tpc_serializers
from strategies.swing.tpc.core.logic import _manage_tpc_position, on_fill as tpc_on_fill
from strategies.swing.tpc.core.state import TPCBarInput, TPCCoreState, TPCFill, TPCSecondEntrySeed
from strategies.swing.tpc.config import SYMBOL_CONFIGS as TPC_SYMBOL_CONFIGS
from strategies.swing.tpc.gates import session_filter as tpc_session_filter
from strategies.swing.tpc.models import PullbackType


class _NoopETFCoreLogic:
    @staticmethod
    def on_fill(state, fill):
        return state, [], []

    @staticmethod
    def on_order_update(state, update):
        return state, [], []


def test_tpc_gld_primary_session_not_blocked_by_optimized_avoid_window() -> None:
    ts = datetime(2025, 3, 3, 8, 15, tzinfo=ZoneInfo("America/New_York"))
    assert TPC_SYMBOL_CONFIGS["GLD"].avoid_windows_et == ((11, 0, 12, 0),)
    assert tpc_session_filter(ts, TPC_SYMBOL_CONFIGS["GLD"])


def test_tpc_serializer_restores_empty_live_state() -> None:
    state = tpc_serializers.restore_state(None)

    assert isinstance(state, TPCCoreState)
    assert state.positions == {}
    assert state.second_entry_seeds == {}


def test_tpc_serializer_roundtrips_second_entry_seed() -> None:
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
    state = TPCCoreState(
        second_entry_seeds={
            "QQQ": TPCSecondEntrySeed(
                symbol="QQQ",
                source_setup_id="TPC-QQQ-source",
                direction=int(Direction.LONG),
                pullback_low=97.25,
                pullback_high=103.50,
                stop_time=ts,
                source_grade="a_plus",
                source_score=18.0,
            )
        }
    )

    restored = tpc_serializers.restore_state(tpc_serializers.snapshot_state(state))

    seed = restored.second_entry_seeds["QQQ"]
    assert seed.source_setup_id == "TPC-QQQ-source"
    assert seed.stop_time == ts
    assert seed.source_score == 18.0


def test_15m_alignment_uses_completed_start_stamped_1h_bar() -> None:
    idx15 = pd.DatetimeIndex(
        [
            "2025-01-02 09:30:00+00:00",
            "2025-01-02 09:45:00+00:00",
            "2025-01-02 10:00:00+00:00",
        ]
    )
    idx1h = pd.DatetimeIndex(["2025-01-02 09:00:00+00:00"])
    df15 = pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1]}, index=idx15)
    df1h = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]}, index=idx1h)

    aligned = align_15m_to_1h(df15, df1h)

    assert aligned.tolist() == [-1, 0, 0]


def test_etf_slippage_duck_types_equity_commission_for_sim_broker() -> None:
    cfg = ETFSlippageConfig(commission_per_share=0.005)
    assert cfg.commission_per_contract == 0.005


def test_tpc_all_symbol_overrides_patch_real_strategy_fields() -> None:
    cfg = TPCBacktestConfig().with_overrides({"all.max_extension_atr_mult": 1.75})
    assert cfg.symbol_configs["QQQ"].max_extension_atr_mult == 1.75
    assert cfg.symbol_configs["GLD"].max_extension_atr_mult == 1.75


def test_tpc_overrides_apply_warmup_and_symbol_fields_together() -> None:
    cfg = TPCBacktestConfig().with_overrides({"warmup_15m": 1200, "all.max_extension_atr_mult": 1.85})

    assert cfg.warmup_15m == 1200
    assert cfg.symbol_configs["QQQ"].max_extension_atr_mult == 1.85
    assert cfg.symbol_configs["GLD"].max_extension_atr_mult == 1.85


def test_tpc_second_entry_fields_are_overridable() -> None:
    cfg = TPCBacktestConfig().with_overrides({
        "all.type_c_mode": "real_reentry",
        "all.second_entry_score_min": 15,
        "all.second_entry_min_source_score": 16.0,
        "all.second_entry_requires_source_a_plus": True,
        "all.min_stop_atr_mult": 0.08,
        "all.confirmation_max_count": 3,
    })

    assert cfg.symbol_configs["QQQ"].type_c_mode == "real_reentry"
    assert cfg.symbol_configs["GLD"].second_entry_score_min == 15
    assert cfg.symbol_configs["GLD"].second_entry_min_source_score == 16.0
    assert cfg.symbol_configs["QQQ"].second_entry_requires_source_a_plus is True
    assert cfg.symbol_configs["QQQ"].min_stop_atr_mult == 0.08
    assert cfg.symbol_configs["GLD"].confirmation_max_count == 3


def test_tpc_structural_alpha_fields_are_overridable() -> None:
    cfg = TPCBacktestConfig().with_overrides({
        "all.min_adx_4h": 18.0,
        "all.require_di_alignment": True,
        "all.min_ma50_slope_atr_4h": 0.04,
        "all.addon_enabled": True,
        "all.addon_trigger_r": 1.5,
        "all.addon_max_total_risk_pct": 0.03,
        "all.mfe_giveback_trigger_r": 3.0,
        "all.mfe_giveback_retain_frac": 0.55,
    })

    assert cfg.symbol_configs["QQQ"].min_adx_4h == 18.0
    assert cfg.symbol_configs["GLD"].require_di_alignment is True
    assert cfg.symbol_configs["QQQ"].min_ma50_slope_atr_4h == 0.04
    assert cfg.symbol_configs["GLD"].addon_enabled is True
    assert cfg.symbol_configs["QQQ"].addon_trigger_r == 1.5
    assert cfg.symbol_configs["GLD"].addon_max_total_risk_pct == 0.03
    assert cfg.symbol_configs["QQQ"].mfe_giveback_trigger_r == 3.0
    assert cfg.symbol_configs["GLD"].mfe_giveback_retain_frac == 0.55


def test_shared_core_entry_fill_places_stop_and_vwap_profit_target() -> None:
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
    setup = SetupSnapshot(
        setup_id="TPC-QQQ-1",
        strategy_id="TPC",
        symbol="QQQ",
        direction=Direction.LONG,
        grade="a_plus",
        setup_type="vwap_reclaim",
        entry_model="vwap_reclaim",
        state="entry_ready",
        created_ts=ts,
        entry_price=100.0,
        stop_price=98.0,
        qty=10,
        score=8.5,
        risk_pct=0.006,
        t1_r=2.0,
        t1_partial_pct=0.40,
        t2_r=3.0,
        t2_partial_pct=0.35,
        target_price=105.0,
    )
    state = ETFCoreState(pending_orders={"TPC-QQQ-1": setup}, setups={"TPC-QQQ-1": setup})

    next_state, actions, events = on_fill_common(
        state,
        ETFFill(oms_order_id="TPC-QQQ-1", fill_price=100.0, fill_qty=10, symbol="QQQ", fill_time=ts),
        strategy_id="TPC",
    )

    assert "QQQ" in next_state.positions
    assert any(isinstance(action, SubmitProtectiveStop) for action in actions)
    target = next(action for action in actions if isinstance(action, SubmitProfitTarget))
    assert target.limit_price == 105.0
    assert target.qty == 4
    assert events[-1].code == "ENTRY_FILLED"


def test_shared_core_entry_fill_can_place_full_t1_exit_target() -> None:
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
    setup = SetupSnapshot(
        setup_id="TPC-QQQ-full-t1",
        strategy_id="TPC",
        symbol="QQQ",
        direction=Direction.LONG,
        grade="A+",
        setup_type="failed_breakdown",
        entry_model="failed_breakdown",
        state="entry_ready",
        created_ts=ts,
        entry_price=100.0,
        stop_price=98.0,
        qty=10,
        score=8.5,
        risk_pct=0.006,
        t1_r=1.5,
        t1_partial_pct=1.0,
        t2_r=3.0,
        t2_partial_pct=0.0,
        target_price=103.0,
        meta={"exit_all_at_t1": True},
    )
    state = ETFCoreState(pending_orders={setup.setup_id: setup}, setups={setup.setup_id: setup})

    _next_state, actions, _events = on_fill_common(
        state,
        ETFFill(oms_order_id=setup.setup_id, fill_price=100.0, fill_qty=10, symbol="QQQ", fill_time=ts),
        strategy_id="TPC",
    )

    target = next(action for action in actions if isinstance(action, SubmitProfitTarget))
    assert target.qty == 10
    assert target.limit_price == 103.0


def test_shared_core_addon_fill_resizes_position_and_stop() -> None:
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-addon",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=6,
        qty_initial=10,
        entry_price=100.0,
        current_stop=101.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="classic_38_62",
        grade="valid",
        entry_model="confirmation_close",
        score=18.0,
        stop_order_id="stop-1",
        meta={"addon_pending": True},
    )
    state = ETFCoreState(positions={"QQQ": position})

    next_state, actions, events = on_fill_common(
        state,
        ETFFill(
            oms_order_id="TPC-QQQ-addon-8",
            fill_price=104.25,
            fill_qty=2,
            symbol="QQQ",
            fill_time=ts,
            order_role="add_on_entry",
        ),
        strategy_id="TPC",
    )

    resized = next(action for action in actions if isinstance(action, ReplaceProtectiveStop))
    assert next_state.positions["QQQ"].qty_open == 8
    assert next_state.positions["QQQ"].qty_initial == 12
    assert next_state.positions["QQQ"].meta["addon_done"] is True
    assert next_state.positions["QQQ"].meta["addon_pending"] is False
    assert resized.qty == 8
    assert resized.reason == "addon_resize"
    assert events[-1].code == "ADDON_FILLED"


def test_shared_core_runner_requires_vwap_acceptance_uses_flatten_action() -> None:
    ts = datetime(2025, 3, 3, 14, 45, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-runner",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=3,
        qty_initial=10,
        entry_price=100.0,
        current_stop=100.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="wick_rejection",
        grade="A+",
        entry_model="wick_rejection",
        score=9.0,
        stop_order_id="stop-1",
        t1_done=True,
        bars_held_15m=8,
        meta={
            "target_price": 103.0,
            "runner_requires_vwap_acceptance": True,
            "runner_acceptance_grace_bars": 1,
            "runner_acceptance_close_atr15": 0.05,
            "t1_done_bars_15m": 6,
        },
    )
    bar_input = ETFBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=102.8, high=103.0, low=102.5, close=102.9),
        indicators={"atr_15m": 1.0},
    )

    actions = default_manage_position(ETFCoreState(), bar_input, object(), position)

    flatten = next(action for action in actions if isinstance(action, FlattenPosition))
    assert flatten.reason == "VWAP_ACCEPTANCE_FAIL"
    assert flatten.qty == 3


def test_shared_core_mfe_giveback_uses_flatten_action() -> None:
    ts = datetime(2025, 3, 3, 14, 45, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-giveback",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=5,
        qty_initial=10,
        entry_price=100.0,
        current_stop=98.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="failed_breakdown",
        grade="A+",
        entry_model="failed_breakdown",
        score=9.0,
        stop_order_id="stop-1",
        mfe_price=104.0,
        bars_held_15m=6,
        meta={
            "mfe_giveback_trigger_r": 1.0,
            "mfe_giveback_retain_frac": 0.5,
            "mfe_giveback_lock_r": 0.0,
        },
    )
    bar_input = ETFBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=102.0, high=102.2, low=100.8, close=101.0),
    )

    actions = default_manage_position(ETFCoreState(), bar_input, object(), position)

    flatten = next(action for action in actions if isinstance(action, FlattenPosition))
    assert flatten.reason == "MFE_GIVEBACK"
    assert flatten.qty == 5


def test_shared_core_early_failure_exit_uses_flatten_action() -> None:
    ts = datetime(2025, 3, 3, 15, 15, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-early-failure",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=5,
        qty_initial=10,
        entry_price=100.0,
        current_stop=98.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="wick_rejection",
        grade="A+",
        entry_model="wick_rejection",
        score=8.5,
        stop_order_id="stop-1",
        mfe_price=100.35,
        bars_held_15m=6,
        meta={
            "early_failure_exit_bars_15m": 6,
            "early_failure_max_mfe_r": 0.25,
            "early_failure_max_current_r": -0.30,
        },
    )
    bar_input = ETFBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=99.6, high=99.8, low=99.0, close=99.2),
    )

    actions = default_manage_position(ETFCoreState(), bar_input, object(), position)

    flatten = next(action for action in actions if isinstance(action, FlattenPosition))
    assert flatten.reason == "EARLY_FAILURE"
    assert round(flatten.metadata["current_r"], 2) == -0.40


def test_shared_core_structure_trail_after_t1_replaces_stop() -> None:
    ts = datetime(2025, 3, 3, 15, 30, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-trail",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=5,
        qty_initial=10,
        entry_price=100.0,
        current_stop=100.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="failed_breakdown",
        grade="A+",
        entry_model="failed_breakdown",
        score=9.0,
        stop_order_id="stop-1",
        t1_done=True,
        bars_held_15m=8,
        meta={
            "structure_trail_after_t1_30m_bars": 3,
            "structure_trail_use_vwap_after_t1": True,
            "structure_trail_min_lock_r": 0.25,
            "t2_r": 3.0,
        },
    )
    bars_30m = BarWindow(
        opens=np.array([101.0, 101.5, 102.0]),
        highs=np.array([102.0, 102.5, 103.0]),
        lows=np.array([101.0, 101.5, 102.0]),
        closes=np.array([101.8, 102.2, 102.8]),
        volumes=np.ones(3),
        times=(ts, ts, ts),
    )
    bar_input = ETFBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=102.6, high=103.1, low=102.4, close=103.0),
        bars_30m=bars_30m,
        indicators={"vwap_30m": 101.2},
    )

    actions = default_manage_position(ETFCoreState(), bar_input, object(), position)

    trail = next(action for action in actions if isinstance(action, ReplaceProtectiveStop))
    assert trail.reason == "structure_trail"
    assert trail.stop_price == 101.2
    assert trail.qty == 5


def test_shared_core_continuation_addon_requires_acceptance_and_caps_risk() -> None:
    ts = datetime(2025, 3, 3, 15, 45, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-continuation-addon",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=5,
        qty_initial=10,
        entry_price=100.0,
        current_stop=101.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="failed_breakdown",
        grade="A+",
        entry_model="failed_breakdown",
        score=9.0,
        stop_order_id="stop-1",
        t1_done=True,
        mfe_price=104.0,
        bars_held_15m=9,
        meta={
            "target_price": 102.0,
            "t2_r": 3.0,
            "continuation_addon_enabled": True,
            "continuation_addon_trigger_r": 1.40,
            "continuation_addon_size_mult": 0.25,
            "continuation_addon_min_score": 8.5,
            "continuation_addon_requires_t1": True,
            "continuation_addon_require_vwap_acceptance": True,
            "continuation_addon_acceptance_atr15": 0.03,
            "continuation_addon_require_ema20_15m": True,
            "continuation_addon_max_total_risk_pct": 0.012,
            "continuation_addon_max_notional_pct": 0.34,
        },
    )
    bar_input = ETFBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=103.0, high=103.5, low=102.8, close=103.2),
        indicators={"atr_15m": 1.0, "ema20_15m": 102.5},
        equity=100_000.0,
    )

    actions = default_manage_position(ETFCoreState(), bar_input, object(), position)

    addon = next(action for action in actions if isinstance(action, SubmitAddOnEntry))
    assert addon.qty == 2
    assert addon.side == "BUY"
    assert addon.risk_context["stop_for_risk"] == 101.0
    assert position.meta["addon_pending"] is True


def test_shared_core_addon_terminal_update_clears_pending_flag() -> None:
    ts = datetime(2025, 3, 3, 16, 0, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-addon-expired",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=5,
        qty_initial=10,
        entry_price=100.0,
        current_stop=101.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="failed_breakdown",
        grade="A+",
        entry_model="failed_breakdown",
        score=9.0,
        stop_order_id="stop-1",
        meta={"addon_pending": True},
    )
    state = ETFCoreState(positions={"QQQ": position})

    next_state, _actions, events = on_order_update_common(
        state,
        ETFOrderUpdate(
            oms_order_id="TPC-QQQ-addon-expired-addon-9",
            status="expired",
            symbol="QQQ",
            timestamp=ts,
            order_role="add_on_entry",
        ),
        strategy_id="TPC",
    )

    assert next_state.positions["QQQ"].meta["addon_pending"] is False
    assert events[-1].code == "ADDON_ORDER_TERMINAL"


def test_tpc_manage_position_emits_proof_based_addon() -> None:
    ts = datetime(2025, 3, 3, 15, 0, tzinfo=ZoneInfo("UTC"))
    cfg = TPCBacktestConfig().with_overrides({
        "QQQ.addon_enabled": True,
        "QQQ.addon_trigger_r": 1.5,
        "QQQ.addon_size_mult": 0.25,
        "QQQ.addon_min_score": 17,
        "QQQ.addon_max_total_risk_pct": 0.03,
    }).symbol_configs["QQQ"]
    position = ETFPosition(
        setup_id="TPC-QQQ-proof-addon",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=6,
        qty_initial=10,
        entry_price=100.0,
        current_stop=101.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="classic_38_62",
        grade="a_plus",
        entry_model="confirmation_close",
        score=18.0,
        stop_order_id="stop-1",
        t1_done=True,
        mfe_price=104.5,
    )
    bars = BarWindow(
        opens=np.array([101.0, 102.0, 102.5, 103.0, 103.4]),
        highs=np.array([101.5, 102.3, 103.0, 103.6, 104.2]),
        lows=np.array([100.8, 101.6, 102.0, 102.8, 103.2]),
        closes=np.array([101.2, 102.1, 102.7, 103.4, 104.1]),
        volumes=np.ones(5),
        times=(ts, ts, ts, ts, ts),
    )
    bar_input = TPCBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=103.4, high=104.2, low=103.2, close=104.1),
        bars_15m=bars,
        indicators={"vwap_15m": 103.0, "ema20_15m": 102.5},
        equity=100_000.0,
    )

    actions = _manage_tpc_position(TPCCoreState(), bar_input, cfg, position)

    addon = next(action for action in actions if isinstance(action, SubmitAddOnEntry))
    assert addon.qty == 2
    assert addon.side == "BUY"
    assert addon.risk_context["stop_for_risk"] == 101.0
    assert position.meta["addon_pending"] is True


def test_tpc_manage_position_uses_mfe_giveback_flatten() -> None:
    ts = datetime(2025, 3, 3, 15, 0, tzinfo=ZoneInfo("UTC"))
    cfg = TPCBacktestConfig().with_overrides({
        "QQQ.mfe_giveback_trigger_r": 3.0,
        "QQQ.mfe_giveback_retain_frac": 0.50,
        "QQQ.mfe_giveback_lock_r": 1.25,
    }).symbol_configs["QQQ"]
    position = ETFPosition(
        setup_id="TPC-QQQ-giveback",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=5,
        qty_initial=10,
        entry_price=100.0,
        current_stop=101.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type="classic_38_62",
        grade="a_plus",
        entry_model="confirmation_close",
        score=18.0,
        stop_order_id="stop-1",
        t1_done=True,
        mfe_price=108.0,
    )
    bar_input = TPCBarInput(
        symbol="QQQ",
        bar_15m=BarData(timestamp=ts, open=104.0, high=104.2, low=103.2, close=103.5),
        equity=100_000.0,
    )

    actions = _manage_tpc_position(TPCCoreState(), bar_input, cfg, position)

    flatten = next(action for action in actions if isinstance(action, FlattenPosition))
    assert flatten.reason == "MFE_GIVEBACK"
    assert flatten.qty == 5


def test_shared_core_terminal_order_update_clears_pending_entry() -> None:
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
    setup = SetupSnapshot(
        setup_id="TPC-QQQ-1",
        strategy_id="TPC",
        symbol="QQQ",
        direction=Direction.LONG,
        grade="valid",
        setup_type="classic_38_62",
        entry_model="confirmation_close",
        state="entry_ready",
        created_ts=ts,
        entry_price=100.0,
        stop_price=98.0,
        qty=10,
        score=14.0,
        risk_pct=0.006,
        t1_r=1.5,
        t1_partial_pct=0.45,
        t2_r=2.75,
        t2_partial_pct=0.275,
    )
    state = ETFCoreState(pending_orders={"TPC-QQQ-1": setup})

    next_state, _actions, events = on_order_update_common(
        state,
        ETFOrderUpdate(oms_order_id="TPC-QQQ-1", status="expired", symbol="QQQ", timestamp=ts),
        strategy_id="TPC",
    )

    assert "TPC-QQQ-1" not in next_state.pending_orders
    assert events[-1].code == "ORDER_TERMINAL"


def test_tpc_stop_before_t1_records_second_entry_seed() -> None:
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
    position = ETFPosition(
        setup_id="TPC-QQQ-1",
        symbol="QQQ",
        direction=Direction.LONG,
        qty_open=10,
        qty_initial=10,
        entry_price=100.0,
        current_stop=98.0,
        initial_stop=98.0,
        entry_ts=ts,
        risk_per_share=2.0,
        setup_type=PullbackType.TYPE_A.value,
        grade="valid",
        entry_model="structure_stop",
        score=15.0,
        meta={"pullback_low": 97.75, "pullback_high": 103.25},
    )
    state = TPCCoreState(positions={"QQQ": position})

    next_state, _actions, events = tpc_on_fill(
        state,
        TPCFill(
            oms_order_id="QQQ-stop-TPC-QQQ-1",
            fill_price=97.95,
            fill_qty=10,
            symbol="QQQ",
            fill_time=ts,
            order_role="stop",
            exit_type="STOP",
        ),
    )

    assert events[-1].code == "STOP_FILLED"
    assert "QQQ" in next_state.second_entry_seeds
    assert next_state.second_entry_seeds["QQQ"].source_setup_id == "TPC-QQQ-1"


def test_etf_engine_clips_stale_exit_qty_and_emits_canonical_net_trade_outcome() -> None:
    ts = datetime(2025, 3, 3, 15, 0, tzinfo=ZoneInfo("UTC"))
    engine = ETFStrategyBacktestEngine(
        strategy_id="TPC",
        configs={"QQQ": TPC_SYMBOL_CONFIGS["QQQ"]},
        core_logic=_NoopETFCoreLogic,
        state_factory=ETFCoreState,
        bar_input_factory=ETFBarInput,
        fill_factory=ETFFill,
        order_update_factory=ETFOrderUpdate,
        indicator_module=object(),
    )
    record = TradeRecord(
        symbol="QQQ",
        direction=int(Direction.LONG),
        entry_time=ts,
        entry_price=100.0,
        qty=10,
        initial_stop=98.0,
    )
    open_trades = {
        "QQQ": _OpenTrade(
            record=record,
            qty_open=5,
            qty_initial=10,
            direction=Direction.LONG,
            setup_id="TPC-QQQ-stale-exit",
            risk_per_share=2.0,
            risk_dollars=20.0,
            realised_pnl=190.0,
            commission=1.0,
        )
    }
    order = SimOrder("QQQ-flatten-stale", "QQQ", OrderSide.SELL, OrderType.MARKET, 10, tick_size=0.01, tag="flatten")
    symbol_results = {"QQQ": SymbolResult(symbol="QQQ")}

    cash, _state, _events = engine._handle_fill(
        state=ETFCoreState(),
        result=FillResult(order, FillStatus.FILLED, fill_price=110.0, fill_time=ts, commission=1.0),
        positions={"QQQ": (5, 100.0)},
        open_trades=open_trades,
        order_context={order.order_id: {"role": "flatten", "exit_type": "TEST_FLATTEN"}},
        symbol_results=symbol_results,
        cash=100_000.0,
        broker=SimBroker(),
        bar_index=10,
    )

    closed = symbol_results["QQQ"].trades[0]
    outcome = trade_outcomes_from_records([closed])[0]

    assert cash == 100_549.5
    assert closed.pnl_dollars == 239.5
    assert closed.net_pnl == 239.5
    assert closed.gross_pnl == 241.0
    assert closed.commission == 1.5
    assert outcome["net_pnl"] == 239.5
    assert outcome["gross_pnl"] == 241.0


def test_new_etf_auto_plugins_expose_seven_nonempty_tpc_phases() -> None:
    for candidate_fn in (tpc_candidates,):
        for phase in range(1, 8):
            assert candidate_fn(phase)


def test_tpc_balance_candidate_explicitly_preserves_qqq_type_b_branch() -> None:
    phase_six = dict(tpc_candidates(6))
    phase_seven = dict(tpc_candidates(7))

    assert phase_six["qqq_balance_without_gld_loosen"]["QQQ.type_b_enabled"] is True
    assert phase_six["qqq_balance_without_gld_loosen"]["QQQ.score_b_min"] == 15
    assert phase_seven["qqq_quality_supply_stack"]["QQQ.type_b_enabled"] is True
    assert phase_seven["qqq_quality_supply_stack"]["QQQ.score_a_min"] == 16


def test_tpc_immutable_score_has_no_more_than_seven_components() -> None:
    assert len(TPC_SCORING_WEIGHTS) <= 7


def test_tpc_discrimination_score_does_not_reward_compounded_return_directly() -> None:
    assert "expected_dollars" not in TPC_SCORING_WEIGHTS
    assert "false_positive_control" in TPC_SCORING_WEIGHTS
    assert "symbol_balance" in TPC_SCORING_WEIGHTS


def test_etf_composite_score_keeps_low_sample_candidates_in_play() -> None:
    metrics = _etf_metrics(total_trades=6, net_return_pct=-0.5, avg_r=-0.2, trades_per_month=0.12, profit_factor=0.0)

    score = composite_score(metrics, {"min_valid_trades": 1, "max_dd_pct": 30.0})

    assert not score.rejected
    assert score.total > 0.0


def test_etf_composite_score_rewards_return_and_frequency() -> None:
    base = _etf_metrics(total_trades=8, net_return_pct=-1.0, avg_r=-0.2, trades_per_month=0.15, profit_factor=0.4)
    better = _etf_metrics(total_trades=36, net_return_pct=4.0, avg_r=0.05, trades_per_month=0.70, profit_factor=1.05)

    assert composite_score(better).total > composite_score(base).total


def _etf_metrics(
    *,
    total_trades: int,
    net_return_pct: float,
    avg_r: float,
    trades_per_month: float,
    profit_factor: float,
) -> ETFMetrics:
    return ETFMetrics(
        total_trades=total_trades,
        net_return_pct=net_return_pct,
        profit_factor=profit_factor,
        avg_r=avg_r,
        total_r=avg_r * total_trades,
        win_rate=0.35,
        max_dd_pct=3.0,
        sharpe=0.0,
        trades_per_month=trades_per_month,
        return_per_trade_pct=net_return_pct / max(total_trades, 1),
    )
