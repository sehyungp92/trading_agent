"""Round 8 TPC additive candidates for 30m pullback alpha extraction."""
from __future__ import annotations

from typing import Any


ROUND8_STRUCTURAL_SEED: dict[str, Any] = {}


def get_round8_phase_candidates(phase: int) -> list[tuple[str, dict[str, Any]]]:
    if phase == 1:
        return _phase_1_additive_30m_pullback()
    if phase == 2:
        return _phase_2_pb30_ema20_context_filters()
    if phase == 3:
        return _phase_3_symbol_specific_ema20_context()
    if phase == 4:
        return _phase_4_risk_and_exit_extraction()
    return []


def _pb30_base() -> dict[str, Any]:
    return {
        "all.pb30_pullback_enabled": True,
        "all.pb30_pullback_min_bars_30m": 6,
        "all.pb30_pullback_max_bars_30m": 20,
        "all.pb30_confirmation_required": 1,
        "all.pb30_confirmation_combo_mode": "structure_or_vwap",
        "all.pb30_entry_order_model": "",
    }


def _pb30_symbol(symbol: str) -> dict[str, Any]:
    return {
        f"{symbol}.pb30_pullback_enabled": True,
        f"{symbol}.pb30_pullback_min_bars_30m": 6,
        f"{symbol}.pb30_pullback_max_bars_30m": 20,
        f"{symbol}.pb30_confirmation_required": 1,
        f"{symbol}.pb30_confirmation_combo_mode": "structure_or_vwap",
        f"{symbol}.pb30_entry_order_model": "",
    }


def _pb30_ema20_context(mode: str, *, distance_atr: float = 0.15, lookback: int = 0) -> dict[str, Any]:
    return {
        **_pb30_base(),
        "all.pb30_ema20_context_enabled": True,
        "all.pb30_ema20_context_mode": mode,
        "all.pb30_ema20_context_distance_atr": distance_atr,
        "all.pb30_ema20_context_lookback_bars_30m": lookback,
    }


def _pb30_ma_transition(
    mode: str,
    *,
    lookback: int = 6,
    min_slope_atr: float = 0.0,
    window: int = 12,
) -> dict[str, Any]:
    return {
        **_pb30_base(),
        "all.pb30_ma_transition_enabled": True,
        "all.pb30_ma_transition_mode": mode,
        "all.pb30_ma_transition_lookback_bars_30m": lookback,
        "all.pb30_ma_transition_min_slope_atr": min_slope_atr,
        "all.pb30_ma_transition_window_bars_30m": window,
    }


def _pb30_symbol_ema20_context(symbol: str, mode: str, *, distance_atr: float = 0.15, lookback: int = 0) -> dict[str, Any]:
    return {
        **_pb30_symbol(symbol),
        f"{symbol}.pb30_ema20_context_enabled": True,
        f"{symbol}.pb30_ema20_context_mode": mode,
        f"{symbol}.pb30_ema20_context_distance_atr": distance_atr,
        f"{symbol}.pb30_ema20_context_lookback_bars_30m": lookback,
    }


def _pb30_symbol_ma_transition(
    symbol: str,
    mode: str,
    *,
    lookback: int = 6,
    min_slope_atr: float = 0.0,
    window: int = 12,
) -> dict[str, Any]:
    return {
        **_pb30_symbol(symbol),
        f"{symbol}.pb30_ma_transition_enabled": True,
        f"{symbol}.pb30_ma_transition_mode": mode,
        f"{symbol}.pb30_ma_transition_lookback_bars_30m": lookback,
        f"{symbol}.pb30_ma_transition_min_slope_atr": min_slope_atr,
        f"{symbol}.pb30_ma_transition_window_bars_30m": window,
    }


def _merge(*mutations: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in mutations:
        merged.update(item)
    return merged


def _relax_orderly(mutations: dict[str, Any]) -> dict[str, Any]:
    return {**mutations, "all.pb30_pullback_orderly_required": False}


def _relax_symbol_orderly(symbol: str, mutations: dict[str, Any]) -> dict[str, Any]:
    return {**mutations, f"{symbol}.pb30_pullback_orderly_required": False}


def _risk_stack(*, max_risk: float, notional: float, a: float, a_plus: float, b: float) -> dict[str, Any]:
    return {
        "all.max_risk_pct": max_risk,
        "all.max_position_notional_pct": notional,
        "all.risk_a_pct": a,
        "all.risk_a_plus_pct": a_plus,
        "all.risk_b_pct": b,
        "all.min_stop_atr_mult": 0.15,
    }


def _phase_1_additive_30m_pullback() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("add_pb30_scaled_confirm1", _pb30_base()),
        ("add_pb30_scaled_confirm2", {**_pb30_base(), "all.pb30_confirmation_required": 2}),
        ("add_pb30_confirm1_any", {**_pb30_base(), "all.pb30_confirmation_combo_mode": "any"}),
        ("add_pb30_duration_5_18", {**_pb30_base(), "all.pb30_pullback_min_bars_30m": 5, "all.pb30_pullback_max_bars_30m": 18}),
        ("add_pb30_duration_7_22", {**_pb30_base(), "all.pb30_pullback_min_bars_30m": 7, "all.pb30_pullback_max_bars_30m": 22}),
        ("add_pb30_orderly", {**_pb30_base(), "all.pb30_pullback_orderly_required": True}),
        ("add_pb30_value_hits2", {**_pb30_base(), "all.pb30_type_a_value_hits_min": 2}),
        (
            "add_pb30_orderly_value2",
            {**_pb30_base(), "all.pb30_pullback_orderly_required": True, "all.pb30_type_a_value_hits_min": 2},
        ),
        ("add_pb30_fib_38_70", {**_pb30_base(), "all.pb30_fib_a_low": 0.38, "all.pb30_fib_a_high": 0.70}),
        ("add_pb30_fib_42_72", {**_pb30_base(), "all.pb30_fib_a_low": 0.42, "all.pb30_fib_a_high": 0.72}),
    ]


def _phase_2_pb30_ema20_context_filters() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("pb30_ma_fast_slope_6_000", _pb30_ma_transition("fast_slope", lookback=6, min_slope_atr=0.00)),
        ("pb30_ma_fast_slope_6_002", _pb30_ma_transition("fast_slope", lookback=6, min_slope_atr=0.02)),
        ("pb30_ma_fast_slope_8_002", _pb30_ma_transition("fast_slope", lookback=8, min_slope_atr=0.02)),
        ("pb30_ma_stack", _pb30_ma_transition("stack", lookback=6, min_slope_atr=0.00)),
        ("pb30_ma_slope_stack_6_000", _pb30_ma_transition("slope_and_stack", lookback=6, min_slope_atr=0.00)),
        ("pb30_ma_slope_stack_6_002", _pb30_ma_transition("slope_and_stack", lookback=6, min_slope_atr=0.02)),
        ("pb30_ma_fast_slow_slope_8_000", _pb30_ma_transition("fast_slow_slope", lookback=8, min_slope_atr=0.00)),
        ("pb30_ma_transition_12", _pb30_ma_transition("transition", lookback=6, min_slope_atr=0.00, window=12)),
        ("pb30_ma_transition_16_002", _pb30_ma_transition("transition", lookback=8, min_slope_atr=0.02, window=16)),
        ("pb30_ma_transition_only_12", _pb30_ma_transition("transition_only", lookback=6, min_slope_atr=0.00, window=12)),
    ]


def _phase_3_symbol_specific_ema20_context() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("gld_pb30_ma_fast_slope_6_002", _pb30_symbol_ma_transition("GLD", "fast_slope", lookback=6, min_slope_atr=0.02)),
        ("gld_pb30_ma_slope_stack_6_000", _pb30_symbol_ma_transition("GLD", "slope_and_stack", lookback=6)),
        ("gld_pb30_ma_transition_12", _pb30_symbol_ma_transition("GLD", "transition", lookback=6, window=12)),
        ("qqq_pb30_ma_fast_slope_6_000", _pb30_symbol_ma_transition("QQQ", "fast_slope", lookback=6)),
        ("qqq_pb30_ma_slope_stack_8_000", _pb30_symbol_ma_transition("QQQ", "slope_and_stack", lookback=8)),
        ("qqq_pb30_ma_transition_16", _pb30_symbol_ma_transition("QQQ", "transition", lookback=8, window=16)),
        (
            "pb30_ma_transition_ema20_hold",
            _merge(_pb30_ma_transition("transition", lookback=6, window=12), _pb30_ema20_context("hold")),
        ),
        (
            "pb30_ma_transition_ema20_confirm15",
            _merge(_pb30_ma_transition("transition", lookback=6, window=12), _pb30_ema20_context("confirm")),
        ),
        (
            "pb30_ma_transition_ema20_reclaim",
            _merge(_pb30_ma_transition("transition", lookback=6, window=12), _pb30_ema20_context("reclaim")),
        ),
        ("pb30_ma_slope_stack_value2", {**_pb30_ma_transition("slope_and_stack", lookback=6), "all.pb30_type_a_value_hits_min": 2}),
        ("pb30_ma_fast_slope_confirm2", {**_pb30_ma_transition("fast_slope", lookback=6), "all.pb30_confirmation_required": 2}),
        (
            "pb30_relax_orderly_ma_transition_ema20_touch",
            _relax_orderly(_merge(_pb30_ma_transition("transition", lookback=6, window=12), _pb30_ema20_context("touch"))),
        ),
        (
            "pb30_relax_orderly_ma_transition_ema20_confirm15",
            _relax_orderly(_merge(_pb30_ma_transition("transition", lookback=6, window=12), _pb30_ema20_context("confirm"))),
        ),
        (
            "gld_relax_orderly_ma_transition_touch",
            _relax_symbol_orderly("GLD", _merge(_pb30_symbol_ma_transition("GLD", "transition"), _pb30_symbol_ema20_context("GLD", "touch"))),
        ),
        (
            "qqq_relax_orderly_ma_transition_touch",
            _relax_symbol_orderly("QQQ", _merge(_pb30_symbol_ma_transition("QQQ", "transition"), _pb30_symbol_ema20_context("QQQ", "touch"))),
        ),
    ]


def _phase_4_risk_and_exit_extraction() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("evidence_risk_temper_5x_020", _risk_stack(max_risk=0.020, notional=5.0, a=0.013, a_plus=0.020, b=0.0085)),
        ("evidence_risk_temper_4x_018", _risk_stack(max_risk=0.018, notional=4.0, a=0.012, a_plus=0.018, b=0.0080)),
        ("evidence_risk_temper_3x_016", _risk_stack(max_risk=0.016, notional=3.0, a=0.010, a_plus=0.016, b=0.0070)),
        (
            "evidence_mfe_giveback_225_045_lock075",
            {
                "all.mfe_giveback_trigger_r": 2.25,
                "all.mfe_giveback_retain_frac": 0.45,
                "all.mfe_giveback_lock_r": 0.75,
                "all.mfe_giveback_after_t1_only": False,
            },
        ),
        (
            "evidence_mfe_giveback_250_050_lock075",
            {
                "all.mfe_giveback_trigger_r": 2.50,
                "all.mfe_giveback_retain_frac": 0.50,
                "all.mfe_giveback_lock_r": 0.75,
                "all.mfe_giveback_after_t1_only": False,
            },
        ),
        ("evidence_stall_exit_40", {"all.stall_exit_bars_15m": 40, "all.stall_exit_min_mfe_r": 1.0, "all.stall_exit_max_current_r": 0.2}),
        ("evidence_max_hold_48_lowmfe", {"all.max_hold_bars_15m": 48, "all.time_stop_min_mfe_r": 0.5}),
        ("evidence_t2_trim_250", {"all.t2_r": 2.50}),
        ("evidence_t2_extend_300", {"all.t2_r": 3.00}),
    ]
