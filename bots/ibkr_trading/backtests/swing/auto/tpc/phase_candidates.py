"""TPC phased candidate mutations.

Round 8 is a discrimination-first round. The round 7 diagnostics showed that
the strategy still admits too many never-worked/low-MFE trades, GLD dominates
the sample, and headline return is partly a scaling/compounding artefact. These
candidates therefore test causal filters, entry selectivity, and symbol balance
before any monetization or risk scaling.
"""
from __future__ import annotations

from typing import Any


def get_phase_candidates(phase: int) -> list[tuple[str, dict[str, Any]]]:
    if phase == 1:
        return _phase_1_gld_signal_discrimination()
    if phase == 2:
        return _phase_2_regime_and_room_discrimination()
    if phase == 3:
        return _phase_3_session_and_symbol_balance()
    if phase == 4:
        return _phase_4_entry_quality_discrimination()
    if phase == 5:
        return _phase_5_pullback_purity()
    if phase == 6:
        return _phase_6_discrimination_combos()
    if phase == 7:
        return _phase_7_qqq_excellent_supply()
    return []


def _risk_stack(
    *,
    max_risk: float,
    notional: float,
    a: float,
    a_plus: float,
    b: float,
    min_stop: float = 0.0,
) -> dict[str, Any]:
    return {
        "all.max_risk_pct": max_risk,
        "all.risk_a_pct": a,
        "all.risk_a_plus_pct": a_plus,
        "all.risk_b_pct": b,
        "all.max_position_notional_pct": notional,
        "all.min_stop_atr_mult": min_stop,
    }


def _dynamic_risk(
    *,
    max_risk: float,
    notional: float,
    min_mult: float,
    max_mult: float,
    floor: float = 14.0,
    ceiling: float = 21.0,
    curve: float = 1.0,
    min_stop: float = 0.0,
) -> dict[str, Any]:
    return {
        "all.dynamic_risk_enabled": True,
        "all.dynamic_risk_score_floor": floor,
        "all.dynamic_risk_score_ceiling": ceiling,
        "all.dynamic_risk_min_mult": min_mult,
        "all.dynamic_risk_max_mult": max_mult,
        "all.dynamic_risk_curve": curve,
        "all.max_risk_pct": max_risk,
        "all.max_position_notional_pct": notional,
        "all.min_stop_atr_mult": min_stop,
    }


def _real_second_entry(
    *,
    score_min: int = 14,
    max_wait: int = 16,
    requires_a_plus: bool = False,
    min_source_score: float = 0.0,
    source_a_plus: bool = False,
    mode: str = "real_reentry",
) -> dict[str, Any]:
    return {
        "all.type_c_enabled": True,
        "all.type_c_mode": mode,
        "all.type_c_requires_a_plus": requires_a_plus,
        "all.second_entry_min_wait_bars_15m": 1,
        "all.second_entry_max_wait_bars_15m": max_wait,
        "all.second_entry_require_vwap": True,
        "all.second_entry_require_structure": True,
        "all.second_entry_score_min": score_min,
        "all.second_entry_min_source_score": min_source_score,
        "all.second_entry_requires_source_a_plus": source_a_plus,
    }


def _addon_continuation(
    *,
    trigger_r: float,
    size_mult: float,
    min_score: int,
    max_total_risk_pct: float = 0.0,
    max_notional_pct: float = 0.0,
) -> dict[str, Any]:
    return {
        "all.addon_enabled": True,
        "all.addon_trigger_r": trigger_r,
        "all.addon_size_mult": size_mult,
        "all.addon_min_score": min_score,
        "all.addon_requires_t1": True,
        "all.addon_require_vwap_hold": True,
        "all.addon_require_structure_hold": True,
        "all.addon_max_total_risk_pct": max_total_risk_pct,
        "all.addon_max_notional_pct": max_notional_pct,
    }


def _mfe_giveback(
    *,
    trigger_r: float,
    retain_frac: float,
    lock_r: float,
    after_t1_only: bool = True,
) -> dict[str, Any]:
    return {
        "all.mfe_giveback_trigger_r": trigger_r,
        "all.mfe_giveback_retain_frac": retain_frac,
        "all.mfe_giveback_lock_r": lock_r,
        "all.mfe_giveback_after_t1_only": after_t1_only,
    }


def _phase_1_gld_signal_discrimination() -> list[tuple[str, dict[str, Any]]]:
    """Tighten GLD setup evidence without touching risk or monetization."""

    return [
        ("gld_score_a16_b16", {"GLD.score_a_min": 16, "GLD.score_b_min": 16}),
        ("gld_score_a17_b17", {"GLD.score_a_min": 17, "GLD.score_b_min": 17}),
        ("gld_a_plus_only_classic", {"GLD.score_a_min": 17, "GLD.type_a_value_hits_min": 2}),
        ("gld_confirm2_structure_vwap", {"GLD.confirmation_required": 2, "GLD.confirmation_combo_mode": "structure_vwap"}),
        ("gld_confirm2_preferred", {"GLD.confirmation_required": 2, "GLD.confirmation_combo_mode": "preferred"}),
        ("gld_require_structure", {"GLD.require_structure_confirmation": True}),
        ("gld_require_vwap", {"GLD.require_vwap_confirmation": True}),
        ("gld_require_structure_and_vwap", {
            "GLD.require_structure_confirmation": True,
            "GLD.require_vwap_confirmation": True,
        }),
        ("gld_value_hits2", {"GLD.type_a_value_hits_min": 2}),
        ("gld_value_hits3", {"GLD.type_a_value_hits_min": 3}),
        ("gld_orderly_pullbacks", {"GLD.pullback_orderly_required": True}),
        ("gld_volume_contract_125", {"GLD.pullback_volume_contract_max": 1.25}),
        ("gld_no_type_b_or_c", {"GLD.type_b_enabled": False, "GLD.type_c_enabled": False}),
    ]


def _phase_2_regime_and_room_discrimination() -> list[tuple[str, dict[str, Any]]]:
    """Require stronger trend/room context for entries that currently die early."""

    return [
        ("all_adx15_di", {"all.min_adx_4h": 15.0, "all.require_di_alignment": True}),
        ("all_adx18_di", {"all.min_adx_4h": 18.0, "all.require_di_alignment": True}),
        ("gld_adx16_di", {"GLD.min_adx_4h": 16.0, "GLD.require_di_alignment": True}),
        ("gld_adx20_di", {"GLD.min_adx_4h": 20.0, "GLD.require_di_alignment": True}),
        ("all_ma50_003_ma100_006_di", {
            "all.min_ma50_slope_atr_4h": 0.03,
            "all.min_ma100_slope_atr_4h": 0.06,
            "all.require_di_alignment": True,
        }),
        ("gld_ma50_004_ma100_006_di", {
            "GLD.min_ma50_slope_atr_4h": 0.04,
            "GLD.min_ma100_slope_atr_4h": 0.06,
            "GLD.require_di_alignment": True,
        }),
        ("gld_room30", {"GLD.daily_room_min_r": 3.0}),
        ("gld_room35", {"GLD.daily_room_min_r": 3.5}),
        ("all_extension175", {"all.max_extension_atr_mult": 1.75}),
        ("gld_extension150", {"GLD.max_extension_atr_mult": 1.50}),
        ("gld_room30_extension175", {"GLD.daily_room_min_r": 3.0, "GLD.max_extension_atr_mult": 1.75}),
        ("qqq_context_min_neg010", {"QQQ.asset_context_min_score": -0.10}),
        ("qqq_context_min010", {"QQQ.asset_context_min_score": 0.10}),
        ("qqq_context_min025", {"QQQ.asset_context_min_score": 0.25}),
        ("gld_context_min_neg050", {"GLD.asset_context_enabled": True, "GLD.asset_context_min_score": -0.50}),
        ("gld_context_min_neg025", {"GLD.asset_context_enabled": True, "GLD.asset_context_min_score": -0.25}),
    ]


def _phase_3_session_and_symbol_balance() -> list[tuple[str, dict[str, Any]]]:
    """Reduce GLD dominance through broad session-quality filters."""

    return [
        ("gld_regular_hours_only", {
            "GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 15, 45)),
        }),
        ("gld_no_late_afternoon", {
            "GLD.primary_windows_et": ((8, 0, 11, 30), (13, 0, 15, 0)),
        }),
        ("gld_midday_gap_only", {
            "GLD.avoid_windows_et": ((11, 0, 13, 0),),
        }),
        ("gld_avoid_first_half_hour", {
            "GLD.avoid_windows_et": ((8, 0, 8, 30), (11, 0, 12, 0)),
        }),
        ("gld_avoid_lunch_and_late", {
            "GLD.avoid_windows_et": ((11, 0, 12, 30), (15, 0, 16, 0)),
        }),
        ("gld_compact_quality_windows", {
            "GLD.primary_windows_et": ((8, 30, 11, 0), (13, 30, 15, 15)),
        }),
        ("qqq_preserve_type_b_supply", {"QQQ.type_b_enabled": True, "QQQ.score_b_min": 15}),
    ]


def _phase_4_entry_quality_discrimination() -> list[tuple[str, dict[str, Any]]]:
    """Let worse GLD setups miss instead of forcing next-bar fills."""

    return [
        ("gld_structure_stop", {"GLD.entry_order_model": "structure_stop"}),
        ("gld_structure_stop_market", {"GLD.entry_order_model": "structure_stop_market"}),
        ("gld_adaptive_structure_limit", {
            "GLD.entry_order_model": "adaptive_structure_stop",
            "GLD.entry_stop_limit_atr_mult": 0.08,
            "GLD.entry_adaptive_stop_limit_min_atr_mult": 0.10,
            "GLD.entry_adaptive_stop_limit_max_atr_mult": 0.26,
        }),
        ("gld_entry_ttl_2h", {"GLD.entry_order_ttl_hours": 2.0}),
        ("gld_entry_ttl_4h", {"GLD.entry_order_ttl_hours": 4.0}),
        ("gld_stop_limit_005", {"GLD.entry_stop_limit_atr_mult": 0.05}),
        ("gld_stop_limit_012", {"GLD.entry_stop_limit_atr_mult": 0.12}),
        ("gld_signal_stop_buffer_015", {"GLD.signal_stop_buffer_atr_mult": 0.15}),
        ("risk_temper_6x_020", _risk_stack(max_risk=0.020, notional=6.0, a=0.014, a_plus=0.020, b=0.009, min_stop=0.15)),
        ("risk_temper_4x_018", _risk_stack(max_risk=0.018, notional=4.0, a=0.012, a_plus=0.018, b=0.008, min_stop=0.15)),
        ("dynamic_risk_alpha7_balanced", _dynamic_risk(
            max_risk=0.022,
            notional=6.0,
            min_mult=0.70,
            max_mult=1.15,
            floor=10.0,
            ceiling=13.0,
            curve=1.25,
            min_stop=0.15,
        )),
    ]


def _phase_5_pullback_purity() -> list[tuple[str, dict[str, Any]]]:
    """Narrow the broad classic pullback lane that admits most failures."""

    return [
        ("all_fib_a_38_70", {"all.fib_a_low": 0.38, "all.fib_a_high": 0.70}),
        ("all_fib_a_42_72", {"all.fib_a_low": 0.42, "all.fib_a_high": 0.72}),
        ("gld_fib_a_38_70", {"GLD.fib_a_low": 0.38, "GLD.fib_a_high": 0.70}),
        ("gld_fib_a_42_72", {"GLD.fib_a_low": 0.42, "GLD.fib_a_high": 0.72}),
        ("gld_fib_a_33_70", {"GLD.fib_a_low": 0.33, "GLD.fib_a_high": 0.70}),
        ("gld_duration_3_10", {"GLD.pullback_min_bars_1h": 3, "GLD.pullback_max_bars_1h": 10}),
        ("gld_duration_4_12", {"GLD.pullback_min_bars_1h": 4, "GLD.pullback_max_bars_1h": 12}),
        ("gld_duration_4_8", {"GLD.pullback_min_bars_1h": 4, "GLD.pullback_max_bars_1h": 8}),
        ("second_entry_source17", {"all.second_entry_score_min": 15, "all.second_entry_min_source_score": 17.0}),
        ("second_entry_source_a_plus", {"all.second_entry_requires_source_a_plus": True}),
        ("disable_second_entry_for_discrimination", {"all.type_c_enabled": False}),
    ]


def _phase_6_discrimination_combos() -> list[tuple[str, dict[str, Any]]]:
    """Broad packages assembled from causal discrimination hypotheses."""

    return [
        ("gld_score16_value2_structure", {
            "GLD.score_a_min": 16,
            "GLD.score_b_min": 16,
            "GLD.type_a_value_hits_min": 2,
            "GLD.require_structure_confirmation": True,
        }),
        ("gld_score16_structure_stop_regular_hours", {
            "GLD.score_a_min": 16,
            "GLD.score_b_min": 16,
            "GLD.entry_order_model": "structure_stop",
            "GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 15, 45)),
        }),
        ("gld_preferred_room3_extension175", {
            "GLD.confirmation_required": 2,
            "GLD.confirmation_combo_mode": "preferred",
            "GLD.daily_room_min_r": 3.0,
            "GLD.max_extension_atr_mult": 1.75,
        }),
        ("gld_regime_quality_fib_38_70", {
            "GLD.min_adx_4h": 16.0,
            "GLD.require_di_alignment": True,
            "GLD.fib_a_low": 0.38,
            "GLD.fib_a_high": 0.70,
        }),
        ("all_regime_value2_no_second_entry", {
            "all.min_ma100_slope_atr_4h": 0.06,
            "all.require_di_alignment": True,
            "all.type_a_value_hits_min": 2,
            "all.type_c_enabled": False,
        }),
        ("gld_quality_stack_preserve_qqq_type_b", {
            "GLD.score_a_min": 16,
            "GLD.type_a_value_hits_min": 2,
            "GLD.entry_order_model": "structure_stop",
            "QQQ.type_b_enabled": True,
            "QQQ.score_b_min": 15,
        }),
        ("decompound_no_addon_discrimination", {
            "all.addon_enabled": False,
            "all.min_ma100_slope_atr_4h": 0.06,
            "all.require_di_alignment": True,
        }),
        ("qqq_balance_without_gld_loosen", {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.20,
            "QQQ.fib_b_high": 0.38,
            "QQQ.score_b_min": 15,
            "all.second_entry_score_min": 15,
            "all.second_entry_min_source_score": 16.0,
        }),
    ]


def _phase_7_qqq_excellent_supply() -> list[tuple[str, dict[str, Any]]]:
    """Add only protected QQQ supply after GLD false-positive filters survive.

    The goal is not generic frequency. These candidates loosen QQQ access only
    when paired with score, room, confirmation, or source-quality constraints so
    accepted mutations should add excellent QQQ trades rather than replace GLD
    overfitting with QQQ overfitting.
    """

    return [
        ("qqq_room16_extension225_score15", {
            "QQQ.daily_room_min_r": 1.6,
            "QQQ.max_extension_atr_mult": 2.25,
            "QQQ.score_a_min": 15,
            "QQQ.score_b_min": 15,
        }),
        ("qqq_room15_extension250_score16", {
            "QQQ.daily_room_min_r": 1.5,
            "QQQ.max_extension_atr_mult": 2.50,
            "QQQ.score_a_min": 16,
            "QQQ.score_b_min": 16,
        }),
        ("qqq_type_b_18_45_score16_confirm2", {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.18,
            "QQQ.fib_b_high": 0.45,
            "QQQ.score_b_min": 16,
            "QQQ.confirmation_required": 2,
        }),
        ("qqq_type_b_16_42_score16_preferred", {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.16,
            "QQQ.fib_b_high": 0.42,
            "QQQ.score_b_min": 16,
            "QQQ.confirmation_required": 2,
            "QQQ.confirmation_combo_mode": "preferred",
        }),
        ("qqq_type_b_a_plus_16_50", {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": True,
            "QQQ.fib_b_low": 0.16,
            "QQQ.fib_b_high": 0.50,
            "QQQ.score_b_min": 15,
        }),
        ("qqq_second_entry_source16_wait20", {
            "QQQ.type_c_enabled": True,
            "QQQ.type_c_mode": "real_reentry",
            "QQQ.type_c_requires_a_plus": False,
            "QQQ.second_entry_min_wait_bars_15m": 1,
            "QQQ.second_entry_max_wait_bars_15m": 20,
            "QQQ.second_entry_require_vwap": True,
            "QQQ.second_entry_require_structure": True,
            "QQQ.second_entry_score_min": 15,
            "QQQ.second_entry_min_source_score": 16.0,
        }),
        ("qqq_second_entry_source_aplus_wait20", {
            "QQQ.type_c_enabled": True,
            "QQQ.type_c_mode": "real_reentry",
            "QQQ.type_c_requires_a_plus": False,
            "QQQ.second_entry_min_wait_bars_15m": 1,
            "QQQ.second_entry_max_wait_bars_15m": 20,
            "QQQ.second_entry_require_vwap": True,
            "QQQ.second_entry_require_structure": True,
            "QQQ.second_entry_score_min": 15,
            "QQQ.second_entry_requires_source_a_plus": True,
        }),
        ("qqq_late_morning_window_score16", {
            "QQQ.primary_windows_et": ((9, 35, 12, 15), (13, 30, 15, 45)),
            "QQQ.score_a_min": 16,
            "QQQ.score_b_min": 16,
            "QQQ.confirmation_required": 2,
        }),
        ("qqq_market_next_bar_high_score", {
            "QQQ.entry_order_model": "market_next_bar",
            "QQQ.score_a_min": 16,
            "QQQ.score_b_min": 16,
            "QQQ.confirmation_required": 2,
        }),
        ("qqq_context_neg010_alpha7_supply", {
            "QQQ.asset_context_min_score": -0.10,
            "QQQ.daily_room_min_r": 1.5,
            "QQQ.max_extension_atr_mult": 2.50,
            "QQQ.score_a_min": 10,
            "QQQ.score_b_min": 9,
        }),
        ("qqq_context010_alpha7_quality", {
            "QQQ.asset_context_min_score": 0.10,
            "QQQ.daily_room_min_r": 1.6,
            "QQQ.max_extension_atr_mult": 2.25,
            "QQQ.score_a_min": 11,
            "QQQ.score_b_min": 10,
        }),
        ("qqq_aplus_short_context_neg010", {
            "QQQ.asset_context_min_score": -0.10,
            "QQQ.shorts_enabled": True,
            "QQQ.shorts_require_a_plus": True,
            "QQQ.min_short_score": 12,
        }),
        ("qqq_quality_supply_stack", {
            "QQQ.daily_room_min_r": 1.6,
            "QQQ.max_extension_atr_mult": 2.25,
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.18,
            "QQQ.fib_b_high": 0.45,
            "QQQ.score_a_min": 16,
            "QQQ.score_b_min": 16,
            "QQQ.confirmation_required": 2,
        }),
    ]
