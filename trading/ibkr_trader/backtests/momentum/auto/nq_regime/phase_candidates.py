from __future__ import annotations

from typing import Any


MODULE_ENABLE_MUTATIONS: dict[str, Any] = {
    "flags.enable_structural_expansion": True,
    "flags.enable_liquidity_reversion": True,
    "flags.enable_second_wind": True,
}

ROUND5_SYNERGY_GUARD_MUTATIONS: dict[str, Any] = {
    "param_overrides.TARGET_ROOM_MIN_R": 0.50,
    "param_overrides.SECOND_WIND_ENTRY_STOP_INVALIDATION_ENABLED": True,
    "param_overrides.MAX_TRADES_PER_DAY": 4,
    "param_overrides.MAX_FULL_RISK_TRADES": 3,
    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
}


def build_round5_seed_from_configs(
    round4a_config: dict[str, Any],
    round4b_config: dict[str, Any],
    round4c_config: dict[str, Any],
) -> dict[str, Any]:
    """Blend the final round_4 single-component winners into an all-module seed."""
    seed = dict(round4a_config)
    seed.update(
        (key, value)
        for key, value in round4b_config.items()
        if key.startswith("param_overrides.STRUCTURAL")
    )
    seed.update(
        (key, value)
        for key, value in round4c_config.items()
        if key.startswith("param_overrides.REVERSION")
    )
    seed.update(MODULE_ENABLE_MUTATIONS)
    seed.update(ROUND5_SYNERGY_GUARD_MUTATIONS)
    return seed


BASE_MUTATIONS: dict[str, Any] = {
    # Round_5 deliberately starts from the strongest all-module blend observed
    # after round_4a/4b/4c, not from the single best isolated module.
    "flags.enable_structural_expansion": True,
    "flags.enable_liquidity_reversion": True,
    "flags.enable_second_wind": True,
    "param_overrides.REGIME_MIN_CONFIDENCE": 0.65,
    "param_overrides.REGIME_MIN_MARGIN": 0.15,
    "param_overrides.STRUCTURAL_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_A_PLUS_SCORE": 10,
    "param_overrides.REVERSION_MIN_SCORE": 8,
    "param_overrides.REVERSION_A_SCORE": 8,
    "param_overrides.REVERSION_A_PLUS_SCORE": 11,
    "param_overrides.SECOND_WIND_MIN_SCORE": 8,
    "param_overrides.SECOND_WIND_A_SCORE": 8,
    "param_overrides.SECOND_WIND_A_PLUS_SCORE": 10,
    "param_overrides.TARGET_ROOM_MIN_R": 0.50,
    "param_overrides.REVERSION_STANDARD_STOP_CAP": 10.0,
    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 12.0,
    "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
    "param_overrides.SECOND_WIND_ENTRY_STOP_INVALIDATION_ENABLED": True,
    "param_overrides.MAX_TRADES_PER_DAY": 4,
    "param_overrides.MAX_FULL_RISK_TRADES": 3,
    "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.0,
    "param_overrides.PROFIT_FLOOR_ENABLED": True,
    "param_overrides.PROFIT_FLOOR_TRIGGER_R": 0.75,
    "param_overrides.PROFIT_FLOOR_LOCK_R": 0.25,
    "param_overrides.MFE_RATCHET_ENABLED": True,
    "param_overrides.MFE_RATCHET_TRIGGER_R": 1.0,
    "param_overrides.MFE_RATCHET_FLOOR_PCT": 0.65,
    "param_overrides.TIME_STOP_ENABLED": False,
    "param_overrides.STRUCTURAL_ENTRY_MODE": "adaptive_retest",
    "param_overrides.STRUCTURAL_RETEST_OFFSET_TICKS": 1,
    "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.55,
    "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.65,
    "param_overrides.STRUCTURAL_STOP_MODEL": "recent_5m",
    "param_overrides.STRUCTURAL_MIN_STOP_PTS": 10.0,
    "param_overrides.STRUCTURAL_MAX_STOP_PTS": 35.0,
    "param_overrides.STRUCTURAL_HYBRID_CLOSE_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 45.0,
    "param_overrides.STRUCTURAL_ADAPTIVE_RETEST_PREFERS_FVG": True,
    "param_overrides.STRUCTURAL_MIDPOINT_RETEST_ENABLED": True,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENABLED": True,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE": "close",
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MAX_AGE_MINUTES": 240,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": 1.0,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_VOLUME_MULTIPLE": 0.8,
    "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 10,
    "param_overrides.STRUCTURAL_LONG_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_ALLOW_MIN_MICRO_SIZE": True,
    "param_overrides.STRUCTURAL_MIN_MICRO_MAX_RISK_PCT": 0.01,
    "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
    "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
    "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.10,
    "param_overrides.REVERSION_ENTRY_MODEL": "swept_level_retest",
    "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 0,
    "param_overrides.REVERSION_MIN_PENETRATION_PTS": 2.0,
    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 12.0,
    "param_overrides.REVERSION_ENABLE_SWING_LEVELS": True,
    "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 36,
    "param_overrides.REVERSION_SWING_RADIUS": 2,
    "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 4,
    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": False,
    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": True,
    "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
    "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
    "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
    "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_midpoint",
    "param_overrides.SECOND_WIND_ATR_STOP_MULT": 0.75,
    "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 1.2,
    "param_overrides.SECOND_WIND_MAX_STOP_PTS": 30.0,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
    "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
    "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
    "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": False,
    "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
    "param_overrides.SECOND_WIND_EMA_TRAIL_EXIT_ENABLED": True,
    "param_overrides.SECOND_WIND_EMA_TRAIL_REQUIRES_PARTIAL": True,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE": 9,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE": 0.60,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE": 1.30,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION": 0.70,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR": 0.08,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT": True,
    "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
    "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": 15,
    "param_overrides.ALLOW_LATE_PM_REVERSION": True,
    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "All-Module Seed And Router Balance",
        [
            "total_r_per_month",
            "trades_per_month",
            "module_coverage",
            "min_module_trades",
        ],
    ),
    2: (
        "Daily Capacity And Opportunity Density",
        [
            "total_trades",
            "trades_per_month",
            "daily_lockout_events",
            "max_drawdown_pct",
        ],
    ),
    3: (
        "PM Continuation Vs Reversion Conflict Resolution",
        [
            "module_second_wind_trades",
            "module_liquidity_reversion_trades",
            "routing_second_wind_selected_to_fill_rate",
            "routing_liquidity_reversion_selected_to_fill_rate",
        ],
    ),
    4: (
        "Structural Expansion Integration",
        [
            "module_structural_expansion_trades",
            "module_structural_expansion_avg_r",
            "module_structural_expansion_profit_factor",
            "routing_structural_expansion_selected",
        ],
    ),
    5: (
        "Liquidity Reversion Quality/Frequency",
        [
            "module_liquidity_reversion_total_r_per_month",
            "module_liquidity_reversion_trades",
            "module_liquidity_reversion_mfe_capture",
        ],
    ),
    6: (
        "Unified Exit Capture",
        [
            "mfe_capture",
            "positive_mfe_loser_rate",
            "module_second_wind_positive_mfe_loser_rate",
            "module_liquidity_reversion_mfe_capture",
        ],
    ),
    7: (
        "Final Synergy Alpha/Frequency Stack",
        [
            "total_r_per_month",
            "trades_per_month",
            "module_coverage",
            "module_second_wind_total_r_per_month",
            "module_liquidity_reversion_total_r_per_month",
        ],
    ),
}


def _merge(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


ROUTER_GLOBAL_LED = {
    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.50,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
}

ROUTER_SW_LED_ONLY = {
    "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": False,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_ENABLED": True,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": 8,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 1.5,
    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
}

STRUCTURAL_CORE = {
    "param_overrides.STRUCTURAL_ENTRY_MODE": "adaptive_retest",
    "param_overrides.STRUCTURAL_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.65,
    "param_overrides.STRUCTURAL_STOP_MODEL": "recent_5m",
    "param_overrides.STRUCTURAL_MIN_STOP_PTS": 10.0,
    "param_overrides.STRUCTURAL_MAX_STOP_PTS": 35.0,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENABLED": True,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE": "close",
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MAX_AGE_MINUTES": 240,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": 1.0,
    "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_VOLUME_MULTIPLE": 0.8,
    "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 10,
    "param_overrides.STRUCTURAL_LONG_MIN_SCORE": 8,
    "param_overrides.STRUCTURAL_ADAPTIVE_RETEST_PREFERS_FVG": True,
    "param_overrides.STRUCTURAL_MIDPOINT_RETEST_ENABLED": True,
}

REVERSION_CORE = {
    "param_overrides.REVERSION_MIN_SCORE": 8,
    "param_overrides.REVERSION_A_SCORE": 8,
    "param_overrides.REVERSION_STANDARD_STOP_CAP": 10.0,
    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 12.0,
    "param_overrides.REVERSION_ENTRY_MODEL": "swept_level_retest",
    "param_overrides.REVERSION_RETEST_OFFSET_TICKS": 0,
    "param_overrides.REVERSION_MIN_PENETRATION_PTS": 2.0,
    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 12.0,
    "param_overrides.REVERSION_ENABLE_SWING_LEVELS": True,
    "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 36,
    "param_overrides.REVERSION_SWING_RADIUS": 2,
    "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 4,
    "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": True,
    "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": False,
}

SECOND_WIND_CORE = {
    "param_overrides.SECOND_WIND_MIN_SCORE": 8,
    "param_overrides.SECOND_WIND_A_SCORE": 8,
    "param_overrides.SECOND_WIND_STOP_CAP": 30.0,
    "param_overrides.SECOND_WIND_MAX_STOP_PTS": 30.0,
    "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_midpoint",
    "param_overrides.SECOND_WIND_ATR_STOP_MULT": 0.75,
    "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 1.2,
    "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.55,
    "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": True,
    "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
    "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
    "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": False,
}


def get_phase_candidates(phase: int, current_mutations: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    del current_mutations
    if phase == 1:
        return [
            ("r5_router_global_led_seed", ROUTER_GLOBAL_LED),
            ("r5_router_sw_led_only_balance", ROUTER_SW_LED_ONLY),
            ("r5_router_global_led_room075", _merge(ROUTER_GLOBAL_LED, {
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
            })),
            ("r5_router_strict_confidence", {
                "param_overrides.REGIME_MIN_CONFIDENCE": 0.70,
                "param_overrides.REGIME_MIN_MARGIN": 0.18,
            }),
            ("r5_router_relaxed_margin_with_led", _merge(ROUTER_GLOBAL_LED, {
                "param_overrides.REGIME_MIN_MARGIN": 0.10,
            })),
            ("r5_router_a_plus_fallback", {
                "param_overrides.ROUTE_ALLOW_A_PLUS_FALLBACK": True,
                "param_overrides.ROUTE_FALLBACK_MIN_SCORE": 11,
            }),
            ("r5_router_score9_room025", {
                "param_overrides.ROUTE_CANDIDATE_LED_ENABLED": True,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": 9,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.25,
            }),
        ]
    if phase == 2:
        return [
            ("r5_capacity_3_full2", {
                "param_overrides.MAX_TRADES_PER_DAY": 3,
                "param_overrides.MAX_FULL_RISK_TRADES": 2,
            }),
            ("r5_capacity_4_full3", {
                "param_overrides.MAX_TRADES_PER_DAY": 4,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
            }),
            ("r5_capacity_5_full3_guarded", {
                "param_overrides.MAX_TRADES_PER_DAY": 5,
                "param_overrides.MAX_FULL_RISK_TRADES": 3,
                "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.5,
            }),
            ("r5_capacity_5_full4_aggressive", {
                "param_overrides.MAX_TRADES_PER_DAY": 5,
                "param_overrides.MAX_FULL_RISK_TRADES": 4,
                "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.5,
            }),
            ("r5_ttl_retest90", {"param_overrides.ENTRY_TTL_RETEST_MINUTES": 90}),
            ("r5_ttl_retest150", {"param_overrides.ENTRY_TTL_RETEST_MINUTES": 150}),
            ("r5_target_room075_quality", {
                "param_overrides.TARGET_ROOM_MIN_R": 0.75,
                "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": 0.75,
            }),
        ]
    if phase == 3:
        return [
            ("r5_sw_led_room3_quality", _merge(SECOND_WIND_CORE, {
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.55,
            })),
            ("r5_sw_pm_score060", {
                "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": 0.60,
            }),
            ("r5_sw_trigger_close_conversion", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "trigger_close",
                "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": 20,
            }),
            ("r5_sw_ema_pullback_ttl120", {
                "param_overrides.SECOND_WIND_ENTRY_MODEL": "ema_pullback",
                "param_overrides.ENTRY_TTL_RETEST_MINUTES": 120,
            }),
            ("r5_sw_enable_second_leg_strict", {
                "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_SCORE": 9,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE": 0.60,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE": 1.20,
                "param_overrides.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION": 0.70,
                "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE": True,
            }),
            ("r5_sw_keep_micro_range_prune_vwap", {
                "param_overrides.SECOND_WIND_VWAP_RECLAIM_ENABLED": False,
                "param_overrides.SECOND_WIND_MICRO_COMPRESSION_ENABLED": True,
                "param_overrides.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED": True,
            }),
            ("r5_pm_reversion_guard_score9", {
                "param_overrides.REVERSION_MIN_SCORE": 9,
                "param_overrides.REVERSION_A_SCORE": 9,
                "param_overrides.ALLOW_LATE_PM_REVERSION": True,
            }),
        ]
    if phase == 4:
        return [
            ("r5_struct_adaptive_core", STRUCTURAL_CORE),
            ("r5_struct_hybrid_close_score8", _merge(STRUCTURAL_CORE, {
                "param_overrides.STRUCTURAL_ENTRY_MODE": "hybrid_close_adaptive",
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 45.0,
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 45.0,
            })),
            ("r5_struct_close70", {
                "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.70,
            }),
            ("r5_struct_stop45", {
                "param_overrides.STRUCTURAL_MAX_STOP_PTS": 45.0,
                "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 45.0,
            }),
            ("r5_struct_pullback_require_trend", {
                "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_REQUIRE_TREND": True,
                "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": 1.25,
            }),
            ("r5_struct_floor_0p5_lock0p2", {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
            }),
            ("r5_struct_no_minmicro_quality", {
                "param_overrides.STRUCTURAL_ALLOW_MIN_MICRO_SIZE": False,
                "param_overrides.STRUCTURAL_MIN_SCORE": 9,
            }),
            ("r5_struct_wide_quicker_t1", {
                "param_overrides.STRUCTURAL_WIDE_TARGET1_R": 0.35,
                "param_overrides.STRUCTURAL_WIDE_TARGET2_R": 0.80,
            }),
        ]
    if phase == 5:
        return [
            ("r5_reversion_core_touch", REVERSION_CORE),
            ("r5_reversion_swing_broad", _merge(REVERSION_CORE, {
                "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 48,
                "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 6,
            })),
            ("r5_reversion_score9_room05", {
                "param_overrides.REVERSION_MIN_SCORE": 9,
                "param_overrides.REVERSION_A_SCORE": 9,
                "param_overrides.TARGET_ROOM_MIN_R": 0.50,
            }),
            ("r5_reversion_tight_stop8", {
                "param_overrides.REVERSION_STANDARD_STOP_CAP": 8.0,
                "param_overrides.REVERSION_A_PLUS_STOP_CAP": 10.0,
            }),
            ("r5_reversion_adaptive_market", {
                "param_overrides.REVERSION_ENTRY_MODEL": "adaptive_reclaim_retest",
                "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_SCORE": 10,
                "param_overrides.REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS": 6.0,
                "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R": 1.5,
            }),
            ("r5_reversion_reaction_floor", {
                "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": False,
                "param_overrides.REVERSION_VWAP_REACTION_EXIT_ENABLED": True,
                "param_overrides.REVERSION_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.REVERSION_PROFIT_FLOOR_TRIGGER_R": 0.75,
                "param_overrides.REVERSION_PROFIT_FLOOR_LOCK_R": 0.25,
            }),
            ("r5_reversion_value_factor1", {
                "param_overrides.REVERSION_MIN_VALUE_FACTORS": 1,
                "param_overrides.REVERSION_MAX_PENETRATION_PTS": 10.0,
            }),
        ]
    if phase == 6:
        return [
            ("r5_global_floor_0p5_lock0p25", {
                "param_overrides.PROFIT_FLOOR_ENABLED": True,
                "param_overrides.PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.PROFIT_FLOOR_LOCK_R": 0.25,
            }),
            ("r5_global_ratchet_1p0_70pct", {
                "param_overrides.MFE_RATCHET_ENABLED": True,
                "param_overrides.MFE_RATCHET_TRIGGER_R": 1.00,
                "param_overrides.MFE_RATCHET_FLOOR_PCT": 0.70,
            }),
            ("r5_second_wind_floor_1p0_lock0p5", {
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 1.00,
                "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.50,
            }),
            ("r5_second_wind_ratchet_1p5_50pct", {
                "param_overrides.SECOND_WIND_MFE_RATCHET_ENABLED": True,
                "param_overrides.SECOND_WIND_MFE_RATCHET_TRIGGER_R": 1.50,
                "param_overrides.SECOND_WIND_MFE_RATCHET_FLOOR_PCT": 0.50,
            }),
            ("r5_reversion_touch_plus_runner", {
                "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": True,
                "param_overrides.REVERSION_TARGET1_QTY_FRACTION": 0.30,
                "param_overrides.REVERSION_TARGET2_QTY_FRACTION": 0.20,
                "param_overrides.REVERSION_MFE_RATCHET_ENABLED": True,
                "param_overrides.REVERSION_MFE_RATCHET_TRIGGER_R": 1.25,
                "param_overrides.REVERSION_MFE_RATCHET_FLOOR_PCT": 0.55,
            }),
            ("r5_structural_ratchet_1p0_70pct", {
                "param_overrides.STRUCTURAL_MFE_RATCHET_ENABLED": True,
                "param_overrides.STRUCTURAL_MFE_RATCHET_TRIGGER_R": 1.00,
                "param_overrides.STRUCTURAL_MFE_RATCHET_FLOOR_PCT": 0.70,
            }),
            ("r5_unified_time_stops", {
                "param_overrides.REVERSION_TIME_STOP_ENABLED": True,
                "param_overrides.REVERSION_TIME_STOP_BARS": 8,
                "param_overrides.REVERSION_TIME_STOP_MIN_MFE_R": 0.50,
                "param_overrides.SECOND_WIND_TIME_STOP_ENABLED": True,
                "param_overrides.SECOND_WIND_TIME_STOP_BARS": 4,
                "param_overrides.SECOND_WIND_TIME_STOP_MIN_MFE_R": 0.50,
            }),
        ]
    if phase == 7:
        return [
            ("r5_final_frequency_stack", _merge(
                ROUTER_GLOBAL_LED,
                STRUCTURAL_CORE,
                REVERSION_CORE,
                SECOND_WIND_CORE,
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 5,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.5,
                },
            )),
            ("r5_final_balanced_stack", _merge(
                ROUTER_GLOBAL_LED,
                STRUCTURAL_CORE,
                REVERSION_CORE,
                SECOND_WIND_CORE,
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 4,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.TARGET_ROOM_MIN_R": 0.50,
                },
            )),
            ("r5_final_quality_stack", _merge(
                ROUTER_SW_LED_ONLY,
                STRUCTURAL_CORE,
                REVERSION_CORE,
                SECOND_WIND_CORE,
                {
                    "param_overrides.STRUCTURAL_MIN_SCORE": 9,
                    "param_overrides.REVERSION_MIN_SCORE": 9,
                    "param_overrides.REVERSION_A_SCORE": 9,
                    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                    "param_overrides.MAX_TRADES_PER_DAY": 4,
                    "param_overrides.MAX_FULL_RISK_TRADES": 2,
                },
            )),
            ("r5_final_reversion_capacity_with_module_guards", _merge(
                ROUTER_GLOBAL_LED,
                REVERSION_CORE,
                {
                    "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 48,
                    "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 6,
                    "param_overrides.MAX_TRADES_PER_DAY": 5,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                },
            )),
            ("r5_final_pm_alpha_protection", _merge(
                SECOND_WIND_CORE,
                {
                    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 3.0,
                    "param_overrides.SECOND_WIND_MIN_PM_SCORE": 0.60,
                    "param_overrides.SECOND_WIND_PROFIT_FLOOR_ENABLED": True,
                    "param_overrides.SECOND_WIND_PROFIT_FLOOR_TRIGGER_R": 1.00,
                    "param_overrides.SECOND_WIND_PROFIT_FLOOR_LOCK_R": 0.50,
                },
            )),
            ("r5_final_structural_protection", {
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_ENABLED": True,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_TRIGGER_R": 0.50,
                "param_overrides.STRUCTURAL_PROFIT_FLOOR_LOCK_R": 0.20,
                "param_overrides.STRUCTURAL_WIDE_TARGET1_R": 0.35,
                "param_overrides.STRUCTURAL_WIDE_TARGET2_R": 0.80,
            }),
            ("r5_final_exit_capture_stack", {
                "param_overrides.PROFIT_FLOOR_ENABLED": True,
                "param_overrides.PROFIT_FLOOR_TRIGGER_R": 0.75,
                "param_overrides.PROFIT_FLOOR_LOCK_R": 0.25,
                "param_overrides.MFE_RATCHET_ENABLED": True,
                "param_overrides.MFE_RATCHET_TRIGGER_R": 1.0,
                "param_overrides.MFE_RATCHET_FLOOR_PCT": 0.65,
                "param_overrides.REVERSION_VWAP_TOUCH_EXIT_ENABLED": True,
                "param_overrides.SECOND_WIND_EMA_TRAIL_EXIT_ENABLED": True,
            }),
        ]
    return []
