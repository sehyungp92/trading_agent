"""Downturn experiment categories — ~80 experiments in 6 categories.

Each experiment is a (name, mutations_dict) tuple.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# REGIME (~12)
# ---------------------------------------------------------------------------

def _regime_experiments() -> list[tuple[str, dict]]:
    exps = []
    # ADX trending threshold
    for v in [18, 20, 22, 28, 30]:
        exps.append((f"regime_adx_trending_{v}", {"param_overrides.adx_trending_threshold": v}))
    # ADX range threshold
    for v in [10, 12, 18, 20]:
        exps.append((f"regime_adx_range_{v}", {"param_overrides.adx_range_threshold": v}))
    # EMA fast/slow periods
    for v in [10, 15, 30]:
        exps.append((f"regime_ema_fast_{v}", {"param_overrides.ema_fast_period": v}))
    for v in [30, 40, 60, 80]:
        exps.append((f"regime_ema_slow_{v}", {"param_overrides.ema_slow_period": v}))
    # SMA200
    for v in [150, 180, 250]:
        exps.append((f"regime_sma200_{v}", {"param_overrides.sma200_period": v}))
    return exps


# ---------------------------------------------------------------------------
# REVERSAL_SIGNAL (~10)
# ---------------------------------------------------------------------------

def _reversal_signal_experiments() -> list[tuple[str, dict]]:
    exps = []
    # Divergence magnitude threshold
    for v in [0.08, 0.10, 0.12, 0.20, 0.25]:
        exps.append((f"rev_div_mag_{v}", {"param_overrides.divergence_mag_threshold": v}))
    # Corridor cap multiplier
    for v in [1.5, 1.8, 2.5, 3.0]:
        exps.append((f"rev_corridor_{v}", {"param_overrides.corridor_cap_mult": v}))
    # Vol coil ratio threshold
    for v in [0.60, 0.70, 0.80, 0.90]:
        exps.append((f"rev_volcoil_ratio_{v}", {
            "param_overrides.vol_coil_ratio_threshold": v,
        }))
    # Disable 2-of-3 gate (relax to 1-of-3)
    exps.append(("rev_disable_trend_weakness", {"flags.reversal_trend_weakness_gate": False}))
    # Disable reversal entirely
    exps.append(("rev_disable_engine", {"flags.reversal_engine": False}))
    return exps


# ---------------------------------------------------------------------------
# BREAKDOWN_SIGNAL (~12)
# ---------------------------------------------------------------------------

def _breakdown_signal_experiments() -> list[tuple[str, dict]]:
    exps = []
    # Containment threshold
    for v in [0.70, 0.75, 0.82, 0.85, 0.90]:
        exps.append((f"bd_contain_{v}", {"param_overrides.box_containment_min": v}))
    # Box width min mult
    for v in [0.30, 0.40, 0.60, 0.75]:
        exps.append((f"bd_width_mult_{v}", {"param_overrides.box_width_min_mult": v}))
    # Displacement quantile
    for v in [0.55, 0.60, 0.65, 0.75, 0.80]:
        exps.append((f"bd_disp_q_{v}", {"param_overrides.displacement_quantile": v}))
    # Adaptive L bucket thresholds
    for low, mid, high in [(15, 25, 40), (18, 28, 44), (22, 36, 52)]:
        exps.append((f"bd_L_{low}_{mid}_{high}", {
            "param_overrides.box_l_low": low,
            "param_overrides.box_l_mid": mid,
            "param_overrides.box_l_high": high,
        }))
    # Disable spike reject
    exps.append(("bd_no_spike_reject", {"flags.breakdown_spike_reject": False}))
    # Disable chop filter
    exps.append(("bd_no_chop", {"flags.breakdown_chop_filter": False}))
    # Disable breakdown entirely
    exps.append(("bd_disable_engine", {"flags.breakdown_engine": False}))
    return exps


# ---------------------------------------------------------------------------
# FADE_SIGNAL (~10)
# ---------------------------------------------------------------------------

def _fade_signal_experiments() -> list[tuple[str, dict]]:
    exps = []
    # VWAP cap values
    for core, ext in [(0.20, 0.35), (0.25, 0.40), (0.35, 0.55), (0.40, 0.65)]:
        exps.append((f"fade_cap_{core}_{ext}", {
            "param_overrides.vwap_cap_core": core,
            "param_overrides.vwap_cap_extended": ext,
        }))
    # Momentum slope lookback
    for v in [2, 4, 5]:
        exps.append((f"fade_mom_lb_{v}", {"param_overrides.mom_slope_lookback": v}))
    # Rejection lookback bars
    for v in [5, 6, 10, 12]:
        exps.append((f"fade_rej_lb_{v}", {"param_overrides.rejection_lookback_bars": v}))
    # Disable momentum confirm
    exps.append(("fade_no_momentum", {"flags.fade_momentum_confirm": False}))
    # Disable bear regime required
    exps.append(("fade_no_bear_req", {"flags.fade_bear_regime_required": False}))
    # Disable fade entirely
    exps.append(("fade_disable_engine", {"flags.fade_engine": False}))
    return exps


# ---------------------------------------------------------------------------
# EXIT (~18)
# ---------------------------------------------------------------------------

def _exit_experiments() -> list[tuple[str, dict]]:
    exps = []
    # TP R-levels for aligned bear
    for tp1 in [1.0, 1.2, 1.8, 2.0]:
        exps.append((f"exit_tp1_aligned_{tp1}", {"param_overrides.tp1_r_aligned": tp1}))
    for tp2 in [2.0, 2.5, 3.5, 4.0]:
        exps.append((f"exit_tp2_aligned_{tp2}", {"param_overrides.tp2_r_aligned": tp2}))
    for tp3 in [4.0, 5.0, 6.0, 8.0]:
        exps.append((f"exit_tp3_aligned_{tp3}", {"param_overrides.tp3_r_aligned": tp3}))
    # TP R-levels for emerging bear (36 trades = dominant regime)
    for tp1 in [0.75, 0.80, 1.2, 1.5]:
        exps.append((f"exit_tp1_emerg_{tp1}", {"param_overrides.tp1_r_emerging": tp1}))
    for tp2 in [1.5, 1.8, 2.5, 3.0]:
        exps.append((f"exit_tp2_emerg_{tp2}", {"param_overrides.tp2_r_emerging": tp2}))
    # TP percentage splits for emerging bear
    for pct in [0.25, 0.30, 0.40, 0.50]:
        exps.append((f"exit_tp1pct_emerg_{pct}", {"param_overrides.tp1_pct_emerging": pct}))
    # Chandelier lookback and base mult
    for v in [10, 12, 18, 20]:
        exps.append((f"exit_chand_lb_{v}", {"param_overrides.chandelier_lookback": v}))
    # Chandelier mult floor/ceiling (controls trailing tightness)
    for v in [1.5, 1.8, 2.2, 2.5]:
        exps.append((f"exit_chand_floor_{v}", {"param_overrides.chandelier_mult_floor": v}))
    for v in [3.0, 3.5, 4.5, 5.0]:
        exps.append((f"exit_chand_ceil_{v}", {"param_overrides.chandelier_mult_ceiling": v}))
    # BE stop buffer (currently 0.20 ATR — too tight?)
    for v in [0.10, 0.15, 0.30, 0.40, 0.50]:
        exps.append((f"exit_be_buffer_{v}", {"param_overrides.be_stop_buffer_mult": v}))
    # Profit floor trail (lock fraction of profit)
    for lock in [0.30, 0.40, 0.50, 0.60]:
        for thresh in [0.3, 0.5, 0.75]:
            exps.append((f"exit_pfloor_{lock}_{thresh}", {
                "flags.profit_floor_trail": True,
                "param_overrides.profit_floor_lock_pct": lock,
                "param_overrides.profit_floor_r_threshold": thresh,
            }))
    # Stale exit bar counts
    for v in [20, 24, 32, 36]:
        exps.append((f"exit_stale_fade_{v}", {"param_overrides.stale_bars_fade": v}))
    for v in [8, 10, 16, 20]:
        exps.append((f"exit_stale_bd_{v}", {"param_overrides.stale_bars_breakdown": v}))
    for v in [8, 10, 16, 20]:
        exps.append((f"exit_stale_rev_{v}", {"param_overrides.stale_bars_reversal": v}))
    # Climax mult
    for v in [2.0, 2.2, 3.0, 3.5]:
        exps.append((f"exit_climax_{v}", {"param_overrides.climax_mult": v}))
    # Disable tiered exits
    exps.append(("exit_no_tiered", {"flags.tiered_exits": False}))
    # Disable chandelier
    exps.append(("exit_no_chandelier", {"flags.chandelier_trailing": False}))
    # Disable stale
    exps.append(("exit_no_stale", {"flags.stale_exit": False}))
    # Disable climax
    exps.append(("exit_no_climax", {"flags.climax_exit": False}))
    return exps


# ---------------------------------------------------------------------------
# SIZING (~12)
# ---------------------------------------------------------------------------

def _sizing_experiments() -> list[tuple[str, dict]]:
    exps = []
    # Base risk pct
    for v in [0.005, 0.0075, 0.012, 0.015, 0.02]:
        exps.append((f"size_risk_{v}", {"param_overrides.base_risk_pct": v}))
    # Vol state boundaries
    for v in [0.75, 0.82, 0.88]:
        exps.append((f"size_high_vol_pctl_{v}", {"param_overrides.high_vol_atr_pctl": v}))
    for v in [0.90, 0.93, 0.97]:
        exps.append((f"size_shock_pctl_{v}", {"param_overrides.shock_atr_pctl": v}))
    # Regime sizing multipliers
    for regime, key in [
        ("aligned", "regime_mult_aligned"),
        ("emerging", "regime_mult_emerging"),
        ("neutral", "regime_mult_neutral"),
        ("counter", "regime_mult_counter"),
        ("range", "regime_mult_range"),
    ]:
        for mult in [0.5, 0.75, 1.0, 1.25]:
            exps.append((f"size_{regime}_{mult}", {"param_overrides." + key: mult}))
    # Circuit breaker threshold
    for v in [-2000, -4000, -5000]:
        exps.append((f"size_cb_{abs(v)}", {"param_overrides.circuit_breaker_threshold": v}))
    # Disable vol states
    exps.append(("size_no_vol_states", {"flags.use_volatility_states": False}))
    # Disable strong bear bonus
    exps.append(("size_no_strong_bonus", {"flags.use_strong_bear_bonus": False}))
    return exps


# ---------------------------------------------------------------------------
# FREQUENCY & REGIME FILTERS (~20)
# ---------------------------------------------------------------------------

def _frequency_experiments() -> list[tuple[str, dict]]:
    """Experiments to boost trade frequency and improve correction bias."""
    exps = []
    # Block counter-regime trades (consistently lose money)
    exps.append(("freq_block_counter", {"flags.block_counter_regime": True}))
    # Correction sizing bonus
    for mult in [1.15, 1.25, 1.40, 1.50]:
        exps.append((f"freq_corr_bonus_{mult}", {
            "flags.correction_sizing_bonus": True,
            "param_overrides.correction_sizing_mult": mult,
        }))
    # Non-correction sizing penalty
    for mult in [0.40, 0.50, 0.60, 0.75]:
        exps.append((f"freq_non_corr_{mult}", {
            "flags.non_correction_penalty": True,
            "param_overrides.non_correction_sizing_mult": mult,
        }))
    # Combined: block counter + correction bonus
    exps.append(("freq_counter_block_plus_corr_bonus", {
        "flags.block_counter_regime": True,
        "flags.correction_sizing_bonus": True,
        "param_overrides.correction_sizing_mult": 1.30,
    }))
    # Combined: correction bonus + non-correction penalty (maximize correction bias)
    for bonus, penalty in [(1.25, 0.60), (1.40, 0.50), (1.50, 0.40)]:
        exps.append((f"freq_corr_bias_{bonus}_{penalty}", {
            "flags.correction_sizing_bonus": True,
            "flags.non_correction_penalty": True,
            "param_overrides.correction_sizing_mult": bonus,
            "param_overrides.non_correction_sizing_mult": penalty,
        }))
    # Allow fade during NEUTRAL regime (expands signal universe)
    exps.append(("freq_fade_allow_neutral", {"flags.fade_bear_regime_required": False}))
    # Relax fade cap to widen entry zone (more signals)
    for core, ext in [(0.45, 0.70), (0.50, 0.80)]:
        exps.append((f"freq_fade_wide_{core}_{ext}", {
            "param_overrides.vwap_cap_core": core,
            "param_overrides.vwap_cap_extended": ext,
        }))
    return exps


# ---------------------------------------------------------------------------
# STRUCTURAL (~18) — fixes for structural gaps (correction coverage, regime gates)
# ---------------------------------------------------------------------------

def _structural_experiments() -> list[tuple[str, dict]]:
    """Experiments targeting structural gaps: correction coverage, regime gates, exits."""
    exps = []

    # Correction-window regime override (P0 — highest impact)
    exps.append(("struct_corr_override", {"flags.correction_regime_override": True}))

    # Short-SMA alternative trend signal (P1)
    for period in [30, 50, 75]:
        exps.append((f"struct_short_sma_{period}", {
            "flags.short_sma_trend": True,
            "param_overrides.short_sma_period": period,
        }))

    # Allow reversal during strong bear (P2)
    exps.append(("struct_rev_strong_bear", {"flags.allow_reversal_strong_bear": True}))

    # Post-TP1 chandelier widening (P2)
    for mult in [3.0, 3.5, 4.0, 5.0]:
        exps.append((f"struct_post_tp1_chand_{mult}", {
            "param_overrides.post_tp1_chandelier_mult": mult,
        }))

    # Stale-to-TP cosmetic fix (P3)
    exps.append(("struct_stale_to_tp", {"flags.stale_to_tp": True}))

    # Combos: correction override + short SMA
    for period in [30, 50, 75]:
        exps.append((f"struct_corr_override_sma_{period}", {
            "flags.correction_regime_override": True,
            "flags.short_sma_trend": True,
            "param_overrides.short_sma_period": period,
        }))

    # Combo: correction override + reversal strong bear
    exps.append(("struct_corr_override_rev_sb", {
        "flags.correction_regime_override": True,
        "flags.allow_reversal_strong_bear": True,
    }))

    # Combos: correction override + post-TP1 widening
    for mult in [3.0, 3.5, 4.0, 5.0]:
        exps.append((f"struct_corr_override_tp1_{mult}", {
            "flags.correction_regime_override": True,
            "param_overrides.post_tp1_chandelier_mult": mult,
        }))

    return exps


# ---------------------------------------------------------------------------
# FAST_CRASH (~8) -- fast-crash override paths E/F/G
# ---------------------------------------------------------------------------

def _fast_crash_experiments() -> list[tuple[str, dict]]:
    """Experiments for real-time crash detection overriding composite regime."""
    exps = []
    # Daily drop thresholds (Path F/E)
    for v in [-0.015, -0.020, -0.025]:
        exps.append((f"fc_daily_{abs(v):.3f}", {
            "flags.fast_crash_override": True,
            "param_overrides.crash_daily_threshold": v,
        }))
    # Cumulative decline variants (Path G) — include looser thresholds for mild corrections
    for period, thresh in [(3, -0.02), (5, -0.02), (5, -0.025), (5, -0.03), (5, -0.04)]:
        exps.append((f"fc_cum_{period}d_{abs(thresh):.3f}", {
            "flags.fast_crash_override": True,
            "param_overrides.crash_cumulative_period": period,
            "param_overrides.crash_cumulative_threshold": thresh,
        }))
    # Combos with conviction scoring (include low thresholds for mild corrections)
    for daily, conv in [(-0.020, 50), (-0.025, 40), (-0.015, 30), (-0.020, 20)]:
        exps.append((f"fc_daily_{abs(daily):.3f}_conv{conv}", {
            "flags.fast_crash_override": True,
            "flags.conviction_scoring": True,
            "param_overrides.crash_daily_threshold": daily,
            "param_overrides.conviction_threshold": conv,
        }))
    return exps


# ---------------------------------------------------------------------------
# CONVICTION (~6) -- bear conviction quality gate
# ---------------------------------------------------------------------------

def _conviction_experiments() -> list[tuple[str, dict]]:
    """Experiments for bear conviction scoring as quality gate."""
    exps = []
    # Standalone conviction thresholds (include low thresholds for mild corrections)
    for v in [20, 30, 40, 50, 60]:
        exps.append((f"conv_{v}", {
            "flags.conviction_scoring": True,
            "param_overrides.conviction_threshold": v,
        }))
    # Conviction + correction override
    exps.append(("conv_50_corr_override", {
        "flags.conviction_scoring": True,
        "flags.correction_regime_override": True,
        "param_overrides.conviction_threshold": 50,
    }))
    # Conviction + fast crash (include low thresholds)
    for v in [20, 30, 40, 50]:
        exps.append((f"conv_{v}_fc", {
            "flags.conviction_scoring": True,
            "flags.fast_crash_override": True,
            "param_overrides.conviction_threshold": v,
        }))
    return exps


# ---------------------------------------------------------------------------
# EXIT_V2 (~12) -- multi-tier profit floor, BE trigger, regime chandelier, scale-out
# ---------------------------------------------------------------------------

def _exit_v2_experiments() -> list[tuple[str, dict]]:
    """Exit enhancements."""
    exps = []
    # Multi-tier profit floor with scales
    for scale in [0.8, 1.0, 1.2, 1.5]:
        exps.append((f"ev2_mtpf_{scale}", {
            "flags.multi_tier_profit_floor": True,
            "param_overrides.profit_floor_scale": scale,
        }))
    # Earlier BE trigger
    for v in [0.50, 0.75]:
        exps.append((f"ev2_be_{v:.2f}", {
            "param_overrides.be_trigger_r": v,
        }))
    # Regime-adaptive chandelier
    exps.append(("ev2_regime_chand", {
        "flags.regime_adaptive_chandelier": True,
    }))
    exps.append(("ev2_regime_chand_wide", {
        "flags.regime_adaptive_chandelier": True,
        "param_overrides.chandelier_regime_mult_aligned": 1.25,
    }))
    # Scale-out variants
    for target, pct in [(2.5, 0.25), (3.0, 0.30), (3.5, 0.33), (4.0, 0.33)]:
        exps.append((f"ev2_scaleout_{target:.0f}_{int(pct*100)}", {
            "flags.scale_out_enabled": True,
            "param_overrides.scale_out_target_r": target,
            "param_overrides.scale_out_pct": pct,
        }))
    return exps


# ---------------------------------------------------------------------------
# CORRECTION_REVERSAL (~4) -- selective reversal in corrections
# ---------------------------------------------------------------------------

def _correction_reversal_experiments() -> list[tuple[str, dict]]:
    """Experiments to revive reversal engine in correction windows."""
    exps = []
    exps.append(("cr_rev_in_corr", {
        "flags.allow_reversal_in_correction": True,
    }))
    exps.append(("cr_rev_in_corr_fc", {
        "flags.allow_reversal_in_correction": True,
        "flags.fast_crash_override": True,
    }))
    exps.append(("cr_rev_in_corr_conv", {
        "flags.allow_reversal_in_correction": True,
        "flags.conviction_scoring": True,
        "param_overrides.conviction_threshold": 50,
    }))
    exps.append(("cr_rev_in_corr_override", {
        "flags.allow_reversal_in_correction": True,
        "flags.correction_regime_override": True,
    }))
    return exps


# ---------------------------------------------------------------------------
# BEAR_STRUCTURE -- ADX hysteresis + paths B/C + BEAR_FORMING
# ---------------------------------------------------------------------------

def _bear_structure_experiments() -> list[tuple[str, dict]]:
    """Bear structure override experiments (gradual correction detection)."""
    exps = []

    # Standalone bear_structure with different min_conditions
    for min_c in [2, 3]:
        exps.append((f"bs_min{min_c}", {
            "flags.bear_structure_override": True,
            "param_overrides.bear_structure_min_conditions": min_c,
        }))

    # ADX hysteresis thresholds
    for on, off in [(20, 12), (25, 15), (30, 18)]:
        exps.append((f"bs_adx_{on}_{off}", {
            "flags.bear_structure_override": True,
            "param_overrides.bear_structure_adx_on": on,
            "param_overrides.bear_structure_adx_off": off,
        }))

    # Path B conviction variants
    for conv in [40, 50, 60]:
        exps.append((f"bs_pathB_conv{conv}", {
            "flags.bear_structure_override": True,
            "param_overrides.bear_structure_path_b_conviction": conv,
        }))

    # Path C structural variants
    for di, sep in [(6, 0.10), (8, 0.15), (10, 0.20)]:
        exps.append((f"bs_pathC_di{di}_sep{sep}", {
            "flags.bear_structure_override": True,
            "param_overrides.bear_structure_path_c_di_gap": di,
            "param_overrides.bear_structure_path_c_ema_sep": sep,
        }))

    # Combined with fast_crash
    for crash_t in [-0.015, -0.020]:
        exps.append((f"bs_fc_{abs(crash_t)}", {
            "flags.bear_structure_override": True,
            "flags.fast_crash_override": True,
            "param_overrides.crash_mild_threshold": crash_t,
        }))

    # Combined with conviction scoring
    for conv in [30, 40, 50]:
        exps.append((f"bs_conv{conv}", {
            "flags.bear_structure_override": True,
            "flags.conviction_scoring": True,
            "param_overrides.conviction_threshold": conv,
        }))

    # Combined with correction_regime_override (all three layers)
    exps.append(("bs_corr_override", {
        "flags.bear_structure_override": True,
        "flags.correction_regime_override": True,
    }))

    # All three override layers + fast_crash
    exps.append(("bs_fc_corr_all", {
        "flags.bear_structure_override": True,
        "flags.fast_crash_override": True,
        "flags.correction_regime_override": True,
    }))

    return exps


# ---------------------------------------------------------------------------
# EXIT_ADAPTIVE (~12) — R6 MFE-tiered adaptive profit floor lock
# ---------------------------------------------------------------------------

def _exit_adaptive_experiments() -> list[tuple[str, dict]]:
    """R6: Adaptive lock_pct that increases capture from big MFE winners."""
    exps = []
    # Different MFE tier thresholds
    for t1 in [1.5, 2.0, 3.0]:
        exps.append((f"ea_t1_{t1}", {
            "flags.adaptive_profit_floor": True,
            "param_overrides.adaptive_lock_t1": t1,
        }))
    # Different bonus amounts
    for b1 in [0.05, 0.15, 0.20]:
        exps.append((f"ea_b1_{int(b1*100):02d}", {
            "flags.adaptive_profit_floor": True,
            "param_overrides.adaptive_lock_bonus_1": b1,
        }))
    # Aggressive capture at high MFE
    exps.append(("ea_high_cap", {
        "flags.adaptive_profit_floor": True,
        "param_overrides.adaptive_lock_bonus_3": 0.30,
    }))
    # Combined with different base lock_pct
    for base in [0.50, 0.60]:
        exps.append((f"ea_base_{int(base*100)}", {
            "flags.adaptive_profit_floor": True,
            "param_overrides.profit_floor_lock_pct": base,
        }))
    # Combined with higher threshold (let winners develop)
    for thresh in [1.0, 1.5]:
        exps.append((f"ea_thresh_{thresh}", {
            "flags.adaptive_profit_floor": True,
            "param_overrides.profit_floor_r_threshold": thresh,
        }))
    # Aggressive full package
    exps.append(("ea_aggressive", {
        "flags.adaptive_profit_floor": True,
        "param_overrides.adaptive_lock_bonus_1": 0.15,
        "param_overrides.adaptive_lock_bonus_2": 0.20,
        "param_overrides.adaptive_lock_bonus_3": 0.30,
    }))
    return exps


# ---------------------------------------------------------------------------
# DRAWDOWN_OVERRIDE (~8) — R6 real-time rolling-high drawdown regime override
# ---------------------------------------------------------------------------

def _drawdown_override_experiments() -> list[tuple[str, dict]]:
    """R6: Real-time drawdown override for bull-market corrections."""
    exps = []
    # Different thresholds
    for t in [0.02, 0.03, 0.04]:
        exps.append((f"dd_t{int(t*100):02d}", {
            "flags.drawdown_regime_override": True,
            "param_overrides.drawdown_threshold": t,
        }))
    # Different lookback
    for lb in [10, 15, 20]:
        exps.append((f"dd_lb{lb}", {
            "flags.drawdown_regime_override": True,
            "param_overrides.drawdown_lookback": lb,
        }))
    # Combined with existing overrides
    exps.append(("dd_fc", {
        "flags.drawdown_regime_override": True,
        "flags.fast_crash_override": True,
    }))
    exps.append(("dd_bs", {
        "flags.drawdown_regime_override": True,
        "flags.bear_structure_override": True,
    }))
    return exps


# ---------------------------------------------------------------------------
# PROGRESSIVE_SMA_V2 (~6) — R6 Rev2: direct regime override (not just daily_trend)
# ---------------------------------------------------------------------------

def _progressive_sma_experiments() -> list[tuple[str, dict]]:
    """R6 Rev2: Progressive SMA with direct regime override in warmup period."""
    exps = []
    # Standalone with different min bars
    for min_bars in [50, 100]:
        exps.append((f"psma_v2_{min_bars}", {
            "flags.progressive_sma": True,
            "param_overrides.progressive_sma_min": min_bars,
        }))
    # Combined with drawdown override
    exps.append(("psma_v2_dd", {
        "flags.progressive_sma": True,
        "flags.drawdown_regime_override": True,
    }))
    return exps


# ---------------------------------------------------------------------------
# MOMENTUM_SIGNAL_V2 (~6) — R6 Rev2: with cooldown gate to prevent displacement
# ---------------------------------------------------------------------------

def _momentum_signal_experiments() -> list[tuple[str, dict]]:
    """R6 Rev2: Momentum impulse with cooldown gate (prevents VWAP displacement)."""
    exps = []
    # Different cooldown values
    for cd in [24, 36, 48]:
        exps.append((f"ms_v2_cd{cd}", {
            "flags.momentum_signal": True,
            "param_overrides.momentum_cooldown_bars": cd,
        }))
    # Combined with drawdown override
    exps.append(("ms_v2_dd_cd36", {
        "flags.momentum_signal": True,
        "flags.drawdown_regime_override": True,
        "param_overrides.momentum_cooldown_bars": 36,
    }))
    # Combined with progressive SMA
    exps.append(("ms_v2_psma_cd36", {
        "flags.momentum_signal": True,
        "flags.progressive_sma": True,
        "param_overrides.momentum_cooldown_bars": 36,
    }))
    # All regime features
    exps.append(("ms_v2_all", {
        "flags.momentum_signal": True,
        "flags.drawdown_regime_override": True,
        "flags.progressive_sma": True,
        "param_overrides.momentum_cooldown_bars": 36,
    }))
    return exps


# ---------------------------------------------------------------------------
# HOLD_PERIOD (~6) — R6 minimum hold period to prevent scalping
# ---------------------------------------------------------------------------

def _hold_period_experiments() -> list[tuple[str, dict]]:
    """R6: Skip exits for first N bars to allow winners to develop."""
    exps = []
    for bars in [3, 6, 12, 24]:
        exps.append((f"hp_{bars}", {
            "flags.min_hold_period": True,
            "param_overrides.min_hold_bars": bars,
        }))
    # Combined with adaptive exit
    for bars in [6, 12]:
        exps.append((f"hp_{bars}_ea", {
            "flags.min_hold_period": True,
            "param_overrides.min_hold_bars": bars,
            "flags.adaptive_profit_floor": True,
        }))
    return exps


# ---------------------------------------------------------------------------
# TRAIL_REDESIGN (~12) — R6 Rev2: chandelier as primary trail + profit floor safety net
# ---------------------------------------------------------------------------

def _trail_redesign_experiments() -> list[tuple[str, dict]]:
    """R6 Rev2: Chandelier trailing as primary trail with adjusted profit floor."""
    exps = []
    # Chandelier ON, profit floor threshold raised (no longer primary exit)
    for thresh in [1.0, 1.5, 2.0]:
        exps.append((f"trail_chand_pf{thresh}", {
            "flags.chandelier_trailing": True,
            "param_overrides.profit_floor_r_threshold": thresh,
        }))
    # Chandelier + adaptive profit floor (big-winner capture)
    exps.append(("trail_chand_adapt", {
        "flags.chandelier_trailing": True,
        "flags.adaptive_profit_floor": True,
        "param_overrides.profit_floor_r_threshold": 1.5,
    }))
    # Chandelier + min hold (let trades develop before any trail kicks in)
    for bars in [6, 12, 24]:
        exps.append((f"trail_chand_hold{bars}", {
            "flags.chandelier_trailing": True,
            "flags.min_hold_period": True,
            "param_overrides.min_hold_bars": bars,
            "param_overrides.profit_floor_r_threshold": 1.5,
        }))
    # Full package: chandelier + adaptive + hold + regime-adaptive width
    for bars, thresh in [(6, 1.5), (12, 1.5), (24, 2.0)]:
        exps.append((f"trail_full_{bars}", {
            "flags.chandelier_trailing": True,
            "flags.adaptive_profit_floor": True,
            "flags.min_hold_period": True,
            "flags.regime_adaptive_chandelier": True,
            "param_overrides.min_hold_bars": bars,
            "param_overrides.profit_floor_r_threshold": thresh,
        }))
    # Chandelier lookback variants
    for lb in [8, 20]:
        exps.append((f"trail_chand_lb{lb}", {
            "flags.chandelier_trailing": True,
            "param_overrides.chandelier_lookback": lb,
            "param_overrides.profit_floor_r_threshold": 1.5,
        }))
    return exps


# ---------------------------------------------------------------------------
# R8 CATEGORY A: Intraday Regime Proxy (~6)
# ---------------------------------------------------------------------------

def _r8_intraday_regime_experiments() -> list[tuple[str, dict]]:
    """R8: Bypass daily-bar alignment lag using intraday bars for regime."""
    return [
        ("r8a_four_hour_only_regime", {"flags.four_hour_only_regime": True}),
        ("r8a_intraday_1h_ema_100", {
            "flags.intraday_regime_proxy": True,
            "flags.intraday_regime_ema_period": 100,
        }),
        ("r8a_intraday_1h_ema_50", {
            "flags.intraday_regime_proxy": True,
            "flags.intraday_regime_ema_period": 50,
        }),
        ("r8a_multi_tf_regime_vote", {"flags.multi_tf_regime_vote": True}),
        ("r8a_atr_expansion", {"flags.regime_proxy_atr_expansion": True}),
        ("r8a_correction_intraday", {"flags.correction_intraday_detect": True}),
    ]


# ---------------------------------------------------------------------------
# R8 CATEGORY B: Reversal Engine Revival (~6)
# ---------------------------------------------------------------------------

def _r8_reversal_revival_experiments() -> list[tuple[str, dict]]:
    """R8: Relax gates or use alternative timeframes to revive dead reversal engine."""
    return [
        ("r8b_reversal_1of3_gate", {"flags.reversal_min_gate_count": 1}),
        ("r8b_reversal_no_extension", {"flags.reversal_no_extension_gate": True}),
        ("r8b_reversal_wider_div_005", {"param_overrides.divergence_mag_threshold": 0.05}),
        ("r8b_reversal_wider_corridor_3", {"flags.reversal_wider_corridor": 3.0}),
        ("r8b_reversal_hourly_pivots", {"flags.reversal_hourly_pivots": True}),
        ("r8b_reversal_combined_relax", {
            "flags.reversal_min_gate_count": 1,
            "flags.reversal_no_extension_gate": True,
            "flags.reversal_wider_corridor": 3.0,
        }),
    ]


# ---------------------------------------------------------------------------
# R8 CATEGORY C: Entry Filters (~5)
# ---------------------------------------------------------------------------

def _r8_entry_filter_experiments() -> list[tuple[str, dict]]:
    """R8: Reduce 66.5% stop rate by filtering weak entries."""
    return [
        ("r8c_correction_only_mode", {"flags.correction_only_mode": True}),
        ("r8c_correction_only_fade", {"flags.correction_only_fade": True}),
        ("r8c_vol_pctl_40", {"flags.vol_percentile_gate": 40.0}),
        ("r8c_vol_pctl_60", {"flags.vol_percentile_gate": 60.0}),
        ("r8c_regime_confidence_40", {"flags.regime_confidence_gate": 40.0}),
    ]


# ---------------------------------------------------------------------------
# R8 CATEGORY D: Exit Improvements (~6)
# ---------------------------------------------------------------------------

def _r8_exit_improvement_experiments() -> list[tuple[str, dict]]:
    """R8: Reduce stop drag by adjusting stop mechanics."""
    return [
        ("r8d_wider_initial_stop_1_5x", {"flags.wider_initial_stop_mult": 1.5}),
        ("r8d_atr_scaled_stop", {"flags.atr_scaled_initial_stop": True}),
        ("r8d_faster_be_03r", {"param_overrides.be_trigger_r": 0.3}),
        ("r8d_slower_be_08r", {"param_overrides.be_trigger_r": 0.8}),
        ("r8d_partial_at_be_25pct", {"flags.partial_at_breakeven": 0.25}),
        ("r8d_time_stop_widening", {
            "flags.time_stop_widening": True,
            "flags.time_stop_widening_bars": 48,
        }),
    ]


# ---------------------------------------------------------------------------
# R2 COUNTER BLOCKING (~5) -- eliminate toxic counter-regime trades
# ---------------------------------------------------------------------------

def _r2_counter_blocking_experiments() -> list[tuple[str, dict]]:
    """R2: Direct counter-regime blocking and neutral damping."""
    return [
        ("r2_counter_mult_0", {"param_overrides.regime_mult_counter": 0.0}),
        ("r2_counter_mult_01", {"param_overrides.regime_mult_counter": 0.1}),
        ("r2_neutral_mult_0", {"param_overrides.regime_mult_neutral": 0.0}),
        ("r2_neutral_mult_025", {"param_overrides.regime_mult_neutral": 0.25}),
        ("r2_counter_neutral_zero", {
            "param_overrides.regime_mult_counter": 0.0,
            "param_overrides.regime_mult_neutral": 0.0,
        }),
    ]


# ---------------------------------------------------------------------------
# R2 REGIME COMBOS (~6) -- interaction combos for regime overhaul
# ---------------------------------------------------------------------------

def _r2_regime_combo_experiments() -> list[tuple[str, dict]]:
    """R2: Combined regime overhaul experiments."""
    return [
        ("r2_block_corr_override", {
            "flags.block_counter_regime": True,
            "flags.correction_regime_override": True,
        }),
        ("r2_block_bear_struct", {
            "flags.block_counter_regime": True,
            "flags.bear_structure_override": True,
        }),
        ("r2_block_fast_crash_015", {
            "flags.block_counter_regime": True,
            "flags.fast_crash_override": True,
            "param_overrides.crash_daily_threshold": -0.015,
        }),
        ("r2_full_regime_overhaul", {
            "flags.block_counter_regime": True,
            "flags.correction_regime_override": True,
            "flags.bear_structure_override": True,
            "flags.fast_crash_override": True,
            "flags.conviction_scoring": True,
            "flags.short_sma_trend": True,
            "param_overrides.conviction_threshold": 30,
        }),
        ("r2_block_plus_corr_sizing", {
            "flags.block_counter_regime": True,
            "flags.correction_sizing_bonus": True,
            "param_overrides.correction_sizing_mult": 1.30,
        }),
        ("r2_corr_only_plus_sizing", {
            "flags.correction_only_mode": True,
            "flags.correction_sizing_bonus": True,
            "param_overrides.correction_sizing_mult": 1.40,
        }),
    ]


# ---------------------------------------------------------------------------
# R2 REGIME PARAMS (~7) -- ADX/EMA parameter sweeps
# ---------------------------------------------------------------------------

def _r2_regime_param_experiments() -> list[tuple[str, dict]]:
    """R2: ADX and EMA parameter sweeps to fix regime classification."""
    return [
        ("r2_adx_trending_22", {"param_overrides.adx_trending_threshold": 22}),
        ("r2_adx_trending_25", {"param_overrides.adx_trending_threshold": 25}),
        ("r2_adx_trending_30", {"param_overrides.adx_trending_threshold": 30}),
        ("r2_adx_range_12", {"param_overrides.adx_range_threshold": 12}),
        ("r2_adx_range_15", {"param_overrides.adx_range_threshold": 15}),
        ("r2_ema_fast_10", {"param_overrides.ema_fast_period": 10}),
        ("r2_ema_fast_20", {"param_overrides.ema_fast_period": 20}),
    ]


# ---------------------------------------------------------------------------
# R3 TARGETED NEXT-ROUND EXPERIMENTS
# ---------------------------------------------------------------------------

def _r3_alpha_experiments() -> list[tuple[str, dict]]:
    """Target correction alpha coverage without changing fill semantics."""
    return [
        ("r3_dd_looser_2pct_lb10", {
            "flags.drawdown_regime_override": True,
            "param_overrides.drawdown_threshold": 0.02,
            "param_overrides.drawdown_lookback": 10,
        }),
        ("r3_dd_slower_3pct_lb15", {
            "flags.drawdown_regime_override": True,
            "param_overrides.drawdown_threshold": 0.03,
            "param_overrides.drawdown_lookback": 15,
        }),
        ("r3_fc_mild_015", {
            "flags.fast_crash_override": True,
            "param_overrides.crash_daily_threshold": -0.015,
        }),
        ("r3_fc_conv_020_30", {
            "flags.fast_crash_override": True,
            "flags.conviction_scoring": True,
            "param_overrides.crash_daily_threshold": -0.020,
            "param_overrides.conviction_threshold": 30,
        }),
        ("r3_bs_min2", {
            "flags.bear_structure_override": True,
            "param_overrides.bear_structure_min_conditions": 2,
        }),
        ("r3_bs_adx_20_12", {
            "flags.bear_structure_override": True,
            "param_overrides.bear_structure_adx_on": 20,
            "param_overrides.bear_structure_adx_off": 12,
        }),
        ("r3_psma_min50", {
            "flags.progressive_sma": True,
            "param_overrides.progressive_sma_min": 50,
        }),
        ("r3_psma_min100", {
            "flags.progressive_sma": True,
            "param_overrides.progressive_sma_min": 100,
        }),
        ("r3_momentum_roc003_cd24", {
            "flags.momentum_signal": True,
            "param_overrides.momentum_roc_threshold": -0.003,
            "param_overrides.momentum_cooldown_bars": 24,
        }),
        ("r3_momentum_roc006_cd36", {
            "flags.momentum_signal": True,
            "param_overrides.momentum_roc_threshold": -0.006,
            "param_overrides.momentum_cooldown_bars": 36,
        }),
        ("r3_fade_allow_neutral", {"flags.fade_bear_regime_required": False}),
        ("r3_fade_wide_045_070", {
            "param_overrides.vwap_cap_core": 0.45,
            "param_overrides.vwap_cap_extended": 0.70,
        }),
        ("r3_corr_bonus_125", {
            "flags.correction_sizing_bonus": True,
            "param_overrides.correction_sizing_mult": 1.25,
        }),
        ("r3_corr_bias_125_060", {
            "flags.correction_sizing_bonus": True,
            "flags.non_correction_penalty": True,
            "param_overrides.correction_sizing_mult": 1.25,
            "param_overrides.non_correction_sizing_mult": 0.60,
        }),
        ("r3_vol_gate_off", {"flags.vol_percentile_gate": 0.0}),
        ("r3_bd_revival_default", {"flags.breakdown_engine": True}),
        ("r3_bd_revival_balanced", {
            "flags.breakdown_engine": True,
            "param_overrides.box_containment_min": 0.70,
            "param_overrides.displacement_quantile": 0.55,
        }),
        ("r3_bd_revival_loose_nochop", {
            "flags.breakdown_engine": True,
            "flags.breakdown_chop_filter": False,
            "param_overrides.box_containment_min": 0.60,
            "param_overrides.displacement_quantile": 0.50,
        }),
    ]


def _r3_entry_experiments() -> list[tuple[str, dict]]:
    """Entry quality and execution experiments routed through neutral orders."""
    return [
        ("r3_entry_ttl_12", {"param_overrides.entry_ttl_bars": 12}),
        ("r3_entry_ttl_24", {"param_overrides.entry_ttl_bars": 24}),
        ("r3_entry_ttl_48", {"param_overrides.entry_ttl_bars": 48}),
        ("r3_entry_ttl_96", {"param_overrides.entry_ttl_bars": 96}),
        ("r3_entry_buffer_1", {"param_overrides.entry_buffer_ticks": 1}),
        ("r3_entry_buffer_3", {"param_overrides.entry_buffer_ticks": 3}),
        ("r3_entry_buffer_4", {"param_overrides.entry_buffer_ticks": 4}),
        ("r3_trigger_buffer_1", {"param_overrides.trigger_low_buffer_ticks": 1}),
        ("r3_trigger_buffer_3", {"param_overrides.trigger_low_buffer_ticks": 3}),
        ("r3_limit_offset_2", {"param_overrides.entry_limit_offset_ticks": 2}),
        ("r3_limit_offset_6", {"param_overrides.entry_limit_offset_ticks": 6}),
        ("r3_fade_stop_035", {"param_overrides.fade_stop_atr_mult": 0.35}),
        ("r3_fade_stop_065", {"param_overrides.fade_stop_atr_mult": 0.65}),
        ("r3_fade_stop_085", {"param_overrides.fade_stop_atr_mult": 0.85}),
        ("r3_rev_stop_050", {"param_overrides.reversal_stop_atr_mult": 0.50}),
        ("r3_rev_stop_100", {"param_overrides.reversal_stop_atr_mult": 1.00}),
        ("r3_bd_stop_005", {"param_overrides.breakdown_stop_atr_mult": 0.05}),
        ("r3_bd_stop_020", {"param_overrides.breakdown_stop_atr_mult": 0.20}),
        ("r3_max_daily_2", {"param_overrides.max_daily_entries": 2}),
        ("r3_max_daily_5", {"param_overrides.max_daily_entries": 5}),
        ("r3_max_daily_8", {"param_overrides.max_daily_entries": 8}),
        ("r3_friction_005", {"param_overrides.friction_min_atr_pctl": 0.05}),
        ("r3_friction_015", {"param_overrides.friction_min_atr_pctl": 0.15}),
        ("r3_friction_025", {"param_overrides.friction_min_atr_pctl": 0.25}),
        ("r3_vol_gate_30", {"flags.vol_percentile_gate": 30.0}),
        ("r3_vol_gate_50", {"flags.vol_percentile_gate": 50.0}),
        ("r3_entry_vol_gate_off", {"flags.vol_percentile_gate": 0.0}),
        ("r3_conviction_gate_30", {"flags.regime_confidence_gate": 30.0}),
        ("r3_conviction_gate_45", {"flags.regime_confidence_gate": 45.0}),
    ]


def _r3_exit_experiments() -> list[tuple[str, dict]]:
    """Target MFE capture without overfitting to individual exits."""
    return [
        ("r3_hold_8", {"flags.min_hold_period": True, "param_overrides.min_hold_bars": 8}),
        ("r3_hold_12", {"flags.min_hold_period": True, "param_overrides.min_hold_bars": 12}),
        ("r3_hold_18", {"flags.min_hold_period": True, "param_overrides.min_hold_bars": 18}),
        ("r3_hold_24", {"flags.min_hold_period": True, "param_overrides.min_hold_bars": 24}),
        ("r3_be_060", {"param_overrides.be_trigger_r": 0.60}),
        ("r3_be_100", {"param_overrides.be_trigger_r": 1.00}),
        ("r3_pf_threshold_100", {
            "flags.profit_floor_trail": True,
            "param_overrides.profit_floor_r_threshold": 1.00,
        }),
        ("r3_pf_threshold_180", {
            "flags.profit_floor_trail": True,
            "param_overrides.profit_floor_r_threshold": 1.80,
        }),
        ("r3_pf_threshold_220", {
            "flags.profit_floor_trail": True,
            "param_overrides.profit_floor_r_threshold": 2.20,
        }),
        ("r3_pf_lock_050", {
            "flags.profit_floor_trail": True,
            "param_overrides.profit_floor_lock_pct": 0.50,
        }),
        ("r3_pf_lock_070", {
            "flags.profit_floor_trail": True,
            "param_overrides.profit_floor_lock_pct": 0.70,
        }),
        ("r3_chand_lb12", {
            "flags.chandelier_trailing": True,
            "param_overrides.chandelier_lookback": 12,
        }),
        ("r3_chand_lb24", {
            "flags.chandelier_trailing": True,
            "param_overrides.chandelier_lookback": 24,
        }),
        ("r3_chand_floor_180", {
            "flags.chandelier_trailing": True,
            "param_overrides.chandelier_mult_floor": 1.80,
        }),
        ("r3_chand_floor_250", {
            "flags.chandelier_trailing": True,
            "param_overrides.chandelier_mult_floor": 2.50,
        }),
        ("r3_adaptive_high_cap", {
            "flags.adaptive_profit_floor": True,
            "param_overrides.adaptive_lock_bonus_3": 0.30,
        }),
        ("r3_scaleout_25_25", {
            "flags.scale_out_enabled": True,
            "param_overrides.scale_out_target_r": 2.5,
            "param_overrides.scale_out_pct": 0.25,
        }),
        ("r3_scaleout_35_30", {
            "flags.scale_out_enabled": True,
            "param_overrides.scale_out_target_r": 3.5,
            "param_overrides.scale_out_pct": 0.30,
        }),
        ("r3_stale_fade_24", {"param_overrides.stale_bars_fade": 24}),
        ("r3_stale_fade_40", {"param_overrides.stale_bars_fade": 40}),
    ]


def _r3_risk_experiments() -> list[tuple[str, dict]]:
    """Sizing/risk polish after entries and exits are selected."""
    return [
        ("r3_risk_0055", {"param_overrides.base_risk_pct": 0.0055}),
        ("r3_risk_0070", {"param_overrides.base_risk_pct": 0.0070}),
        ("r3_risk_0080", {"param_overrides.base_risk_pct": 0.0080}),
        ("r3_risk_0100", {"param_overrides.base_risk_pct": 0.0100}),
        ("r3_aligned_mult_100", {"param_overrides.regime_mult_aligned": 1.00}),
        ("r3_aligned_mult_125", {"param_overrides.regime_mult_aligned": 1.25}),
        ("r3_aligned_mult_150", {"param_overrides.regime_mult_aligned": 1.50}),
        ("r3_emerging_mult_075", {"param_overrides.regime_mult_emerging": 0.75}),
        ("r3_emerging_mult_100", {"param_overrides.regime_mult_emerging": 1.00}),
        ("r3_emerging_mult_125", {"param_overrides.regime_mult_emerging": 1.25}),
        ("r3_range_mult_075", {"param_overrides.regime_mult_range": 0.75}),
        ("r3_range_mult_100", {"param_overrides.regime_mult_range": 1.00}),
        ("r3_counter_mult_000", {"param_overrides.regime_mult_counter": 0.0}),
        ("r3_counter_mult_010", {"param_overrides.regime_mult_counter": 0.10}),
        ("r3_corr_bonus_140", {
            "flags.correction_sizing_bonus": True,
            "param_overrides.correction_sizing_mult": 1.40,
        }),
        ("r3_non_corr_penalty_050", {
            "flags.non_correction_penalty": True,
            "param_overrides.non_correction_sizing_mult": 0.50,
        }),
    ]


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

EXPERIMENT_CATEGORIES: dict[str, list[tuple[str, dict]]] = {
    "REGIME": _regime_experiments(),
    "REVERSAL_SIGNAL": _reversal_signal_experiments(),
    "BREAKDOWN_SIGNAL": _breakdown_signal_experiments(),
    "FADE_SIGNAL": _fade_signal_experiments(),
    "EXIT": _exit_experiments(),
    "SIZING": _sizing_experiments(),
    "FREQUENCY": _frequency_experiments(),
    "STRUCTURAL": _structural_experiments(),
    "FAST_CRASH": _fast_crash_experiments(),
    "CONVICTION": _conviction_experiments(),
    "EXIT_V2": _exit_v2_experiments(),
    "CORRECTION_REVERSAL": _correction_reversal_experiments(),
    "BEAR_STRUCTURE": _bear_structure_experiments(),
    # R6 categories
    "EXIT_ADAPTIVE": _exit_adaptive_experiments(),
    "DRAWDOWN_OVERRIDE": _drawdown_override_experiments(),
    "PROGRESSIVE_SMA": _progressive_sma_experiments(),
    "MOMENTUM_SIGNAL": _momentum_signal_experiments(),
    "HOLD_PERIOD": _hold_period_experiments(),
    # R6 Rev2 categories
    "TRAIL_REDESIGN": _trail_redesign_experiments(),
    # R8 categories
    "R8_INTRADAY_REGIME": _r8_intraday_regime_experiments(),
    "R8_REVERSAL_REVIVAL": _r8_reversal_revival_experiments(),
    "R8_ENTRY_FILTERS": _r8_entry_filter_experiments(),
    "R8_EXIT_IMPROVEMENTS": _r8_exit_improvement_experiments(),
    # R2 categories
    "R2_COUNTER_BLOCKING": _r2_counter_blocking_experiments(),
    "R2_REGIME_COMBOS": _r2_regime_combo_experiments(),
    "R2_REGIME_PARAMS": _r2_regime_param_experiments(),
    "R3_ALPHA": _r3_alpha_experiments(),
    "R3_ENTRY": _r3_entry_experiments(),
    "R3_EXIT": _r3_exit_experiments(),
    "R3_RISK": _r3_risk_experiments(),
}


def get_all_experiments() -> list[tuple[str, dict]]:
    """Return all experiments across all categories."""
    all_exps = []
    for cat_exps in EXPERIMENT_CATEGORIES.values():
        all_exps.extend(cat_exps)
    return all_exps


def get_category_experiments(categories: list[str]) -> list[tuple[str, dict]]:
    """Return experiments for specific categories."""
    exps = []
    for cat in categories:
        exps.extend(EXPERIMENT_CATEGORIES.get(cat, []))
    return exps
