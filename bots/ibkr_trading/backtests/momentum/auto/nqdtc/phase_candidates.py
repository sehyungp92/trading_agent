"""NQDTC targeted phased auto-optimization candidates.

The suite starts from the latest optimized config and probes structural levers
that map directly to the diagnostics: exit MFE giveback, score discrimination,
entry under-diversification, and frequency interactions.
"""
from __future__ import annotations

from typing import Any


def get_phase_candidates(
    phase: int,
    prior_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Return (name, mutations) pairs for the given phase."""
    prior = prior_mutations or {}
    candidates: list[tuple[str, dict]] = []

    if suggested_experiments:
        seen = set()
        for name, muts in suggested_experiments:
            if name in seen:
                continue
            seen.add(name)
            candidates.append((name, muts))

    if phase == 1:
        candidates.extend(_phase_1_exit_monetization())
    elif phase == 2:
        candidates.extend(_phase_2_signal_discrimination())
    elif phase == 3:
        candidates.extend(_phase_3_entry_diversification())
    elif phase == 4:
        candidates.extend(_phase_4_frequency_interactions(prior))
    elif phase == 5:
        candidates.extend(_phase_5_evidence_guided_frequency())

    seen_names: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, muts in candidates:
        if name in seen_names:
            continue
        seen_names.add(name)
        unique.append((name, muts))
    return unique


def _exit_tp2(
    *,
    tp1_r: float = 1.2,
    tp1_pct: float = 0.45,
    tp2_r: float = 2.4,
    tp2_pct: float = 0.20,
    cap_mode: str = "degraded_only",
) -> dict[str, Any]:
    return {
        "param_overrides.TP1_R": tp1_r,
        "param_overrides.TP1_PARTIAL_PCT": tp1_pct,
        "param_overrides.TP2_R": tp2_r,
        "param_overrides.TP2_PARTIAL_PCT": tp2_pct,
        "param_overrides.TP1_ONLY_CAP_MODE": cap_mode,
    }


def _mfe_tiers(
    *,
    t1: tuple[float, float] = (2.0, 0.8),
    t2: tuple[float, float] = (3.0, 1.35),
    t3: tuple[float, float] = (4.0, 2.0),
) -> dict[str, Any]:
    return {
        "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
        "param_overrides.MFE_RATCHET_T1_R": t1[0],
        "param_overrides.MFE_RATCHET_T1_LOCK_R": t1[1],
        "param_overrides.MFE_RATCHET_T2_R": t2[0],
        "param_overrides.MFE_RATCHET_T2_LOCK_R": t2[1],
        "param_overrides.MFE_RATCHET_T3_R": t3[0],
        "param_overrides.MFE_RATCHET_T3_LOCK_R": t3[1],
    }


def _context_filter() -> dict[str, Any]:
    return {
        "param_overrides.WEAK_SCORE_BAND_FILTER_ENABLED": True,
        "param_overrides.WEAK_SCORE_BAND_MAX_BOX_WIDTH": 225.0,
        "param_overrides.WEAK_SCORE_BAND_MIN_RVOL": 1.75,
        "param_overrides.WIDE_BOX_SCORE_FILTER_ENABLED": True,
        "param_overrides.WIDE_BOX_MIN_WIDTH": 275.0,
        "param_overrides.WIDE_BOX_MIN_SCORE": 3.0,
        "param_overrides.WIDE_BOX_MIN_RVOL": 1.75,
    }


def _phase_1_exit_monetization() -> list[tuple[str, dict]]:
    """Target MFE giveback and absent TP2 without broad entry changes."""
    return [
        ("tp2_degraded_only_2.25_p20", _exit_tp2(tp2_r=2.25, tp2_pct=0.20, cap_mode="degraded_only")),
        ("tp2_degraded_only_2.50_p20", _exit_tp2(tp2_r=2.50, tp2_pct=0.20, cap_mode="degraded_only")),
        ("tp2_off_2.25_p15", _exit_tp2(tp2_r=2.25, tp2_pct=0.15, cap_mode="off")),
        ("tp2_off_2.75_p15", _exit_tp2(tp2_r=2.75, tp2_pct=0.15, cap_mode="off")),
        ("mfe_tiers_balanced", _mfe_tiers()),
        ("mfe_tiers_fast_lock", _mfe_tiers(t1=(1.75, 0.75), t2=(2.75, 1.30), t3=(3.75, 1.90))),
        ("mfe_tiers_loose_high", _mfe_tiers(t1=(2.25, 0.80), t2=(3.25, 1.40), t3=(4.50, 2.15))),
        ("tp2_mfe_balanced", {**_exit_tp2(tp2_r=2.50, tp2_pct=0.15, cap_mode="degraded_only"), **_mfe_tiers()}),
        ("tp2_mfe_off_cap", {**_exit_tp2(tp2_r=2.50, tp2_pct=0.15, cap_mode="off"), **_mfe_tiers()}),
        ("ratchet_lock_0.45", {"param_overrides.RATCHET_LOCK_PCT": 0.45}),
        ("ratchet_0.45_at_0.75", {"param_overrides.RATCHET_LOCK_PCT": 0.45, "param_overrides.RATCHET_THRESHOLD_R": 0.75}),
        ("chandelier_high_tighter", {
            "param_overrides.CHANDELIER_TIER2_MULT": 1.35,
            "param_overrides.CHANDELIER_TIER3_MULT": 1.05,
            "param_overrides.CHANDELIER_TIER4_MULT": 0.85,
        }),
    ]


def _phase_2_signal_discrimination() -> list[tuple[str, dict]]:
    """Use score with box/RVOL/regime context, not blunt lower thresholds."""
    context = _context_filter()
    return [
        ("weak_score_context", {
            "param_overrides.WEAK_SCORE_BAND_FILTER_ENABLED": True,
            "param_overrides.WEAK_SCORE_BAND_MAX_BOX_WIDTH": 225.0,
            "param_overrides.WEAK_SCORE_BAND_MIN_RVOL": 1.75,
        }),
        ("wide_box_context", {
            "param_overrides.WIDE_BOX_SCORE_FILTER_ENABLED": True,
            "param_overrides.WIDE_BOX_MIN_WIDTH": 275.0,
            "param_overrides.WIDE_BOX_MIN_SCORE": 3.0,
            "param_overrides.WIDE_BOX_MIN_RVOL": 1.75,
        }),
        ("score_context_full", context),
        ("score_1.25_with_context", {**context, "param_overrides.SCORE_NORMAL": 1.25}),
        ("rvol_1.35_with_context", {**context, "param_overrides.RVOL_SCORE_THRESH": 1.35}),
        ("score_1.25_rvol_1.35_context", {
            **context,
            "param_overrides.SCORE_NORMAL": 1.25,
            "param_overrides.RVOL_SCORE_THRESH": 1.35,
        }),
        ("max_box_300", {"param_overrides.MAX_BOX_WIDTH": 300.0}),
        ("max_box_350_context", {**context, "param_overrides.MAX_BOX_WIDTH": 350.0}),
        ("aligned_recovery_score_3_context", {
            **context,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
        }),
        ("aligned_recovery_score_3_25_context", {
            **context,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.25,
        }),
        ("neutral_recovery_score_3_context", {
            **context,
            "param_overrides.BLOCK_NEUTRAL_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
        }),
        ("neutral_aligned_score_3_25_context", {
            **context,
            "param_overrides.BLOCK_NEUTRAL_REGIME": False,
            "param_overrides.BLOCK_ALIGNED_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.25,
        }),
    ]


def _phase_3_entry_diversification() -> list[tuple[str, dict]]:
    """Probe A/B/C mechanisms with explicit quality guards."""
    context = _context_filter()
    return [
        ("b_range_p90", {"param_overrides.B_ALLOW_RANGE": True, "param_overrides.B_MIN_DISP_Q": 0.90}),
        ("b_range_p85", {"param_overrides.B_ALLOW_RANGE": True, "param_overrides.B_MIN_DISP_Q": 0.85}),
        ("b_range_context_p85", {**context, "param_overrides.B_ALLOW_RANGE": True, "param_overrides.B_MIN_DISP_Q": 0.85}),
        ("a_latch_only_stop_175", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": False,
            "flags.entry_a_latch": True,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("a_latch_only_context_stop_175", {
            **context,
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": False,
            "flags.entry_a_latch": True,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("a_latch_tight_offset", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": False,
            "flags.entry_a_latch": True,
            "param_overrides.A_STOP_ATR_MULT": 0.35,
            "param_overrides.MAX_STOP_WIDTH_PTS": 175.0,
        }),
        ("c_cont_mfe_0.50", {
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.50,
        }),
        ("c_cont_mfe_0.75", {
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.75,
        }),
        ("c_cont_context_mfe_0.50", {
            **context,
            "flags.entry_c_continuation": True,
            "param_overrides.C_CONT_ENTRY_ENABLED": True,
            "param_overrides.C_CONT_MFE_GATE_R": 0.50,
        }),
        ("c_standard_tighter_offset", {"param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.12}),
        ("c_standard_wider_offset", {"param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.24}),
        ("c_hold_2_bars", {"param_overrides.C_HOLD_BARS": 2}),
    ]


def _phase_4_frequency_interactions(prior: dict[str, Any]) -> list[tuple[str, dict]]:
    """Fine-tune accepted structural changes and controlled frequency probes."""
    candidates: list[tuple[str, dict]] = [
        ("cooldown_15", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 15}),
        ("cooldown_10_with_context", {**_context_filter(), "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 10}),
        ("cooldown_0_mfe", {**_mfe_tiers(), "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 0}),
        ("loss_streak_3", {"param_overrides.LOSS_STREAK_THRESHOLD": 3}),
        ("loss_streak_3_skip_3", {"param_overrides.LOSS_STREAK_THRESHOLD": 3, "param_overrides.LOSS_STREAK_SKIP_BARS": 3}),
        ("allow_05_context", {**_context_filter(), "flags.block_05_et": False}),
        ("allow_09_context", {**_context_filter(), "flags.block_09_et": False}),
        ("allow_05_09_context", {**_context_filter(), "flags.block_05_et": False, "flags.block_09_et": False}),
        ("b_range_cooldown_15", {
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_MIN_DISP_Q": 0.85,
            "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 15,
        }),
        ("exit_entry_combo", {
            **_exit_tp2(tp2_r=2.50, tp2_pct=0.15, cap_mode="degraded_only"),
            **_mfe_tiers(),
            "param_overrides.B_ALLOW_RANGE": True,
            "param_overrides.B_MIN_DISP_Q": 0.85,
        }),
    ]

    ranges = {
        "param_overrides.SCORE_NORMAL": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.RVOL_SCORE_THRESH": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.MIN_BOX_WIDTH": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": [(0.50, "m50"), (1.50, "p50")],
        "param_overrides.TP1_R": [(0.95, "m05"), (1.05, "p05")],
        "param_overrides.TP1_PARTIAL_PCT": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.RATCHET_LOCK_PCT": [(0.90, "m10"), (1.10, "p10")],
        "param_overrides.RATCHET_THRESHOLD_R": [(0.90, "m10"), (1.10, "p10")],
    }
    for key, adjustments in ranges.items():
        value = prior.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        base_name = key.replace("param_overrides.", "")
        for factor, label in adjustments:
            adjusted = value * factor
            if isinstance(value, int):
                adjusted = int(round(adjusted))
            candidates.append((f"finetune_{base_name}_{label}", {key: adjusted}))

    return candidates


def _phase_5_evidence_guided_frequency() -> list[tuple[str, dict]]:
    """Promote only frequency levers that passed direct quantitative probes."""
    return [
        ("box_min_50", {"param_overrides.MIN_BOX_WIDTH": 50}),
        ("box_min_75", {"param_overrides.MIN_BOX_WIDTH": 75}),
        ("box_min_100", {"param_overrides.MIN_BOX_WIDTH": 100}),
        ("allow_05", {"flags.block_05_et": False}),
        ("box_min_50_allow_05", {
            "param_overrides.MIN_BOX_WIDTH": 50,
            "flags.block_05_et": False,
        }),
        ("a_retest_box225_score3", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.A_MIN_SCORE": 3.0,
        }),
        ("a_retest_box225_boxmin50", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.MIN_BOX_WIDTH": 50,
        }),
        ("a_retest_box225_boxmin75", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.MIN_BOX_WIDTH": 75,
        }),
        ("a_retest_box225_boxmin50_allow05", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": False,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.MIN_BOX_WIDTH": 50,
            "flags.block_05_et": False,
        }),
        ("a_both_box225_no_weak", {
            "param_overrides.A_ENTRY_ENABLED": True,
            "flags.entry_a_retest": True,
            "flags.entry_a_latch": True,
            "param_overrides.A_TTL_5M_BARS": 12,
            "param_overrides.A_MAX_BOX_WIDTH": 225.0,
            "param_overrides.A_BLOCK_WEAK_SCORE_BAND": True,
        }),
    ]
