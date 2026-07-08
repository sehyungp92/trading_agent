"""
Momentum strategy experiment catalog (~370 experiments).

Frozen experiment definitions for NQDTC, Vdubus, and portfolio-level tests.
Used by the auto-runner harness to sweep ablations, parameter ranges, interactions,
portfolio configurations, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    id: str                       # e.g. "abl_nqdtc_score_threshold"
    type: str                     # ABLATION | PARAM_SWEEP | INTERACTION | PORTFOLIO | DIAGNOSTIC
    strategy: str                 # "nqdtc" | "vdubus" | "portfolio"
    description: str
    hypothesis: str
    priority: int                 # 1=highest
    mutations: dict = field(default_factory=dict)


E = Experiment

# ---------------------------------------------------------------------------
# Priority 1 — Ablation experiments (~95)
# ---------------------------------------------------------------------------


def _nqdtc_ablation() -> list[Experiment]:
    """37 ablation experiments for NQDTCAblationFlags."""
    exps: list[Experiment] = []

    true_flags = [
        ("displacement_threshold", "Disable displacement threshold", "Displacement filter may reject valid micro-setups"),
        ("score_threshold", "Disable score threshold", "Score filter may reject marginal but profitable trades"),
        ("breakout_quality_reject", "Disable breakout quality reject", "Quality reject may over-filter"),
        ("dirty_mechanism", "Disable dirty mechanism", "Dirty mechanism may miss continuation entries"),
        ("continuation_mode", "Disable continuation mode", "Continuation mode may add noise"),
        ("chop_halt", "Disable chop halt", "Chop halt may block trades in transitional markets"),
        ("chop_degraded", "Disable chop degraded", "Chop degraded sizing may be too conservative"),
        ("news_blackout", "Disable news blackout", "News blackout may miss post-event momentum"),
        ("friction_gate", "Disable friction gate", "Friction gate may over-filter in trending markets"),
        ("micro_guard", "Disable micro guard", "Micro guard may reject valid micro-displacement setups"),
        ("daily_stop", "Disable daily stop", "Daily stop may end profitable days early"),
        ("weekly_stop", "Disable weekly stop", "Weekly stop may cut recovery weeks short"),
        ("monthly_stop", "Disable monthly stop", "Monthly stop may be too conservative"),
        ("drawdown_throttle", "Disable drawdown throttle", "Drawdown throttle may slow recovery"),
        ("entry_a_retest", "Disable entry A retest", "Entry A retest may be the weakest entry type"),
        ("entry_a_latch", "Disable entry A latch", "Entry A latch may add false signals"),
        ("entry_b_sweep", "Disable entry B sweep", "Entry B sweep may catch low-quality reversals"),
        ("entry_c_standard", "Disable entry C standard", "Entry C standard may underperform"),
        ("rth_entries", "Disable RTH entries restriction", "RTH-only may miss ETH setups"),
        ("tiered_exits", "Disable tiered exits", "Tiered exits may reduce simplicity without adding edge"),
        ("chandelier_trailing", "Disable chandelier trailing", "Chandelier trailing may exit too early"),
        ("stale_exit", "Disable stale exit", "Stale exit may cut trades prematurely"),
        ("profit_funded_be", "Disable profit-funded breakeven", "Profit-funded BE may lock out recovery"),
        ("overnight_bridge", "Disable overnight bridge", "Overnight bridge may add gap risk"),
        ("early_chandelier", "Disable early chandelier", "Early chandelier may trim winners too soon"),
        ("loss_streak_cooldown", "Disable loss streak cooldown", "Cooldown may miss recovery trades"),
        ("block_05_et", "Disable 05 ET block", "5 AM block may miss pre-RTH momentum"),
        ("block_04_et", "Disable 04 ET block", "4 AM block may miss European session overlap"),
        ("block_06_et", "Disable 06 ET block", "6 AM block may miss early US prep moves"),
        ("block_09_et", "Disable 09 ET block", "9 AM block may miss open momentum"),
        ("block_12_et", "Disable 12 ET block", "Noon block may miss afternoon continuation"),
    ]
    for flag, desc, hyp in true_flags:
        exps.append(E(
            f"abl_nqdtc_{flag}",
            "ABLATION", "nqdtc", desc, hyp, 1,
            {f"flags.{flag}": False},
        ))

    false_flags = [
        ("entry_c_continuation", "Enable entry C continuation", "Continuation entries may add profitable setups"),
        ("max_loss_cap", "Enable max loss cap", "Max loss cap may improve tail risk"),
        ("max_stop_width", "Enable max stop width", "Max stop width may filter bad R:R setups"),
        ("block_thursday", "Enable Thursday block", "Thursday may be a weak session day"),
        ("early_be", "Enable early breakeven", "Early BE may protect capital on slow starters"),
        ("es_daily_trend", "Enable ES daily trend filter", "ES daily trend may improve direction accuracy"),
    ]
    for flag, desc, hyp in false_flags:
        exps.append(E(
            f"abl_nqdtc_{flag}",
            "ABLATION", "nqdtc", desc, hyp, 1,
            {f"flags.{flag}": True},
        ))

    return exps


def _vdubus_ablation() -> list[Experiment]:
    """42 ablation experiments for VdubusAblationFlags."""
    exps: list[Experiment] = []

    true_flags = [
        ("daily_trend_gate", "Disable daily trend gate", "Daily trend gate may reject valid counter-trend setups"),
        ("shock_block", "Disable shock block", "Shock block may miss high-vol continuation"),
        ("hourly_alignment", "Disable hourly alignment", "Hourly alignment may over-constrain entries"),
        ("type_a_enabled", "Disable type A entries", "Type A entries may underperform"),
        ("type_b_enabled", "Disable type B entries", "Type B entries may underperform"),
        ("vwap_cap_gate", "Disable VWAP cap gate", "VWAP cap may block valid extended entries"),
        ("extension_sanity", "Disable extension sanity check", "Extension sanity may over-filter"),
        ("touch_lookback_gate", "Disable touch lookback gate", "Touch lookback may reject fresh setups"),
        ("slope_gate", "Disable slope gate", "Slope gate may reject valid low-angle setups"),
        ("predator_overlay", "Disable predator overlay", "Predator overlay may add false positives"),
        ("momentum_floor", "Disable momentum floor", "Momentum floor may reject slow-burn entries"),
        ("min_max_stop", "Disable min/max stop", "Min/max stop may distort R:R"),
        ("ttl_cancel", "Disable TTL cancel", "TTL cancel may kill orders about to fill"),
        ("teleport_skip", "Disable teleport skip", "Teleport skip may miss gap continuations"),
        ("fallback_market", "Disable fallback market orders", "Fallback market may cause slippage"),
        ("viability_filter", "Disable viability filter", "Viability filter may reject recoverable setups"),
        ("direction_caps", "Disable direction caps", "Direction caps may limit profitable trends"),
        ("heat_cap", "Disable heat cap", "Heat cap may be too conservative in trending markets"),
        ("vwap_failure_exit", "Disable VWAP failure exit", "VWAP failure exit may cut trades too early"),
        ("stale_exit", "Disable stale exit", "Stale exit may remove slow winners"),
        ("plus_1r_partial", "Disable +1R partial", "+1R partial may reduce winner magnitude"),
        ("decision_gate", "Disable decision gate", "Decision gate may add unnecessary delay"),
        ("overnight_widening", "Disable overnight widening", "Overnight widening may waste stop distance"),
        ("vwap_a_failure", "Disable VWAP A failure", "VWAP A failure may exit too aggressively"),
        ("friday_override", "Disable Friday override", "Friday override may miss end-of-week momentum"),
        ("free_ride_stale", "Disable free ride stale", "Free ride stale may exit profitable runners"),
        ("free_profit_lock", "Disable free profit lock", "Free profit lock may reduce upside"),
        ("max_duration", "Disable max duration", "Max duration may cut long-running winners"),
        ("post_partial_trail_tighten", "Disable post-partial trail tighten", "Trail tighten may exit too early after partial"),
        ("drawdown_throttle", "Disable drawdown throttle", "Drawdown throttle may slow recovery"),
        ("event_blocking", "Disable event blocking", "Event blocking may miss post-event moves"),
        ("expanded_dead_zone", "Disable expanded dead zone", "Expanded dead zone may be too wide"),
        ("choppiness_gate", "Disable choppiness gate", "Choppiness gate may over-filter"),
        ("rth_only_shorts", "Disable RTH-only shorts", "RTH-only shorts may miss ETH short setups"),
        ("block_20h_hour", "Disable 20h hour block", "20h block may miss valid late-session entries"),
        ("close_skip_partial", "Disable close skip partial", "Close skip partial may miss end-of-day exits"),
    ]
    for flag, desc, hyp in true_flags:
        exps.append(E(
            f"abl_vdubus_{flag}",
            "ABLATION", "vdubus", desc, hyp, 1,
            {f"flags.{flag}": False},
        ))

    false_flags = [
        ("adaptive_stale", "Enable adaptive stale", "Adaptive stale may improve exit timing"),
        ("dow_sizing", "Enable DOW sizing", "DOW sizing may capture day-of-week patterns"),
        ("entry_quality_gate", "Enable entry quality gate", "Quality gate may filter low-EV entries"),
        ("evening_vwap_cap", "Enable evening VWAP cap", "Evening VWAP cap may reduce late-session risk"),
        ("mfe_ratchet", "Enable MFE ratchet", "MFE ratchet may lock profits more effectively"),
        ("bar_quality_gate", "Enable bar quality gate", "Bar quality gate may filter noisy entries"),
        ("stale_mfe_exempt", "Enable stale MFE exemption", "Exempt high-MFE trades from stale exit may preserve winners"),
    ]
    for flag, desc, hyp in false_flags:
        exps.append(E(
            f"abl_vdubus_{flag}",
            "ABLATION", "vdubus", desc, hyp, 1,
            {f"flags.{flag}": True},
        ))

    return exps


# ---------------------------------------------------------------------------
# Priority 2 — Parameter Sweeps (~150)
# ---------------------------------------------------------------------------


def _nqdtc_param_sweeps() -> list[Experiment]:
    """55 parameter sweep experiments for NQDTC."""
    exps: list[Experiment] = []

    sweeps: list[tuple[str, str, list]] = [
        ("DISPLACEMENT_MIN_POINTS", "param_overrides.DISPLACEMENT_MIN_POINTS", [15, 20, 30, 35]),
        ("SCORE_MIN", "param_overrides.SCORE_MIN", [3, 4, 6, 7]),
        ("TP1_R", "param_overrides.TP1_R", [0.8, 1.0, 1.5, 2.0]),
        ("TP2_R", "param_overrides.TP2_R", [2.0, 2.5, 3.5, 4.0]),
        ("TP1_PARTIAL_PCT", "param_overrides.TP1_PARTIAL_PCT", [0.25, 0.40, 0.60, 0.75]),
        ("CHANDELIER_ATR_MULT", "param_overrides.CHANDELIER_ATR_MULT", [2.0, 2.5, 3.5, 4.0]),
        ("CHANDELIER_LOOKBACK", "param_overrides.CHANDELIER_LOOKBACK", [10, 15, 25, 30]),
        ("STALE_BARS", "param_overrides.STALE_BARS", [12, 15, 20, 25]),
        ("PROFIT_BE_R", "param_overrides.PROFIT_BE_R", [0.3, 0.5, 0.8, 1.0]),
        ("OVERNIGHT_WIDEN_MULT", "param_overrides.OVERNIGHT_WIDEN_MULT", [1.2, 1.5, 2.0, 2.5]),
        ("LOSS_STREAK_THRESHOLD", "param_overrides.LOSS_STREAK_THRESHOLD", [2, 4, 5]),
        ("LOSS_STREAK_SKIP", "param_overrides.LOSS_STREAK_SKIP", [1, 2, 3]),
        ("DIRTY_THRESHOLD_BARS", "param_overrides.DIRTY_THRESHOLD_BARS", [2, 3, 5]),
        ("CONTINUATION_LOOKBACK", "param_overrides.CONTINUATION_LOOKBACK", [3, 5, 8, 10]),
        ("min_stop_distance", "flags.min_stop_distance", [2.0, 4.0, 5.0, 6.0]),
    ]
    for name, key, values in sweeps:
        for v in values:
            exps.append(E(
                f"sweep_nqdtc_{name}_{v}",
                "PARAM_SWEEP", "nqdtc",
                f"NQDTC {name}={v}",
                f"Testing {name} at {v} may reveal performance sensitivity",
                2,
                {key: v},
            ))

    return exps


def _vdubus_param_sweeps() -> list[Experiment]:
    """60 parameter sweep experiments for Vdubus."""
    exps: list[Experiment] = []

    sweeps: list[tuple[str, str, list]] = [
        ("VWAP_CAP_POINTS", "param_overrides.VWAP_CAP_POINTS", [3.0, 5.0, 8.0, 10.0]),
        ("EXTENSION_ATR_MULT", "param_overrides.EXTENSION_ATR_MULT", [1.5, 2.0, 3.0, 4.0]),
        ("TOUCH_LOOKBACK_BARS", "param_overrides.TOUCH_LOOKBACK_BARS", [3, 5, 8, 10]),
        ("SLOPE_MIN_ANGLE", "param_overrides.SLOPE_MIN_ANGLE", [0.05, 0.10, 0.20, 0.30]),
        ("MOMENTUM_FLOOR_PCT", "param_overrides.MOMENTUM_FLOOR_PCT", [0.001, 0.002, 0.005, 0.008]),
        ("TTL_CANCEL_BARS", "param_overrides.TTL_CANCEL_BARS", [3, 5, 8, 10]),
        ("STALE_BARS", "param_overrides.STALE_BARS", [8, 12, 18, 24]),
        ("PLUS_1R_PARTIAL_PCT", "param_overrides.PLUS_1R_PARTIAL_PCT", [0.30, 0.50, 0.75]),
        ("OVERNIGHT_WIDEN_MULT", "param_overrides.OVERNIGHT_WIDEN_MULT", [1.2, 1.5, 2.0, 2.5]),
        ("MAX_DURATION_BARS", "param_overrides.MAX_DURATION_BARS", [30, 50, 80, 100]),
        ("TRAIL_ACTIVATION_R", "param_overrides.TRAIL_ACTIVATION_R", [0.5, 0.8, 1.2, 1.5]),
        ("TRAIL_ATR_MULT", "param_overrides.TRAIL_ATR_MULT", [1.5, 2.0, 3.0, 4.0]),
        ("DECISION_GATE_BARS", "param_overrides.DECISION_GATE_BARS", [5, 8, 12]),
        ("DEAD_ZONE_START_HOUR", "param_overrides.DEAD_ZONE_START_HOUR", [10, 11, 12]),
        ("DEAD_ZONE_END_HOUR", "param_overrides.DEAD_ZONE_END_HOUR", [14, 15, 16]),
        ("CHOP_THRESHOLD", "param_overrides.CHOP_THRESHOLD", [40, 50, 65, 75]),
    ]
    for name, key, values in sweeps:
        for v in values:
            exps.append(E(
                f"sweep_vdubus_{name}_{v}",
                "PARAM_SWEEP", "vdubus",
                f"Vdubus {name}={v}",
                f"Testing {name} at {v} may reveal performance sensitivity",
                2,
                {key: v},
            ))

    return exps


# ---------------------------------------------------------------------------
# Priority 3 — Interaction experiments (~40)
# ---------------------------------------------------------------------------


def _nqdtc_interactions() -> list[Experiment]:
    """15 interaction experiments for NQDTC."""
    return [
        E("int_nqdtc_tight_exits", "INTERACTION", "nqdtc",
          "Tight exits: low TP1 + tight chandelier + short stale",
          "Tighter exits may improve win rate on small moves",
          3,
          {"param_overrides.TP1_R": 0.8, "param_overrides.CHANDELIER_ATR_MULT": 2.0,
           "param_overrides.STALE_BARS": 12}),

        E("int_nqdtc_loose_exits", "INTERACTION", "nqdtc",
          "Loose exits: high TP1 + wide chandelier + long stale",
          "Looser exits may let winners run further",
          3,
          {"param_overrides.TP1_R": 1.5, "param_overrides.CHANDELIER_ATR_MULT": 3.5,
           "param_overrides.STALE_BARS": 25}),

        E("int_nqdtc_aggressive_entry", "INTERACTION", "nqdtc",
          "Aggressive entry: continuation + RTH + no displacement filter",
          "Widening entry criteria may increase opportunity",
          3,
          {"flags.entry_c_continuation": True, "flags.rth_entries": True,
           "flags.displacement_threshold": False}),

        E("int_nqdtc_conservative_entry", "INTERACTION", "nqdtc",
          "Conservative entry: score + quality + friction all on",
          "Maximum entry filtering may improve quality",
          3,
          {"flags.score_threshold": True, "flags.breakout_quality_reject": True,
           "flags.friction_gate": True}),

        E("int_nqdtc_max_continuation", "INTERACTION", "nqdtc",
          "Max continuation: mode + C continuation + long lookback",
          "Full continuation config may capture trend extensions",
          3,
          {"flags.continuation_mode": True, "flags.entry_c_continuation": True,
           "param_overrides.CONTINUATION_LOOKBACK": 8}),

        E("int_nqdtc_no_time_blocks", "INTERACTION", "nqdtc",
          "No time blocks: all hour blocks disabled",
          "Time blocks may be removing valid setups unnecessarily",
          3,
          {"flags.block_05_et": False, "flags.block_04_et": False,
           "flags.block_06_et": False, "flags.block_09_et": False,
           "flags.block_12_et": False}),

        E("int_nqdtc_all_time_blocks", "INTERACTION", "nqdtc",
          "All time blocks: every hour block + Thursday enabled",
          "Maximum time filtering may reveal session-dependent edge",
          3,
          {"flags.block_05_et": True, "flags.block_04_et": True,
           "flags.block_06_et": True, "flags.block_09_et": True,
           "flags.block_12_et": True, "flags.block_thursday": True}),

        E("int_nqdtc_tight_risk", "INTERACTION", "nqdtc",
          "Tight risk: all stop levels + drawdown throttle",
          "Full risk controls may smooth equity curve",
          3,
          {"flags.daily_stop": True, "flags.weekly_stop": True,
           "flags.monthly_stop": True, "flags.drawdown_throttle": True}),

        E("int_nqdtc_loose_risk", "INTERACTION", "nqdtc",
          "Loose risk: no daily/weekly/monthly stops",
          "Removing risk stops may reveal true strategy potential",
          3,
          {"flags.daily_stop": False, "flags.weekly_stop": False,
           "flags.monthly_stop": False}),

        E("int_nqdtc_entry_a_only", "INTERACTION", "nqdtc",
          "Entry A only: disable B and C entries",
          "Isolating entry A may reveal its standalone performance",
          3,
          {"flags.entry_b_sweep": False, "flags.entry_c_standard": False}),

        E("int_nqdtc_entry_b_only", "INTERACTION", "nqdtc",
          "Entry B only: disable A retest, A latch, C standard",
          "Isolating entry B may reveal sweep entry quality",
          3,
          {"flags.entry_a_retest": False, "flags.entry_a_latch": False,
           "flags.entry_c_standard": False}),

        E("int_nqdtc_fast_be", "INTERACTION", "nqdtc",
          "Fast breakeven: early BE + low profit threshold",
          "Faster BE may protect capital on marginal trades",
          3,
          {"flags.early_be": True, "param_overrides.PROFIT_BE_R": 0.5}),

        E("int_nqdtc_es_trend_filter", "INTERACTION", "nqdtc",
          "ES trend filter: ES daily trend + RTH entries",
          "ES trend alignment may improve direction accuracy",
          3,
          {"flags.es_daily_trend": True, "flags.rth_entries": True}),

        E("int_nqdtc_overnight_aggressive", "INTERACTION", "nqdtc",
          "Overnight aggressive: bridge on + tight widening",
          "Tighter overnight widening may improve gap risk management",
          3,
          {"flags.overnight_bridge": True, "param_overrides.OVERNIGHT_WIDEN_MULT": 1.2}),

        E("int_nqdtc_max_loss_with_width", "INTERACTION", "nqdtc",
          "Max loss + max stop width caps enabled",
          "Combined loss/width caps may improve tail risk profile",
          3,
          {"flags.max_loss_cap": True, "flags.max_stop_width": True}),
    ]


def _vdubus_interactions() -> list[Experiment]:
    """13 interaction experiments for Vdubus."""
    return [
        E("int_vdubus_tight_exits", "INTERACTION", "vdubus",
          "Tight exits: short stale + high partial + trail tighten",
          "Tighter exits may lock profits sooner",
          3,
          {"param_overrides.STALE_BARS": 8, "param_overrides.PLUS_1R_PARTIAL_PCT": 0.50,
           "flags.post_partial_trail_tighten": True}),

        E("int_vdubus_loose_exits", "INTERACTION", "vdubus",
          "Loose exits: long stale + low partial + long duration",
          "Looser exits may let winners run",
          3,
          {"param_overrides.STALE_BARS": 24, "param_overrides.PLUS_1R_PARTIAL_PCT": 0.30,
           "param_overrides.MAX_DURATION_BARS": 100}),

        E("int_vdubus_aggressive_entry", "INTERACTION", "vdubus",
          "Aggressive entry: both types + no VWAP cap + no extension check",
          "Widening entry criteria may increase trade count",
          3,
          {"flags.type_a_enabled": True, "flags.type_b_enabled": True,
           "flags.vwap_cap_gate": False, "flags.extension_sanity": False}),

        E("int_vdubus_conservative_entry", "INTERACTION", "vdubus",
          "Conservative entry: slope + predator + momentum + touch lookback",
          "Maximum entry filtering may improve quality",
          3,
          {"flags.slope_gate": True, "flags.predator_overlay": True,
           "flags.momentum_floor": True, "flags.touch_lookback_gate": True}),

        E("int_vdubus_no_risk_caps", "INTERACTION", "vdubus",
          "No risk caps: direction + heat + drawdown all off",
          "Removing risk caps may reveal true strategy potential",
          3,
          {"flags.direction_caps": False, "flags.heat_cap": False,
           "flags.drawdown_throttle": False}),

        E("int_vdubus_tight_risk", "INTERACTION", "vdubus",
          "Tight risk: direction + heat + drawdown + event blocking",
          "Full risk controls may smooth equity curve",
          3,
          {"flags.direction_caps": True, "flags.heat_cap": True,
           "flags.drawdown_throttle": True, "flags.event_blocking": True}),

        E("int_vdubus_vwap_focused", "INTERACTION", "vdubus",
          "VWAP focused: all VWAP-related gates and exits on",
          "Full VWAP integration may improve mean-reversion edge",
          3,
          {"flags.vwap_failure_exit": True, "flags.vwap_a_failure": True,
           "flags.vwap_cap_gate": True, "flags.evening_vwap_cap": True}),

        E("int_vdubus_overnight_safe", "INTERACTION", "vdubus",
          "Overnight safe: widening + Friday override + free ride stale",
          "Overnight protection combo may reduce gap risk",
          3,
          {"flags.overnight_widening": True, "flags.friday_override": True,
           "flags.free_ride_stale": True}),

        E("int_vdubus_minimal_dead_zone", "INTERACTION", "vdubus",
          "Minimal dead zone: expanded off + 20h block off",
          "Smaller dead zone may allow more setups",
          3,
          {"flags.expanded_dead_zone": False, "flags.block_20h_hour": False}),

        E("int_vdubus_full_dead_zone", "INTERACTION", "vdubus",
          "Full dead zone: expanded + 20h block + wide hours",
          "Maximum dead zone may avoid all choppy periods",
          3,
          {"flags.expanded_dead_zone": True, "flags.block_20h_hour": True,
           "param_overrides.DEAD_ZONE_START_HOUR": 10,
           "param_overrides.DEAD_ZONE_END_HOUR": 16}),

        E("int_vdubus_experimental_on", "INTERACTION", "vdubus",
          "All experimental flags on: adaptive stale + DOW sizing + quality gates + MFE",
          "Enabling all experimental features may reveal combined value",
          3,
          {"flags.adaptive_stale": True, "flags.dow_sizing": True,
           "flags.entry_quality_gate": True, "flags.mfe_ratchet": True}),

        E("int_vdubus_close_optimized", "INTERACTION", "vdubus",
          "Close optimized: skip partial near close + shorter stale",
          "Close-aware exits may improve end-of-day behavior",
          3,
          {"flags.close_skip_partial": True, "param_overrides.STALE_BARS": 12}),

        E("int_vdubus_chop_aware", "INTERACTION", "vdubus",
          "Chop aware: choppiness gate + moderate threshold + slope gate",
          "Chop detection with slope may filter ranging markets",
          3,
          {"flags.choppiness_gate": True, "param_overrides.CHOP_THRESHOLD": 50,
           "flags.slope_gate": True}),
    ]


# ---------------------------------------------------------------------------
# Priority 4 — Portfolio experiments (~65)
# ---------------------------------------------------------------------------


def _portfolio_param_sweeps() -> list[Experiment]:
    """Portfolio-level parameter sweeps."""
    exps: list[Experiment] = []

    # Heat cap
    for v in [2.0, 3.0, 4.0, 4.5]:
        exps.append(E(
            f"port_heat_cap_{v}", "PORTFOLIO", "portfolio",
            f"Portfolio heat cap = {v}R",
            f"Heat cap at {v}R may change concurrent risk profile",
            4, {"portfolio.heat_cap_R": v}))

    # Directional cap
    for v in [2.0, 3.0, 4.0]:
        exps.append(E(
            f"port_directional_cap_{v}", "PORTFOLIO", "portfolio",
            f"Portfolio directional cap = {v}R",
            f"Directional cap at {v}R may change directional exposure",
            4, {"portfolio.directional_cap_R": v}))

    # Daily stop
    for v in [1.0, 2.0, 2.5]:
        exps.append(E(
            f"port_daily_stop_{v}", "PORTFOLIO", "portfolio",
            f"Portfolio daily stop = {v}R",
            f"Daily stop at {v}R may change daily loss profile",
            4, {"portfolio.portfolio_daily_stop_R": v}))

    # Weekly stop
    for v in [0, 8.0, 10.0, 15.0]:
        exps.append(E(
            f"port_weekly_stop_{v}", "PORTFOLIO", "portfolio",
            f"Portfolio weekly stop = {v}R",
            f"Weekly stop at {v}R may change weekly loss profile",
            4, {"portfolio.portfolio_weekly_stop_R": v}))

    # Max total positions
    for v in [2, 4, 5]:
        exps.append(E(
            f"port_max_positions_{v}", "PORTFOLIO", "portfolio",
            f"Portfolio max total positions = {v}",
            f"Max positions at {v} may change concurrent trade profile",
            4, {"portfolio.max_total_positions": v}))

    # NQDTC direction filter
    for v in [True, False]:
        exps.append(E(
            f"port_nqdtc_direction_filter_{v}", "PORTFOLIO", "portfolio",
            f"NQDTC direction filter = {v}",
            f"Direction filter {'on' if v else 'off'} may change NQDTC/Vdubus interaction",
            4, {"portfolio.nqdtc_direction_filter_enabled": v}))

    # NQDTC agree size mult
    for v in [1.0, 1.25, 1.75, 2.0]:
        exps.append(E(
            f"port_nqdtc_agree_size_{v}", "PORTFOLIO", "portfolio",
            f"NQDTC agree size mult = {v}",
            f"Agree sizing at {v}x may change aligned-trade risk",
            4, {"portfolio.nqdtc_agree_size_mult": v}))

    # NQDTC oppose size mult
    for v in [0.0, 0.25, 0.50, 1.0]:
        exps.append(E(
            f"port_nqdtc_oppose_size_{v}", "PORTFOLIO", "portfolio",
            f"NQDTC oppose size mult = {v}",
            f"Oppose sizing at {v}x may change hedging behavior",
            4, {"portfolio.nqdtc_oppose_size_mult": v}))

    return exps


def _portfolio_allocation_sweeps() -> list[Experiment]:
    """Strategy allocation sweeps: base risk, daily stops, concurrency, etc."""
    exps: list[Experiment] = []

    # NQDTC base risk [index 1]
    for v in [0.005, 0.010, 0.012]:
        exps.append(E(
            f"port_nqdtc_risk_{v}", "PORTFOLIO", "portfolio",
            f"NQDTC base risk = {v}",
            f"NQDTC risk at {v} may change strategy contribution",
            4, {"portfolio.strategies[1].base_risk_pct": v}))

    # Vdubus base risk [index 0]
    for v in [0.005, 0.010, 0.012]:
        exps.append(E(
            f"port_vdubus_risk_{v}", "PORTFOLIO", "portfolio",
            f"Vdubus base risk = {v}",
            f"Vdubus risk at {v} may change strategy contribution",
            4, {"portfolio.strategies[0].base_risk_pct": v}))

    # NQDTC daily stop [index 1]
    for v in [2.0, 3.0, 3.5]:
        exps.append(E(
            f"port_nqdtc_daily_stop_{v}", "PORTFOLIO", "portfolio",
            f"NQDTC daily stop = {v}R",
            f"NQDTC daily stop at {v}R may change drawdown profile",
            4, {"portfolio.strategies[1].daily_stop_R": v}))

    # NQDTC continuation size mult [index 1]
    for v in [0.40, 0.50, 0.85, 1.0]:
        exps.append(E(
            f"port_nqdtc_cont_size_{v}", "PORTFOLIO", "portfolio",
            f"NQDTC continuation size mult = {v}",
            f"Continuation sizing at {v}x may change add-on risk",
            4, {"portfolio.strategies[1].continuation_size_mult": v}))

    # NQDTC reversal only [index 1]
    exps.append(E(
        "port_nqdtc_reversal_only", "PORTFOLIO", "portfolio",
        "NQDTC reversal only mode",
        "Restricting NQDTC to reversals may improve quality",
        4, {"portfolio.strategies[1].reversal_only": True}))

    return exps


def _portfolio_drawdown_tiers() -> list[Experiment]:
    """Drawdown tier configuration experiments."""
    return [
        E("port_drawdown_aggressive", "PORTFOLIO", "portfolio",
          "Aggressive drawdown tiers: fast throttle",
          "Aggressive throttling may reduce max drawdown at cost of recovery",
          4, {"portfolio.drawdown_tiers": ((0.05, 1.0), (0.08, 0.50), (0.12, 0.25), (1.0, 0.0))}),

        E("port_drawdown_relaxed", "PORTFOLIO", "portfolio",
          "Relaxed drawdown tiers: slow throttle",
          "Relaxed throttling may improve recovery speed",
          4, {"portfolio.drawdown_tiers": ((0.12, 1.0), (0.18, 0.50), (0.22, 0.25), (1.0, 0.0))}),

        E("port_drawdown_none", "PORTFOLIO", "portfolio",
          "No drawdown throttle",
          "Removing throttle reveals true unfiltered performance",
          4, {"portfolio.drawdown_tiers": ((1.0, 1.0),)}),
    ]


def _portfolio_strategy_exclusion() -> list[Experiment]:
    """Strategy exclusion experiments for the remaining momentum engines."""
    return [
        E("port_exclude_nqdtc", "PORTFOLIO", "portfolio",
          "Exclude NQDTC: run Vdubus only",
          "Removing NQDTC may reveal if it adds portfolio value",
          4, {"portfolio.run_nqdtc": False}),

        E("port_exclude_vdubus", "PORTFOLIO", "portfolio",
          "Exclude Vdubus: run NQDTC only",
          "Removing Vdubus may reveal if it adds portfolio value",
          4, {"portfolio.run_vdubus": False}),
    ]

def _portfolio_preset_comparisons() -> list[Experiment]:
    """Preset configuration comparisons."""
    presets = [
        ("make_10k_config", "10K starter config"),
        ("v3", "V3 configuration"),
        ("v4", "V4 configuration"),
        ("v5", "V5 configuration"),
        ("optimized", "Optimized configuration"),
    ]
    return [
        E(f"port_preset_{name}", "PORTFOLIO", "portfolio",
          f"Preset comparison: {desc}",
          f"Preset {name} may reveal optimal configuration baseline",
          4, {"preset": name})
        for name, desc in presets
    ]


# ---------------------------------------------------------------------------
# Priority 5 — Diagnostic experiments (~20)
# ---------------------------------------------------------------------------


def _diagnostic_experiments() -> list[Experiment]:
    """Diagnostic experiments for session, time, cost, and regime analysis."""
    exps: list[Experiment] = []

    # Session isolation (6 experiments)
    sessions = [
        ("eth_asia", "ETH Asia session (18:00-02:00 ET)", "Asia session may have distinct characteristics"),
        ("eth_europe", "ETH Europe session (02:00-08:00 ET)", "Europe session may have distinct characteristics"),
        ("rth_open", "RTH open session (09:30-11:00 ET)", "RTH open may be the highest-edge window"),
        ("rth_midday", "RTH midday session (11:00-14:00 ET)", "Midday may be a dead zone for momentum"),
        ("rth_close", "RTH close session (14:00-16:00 ET)", "Close session may have distinct mean-reversion edge"),
        ("eth_evening", "ETH evening session (16:00-18:00 ET)", "Evening session may be low-liquidity noise"),
    ]
    for session_id, desc, hyp in sessions:
        exps.append(E(
            f"diag_session_{session_id}", "DIAGNOSTIC", "portfolio",
            f"Session isolation: {desc}",
            hyp, 5,
            {"session_filter": session_id}))

    # Overnight hold tests
    exps.append(E(
        "diag_overnight_only", "DIAGNOSTIC", "portfolio",
        "Overnight holds only: trades that span sessions",
        "Overnight holds may carry disproportionate gap risk",
        5, {"overnight_only": True}))

    exps.append(E(
        "diag_no_overnight", "DIAGNOSTIC", "portfolio",
        "No overnight holds: close all positions before session end",
        "Avoiding overnight may reduce gap risk at cost of trend capture",
        5, {"no_overnight": True}))

    # VIX regime blocks
    exps.append(E(
        "diag_vix_low", "DIAGNOSTIC", "portfolio",
        "VIX regime: low volatility only (VIX < 20)",
        "Low VIX may favor mean-reversion strategies",
        5, {"vix_max": 20}))

    exps.append(E(
        "diag_vix_high", "DIAGNOSTIC", "portfolio",
        "VIX regime: high volatility only (VIX > 25)",
        "High VIX may favor momentum strategies",
        5, {"vix_min": 25}))

    exps.append(E(
        "diag_vix_mid", "DIAGNOSTIC", "portfolio",
        "VIX regime: mid volatility (15 < VIX < 25)",
        "Mid VIX may be the sweet spot for balanced strategies",
        5, {"vix_range": (15, 25)}))

    # Time-of-day windows
    tod_windows = [
        (9, 12, "morning", "Morning hours may capture most edge"),
        (12, 15, "afternoon", "Afternoon may be lower-quality for momentum"),
        (15, 17, "late_afternoon", "Late afternoon may have close-driven edge"),
    ]
    for start, end, label, hyp in tod_windows:
        exps.append(E(
            f"diag_tod_{label}", "DIAGNOSTIC", "portfolio",
            f"Time-of-day window: {start}:00-{end}:00 ET",
            hyp, 5,
            {"tod_start": start, "tod_end": end}))

    # Cost stress tests
    for v in [0.31, 1.24, 1.86]:
        exps.append(E(
            f"diag_cost_stress_{v}", "DIAGNOSTIC", "portfolio",
            f"Cost stress: commission = ${v}/contract",
            f"Commission at ${v} may reveal cost sensitivity",
            5, {"slippage.commission_per_contract": v}))

    # Equity scaling
    for v in [5000, 25000, 50000]:
        exps.append(E(
            f"diag_equity_{v}", "DIAGNOSTIC", "portfolio",
            f"Equity scaling: initial equity = ${v:,}",
            f"${v:,} equity may reveal sizing and position count effects",
            5, {"initial_equity": v}))

    # Date range experiments
    date_ranges = [
        ("h1_2024", "2024-01-01", "2024-06-30", "H1 2024 may have different regime characteristics"),
        ("h2_2024", "2024-07-01", "2024-12-31", "H2 2024 may have different regime characteristics"),
        ("full_2023", "2023-01-01", "2023-12-31", "2023 may reveal out-of-sample performance"),
    ]
    for label, start, end, hyp in date_ranges:
        exps.append(E(
            f"diag_daterange_{label}", "DIAGNOSTIC", "portfolio",
            f"Date range: {start} to {end}",
            hyp, 5,
            {"start_date": start, "end_date": end}))

    # Seasonality filters
    seasonality_events = [
        ("fomc", "FOMC meeting windows", "FOMC may create regime-specific edge"),
        ("opex", "Options expiration windows", "OpEx may create pinning effects"),
        ("nfp", "Non-farm payroll windows", "NFP may create high-vol dislocations"),
    ]
    for event_id, desc, hyp in seasonality_events:
        exps.append(E(
            f"diag_seasonality_{event_id}", "DIAGNOSTIC", "portfolio",
            f"Seasonality filter: {desc}",
            hyp, 5,
            {"seasonality_filter": event_id}))

    return exps


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_experiment_queue(strategy_filter: str = "all") -> list[Experiment]:
    """Build priority-ordered experiment queue.

    Args:
        strategy_filter: "nqdtc", "vdubus", "portfolio", or "all"

    Returns:
        List of Experiment sorted by (priority, id).
    """
    all_experiments: list[Experiment] = []

    # Priority 1: Ablation
    if strategy_filter in ("all", "nqdtc"):
        all_experiments.extend(_nqdtc_ablation())
    if strategy_filter in ("all", "vdubus"):
        all_experiments.extend(_vdubus_ablation())

    # Priority 2: Parameter Sweeps
    if strategy_filter in ("all", "nqdtc"):
        all_experiments.extend(_nqdtc_param_sweeps())
    if strategy_filter in ("all", "vdubus"):
        all_experiments.extend(_vdubus_param_sweeps())

    # Priority 3: Interactions
    if strategy_filter in ("all", "nqdtc"):
        all_experiments.extend(_nqdtc_interactions())
    if strategy_filter in ("all", "vdubus"):
        all_experiments.extend(_vdubus_interactions())

    # Priority 4: Portfolio (always included for "all" or "portfolio")
    if strategy_filter in ("all", "portfolio"):
        all_experiments.extend(_portfolio_param_sweeps())
        all_experiments.extend(_portfolio_allocation_sweeps())
        all_experiments.extend(_portfolio_drawdown_tiers())
        all_experiments.extend(_portfolio_strategy_exclusion())
        all_experiments.extend(_portfolio_preset_comparisons())

    # Priority 5: Diagnostic (always included for "all" or "portfolio")
    if strategy_filter in ("all", "portfolio"):
        all_experiments.extend(_diagnostic_experiments())

    # Sort by priority then ID for deterministic ordering
    all_experiments.sort(key=lambda e: (e.priority, e.id))

    return all_experiments


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queue = build_experiment_queue()
    print(f"Total experiments: {len(queue)}")
    by_type: dict[str, int] = {}
    by_strategy: dict[str, int] = {}
    by_priority: dict[int, int] = {}
    for exp in queue:
        by_type[exp.type] = by_type.get(exp.type, 0) + 1
        by_strategy[exp.strategy] = by_strategy.get(exp.strategy, 0) + 1
        by_priority[exp.priority] = by_priority.get(exp.priority, 0) + 1

    print("\nBy type:")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")

    print("\nBy strategy:")
    for s, c in sorted(by_strategy.items()):
        print(f"  {s}: {c}")

    print("\nBy priority:")
    for p, c in sorted(by_priority.items()):
        print(f"  P{p}: {c}")
