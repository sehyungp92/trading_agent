"""Experiment definitions and priority queue for auto backtesting."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    id: str                       # e.g. "abl_alcb_regime_gate"
    type: str                     # ABLATION | PARAM_SWEEP | INTERACTION | PORTFOLIO
    strategy: str                 # "alcb" | "iaric" | "portfolio"
    tier: int                     # 2 (default), 1 for flags only wired in Tier 1
    description: str
    hypothesis: str
    priority: int                 # 1=highest
    mutations: dict = field(default_factory=dict)


def build_experiment_queue(strategy_filter: str = "all") -> list[Experiment]:
    """Build priority-ordered experiment queue.

    Args:
        strategy_filter: "alcb", "iaric", or "all"
    """
    experiments: list[Experiment] = []

    if strategy_filter in ("all", "alcb"):
        experiments.extend(_alcb_ablation())
    if strategy_filter in ("all", "iaric"):
        experiments.extend(_iaric_ablation())
    if strategy_filter in ("all", "alcb"):
        experiments.extend(_alcb_param_sweeps())
    if strategy_filter in ("all", "iaric"):
        experiments.extend(_iaric_param_sweeps())
    if strategy_filter in ("all", "alcb"):
        experiments.extend(_alcb_interactions())
    if strategy_filter in ("all", "iaric"):
        experiments.extend(_iaric_interactions())
    if strategy_filter in ("all", "iaric"):
        experiments.extend(_iaric_structural())
    if strategy_filter in ("all", "iaric"):
        experiments.extend(_iaric_t2_tuning())
    if strategy_filter in ("all", "iaric"):
        experiments.extend(_iaric_t2_structural_v2())
    if strategy_filter == "all":
        experiments.extend(_portfolio_experiments())

    return sorted(experiments, key=lambda e: (e.priority, e.id))


# ---------------------------------------------------------------------------
# Priority 1: Ablation experiments
# ---------------------------------------------------------------------------

def _alcb_ablation() -> list[Experiment]:
    """ALCB T1 momentum continuation ablation experiments."""
    base = []

    # Core gates
    base.append(Experiment(
        id="abl_alcb_regime_gate", type="ABLATION", strategy="alcb", tier=2,
        description="Disable regime gate",
        hypothesis="Tier C may have valid momentum trades",
        priority=1, mutations={"ablation.use_regime_gate": False},
    ))
    base.append(Experiment(
        id="abl_alcb_sector_limit", type="ABLATION", strategy="alcb", tier=2,
        description="Disable sector limit",
        hypothesis="Sector cap may reject hot sectors with clustering momentum",
        priority=1, mutations={"ablation.use_sector_limit": False},
    ))
    base.append(Experiment(
        id="abl_alcb_heat_cap", type="ABLATION", strategy="alcb", tier=2,
        description="Disable heat cap",
        hypothesis="Heat cap may be too conservative for intraday momentum",
        priority=1, mutations={"ablation.use_heat_cap": False},
    ))
    base.append(Experiment(
        id="abl_alcb_long_only", type="ABLATION", strategy="alcb", tier=2,
        description="Disable long-only mode (allow shorts)",
        hypothesis="Shorts may capture gap-down momentum",
        priority=1, mutations={"ablation.use_long_only": False},
    ))

    # Entry filters
    base.append(Experiment(
        id="abl_alcb_rvol_filter", type="ABLATION", strategy="alcb", tier=2,
        description="Disable RVOL filter",
        hypothesis="RVOL may be redundant with momentum score",
        priority=1, mutations={"ablation.use_rvol_filter": False},
    ))
    base.append(Experiment(
        id="abl_alcb_cpr_filter", type="ABLATION", strategy="alcb", tier=2,
        description="Disable CPR filter",
        hypothesis="CPR gate may be too aggressive on breakout bars",
        priority=1, mutations={"ablation.use_cpr_filter": False},
    ))
    base.append(Experiment(
        id="abl_alcb_avwap_filter", type="ABLATION", strategy="alcb", tier=2,
        description="Disable AVWAP filter",
        hypothesis="AVWAP may filter profitable late-day entries",
        priority=1, mutations={"ablation.use_avwap_filter": False},
    ))
    base.append(Experiment(
        id="abl_alcb_momentum_score", type="ABLATION", strategy="alcb", tier=2,
        description="Disable momentum score gate",
        hypothesis="Score gate may over-filter valid breakouts",
        priority=1, mutations={"ablation.use_momentum_score_gate": False},
    ))
    base.append(Experiment(
        id="abl_alcb_pdh_breakout", type="ABLATION", strategy="alcb", tier=2,
        description="Disable prior-day-high breakout",
        hypothesis="OR-only may be cleaner signal",
        priority=1, mutations={"ablation.use_prior_day_high_breakout": False},
    ))

    # Exit mechanisms
    base.append(Experiment(
        id="abl_alcb_flow_reversal", type="ABLATION", strategy="alcb", tier=2,
        description="Disable flow reversal exit",
        hypothesis="Quantify flow reversal edge vs holding to stop",
        priority=1, mutations={"ablation.use_flow_reversal_exit": False},
    ))
    base.append(Experiment(
        id="abl_alcb_carry_logic", type="ABLATION", strategy="alcb", tier=2,
        description="Disable carry logic (flatten all EOD)",
        hypothesis="Quantify overnight carry edge",
        priority=1, mutations={"ablation.use_carry_logic": False},
    ))
    base.append(Experiment(
        id="abl_alcb_partial_takes", type="ABLATION", strategy="alcb", tier=2,
        description="Disable partial takes",
        hypothesis="Partials may reduce overall R by cutting winners",
        priority=1, mutations={"ablation.use_partial_takes": False},
    ))

    return base


def _iaric_ablation() -> list[Experiment]:
    """IARIC ablation experiments for both T1 and T2.

    T1 diagnostic baseline (337 trades, $10K equity):
      51% WR, PF 2.01, Sharpe 3.02, +121.77R, $3,745 PnL
      Exits: CARRY 40% ($4,262), EOD_FLATTEN 29% ($419), FLOW_REV 11% ($1,101), STOP 20% (-$2,038)
    """
    base = []

    # --- T1 ablations (all 5 wired flags) ---
    base.append(Experiment(
        id="abl_iaric_t1_regime_gate", type="ABLATION", strategy="iaric", tier=1,
        description="Disable regime gate (T1)",
        hypothesis="Regime C skip may reject valid setups; disabling adds trades in weak markets",
        priority=1, mutations={"ablation.use_regime_gate": False},
    ))
    base.append(Experiment(
        id="abl_iaric_t1_carry_logic", type="ABLATION", strategy="iaric", tier=1,
        description="Disable carry logic (T1)",
        hypothesis="Carry is #1 profit source ($4,262/134 trades); disabling quantifies overnight edge",
        priority=1, mutations={"ablation.use_carry_logic": False},
    ))
    base.append(Experiment(
        id="abl_iaric_t1_sector_limit", type="ABLATION", strategy="iaric", tier=1,
        description="Disable sector limit (T1)",
        hypothesis="W9: Tech dominates 4/5 worst DD episodes; sector limits may or may not help",
        priority=1, mutations={"ablation.use_sector_limit": False},
    ))
    base.append(Experiment(
        id="abl_iaric_t1_flow_reversal", type="ABLATION", strategy="iaric", tier=1,
        description="Disable flow reversal exit (T1)",
        hypothesis="Flow reversal has 81.6% WR (+$1,101); disabling tests if exits are premature",
        priority=1, mutations={"ablation.use_flow_reversal_exit": False},
    ))
    base.append(Experiment(
        id="abl_iaric_t1_conviction", type="ABLATION", strategy="iaric", tier=1,
        description="Disable conviction scaling (T1)",
        hypothesis="W7: Conviction system shows '?' for all trades; may undersize good setups",
        priority=1, mutations={"ablation.use_conviction_scaling": False},
    ))

    # --- T2 ablations (wired in T2 engine only) ---
    base.append(Experiment(
        id="abl_iaric_regime_gate", type="ABLATION", strategy="iaric", tier=2,
        description="Disable regime gate (T2)",
        hypothesis="May reject valid setups",
        priority=1, mutations={"ablation.use_regime_gate": False},
    ))
    base.append(Experiment(
        id="abl_iaric_carry_logic", type="ABLATION", strategy="iaric", tier=2,
        description="Disable carry logic (T2)",
        hypothesis="Overnight risk (key weakness)",
        priority=1, mutations={"ablation.use_carry_logic": False},
    ))
    base.append(Experiment(
        id="abl_iaric_sector_limit", type="ABLATION", strategy="iaric", tier=2,
        description="Disable sector limit (T2)",
        hypothesis="Concentration limits",
        priority=1, mutations={"ablation.use_sector_limit": False},
    ))
    base.append(Experiment(
        id="abl_iaric_time_stop", type="ABLATION", strategy="iaric", tier=2,
        description="Disable time stop (T2)",
        hypothesis="May cut intraday winners",
        priority=1, mutations={"ablation.use_time_stop": False},
    ))
    base.append(Experiment(
        id="abl_iaric_partial_take", type="ABLATION", strategy="iaric", tier=2,
        description="Disable partial take (T2)",
        hypothesis="Calibration question",
        priority=1, mutations={"ablation.use_partial_take": False},
    ))

    # Unwired flags — expect delta=0
    base.append(Experiment(
        id="abl_iaric_sponsorship", type="ABLATION", strategy="iaric", tier=2,
        description="Disable sponsorship filter (expected unwired)",
        hypothesis="Expect delta=0",
        priority=1, mutations={"ablation.use_sponsorship_filter": False},
    ))
    base.append(Experiment(
        id="abl_iaric_avwap_exit", type="ABLATION", strategy="iaric", tier=2,
        description="Disable AVWAP breakdown exit (expected unwired)",
        hypothesis="Expect delta=0",
        priority=1, mutations={"ablation.use_avwap_breakdown_exit": False},
    ))

    return base


# ---------------------------------------------------------------------------
# Priority 2: Parameter sweeps (conditional on ablation results)
# ---------------------------------------------------------------------------

def _alcb_param_sweeps() -> list[Experiment]:
    """ALCB T1 momentum continuation parameter sweeps."""
    sweeps = []

    alcb_params = {
        # Opening range
        "opening_range_bars": [3, 6, 9, 12],
        # Entry filters
        "rvol_threshold": [1.0, 1.3, 1.5, 2.0],
        "cpr_threshold": [0.5, 0.6, 0.7],
        # Risk sizing
        "base_risk_fraction": [0.010, 0.012, 0.015, 0.020],
        # Stop placement
        "stop_atr_multiple": [0.5, 0.75, 1.0, 1.5],
        # Partial takes
        "partial_r_trigger": [0.75, 1.0, 1.5, 2.0],
        "partial_fraction": [0.30, 0.50, 0.70],
        # Momentum score gate
        "momentum_score_min": [2, 3, 4, 5],
        # Entry window
        "entry_window_end": ["10:00", "11:00", "12:00", "13:00"],
        # Portfolio limits
        "max_positions": [3, 5, 8],
        "max_positions_per_sector": [1, 2, 3],
        # EOD timing (minutes before close)
        "eod_flatten_time": ["15:30", "15:45", "15:55"],
        # Carry
        "carry_min_r": [0.3, 0.5, 1.0],
        "max_carry_days": [1, 2, 3],
    }

    hyp = {
        "opening_range_bars": "15/30/45/60 min OR window",
        "rvol_threshold": "Volume confirmation strength",
        "cpr_threshold": "Bar close quality gate",
        "base_risk_fraction": "Risk per trade",
        "stop_atr_multiple": "Stop width in ATR",
        "partial_r_trigger": "When to take partial profit",
        "partial_fraction": "How much to take at partial",
        "momentum_score_min": "Entry quality gate",
        "entry_window_end": "Latest entry time",
        "max_positions": "Portfolio breadth",
        "max_positions_per_sector": "Sector concentration",
        "eod_flatten_time": "Minutes before close to flatten",
        "carry_min_r": "Minimum R for overnight carry",
        "max_carry_days": "Maximum overnight carry duration",
    }

    for param_name, values in alcb_params.items():
        h = hyp.get(param_name, f"Sweep {param_name}")
        for val in values:
            sweeps.append(Experiment(
                id=f"sweep_alcb_{param_name}_{val}",
                type="PARAM_SWEEP", strategy="alcb", tier=2,
                description=f"ALCB {param_name}={val}",
                hypothesis=h,
                priority=2,
                mutations={f"param_overrides.{param_name}": val},
            ))

    return sweeps


def _iaric_param_sweeps() -> list[Experiment]:
    """IARIC T1 parameter sweep experiments informed by diagnostic findings.

    Diagnostic weaknesses driving the sweep design:
    W1: 20% stop hits = pure drag (-$2,038); sweep stop_risk_cap_pct
    W2: 29% EOD_FLATTEN = marginal ($419 on 97 trades); sweep conviction filter
    W3: Tuesday PF=1.08 (+1.94R on 64 trades); test day-of-week sizing
    W4: Financials worst sector (Mean R +0.107); sweep sector limits
    W5: Sizing at 0.25% may be too conservative for Sharpe 3+ strategy
    W6: Carry is #1 source ($4,262) but capped at 3 days; sweep duration
    W7: Conviction '?' for all trades; sweep multiplier thresholds
    W8: 81% conversion rate suggests loose selection; sweep regime/breadth gates
    W9: Tech dominates drawdowns; sweep sector concentration caps
    """
    sweeps = []

    hyp = {
        # W1: Stop risk calibration — 68 stop hits at exactly -1R each
        "stop_risk_cap_pct": "W1: Default 2% cap; tighter = smaller loss per stop but more stops",
        # W5: Position sizing — Sharpe 3+ supports more aggressive sizing
        "base_risk_fraction": "W5: Default 0.25%; strategy edge supports more",
        # W6: Carry duration — #1 profit source at 40% of trades
        "max_carry_days": "W6: Default 3d; longer carry may capture more trend",
        # W6: Carry quality gate — carry at 0R may include marginal positions
        "min_carry_r": "W6: Require minimum R to qualify for carry; filters marginal holds",
        # W7: Conviction entry gate — currently any positive conviction enters
        "min_conviction_multiplier": "W2+W7: Filter low-conviction entries that become EOD_FLATTEN",
        # W8: Flow reversal sensitivity — 38 exits at 81.6% WR
        "flow_reversal_lookback": "W8: Default 2 days; longer = less sensitive exit trigger",
        # Selection pipeline: regime and breadth thresholds
        "tier_a_min": "W8: Regime A threshold; higher = fewer but higher-quality trading days",
        "tier_b_min": "W8: Regime B threshold; higher = stricter B qualification",
        "breadth_threshold_pct": "W8: Market breadth gate; higher = more selective market filter",
        # W9: Position and sector limits
        "max_positions_per_sector": "W9: Tech concentration in drawdowns; tighter sector cap",
        # Selection sensitivity
        "anchor_lookback_sessions": "Anchor search window; wider may find better setups",
        "avwap_acceptance_band_pct": "AVWAP acceptance width; wider = more entries",
    }

    # --- W1: Stop risk calibration ---
    for val in [0.010, 0.015, 0.025, 0.030]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_stop_cap_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC stop_risk_cap_pct={val}",
            hypothesis=hyp["stop_risk_cap_pct"],
            priority=2,
            mutations={"param_overrides.stop_risk_cap_pct": val},
        ))

    # --- W5: Position sizing ---
    for val in [0.0035, 0.005, 0.0075, 0.010, 0.015]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_risk_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC base_risk_fraction={val}",
            hypothesis=hyp["base_risk_fraction"],
            priority=2,
            mutations={"param_overrides.base_risk_fraction": val},
        ))

    # --- W6: Carry duration ---
    for val in [1, 2, 5, 7, 10]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_carry_days_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC max_carry_days={val}",
            hypothesis=hyp["max_carry_days"],
            priority=2,
            mutations={"param_overrides.max_carry_days": val},
        ))

    # --- W6: Carry quality gate ---
    for val in [0.25, 0.50, 0.75, 1.0]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_carry_minr_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC min_carry_r={val}",
            hypothesis=hyp["min_carry_r"],
            priority=2,
            mutations={"param_overrides.min_carry_r": val},
        ))

    # --- W2+W7: Conviction entry gate ---
    for val in [0.25, 0.50, 0.75, 1.0]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_min_conv_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC min_conviction_multiplier={val}",
            hypothesis=hyp["min_conviction_multiplier"],
            priority=2,
            mutations={"param_overrides.min_conviction_multiplier": val},
        ))

    # --- W8: Flow reversal sensitivity ---
    for val in [1, 3, 4, 5]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_flow_lookback_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC flow_reversal_lookback={val}",
            hypothesis=hyp["flow_reversal_lookback"],
            priority=2,
            mutations={"param_overrides.flow_reversal_lookback": val},
        ))

    # --- W8: Regime thresholds ---
    for val in [0.55, 0.60, 0.70, 0.75, 0.80]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_tier_a_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC tier_a_min={val}",
            hypothesis=hyp["tier_a_min"],
            priority=2,
            mutations={"param_overrides.tier_a_min": val},
        ))
    for val in [0.30, 0.35, 0.45, 0.50]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_tier_b_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC tier_b_min={val}",
            hypothesis=hyp["tier_b_min"],
            priority=2,
            mutations={"param_overrides.tier_b_min": val},
        ))

    # --- W8: Breadth gate ---
    for val in [45.0, 50.0, 60.0, 65.0]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_breadth_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC breadth_threshold_pct={val}",
            hypothesis=hyp["breadth_threshold_pct"],
            priority=2,
            mutations={"param_overrides.breadth_threshold_pct": val},
        ))

    # --- W9: Sector limits (via config top-level) ---
    for val in [1, 2, 4, 5]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_sector_lim_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC max_per_sector={val}",
            hypothesis=hyp["max_positions_per_sector"],
            priority=2,
            mutations={"max_per_sector": val},
        ))

    # --- Position limits ---
    for val in [4, 6, 10, 12]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_maxpos_a_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC max_positions_tier_a={val}",
            hypothesis="Position limit for Tier A regimes",
            priority=2,
            mutations={"max_positions_tier_a": val},
        ))
    for val in [2, 3, 6, 8]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_maxpos_b_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC max_positions_tier_b={val}",
            hypothesis="Position limit for Tier B regimes",
            priority=2,
            mutations={"max_positions_tier_b": val},
        ))

    # --- Selection sensitivity ---
    for val in [20, 30, 50, 60]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_anchor_lb_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC anchor_lookback_sessions={val}",
            hypothesis=hyp["anchor_lookback_sessions"],
            priority=2,
            mutations={"param_overrides.anchor_lookback_sessions": val},
        ))
    for val in [0.005, 0.015, 0.020, 0.030]:
        sweeps.append(Experiment(
            id=f"sweep_iaric_avwap_accept_{val}", type="PARAM_SWEEP",
            strategy="iaric", tier=1,
            description=f"IARIC avwap_acceptance_band_pct={val}",
            hypothesis=hyp["avwap_acceptance_band_pct"],
            priority=2,
            mutations={"param_overrides.avwap_acceptance_band_pct": val},
        ))

    # --- T2 sweeps (retained for when T2 data available) ---
    t2_params = {
        "partial_r_multiple": [1.0, 1.5, 2.0, 2.5],
        "partial_exit_fraction": [0.25, 0.33, 0.50, 0.67],
        "time_stop_minutes": [180, 240, 300, 360],
    }
    for param_name, values in t2_params.items():
        for val in values:
            sweeps.append(Experiment(
                id=f"sweep_iaric_{param_name}_{val}",
                type="PARAM_SWEEP", strategy="iaric", tier=2,
                description=f"IARIC {param_name}={val}",
                hypothesis=f"T2 sweep {param_name}",
                priority=2,
                mutations={f"param_overrides.{param_name}": val},
            ))

    return sweeps


# ---------------------------------------------------------------------------
# Priority 3: Interaction effects (targeted pairs, 2×2 design)
# ---------------------------------------------------------------------------

def _alcb_interactions() -> list[Experiment]:
    """ALCB T1 momentum interaction experiments."""
    experiments = []

    # --- Ablation flag pairs ---
    pairs = [
        ("rvol_filter", "avwap_filter",
         "RVOL and AVWAP may be redundant entry gates"),
        ("flow_reversal_exit", "carry_logic",
         "Flow reversal and carry interact on EOD decisions"),
        ("regime_gate", "sector_limit",
         "Regime and sector gates may compound over-filtering"),
        ("prior_day_high_breakout", "momentum_score_gate",
         "PDH breakout type and score gate interaction"),
    ]

    for flag_a, flag_b, hypothesis in pairs:
        for val_a in [True, False]:
            for val_b in [True, False]:
                if val_a and val_b:
                    continue  # skip baseline (both on)
                experiments.append(Experiment(
                    id=f"int_alcb_{flag_a}_{val_a}_{flag_b}_{val_b}",
                    type="INTERACTION", strategy="alcb", tier=2,
                    description=f"ALCB {flag_a}={val_a} × {flag_b}={val_b}",
                    hypothesis=hypothesis,
                    priority=3,
                    mutations={
                        f"ablation.use_{flag_a}": val_a,
                        f"ablation.use_{flag_b}": val_b,
                    },
                ))

    # --- Part 2: Compound parameter experiments ---
    compound = [
        ("or6_window12", "OR=30min + entry window to 12:00",
         "Standard OR with extended entry window",
         {"param_overrides.opening_range_bars": 6,
          "param_overrides.entry_window_end": "12:00"}),
        ("or9_window13", "OR=45min + entry window to 13:00",
         "Wider OR with late entries",
         {"param_overrides.opening_range_bars": 9,
          "param_overrides.entry_window_end": "13:00"}),
        ("tight_stop_early_partial", "Stop=0.5ATR + partial at 0.75R",
         "Tighter stop with earlier profit lock",
         {"param_overrides.stop_atr_multiple": 0.5,
          "param_overrides.partial_r_trigger": 0.75}),
        ("wide_stop_late_partial", "Stop=1.5ATR + partial at 2.0R",
         "Wider stop allowing more room, later partial",
         {"param_overrides.stop_atr_multiple": 1.5,
          "param_overrides.partial_r_trigger": 2.0}),
        ("aggressive_carry", "Carry min_r=0.3 + max_days=3",
         "More aggressive overnight carry",
         {"param_overrides.carry_min_r": 0.3,
          "param_overrides.max_carry_days": 3}),
        ("score4_rvol2", "Momentum score >= 4 + RVOL >= 2.0",
         "Higher quality gate on both dimensions",
         {"param_overrides.momentum_score_min": 4,
          "param_overrides.rvol_threshold": 2.0}),
    ]

    for eid, desc, hyp, mutations in compound:
        experiments.append(Experiment(
            id=f"int_alcb_{eid}",
            type="INTERACTION", strategy="alcb", tier=2,
            description=desc,
            hypothesis=hyp,
            priority=3,
            mutations=mutations,
        ))

    return experiments


def _iaric_interactions() -> list[Experiment]:
    """IARIC compound experiments targeting multiple weaknesses simultaneously.

    Part 1: Ablation flag pairs (T1)
    Part 2: Compound parameter experiments addressing diagnostic weaknesses
    """
    experiments = []

    # --- Part 1: Ablation flag pairs (T1) ---
    pairs = [
        ("flow_reversal_exit", "carry_logic", 1,
         "W6+W8: Flow reversal (81% WR) + carry (#1 source) interact on overnight holds"),
        ("conviction_scaling", "regime_gate", 1,
         "W7: Conviction sizing + regime filtering may compound or cancel"),
        ("sector_limit", "carry_logic", 1,
         "W9+W6: Sector cap interacts with carry concentration in Tech"),
    ]

    for flag_a, flag_b, tier, hypothesis in pairs:
        for val_a in [True, False]:
            for val_b in [True, False]:
                if val_a and val_b:
                    continue
                experiments.append(Experiment(
                    id=f"int_iaric_{flag_a}_{val_a}_{flag_b}_{val_b}",
                    type="INTERACTION", strategy="iaric", tier=tier,
                    description=f"IARIC {flag_a}={val_a} x {flag_b}={val_b}",
                    hypothesis=hypothesis,
                    priority=3,
                    mutations={
                        f"ablation.use_{flag_a}": val_a,
                        f"ablation.use_{flag_b}": val_b,
                    },
                ))

    # --- Part 2: Compound parameter experiments ---
    compound = [
        # W5+W6: Aggressive sizing + extended carry — maximize the core edge
        ("aggressive_carry",
         "Carry-only + close stop + risk 0.5% + carry 5d",
         "W5+W6: Carry is #1 source; eliminate drag + go bigger and hold longer",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.base_risk_fraction": 0.005,
          "param_overrides.max_carry_days": 5}),
        ("max_carry_edge",
         "Larger risk (0.75%) + longer carry (7d) + carry gate 0.25R",
         "W5+W6: Push carry edge harder with quality gate to filter marginal overnight holds",
         {"param_overrides.base_risk_fraction": 0.0075,
          "param_overrides.max_carry_days": 7,
          "param_overrides.min_carry_r": 0.25}),

        # W1+W5: Wider stops + bigger sizing — lose less per stop, earn more per winner
        ("wider_stop_bigger_size",
         "Stop cap 3% + risk fraction 0.5%",
         "W1+W5: Wider stops reduce stop-hit rate (20%); bigger sizing captures more per trade",
         {"param_overrides.stop_risk_cap_pct": 0.03,
          "param_overrides.base_risk_fraction": 0.005}),
        ("tighter_stop_bigger_size",
         "Stop cap 1.5% + risk fraction 0.75%",
         "W1+W5: Tighter stops = smaller $ per stop + bigger sizing = more $ per winner",
         {"param_overrides.stop_risk_cap_pct": 0.015,
          "param_overrides.base_risk_fraction": 0.0075}),

        # W2+W7: Quality filter — reduce marginal EOD_FLATTEN trades
        ("conviction_quality_gate",
         "Min conviction 0.5 + tighter regime (tier_a_min=0.70)",
         "W2+W7: Filter low-conviction entries that become marginal EOD_FLATTEN trades",
         {"param_overrides.min_conviction_multiplier": 0.50,
          "param_overrides.tier_a_min": 0.70}),
        ("strict_quality_gate",
         "Min conviction 0.75 + breadth 60% + tier_a_min 0.75",
         "W2+W7+W8: Maximum selectivity — fewer but much higher quality entries",
         {"param_overrides.min_conviction_multiplier": 0.75,
          "param_overrides.breadth_threshold_pct": 60.0,
          "param_overrides.tier_a_min": 0.75}),

        # W9: Tech drawdown mitigation
        ("tech_cap_tight",
         "Sector limit 2 + maxpos_a 6",
         "W9: Tech dominates 4/5 worst DD; tighter sector + fewer positions reduces concentration",
         {"max_per_sector": 2, "max_positions_tier_a": 6}),

        # W6+W8: Extended carry with stricter flow exit
        ("long_carry_strict_flow",
         "Carry 7d + flow lookback 3 + min carry R 0.5",
         "W6+W8: Hold longer but require 3 negative flow days and 0.5R to qualify",
         {"param_overrides.max_carry_days": 7,
          "param_overrides.flow_reversal_lookback": 3,
          "param_overrides.min_carry_r": 0.50}),

        # W8: Wider funnel — test whether more trades improve diversification
        ("wider_funnel",
         "Breadth 45% + tier_a_min 0.55 + maxpos_a 12 + anchor_lookback 50",
         "W8: Looser selection may add marginal trades but improve diversification",
         {"param_overrides.breadth_threshold_pct": 45.0,
          "param_overrides.tier_a_min": 0.55,
          "max_positions_tier_a": 12,
          "param_overrides.anchor_lookback_sessions": 50}),

        # Full recalibration — address all weaknesses together
        ("full_recalibration",
         "Carry-only + close stop + risk 0.5% + carry 5d + carry_r 0.25 + conv 0.25 + sector 2",
         "W1-W9: Comprehensive recalibration with structural alpha features",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.base_risk_fraction": 0.005,
          "param_overrides.max_carry_days": 5,
          "param_overrides.min_carry_r": 0.25,
          "param_overrides.min_conviction_multiplier": 0.25,
          "max_per_sector": 2}),
        ("aggressive_recalibration",
         "Carry-only + close stop + risk 1% + carry 7d + carry_r 0.5 + conv 0.5 + flow_lb 3",
         "Aggressive version: structural alpha + bigger sizing, longer carry, stricter quality gates",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.base_risk_fraction": 0.010,
          "param_overrides.max_carry_days": 7,
          "param_overrides.min_carry_r": 0.50,
          "param_overrides.stop_risk_cap_pct": 0.015,
          "param_overrides.min_conviction_multiplier": 0.50,
          "param_overrides.flow_reversal_lookback": 3}),
    ]

    for eid, desc, hyp, mutations in compound:
        experiments.append(Experiment(
            id=f"int_iaric_{eid}",
            type="INTERACTION", strategy="iaric", tier=1,
            description=desc,
            hypothesis=hyp,
            priority=3,
            mutations=mutations,
        ))

    # --- T2 ablation pairs (retained) ---
    t2_pairs = [
        ("carry_logic", "regime_gate", 2,
         "Carry + regime filtering interact"),
        ("time_stop", "partial_take", 2,
         "Holding period vs profit taking interact"),
    ]
    for flag_a, flag_b, tier, hypothesis in t2_pairs:
        for val_a in [True, False]:
            for val_b in [True, False]:
                if val_a and val_b:
                    continue
                experiments.append(Experiment(
                    id=f"int_iaric_{flag_a}_{val_a}_{flag_b}_{val_b}",
                    type="INTERACTION", strategy="iaric", tier=tier,
                    description=f"IARIC {flag_a}={val_a} x {flag_b}={val_b}",
                    hypothesis=hypothesis,
                    priority=3,
                    mutations={
                        f"ablation.use_{flag_a}": val_a,
                        f"ablation.use_{flag_b}": val_b,
                    },
                ))

    return experiments


# ---------------------------------------------------------------------------
# Priority 2: Structural alpha amplification (IARIC T1)
# ---------------------------------------------------------------------------


def _iaric_structural() -> list[Experiment]:
    """IARIC structural alpha amplification experiments.

    Target the core finding that carry is 100% of edge:
    - CARRY_EXIT + FLOW_REVERSAL = +$5,363 (143% of total PnL)
    - EOD_FLATTEN + STOP_HIT = -$1,619 (net drag)

    Structural features eliminate non-carry drag and match live behavior.
    """
    experiments: list[Experiment] = []

    # --- Single structural features ---
    singles = [
        ("struct_iaric_carry_only",
         "Carry-only entry: skip trades that can never carry overnight",
         "Non-carry trades (EOD_FLATTEN+STOP_HIT) lose $1,619; eliminating them is pure profit",
         {"param_overrides.carry_only_entry": True}),
        ("struct_iaric_close_stop",
         "Close-price stop: exit underwater at close instead of hard intraday stop",
         "Hard stop L<=stop is unrealistic vs live time-based stop; close stop lets recoveries play out",
         {"param_overrides.use_close_stop": True}),
        ("struct_iaric_strong_only",
         "Strong-only entry: require STRONG sponsorship to enter",
         "NEUTRAL sponsorship entries have lower conviction and can never carry; filtering reduces drag",
         {"param_overrides.strong_only_entry": True}),
        ("struct_iaric_conviction_order",
         "Conviction-ordered entry: sort tradable by conviction desc before position fill",
         "Default daily_rank ordering may fill lower-conviction trades first when max_pos binds",
         {"param_overrides.entry_order_by_conviction": True}),
        ("struct_iaric_intraday_flow",
         "Intraday flow check: exit at close if flow reversed on same-day positions",
         "Flow reversal (81% WR) only checks overnight carries; same-day positions miss this signal",
         {"param_overrides.intraday_flow_check": True}),
        ("struct_iaric_regime_b_carry",
         "Regime B carry: allow overnight carry in Regime B at 0.6x size multiplier",
         "Regime B is ~40% of trading days; STRONG+in-profit B trades may have overnight edge",
         {"param_overrides.regime_b_carry_mult": 0.6}),
        ("struct_iaric_top_quartile",
         "Top-quartile carry gate: require close in top 25% of daily range to carry",
         "Live carry_eligible() requires close_pct >= 0.75; matching this filters marginal carries",
         {"param_overrides.carry_top_quartile": True}),
    ]

    for eid, desc, hyp, mutations in singles:
        experiments.append(Experiment(
            id=eid, type="STRUCTURAL", strategy="iaric", tier=1,
            description=desc, hypothesis=hyp, priority=2,
            mutations=mutations,
        ))

    # --- Compound structural experiments ---
    compounds = [
        ("struct_iaric_carry_only_close_stop",
         "Carry-only + close-price stop: two biggest drags fixed",
         "Eliminate non-carry entries + replace hard stop with realistic close-price stop",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True}),
        ("struct_iaric_carry_only_extended",
         "Carry-only + extended carry (5d) + carry gate 0.25R",
         "Eliminate drag + extend the core carry edge with quality gate",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.max_carry_days": 5,
          "param_overrides.min_carry_r": 0.25}),
        ("struct_iaric_strong_close_topq",
         "Strong-only + close stop + top-quartile carry gate",
         "Quality gates everywhere: entry, stop, and carry",
         {"param_overrides.strong_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.carry_top_quartile": True}),
        ("struct_iaric_carry_only_regime_b",
         "Carry-only entry + Regime B carry at 0.5x",
         "Carry-focused strategy with wider carry universe via Regime B",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.regime_b_carry_mult": 0.5}),
        ("struct_iaric_full_quality",
         "All quality gates: carry-only + close stop + conviction + flow + top quartile",
         "Maximum quality filtering across entry, stop, ordering, flow, and carry gates",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.entry_order_by_conviction": True,
          "param_overrides.intraday_flow_check": True,
          "param_overrides.carry_top_quartile": True}),
        ("struct_iaric_full_aggressive",
         "All structural + risk 0.5% + carry 7d",
         "Full structural stack with bigger sizing and extended carry",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.entry_order_by_conviction": True,
          "param_overrides.intraday_flow_check": True,
          "param_overrides.carry_top_quartile": True,
          "param_overrides.regime_b_carry_mult": 0.6,
          "param_overrides.base_risk_fraction": 0.005,
          "param_overrides.max_carry_days": 7}),
        ("struct_iaric_max_extraction",
         "All structural + risk 0.75% + carry 10d + carry_r 0 + flow_lb 3",
         "Push to extraction limit: all features + aggressive sizing and carry",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.entry_order_by_conviction": True,
          "param_overrides.intraday_flow_check": True,
          "param_overrides.carry_top_quartile": True,
          "param_overrides.regime_b_carry_mult": 0.6,
          "param_overrides.base_risk_fraction": 0.0075,
          "param_overrides.max_carry_days": 10,
          "param_overrides.min_carry_r": 0.0,
          "param_overrides.flow_reversal_lookback": 3}),
        ("struct_iaric_kitchen_sink",
         "All structural + risk 1% + carry 7d + carry_r 0.25 + sector 2",
         "Theoretical maximum: every lever turned to aggressive setting",
         {"param_overrides.carry_only_entry": True,
          "param_overrides.use_close_stop": True,
          "param_overrides.entry_order_by_conviction": True,
          "param_overrides.intraday_flow_check": True,
          "param_overrides.carry_top_quartile": True,
          "param_overrides.regime_b_carry_mult": 0.6,
          "param_overrides.strong_only_entry": True,
          "param_overrides.base_risk_fraction": 0.01,
          "param_overrides.max_carry_days": 7,
          "param_overrides.min_carry_r": 0.25,
          "max_per_sector": 2}),
    ]

    for eid, desc, hyp, mutations in compounds:
        experiments.append(Experiment(
            id=eid, type="STRUCTURAL", strategy="iaric", tier=1,
            description=desc, hypothesis=hyp, priority=2,
            mutations=mutations,
        ))

    return experiments


# ---------------------------------------------------------------------------
# Priority 3: IARIC T2 intraday engine tuning
# ---------------------------------------------------------------------------

def _iaric_t2_tuning() -> list[Experiment]:
    """IARIC T2 v2 experiments for the new intraday engine.

    T2 v2 uses T1's proven watchlist + intraday execution improvements:
    adaptive trailing stops, VWAP pullback / ORB entries, carry scoring,
    day-of-week sizing, staleness timeout, PM re-entry.
    """
    experiments: list[Experiment] = []

    # --- Adaptive stop ablations ---
    experiments.append(Experiment(
        id="abl_t2v2_adaptive_stop", type="ABLATION", strategy="iaric", tier=2,
        description="T2v2 ablation: use hard -1R stop instead of adaptive trail",
        hypothesis="Adaptive stops should recover ~$2K of hard stop drag",
        priority=1,
        mutations={"param_overrides.t2_initial_atr_mult": 99.0,
                   "param_overrides.t2_breakeven_r": 99.0},
    ))

    # --- Adaptive stop param sweeps ---
    stop_sweeps = [
        ("t2v2_init_atr_1.0", 1.0, "Tighter initial stop (1.0 ATR vs 1.5)"),
        ("t2v2_init_atr_2.0", 2.0, "Wider initial stop (2.0 ATR vs 1.5)"),
        ("t2v2_be_r_0.3", None, "Earlier breakeven at +0.3R (vs +0.5R)"),
        ("t2v2_be_r_0.75", None, "Later breakeven at +0.75R (vs +0.5R)"),
        ("t2v2_trail_0.75", None, "Tighter profit trail (0.75 ATR vs 1.0)"),
        ("t2v2_trail_1.5", None, "Wider profit trail (1.5 ATR vs 1.0)"),
    ]
    for eid, val, hyp in stop_sweeps:
        if "init_atr" in eid:
            mutations = {"param_overrides.t2_initial_atr_mult": val}
        elif "be_r_0.3" in eid:
            mutations = {"param_overrides.t2_breakeven_r": 0.3}
        elif "be_r_0.75" in eid:
            mutations = {"param_overrides.t2_breakeven_r": 0.75}
        elif "trail_0.75" in eid:
            mutations = {"param_overrides.t2_profit_trail_atr": 0.75}
        elif "trail_1.5" in eid:
            mutations = {"param_overrides.t2_profit_trail_atr": 1.5}
        else:
            continue
        experiments.append(Experiment(
            id=eid, type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"T2v2 adaptive stop: {eid.split('_', 1)[1]}",
            hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    # --- Entry timing sweeps ---
    entry_sweeps = [
        ("t2v2_fallback_bar4", {"param_overrides.t2_fallback_entry_bar": 4},
         "Earlier fallback at 9:50 (bar 4 vs bar 6)"),
        ("t2v2_fallback_bar12", {"param_overrides.t2_fallback_entry_bar": 12},
         "Later fallback at 10:30 (bar 12 vs bar 6)"),
        ("t2v2_no_pm_reentry", {"param_overrides.t2_pm_reentry": False},
         "Disable afternoon re-entry for stopped-out symbols"),
    ]
    for eid, mutations, hyp in entry_sweeps:
        experiments.append(Experiment(
            id=eid, type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"T2v2 entry: {eid.split('_', 1)[1]}",
            hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    # --- Carry scoring sweeps ---
    carry_sweeps = [
        ("t2v2_carry_thresh_55", {"param_overrides.t2_carry_threshold": 55.0},
         "Lower carry threshold (55 vs 65): more carries"),
        ("t2v2_carry_thresh_75", {"param_overrides.t2_carry_threshold": 75.0},
         "Higher carry threshold (75 vs 65): more selective"),
        ("t2v2_carry_partial_35", {"param_overrides.t2_carry_partial_threshold": 35.0},
         "Lower partial carry threshold (35 vs 45): more partial carries"),
    ]
    for eid, mutations, hyp in carry_sweeps:
        experiments.append(Experiment(
            id=eid, type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"T2v2 carry: {eid.split('_', 2)[2]}",
            hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    # --- Day-of-week sizing ---
    dow_sweeps = [
        ("t2v2_tue_skip", {"param_overrides.t2_tuesday_mult": 0.0},
         "Skip Tuesdays entirely (46.8% WR poison day)"),
        ("t2v2_tue_075", {"param_overrides.t2_tuesday_mult": 0.75},
         "Smaller Tuesday reduction (0.75 vs 0.50)"),
        ("t2v2_fri_1.5", {"param_overrides.t2_friday_mult": 1.5},
         "Stronger Friday boost (1.5x vs 1.25x)"),
        ("t2v2_no_dow", {"param_overrides.t2_tuesday_mult": 1.0,
                         "param_overrides.t2_friday_mult": 1.0},
         "Disable all day-of-week adjustments"),
    ]
    for eid, mutations, hyp in dow_sweeps:
        experiments.append(Experiment(
            id=eid, type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"T2v2 DOW: {eid.split('_', 1)[1]}",
            hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    # --- Staleness sweeps ---
    stale_sweeps = [
        ("t2v2_stale_1h", {"param_overrides.t2_staleness_hours": 1.0},
         "Aggressive staleness timeout (1h vs 2h)"),
        ("t2v2_stale_3h", {"param_overrides.t2_staleness_hours": 3.0},
         "Patient staleness timeout (3h vs 2h)"),
        ("t2v2_no_stale", {"param_overrides.t2_staleness_hours": 99.0},
         "Disable staleness timeout entirely"),
    ]
    for eid, mutations, hyp in stale_sweeps:
        experiments.append(Experiment(
            id=eid, type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"T2v2 staleness: {eid.split('_', 1)[1]}",
            hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    # --- Risk sizing ---
    risk_sweeps = [
        ("t2v2_risk_0075", {"param_overrides.base_risk_fraction": 0.0075},
         "Higher risk fraction (0.75% vs 0.50%)"),
        ("t2v2_risk_010", {"param_overrides.base_risk_fraction": 0.01},
         "Higher risk fraction (1.0% vs 0.50%)"),
    ]
    for eid, mutations, hyp in risk_sweeps:
        experiments.append(Experiment(
            id=eid, type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"T2v2 risk: {eid.split('_', 1)[1]}",
            hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    # --- Compound interaction experiments ---
    compounds = [
        ("t2v2_optimal_stops",
         "T2v2 optimized stop combo: tight init + early breakeven + tight trail",
         "Combines best stop params for max stop drag reduction",
         {"param_overrides.t2_initial_atr_mult": 1.0,
          "param_overrides.t2_breakeven_r": 0.3,
          "param_overrides.t2_profit_trail_atr": 0.75}),
        ("t2v2_aggressive_carry",
         "T2v2 aggressive carry: lower threshold + partial + Friday boost",
         "Maximize carry frequency on strongest days",
         {"param_overrides.t2_carry_threshold": 55.0,
          "param_overrides.t2_carry_partial_threshold": 35.0,
          "param_overrides.t2_friday_mult": 1.5}),
        ("t2v2_conservative",
         "T2v2 conservative: wider stops + no PM reentry + higher carry bar",
         "Defensive configuration reducing whipsaw risk",
         {"param_overrides.t2_initial_atr_mult": 2.0,
          "param_overrides.t2_pm_reentry": False,
          "param_overrides.t2_carry_threshold": 75.0}),
    ]
    for eid, desc, hyp, mutations in compounds:
        experiments.append(Experiment(
            id=eid, type="INTERACTION", strategy="iaric", tier=2,
            description=desc, hypothesis=hyp, priority=3,
            mutations=mutations,
        ))

    return experiments


# ---------------------------------------------------------------------------
# Priority 2: IARIC T2 v2 structural improvements (Phase 5)
# ---------------------------------------------------------------------------

def _iaric_t2_structural_v2() -> list[Experiment]:
    """IARIC T2 v2 structural improvements targeting 3 shifts:
    1. Capture open-to-close return (like T1)
    2. Carry-first architecture (carry by default)
    3. Use intraday data for risk management, not entry
    """
    experiments: list[Experiment] = []

    # --- Shift 1: OPEN_ENTRY (highest impact) ---
    experiments.append(Experiment(
        id="t2v2_open_entry", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Enter at bar 1 (near open), capturing T1's open-to-close return",
        hypothesis="T1 enters at open with 91.4% WR on EOD_FLATTEN; T2 misses first hour",
        priority=2,
        mutations={"param_overrides.t2_open_entry": True},
    ))
    experiments.append(Experiment(
        id="t2v2_open_entry_half", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Open entry with 0.5x sizing to reduce risk while capturing return",
        hypothesis="Half size may improve risk-adjusted returns on open entries",
        priority=2,
        mutations={"param_overrides.t2_open_entry": True,
                   "param_overrides.t2_open_entry_size_mult": 0.5},
    ))
    experiments.append(Experiment(
        id="t2v2_open_entry_wide", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Open entry with 2.0 ATR stop to avoid morning volatility stops",
        hypothesis="Wider stop survives first-hour noise better",
        priority=2,
        mutations={"param_overrides.t2_open_entry": True,
                   "param_overrides.t2_open_entry_stop_atr": 2.0},
    ))

    # --- Shift 2: DEFAULT_CARRY (second highest impact) ---
    experiments.append(Experiment(
        id="t2v2_default_carry", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Carry all profitable positions by default (carry-first architecture)",
        hypothesis="Increases carry rate from 39% to 60-70%, converts EOD_FLATTEN losers",
        priority=2,
        mutations={"param_overrides.t2_default_carry_profitable": True},
    ))
    experiments.append(Experiment(
        id="t2v2_default_carry_r01", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Default carry requiring min +0.1R unrealized",
        hypothesis="Small R gate filters marginal carries",
        priority=2,
        mutations={"param_overrides.t2_default_carry_profitable": True,
                   "param_overrides.t2_default_carry_min_r": 0.1},
    ))
    experiments.append(Experiment(
        id="t2v2_default_carry_tight", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Default carry with 0.5 ATR protective stop",
        hypothesis="Tighter overnight stop reduces gap risk on weaker carries",
        priority=2,
        mutations={"param_overrides.t2_default_carry_profitable": True,
                   "param_overrides.t2_default_carry_stop_atr": 0.5},
    ))
    experiments.append(Experiment(
        id="t2v2_default_carry_wide", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Default carry with 1.5 ATR protective stop",
        hypothesis="Wider stop gives more room for overnight gap recovery",
        priority=2,
        mutations={"param_overrides.t2_default_carry_profitable": True,
                   "param_overrides.t2_default_carry_stop_atr": 1.5},
    ))

    # --- Shift 3: Entry strength scoring ---
    for gate in [30.0, 40.0, 50.0]:
        experiments.append(Experiment(
            id=f"t2v2_entry_str_{int(gate)}", type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"Entry strength gate at {int(gate)} (uses trend, RS, persistence, VWAP, OR)",
            hypothesis=f"Filters out weakest entries; 86% are zero-edge FALLBACK",
            priority=2,
            mutations={"param_overrides.t2_entry_strength_gate": gate},
        ))
    experiments.append(Experiment(
        id="t2v2_entry_str_sizing", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Scale position size by entry strength score (0-100 → 0.1-1.0x)",
        hypothesis="Size up strongest entries, size down weakest",
        priority=2,
        mutations={"param_overrides.t2_entry_strength_sizing": True},
    ))

    # --- Extended ORB window ---
    for w in [3, 5]:
        experiments.append(Experiment(
            id=f"t2v2_orb_window_{w}", type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"ORB fires bars 6-{5+w} (was bar 6 only)",
            hypothesis=f"ORB has AvgR=+1.991 but only 20 triggers; wider window captures more",
            priority=3,
            mutations={"param_overrides.t2_orb_window_bars": w},
        ))
    experiments.append(Experiment(
        id="t2v2_orb_no_bullish", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Remove bullish bar requirement from ORB trigger",
        hypothesis="CPR check may be sufficient without bullish bar gate",
        priority=3,
        mutations={"param_overrides.t2_orb_require_bullish": False},
    ))
    experiments.append(Experiment(
        id="t2v2_orb_window_3_any", type="INTERACTION", strategy="iaric", tier=2,
        description="ORB window 3 bars + no bullish requirement",
        hypothesis="Maximizes ORB trigger count for highest-edge entry type",
        priority=3,
        mutations={"param_overrides.t2_orb_window_bars": 3,
                   "param_overrides.t2_orb_require_bullish": False},
    ))

    # --- Multi-bar VWAP pullback ---
    experiments.append(Experiment(
        id="t2v2_vwap_multibar", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Accept VWAP pullback across multiple bars (dip + reclaim)",
        hypothesis="Single-bar requirement is too strict for 5m bars",
        priority=3,
        mutations={"param_overrides.t2_vwap_pullback_multibar": True},
    ))
    experiments.append(Experiment(
        id="t2v2_vwap_multibar_loose", type="INTERACTION", strategy="iaric", tier=2,
        description="Multi-bar VWAP pullback + lower volume threshold (60%)",
        hypothesis="Combined relaxation maximizes VWAP trigger rate",
        priority=3,
        mutations={"param_overrides.t2_vwap_pullback_multibar": True,
                   "param_overrides.t2_vwap_reclaim_vol_pct": 0.60},
    ))

    # --- Graduated staleness ---
    experiments.append(Experiment(
        id="t2v2_stale_tighten", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Staleness: tighten stop instead of closing (57 exits lose $1,516)",
        hypothesis="Tightening lets profitable ones survive while capping losers",
        priority=2,
        mutations={"param_overrides.t2_staleness_action": "TIGHTEN"},
    ))
    experiments.append(Experiment(
        id="t2v2_stale_skip", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Disable staleness exit entirely, let EOD/stops handle",
        hypothesis="EOD carry scoring may be sufficient without staleness",
        priority=2,
        mutations={"param_overrides.t2_staleness_action": "SKIP"},
    ))

    # --- AVWAP breakdown tighten ---
    experiments.append(Experiment(
        id="t2v2_avwap_tighten", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="AVWAP breakdown: tighten stop instead of immediate exit",
        hypothesis="44 trades at 0% WR — tightening may salvage recoveries",
        priority=2,
        mutations={"param_overrides.t2_avwap_breakdown_action": "TIGHTEN"},
    ))

    # --- Regime B sizing ---
    for mult, label in [(0.5, "half"), (0.25, "quarter")]:
        experiments.append(Experiment(
            id=f"t2v2_regime_b_{label}", type="PARAM_SWEEP", strategy="iaric", tier=2,
            description=f"Tier B sizing at {mult}x (70 trades, 21.4% WR)",
            hypothesis="Reduce exposure in uncertain regime while keeping carry optionality",
            priority=3,
            mutations={"param_overrides.t2_regime_b_sizing_mult": mult},
        ))

    # --- Breakeven on closes ---
    experiments.append(Experiment(
        id="t2v2_be_closes", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Breakeven triggers on highest close instead of intra-bar MFE",
        hypothesis="Reduces whipsaw from brief intra-bar spikes triggering breakeven",
        priority=3,
        mutations={"param_overrides.t2_breakeven_use_closes": True},
    ))

    # --- Fallback quality gates ---
    experiments.append(Experiment(
        id="t2v2_fallback_vwap", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Require price above VWAP for FALLBACK entries",
        hypothesis="VWAP gate filters out weakest fallback entries",
        priority=3,
        mutations={"param_overrides.t2_fallback_require_above_vwap": True},
    ))
    experiments.append(Experiment(
        id="t2v2_fallback_mom2", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Require 2 consecutive green bars before FALLBACK entry",
        hypothesis="Momentum filter removes entries into declining price",
        priority=3,
        mutations={"param_overrides.t2_fallback_momentum_bars": 2},
    ))
    experiments.append(Experiment(
        id="t2v2_fallback_mom3", type="PARAM_SWEEP", strategy="iaric", tier=2,
        description="Require 3 consecutive green bars before FALLBACK entry",
        hypothesis="Stricter momentum filter for higher-quality fallback entries",
        priority=3,
        mutations={"param_overrides.t2_fallback_momentum_bars": 3},
    ))
    experiments.append(Experiment(
        id="t2v2_fallback_vwap_mom2", type="INTERACTION", strategy="iaric", tier=2,
        description="VWAP gate + 2 green bars for FALLBACK entry",
        hypothesis="Combined filters maximize fallback quality",
        priority=3,
        mutations={"param_overrides.t2_fallback_require_above_vwap": True,
                   "param_overrides.t2_fallback_momentum_bars": 2},
    ))

    # --- AVWAP disable ---
    experiments.append(Experiment(
        id="t2v2_avwap_disable", type="ABLATION", strategy="iaric", tier=2,
        description="Disable AVWAP breakdown exit entirely",
        hypothesis="44 trades at 0% WR — removal may improve by avoiding premature exits",
        priority=2,
        mutations={"ablation.use_avwap_breakdown_exit": False},
    ))

    # --- Compound structural bundles ---
    experiments.append(Experiment(
        id="t2v2_shift1_shift2", type="INTERACTION", strategy="iaric", tier=2,
        description="Open entry + default carry (both structural shifts combined)",
        hypothesis="Captures open return AND carries it overnight = T1-like behavior",
        priority=2,
        mutations={"param_overrides.t2_open_entry": True,
                   "param_overrides.t2_default_carry_profitable": True},
    ))
    experiments.append(Experiment(
        id="t2v2_all_shifts", type="INTERACTION", strategy="iaric", tier=2,
        description="All 3 shifts: open entry + default carry + entry strength gate",
        hypothesis="Full structural transformation targets >$14K net",
        priority=2,
        mutations={"param_overrides.t2_open_entry": True,
                   "param_overrides.t2_default_carry_profitable": True,
                   "param_overrides.t2_entry_strength_gate": 30.0},
    ))

    return experiments


# ---------------------------------------------------------------------------
# Priority 4: Portfolio integration
# ---------------------------------------------------------------------------

def _portfolio_experiments() -> list[Experiment]:
    experiments = []

    # Family directional cap sweep
    for val in [6.0, 7.0, 8.0, 9.0, 10.0]:
        experiments.append(Experiment(
            id=f"pf_dir_cap_{val}",
            type="PORTFOLIO", strategy="portfolio", tier=2,
            description=f"Portfolio family_directional_cap_r={val}",
            hypothesis="Optimal directional cap",
            priority=4,
            mutations={"family_directional_cap_r": val},
        ))

    # Combined heat cap sweep
    for val in [8.0, 9.0, 10.0, 11.0, 12.0]:
        experiments.append(Experiment(
            id=f"pf_heat_cap_{val}",
            type="PORTFOLIO", strategy="portfolio", tier=2,
            description=f"Portfolio combined_heat_cap_r={val}",
            hypothesis="Optimal heat cap",
            priority=4,
            mutations={"combined_heat_cap_r": val},
        ))

    # Symbol collision toggle
    for val in [True, False]:
        experiments.append(Experiment(
            id=f"pf_collision_{val}",
            type="PORTFOLIO", strategy="portfolio", tier=2,
            description=f"Portfolio symbol_collision_half_size={val}",
            hypothesis="Half-size vs full-size on collision",
            priority=4,
            mutations={"symbol_collision_half_size": val},
        ))

    return experiments
