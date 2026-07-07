"""Downturn 3-decision loop analyzer — improve_scoring / improve_diagnostics / advance.

3 improvements over the previous pattern:
  A. Cross-engine delta tracking (per-engine health assessment)
  B. Correction-window attribution (split PnL into correction vs non-correction)
  C. Engine-routed experiment suggestions (target experiments to weak engines)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics
from backtests.momentum.auto.downturn.phase_diagnostics import get_diagnostic_gaps
from backtests.momentum.auto.downturn.phase_gates import GateResult, check_phase_gate
from strategies.momentum.downturn.bt_models import EngineTag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ultimate targets
# ---------------------------------------------------------------------------

ULTIMATE_TARGETS = {
    "correction_pnl_pct": 25.0,
    "profit_factor": 2.0,
    "net_return_pct": 40.0,
    "max_dd_pct": 15.0,   # lower is better
    "calmar": 2.0,
    "sharpe": 0.8,
    "exit_efficiency": 0.40,
    "signal_to_entry_ratio": 0.25,
    "total_trades": 120,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Signal Detection", ["signal_to_entry_ratio", "correction_pnl_pct", "total_trades"]),
    2: ("Capture", ["exit_efficiency", "profit_factor", "correction_pnl_pct"]),
    3: ("Risk Control", ["calmar", "max_dd_pct", "sharpe"]),
    4: ("Fine-tuning", ["calmar", "net_return_pct", "correction_pnl_pct"]),
}


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass
class PhaseAnalysis:
    """Result of post-phase analysis."""
    phase: int
    goal_progress: dict[str, dict] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    engine_health: dict[str, str] = field(default_factory=dict)
    correction_attribution: dict[str, float] = field(default_factory=dict)
    scoring_assessment: str = ""
    diagnostic_gaps: list[str] = field(default_factory=list)
    suggested_experiments: list[tuple[str, dict]] = field(default_factory=list)
    recommendation: str = ""
    recommendation_reason: str = ""
    scoring_weight_overrides: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze_phase(
    phase: int,
    metrics: DownturnMetrics,
    gate_result: GateResult,
    greedy_result: dict | None = None,
    state_dict: dict | None = None,
    all_trades: list | None = None,
    scoring_retries: int = 0,
    diagnostic_retries: int = 0,
) -> PhaseAnalysis:
    """Run post-phase analysis with 3-decision loop."""
    analysis = PhaseAnalysis(phase=phase)

    # Goal progress
    analysis.goal_progress = _compute_goal_progress(metrics)

    # Strengths/weaknesses
    analysis.strengths, analysis.weaknesses = _assess_strengths_weaknesses(metrics)

    # Improvement A: Engine health assessment
    analysis.engine_health = _assess_engine_health(metrics)

    # Improvement B: Correction attribution
    analysis.correction_attribution = _compute_correction_attribution(metrics, all_trades)

    # Scoring assessment
    analysis.scoring_assessment = _assess_scoring(
        metrics, gate_result, greedy_result,
    )

    # Diagnostic gaps
    analysis.diagnostic_gaps = get_diagnostic_gaps(phase, metrics)

    # Improvement C: Engine-routed experiment suggestions
    analysis.suggested_experiments = _suggest_experiments(
        phase, metrics, analysis.engine_health, analysis.weaknesses,
    )

    # Decision logic
    rec, reason, weight_overrides = _recommend_action(
        analysis, scoring_retries, diagnostic_retries,
    )
    analysis.recommendation = rec
    analysis.recommendation_reason = reason
    analysis.scoring_weight_overrides = weight_overrides

    return analysis


# ---------------------------------------------------------------------------
# Goal progress
# ---------------------------------------------------------------------------

def _compute_goal_progress(metrics: DownturnMetrics) -> dict[str, dict]:
    progress = {}
    for key, target in ULTIMATE_TARGETS.items():
        actual = getattr(metrics, key, 0)
        # max_dd_pct: lower is better
        if key == "max_dd_pct":
            pct = (target / actual * 100) if actual > 0 else 100.0
        else:
            pct = (actual / target * 100) if target > 0 else 0.0
        progress[key] = {"target": target, "actual": actual, "pct_of_target": min(pct, 200.0)}
    return progress


# ---------------------------------------------------------------------------
# Strengths / weaknesses
# ---------------------------------------------------------------------------

def _assess_strengths_weaknesses(
    metrics: DownturnMetrics,
) -> tuple[list[str], list[str]]:
    strengths = []
    weaknesses = []

    if metrics.profit_factor >= 1.5:
        strengths.append(f"Strong PF ({metrics.profit_factor:.2f})")
    elif metrics.profit_factor < 1.0:
        weaknesses.append(f"PF below breakeven ({metrics.profit_factor:.2f})")

    if metrics.correction_pnl_pct >= 10:
        strengths.append(f"Good correction PnL ({metrics.correction_pnl_pct:.1f}%)")
    elif metrics.correction_pnl_pct < 2:
        weaknesses.append(f"Low correction PnL ({metrics.correction_pnl_pct:.1f}%)")

    if metrics.max_dd_pct <= 0.15:
        strengths.append(f"Controlled drawdown ({metrics.max_dd_pct:.1%})")
    elif metrics.max_dd_pct > 0.25:
        weaknesses.append(f"High drawdown ({metrics.max_dd_pct:.1%})")

    if metrics.exit_efficiency >= 0.35:
        strengths.append(f"Good exit efficiency ({metrics.exit_efficiency:.2f})")
    elif metrics.exit_efficiency < 0.15:
        weaknesses.append(f"Poor exit efficiency ({metrics.exit_efficiency:.2f})")

    if metrics.signal_to_entry_ratio >= 0.25:
        strengths.append(f"Good signal discrimination ({metrics.signal_to_entry_ratio:.2f})")
    elif metrics.signal_to_entry_ratio < 0.10:
        weaknesses.append(f"Low signal→entry conversion ({metrics.signal_to_entry_ratio:.2f})")

    # Trade frequency
    if metrics.total_trades >= 100:
        strengths.append(f"High trade frequency ({metrics.total_trades} trades)")
    elif metrics.total_trades < 40:
        weaknesses.append(f"Low trade frequency ({metrics.total_trades} trades)")

    # TP hit rates
    tp_rates = metrics.tp_hit_rates or {}
    if tp_rates.get("tp1", 0) < 0.20:
        weaknesses.append(f"Low TP1 hit rate ({tp_rates.get('tp1', 0):.1%})")
    if tp_rates.get("tp2", 0) == 0 and tp_rates.get("tp3", 0) == 0:
        weaknesses.append("TP2/TP3 never hit — exits give back MFE")

    # Win rate
    if metrics.win_rate >= 0.55:
        strengths.append(f"Strong win rate ({metrics.win_rate:.1%})")
    elif metrics.win_rate < 0.35:
        weaknesses.append(f"Low win rate ({metrics.win_rate:.1%})")

    # Engine-specific: reversal dead
    if metrics.reversal_trades == 0:
        weaknesses.append("Reversal engine dead (0 trades)")

    return strengths, weaknesses


# ---------------------------------------------------------------------------
# Improvement A: Engine health
# ---------------------------------------------------------------------------

def _assess_engine_health(metrics: DownturnMetrics) -> dict[str, str]:
    health = {}
    for tag, trades, wr, avg_r in [
        ("reversal", metrics.reversal_trades, metrics.reversal_wr, metrics.reversal_avg_r),
        ("breakdown", metrics.breakdown_trades, metrics.breakdown_wr, metrics.breakdown_avg_r),
        ("fade", metrics.fade_trades, metrics.fade_wr, metrics.fade_avg_r),
    ]:
        if trades < 3:
            health[tag] = "insufficient_data"
        elif avg_r < -0.5:
            health[tag] = "harmful"
        elif wr < 0.35 and avg_r < 0:
            health[tag] = "underperforming"
        else:
            health[tag] = "healthy"
    return health


# ---------------------------------------------------------------------------
# Improvement B: Correction attribution
# ---------------------------------------------------------------------------

def _compute_correction_attribution(
    metrics: DownturnMetrics,
    all_trades: list | None,
) -> dict[str, float]:
    if not all_trades:
        return {"correction_pnl": 0, "non_correction_pnl": 0, "ratio": 0}

    corr_pnl = sum(t.pnl for t in all_trades if t.in_correction_window)
    non_corr_pnl = sum(t.pnl for t in all_trades if not t.in_correction_window)
    total = corr_pnl + non_corr_pnl
    ratio = corr_pnl / total if total != 0 else 0

    return {
        "correction_pnl": corr_pnl,
        "non_correction_pnl": non_corr_pnl,
        "ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Improvement C: Engine-routed experiment suggestions
# ---------------------------------------------------------------------------

def _suggest_experiments(
    phase: int,
    metrics: DownturnMetrics,
    engine_health: dict[str, str],
    weaknesses: list[str],
) -> list[tuple[str, dict]]:
    suggestions = []

    # --- Engine health-routed suggestions ---
    for engine, status in engine_health.items():
        if status == "harmful":
            suggestions.append((f"ablate_{engine}", {f"flags.{engine}_engine": False}))
        elif status == "insufficient_data":
            # Engine dead — relax gates to activate it
            if engine == "reversal":
                suggestions.extend([
                    ("rev_relax_div_threshold_0.08",
                     {"param_overrides.divergence_mag_threshold": 0.08}),
                    ("rev_relax_trend_gate",
                     {"flags.reversal_trend_weakness_gate": False}),
                    ("rev_relax_extension_gate",
                     {"flags.reversal_extension_gate": False}),
                    ("rev_relax_corridor_cap",
                     {"flags.reversal_corridor_cap": False}),
                ])
            elif engine == "breakdown":
                suggestions.extend([
                    ("bd_relax_containment_0.60",
                     {"param_overrides.box_containment_min": 0.60}),
                    ("bd_no_chop_filter",
                     {"flags.breakdown_chop_filter": False}),
                    ("bd_relax_displacement_0.50",
                     {"param_overrides.displacement_quantile": 0.50}),
                ])
            elif engine == "fade":
                suggestions.extend([
                    ("fade_no_bear_required",
                     {"flags.fade_bear_regime_required": False}),
                    ("fade_no_momentum_confirm",
                     {"flags.fade_momentum_confirm": False}),
                ])
        elif status == "underperforming":
            if engine == "reversal":
                suggestions.extend([
                    ("rev_relax_div_threshold_0.10",
                     {"param_overrides.divergence_mag_threshold": 0.10}),
                    ("rev_relax_trend_gate",
                     {"flags.reversal_trend_weakness_gate": False}),
                ])
            elif engine == "breakdown":
                suggestions.extend([
                    ("bd_relax_containment_0.70",
                     {"param_overrides.box_containment_min": 0.70}),
                    ("bd_relax_displacement_0.55",
                     {"param_overrides.displacement_quantile": 0.55}),
                ])
            elif engine == "fade":
                suggestions.extend([
                    ("fade_widen_cap_0.40",
                     {"param_overrides.vwap_cap_core": 0.40}),
                    ("fade_no_bear_required",
                     {"flags.fade_bear_regime_required": False}),
                ])

    # --- Weakness-driven suggestions ---
    weakness_text = " ".join(weaknesses).lower()

    # Low trade frequency — relax gates to allow more entries
    if "low trade frequency" in weakness_text or metrics.total_trades < 50:
        suggestions.extend([
            ("relax_dead_zones", {"flags.use_dead_zones": False}),
            ("relax_entry_windows", {"flags.use_entry_windows": False}),
            ("relax_friction_gate", {"flags.friction_gate": False}),
            ("wider_regime_neutral",
             {"param_overrides.regime_mult_neutral": 0.80}),
        ])

    # Low correction-window PnL — regime detection too slow or misaligned
    if metrics.correction_pnl_pct < 5.0:
        suggestions.extend([
            ("regime_faster_ema_10",
             {"param_overrides.ema_fast_period": 10}),
            ("regime_adx_trending_20",
             {"param_overrides.adx_trending_threshold": 20}),
            ("regime_sma200_150",
             {"param_overrides.sma200_period": 150}),
        ])

    # TP2/TP3 never hit — exits too tight or TPs too ambitious
    if "tp2/tp3 never hit" in weakness_text:
        suggestions.extend([
            ("tp2_lower_2.0R",
             {"param_overrides.tp2_r_aligned": 2.0}),
            ("tp3_lower_3.5R",
             {"param_overrides.tp3_r_aligned": 3.5}),
            ("tp1_higher_2.0R",
             {"param_overrides.tp1_r_aligned": 2.0}),
            ("chandelier_wider_20",
             {"param_overrides.chandelier_lookback": 20}),
        ])

    # Low TP1 hit rate — stops too tight
    if "low tp1 hit rate" in weakness_text:
        suggestions.extend([
            ("wider_stop_mult",
             {"param_overrides.climax_mult": 3.0}),
            ("longer_stale_fade_36",
             {"param_overrides.stale_bars_fade": 36}),
        ])

    # Poor exit efficiency — trades give back MFE
    if metrics.exit_efficiency < 0.15:
        suggestions.extend([
            ("faster_be_move",
             {"param_overrides.tp1_r_aligned": 1.0}),
            ("tighter_chandelier_8",
             {"param_overrides.chandelier_lookback": 8}),
        ])

    # High drawdown — sizing or risk control
    if metrics.max_dd_pct > 0.25:
        suggestions.extend([
            ("lower_risk_pct_0.008",
             {"param_overrides.base_risk_pct": 0.008}),
            ("reduce_counter_mult_0.20",
             {"param_overrides.regime_mult_counter": 0.20}),
            ("circuit_breaker_tighter",
             {"param_overrides.daily_circuit_breaker": 0.02}),
        ])

    # Deduplicate by name
    seen = set()
    unique = []
    for name, muts in suggestions:
        if name not in seen:
            seen.add(name)
            unique.append((name, muts))
    return unique


# ---------------------------------------------------------------------------
# Scoring assessment
# ---------------------------------------------------------------------------

def _assess_scoring(
    metrics: DownturnMetrics,
    gate_result: GateResult,
    greedy_result: dict | None,
) -> str:
    """Assess if scoring function is effective."""
    if gate_result.passed:
        return "EFFECTIVE"

    if greedy_result:
        rounds = greedy_result.get("rounds", [])
        if rounds:
            improvements = sum(1 for r in rounds if r.get("accepted", False))
            if improvements == 0:
                return "INEFFECTIVE"
            # Score improved but gates didn't
            if improvements > 3 and not gate_result.passed:
                return "MISALIGNED"

    category = gate_result.failure_category
    if category == "scoring_ineffective":
        return "INEFFECTIVE"
    if category == "diagnostic_needed":
        return "MARGINAL"

    return "MARGINAL"


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _recommend_action(
    analysis: PhaseAnalysis,
    scoring_retries: int,
    diagnostic_retries: int,
) -> tuple[str, str, dict[str, float] | None]:
    """3-decision loop: improve_scoring / improve_diagnostics / advance."""

    # Check correction attribution for structural issues
    corr = analysis.correction_attribution
    if corr.get("correction_pnl", 0) < 0 and corr.get("non_correction_pnl", 0) > 0:
        if scoring_retries < 2:
            return (
                "improve_scoring",
                "Strategy profits from general trends but loses during corrections — "
                "boost correction_pnl weight",
                {"correction_pnl": 0.40, "signal_quality": 0.20,
                 "net_profit": 0.15, "profit_factor": 0.10,
                 "calmar": 0.05, "inv_drawdown": 0.10},
            )

    # 1. Scoring assessment
    if analysis.scoring_assessment in ("INEFFECTIVE", "MISALIGNED") and scoring_retries < 2:
        weight_overrides = _compute_weight_adjustment(analysis)
        return (
            "improve_scoring",
            f"Scoring {analysis.scoring_assessment} — adjusting weights",
            weight_overrides,
        )

    # 2. Diagnostic gaps
    if analysis.diagnostic_gaps and diagnostic_retries < 1:
        return (
            "improve_diagnostics",
            f"Diagnostic gaps: {', '.join(analysis.diagnostic_gaps[:3])}",
            None,
        )

    # 3. Advance
    reason = "clean" if not analysis.weaknesses else "forced (budget exhausted)"
    return ("advance", f"Advance to next phase — {reason}", None)


def _compute_weight_adjustment(analysis: PhaseAnalysis) -> dict[str, float]:
    """Compute adjusted weights based on weaknesses."""
    # Start from current phase weights
    from backtests.momentum.auto.downturn.phase_scoring import PHASE_WEIGHTS
    from backtests.momentum.auto.downturn.scoring import BASE_WEIGHTS
    base = PHASE_WEIGHTS.get(analysis.phase) or dict(BASE_WEIGHTS)
    adjusted = dict(base)

    # Boost underperforming components
    for weakness in analysis.weaknesses:
        if "correction pnl" in weakness.lower():
            adjusted["correction_pnl"] = min(adjusted.get("correction_pnl", 0.25) + 0.10, 0.45)
        if "pf" in weakness.lower() or "profit" in weakness.lower():
            adjusted["profit_factor"] = min(adjusted.get("profit_factor", 0.15) + 0.05, 0.30)
        if "drawdown" in weakness.lower():
            adjusted["inv_drawdown"] = min(adjusted.get("inv_drawdown", 0.10) + 0.05, 0.25)

    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted
