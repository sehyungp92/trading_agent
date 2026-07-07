"""Post-phase analysis for regime optimization.

Evaluates results after each phase to assess progress toward the goal
of a high-value market regime predictor with maximum predictive capabilities.
Identifies strengths, weaknesses, scoring effectiveness, and recommends
next actions (proceed, rerun with adjusted scoring, or expand experiments).

All analysis is logged to the phase log file for full auditability.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ultimate targets for a production-ready regime predictor
_ULTIMATE_TARGETS = {
    "sharpe": 1.0,
    "calmar": 1.0,
    "max_drawdown_pct": 0.15,
    "cagr": 0.08,
    "sortino": 1.5,
    "n_active_regimes": 4,
    "regime_entropy": 0.85,
    "crisis_response": 0.50,
    "transition_rate_min": 0.005,
    "transition_rate_max": 0.025,
}

# Phase-specific focus areas (what each phase should improve)
_PHASE_FOCUS = {
    1: ("HMM dynamics", ["n_active_regimes", "regime_entropy", "transition_rate"]),
    2: ("Feature engineering", ["regime_entropy", "sharpe", "crisis_response"]),
    3: ("Crisis & portfolio", ["crisis_response", "max_drawdown_pct", "calmar"]),
    4: ("Fine-tuning", ["sharpe", "calmar", "sortino"]),
}


@dataclass
class PhaseAnalysis:
    """Structured result of post-phase analysis."""
    phase: int
    goal_progress: dict[str, dict]
    strengths: list[str]
    weaknesses: list[str]
    scoring_assessment: str
    suggested_experiments: list[tuple[str, dict[str, Any]]]
    recommendation: str  # "proceed", "rerun", "expand_and_proceed"
    recommendation_reason: str
    report: str


def analyze_phase(
    phase: int,
    greedy_result,
    regime_stats: dict,
    metrics,
    state,
    gate_passed: bool = True,
) -> PhaseAnalysis:
    """Run comprehensive post-phase analysis.

    Args:
        phase: Phase number (1-4).
        greedy_result: GreedyResult from optimization.
        regime_stats: Dict from compute_regime_stats().
        metrics: PortfolioMetrics dataclass.
        state: PhaseState with cumulative history.
        gate_passed: Whether the phase gate check passed.

    Returns:
        PhaseAnalysis with full structured results and text report.
    """
    goal_progress = _assess_goal_progress(metrics, regime_stats)
    strengths = _identify_strengths(metrics, regime_stats, state, phase)
    weaknesses = _identify_weaknesses(metrics, regime_stats, state, phase)
    scoring_assessment = _assess_scoring(greedy_result, metrics, regime_stats, phase)
    suggested = _suggest_experiments(weaknesses, regime_stats, metrics, phase)
    recommendation, reason = _recommend_action(
        phase, gate_passed, greedy_result, metrics, regime_stats,
        weaknesses, scoring_assessment,
    )

    report = _generate_report(
        phase, goal_progress, strengths, weaknesses,
        scoring_assessment, suggested, recommendation, reason,
        greedy_result, metrics, regime_stats,
    )

    analysis = PhaseAnalysis(
        phase=phase,
        goal_progress=goal_progress,
        strengths=strengths,
        weaknesses=weaknesses,
        scoring_assessment=scoring_assessment,
        suggested_experiments=suggested,
        recommendation=recommendation,
        recommendation_reason=reason,
        report=report,
    )

    # Log full analysis
    logger.info("=== Post-Phase %d Analysis ===", phase)
    for line in report.split("\n"):
        if line.strip():
            logger.info(line)

    return analysis


def save_phase_analysis(analysis: PhaseAnalysis, path: Path) -> None:
    """Persist analysis to JSON for later reference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": analysis.phase,
        "goal_progress": analysis.goal_progress,
        "strengths": analysis.strengths,
        "weaknesses": analysis.weaknesses,
        "scoring_assessment": analysis.scoring_assessment,
        "suggested_experiments": [
            {"name": n, "mutations": m} for n, m in analysis.suggested_experiments
        ],
        "recommendation": analysis.recommendation,
        "recommendation_reason": analysis.recommendation_reason,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Phase analysis saved to %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assess_goal_progress(metrics, regime_stats: dict) -> dict:
    """Evaluate each metric against ultimate production targets."""
    progress = {}

    checks = [
        ("sharpe", _ULTIMATE_TARGETS["sharpe"], metrics.sharpe, True),
        ("calmar", _ULTIMATE_TARGETS["calmar"], metrics.calmar, True),
        ("cagr", _ULTIMATE_TARGETS["cagr"], metrics.cagr, True),
        ("sortino", _ULTIMATE_TARGETS["sortino"], metrics.sortino, True),
        ("max_drawdown", _ULTIMATE_TARGETS["max_drawdown_pct"],
         metrics.max_drawdown_pct, False),
    ]
    for name, target, current, higher_is_better in checks:
        if higher_is_better:
            pct = max(0.0, min(100.0, current / target * 100)) if target > 0 else 0.0
            status = "ACHIEVED" if current >= target else "IN_PROGRESS"
        else:
            pct = max(0.0, min(100.0, target / current * 100)) if current > 0 else 100.0
            status = "ACHIEVED" if current <= target else "IN_PROGRESS"

        progress[name] = {
            "current": round(current, 4),
            "target": round(target, 4),
            "pct_progress": round(pct, 1),
            "status": status,
        }

    # Regime quality metrics
    n_regimes = regime_stats.get("n_active_regimes", 0)
    progress["n_active_regimes"] = {
        "current": n_regimes,
        "target": _ULTIMATE_TARGETS["n_active_regimes"],
        "pct_progress": round(min(100.0, n_regimes / 4 * 100), 1),
        "status": "ACHIEVED" if n_regimes >= 4 else "IN_PROGRESS",
    }

    entropy = regime_stats.get("regime_entropy", 0)
    progress["regime_entropy"] = {
        "current": round(entropy, 4),
        "target": _ULTIMATE_TARGETS["regime_entropy"],
        "pct_progress": round(min(100.0, entropy / 0.85 * 100), 1),
        "status": "ACHIEVED" if entropy >= 0.85 else "IN_PROGRESS",
    }

    crisis = regime_stats.get("crisis_response", 0)
    progress["crisis_response"] = {
        "current": round(crisis, 4),
        "target": _ULTIMATE_TARGETS["crisis_response"],
        "pct_progress": round(min(100.0, crisis / 0.50 * 100), 1),
        "status": "ACHIEVED" if crisis >= 0.50 else "IN_PROGRESS",
    }

    tr = regime_stats.get("transition_rate", 0)
    tr_min = _ULTIMATE_TARGETS["transition_rate_min"]
    tr_max = _ULTIMATE_TARGETS["transition_rate_max"]
    if tr_min <= tr <= tr_max:
        tr_pct, tr_status = 100.0, "ACHIEVED"
    elif tr < tr_min:
        tr_pct = round(min(100.0, tr / tr_min * 100), 1)
        tr_status = "IN_PROGRESS"
    else:
        tr_pct = round(min(100.0, tr_max / tr * 100), 1)
        tr_status = "IN_PROGRESS"
    progress["transition_rate"] = {
        "current": round(tr, 5),
        "target": f"{tr_min}-{tr_max}",
        "pct_progress": tr_pct,
        "status": tr_status,
    }

    return progress


def _identify_strengths(metrics, regime_stats, state, phase) -> list[str]:
    """Identify what is working well after this phase."""
    strengths = []

    # Score improvement from previous phase
    prev_score = 0.0
    for p in range(1, phase):
        if p in state.phase_results:
            prev_score = state.phase_results[p].get("final_score", prev_score)

    curr_score = state.phase_results.get(phase, {}).get("final_score", 0)
    if curr_score > prev_score and prev_score > 0:
        strengths.append(
            f"Score improved {prev_score:.4f} -> {curr_score:.4f} "
            f"(+{curr_score - prev_score:.4f})"
        )

    # Regime health
    n_regimes = regime_stats.get("n_active_regimes", 0)
    if n_regimes >= 4:
        strengths.append("All 4 regimes active (G/R/S/D)")
    elif n_regimes >= 3:
        strengths.append(f"{n_regimes} regimes active (target: 4)")

    entropy = regime_stats.get("regime_entropy", 0)
    if entropy > 0.85:
        strengths.append(f"Excellent regime entropy ({entropy:.3f})")
    elif entropy > 0.70:
        strengths.append(f"Good regime entropy ({entropy:.3f})")

    # Distribution balance
    dist = regime_stats.get("dominant_dist", {})
    min_share = min(dist.values()) if dist else 0
    if min_share > 0.10:
        strengths.append(f"Well-balanced regime distribution (min share {min_share:.1%})")

    # Financial metrics
    if metrics.sharpe > 0.8:
        strengths.append(f"Strong Sharpe ratio ({metrics.sharpe:.3f})")
    elif metrics.sharpe > 0.5:
        strengths.append(f"Decent Sharpe ratio ({metrics.sharpe:.3f})")

    if metrics.calmar > 0.5:
        strengths.append(f"Good risk-adjusted returns (Calmar={metrics.calmar:.3f})")

    if metrics.max_drawdown_pct < 0.15:
        strengths.append(f"Well-controlled drawdowns ({metrics.max_drawdown_pct:.1%})")

    if metrics.cagr > 0.08:
        strengths.append(f"Strong CAGR ({metrics.cagr:.2%})")

    crisis = regime_stats.get("crisis_response", 0)
    if crisis > 0.5:
        strengths.append(f"Strong crisis detection ({crisis:.3f})")
    elif crisis > 0.3:
        strengths.append(f"Moderate crisis detection ({crisis:.3f})")

    return strengths


def _identify_weaknesses(metrics, regime_stats, state, phase) -> list[str]:
    """Identify what is limiting predictive capability."""
    weaknesses = []

    # Regime structural issues
    n_regimes = regime_stats.get("n_active_regimes", 0)
    if n_regimes < 3:
        weaknesses.append(
            f"CRITICAL: Only {n_regimes} active regimes - HMM collapsing"
        )
    elif n_regimes < 4:
        dist = regime_stats.get("dominant_dist", {})
        missing = [r for r in ["G", "R", "S", "D"] if dist.get(r, 0) < 0.03]
        if missing:
            weaknesses.append(
                f"Regime(s) {missing} underrepresented (<3%)"
            )

    entropy = regime_stats.get("regime_entropy", 0)
    if entropy < 0.60:
        weaknesses.append(
            f"Low regime entropy ({entropy:.3f}) - heavy concentration"
        )
    elif entropy < 0.85:
        weaknesses.append(
            f"Moderate regime entropy ({entropy:.3f}) - room for improvement"
        )

    tr = regime_stats.get("transition_rate", 0)
    if tr < 0.005:
        weaknesses.append(
            f"Very low transition rate ({tr:.4f}) - regimes too sticky"
        )
    elif tr > 0.025:
        weaknesses.append(
            f"High transition rate ({tr:.4f}) - noisy regime switching"
        )

    # Financial weaknesses
    if metrics.sharpe < 0.3:
        weaknesses.append(
            f"Very low Sharpe ({metrics.sharpe:.3f}) - signal has little predictive value"
        )
    elif metrics.sharpe < 0.5:
        weaknesses.append(
            f"Low Sharpe ({metrics.sharpe:.3f}) - poor signal-to-noise"
        )
    elif metrics.sharpe < 0.8:
        weaknesses.append(
            f"Moderate Sharpe ({metrics.sharpe:.3f}) - needs improvement for production"
        )

    if metrics.max_drawdown_pct > 0.25:
        weaknesses.append(
            f"Excessive drawdown ({metrics.max_drawdown_pct:.1%}) - risk management needed"
        )
    elif metrics.max_drawdown_pct > 0.15:
        weaknesses.append(
            f"Elevated drawdown ({metrics.max_drawdown_pct:.1%}) - "
            "consider leverage/risk tuning"
        )

    if metrics.calmar < 0.3:
        weaknesses.append(
            f"Poor Calmar ({metrics.calmar:.3f}) - returns don't justify drawdown risk"
        )

    if metrics.cagr < 0.05:
        weaknesses.append(
            f"Low CAGR ({metrics.cagr:.2%}) - regime prediction not adding value"
        )

    crisis = regime_stats.get("crisis_response", 0)
    if crisis < 0.2:
        weaknesses.append(
            f"Very poor crisis detection ({crisis:.3f}) - "
            "model misses major regime shifts"
        )
    elif crisis < 0.4:
        weaknesses.append(
            f"Weak crisis detection ({crisis:.3f}) - slow adaptation to stress"
        )

    # Check for regressions from previous phase
    prev_fm = {}
    for p in range(1, phase):
        if p in state.phase_results:
            fm = state.phase_results[p].get("final_metrics", {})
            if fm:
                prev_fm = fm

    if prev_fm:
        for key, label in [
            ("sharpe", "Sharpe"),
            ("calmar", "Calmar"),
            ("cagr", "CAGR"),
            ("max_drawdown_pct", "Max DD"),
        ]:
            prev_val = prev_fm.get(key, 0)
            curr_val = getattr(metrics, key, 0)
            if key == "max_drawdown_pct":
                # For DD, regression = getting worse (higher)
                if prev_val > 0 and curr_val > prev_val * 1.15:
                    weaknesses.append(
                        f"REGRESSION: {label} worsened from "
                        f"{prev_val:.1%} to {curr_val:.1%}"
                    )
            else:
                if prev_val > 0 and curr_val < prev_val * 0.90:
                    weaknesses.append(
                        f"REGRESSION: {label} dropped from "
                        f"{prev_val:.3f} to {curr_val:.3f}"
                    )

    return weaknesses


def _assess_scoring(greedy_result, metrics, regime_stats, phase) -> str:
    """Assess whether the scoring function drove the right improvements."""
    score_delta = greedy_result.final_score - greedy_result.baseline_score
    n_accepted = len(greedy_result.rounds)

    if n_accepted == 0:
        return (
            "INEFFECTIVE: No candidates accepted. The scoring function may be "
            "too strict or candidates don't address the current bottleneck. "
            "Consider adjusting scorer weights or expanding candidate ranges."
        )

    if score_delta < 0.005:
        return (
            f"MARGINAL: Score improved by only {score_delta:.4f} across "
            f"{n_accepted} mutations. The scoring function may not be "
            "sensitive enough to meaningful improvements. Consider increasing "
            "weight on the weakest metrics."
        )

    # Check for score-metric divergence
    focus_name, focus_metrics = _PHASE_FOCUS.get(phase, ("", []))
    divergences = []

    if "sharpe" in focus_metrics and metrics.sharpe < 0.5 and phase >= 2:
        divergences.append(f"Sharpe still low ({metrics.sharpe:.3f})")
    if "crisis_response" in focus_metrics:
        crisis = regime_stats.get("crisis_response", 0)
        if crisis < 0.3:
            divergences.append(f"Crisis response still poor ({crisis:.3f})")
    if "max_drawdown_pct" in focus_metrics and metrics.max_drawdown_pct > 0.20:
        divergences.append(f"Drawdown still high ({metrics.max_drawdown_pct:.1%})")

    if divergences:
        return (
            f"MISALIGNED: Score improved (+{score_delta:.4f}) but key phase "
            f"focus metrics lagging: {'; '.join(divergences)}. "
            "Scoring may be over-weighting non-priority dimensions. "
            "Consider rebalancing phase scorer weights toward focus areas."
        )

    return (
        f"EFFECTIVE: Score improved by {score_delta:.4f} with {n_accepted} "
        f"accepted mutations. Financial metrics (Sharpe={metrics.sharpe:.3f}, "
        f"Calmar={metrics.calmar:.3f}) and regime health "
        f"(entropy={regime_stats.get('regime_entropy', 0):.3f}) "
        "are progressing together."
    )


def _suggest_experiments(weaknesses, regime_stats, metrics, phase) -> list[tuple[str, dict]]:
    """Generate experiment suggestions based on identified weaknesses.

    Suggestions are phase-aware: early phases (1-2) suggest HMM-level fixes
    (dynamics, features) while later phases (3-4) suggest portfolio-level tuning.
    """
    experiments: list[tuple[str, dict]] = []
    wk_text = " ".join(weaknesses).lower()

    # HMM collapse / few regimes (always HMM-level)
    if "collapsing" in wk_text or ("only" in wk_text and "regime" in wk_text):
        experiments.extend([
            ("exp_sticky_diag_3", {"sticky_diag": 3.0}),
            ("exp_sticky_diag_2", {"sticky_diag": 2.0}),
            ("exp_rolling_4y", {
                "use_expanding_window": False, "rolling_window_years": 4,
            }),
            ("exp_refit_3ME", {"refit_freq": "3ME"}),
        ])

    # Underrepresented regimes (HMM-level)
    if "underrepresented" in wk_text:
        experiments.extend([
            ("exp_lower_sticky_offdiag", {"sticky_offdiag": 2.0}),
            ("exp_cov_full", {"covariance_type": "full"}),
            ("exp_z_window_126", {"z_window": 126}),
        ])

    # Low/moderate entropy (HMM-level)
    if "entropy" in wk_text and ("low" in wk_text or "moderate" in wk_text):
        experiments.extend([
            ("exp_lower_sticky_combo", {"sticky_diag": 3.0, "sticky_offdiag": 2.0}),
            ("exp_cov_full", {"covariance_type": "full"}),
        ])

    # Transition rate issues — phase-aware
    if "too sticky" in wk_text:
        experiments.extend([
            ("exp_ll_tol_3", {"refit_ll_tolerance": 3.0}),
            ("exp_refit_4ME", {"refit_freq": "4ME"}),
            ("exp_perturb_0.3", {"warm_start_perturb_std": 0.3}),
        ])
    elif "noisy" in wk_text and "transition" in wk_text:
        if phase <= 2:
            # HMM-level stabilization
            experiments.extend([
                ("exp_sticky_25", {"sticky_diag": 25.0}),
                ("exp_sticky_15", {"sticky_diag": 15.0}),
                ("exp_z_window_504", {"z_window": 504}),
                ("exp_expanding_window", {"use_expanding_window": True}),
            ])
        else:
            # Portfolio-level — can't fix HMM, stabilize via confidence
            experiments.extend([
                ("exp_stability_weight_0.8", {"stability_weight": 0.8}),
                ("exp_conf_floor_0.5", {"conf_floor": 0.5}),
            ])

    # Poor crisis response — phase-aware suggestions
    # Crisis response = HMM regime label accuracy during known crises.
    # Phase 1: fix HMM dynamics so regime transitions happen during crises
    # Phase 2: fix features so HMM can distinguish crisis from non-crisis
    # Phase 3+: tune portfolio-level crisis overlay
    if "crisis" in wk_text and ("poor" in wk_text or "weak" in wk_text):
        if phase == 1:
            # HMM dynamics: faster adaptation to crisis regime shifts
            experiments.extend([
                ("exp_crisis_rolling_5y_cold", {
                    "use_expanding_window": False, "rolling_window_years": 5,
                    "use_warm_start": False,
                }),
                ("exp_crisis_refit_QE", {"refit_freq": "QE"}),
                ("exp_crisis_ll_tol_3", {"refit_ll_tolerance": 3.0}),
                ("exp_crisis_sticky_3", {"sticky_diag": 3.0}),
                ("exp_crisis_combo_fast", {
                    "sticky_diag": 3.0, "refit_freq": "QE",
                    "use_expanding_window": False, "rolling_window_years": 5,
                }),
            ])
        elif phase == 2:
            # Feature engineering: features that distinguish crisis periods
            experiments.extend([
                ("exp_crisis_add_commodity", {"use_commodity_feature": True}),
                ("exp_crisis_add_both_features", {
                    "use_commodity_feature": True,
                    "use_real_rates_feature": True,
                }),
                ("exp_crisis_drop_noisy_add_commodity", {
                    "use_commodity_feature": True,
                    "drop_momentum_breadth": True,
                }),
                ("exp_crisis_z_window_126", {"z_window": 126}),
                ("exp_crisis_cov_diag", {"covariance_type": "diag"}),
            ])
        else:
            # Phase 3+: portfolio-level crisis overlay tuning
            experiments.extend([
                ("exp_crisis_vix_80", {"crisis_weights": (0.8, 0.15, 0.05)}),
                ("exp_crisis_logit_steep", {
                    "crisis_logit_a": 3.0, "crisis_logit_b": -0.3,
                }),
                ("exp_crisis_spread_heavy", {"crisis_weights": (0.3, 0.6, 0.1)}),
            ])

    # Low Sharpe / signal quality — phase-aware
    if "sharpe" in wk_text and ("low" in wk_text or "moderate" in wk_text):
        if phase <= 2:
            # HMM-level: better regime separation improves signal
            experiments.extend([
                ("exp_cov_diag", {"covariance_type": "diag"}),
                ("exp_z_minp_90", {"z_minp": 90}),
            ])
        else:
            # Portfolio-level: leverage and confidence tuning
            experiments.extend([
                ("exp_target_vol_12", {"base_target_vol_annual": 0.12}),
                ("exp_L_max_1.5", {"L_max": 1.5}),
                ("exp_conf_floor_0.3", {"conf_floor": 0.3}),
            ])

    # Drawdown issues
    if "drawdown" in wk_text and ("excessive" in wk_text or "elevated" in wk_text):
        experiments.extend([
            ("exp_L_max_0.8", {"L_max": 0.8}),
            ("exp_target_vol_6", {"base_target_vol_annual": 0.06}),
            ("exp_kappa_0.8", {"kappa_totalvol_cap": 0.8}),
            ("exp_sigma_floor_0.05", {"sigma_floor_annual": 0.05}),
        ])

    # Low CAGR
    if "cagr" in wk_text and "low" in wk_text:
        experiments.extend([
            ("exp_L_max_1.5", {"L_max": 1.5}),
            ("exp_target_vol_12", {"base_target_vol_annual": 0.12}),
        ])

    # Poor Calmar
    if "calmar" in wk_text and "poor" in wk_text:
        experiments.extend([
            ("exp_vent_lambda_0.5", {"ventilator_lambda": 0.5}),
            ("exp_delta_rho_0.15", {"delta_rho_threshold": 0.15}),
        ])

    # Regressions suggest over-tuning in one dimension
    if "regression" in wk_text:
        experiments.extend([
            ("exp_stability_weight_0.6", {"stability_weight": 0.6}),
            ("exp_conf_floor_0.5", {"conf_floor": 0.5}),
        ])

    # Deduplicate
    seen: set[str] = set()
    deduped = []
    for name, muts in experiments:
        if name not in seen:
            seen.add(name)
            deduped.append((name, muts))

    return deduped


def _recommend_action(
    phase, gate_passed, greedy_result, metrics, regime_stats,
    weaknesses, scoring_assessment,
) -> tuple[str, str]:
    """Recommend next action based on full analysis."""
    score_delta = greedy_result.final_score - greedy_result.baseline_score

    # Scoring was ineffective -> rerun with different approach
    if "INEFFECTIVE" in scoring_assessment:
        return (
            "rerun",
            "Scoring function failed to identify improvements. Adjust phase scorer "
            "weights to better target current weaknesses, then rerun this phase.",
        )

    # Scoring was misaligned -> expand and proceed (same scoring = same
    # misalignment on rerun; next phase has different scoring focus)
    if "MISALIGNED" in scoring_assessment:
        return (
            "expand_and_proceed",
            "Scoring function improved score but not key focus metrics. "
            "Adding suggested experiments to next phase whose scoring "
            "better targets the lagging dimensions.",
        )

    # Gate failed with structural issues -> must fix before proceeding
    if not gate_passed:
        critical = [w for w in weaknesses if "CRITICAL" in w]
        if critical:
            return (
                "rerun",
                f"Gate failed with structural issue: {critical[0]}. "
                "Need fundamental parameter changes before proceeding.",
            )
        return (
            "expand_and_proceed",
            "Gate failed but no structural issues. Adding suggested experiments "
            "to next phase and proceeding (next phase may address remaining gaps).",
        )

    # Gate passed - check for regressions
    regressions = [w for w in weaknesses if "REGRESSION" in w]
    if regressions:
        return (
            "expand_and_proceed",
            f"Gate passed but with metric regressions: {regressions[0]}. "
            "Adding compensating experiments to next phase.",
        )

    # Normal progression
    n_rounds = len(greedy_result.rounds)
    return (
        "proceed",
        f"Phase {phase} gate passed. Score improved by {score_delta:.4f} "
        f"with {n_rounds} accepted mutation(s). Proceeding to next phase.",
    )


def _generate_report(
    phase, goal_progress, strengths, weaknesses,
    scoring_assessment, suggested, recommendation, reason,
    greedy_result, metrics, regime_stats,
) -> str:
    """Generate comprehensive analysis report text."""
    buf = StringIO()
    w = buf.write

    w(f"\n{'='*70}\n")
    w(f"  POST-PHASE {phase} ANALYSIS - Regime Predictor Optimization\n")
    w(f"{'='*70}\n\n")

    # Section 1: Goal progress
    w("1. GOAL PROGRESS (toward production-ready regime predictor)\n")
    w("-" * 60 + "\n")
    achieved = sum(1 for v in goal_progress.values() if v["status"] == "ACHIEVED")
    total = len(goal_progress)
    w(f"   Overall: {achieved}/{total} targets achieved\n\n")

    for name, info in goal_progress.items():
        marker = "[Y]" if info["status"] == "ACHIEVED" else "[ ]"
        pct = info["pct_progress"] if isinstance(info["pct_progress"], (int, float)) else 0
        bar_len = max(0, min(20, int(pct / 5)))
        bar = "#" * bar_len + "." * (20 - bar_len)
        current_str = str(info["current"])
        target_str = str(info["target"])
        w(f"   {marker} {name:<20s} {current_str:>10} / {target_str:<10} "
          f"[{bar}] {info['pct_progress']:>5.1f}%\n")

    # Section 2: Strengths
    w(f"\n2. STRENGTHS\n")
    w("-" * 60 + "\n")
    if strengths:
        for s in strengths:
            w(f"   + {s}\n")
    else:
        w("   (none identified)\n")

    # Section 3: Weaknesses
    w(f"\n3. WEAKNESSES & BOTTLENECKS\n")
    w("-" * 60 + "\n")
    if weaknesses:
        for wk in weaknesses:
            pfx = "!!!" if "CRITICAL" in wk or "REGRESSION" in wk else " - "
            w(f"   {pfx} {wk}\n")
    else:
        w("   (none identified)\n")

    # Section 4: Scoring assessment
    w(f"\n4. SCORING FUNCTION ASSESSMENT\n")
    w("-" * 60 + "\n")
    w(f"   {scoring_assessment}\n")

    # Section 5: Optimization summary
    w(f"\n5. OPTIMIZATION SUMMARY\n")
    w("-" * 60 + "\n")
    w(f"   Baseline score:    {greedy_result.baseline_score:.4f}\n")
    w(f"   Final score:       {greedy_result.final_score:.4f}\n")
    delta = greedy_result.final_score - greedy_result.baseline_score
    w(f"   Score delta:       {delta:+.4f}\n")
    w(f"   Rounds:            {len(greedy_result.rounds)}\n")
    w(f"   Candidates tested: {greedy_result.total_candidates_tested}\n")
    w(f"   Elapsed:           {greedy_result.elapsed_seconds:.0f}s\n")
    if greedy_result.rounds:
        w(f"   Accepted mutations:\n")
        for r in greedy_result.rounds:
            w(f"     Round {r.round_num}: {r.candidate_id} (+{r.delta:.4f})\n")

    # Section 6: Current cumulative config
    w(f"\n6. CUMULATIVE CONFIGURATION\n")
    w("-" * 60 + "\n")
    for k, v in sorted(greedy_result.accepted_mutations.items()):
        if k != "rebalance_freq":
            w(f"   {k}: {v}\n")

    # Section 7: Regime snapshot
    w(f"\n7. REGIME SNAPSHOT\n")
    w("-" * 60 + "\n")
    w(f"   Active regimes:    {regime_stats.get('n_active_regimes', '?')}\n")
    w(f"   Entropy:           {regime_stats.get('regime_entropy', 0):.4f}\n")
    w(f"   Transition rate:   {regime_stats.get('transition_rate', 0):.5f}\n")
    w(f"   Crisis response:   {regime_stats.get('crisis_response', 0):.4f}\n")
    dist = regime_stats.get("dominant_dist", {})
    if dist:
        parts = [f"{k}={v:.1%}" for k, v in sorted(dist.items())]
        w(f"   Distribution:      {', '.join(parts)}\n")

    # Section 8: Experiment suggestions
    w(f"\n8. SUGGESTED EXPERIMENTS FOR NEXT PHASE\n")
    w("-" * 60 + "\n")
    if suggested:
        for name, muts in suggested:
            w(f"   * {name}: {muts}\n")
    else:
        w("   (no additional experiments suggested)\n")

    # Section 9: Recommendation
    w(f"\n9. RECOMMENDATION\n")
    w("-" * 60 + "\n")
    next_label = f"Phase {phase + 1}" if phase < 4 else "validation"
    action_label = {
        "proceed": f">> PROCEED to {next_label}",
        "rerun": f">> RERUN Phase {phase} with adjusted scoring",
        "expand_and_proceed": f">> ADD experiments and PROCEED to {next_label}",
    }
    w(f"   Action: {action_label.get(recommendation, recommendation)}\n")
    w(f"   Reason: {reason}\n")

    w(f"\n{'='*70}\n")

    return buf.getvalue()
