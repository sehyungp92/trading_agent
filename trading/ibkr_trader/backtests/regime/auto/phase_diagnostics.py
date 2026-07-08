"""Inter-phase diagnostics for multi-phase regime optimization.

Generates 5 phase-specific diagnostic sections:
1. Phase Delta — before/after comparison with previous phase
2. Mutation Analysis — accepted candidates by category
3. Gate Assessment — each criterion with target vs actual
4. Crisis Audit — regime assignments during known crises
5. Recommendations — what to try next based on failure category
"""
from __future__ import annotations

from io import StringIO
from typing import Any

from backtests.regime.analysis.metrics import PortfolioMetrics
from backtests.regime.auto.phase_gates import check_phase_gate
from backtests.regime.auto.phase_scoring import CRISIS_PERIODS, REGIME_COLS


def generate_phase_diagnostics(
    phase: int,
    regime_stats: dict,
    metrics: PortfolioMetrics,
    greedy_result: Any,
    state: Any,
) -> str:
    """Generate phase-specific diagnostics report.

    Args:
        phase: Current phase number.
        regime_stats: From compute_regime_stats().
        metrics: Portfolio metrics from final run.
        greedy_result: GreedyResult from run_greedy().
        state: PhaseState with history.

    Returns:
        Multi-section text report.
    """
    buf = StringIO()
    _sep = "=" * 70

    buf.write(f"\n{_sep}\n")
    buf.write(f" PHASE {phase} DIAGNOSTICS\n")
    buf.write(f"{_sep}\n")

    # Section 1: Phase Delta
    _write_phase_delta(buf, phase, metrics, regime_stats, state)

    # Section 2: Mutation Analysis
    _write_mutation_analysis(buf, greedy_result)

    # Section 3: Gate Assessment
    _write_gate_assessment(buf, phase, metrics, regime_stats, greedy_result)

    # Section 4: Crisis Audit
    _write_crisis_audit(buf, regime_stats)

    # Section 5: Recommendations
    _write_recommendations(buf, phase, metrics, regime_stats, greedy_result)

    return buf.getvalue()


def _write_phase_delta(
    buf: StringIO,
    phase: int,
    metrics: PortfolioMetrics,
    regime_stats: dict,
    state: Any,
) -> None:
    buf.write(f"\n--- 1. Phase Delta ---\n")

    # Compare with previous phase results (if available)
    prev_phase = phase - 1
    prev_result = state.phase_results.get(prev_phase, {}) if prev_phase > 0 else {}

    buf.write(f"  Current phase {phase} results:\n")
    buf.write(f"    Sharpe:           {metrics.sharpe:.4f}\n")
    buf.write(f"    CAGR:             {metrics.cagr:.2%}\n")
    buf.write(f"    Max DD:           {metrics.max_drawdown_pct:.2%}\n")
    buf.write(f"    Calmar:           {metrics.calmar:.4f}\n")
    buf.write(f"    Active regimes:   {regime_stats['n_active_regimes']}\n")
    buf.write(f"    Regime entropy:   {regime_stats['regime_entropy']:.4f}\n")
    buf.write(f"    Transition rate:  {regime_stats['transition_rate']:.4f}\n")
    buf.write(f"    Crisis response:  {regime_stats['crisis_response']:.4f}\n")

    if prev_result:
        buf.write(f"\n  Previous phase {prev_phase} score: {prev_result.get('final_score', 'N/A')}\n")
    else:
        buf.write(f"\n  No previous phase for comparison (this is phase {phase})\n")

    buf.write(f"\n  Regime distribution:\n")
    for regime, frac in regime_stats.get("dominant_dist", {}).items():
        bar = "#" * int(frac * 50)
        buf.write(f"    {regime}: {frac:6.1%}  {bar}\n")


def _write_mutation_analysis(buf: StringIO, greedy_result: Any) -> None:
    buf.write(f"\n--- 2. Mutation Analysis ---\n")

    rounds = greedy_result.rounds if hasattr(greedy_result, "rounds") else []
    if not rounds:
        buf.write("  No mutations accepted.\n")
        return

    buf.write(f"  Accepted {len(rounds)} mutations:\n")
    for r in rounds:
        buf.write(f"    Round {r.round_num}: {r.candidate_id}  "
                  f"delta=+{r.delta:.4f}  "
                  f"({r.score_before:.4f} -> {r.score_after:.4f})\n")

    # Categorize mutations
    categories = {}
    for r in rounds:
        muts = r.candidate_mutations if hasattr(r, "candidate_mutations") else {}
        for key in muts:
            cat = _categorize_param(key)
            categories.setdefault(cat, []).append(r.candidate_id)

    buf.write(f"\n  By category:\n")
    for cat, names in sorted(categories.items()):
        buf.write(f"    {cat}: {', '.join(names)}\n")


def _write_gate_assessment(
    buf: StringIO,
    phase: int,
    metrics: PortfolioMetrics,
    regime_stats: dict,
    greedy_result: Any,
) -> None:
    buf.write(f"\n--- 3. Gate Assessment ---\n")

    rounds = greedy_result.rounds if hasattr(greedy_result, "rounds") else []
    greedy_data = {
        "n_rounds": len(rounds),
        "max_rounds": getattr(greedy_result, "max_rounds", 20),
        "rounds": [{"candidate_id": r.candidate_id} for r in rounds],
    }
    gate = check_phase_gate(phase, metrics, regime_stats, greedy_data)

    status = "PASSED" if gate.passed else "FAILED"
    buf.write(f"  Overall: {status}\n")
    for c in gate.criteria:
        mark = "PASS" if c.passed else "FAIL"
        buf.write(f"    [{mark}] {c.name}: {c.actual:.4f} (target: {c.target:.4f})\n")

    if gate.failure_category:
        buf.write(f"\n  Failure category: {gate.failure_category}\n")


def _write_crisis_audit(buf: StringIO, regime_stats: dict) -> None:
    buf.write(f"\n--- 4. Crisis Audit ---\n")

    crisis_response = regime_stats.get("crisis_response", 0.0)
    buf.write(f"  Overall crisis response: {crisis_response:.4f}\n")

    # We don't have per-crisis breakdown in regime_stats by default,
    # but we show what's expected
    buf.write(f"\n  Expected crisis assignments:\n")
    for name, (start, end, expected) in CRISIS_PERIODS.items():
        buf.write(f"    {name} ({start} to {end}): expected {expected}\n")


def _write_recommendations(
    buf: StringIO,
    phase: int,
    metrics: PortfolioMetrics,
    regime_stats: dict,
    greedy_result: Any,
) -> None:
    buf.write(f"\n--- 5. Recommendations ---\n")

    recs = []

    # Phase-specific heuristics from the plan
    if regime_stats["regime_entropy"] > 0.5 and metrics.sharpe < 0.4:
        recs.append("Entropy improved but financials degraded — reduce regime health weight by 10%")
    if regime_stats["n_active_regimes"] < 3 and phase == 1:
        recs.append("Still < 3 active regimes — sticky prior may need to go even lower (try 2-3)")
    if regime_stats["crisis_response"] < 0.2 and phase >= 3:
        recs.append("Low crisis response despite good diversity — investigate ventilator disconnect")
    if regime_stats.get("historical_alignment", 0.0) < 0.3 and phase >= 4:
        recs.append("Low historical alignment — features likely wrong, go back to Phase 2")

    n_rounds = len(greedy_result.rounds) if hasattr(greedy_result, "rounds") else 0
    if n_rounds == 0:
        recs.append("No mutations accepted — all candidates may be rejected; check hard reject thresholds")
    elif n_rounds == 1:
        recs.append("Only 1 mutation accepted — candidates may be too narrow; consider wider ranges")

    if not recs:
        recs.append("Results look healthy. Proceed to next phase or gate check.")

    for r in recs:
        buf.write(f"  - {r}\n")


def _categorize_param(key: str) -> str:
    """Categorize a parameter key into a human-readable category."""
    categories = {
        "sticky": "HMM prior",
        "warm_start": "Warm start",
        "rolling_window": "Window",
        "use_expanding": "Window",
        "refit": "Refit",
        "ll_tol": "OOS guard",
        "z_window": "Z-score",
        "z_minp": "Z-score",
        "covariance_type": "HMM architecture",
        "commodity": "Features",
        "real_rates": "Features",
        "drop_": "Features",
        "cov_window": "Covariance",
        "crisis": "Crisis overlay",
        "conf_floor": "Confidence",
        "stability": "Confidence",
        "ventilator": "Ventilator",
        "delta_rho": "Ventilator",
        "L_max": "Leverage",
        "kappa": "Leverage",
        "base_target_vol": "Leverage",
        "sigma_floor": "Risk budget",
        "per_strat": "Risk budget",
    }
    for prefix, cat in categories.items():
        if prefix in key:
            return cat
    return "Other"
