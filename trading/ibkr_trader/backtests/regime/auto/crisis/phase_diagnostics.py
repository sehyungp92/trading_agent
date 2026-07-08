"""Crisis detection phase diagnostics -- latency, FP rates, alert distribution."""
from __future__ import annotations

from typing import Any

from .scoring import CrisisMetrics


def generate_phase_diagnostics(
    phase: int,
    metrics: CrisisMetrics,
    greedy_result: dict | None,
    alerts_df: Any = None,
    force_all_modules: bool = False,
) -> str:
    """Generate crisis-detection-specific phase diagnostics report."""
    lines: list[str] = []
    lines.append(f"{'='*60}")
    lines.append(f"Crisis Detection Phase {phase} Diagnostics")
    lines.append(f"{'='*60}")

    # D1: Detection Performance (always)
    lines.append("\n--- D1: Detection Performance ---")
    lines.append(f"Crises detected: {metrics.crises_detected}/{metrics.total_crises}")
    lines.append(f"Avg latency:     {metrics.avg_latency:.1f} days")
    lines.append(f"Max latency:     {metrics.max_latency:.1f} days")
    lines.append(f"Action latency:  {metrics.avg_action_latency:.1f} days avg")
    lines.append(f"Advisory latency:{metrics.avg_advisory_latency:.1f} days avg")
    if metrics.gfc_latency >= 0:
        lines.append(f"GFC latency:     {metrics.gfc_latency:.0f} days")
    else:
        lines.append("GFC latency:     NOT DETECTED")
    if metrics.covid_latency >= 0:
        lines.append(f"COVID latency:   {metrics.covid_latency:.0f} days")
    else:
        lines.append("COVID latency:   NOT DETECTED")

    # D2: False Positive Rates (always)
    lines.append("\n--- D2: False Positive Rates ---")
    lines.append(f"WARNING FP rate: {metrics.warning_fp_rate:.2%}")
    lines.append(f"CRISIS FP rate:  {metrics.crisis_fp_rate:.2%}")
    lines.append(f"Advisory FP rate:{metrics.advisory_fp_rate:.2%}")
    lines.append(f"Pre-action FP:   {metrics.preaction_fp_rate:.2%}")
    lines.append(f"Avg peak level:  {metrics.avg_peak_level:.2f}")

    # D3: Alert Level Distribution (always)
    lines.append("\n--- D3: Alert Level Distribution ---")
    lines.append(f"NORMAL:          {metrics.pct_normal:.1%}")
    lines.append(f"WATCH:           {metrics.pct_watch:.1%}")
    lines.append(f"WARNING:         {metrics.pct_warning:.1%}")
    lines.append(f"CRISIS:          {metrics.pct_crisis:.1%}")
    lines.append(f"Advisory WATCH:  {metrics.pct_advisory_watch:.1%}")
    lines.append(f"Pre-action WATCH:{metrics.pct_preaction_watch:.1%}")
    elevated = metrics.pct_warning + metrics.pct_crisis
    lines.append(f"Elevated total:  {elevated:.1%} (target ~7%)")

    # D4: Stability (always)
    lines.append("\n--- D4: Stability ---")
    lines.append(f"Total transitions:   {metrics.total_transitions}")
    lines.append(f"Transitions/year:    {metrics.transitions_per_year:.1f}")

    # D5: Greedy Result (always)
    if greedy_result:
        lines.append("\n--- D5: Greedy Result ---")
        lines.append(f"Base score:      {greedy_result.get('base_score', 0):.4f}")
        lines.append(f"Final score:     {greedy_result.get('final_score', 0):.4f}")
        lines.append(f"Accepted:        {greedy_result.get('accepted_count', 0)}")
        lines.append(f"Total candidates:{greedy_result.get('total_candidates', 0)}")
        kept = greedy_result.get("kept_features", [])
        if kept:
            lines.append(f"Kept features:   {', '.join(kept)}")

    # D6: Recovery Speed
    lines.append("\n--- D6: Recovery Speed ---")
    if hasattr(metrics, "avg_recovery_days"):
        qual = ("FAST" if metrics.avg_recovery_days <= 5 else
                "MODERATE" if metrics.avg_recovery_days <= 15 else "SLOW")
        lines.append(f"Avg post-crisis elevated days: {metrics.avg_recovery_days:.1f} [{qual}]")
        lines.append(f"Max post-crisis elevated days: {metrics.max_recovery_days:.1f}")
    else:
        lines.append("Recovery metrics not available")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


def get_diagnostic_gaps(phase: int, metrics: CrisisMetrics) -> list[str]:
    """Identify diagnostic gaps for the current phase."""
    gaps: list[str] = []

    if metrics.crises_detected < metrics.total_crises:
        gaps.append(
            f"Only {metrics.crises_detected}/{metrics.total_crises} crises detected -- coverage gap"
        )
    if metrics.avg_latency > 25.0:
        gaps.append(
            f"Average detection latency {metrics.avg_latency:.1f}d -- too slow"
        )
    if metrics.avg_action_latency > 14.0:
        gaps.append(
            f"Average portfolio action latency {metrics.avg_action_latency:.1f}d "
            "-- early action too slow"
        )
    if metrics.avg_advisory_latency > 10.0:
        gaps.append(
            f"Average external advisory latency {metrics.avg_advisory_latency:.1f}d "
            "-- advisory too slow"
        )
    if metrics.warning_fp_rate > 0.08:
        gaps.append(
            f"WARNING FP rate {metrics.warning_fp_rate:.2%} -- too many false alarms"
        )
    if metrics.crisis_fp_rate > 0.03:
        gaps.append(
            f"CRISIS FP rate {metrics.crisis_fp_rate:.2%} -- false crisis alerts"
        )
    if metrics.preaction_fp_rate > 0.12:
        gaps.append(
            f"Pre-action FP rate {metrics.preaction_fp_rate:.2%} -- early action too noisy"
        )
    if metrics.pct_preaction_watch < 0.02:
        gaps.append(
            f"Pre-action WATCH {metrics.pct_preaction_watch:.1%} -- early layer underused"
        )
    elif metrics.pct_preaction_watch > 0.10:
        gaps.append(
            f"Pre-action WATCH {metrics.pct_preaction_watch:.1%} -- early layer saturated"
        )
    if metrics.advisory_fp_rate > 0.50:
        gaps.append(
            f"Advisory FP rate {metrics.advisory_fp_rate:.2%} -- external WATCH too noisy"
        )
    if metrics.transitions_per_year > 25:
        gaps.append(
            f"Transitions/year {metrics.transitions_per_year:.0f} "
            "-- unstable state machine"
        )

    elevated = metrics.pct_warning + metrics.pct_crisis
    if elevated > 0.15:
        gaps.append(f"Elevated time {elevated:.1%} -- thresholds too sensitive")
    elif elevated < 0.03:
        gaps.append(f"Elevated time {elevated:.1%} -- thresholds too conservative")

    if hasattr(metrics, "avg_recovery_days") and metrics.avg_recovery_days > 15:
        gaps.append(f"recovery_gap: avg {metrics.avg_recovery_days:.1f}d post-crisis elevated")

    if metrics.gfc_latency < 0:
        gaps.append(
            "GFC (2008) not detected -- need lower thresholds or "
            "fewer conjunction requirements"
        )
    if metrics.covid_latency < 0:
        gaps.append(
            "COVID (2020) not detected -- need lower thresholds or "
            "fewer conjunction requirements"
        )

    return gaps


def suggest_experiments(
    phase: int,
    metrics: CrisisMetrics,
    weaknesses: list[str],
) -> list[tuple[str, dict]]:
    """Generate follow-up experiment suggestions based on weaknesses."""
    suggestions: list[tuple[str, dict]] = []
    weakness_text = " ".join(weaknesses).lower()

    if "coverage" in weakness_text or metrics.crises_detected < 7:
        suggestions.extend([
            ("suggest_lower_vix", {
                "VIX_WATCH": 20.0, "VIX_WARNING": 27.0, "VIX_CRISIS": 32.0,
            }),
            ("suggest_relax_conjunction", {
                "WARNING_MIN_PRIMARY": 1, "CRISIS_MIN_PRIMARY": 1,
            }),
        ])

    if "fp" in weakness_text or "false" in weakness_text:
        suggestions.extend([
            ("suggest_raise_vix", {
                "VIX_WATCH": 27.0, "VIX_WARNING": 32.0, "VIX_CRISIS": 38.0,
            }),
            ("suggest_tighten_conjunction", {
                "WARNING_MIN_PRIMARY": 2, "CRISIS_MIN_PRIMARY": 3,
            }),
        ])

    if "latency" in weakness_text or "slow" in weakness_text:
        suggestions.extend([
            ("suggest_aggressive_thresholds", {
                "VIX_WATCH": 20.0, "VIX_WARNING": 25.0, "VIX_CRISIS": 30.0,
            }),
            ("suggest_fast_hysteresis", {
                "DEESCALATE_CRISIS_DAYS": 2, "DEESCALATE_WARNING_DAYS": 3,
            }),
            ("suggest_tighter_spy_dd", {
                "SPY_DD_WATCH": -0.03, "SPY_DD_WARNING": -0.05,
                "SPY_DD_CRISIS": -0.08,
            }),
            ("suggest_hybrid_aggressive", {
                "HYBRID_WARNING_MIN_CRISIS": 1,
                "HYBRID_WARNING_MIN_PRIMARY": 1,
            }),
            ("suggest_fast_stress_formation", {
                "STRESS_FORMATION_MIN_SCORE": 2,
                "SHOCK_SPY_3D_RETURN": -0.03,
                "SHOCK_SPY_5D_RETURN": -0.05,
                "SHOCK_VIX_3D_CHANGE": 6.0,
                "SHOCK_MIN_VIX": 22.0,
                "GRIND_SPREAD_20D_CHANGE_BPS": 50.0,
                "GRIND_SPY_20D_RETURN": -0.04,
                "GRIND_VIX_PERSIST_DAYS": 3,
            }),
            ("suggest_early_shock_equity_vol", {
                "STRESS_FORMATION_MIN_SCORE": 2,
                "SHOCK_SPY_3D_RETURN": -0.025,
                "SHOCK_SPY_5D_RETURN": -0.045,
                "SHOCK_VIX_3D_CHANGE": 5.0,
                "SHOCK_MIN_VIX": 22.0,
                "SHOCK_CORR_MIN": 0.45,
                "SHOCK_CORR_SPY_5D_RETURN": -0.02,
            }),
        ])

    if (
        "pre-action" in weakness_text
        or "early action" in weakness_text
        or metrics.avg_action_latency > 14.0
    ):
        suggestions.extend([
            ("suggest_preaction_more_sensitive", {
                "STRESS_FORMATION_MIN_SCORE": 1,
                "SHOCK_SPY_3D_RETURN": -0.035,
                "SHOCK_SPY_5D_RETURN": -0.055,
            }),
            ("suggest_preaction_less_noisy", {
                "STRESS_FORMATION_MIN_SCORE": 3,
                "GRIND_SPREAD_20D_CHANGE_BPS": 100.0,
                "GRIND_VIX_PERSIST_DAYS": 7,
            }),
        ])

    if "unstable" in weakness_text or "transition" in weakness_text:
        suggestions.extend([
            ("suggest_slow_hysteresis", {
                "DEESCALATE_CRISIS_DAYS": 5, "DEESCALATE_WARNING_DAYS": 7,
                "DEESCALATE_WATCH_DAYS": 3,
            }),
        ])

    if "recovery" in weakness_text or "de-escalat" in weakness_text:
        suggestions.extend([
            ("suggest_accel_deesc_3", {"ACCEL_DEESCALATE_NORMAL_DAYS": 3}),
            ("suggest_accel_deesc_5", {"ACCEL_DEESCALATE_NORMAL_DAYS": 5}),
            ("suggest_fast_hysteresis_recovery", {
                "DEESCALATE_CRISIS_DAYS": 1, "DEESCALATE_WARNING_DAYS": 1,
                "DEESCALATE_WATCH_DAYS": 1,
            }),
        ])

    return suggestions
