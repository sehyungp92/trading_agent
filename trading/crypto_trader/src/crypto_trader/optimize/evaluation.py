"""5-dimension end-of-round evaluation report.

Provides:
- build_evaluation_report(): builds dict-based dimension data from metrics + journal
- build_end_of_round_report(): formats text report from EndOfRoundArtifacts
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from crypto_trader.backtest.metrics import PerformanceMetrics
from crypto_trader.optimize.types import EndOfRoundArtifacts


def build_evaluation_report(
    metrics: PerformanceMetrics,
    journal_entries: list[Any] | None = None,
    *,
    insights: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Build 5-dimension evaluation from metrics and optional journal entries.

    Args:
        metrics: PerformanceMetrics from backtest.
        journal_entries: Optional journal entries for grade/method counting.
        insights: Optional DiagnosticInsights for trade-level enrichment.

    Returns a dict with dimension names as keys and data dicts as values.
    Used by momentum_plugin to build dimension text reports.
    """
    entries = journal_entries or []

    return {
        "Signal Extraction": _evaluate_signal_extraction(metrics, entries, insights),
        "Signal Discrimination": _evaluate_signal_discrimination(metrics, entries, insights),
        "Entry Mechanism": _evaluate_entry_mechanism(metrics, entries, insights),
        "Trade Management": _evaluate_trade_management(metrics, entries, insights),
        "Exit Mechanism": _evaluate_exit_mechanism(metrics, entries, insights),
    }


def build_end_of_round_report(
    strategy_name: str,
    state: Any,
    artifacts: EndOfRoundArtifacts,
) -> str:
    """Format text report from plugin-provided EndOfRoundArtifacts.

    Generates:
    - Phase progression summary (score changes per phase)
    - Cumulative mutations applied
    - 5 dimension reports (text from artifacts.dimension_reports)
    - Extra sections
    - Overall verdict
    """
    lines = [f"{'=' * 60}"]
    lines.append(f"END-OF-ROUND EVALUATION: {strategy_name}")
    lines.append(f"{'=' * 60}")

    # Phase progression
    if hasattr(state, "phase_results") and state.phase_results:
        lines.append("\n--- Phase Progression ---")
        for phase_num in sorted(state.phase_results.keys()):
            result = state.phase_results[phase_num]
            base_s = result.get("base_score", 0.0)
            final_s = result.get("final_score", 0.0)
            focus = result.get("focus", "")
            accepted = result.get("accepted_count", 0)
            lines.append(
                f"  Phase {phase_num} ({focus}): "
                f"{base_s:.4f} -> {final_s:.4f} "
                f"({accepted} accepted)"
            )

    # Cumulative mutations
    if hasattr(state, "cumulative_mutations") and state.cumulative_mutations:
        lines.append("\n--- Cumulative Mutations ---")
        for key, val in sorted(state.cumulative_mutations.items()):
            lines.append(f"  {key}: {val}")

    # Dimension reports
    if artifacts.dimension_reports:
        lines.append("\n--- Dimension Reports ---")
        for dim_name, dim_text in artifacts.dimension_reports.items():
            lines.append(f"\n  [{dim_name}]")
            for line in dim_text.split("\n"):
                lines.append(f"    {line}")

    # Final diagnostics
    if artifacts.final_diagnostics_text:
        lines.append("\n--- Final Diagnostics ---")
        lines.append(artifacts.final_diagnostics_text)

    # Extra sections
    if artifacts.extra_sections:
        for section_name, section_text in artifacts.extra_sections.items():
            lines.append(f"\n--- {section_name} ---")
            lines.append(section_text)

    # Overall verdict
    if artifacts.overall_verdict:
        lines.append(f"\n{'=' * 60}")
        lines.append(f"VERDICT: {artifacts.overall_verdict}")
        lines.append(f"{'=' * 60}")

    return "\n".join(lines)


def format_dimension_text(name: str, data: dict[str, Any]) -> str:
    """Format a single dimension dict into readable text.

    Handles nested dicts and renders 'assessment' key as a header line.
    """
    lines: list[str] = []
    for k, v in data.items():
        if k == "assessment":
            lines.insert(0, f">> {v}")  # Assessment first
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for k2, v2 in v.items():
                lines.append(f"  {k2}: {v2:.4f}" if isinstance(v2, float) else f"  {k2}: {v2}")
        elif isinstance(v, list):
            lines.append(f"{k}: {', '.join(str(x) for x in v)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


# ── Internal helpers (used by momentum_plugin) ──────────────────────


def _evaluate_signal_extraction(
    metrics: PerformanceMetrics, entries: list[Any], insights: Any = None,
) -> dict[str, Any]:
    """Dimension 1: Signal detection and trade generation."""
    setup_grades = Counter()
    for e in entries:
        grade = getattr(e, "setup_grade", "unknown")
        setup_grades[grade] += 1

    result: dict[str, Any] = {
        "total_trades": metrics.total_trades,
        "a_setups": setup_grades.get("A", 0),
        "b_setups": setup_grades.get("B", 0),
        "grade_distribution": dict(setup_grades),
    }

    if insights is not None:
        result["mean_r"] = insights.mean_r
        if insights.per_confirmation:
            result["per_confirmation"] = insights.per_confirmation

        # R-distribution skewness — positive skew = fat right tail (good)
        r_skew = insights.r_stats.get("skew", 0.0)
        result["r_skew"] = r_skew

        # Per-asset edge breakdown
        if insights.per_asset:
            result["per_asset_edge"] = {
                sym: stats.get("avg_r", 0)
                for sym, stats in insights.per_asset.items()
            }

        # Concentration — few trades driving all profit?
        result["top1_pct"] = insights.concentration.get("top1_pct", 0)
        result["top20_pct"] = insights.concentration.get("top20_pct", 0)

        # Assessment — distinguish broad vs concentrated alpha
        concentrated = insights.concentration.get("top1_pct", 0) > 0.5
        if insights.mean_r > 0.2 and insights.profit_factor > 1.5:
            base = "Alpha: capturing meaningful alpha"
            if concentrated:
                base += " (concentrated — few trades driving profit)"
            result["assessment"] = base
        elif insights.mean_r > 0 and insights.profit_factor > 1.0:
            result["assessment"] = "Alpha: marginal edge — signal quality needs improvement"
        else:
            result["assessment"] = "Alpha: no edge — fundamental signal review needed"

    return result


def _evaluate_signal_discrimination(
    metrics: PerformanceMetrics, entries: list[Any], insights: Any = None,
) -> dict[str, Any]:
    """Dimension 2: Signal quality differentiation."""
    result: dict[str, Any] = {
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "a_setup_win_rate": metrics.a_setup_win_rate,
        "b_setup_win_rate": metrics.b_setup_win_rate,
        "a_b_gap": metrics.a_setup_win_rate - metrics.b_setup_win_rate,
    }

    if insights is not None:
        # Identify value-destroying confirmations with waste quantification
        bad_confs = [c for c, s in insights.per_confirmation.items()
                     if s["avg_r"] < 0 and s["n"] >= 2]
        total_r_lost = sum(
            s["total_r"] for c, s in insights.per_confirmation.items()
            if s["avg_r"] < 0 and s["n"] >= 2
        )
        if bad_confs:
            result["value_destroying_confirmations"] = bad_confs
            result["r_lost_to_bad_signals"] = total_r_lost
            result["assessment"] = (
                f"Discrimination: value-destroying signals: {', '.join(bad_confs)} "
                f"(total R lost: {total_r_lost:.2f})"
            )
        else:
            result["assessment"] = "Discrimination: all signals positive expectancy"

        # Grade WR gap quality — use R multiples not just WR
        a_data = insights.grade.get("A", {})
        b_data = insights.grade.get("B", {})
        if a_data and b_data:
            result["grade_wr_gap"] = a_data.get("wr", 0) - b_data.get("wr", 0)
            result["grade_r_gap"] = a_data.get("avg_r", 0) - b_data.get("avg_r", 0)

        # Confluence effectiveness ladder
        if len(insights.confluence) >= 2:
            sorted_confs = sorted(insights.confluence.items())
            result["confluence_ladder"] = {
                str(k): v.get("avg_r", 0)
                for k, v in sorted_confs
            }

    return result


def _evaluate_entry_mechanism(
    metrics: PerformanceMetrics, entries: list[Any], insights: Any = None,
) -> dict[str, Any]:
    """Dimension 3: Entry quality assessment."""
    entry_methods = Counter()
    for e in entries:
        method = getattr(e, "entry_method", "unknown")
        entry_methods[method] += 1

    result: dict[str, Any] = {
        "avg_mae_r": metrics.avg_mae_r,
        "entry_method_breakdown": dict(entry_methods),
    }

    if insights is not None:
        result["avg_mae_r_from_insights"] = insights.mfe_capture.get("avg_mae_r", 0)

        # Per-direction entry quality
        long_data = insights.direction.get("long", {})
        short_data = insights.direction.get("short", {})
        if long_data:
            result["long_avg_r"] = long_data.get("avg_r", 0)
        if short_data:
            result["short_avg_r"] = short_data.get("avg_r", 0)

        # Identify if entries are worse on one side
        long_r = long_data.get("avg_r", 0) if long_data else 0
        short_r = short_data.get("avg_r", 0) if short_data else 0
        direction_note = ""
        if long_data.get("n", 0) >= 2 and short_data.get("n", 0) >= 2:
            if long_r < -0.3 and short_r > 0:
                direction_note = " (long entries significantly worse)"
            elif short_r < -0.3 and long_r > 0:
                direction_note = " (short entries significantly worse)"

        # Assessment based on MAE context
        avg_mae = insights.mfe_capture.get("avg_mae_r", 0)
        if avg_mae > -0.3:
            result["assessment"] = f"Entry: tight entries — low adverse excursion{direction_note}"
        elif avg_mae > -0.6:
            result["assessment"] = f"Entry: moderate MAE — entries acceptable{direction_note}"
        else:
            result["assessment"] = f"Entry: high MAE — entries need improvement{direction_note}"

    return result


def _evaluate_trade_management(
    metrics: PerformanceMetrics, entries: list[Any], insights: Any = None,
) -> dict[str, Any]:
    """Dimension 4: In-trade management assessment."""
    exit_reasons = Counter()
    for e in entries:
        reason = getattr(e, "exit_reason", "unknown")
        exit_reasons[reason] += 1

    time_stops = exit_reasons.get("soft_time_stop", 0) + exit_reasons.get(
        "hard_time_stop", 0
    )

    result: dict[str, Any] = {
        "avg_bars_held": metrics.avg_bars_held,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "time_stop_rate": time_stops / max(metrics.total_trades, 1),
        "exit_reason_breakdown": dict(exit_reasons),
    }

    if insights is not None:
        cap = insights.mfe_capture.get("avg_capture_pct", 0)
        give = insights.mfe_capture.get("avg_giveback_pct", 0)
        result["avg_capture_pct"] = cap
        result["avg_giveback_pct"] = give

        # Duration analysis — trades held too long/short?
        avg_bars = insights.duration.get("avg_bars", 0)
        result["avg_bars_from_insights"] = avg_bars

        # Stagnation detection — high bars + low MFE
        avg_mfe = insights.mfe_capture.get("avg_mfe_r", 0)
        if avg_bars > 15 and avg_mfe < 0.5:
            result["stagnation_detected"] = True
        else:
            result["stagnation_detected"] = False

        # Per-direction management quality
        long_data = insights.direction.get("long", {})
        short_data = insights.direction.get("short", {})
        if long_data and short_data:
            result["direction_management"] = {
                "long_avg_r": long_data.get("avg_r", 0),
                "short_avg_r": short_data.get("avg_r", 0),
            }

        if cap < 0.35:
            stag_note = " (stagnation detected)" if result["stagnation_detected"] else ""
            result["assessment"] = f"Management: poor capture ({cap:.0%}) — significant alpha left on table{stag_note}"
        elif cap < 0.55:
            result["assessment"] = f"Management: moderate capture ({cap:.0%}) — room for improvement"
        else:
            result["assessment"] = f"Management: good capture ({cap:.0%})"

    return result


def _evaluate_exit_mechanism(
    metrics: PerformanceMetrics, entries: list[Any], insights: Any = None,
) -> dict[str, Any]:
    """Dimension 5: Exit quality assessment."""
    exit_reasons = Counter()
    for e in entries:
        reason = getattr(e, "exit_reason", "unknown")
        exit_reasons[reason] += 1

    trail_exits = exit_reasons.get("trailing_stop", 0)
    tp_exits = exit_reasons.get("tp1", 0) + exit_reasons.get("tp2", 0)

    result: dict[str, Any] = {
        "exit_efficiency": metrics.exit_efficiency,
        "avg_mfe_r": metrics.avg_mfe_r,
        "trail_exits": trail_exits,
        "tp_exits": tp_exits,
        "exit_type_breakdown": dict(exit_reasons),
    }

    if insights is not None:
        # Per-exit-reason P&L share
        if insights.exit_attribution:
            top_exits = sorted(
                insights.exit_attribution.items(),
                key=lambda x: abs(x[1].get("pnl_share", 0)),
                reverse=True,
            )[:3]
            result["top_exit_pnl_shares"] = {
                reason: stats["pnl_share"]
                for reason, stats in top_exits
            }

            # Full exit attribution detail
            result["exit_attribution_detail"] = {
                reason: {"n": stats.get("n", 0), "avg_r": stats.get("avg_r", 0)}
                for reason, stats in insights.exit_attribution.items()
            }

        # "Right-then-stopped" detection — trades with MFE > 1R but final R < 0
        right_then_stopped = [
            t for t in insights.worst_trades
            if (t.get("mfe_r") or 0) > 1.0 and (t.get("r_multiple") or 0) < 0
        ]
        result["right_then_stopped_count"] = len(right_then_stopped)
        if right_then_stopped:
            result["right_then_stopped"] = [
                {"symbol": t.get("symbol", ""), "mfe_r": t.get("mfe_r", 0),
                 "r_multiple": t.get("r_multiple", 0)}
                for t in right_then_stopped
            ]

        # Trail vs TP effectiveness
        trail_attr = insights.exit_attribution.get("trailing_stop", {})
        tp1_attr = insights.exit_attribution.get("tp1", {})
        if trail_attr.get("n", 0) > 0 and tp1_attr.get("n", 0) > 0:
            result["trail_vs_tp_avg_r"] = {
                "trail": trail_attr.get("avg_r", 0),
                "tp1": tp1_attr.get("avg_r", 0),
            }

        eff = metrics.exit_efficiency
        rts_note = (f" ({len(right_then_stopped)} right-then-stopped trades)"
                     if right_then_stopped else "")
        if eff < 0.35:
            result["assessment"] = f"Exit: poor efficiency ({eff:.0%}) — exits leaving too much on table{rts_note}"
        elif eff < 0.55:
            result["assessment"] = f"Exit: moderate efficiency ({eff:.0%}) — room for improvement{rts_note}"
        else:
            result["assessment"] = f"Exit: good efficiency ({eff:.0%}){rts_note}"

    return result
