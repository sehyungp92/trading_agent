"""Comparative round diagnostics for IARIC pullback phased auto runs."""
from __future__ import annotations

from typing import Any

from backtests.stock.analysis.iaric_pullback_diagnostics import compute_pullback_diagnostic_snapshot


def _hdr(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}"


def _ctx_snapshot(ctx: dict[str, Any]) -> dict[str, Any]:
    return compute_pullback_diagnostic_snapshot(
        ctx.get("trades", []),
        metrics=ctx.get("metrics"),
        replay=ctx.get("replay"),
        daily_selections=ctx.get("daily_selections"),
        candidate_ledger=ctx.get("candidate_ledger"),
        funnel_counters=ctx.get("funnel_counters"),
        rejection_log=ctx.get("rejection_log"),
        shadow_outcomes=ctx.get("shadow_outcomes"),
        selection_attribution=ctx.get("selection_attribution"),
        fsm_log=ctx.get("fsm_log"),
    )


def _compare_line(label: str, base: float, other: float, fmt: str = "{:+.3f}") -> str:
    return f"  {label}: {fmt.format(base)} -> {fmt.format(other)} ({fmt.format(other - base)})"


def build_pullback_round_comparison_report(
    baseline_ctx: dict[str, Any],
    final_ctx: dict[str, Any],
    *,
    baseline_label: str = "Rebased Baseline",
    final_label: str = "Final Bundle",
    phase4_ctx: dict[str, Any] | None = None,
    phase5_ctx: dict[str, Any] | None = None,
) -> str:
    base = _ctx_snapshot(baseline_ctx)
    final = _ctx_snapshot(final_ctx)
    p4 = _ctx_snapshot(phase4_ctx) if phase4_ctx else None
    p5 = _ctx_snapshot(phase5_ctx) if phase5_ctx else None

    lines = [_hdr("Pullback Round Comparison")]
    lines.append(f"  {baseline_label}: trades={int(base['overview']['n'])}, avg_r={base['overview']['avg_r']:+.3f}, PF={base['overview']['pf']:.2f}, sharpe={base['overview']['sharpe']:.2f}")
    lines.append(f"  {final_label}: trades={int(final['overview']['n'])}, avg_r={final['overview']['avg_r']:+.3f}, PF={final['overview']['pf']:.2f}, sharpe={final['overview']['sharpe']:.2f}")
    lines.append("")
    lines.append("Signal extraction:")
    lines.append(_compare_line("accepted avg_r", base["shadow"]["actual"]["avg_r"], final["shadow"]["actual"]["avg_r"]))
    lines.append(_compare_line("rejected shadow avg_r", base["shadow"]["shadow"]["avg_r"], final["shadow"]["shadow"]["avg_r"]))
    lines.append(_compare_line("accept rate", base["funnel"]["accept_rate"], final["funnel"]["accept_rate"], "{:.1%}"))
    lines.append("")
    lines.append("Discrimination and crowding:")
    lines.append(_compare_line("crowded-day entered avg_r", base["selection"]["entered_avg_r"], final["selection"]["entered_avg_r"]))
    lines.append(_compare_line("crowded-day skipped shadow avg_r", base["selection"]["skipped_avg_shadow_r"], final["selection"]["skipped_avg_shadow_r"]))
    lines.append(f"  missed-alpha days: {base['selection']['days_with_missed_alpha']} -> {final['selection']['days_with_missed_alpha']} ({final['selection']['days_with_missed_alpha'] - base['selection']['days_with_missed_alpha']:+d})")
    lines.append("")
    lines.append("Exit frontier:")
    base_best = max(base["exit_frontier"], key=lambda item: item["avg_r"], default=None)
    final_best = max(final["exit_frontier"], key=lambda item: item["avg_r"], default=None)
    if base_best and final_best:
        lines.append(f"  best baseline variant: {base_best['label']} @ {base_best['avg_r']:+.3f}R")
        lines.append(f"  best final variant: {final_best['label']} @ {final_best['avg_r']:+.3f}R")
    lines.append(_compare_line("actual avg_r", base["overview"]["avg_r"], final["overview"]["avg_r"]))
    lines.append("")
    lines.append("Carry funnel:")
    lines.append(f"  EOD flatten trades: {base['carry_funnel']['eod']} -> {final['carry_funnel']['eod']} ({final['carry_funnel']['eod'] - base['carry_funnel']['eod']:+d})")
    lines.append(f"  profitable-at-close count: {base['carry_funnel']['profitable']} -> {final['carry_funnel']['profitable']} ({final['carry_funnel']['profitable'] - base['carry_funnel']['profitable']:+d})")
    lines.append(f"  quality-gated carry count: {base['carry_funnel']['flow_ok']} -> {final['carry_funnel']['flow_ok']} ({final['carry_funnel']['flow_ok'] - base['carry_funnel']['flow_ok']:+d})")
    lines.append("")
    lines.append("Concentration:")
    base_top_day = max(base["concentration"]["day_rows"], key=lambda item: item["n"], default={"label": "-", "n": 0, "avg_r": 0.0})
    final_top_day = max(final["concentration"]["day_rows"], key=lambda item: item["n"], default={"label": "-", "n": 0, "avg_r": 0.0})
    lines.append(f"  top weekday: {base_top_day['label']} n={base_top_day['n']} avg_r={base_top_day['avg_r']:+.3f} -> {final_top_day['label']} n={final_top_day['n']} avg_r={final_top_day['avg_r']:+.3f}")
    base_top_sector = max(base["concentration"]["sector_rows"], key=lambda item: item["n"], default={"label": "-", "n": 0, "avg_r": 0.0})
    final_top_sector = max(final["concentration"]["sector_rows"], key=lambda item: item["n"], default={"label": "-", "n": 0, "avg_r": 0.0})
    lines.append(f"  top sector: {base_top_sector['label']} n={base_top_sector['n']} avg_r={base_top_sector['avg_r']:+.3f} -> {final_top_sector['label']} n={final_top_sector['n']} avg_r={final_top_sector['avg_r']:+.3f}")
    if p4 and p5:
        lines.append("")
        lines.append("Phase 4 -> Phase 5 overlay check:")
        lines.append(_compare_line("phase avg_r", p4["overview"]["avg_r"], p5["overview"]["avg_r"]))
        lines.append(_compare_line("phase accept rate", p4["funnel"]["accept_rate"], p5["funnel"]["accept_rate"], "{:.1%}"))
        lines.append(_compare_line("phase skipped shadow avg_r", p4["selection"]["skipped_avg_shadow_r"], p5["selection"]["skipped_avg_shadow_r"]))
    return "\n".join(lines)
