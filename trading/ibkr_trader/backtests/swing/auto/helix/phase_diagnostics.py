"""Helix phase diagnostics -- D1-D6 modules for phase analysis.

D1: Class Attribution (always)
D2: Exit Efficiency (Phase 2+)
D3: Timeframe / Symbol breakdown (always)
D4: Stale / Waste analysis (Phase 2+)
D5: Tail analysis (Phase 3+)
D6: Phase delta (always)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .scoring import HelixMetrics


def generate_phase_diagnostics(
    phase: int,
    metrics: HelixMetrics,
    greedy_result: dict,
    state_dict: dict,
    all_trades: list | None = None,
    force_all_modules: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"HELIX PHASE {phase} DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Trades:          {metrics.total_trades}")
    lines.append(f"  Profit Factor:   {metrics.profit_factor:.2f}")
    lines.append(f"  Net Return:      {metrics.net_return_pct:.1f}%")
    lines.append(f"  Max R DD:        {metrics.max_r_dd:.2f}")
    lines.append(f"  Exit Efficiency: {metrics.exit_efficiency:.3f}")
    lines.append(f"  Waste Ratio:     {metrics.waste_ratio:.3f}")
    lines.append(f"  Tail Pct:        {metrics.tail_pct:.3f}")
    lines.append(f"  Bull PF:         {metrics.bull_pf:.2f}")
    lines.append(f"  Bear PF:         {metrics.bear_pf:.2f}")
    lines.append(f"  Min Regime PF:   {metrics.min_regime_pf:.2f}")
    lines.append(f"  Long/Short PF:   {metrics.long_pf:.2f}/{metrics.short_pf:.2f}")
    lines.append(f"  Min Side PF:     {metrics.min_side_pf:.2f}")
    lines.append(f"  Sharpe:          {metrics.sharpe:.2f}")
    lines.append(f"  Calmar (R):      {metrics.calmar_r:.2f}")
    lines.append(f"  Win Rate:        {metrics.win_rate:.1f}%")
    lines.append(f"  Total R:         {metrics.total_r:.2f}")
    lines.append(f"  Stale R:         {metrics.stale_r:.2f}")
    lines.append(f"  Short Hold R:    {metrics.short_hold_r:.2f}")
    lines.append("")

    # Greedy result summary
    kept = greedy_result.get("kept_features", [])
    lines.append("GREEDY RESULT")
    lines.append("-" * 40)
    lines.append(f"  Base Score:      {greedy_result.get('base_score', 0):.4f}")
    lines.append(f"  Final Score:     {greedy_result.get('final_score', 0):.4f}")
    lines.append(f"  Accepted:        {greedy_result.get('accepted_count', 0)}")
    lines.append(f"  Total Candidates:{greedy_result.get('total_candidates', 0)}")
    if kept:
        lines.append(f"  Kept Features:   {', '.join(kept)}")
    lines.append("")

    # D1: Class Attribution (always)
    if all_trades:
        lines.extend(_write_d1_class_attribution(all_trades))

    # D2: Exit Efficiency (Phase 2+ or forced)
    if all_trades and (phase >= 2 or force_all_modules):
        lines.extend(_write_d2_exit_efficiency(all_trades))

    # D3: TF/Symbol breakdown (always)
    if all_trades:
        lines.extend(_write_d3_tf_symbol(all_trades))

    # D4: Stale / Waste (Phase 2+ or forced)
    if all_trades and (phase >= 2 or force_all_modules):
        lines.extend(_write_d4_stale_waste(all_trades))

    # D5: Tail analysis (Phase 3+ or forced)
    if all_trades and (phase >= 3 or force_all_modules):
        lines.extend(_write_d5_tail_analysis(all_trades))

    # D6: Phase delta (always)
    lines.extend(_write_d6_phase_delta(phase, state_dict))

    return "\n".join(lines)


def get_diagnostic_gaps(phase: int, metrics: HelixMetrics) -> list[str]:
    """Return weakness descriptions for improve_diagnostics action."""
    gaps: list[str] = []

    if metrics.exit_efficiency < 0.25:
        gaps.append(
            f"Exit efficiency is very low ({metrics.exit_efficiency:.3f}); "
            "run enhanced D2 to analyze right-then-stopped leakage by class/hold time."
        )

    if metrics.waste_ratio < 0.50:
        gaps.append(
            f"Waste ratio is low ({metrics.waste_ratio:.3f}); "
            "stale/short-hold trades consuming too much gross R."
        )

    if metrics.min_regime_pf < 1.2:
        gaps.append(
            f"Min regime PF is weak ({metrics.min_regime_pf:.2f}); "
            "one regime is significantly underperforming."
        )

    if metrics.min_side_pf < 1.2:
        gaps.append(
            f"Min side PF is weak ({metrics.min_side_pf:.2f}); "
            "one direction is failing to discriminate positive from negative setups."
        )

    if metrics.tail_pct < 0.40:
        gaps.append(
            f"Tail preservation at {metrics.tail_pct:.2f} -- "
            "big winners being clipped. Check trailing/partial thresholds."
        )

    return gaps


def _write_d1_class_attribution(trades: list) -> list[str]:
    """D1: Per-class (A/B/C/D) attribution table."""
    lines = ["D1: CLASS ATTRIBUTION", "-" * 40]

    by_class: dict[str, list] = defaultdict(list)
    for t in trades:
        cls = getattr(t, "setup_class", "?")
        by_class[cls].append(t)

    header = f"  {'Class':<8} {'N':>5} {'WR%':>6} {'PF':>6} {'TotR':>8} {'AvgR':>7} {'MFE':>7} {'Pct':>6}"
    lines.append(header)
    lines.append("  " + "-" * 60)

    total_r = sum(t.r_multiple for t in trades)

    for cls in sorted(by_class):
        cls_trades = by_class[cls]
        n = len(cls_trades)
        wins = [t for t in cls_trades if t.r_multiple > 0]
        wr = len(wins) / n * 100 if n else 0
        gw = sum(t.r_multiple for t in wins)
        gl = abs(sum(t.r_multiple for t in cls_trades if t.r_multiple <= 0))
        pf = gw / gl if gl > 0 else 999
        tot_r = sum(t.r_multiple for t in cls_trades)
        avg_r = tot_r / n if n else 0
        avg_mfe = sum(t.mfe_r for t in cls_trades) / n if n else 0
        pct = tot_r / total_r * 100 if total_r != 0 else 0
        lines.append(f"  {cls:<8} {n:>5} {wr:>5.1f}% {pf:>6.2f} {tot_r:>+7.1f}R {avg_r:>+6.2f} {avg_mfe:>6.2f} {pct:>5.1f}%")

    lines.append("")
    return lines


def _write_d2_exit_efficiency(trades: list) -> list[str]:
    """D2: Exit efficiency by exit reason + right-then-stopped analysis."""
    lines = ["D2: EXIT EFFICIENCY", "-" * 40]

    by_reason: dict[str, list] = defaultdict(list)
    for t in trades:
        reason = getattr(t, "exit_reason", "UNKNOWN")
        by_reason[reason].append(t)

    header = f"  {'Reason':<15} {'N':>5} {'TotR':>8} {'AvgR':>7} {'AvgMFE':>7} {'Capture':>8}"
    lines.append(header)
    lines.append("  " + "-" * 55)

    for reason in sorted(by_reason):
        rt = by_reason[reason]
        n = len(rt)
        tot_r = sum(t.r_multiple for t in rt)
        avg_r = tot_r / n if n else 0
        avg_mfe = sum(t.mfe_r for t in rt) / n if n else 0
        sum_mfe = sum(t.mfe_r for t in rt if t.mfe_r > 0)
        capture = tot_r / sum_mfe if sum_mfe > 0 else 0
        lines.append(f"  {reason:<15} {n:>5} {tot_r:>+7.1f}R {avg_r:>+6.2f} {avg_mfe:>6.2f} {capture:>7.1%}")

    # Right-then-stopped: trades with MFE >= 1R but exited at loss or near-zero
    rts = [t for t in trades if t.mfe_r >= 1.0 and t.r_multiple < 0.5]
    if rts:
        leaked_r = sum(t.mfe_r - t.r_multiple for t in rts)
        lines.append("")
        lines.append(f"  Right-then-stopped: {len(rts)} trades, leaked {leaked_r:.1f}R")
        lines.append(f"  (MFE >= 1R but exited at < 0.5R)")

    lines.append("")
    return lines


def _write_d3_tf_symbol(trades: list) -> list[str]:
    """D3: Timeframe and symbol breakdown."""
    lines = ["D3: TIMEFRAME / SYMBOL BREAKDOWN", "-" * 40]

    # By origin TF
    by_tf: dict[str, list] = defaultdict(list)
    for t in trades:
        tf = getattr(t, "origin_tf", "?")
        by_tf[tf].append(t)

    lines.append("  By Origin Timeframe:")
    for tf in sorted(by_tf):
        tt = by_tf[tf]
        n = len(tt)
        tot_r = sum(t.r_multiple for t in tt)
        gw = sum(t.r_multiple for t in tt if t.r_multiple > 0)
        gl = abs(sum(t.r_multiple for t in tt if t.r_multiple <= 0))
        pf = gw / gl if gl > 0 else 999
        lines.append(f"    {tf}: {n} trades, {tot_r:+.1f}R, PF={pf:.2f}")

    # By symbol
    by_sym: dict[str, list] = defaultdict(list)
    for t in trades:
        sym = getattr(t, "symbol", "?")
        by_sym[sym].append(t)

    lines.append("")
    lines.append("  By Symbol:")
    for sym in sorted(by_sym):
        st = by_sym[sym]
        n = len(st)
        tot_r = sum(t.r_multiple for t in st)
        gw = sum(t.r_multiple for t in st if t.r_multiple > 0)
        gl = abs(sum(t.r_multiple for t in st if t.r_multiple <= 0))
        pf = gw / gl if gl > 0 else 999
        lines.append(f"    {sym}: {n} trades, {tot_r:+.1f}R, PF={pf:.2f}")

    lines.append("")
    return lines


def _write_d4_stale_waste(trades: list) -> list[str]:
    """D4: Stale and waste trade analysis."""
    lines = ["D4: STALE / WASTE ANALYSIS", "-" * 40]

    stale = [t for t in trades if getattr(t, "exit_reason", "") == "STALE"]
    short_hold = [t for t in trades if getattr(t, "bars_held", 0) <= 10 and t.r_multiple < 0]
    gross_win = sum(t.r_multiple for t in trades if t.r_multiple > 0)

    stale_r = sum(t.r_multiple for t in stale)
    short_r = sum(t.r_multiple for t in short_hold)

    lines.append(f"  Stale trades:      {len(stale)} ({stale_r:+.1f}R)")
    lines.append(f"  Short-hold losers: {len(short_hold)} ({short_r:+.1f}R, <=10 bars)")
    lines.append(f"  Gross win R:       {gross_win:.1f}R")
    waste_pct = (abs(stale_r) + abs(short_r)) / gross_win * 100 if gross_win > 0 else 0
    lines.append(f"  Waste as % of gross: {waste_pct:.1f}%")

    # Short hold by class
    if short_hold:
        by_cls: dict[str, int] = defaultdict(int)
        by_cls_r: dict[str, float] = defaultdict(float)
        for t in short_hold:
            cls = getattr(t, "setup_class", "?")
            by_cls[cls] += 1
            by_cls_r[cls] += t.r_multiple
        lines.append("")
        lines.append("  Short-hold losers by class:")
        for cls in sorted(by_cls):
            lines.append(f"    {cls}: {by_cls[cls]} trades, {by_cls_r[cls]:+.1f}R")

    lines.append("")
    return lines


def _write_d5_tail_analysis(trades: list) -> list[str]:
    """D5: Big winner (tail) analysis."""
    lines = ["D5: TAIL / BIG WINNER ANALYSIS", "-" * 40]

    wins = [t for t in trades if t.r_multiple > 0]
    big = [t for t in wins if t.r_multiple >= 3.0]
    huge = [t for t in wins if t.r_multiple >= 5.0]
    gross_win = sum(t.r_multiple for t in wins)

    big_r = sum(t.r_multiple for t in big)
    huge_r = sum(t.r_multiple for t in huge)

    lines.append(f"  Total winners:     {len(wins)}")
    lines.append(f"  Big (>=3R):        {len(big)} ({big_r:.1f}R, {big_r/gross_win*100:.0f}% of gross)" if gross_win > 0 else f"  Big (>=3R):        {len(big)}")
    lines.append(f"  Huge (>=5R):       {len(huge)} ({huge_r:.1f}R)" if huge else f"  Huge (>=5R):       0")

    if big:
        lines.append("")
        lines.append("  Top 10 winners:")
        for t in sorted(big, key=lambda x: x.r_multiple, reverse=True)[:10]:
            sym = getattr(t, "symbol", "?")
            cls = getattr(t, "setup_class", "?")
            held = getattr(t, "bars_held", 0)
            lines.append(f"    {sym} {cls} {t.r_multiple:+.2f}R (MFE={t.mfe_r:.2f}R, held={held}b)")

    lines.append("")
    return lines


def _write_d6_phase_delta(phase: int, state_dict: dict) -> list[str]:
    """D6: Before/after phase comparison."""
    lines = ["D6: PHASE DELTA", "-" * 40]

    phase_results = state_dict.get("phase_results", {})
    current = phase_results.get(phase, phase_results.get(str(phase), {}))
    if not current:
        lines.append("  No phase result available.")
        lines.append("")
        return lines

    base_metrics = {}
    final_metrics = current.get("final_metrics", {})
    # Try to get prior phase metrics for comparison
    prior_key = phase - 1
    prior = phase_results.get(prior_key, phase_results.get(str(prior_key), {}))
    if prior:
        base_metrics = prior.get("final_metrics", {})

    if base_metrics and final_metrics:
        lines.append(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Delta':>10}")
        lines.append("  " + "-" * 55)
        for key in ["total_trades", "profit_factor", "net_return_pct", "max_r_dd",
                     "exit_efficiency", "waste_ratio", "min_side_pf", "tail_pct", "sharpe", "calmar_r"]:
            before = float(base_metrics.get(key, 0))
            after = float(final_metrics.get(key, 0))
            delta = after - before
            lines.append(f"  {key:<20} {before:>10.3f} {after:>10.3f} {delta:>+10.3f}")
    elif final_metrics:
        lines.append("  Final metrics (no prior phase for comparison):")
        for key in ["total_trades", "profit_factor", "net_return_pct", "max_r_dd",
                     "exit_efficiency", "waste_ratio", "min_side_pf", "tail_pct"]:
            val = float(final_metrics.get(key, 0))
            lines.append(f"    {key}: {val:.3f}")

    lines.append("")
    return lines
