"""Swing portfolio diagnostics — cross-strategy analysis.

Analyzes heat utilization, coordination impact, cross-strategy correlation,
and portfolio-level drawdown for the unified swing portfolio.
"""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta


def portfolio_diagnostic_report(
    portfolio_result,
    all_trades: dict[str, list] | None = None,
    heat_rejections: list[dict] | None = None,
    coordination_events: list[dict] | None = None,
) -> str:
    """Generate 10-section portfolio diagnostic report.

    Args:
        portfolio_result: UnifiedPortfolioResult from the engine.
        all_trades: Dict mapping strategy name to trade list.
        heat_rejections: List of heat cap rejection events.
        coordination_events: List of coordination (tighten/boost) events.
    """
    lines = ["=" * 60]
    lines.append("  SWING PORTFOLIO DIAGNOSTICS")
    lines.append("=" * 60)
    lines.append("")

    sections = [
        _portfolio_overview(portfolio_result),
        _strategy_contribution(portfolio_result),
        _heat_utilization(portfolio_result),
        _heat_rejection_analysis(portfolio_result, heat_rejections),
        _cross_strategy_correlation(all_trades),
        _simultaneous_positions(all_trades),
        _coordination_tighten(coordination_events, all_trades),
        _coordination_boost(coordination_events, all_trades),
        _monthly_strategy_mix(all_trades),
        _portfolio_drawdown(portfolio_result),
    ]
    return "\n\n".join(s for s in sections if s)


def _portfolio_overview(result) -> str:
    lines = ["  1. PORTFOLIO OVERVIEW"]
    lines.append("  " + "-" * 40)

    eq = np.array(result.combined_equity) if hasattr(result, 'combined_equity') else np.array([])
    if len(eq) > 1:
        total_return = (eq[-1] - eq[0]) / eq[0] * 100 if eq[0] > 0 else 0
        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(np.min(dd)) * 100
        max_dd_dollars = float(np.min(eq - peak))
        # Sharpe (annualized from daily returns)
        if len(eq) > 2:
            returns = np.diff(eq) / eq[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        lines.append(f"    Total return:          {total_return:+.1f}%")
        lines.append(f"    Final equity:          ${eq[-1]:,.0f}")
        lines.append(f"    Max drawdown:          {max_dd:.1f}% (${max_dd_dollars:+,.0f})")
        lines.append(f"    Sharpe ratio:          {sharpe:.2f}")

    # Aggregate trade stats
    sr = getattr(result, 'strategy_results', {})
    total_trades = sum(s.total_trades for s in sr.values())
    total_pnl = sum(s.total_pnl for s in sr.values())
    total_r = sum(s.total_r for s in sr.values())
    total_wins = sum(s.winning_trades for s in sr.values())
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    lines.append(f"    Total trades:          {total_trades}")
    lines.append(f"    Win rate:              {wr:.1f}%")
    lines.append(f"    Total PnL:             ${total_pnl:+,.0f}")
    lines.append(f"    Total R:               {total_r:+.1f}")

    overlay_pnl = getattr(result, 'overlay_pnl', 0.0)
    if overlay_pnl != 0:
        lines.append(f"    Overlay PnL:           ${overlay_pnl:+,.0f}")

    return "\n".join(lines)


def _strategy_contribution(result) -> str:
    lines = ["  2. STRATEGY CONTRIBUTION"]
    lines.append("  " + "-" * 40)

    sr = getattr(result, 'strategy_results', {})
    if not sr:
        lines.append("    No strategy results available.")
        return "\n".join(lines)

    total_pnl = sum(s.total_pnl for s in sr.values())

    header = f"    {'Strategy':14s} {'Trades':>6s} {'WR%':>6s} {'AvgR':>7s} {'TotalR':>7s} {'PnL $':>10s} {'% PnL':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for name, s in sorted(sr.items(), key=lambda x: -x[1].total_pnl):
        wr = s.winning_trades / s.total_trades * 100 if s.total_trades > 0 else 0
        avg_r = s.total_r / s.total_trades if s.total_trades > 0 else 0
        pct = s.total_pnl / total_pnl * 100 if total_pnl != 0 else 0
        lines.append(
            f"    {name:14s} {s.total_trades:6d} {wr:5.1f}% {avg_r:+7.3f} "
            f"{s.total_r:+7.1f} {s.total_pnl:+10,.0f} {pct:6.1f}%"
        )

    return "\n".join(lines)


def _heat_utilization(result) -> str:
    lines = ["  3. HEAT UTILIZATION"]
    lines.append("  " + "-" * 40)

    hs = getattr(result, 'heat_stats', None)
    if hs:
        lines.append(f"    Avg heat used:         {hs.avg_heat_pct:.1f}% of cap")
        lines.append(f"    Max heat reached:      {hs.max_heat_pct:.1f}%")
        lines.append(f"    Time at cap (100%):    {hs.pct_time_at_cap:.1f}%")

        if hs.avg_heat_pct < 30:
            verdict = "UNDERUTILIZED — portfolio rarely near capacity"
        elif hs.avg_heat_pct > 70:
            verdict = "HEAVILY UTILIZED — frequent capacity constraints"
        else:
            verdict = "MODERATE — balanced utilization"
        lines.append(f"    Verdict: {verdict}")
    else:
        lines.append("    Heat stats not available.")

    return "\n".join(lines)


def _heat_rejection_analysis(result, heat_rejections) -> str:
    lines = ["  4. HEAT REJECTION ANALYSIS"]
    lines.append("  " + "-" * 40)

    sr = getattr(result, 'strategy_results', {})
    total_blocked = sum(s.entries_blocked_by_heat for s in sr.values())
    lines.append(f"    Total entries blocked by heat: {total_blocked}")

    if total_blocked > 0:
        lines.append("")
        header = f"    {'Strategy':14s} {'Blocked':>7s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for name, s in sorted(sr.items(), key=lambda x: -x[1].entries_blocked_by_heat):
            if s.entries_blocked_by_heat > 0:
                lines.append(f"    {name:14s} {s.entries_blocked_by_heat:7d}")

    if heat_rejections:
        lines.append("")
        lines.append("    Rejection reasons:")
        reason_counts = Counter(r.get("reason", "unknown") for r in heat_rejections)
        for reason, cnt in reason_counts.most_common():
            lines.append(f"      {reason}: {cnt}")

    return "\n".join(lines)


def _cross_strategy_correlation(all_trades) -> str:
    lines = ["  5. CROSS-STRATEGY CORRELATION"]
    lines.append("  " + "-" * 40)

    if not all_trades or len(all_trades) < 2:
        lines.append("    Insufficient strategies for correlation analysis.")
        return "\n".join(lines)

    # Build daily return series per strategy
    daily_returns = {}
    all_dates = set()
    for name, trades in all_trades.items():
        dr = defaultdict(float)
        for t in trades:
            exit_t = getattr(t, 'exit_time', None)
            if exit_t is None:
                continue
            if isinstance(exit_t, datetime):
                date_key = exit_t.date()
            else:
                try:
                    import pandas as pd
                    date_key = pd.Timestamp(exit_t).date()
                except Exception:
                    continue
            dr[date_key] += getattr(t, 'r_multiple', 0.0)
            all_dates.add(date_key)
        daily_returns[name] = dr

    if not all_dates:
        lines.append("    No trade data for correlation.")
        return "\n".join(lines)

    sorted_dates = sorted(all_dates)
    names = sorted(daily_returns.keys())

    # Build matrix
    matrix = np.zeros((len(sorted_dates), len(names)))
    for j, name in enumerate(names):
        for i, d in enumerate(sorted_dates):
            matrix[i, j] = daily_returns[name].get(d, 0.0)

    # Correlation matrix
    if matrix.shape[0] > 5:
        corr = np.corrcoef(matrix.T)
        lines.append("")
        # Header
        hdr = "    " + " " * 14 + "".join(f"{n[:8]:>10s}" for n in names)
        lines.append(hdr)
        for i, name in enumerate(names):
            row = f"    {name:14s}"
            for j in range(len(names)):
                if np.isnan(corr[i, j]):
                    row += f"{'N/A':>10s}"
                else:
                    row += f"{corr[i, j]:+10.3f}"
            lines.append(row)

        # Average off-diagonal correlation
        mask = ~np.eye(len(names), dtype=bool)
        off_diag = corr[mask]
        off_diag = off_diag[~np.isnan(off_diag)]
        avg_corr = float(np.mean(off_diag)) if len(off_diag) > 0 else 0

        lines.append("")
        lines.append(f"    Avg off-diagonal corr: {avg_corr:+.3f}")
        if avg_corr < 0.3:
            verdict = "DIVERSIFIED (avg corr < 0.3)"
        elif avg_corr > 0.5:
            verdict = "CONCENTRATED (avg corr > 0.5)"
        else:
            verdict = "MODERATE (avg corr 0.3-0.5)"
        lines.append(f"    Verdict: {verdict}")
    else:
        lines.append("    Insufficient data points for correlation (need > 5 days).")

    return "\n".join(lines)


def _simultaneous_positions(all_trades) -> str:
    lines = ["  6. SIMULTANEOUS POSITIONS"]
    lines.append("  " + "-" * 40)

    if not all_trades:
        lines.append("    No trade data.")
        return "\n".join(lines)

    # Build position intervals
    intervals = []
    for name, trades in all_trades.items():
        for t in trades:
            entry = getattr(t, 'entry_time', None)
            exit_t = getattr(t, 'exit_time', None)
            if entry and exit_t:
                intervals.append((entry, exit_t, name))

    if not intervals:
        lines.append("    No position intervals.")
        return "\n".join(lines)

    # Sample daily: count concurrent positions
    all_times = [i[0] for i in intervals] + [i[1] for i in intervals]
    try:
        min_t = min(all_times)
        max_t = max(all_times)
    except TypeError:
        lines.append("    Cannot compare timestamps.")
        return "\n".join(lines)

    # Simple concurrent count using entry/exit pairs
    max_concurrent = 0
    concurrent_counts = []
    for iv in intervals:
        count = sum(1 for other in intervals if other[0] <= iv[0] <= other[1])
        concurrent_counts.append(count)
        max_concurrent = max(max_concurrent, count)

    avg_concurrent = np.mean(concurrent_counts) if concurrent_counts else 0

    lines.append(f"    Max concurrent positions: {max_concurrent}")
    lines.append(f"    Avg concurrent positions: {avg_concurrent:.1f}")

    # Overlap pairs
    overlap_pairs = Counter()
    for i, iv1 in enumerate(intervals):
        for j, iv2 in enumerate(intervals):
            if j <= i:
                continue
            if iv1[0] <= iv2[1] and iv2[0] <= iv1[1]:
                pair = tuple(sorted([iv1[2], iv2[2]]))
                overlap_pairs[pair] += 1

    if overlap_pairs:
        lines.append("    Top overlap pairs:")
        for (s1, s2), cnt in overlap_pairs.most_common(5):
            lines.append(f"      {s1} + {s2}: {cnt} overlapping trades")

    return "\n".join(lines)


def _coordination_tighten(coordination_events, all_trades) -> str:
    lines = ["  7. COORDINATION: TIGHTEN IMPACT"]
    lines.append("  " + "-" * 40)

    if not coordination_events:
        lines.append("    No coordination events recorded.")
        return "\n".join(lines)

    tighten_events = [e for e in coordination_events if e.get("type") == "tighten"]
    lines.append(f"    Tighten events: {len(tighten_events)}")

    if not tighten_events:
        lines.append("    (No tighten events)")
        return "\n".join(lines)

    lines.append("    Rule: ATRSS entry on symbol X → Helix stop tightened to BE")
    lines.append("")

    # Compare tightened vs untightened helix trades if trade data available
    helix_key = next((k for k in (all_trades or {}) if "helix" in k.lower()), None)
    if all_trades and helix_key:
        helix_trades = all_trades[helix_key]

        # Build tighten lookup: (symbol, time) for time-proximity matching
        tighten_entries = [(e.get("symbol"), e.get("time")) for e in tighten_events]

        def _was_tightened(trade) -> bool:
            sym = getattr(trade, 'symbol', '')
            entry_t = getattr(trade, 'entry_time', None)
            if entry_t is None:
                return False
            for t_sym, t_time in tighten_entries:
                if t_sym != sym or t_time is None:
                    continue
                # Tighten event must occur during the trade's lifetime
                exit_t = getattr(trade, 'exit_time', None)
                try:
                    if entry_t <= t_time and (exit_t is None or t_time <= exit_t):
                        return True
                except TypeError:
                    # Incompatible timestamp types
                    if str(t_sym) == str(sym):
                        return True
            return False

        tightened, untightened = [], []
        for t in helix_trades:
            (tightened if _was_tightened(t) else untightened).append(t)

        if tightened:
            t_r = np.array([getattr(t, 'r_multiple', 0.0) for t in tightened])
            u_r = np.array([getattr(t, 'r_multiple', 0.0) for t in untightened]) if untightened else np.array([0.0])
            lines.append(f"    Tightened Helix trades:   {len(tightened)} (avg R: {np.mean(t_r):+.3f})")
            lines.append(f"    Untightened Helix trades: {len(untightened)} (avg R: {np.mean(u_r):+.3f})")

            delta = float(np.mean(t_r)) - float(np.mean(u_r))
            verdict = "HELPFUL" if delta >= 0 else "HARMFUL"
            lines.append(f"    Delta: {delta:+.3f}R per trade → {verdict}")
        else:
            lines.append("    No Helix trades matched to tighten events.")

    return "\n".join(lines)


def _coordination_boost(coordination_events, all_trades) -> str:
    lines = ["  8. COORDINATION: BOOST IMPACT"]
    lines.append("  " + "-" * 40)

    if not coordination_events:
        lines.append("    No coordination events recorded.")
        return "\n".join(lines)

    boost_events = [e for e in coordination_events if e.get("type") == "boost"]
    lines.append(f"    Boost events: {len(boost_events)}")

    if not boost_events:
        lines.append("    (No boost events)")
        return "\n".join(lines)

    lines.append("    Rule: ATRSS active → Helix gets 1.25x size boost")

    # Compare boosted vs unboosted helix trades
    helix_key = next((k for k in (all_trades or {}) if "helix" in k.lower()), None)
    if all_trades and helix_key:
        helix_trades = all_trades[helix_key]
        boost_entries = [(e.get("symbol"), e.get("time")) for e in boost_events]

        def _was_boosted(trade) -> bool:
            sym = getattr(trade, 'symbol', '')
            entry_t = getattr(trade, 'entry_time', None)
            if entry_t is None:
                return False
            for b_sym, b_time in boost_entries:
                if b_sym != sym or b_time is None:
                    continue
                exit_t = getattr(trade, 'exit_time', None)
                try:
                    if entry_t <= b_time and (exit_t is None or b_time <= exit_t):
                        return True
                except TypeError:
                    if str(b_sym) == str(sym):
                        return True
            return False

        boosted, unboosted = [], []
        for t in helix_trades:
            (boosted if _was_boosted(t) else unboosted).append(t)

        if boosted:
            b_r = np.array([getattr(t, 'r_multiple', 0.0) for t in boosted])
            u_r = np.array([getattr(t, 'r_multiple', 0.0) for t in unboosted]) if unboosted else np.array([0.0])
            lines.append(f"    Boosted Helix trades:    {len(boosted)} (avg R: {np.mean(b_r):+.3f})")
            lines.append(f"    Unboosted Helix trades:  {len(unboosted)} (avg R: {np.mean(u_r):+.3f})")

            delta = float(np.mean(b_r)) - float(np.mean(u_r))
            # Net impact: 0.25x extra allocation on boosted trades
            extra_r = float(np.sum(b_r)) * 0.25 / 1.25  # extra R from the 25% boost
            lines.append(f"    Avg R delta:             {delta:+.3f}")
            lines.append(f"    Extra R from 0.25x boost:{extra_r:+.2f}")
            verdict = "HELPFUL" if extra_r > 0 else "HARMFUL"
            lines.append(f"    Verdict: {verdict}")

    return "\n".join(lines)


def _monthly_strategy_mix(all_trades) -> str:
    lines = ["  9. MONTHLY STRATEGY MIX"]
    lines.append("  " + "-" * 40)

    if not all_trades:
        lines.append("    No trade data.")
        return "\n".join(lines)

    # Group by month
    monthly = defaultdict(lambda: defaultdict(list))
    for name, trades in all_trades.items():
        for t in trades:
            exit_t = getattr(t, 'exit_time', None)
            if exit_t is None:
                continue
            if isinstance(exit_t, datetime):
                month_key = exit_t.strftime("%Y-%m")
            else:
                try:
                    import pandas as pd
                    month_key = pd.Timestamp(exit_t).strftime("%Y-%m")
                except Exception:
                    continue
            monthly[month_key][name].append(t)

    if not monthly:
        lines.append("    No monthly data.")
        return "\n".join(lines)

    strategies = sorted(all_trades.keys())
    hdr = f"    {'Month':>8s}" + "".join(f" {s[:7]:>8s}" for s in strategies) + f" {'Total':>8s}"
    lines.append(hdr)
    lines.append("    " + "-" * (len(hdr) - 4))

    for month in sorted(monthly.keys()):
        row = f"    {month:>8s}"
        total_r = 0.0
        for s in strategies:
            trades = monthly[month].get(s, [])
            r_sum = sum(getattr(t, 'r_multiple', 0.0) for t in trades)
            total_r += r_sum
            row += f" {r_sum:+8.1f}" if trades else f" {'—':>8s}"
        row += f" {total_r:+8.1f}"
        lines.append(row)

    return "\n".join(lines)


def _portfolio_drawdown(result) -> str:
    lines = ["  10. PORTFOLIO DRAWDOWN"]
    lines.append("  " + "-" * 40)

    eq = np.array(result.combined_equity) if hasattr(result, 'combined_equity') else np.array([])
    if len(eq) < 2:
        lines.append("    Insufficient equity data.")
        return "\n".join(lines)

    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    dd_pct = dd / np.where(peak > 0, peak, 1) * 100

    max_dd_idx = int(np.argmin(dd))
    max_dd_dollars = float(dd[max_dd_idx])
    max_dd_pct = float(dd_pct[max_dd_idx])

    lines.append(f"    Max drawdown:          ${max_dd_dollars:+,.0f} ({max_dd_pct:.1f}%)")

    # Find peak before max DD
    peak_idx = int(np.argmax(eq[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    ts = getattr(result, 'combined_timestamps', [])
    if len(ts) > 0 and peak_idx < len(ts) and max_dd_idx < len(ts):
        lines.append(f"    DD peak at:            index {peak_idx}")
        lines.append(f"    DD trough at:          index {max_dd_idx}")

    # Current DD
    current_dd = float(dd[-1])
    current_dd_pct = float(dd_pct[-1])
    lines.append(f"    Current drawdown:      ${current_dd:+,.0f} ({current_dd_pct:.1f}%)")

    return "\n".join(lines)
