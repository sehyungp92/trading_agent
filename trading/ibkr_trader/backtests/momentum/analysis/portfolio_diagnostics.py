"""Portfolio diagnostics — 10-section comprehensive portfolio report.

Provides a full executive summary of the active momentum portfolio
including per-strategy contribution, heat utilization, correlation,
coordination events, monthly mix, drawdown attribution, and recommendations.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

from backtests.momentum.analysis._utils import parse_dt as _parse_dt, trade_date


POINT_VALUE_MNQ = 2.0


def _trade_date(trade) -> str | None:
    """Extract YYYY-MM-DD from entry_time (thin wrapper over shared trade_date)."""
    d = trade_date(trade)
    return d.isoformat() if d else None


def _trade_month(trade) -> str | None:
    dt = _parse_dt(getattr(trade, "entry_time", None))
    return dt.strftime("%Y-%m") if dt else None


def _hours_held(trade) -> float:
    entry = _parse_dt(getattr(trade, "entry_time", None))
    exit_ = _parse_dt(getattr(trade, "exit_time", None))
    if entry is None or exit_ is None:
        return 0.0
    return max(0.0, (exit_ - entry).total_seconds() / 3600)


def _daily_pnl(trades: list) -> dict[str, float]:
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        d = _trade_date(t)
        if d:
            daily[d] += getattr(t, "pnl_dollars", 0.0)
    return dict(daily)


def _stats(trades: list) -> dict:
    if not trades:
        return {"count": 0, "wr": 0, "pf": 0, "net": 0, "avg": 0, "max_dd": 0}
    pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd))
    return {
        "count": len(pnl), "wr": float(np.mean(pnl > 0)),
        "pf": pf, "net": float(np.sum(pnl)),
        "avg": float(np.mean(pnl)), "max_dd": max_dd,
    }


def generate_portfolio_diagnostics_report(
    portfolio_result,
    nqdtc_trades: list,
    vdubus_trades: list,
    initial_equity: float = 10_000.0,
) -> str:
    """Generate 10-section portfolio diagnostics report.

    Args:
        portfolio_result: Object with trades, blocked_trades, equity_curve,
            equity_timestamps, rule_blocks (dict), rule_blocked_pnl (dict),
            max_concurrent (int).
        nqdtc_trades: NQDTC strategy trade list.
        vdubus_trades: VDUBUS strategy trade list.
        initial_equity: Starting equity.

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  MOMENTUM PORTFOLIO DIAGNOSTICS")
    lines.append("=" * 72)
    lines.append("")

    all_trades = getattr(portfolio_result, "trades", nqdtc_trades + vdubus_trades)
    equity_curve = getattr(portfolio_result, "equity_curve", None)
    equity_ts = getattr(portfolio_result, "equity_timestamps", None)
    rule_blocks = getattr(portfolio_result, "rule_blocks", {})
    rule_blocked_pnl = getattr(portfolio_result, "rule_blocked_pnl", {})
    max_concurrent = getattr(portfolio_result, "max_concurrent", 0)

    strats = {"nqdtc": nqdtc_trades, "vdubus": vdubus_trades}

    # ──────────────────────────────────────────────
    # 1. OVERVIEW
    # ──────────────────────────────────────────────
    ps = _stats(all_trades)
    lines.append("  1. OVERVIEW")
    lines.append("  " + "-" * 55)
    lines.append(f"    Total trades:      {ps['count']}")
    lines.append(f"    Net profit:        ${ps['net']:+,.0f}")
    lines.append(f"    Profit factor:     {ps['pf']:.2f}")
    lines.append(f"    Win rate:          {ps['wr']*100:.1f}%")
    lines.append(f"    Max drawdown:      ${ps['max_dd']:+,.0f}")
    lines.append(f"    Return on equity:  {ps['net']/initial_equity*100:.1f}%")

    # ──────────────────────────────────────────────
    # 2. PER-STRATEGY CONTRIBUTION
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  2. PER-STRATEGY CONTRIBUTION")
    lines.append("  " + "-" * 55)
    lines.append(f"    {'Strategy':<10s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} {'Net PnL':>10s} {'% of P&L':>8s}")
    lines.append("    " + "-" * 50)

    for name, strades in strats.items():
        ss = _stats(strades)
        pct = ss["net"] / ps["net"] * 100 if ps["net"] != 0 else 0
        lines.append(
            f"    {name:<10s} {ss['count']:>6d} {ss['wr']*100:>5.1f}% {ss['pf']:>5.2f} "
            f"${ss['net']:>+9,.0f} {pct:>7.1f}%"
        )

    # ──────────────────────────────────────────────
    # 3. HEAT UTILIZATION
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  3. HEAT UTILIZATION")
    lines.append("  " + "-" * 55)

    # Build concurrent-position timeline
    events = []
    for t in all_trades:
        entry = _parse_dt(getattr(t, "entry_time", None))
        exit_ = _parse_dt(getattr(t, "exit_time", None))
        if entry and exit_:
            events.append((entry, 1))
            events.append((exit_, -1))

    if events:
        events.sort(key=lambda x: x[0])
        concurrent = 0
        peak_concurrent = 0
        time_at_cap = 0.0
        total_time = 0.0
        last_t = events[0][0]
        heat_cap = max_concurrent if max_concurrent > 0 else 3

        for evt_time, delta in events:
            elapsed = (evt_time - last_t).total_seconds() / 3600
            total_time += elapsed
            if concurrent >= heat_cap:
                time_at_cap += elapsed
            last_t = evt_time
            concurrent += delta
            peak_concurrent = max(peak_concurrent, concurrent)

        avg_concurrent = sum(
            getattr(t, "bars_held", 1) for t in all_trades
        ) / max(1, ps["count"])

        lines.append(f"    Max concurrent positions: {peak_concurrent}")
        lines.append(f"    Heat cap:                {heat_cap}")
        lines.append(f"    % time at heat cap:      {time_at_cap/max(1,total_time)*100:.1f}%")
        lines.append(f"    Avg bars held:           {avg_concurrent:.1f}")
    else:
        lines.append("    [Insufficient data]")

    # ──────────────────────────────────────────────
    # 4. CROSS-STRATEGY DAILY P&L CORRELATION
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  4. CROSS-STRATEGY DAILY P&L CORRELATION")
    lines.append("  " + "-" * 55)

    daily_series = {n: _daily_pnl(st) for n, st in strats.items()}
    all_dates = sorted(set().union(*(s.keys() for s in daily_series.values())))

    if len(all_dates) >= 5:
        arrays = {n: np.array([daily_series[n].get(d, 0.0) for d in all_dates]) for n in strats}
        names_list = list(strats.keys())
        matrix = np.corrcoef([arrays[n] for n in names_list])

        for i, n1 in enumerate(names_list):
            for j, n2 in enumerate(names_list):
                if j > i:
                    lines.append(f"    {n1} vs {n2}: {matrix[i,j]:.3f}")
    else:
        lines.append("    [Insufficient trading days]")

    # ──────────────────────────────────────────────
    # 5. SIMULTANEOUS POSITIONS
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  5. SIMULTANEOUS POSITIONS ANALYSIS")
    lines.append("  " + "-" * 55)

    overlap_count = 0
    overlap_same_dir = 0
    strat_names = list(strats.keys())
    for i in range(len(strat_names)):
        for j in range(i + 1, len(strat_names)):
            n1, n2 = strat_names[i], strat_names[j]
            for t1 in strats[n1]:
                e1 = _parse_dt(getattr(t1, "entry_time", None))
                x1 = _parse_dt(getattr(t1, "exit_time", None))
                if not e1 or not x1:
                    continue
                for t2 in strats[n2]:
                    e2 = _parse_dt(getattr(t2, "entry_time", None))
                    x2 = _parse_dt(getattr(t2, "exit_time", None))
                    if not e2 or not x2:
                        continue
                    # Check temporal overlap
                    if e1 < x2 and e2 < x1:
                        overlap_count += 1
                        d1 = getattr(t1, "direction", 1)
                        d2 = getattr(t2, "direction", 1)
                        if d1 == d2:
                            overlap_same_dir += 1

    lines.append(f"    Overlapping position pairs: {overlap_count}")
    lines.append(f"    Same direction:            {overlap_same_dir} ({overlap_same_dir/max(1,overlap_count)*100:.0f}%)")
    lines.append(f"    Opposing direction:         {overlap_count - overlap_same_dir}")

    # ──────────────────────────────────────────────
    # 6. COORDINATION EVENTS
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  6. COORDINATION EVENTS (blocks, cooldowns)")
    lines.append("  " + "-" * 55)

    if rule_blocks:
        total_blocks = sum(v if isinstance(v, int) else len(v) for v in rule_blocks.values())
        lines.append(f"    Total blocks: {total_blocks}")
        for rule, val in rule_blocks.items():
            cnt = val if isinstance(val, int) else len(val)
            blocked_pnl = rule_blocked_pnl.get(rule, 0)
            lines.append(f"      {rule:<30s}: {cnt:>4d} blocks, foregone P&L: ${blocked_pnl:+,.0f}")
    else:
        lines.append("    No coordination blocks recorded.")

    blocked_trades = getattr(portfolio_result, "blocked_trades", [])
    if blocked_trades:
        b_pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in blocked_trades)
        lines.append(f"    Blocked trades total:  {len(blocked_trades)}")
        lines.append(f"    Blocked trades P&L:    ${b_pnl:+,.0f}")

    # ──────────────────────────────────────────────
    # 7. MONTHLY MIX TABLE
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  7. MONTHLY MIX TABLE")
    lines.append("  " + "-" * 55)

    months = set()
    monthly_strat: dict[str, dict[str, float]] = {n: defaultdict(float) for n in strats}
    monthly_strat["TOTAL"] = defaultdict(float)

    for name, strades in strats.items():
        for t in strades:
            m = _trade_month(t)
            if m:
                months.add(m)
                pnl = getattr(t, "pnl_dollars", 0.0)
                monthly_strat[name][m] += pnl
                monthly_strat["TOTAL"][m] += pnl

    sorted_months = sorted(months)
    header = f"    {'Month':>8s}"
    for n in list(strats.keys()) + ["TOTAL"]:
        header += f" {n:>10s}"
    lines.append(header)
    lines.append("    " + "-" * (9 + 11 * (len(strats) + 1)))

    for m in sorted_months:
        row = f"    {m:>8s}"
        for n in list(strats.keys()) + ["TOTAL"]:
            row += f" ${monthly_strat[n][m]:>+8,.0f}"
        lines.append(row)

    # ──────────────────────────────────────────────
    # 8. DRAWDOWN ATTRIBUTION
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  8. DRAWDOWN ATTRIBUTION")
    lines.append("  " + "-" * 55)

    # Sort trades by time for temporal equity curve (all_trades may be grouped by strategy)
    sorted_trades = sorted(all_trades, key=lambda t: getattr(t, "entry_time", getattr(t, "exit_time", None)) or 0)
    pnl_all = np.array([getattr(t, "pnl_dollars", 0.0) for t in sorted_trades])
    if len(pnl_all) > 0:
        cum = np.cumsum(pnl_all)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak

        # Find DD trough indices
        sorted_dd_idx = np.argsort(dd)[:3]

        for rank, trough_idx in enumerate(sorted_dd_idx, 1):
            # Find start of this DD episode (last peak before trough)
            peak_val = peak[trough_idx]
            start_idx = 0
            for k in range(trough_idx, -1, -1):
                if cum[k] >= peak_val:
                    start_idx = k
                    break

            episode_trades = sorted_trades[start_idx:trough_idx + 1]
            ep_pnl = float(dd[trough_idx])
            lines.append(f"\n    DD Episode #{rank}: ${ep_pnl:+,.0f}")

            # Attribution per strategy
            for name in strats:
                strat_pnl_in_ep = sum(
                    getattr(t, "pnl_dollars", 0.0) for t in episode_trades
                    if getattr(t, "entry_class", "") == name or
                    getattr(t, "strategy", "") == name
                )
                pct = strat_pnl_in_ep / ep_pnl * 100 if ep_pnl != 0 else 0
                lines.append(f"      {name:<10s}: ${strat_pnl_in_ep:+,.0f} ({pct:.0f}%)")

    # ──────────────────────────────────────────────
    # 9. CAPITAL EFFICIENCY
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  9. CAPITAL EFFICIENCY")
    lines.append("  " + "-" * 55)

    total_hold_hrs = sum(_hours_held(t) for t in all_trades)
    r_per_hr = ps["net"] / total_hold_hrs if total_hold_hrs > 0 else 0

    lines.append(f"    Total hold hours:      {total_hold_hrs:,.1f}")
    lines.append(f"    $ per hour deployed:   ${r_per_hr:+,.2f}")

    for name, strades in strats.items():
        s_pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in strades)
        s_hrs = sum(_hours_held(t) for t in strades)
        s_eff = s_pnl / s_hrs if s_hrs > 0 else 0
        lines.append(f"    {name:<10s}: ${s_eff:+,.2f}/hr ({s_hrs:.0f}h)")

    # ──────────────────────────────────────────────
    # 10. RECOMMENDATIONS
    # ──────────────────────────────────────────────
    lines.append("")
    lines.append("  10. RECOMMENDATIONS")
    lines.append("  " + "-" * 55)

    recs = []

    # Check strategy balance
    strat_stats = {n: _stats(st) for n, st in strats.items()}
    worst_strat = min(strat_stats, key=lambda n: strat_stats[n]["net"])
    best_strat = max(strat_stats, key=lambda n: strat_stats[n]["net"])

    if strat_stats[worst_strat]["net"] < 0:
        recs.append(f"REVIEW {worst_strat}: negative P&L (${strat_stats[worst_strat]['net']:+,.0f}). "
                    f"Consider disabling or re-parameterizing.")

    # Check WR
    for name, ss in strat_stats.items():
        if ss["count"] > 10 and ss["wr"] < 0.40:
            recs.append(f"LOW WIN RATE for {name}: {ss['wr']*100:.0f}% — filter quality may need tightening.")

    # Check correlation
    if len(all_dates) >= 20:
        arrays_list = [np.array([daily_series[n].get(d, 0.0) for d in all_dates]) for n in strats]
        corr_mat = np.corrcoef(arrays_list)
        high_corr_pairs = []
        names_l = list(strats.keys())
        for i in range(len(names_l)):
            for j in range(i + 1, len(names_l)):
                if corr_mat[i, j] > 0.5:
                    high_corr_pairs.append((names_l[i], names_l[j], corr_mat[i, j]))
        if high_corr_pairs:
            for n1, n2, c in high_corr_pairs:
                recs.append(f"HIGH CORRELATION {n1}/{n2} ({c:.2f}): diversification benefit limited.")

    # Check blocked trade cost
    if rule_blocked_pnl:
        costly_rules = [(r, v) for r, v in rule_blocked_pnl.items() if v > 0]
        for rule, cost in costly_rules:
            recs.append(f"BLOCKING COST — '{rule}' blocked ${cost:+,.0f} of profitable trades.")

    if not recs:
        recs.append("No critical issues detected. Portfolio operating within expected parameters.")

    for i, rec in enumerate(recs, 1):
        lines.append(f"    {i}. {rec}")

    return "\n".join(lines)
