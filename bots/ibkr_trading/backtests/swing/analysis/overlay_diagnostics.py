"""Overlay strategy diagnostics — idle capital deployment analysis.

Analyzes the EMA crossover overlay strategy that deploys idle capital
into QQQ and GLD positions when primary strategies are not using capacity.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime


def overlay_diagnostic_report(
    overlay_trades: list,
    daily_data: dict | None = None,
    regime_series: dict | None = None,
) -> str:
    """Generate overlay strategy diagnostic report.

    Args:
        overlay_trades: List of overlay trade records with symbol, direction,
                       entry_time, exit_time, pnl_dollars, r_multiple,
                       entry_price, exit_price, bars_held, commission.
        daily_data: Optional dict of {symbol: NumpyBars} for utilization analysis.
        regime_series: Optional dict of {date: regime_label} for regime bucketing.
    """
    lines = ["=" * 60]
    lines.append("  OVERLAY STRATEGY DIAGNOSTICS")
    lines.append("=" * 60)
    lines.append("")

    if not overlay_trades:
        lines.append("  No overlay trades to analyze.")
        return "\n".join(lines)

    # ── 1. Overview ──
    lines.append("  1. OVERVIEW")
    lines.append("  " + "-" * 40)

    total_pnl = sum(getattr(t, 'pnl_dollars', 0.0) for t in overlay_trades)
    total_comm = sum(getattr(t, 'commission', 0.0) for t in overlay_trades)
    r_arr = np.array([getattr(t, 'r_multiple', 0.0) for t in overlay_trades])
    wr = float(np.mean(r_arr > 0)) * 100 if len(r_arr) > 0 else 0
    avg_r = float(np.mean(r_arr)) if len(r_arr) > 0 else 0

    lines.append(f"    Total overlay trades:    {len(overlay_trades)}")
    lines.append(f"    Total PnL:               ${total_pnl:+,.0f}")
    lines.append(f"    Win rate:                {wr:.1f}%")
    lines.append(f"    Avg R:                   {avg_r:+.3f}")
    lines.append(f"    Total commission:        ${total_comm:,.0f}")

    # ── 2. Symbol Breakdown ──
    lines.append("")
    lines.append("  2. SYMBOL BREAKDOWN")
    lines.append("  " + "-" * 40)

    by_sym = defaultdict(list)
    for t in overlay_trades:
        by_sym[getattr(t, 'symbol', 'unknown')].append(t)

    header = f"    {'Symbol':8s} {'Trades':>6s} {'WR%':>6s} {'AvgR':>7s} {'PnL $':>10s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for sym, trades in sorted(by_sym.items(), key=lambda x: -sum(getattr(t, 'pnl_dollars', 0.0) for t in x[1])):
        sym_r = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
        sym_pnl = sum(getattr(t, 'pnl_dollars', 0.0) for t in trades)
        sym_wr = float(np.mean(sym_r > 0)) * 100 if len(sym_r) > 0 else 0
        sym_avg_r = float(np.mean(sym_r)) if len(sym_r) > 0 else 0
        lines.append(f"    {sym:8s} {len(trades):6d} {sym_wr:5.1f}% {sym_avg_r:+7.3f} {sym_pnl:+10,.0f}")

    # ── 3. Regime Performance ──
    lines.append("")
    lines.append("  3. REGIME PERFORMANCE")
    lines.append("  " + "-" * 40)

    if regime_series:
        by_regime = defaultdict(list)
        for t in overlay_trades:
            entry_t = getattr(t, 'entry_time', None)
            if entry_t is None:
                continue
            if isinstance(entry_t, datetime):
                date_key = entry_t.date()
            else:
                try:
                    import pandas as pd
                    date_key = pd.Timestamp(entry_t).date()
                except Exception:
                    continue
            regime = regime_series.get(date_key, "UNKNOWN")
            by_regime[str(regime)].append(t)

        header = f"    {'Regime':12s} {'Trades':>6s} {'WR%':>6s} {'AvgR':>7s} {'TotalR':>8s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for regime, trades in sorted(by_regime.items(), key=lambda x: -len(x[1])):
            reg_r = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
            lines.append(
                f"    {regime:12s} {len(trades):6d} {float(np.mean(reg_r > 0))*100:5.1f}% "
                f"{float(np.mean(reg_r)):+7.3f} {float(np.sum(reg_r)):+8.1f}"
            )
    else:
        lines.append("    (Regime data not provided)")

    # ── 4. Capital Utilization ──
    lines.append("")
    lines.append("  4. CAPITAL UTILIZATION")
    lines.append("  " + "-" * 40)

    # Compute % of time invested
    total_bars_held = sum(getattr(t, 'bars_held', 0) for t in overlay_trades)
    if overlay_trades:
        all_entry = [getattr(t, 'entry_time', None) for t in overlay_trades]
        all_exit = [getattr(t, 'exit_time', None) for t in overlay_trades]
        valid_entry = [e for e in all_entry if e is not None]
        valid_exit = [e for e in all_exit if e is not None]
        if valid_entry and valid_exit:
            lines.append(f"    Total bars held:         {total_bars_held}")
            lines.append(f"    Avg bars per trade:      {total_bars_held / len(overlay_trades):.1f}")
        else:
            lines.append("    (Timestamp data incomplete)")
    else:
        lines.append("    (No overlay trades)")

    # ── 5. Entry Timing ──
    lines.append("")
    lines.append("  5. ENTRY TIMING")
    lines.append("  " + "-" * 40)

    # Day-of-week analysis
    dow_trades = defaultdict(list)
    for t in overlay_trades:
        entry_t = getattr(t, 'entry_time', None)
        if entry_t is None:
            continue
        if isinstance(entry_t, datetime):
            dow = entry_t.strftime("%A")
        else:
            try:
                import pandas as pd
                dow = pd.Timestamp(entry_t).strftime("%A")
            except Exception:
                continue
        dow_trades[dow].append(t)

    if dow_trades:
        for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            trades = dow_trades.get(dow, [])
            if trades:
                dow_r = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
                lines.append(f"    {dow:12s}: {len(trades):3d} trades, avg R {float(np.mean(dow_r)):+.3f}")

    # ── 6. Cost of Overlay ──
    lines.append("")
    lines.append("  6. COST OF OVERLAY")
    lines.append("  " + "-" * 40)

    gross_pnl = total_pnl + total_comm
    comm_pct = total_comm / gross_pnl * 100 if gross_pnl > 0 else 0
    lines.append(f"    Gross PnL:               ${gross_pnl:+,.0f}")
    lines.append(f"    Commission:              ${total_comm:,.0f}")
    lines.append(f"    Commission as % gross:   {comm_pct:.1f}%")
    lines.append(f"    Net PnL:                 ${total_pnl:+,.0f}")

    # ── 7. Verdict ──
    lines.append("")
    lines.append("  7. VERDICT")
    lines.append("  " + "-" * 40)

    if total_pnl > 0 and wr > 50:
        verdict = "NET_POSITIVE — overlay adds value"
    elif total_pnl > 0:
        verdict = "MARGINAL — positive but low conviction"
    else:
        verdict = "NET_NEGATIVE — overlay detracts from portfolio"

    lines.append(f"    {verdict}")
    lines.append(f"    Net overlay PnL: ${total_pnl:+,.0f} across {len(overlay_trades)} trades")

    return "\n".join(lines)
