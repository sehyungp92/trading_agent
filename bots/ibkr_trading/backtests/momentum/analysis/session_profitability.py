"""Session profitability analysis — per-strategy heatmap across 7 session windows.

Breaks trading sessions into 7 windows (ETH-Asia, ETH-Europe, Pre-Market,
RTH-Open, RTH-Core, RTH-Close, Evening) and evaluates per-strategy performance.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime

from backtests.momentum.analysis._utils import (
    SESSION_ORDER,
    classify_session as _classify_session,
    utc_to_et as _utc_to_et,
)


def _compute_metrics(trades: list) -> dict:
    """Compute WR, avg R, expectancy, and count for a group of trades."""
    if not trades:
        return {"wr": 0.0, "avg_r": 0.0, "expectancy": 0.0, "count": 0}
    pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in trades])
    wins = pnl > 0
    wr = float(np.mean(wins)) if len(pnl) > 0 else 0.0
    avg_r = float(np.mean(pnl))
    expectancy = wr * float(np.mean(pnl[wins])) - (1 - wr) * abs(float(np.mean(pnl[~wins]))) if np.any(wins) and np.any(~wins) else avg_r
    return {"wr": wr, "avg_r": avg_r, "expectancy": expectancy, "count": len(trades)}


def generate_session_profitability_report(trades: list, strategies: dict[str, list] | None = None) -> str:
    """Generate session profitability heatmap report.

    Args:
        trades: Combined trade list (used if strategies is None).
        strategies: Optional dict mapping strategy name -> trade list.
                    If None, all trades are grouped under "all".

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  SESSION PROFITABILITY REPORT")
    lines.append("=" * 72)
    lines.append("")

    if strategies is None:
        strategies = {"all": trades}

    if not any(strategies.values()):
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # Bucket trades by strategy x session
    buckets: dict[str, dict[str, list]] = {
        name: defaultdict(list) for name in strategies
    }

    for name, strat_trades in strategies.items():
        for t in strat_trades:
            et_dt = _utc_to_et(getattr(t, "entry_time", None))
            if et_dt is None:
                continue
            session = _classify_session(et_dt)
            buckets[name][session].append(t)

    # Per-strategy heatmap
    lines.append("  A. SESSION x STRATEGY HEATMAP")
    lines.append("  " + "-" * 60)
    lines.append("")

    header = f"  {'Strategy':<12s}"
    for sess in SESSION_ORDER:
        header += f" | {sess:>11s}"
    lines.append(header)
    lines.append("  " + "-" * (13 + 14 * len(SESSION_ORDER)))

    best_worst: dict[str, tuple[str, str]] = {}

    for name in strategies:
        row_wr = f"  {name:<12s}"
        row_exp = f"  {'':12s}"
        row_cnt = f"  {'':12s}"
        best_exp, worst_exp = -1e9, 1e9
        best_sess, worst_sess = "", ""

        for sess in SESSION_ORDER:
            m = _compute_metrics(buckets[name][sess])
            row_wr += f" | {m['wr']*100:5.1f}%/{m['count']:>3d}"
            row_exp += f" |   ${m['expectancy']:>+7.0f}"
            if m["count"] >= 3:
                if m["expectancy"] > best_exp:
                    best_exp = m["expectancy"]
                    best_sess = sess
                if m["expectancy"] < worst_exp:
                    worst_exp = m["expectancy"]
                    worst_sess = sess

        lines.append(row_wr + "  (WR%/cnt)")
        lines.append(row_exp + "  (expectancy)")
        lines.append("")
        best_worst[name] = (best_sess, worst_sess)

    # Highlights
    lines.append("  B. BEST / WORST SESSION PER STRATEGY")
    lines.append("  " + "-" * 60)
    for name, (best, worst) in best_worst.items():
        lines.append(f"    {name:<12s}  BEST: {best or 'N/A':<14s}  WORST: {worst or 'N/A'}")

    lines.append("")
    lines.append("  Note: Times are approximate ET (UTC-5). Sessions with < 3 trades excluded from ranking.")

    return "\n".join(lines)
