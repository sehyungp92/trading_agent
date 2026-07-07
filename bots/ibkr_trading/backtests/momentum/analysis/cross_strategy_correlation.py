"""Cross-strategy correlation analysis — quantify diversification benefit.

Measures daily P&L correlation, rolling co-movement, signal overlap, and
concurrent drawdown across the 3 momentum strategies.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

from backtests.momentum.analysis._utils import trade_date


def _trade_date(trade) -> str | None:
    """Extract YYYY-MM-DD from entry_time (thin wrapper over shared trade_date)."""
    d = trade_date(trade)
    return d.isoformat() if d else None


def _daily_pnl_series(trades: list) -> dict[str, float]:
    """Aggregate trade PnL by date."""
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        d = _trade_date(t)
        if d:
            daily[d] += getattr(t, "pnl_dollars", 0.0)
    return dict(daily)


def _align_series(*series_list: dict[str, float]) -> tuple[list[str], list[np.ndarray]]:
    """Align multiple date->value dicts to common dates, filling missing with 0."""
    all_dates = sorted(set().union(*(s.keys() for s in series_list)))
    arrays = [np.array([s.get(d, 0.0) for d in all_dates]) for s in series_list]
    return all_dates, arrays


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling Pearson correlation."""
    n = len(a)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        x = a[i - window + 1: i + 1]
        y = b[i - window + 1: i + 1]
        if np.std(x) > 0 and np.std(y) > 0:
            result[i] = float(np.corrcoef(x, y)[0, 1])
    return result


def generate_cross_strategy_correlation_report(trades_by_strategy: dict[str, list]) -> str:
    """Generate cross-strategy correlation and diversification report.

    Args:
        trades_by_strategy: Dict mapping strategy name to trade list,
                            e.g. {"nqdtc": [...], "vdubus": [...], "downturn": [...]}.

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  CROSS-STRATEGY CORRELATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    names = sorted(trades_by_strategy.keys())
    if len(names) < 2:
        lines.append("  Need at least 2 strategies for correlation analysis.")
        return "\n".join(lines)

    # Build daily P&L series
    daily_series = {n: _daily_pnl_series(trades_by_strategy[n]) for n in names}
    dates, arrays = _align_series(*[daily_series[n] for n in names])

    if len(dates) < 5:
        lines.append("  Insufficient trading days for correlation analysis.")
        return "\n".join(lines)

    # ── A. Daily P&L Correlation Matrix ──
    lines.append("  A. DAILY P&L CORRELATION MATRIX")
    lines.append("  " + "-" * 50)

    matrix = np.corrcoef(np.array(arrays))
    header = f"    {'':>12s}" + "".join(f" {n:>12s}" for n in names)
    lines.append(header)
    for i, n in enumerate(names):
        row = f"    {n:>12s}"
        for j in range(len(names)):
            row += f" {matrix[i, j]:>12.3f}"
        lines.append(row)

    avg_corr = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            avg_corr.append(matrix[i, j])
    lines.append(f"\n    Average pairwise correlation: {np.mean(avg_corr):.3f}")

    # ── B. Rolling 20-day correlation ──
    lines.append("")
    lines.append("  B. ROLLING 20-DAY CORRELATION")
    lines.append("  " + "-" * 50)

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], i, j))

    for n1, n2, i, j in pairs:
        rc = _rolling_corr(arrays[i], arrays[j], window=20)
        valid = rc[~np.isnan(rc)]
        if len(valid) > 0:
            lines.append(
                f"    {n1} vs {n2}: mean={np.mean(valid):.3f}, "
                f"min={np.min(valid):.3f}, max={np.max(valid):.3f}, "
                f"std={np.std(valid):.3f}"
            )
            # Pct time negatively correlated
            neg_pct = float(np.mean(valid < 0)) * 100
            lines.append(f"      Negatively correlated {neg_pct:.0f}% of windows")

    # ── C. Signal overlap frequency ──
    lines.append("")
    lines.append("  C. SIGNAL OVERLAP (same direction within 1hr)")
    lines.append("  " + "-" * 50)

    for n1, n2, _, _ in pairs:
        overlaps = 0
        agreement = 0
        t1_list = trades_by_strategy[n1]
        t2_list = trades_by_strategy[n2]

        for t1 in t1_list:
            et1 = getattr(t1, "entry_time", None)
            if et1 is None:
                continue
            if isinstance(et1, datetime):
                dt1 = et1
            else:
                try:
                    import pandas as pd
                    dt1 = pd.Timestamp(et1).to_pydatetime()
                except Exception:
                    continue

            for t2 in t2_list:
                et2 = getattr(t2, "entry_time", None)
                if et2 is None:
                    continue
                if isinstance(et2, datetime):
                    dt2 = et2
                else:
                    try:
                        import pandas as pd
                        dt2 = pd.Timestamp(et2).to_pydatetime()
                    except Exception:
                        continue

                if abs((dt1 - dt2).total_seconds()) <= 3600:
                    overlaps += 1
                    d1 = getattr(t1, "direction", 1)
                    d2 = getattr(t2, "direction", 1)
                    if d1 == d2:
                        agreement += 1

        lines.append(
            f"    {n1} vs {n2}: {overlaps} overlaps, "
            f"{agreement} same-direction ({agreement/max(1,overlaps)*100:.0f}%)"
        )

    # ── D. Max concurrent drawdown ──
    lines.append("")
    lines.append("  D. MAX CONCURRENT DRAWDOWN")
    lines.append("  " + "-" * 50)

    # Portfolio = sum of all strategy daily P&L
    portfolio = sum(arrays)
    cum_port = np.cumsum(portfolio)
    peak_port = np.maximum.accumulate(cum_port)
    dd_port = cum_port - peak_port
    max_dd_port = float(np.min(dd_port))

    # Sum of individual max DDs
    individual_max_dd_sum = 0.0
    for i, n in enumerate(names):
        cum = np.cumsum(arrays[i])
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        ind_max = float(np.min(dd))
        individual_max_dd_sum += ind_max
        lines.append(f"    {n:>12s} max DD: ${ind_max:+,.0f}")

    lines.append(f"    {'Portfolio':>12s} max DD: ${max_dd_port:+,.0f}")
    lines.append(f"    Sum of individual max DDs: ${individual_max_dd_sum:+,.0f}")

    # ── E. Diversification benefit ──
    lines.append("")
    lines.append("  E. DIVERSIFICATION BENEFIT")
    lines.append("  " + "-" * 50)

    if individual_max_dd_sum < 0:
        benefit = (1 - max_dd_port / individual_max_dd_sum) * 100
        lines.append(f"    DD reduction:  {benefit:.1f}%  (portfolio DD vs sum of individual DDs)")
    else:
        lines.append("    Cannot compute — no individual drawdowns.")

    # Sharpe comparison
    port_sharpe = float(np.mean(portfolio)) / float(np.std(portfolio)) * np.sqrt(252) if np.std(portfolio) > 0 else 0
    ind_sharpes = []
    for i, n in enumerate(names):
        s = float(np.mean(arrays[i])) / float(np.std(arrays[i])) * np.sqrt(252) if np.std(arrays[i]) > 0 else 0
        ind_sharpes.append(s)
        lines.append(f"    {n:>12s} annualized Sharpe: {s:.2f}")
    lines.append(f"    {'Portfolio':>12s} annualized Sharpe: {port_sharpe:.2f}")
    lines.append(f"    Sharpe improvement: {port_sharpe - max(ind_sharpes):+.2f} vs best individual")

    return "\n".join(lines)
