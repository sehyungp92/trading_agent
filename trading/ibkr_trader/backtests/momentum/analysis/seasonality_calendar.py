"""Seasonality and calendar analysis — temporal patterns in momentum P&L.

Examines month-over-month trends, day-of-week patterns, macro event impact
(FOMC/NFP/CPI), options expiration week, and holiday-adjacent sessions.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, date, timedelta

from backtests.momentum.analysis._utils import parse_dt as _parse_dt, trade_date as _trade_date


WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# US market holidays (approximate — static dates only)
_STATIC_HOLIDAYS = {
    (1, 1), (7, 4), (12, 25),
}


def _third_friday(year: int, month: int) -> date:
    """Find the third Friday of a given month (OpEx)."""
    d = date(year, month, 1)
    # Find first Friday
    while d.weekday() != 4:
        d += timedelta(days=1)
    # Third Friday = first Friday + 14 days
    return d + timedelta(days=14)


def _is_opex_week(d: date) -> bool:
    """Check if a date falls in OpEx week (Mon-Fri of third Friday's week)."""
    opex = _third_friday(d.year, d.month)
    # Monday of OpEx week
    monday = opex - timedelta(days=opex.weekday())
    friday = monday + timedelta(days=4)
    return monday <= d <= friday


def _is_holiday_adjacent(d: date) -> bool:
    """Check if date is within 1 trading day of a static holiday."""
    for m, dy in _STATIC_HOLIDAYS:
        try:
            h = date(d.year, m, dy)
        except ValueError:
            continue
        delta = abs((d - h).days)
        if 0 < delta <= 2:
            return True
    return False


def generate_seasonality_report(
    trades: list,
    calendar_events: list[str] | None = None,
) -> str:
    """Generate seasonality and calendar pattern report.

    Args:
        trades: Trade records.
        calendar_events: Optional list of "YYYY-MM-DD" strings for macro events
                         (FOMC, NFP, CPI, etc.).

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  SEASONALITY & CALENDAR REPORT")
    lines.append("=" * 72)
    lines.append("")

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # ── A. Month-over-month P&L ──
    lines.append("  A. MONTH-OVER-MONTH P&L")
    lines.append("  " + "-" * 55)

    monthly: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        d = _trade_date(t)
        if d:
            key = f"{d.year}-{d.month:02d}"
            monthly[key].append(getattr(t, "pnl_dollars", 0.0))

    lines.append(f"    {'Month':>8s} {'Trades':>6s} {'WR%':>6s} {'Net PnL':>10s} {'Avg PnL':>10s}")
    lines.append("    " + "-" * 44)

    for key in sorted(monthly.keys()):
        pnl_list = monthly[key]
        arr = np.array(pnl_list)
        wr = float(np.mean(arr > 0)) * 100
        lines.append(
            f"    {key:>8s} {len(arr):>6d} {wr:>5.1f}% ${np.sum(arr):>+9,.0f} ${np.mean(arr):>+9,.0f}"
        )

    # Monthly seasonality (aggregate by calendar month)
    lines.append("")
    lines.append("  Calendar month averages:")
    month_agg: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        d = _trade_date(t)
        if d:
            month_agg[d.month].append(getattr(t, "pnl_dollars", 0.0))

    for m in range(1, 13):
        if m in month_agg:
            arr = np.array(month_agg[m])
            lines.append(
                f"    {MONTH_NAMES[m-1]:>4s}: {len(arr):>3d} trades, "
                f"WR={np.mean(arr>0)*100:.0f}%, net=${np.sum(arr):+,.0f}"
            )

    # ── B. Day-of-week patterns ──
    lines.append("")
    lines.append("  B. DAY-OF-WEEK PATTERNS")
    lines.append("  " + "-" * 55)

    dow: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        d = _trade_date(t)
        if d:
            dow[d.weekday()].append(getattr(t, "pnl_dollars", 0.0))

    lines.append(f"    {'Day':>5s} {'Count':>5s} {'WR%':>6s} {'Net PnL':>10s} {'Avg PnL':>10s} {'Med PnL':>10s}")
    lines.append("    " + "-" * 50)

    for wd in range(5):  # Mon-Fri
        if wd in dow:
            arr = np.array(dow[wd])
            lines.append(
                f"    {WEEKDAY_NAMES[wd]:>5s} {len(arr):>5d} {np.mean(arr>0)*100:>5.1f}% "
                f"${np.sum(arr):>+9,.0f} ${np.mean(arr):>+9,.0f} ${np.median(arr):>+9,.0f}"
            )

    # ── C. Macro event impact ──
    lines.append("")
    lines.append("  C. MACRO EVENT DAY IMPACT")
    lines.append("  " + "-" * 55)

    if calendar_events:
        event_set = set(calendar_events)
        event_trades = [t for t in trades if str(_trade_date(t)) in event_set]
        non_event_trades = [t for t in trades if str(_trade_date(t)) not in event_set]

        e_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in event_trades]) if event_trades else np.array([])
        n_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in non_event_trades]) if non_event_trades else np.array([])

        lines.append(f"    Event days:      {len(event_trades)} trades, "
                     f"WR={np.mean(e_pnl>0)*100:.0f}%, avg=${np.mean(e_pnl):+,.0f}" if len(e_pnl) > 0 else "    Event days: 0 trades")
        lines.append(f"    Non-event days:  {len(non_event_trades)} trades, "
                     f"WR={np.mean(n_pnl>0)*100:.0f}%, avg=${np.mean(n_pnl):+,.0f}" if len(n_pnl) > 0 else "    Non-event days: 0 trades")

        if len(e_pnl) > 0 and len(n_pnl) > 0:
            diff = float(np.mean(e_pnl)) - float(np.mean(n_pnl))
            lines.append(f"    Event-day delta: ${diff:+,.0f}/trade")
    else:
        lines.append("    [Calendar events not provided — skipping macro analysis]")

    # ── D. OpEx week analysis ──
    lines.append("")
    lines.append("  D. OPTIONS EXPIRATION WEEK")
    lines.append("  " + "-" * 55)

    opex_trades = [t for t in trades if _trade_date(t) and _is_opex_week(_trade_date(t))]
    non_opex_trades = [t for t in trades if _trade_date(t) and not _is_opex_week(_trade_date(t))]

    o_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in opex_trades]) if opex_trades else np.array([])
    no_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in non_opex_trades]) if non_opex_trades else np.array([])

    if len(o_pnl) > 0:
        lines.append(f"    OpEx week:      {len(o_pnl)} trades, WR={np.mean(o_pnl>0)*100:.0f}%, avg=${np.mean(o_pnl):+,.0f}")
    else:
        lines.append("    OpEx week:      0 trades")
    if len(no_pnl) > 0:
        lines.append(f"    Non-OpEx:       {len(no_pnl)} trades, WR={np.mean(no_pnl>0)*100:.0f}%, avg=${np.mean(no_pnl):+,.0f}")

    if len(o_pnl) > 0 and len(no_pnl) > 0:
        diff = float(np.mean(o_pnl)) - float(np.mean(no_pnl))
        lines.append(f"    OpEx delta:     ${diff:+,.0f}/trade")

    # ── E. Holiday-adjacent sessions ──
    lines.append("")
    lines.append("  E. HOLIDAY-ADJACENT SESSIONS")
    lines.append("  " + "-" * 55)

    hol_trades = [t for t in trades if _trade_date(t) and _is_holiday_adjacent(_trade_date(t))]
    reg_trades = [t for t in trades if _trade_date(t) and not _is_holiday_adjacent(_trade_date(t))]

    h_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in hol_trades]) if hol_trades else np.array([])
    r_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in reg_trades]) if reg_trades else np.array([])

    if len(h_pnl) > 0:
        lines.append(f"    Holiday-adjacent: {len(h_pnl)} trades, WR={np.mean(h_pnl>0)*100:.0f}%, avg=${np.mean(h_pnl):+,.0f}")
    else:
        lines.append("    Holiday-adjacent: 0 trades")
    if len(r_pnl) > 0:
        lines.append(f"    Regular:          {len(r_pnl)} trades, WR={np.mean(r_pnl>0)*100:.0f}%, avg=${np.mean(r_pnl):+,.0f}")

    return "\n".join(lines)
