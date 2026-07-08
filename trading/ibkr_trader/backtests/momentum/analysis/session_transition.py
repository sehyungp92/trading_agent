"""Session transition analysis — quantify ETH/RTH transition weakness.

Analyzes trade performance across session windows, cross-session trades,
and the known 09:30 whipsaw vulnerability for NQDTC.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, time


# Session windows (Eastern Time)
SESSION_WINDOWS = {
    "PRE_RTH":   (time(8, 30), time(9, 30)),
    "RTH_OPEN":  (time(9, 30), time(10, 0)),
    "MORNING":   (time(10, 0), time(12, 0)),
    "MIDDAY":    (time(12, 0), time(14, 0)),
    "CLOSE":     (time(14, 0), time(16, 0)),
    "POST_RTH":  (time(16, 0), time(18, 0)),
    "EVENING":   (time(18, 0), time(23, 59)),
    "OVERNIGHT": (time(0, 0), time(8, 30)),
}


def _get_et_time(dt_val) -> time | None:
    """Extract Eastern Time from a datetime/timestamp."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        dt = dt_val
    else:
        try:
            import pandas as pd
            dt = pd.Timestamp(dt_val).to_pydatetime()
        except Exception:
            return None
    # Assume timestamps are in ET or UTC; if UTC, subtract 5h (approximate)
    # In practice, momentum backtests use ET-aware timestamps
    return dt.time()


def _classify_window(t: time) -> str:
    """Classify a time into a session window."""
    for name, (start, end) in SESSION_WINDOWS.items():
        if name == "EVENING":
            if t >= start:
                return name
        elif name == "OVERNIGHT":
            if t < time(8, 30):
                return name
        else:
            if start <= t < end:
                return name
    return "UNKNOWN"


def session_transition_report(trades: list, strategy: str = "momentum") -> str:
    """Generate session transition performance report.

    Args:
        trades: List of trade records with entry_time, exit_time, r_multiple,
                direction, and optionally session fields.
        strategy: Strategy name for header.
    """
    lines = ["=" * 60]
    lines.append(f"  {strategy.upper()} SESSION TRANSITION REPORT")
    lines.append("=" * 60)
    lines.append("")

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # ── A. Transition Window Performance ──
    lines.append("  A. SESSION WINDOW PERFORMANCE")
    lines.append("  " + "-" * 50)

    window_trades = defaultdict(list)
    for t in trades:
        entry_t = _get_et_time(getattr(t, 'entry_time', None))
        if entry_t is not None:
            window = _classify_window(entry_t)
            window_trades[window].append(t)

    header = f"    {'Window':12s} {'Trades':>6s} {'WR%':>6s} {'AvgR':>7s} {'TotalR':>8s} {'Verdict':>10s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    overall_avg_r = np.mean([getattr(t, 'r_multiple', 0.0) for t in trades]) if trades else 0

    for window_name in SESSION_WINDOWS:
        wt = window_trades.get(window_name, [])
        if not wt:
            lines.append(f"    {window_name:12s} {'—':>6s} {'—':>6s} {'—':>7s} {'—':>8s} {'—':>10s}")
            continue
        r_arr = np.array([getattr(t, 'r_multiple', 0.0) for t in wt])
        wr = float(np.mean(r_arr > 0)) * 100
        avg_r = float(np.mean(r_arr))
        total_r = float(np.sum(r_arr))

        if avg_r > overall_avg_r + 0.1:
            verdict = "STRONG"
        elif avg_r < overall_avg_r - 0.1:
            verdict = "WEAK"
        else:
            verdict = "NEUTRAL"

        lines.append(
            f"    {window_name:12s} {len(wt):6d} {wr:5.1f}% {avg_r:+7.3f} "
            f"{total_r:+8.1f} {verdict:>10s}"
        )

    # ── B. Cross-Session Trades ──
    lines.append("")
    lines.append("  B. CROSS-SESSION TRADES")
    lines.append("  " + "-" * 50)
    lines.append("    Trades spanning session boundaries (entry in one session, exit in another):")
    lines.append("")

    cross_session = []
    for t in trades:
        entry_t = _get_et_time(getattr(t, 'entry_time', None))
        exit_t = _get_et_time(getattr(t, 'exit_time', None))
        if entry_t is not None and exit_t is not None:
            entry_w = _classify_window(entry_t)
            exit_w = _classify_window(exit_t)
            if entry_w != exit_w:
                cross_session.append((t, entry_w, exit_w))

    lines.append(f"    Cross-session trades: {len(cross_session)} of {len(trades)} ({len(cross_session)/len(trades)*100:.1f}%)")

    if cross_session:
        # Group by transition type
        transitions = defaultdict(list)
        for t, ew, xw in cross_session:
            transitions[f"{ew} → {xw}"].append(t)

        lines.append("")
        header = f"    {'Transition':28s} {'Count':>5s} {'AvgR':>7s}"
        lines.append(header)
        for trans, ts in sorted(transitions.items(), key=lambda x: -len(x[1]))[:8]:
            r_arr = np.array([getattr(t, 'r_multiple', 0.0) for t in ts])
            lines.append(f"    {trans:28s} {len(ts):5d} {np.mean(r_arr):+7.3f}")

    # ── C. 09:30 Whipsaw Analysis (NQDTC-specific) ──
    lines.append("")
    lines.append("  C. 09:30 WHIPSAW ANALYSIS")
    lines.append("  " + "-" * 50)

    early_rth = []  # 09:00-09:45
    post_open = []  # 10:00+
    for t in trades:
        entry_t = _get_et_time(getattr(t, 'entry_time', None))
        if entry_t is None:
            continue
        if time(9, 0) <= entry_t < time(9, 45):
            early_rth.append(t)
        elif entry_t >= time(10, 0):
            post_open.append(t)

    if early_rth:
        r_early = np.array([getattr(t, 'r_multiple', 0.0) for t in early_rth])
        wr_early = float(np.mean(r_early > 0)) * 100
        avg_early = float(np.mean(r_early))
    else:
        wr_early = avg_early = 0

    if post_open:
        r_post = np.array([getattr(t, 'r_multiple', 0.0) for t in post_open])
        wr_post = float(np.mean(r_post > 0)) * 100
        avg_post = float(np.mean(r_post))
    else:
        wr_post = avg_post = 0

    lines.append(f"    Entries 09:00-09:45 ET:  {len(early_rth)} trades, WR {wr_early:.1f}%, avg R {avg_early:+.3f}")
    lines.append(f"    Entries 10:00+ ET:       {len(post_open)} trades, WR {wr_post:.1f}%, avg R {avg_post:+.3f}")

    if early_rth and post_open:
        delta = avg_early - avg_post
        if delta < -0.1:
            lines.append(f"    Verdict: WHIPSAW DRAG — 09:00-09:45 entries are {abs(delta):.3f}R worse")
        elif delta > 0.1:
            lines.append(f"    Verdict: EARLY EDGE — 09:00-09:45 entries are {delta:.3f}R better")
        else:
            lines.append(f"    Verdict: NO SIGNIFICANT DIFFERENCE ({delta:+.3f}R delta)")
    elif not early_rth:
        lines.append("    (No 09:00-09:45 entries — strategy may already block this window)")

    return "\n".join(lines)
