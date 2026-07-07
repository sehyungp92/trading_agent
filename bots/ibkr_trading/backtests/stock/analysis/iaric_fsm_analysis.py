"""IARIC FSM analysis -- unique to IARIC's state-machine architecture.

Sections:
1. Transition matrix -- from/to counts, biggest drop-off
2. State dwell times -- mean/median time in SETUP_DETECTED, ACCEPTING
3. Funnel conversion rates -- per-day detected -> accepting -> ready -> entered -> profitable
4. Stale vs invalidated -- why setups die: timeout vs price crash vs insufficient acceptance
5. Re-entry after invalidation -- same symbol re-setup quality after cooldown
6. FSM velocity -- does faster ACCEPTING progression predict better R?
7. Daily FSM congestion -- active instances by time bucket
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _s1_transition_matrix(fsm_log: list[dict]) -> str:
    """Transition matrix: from/to state counts."""
    lines = [_hdr("FSM-1. Transition Matrix")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in fsm_log:
        fr = entry.get("from_state", "?")
        to = entry.get("to_state", "?")
        matrix[fr][to] += 1

    states = sorted(set(list(matrix.keys()) + [s for d in matrix.values() for s in d]))
    hdr = f"  {'From \\ To ->':<20s}"
    for s in states:
        hdr += f" {s[:12]:>12s}"
    lines.append(hdr)
    lines.append("  " + "-" * (20 + 13 * len(states)))

    for fr in states:
        row = f"  {fr:<20s}"
        for to in states:
            count = matrix[fr][to]
            row += f" {count:>12}" if count else f" {'·':>12s}"
        lines.append(row)

    # Biggest drop-off
    total_out: dict[str, int] = {}
    for fr, tos in matrix.items():
        total_out[fr] = sum(tos.values())
    if total_out:
        max_state = max(total_out, key=total_out.get)
        lines.append(f"\n  Highest volume state: {max_state} ({total_out[max_state]} transitions)")

    return "\n".join(lines)


def _s2_state_dwell_times(fsm_log: list[dict]) -> str:
    """Mean/median time in each state."""
    lines = [_hdr("FSM-2. State Dwell Times")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    # Track state entry times per symbol
    entry_times: dict[tuple[str, str], datetime] = {}  # (symbol, state) -> entered_at
    dwell_times: dict[str, list[float]] = defaultdict(list)  # state -> [minutes]

    for entry in fsm_log:
        sym = entry["symbol"]
        ts = entry["timestamp"]
        from_s = entry["from_state"]
        to_s = entry["to_state"]

        # Record dwell time for from_state
        key = (sym, from_s)
        if key in entry_times:
            dt = (ts - entry_times[key]).total_seconds() / 60
            dwell_times[from_s].append(dt)
            del entry_times[key]

        # Record entry into new state
        entry_times[(sym, to_s)] = ts

    lines.append(f"  {'State':<20s} {'Count':>6} {'Mean(min)':>10} {'Median':>10} {'Max':>10}")
    lines.append("  " + "-" * 60)

    for state in ["IDLE", "SETUP_DETECTED", "ACCEPTING", "READY_TO_ENTER", "INVALIDATED", "IN_POSITION"]:
        times = dwell_times.get(state, [])
        if not times:
            continue
        lines.append(
            f"  {state:<20s} {len(times):>6} {float(np.mean(times)):>10.1f}"
            f" {float(np.median(times)):>10.1f} {max(times):>10.1f}"
        )

    return "\n".join(lines)


def _s3_funnel_conversion(fsm_log: list[dict], trades: list[TradeRecord]) -> str:
    """Per-day funnel: detected -> accepting -> ready -> entered -> profitable."""
    lines = [_hdr("FSM-3. Daily Funnel Conversion")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    daily: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in fsm_log:
        d = entry["date"]
        to_s = entry["to_state"]
        daily[d][to_s] += 1

    # Count profitable entries per day
    profit_by_date: dict[date, int] = defaultdict(int)
    entry_by_date: dict[date, int] = defaultdict(int)
    for t in trades:
        d = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_time
        entry_by_date[d] += 1
        if t.is_winner:
            profit_by_date[d] += 1

    lines.append(f"  {'Date':<12s} {'Detect':>7} {'Accept':>7} {'Ready':>7} {'Enter':>7} {'Profit':>7} {'Conv%':>7}")
    lines.append("  " + "-" * 56)

    dates = sorted(daily.keys())
    for d in dates[:30]:  # Show first 30 days
        det = daily[d].get("SETUP_DETECTED", 0)
        acc = daily[d].get("ACCEPTING", 0)
        rdy = daily[d].get("READY_TO_ENTER", 0)
        ent = entry_by_date.get(d, 0)
        prof = profit_by_date.get(d, 0)
        conv = ent / det if det > 0 else 0
        lines.append(f"  {str(d):<12s} {det:>7} {acc:>7} {rdy:>7} {ent:>7} {prof:>7} {conv:>6.0%}")

    if len(dates) > 30:
        lines.append(f"  ... ({len(dates) - 30} more days)")

    # Summary
    total_det = sum(daily[d].get("SETUP_DETECTED", 0) for d in dates)
    total_ent = sum(entry_by_date.values())
    lines.append(f"\n  Overall: {total_det} detected -> {total_ent} entered ({total_ent/total_det:.1%} conversion)" if total_det else "")

    return "\n".join(lines)


def _s4_invalidation_causes(fsm_log: list[dict]) -> str:
    """Why setups die: timeout vs price crash vs insufficient acceptance."""
    lines = [_hdr("FSM-4. Invalidation Causes")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    inv_reasons: dict[str, int] = defaultdict(int)
    inv_from_state: dict[str, int] = defaultdict(int)

    for entry in fsm_log:
        if entry["to_state"] == "INVALIDATED":
            reason = entry.get("reason", "unknown")
            inv_reasons[reason] += 1
            inv_from_state[entry["from_state"]] += 1

    total = sum(inv_reasons.values())
    if total == 0:
        lines.append("  No invalidations recorded")
        return "\n".join(lines)

    lines.append(f"  Total invalidations: {total}")
    lines.append(f"\n  By reason:")
    for reason, count in sorted(inv_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"    {reason:<25s} {count:>5} ({count/total:.0%})")

    lines.append(f"\n  By source state:")
    for state, count in sorted(inv_from_state.items(), key=lambda x: -x[1]):
        lines.append(f"    {state:<25s} {count:>5} ({count/total:.0%})")

    # Acceptance count at invalidation
    inv_with_acc = [e for e in fsm_log if e["to_state"] == "INVALIDATED" and e.get("acceptance_count", 0) > 0]
    if inv_with_acc:
        accs = [e["acceptance_count"] for e in inv_with_acc]
        lines.append(f"\n  Avg acceptance count at invalidation: {float(np.mean(accs)):.1f}")

    return "\n".join(lines)


def _s5_reentry_quality(fsm_log: list[dict], trades: list[TradeRecord]) -> str:
    """Same symbol re-setup quality after cooldown."""
    lines = [_hdr("FSM-5. Re-entry After Invalidation")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    # Find symbols that had invalidation then re-setup on same day
    inv_events: dict[tuple[date, str], int] = defaultdict(int)
    reentry_events: dict[tuple[date, str], int] = defaultdict(int)

    for entry in fsm_log:
        key = (entry["date"], entry["symbol"])
        if entry["to_state"] == "INVALIDATED":
            inv_events[key] += 1
        if entry["from_state"] == "INVALIDATED" and entry["to_state"] == "IDLE":
            reentry_events[key] += 1

    # Match with trades
    trade_map: dict[tuple[date, str], list[TradeRecord]] = defaultdict(list)
    for t in trades:
        d = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_time
        trade_map[(d, t.symbol)].append(t)

    re_entered = 0
    re_entered_profitable = 0
    for key in reentry_events:
        if key in trade_map:
            re_entered += 1
            if any(t.is_winner for t in trade_map[key]):
                re_entered_profitable += 1

    total_inv = sum(inv_events.values())
    total_re = sum(reentry_events.values())
    lines.append(f"  Invalidations: {total_inv}")
    lines.append(f"  Cooldown resets: {total_re}")
    lines.append(f"  Re-entered after cooldown: {re_entered}")
    if re_entered:
        lines.append(f"  Re-entry profitable: {re_entered_profitable}/{re_entered} ({re_entered_profitable/re_entered:.0%})")

    return "\n".join(lines)


def _s6_fsm_velocity(fsm_log: list[dict], trades: list[TradeRecord]) -> str:
    """Does faster ACCEPTING progression predict better R?"""
    lines = [_hdr("FSM-6. FSM Velocity")]
    if not fsm_log or not trades:
        lines.append("  Insufficient data")
        return "\n".join(lines)

    # Measure time from ACCEPTING -> READY_TO_ENTER per (date, symbol)
    accepting_start: dict[tuple, datetime] = {}
    ready_times: dict[tuple, float] = {}

    for entry in fsm_log:
        key = (entry["date"], entry["symbol"])
        if entry["to_state"] == "ACCEPTING":
            accepting_start[key] = entry["timestamp"]
        elif entry["to_state"] == "READY_TO_ENTER" and key in accepting_start:
            dt = (entry["timestamp"] - accepting_start[key]).total_seconds() / 60
            ready_times[key] = dt

    if not ready_times:
        lines.append("  No ACCEPTING -> READY transitions recorded")
        return "\n".join(lines)

    all_times = list(ready_times.values())
    lines.append(f"  ACCEPTING -> READY_TO_ENTER duration:")
    lines.append(f"    Mean: {float(np.mean(all_times)):.1f} min")
    lines.append(f"    Median: {float(np.median(all_times)):.1f} min")
    lines.append(f"    Range: [{min(all_times):.0f}, {max(all_times):.0f}] min")

    # Split trades by fast/slow acceptance
    median_time = float(np.median(all_times))
    fast_keys = {k for k, v in ready_times.items() if v <= median_time}
    slow_keys = {k for k, v in ready_times.items() if v > median_time}

    fast_trades = [t for t in trades if (t.entry_time.date(), t.symbol) in fast_keys]
    slow_trades = [t for t in trades if (t.entry_time.date(), t.symbol) in slow_keys]

    lines.append(f"\n  Fast (≤{median_time:.0f}min):")
    lines.append(f"    n={len(fast_trades)}, " +
                 (f"Mean R={float(np.mean([t.r_multiple for t in fast_trades])):+.3f}" if fast_trades else "no trades"))
    lines.append(f"  Slow (>{median_time:.0f}min):")
    lines.append(f"    n={len(slow_trades)}, " +
                 (f"Mean R={float(np.mean([t.r_multiple for t in slow_trades])):+.3f}" if slow_trades else "no trades"))

    return "\n".join(lines)


def _s7_daily_congestion(fsm_log: list[dict]) -> str:
    """Active FSM instances by time-of-day bucket."""
    lines = [_hdr("FSM-7. Daily FSM Congestion")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    # Count active setups by 30-minute bucket
    buckets: dict[str, list[int]] = defaultdict(list)

    # Group by date, then count active setups at each transition
    by_date: dict[date, list[dict]] = defaultdict(list)
    for entry in fsm_log:
        by_date[entry["date"]].append(entry)

    for d, entries in by_date.items():
        active_count = 0
        for entry in sorted(entries, key=lambda e: e["timestamp"]):
            if entry["to_state"] in ("SETUP_DETECTED", "ACCEPTING", "READY_TO_ENTER"):
                active_count += 1
            elif entry["to_state"] in ("INVALIDATED", "IDLE", "IN_POSITION"):
                active_count = max(0, active_count - 1)

            hour = entry["timestamp"].hour
            minute = entry["timestamp"].minute
            bucket = f"{hour:02d}:{(minute // 30) * 30:02d}"
            buckets[bucket].append(active_count)

    if not buckets:
        lines.append("  No time-bucketed data")
        return "\n".join(lines)

    lines.append(f"  {'Time':>8s} {'Avg Active':>12s} {'Max':>6s} {'Events':>8s}")
    lines.append("  " + "-" * 38)

    for bucket in sorted(buckets):
        counts = buckets[bucket]
        lines.append(
            f"  {bucket:>8s} {float(np.mean(counts)):>12.1f}"
            f" {max(counts):>6} {len(counts):>8}"
        )

    return "\n".join(lines)


def iaric_fsm_analysis(
    fsm_log: list[dict],
    rejection_log: list[dict],
    trades: list[TradeRecord],
) -> str:
    """Generate the full IARIC FSM analysis report."""
    if not fsm_log:
        return "No FSM log data available."

    sections = [
        _s1_transition_matrix(fsm_log),
        _s2_state_dwell_times(fsm_log),
        _s3_funnel_conversion(fsm_log, trades),
        _s4_invalidation_causes(fsm_log),
        _s5_reentry_quality(fsm_log, trades),
        _s6_fsm_velocity(fsm_log, trades),
        _s7_daily_congestion(fsm_log),
    ]
    return "\n".join(sections)
