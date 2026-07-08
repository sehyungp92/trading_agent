"""Quick-exit replacement trade analysis for ALCB diagnostics.

Determines whether trades that entered QE-freed position slots were
profitable.  Reconstructs the position timeline from the trade list and
identifies entries that were only possible because a quick exit freed a
slot while positions were at max capacity.

Each QE-freed slot can enable exactly one replacement trade (FIFO).

Usage:
    from backtests.stock.analysis.alcb_qe_replacement import (
        qe_replacement_analysis,
    )
    report = qe_replacement_analysis(trades, max_positions=8)
    print(report)
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np

from backtests.stock.models import TradeRecord


_ET = ZoneInfo("America/New_York")


def _group_stats(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "mean_r": 0, "median_r": 0, "total_r": 0, "pf": 0}
    r_vals = [t.r_multiple for t in trades]
    gross_w = sum(r for r in r_vals if r > 0)
    gross_l = abs(sum(r for r in r_vals if r < 0))
    return {
        "n": len(trades),
        "wr": sum(1 for r in r_vals if r > 0) / len(trades),
        "mean_r": float(np.mean(r_vals)),
        "median_r": float(np.median(r_vals)),
        "total_r": sum(r_vals),
        "pf": gross_w / gross_l if gross_l > 0 else float("inf"),
    }


def _fmt(s: dict) -> str:
    pf_str = f"{s['pf']:.2f}" if s["pf"] < 100 else "inf"
    return (
        f"n={s['n']:>4d}  WR={s['wr']:.1%}  mean_R={s['mean_r']:+.3f}  "
        f"med_R={s['median_r']:+.3f}  total_R={s['total_r']:+.1f}  PF={pf_str}"
    )


def _time_bucket(dt) -> str:
    """Round to 30-min bucket label."""
    dt_et = dt.astimezone(_ET)
    m = (dt_et.minute // 30) * 30
    return f"{dt_et.hour:02d}:{m:02d}"


def qe_replacement_analysis(
    trades: list[TradeRecord],
    max_positions: int = 8,
    window_minutes: int = 30,
) -> str:
    """Analyze profitability of trades that entered QE-freed slots.

    Algorithm:
      1. Build chronological event list (entries + exits).
      2. Replay events, tracking active position count.
      3. When a QE fires at max capacity, enqueue a freed-slot token.
      4. When an entry occurs within *window_minutes* of a queued token,
         consume the token and tag the trade as QE-enabled.
      5. Report separate P&L for QE-enabled vs organic trades.
    """
    if not trades:
        return "No trades to analyze."

    window = timedelta(minutes=window_minutes)

    # Build events: (time, priority, type, trade_index)
    # priority: 0=exit first, 1=entry second (same-bar ordering)
    events: list[tuple] = []
    for i, t in enumerate(trades):
        events.append((t.entry_time, 1, "entry", i))
        events.append((t.exit_time, 0, "exit", i))
    events.sort()

    active: set[int] = set()
    qe_slot_queue: deque = deque()  # freed_time tokens (FIFO)
    qe_enabled_indices: set[int] = set()

    for _, _, event_type, idx in events:
        if event_type == "exit":
            was_at_max = len(active) >= max_positions
            active.discard(idx)
            if trades[idx].exit_reason == "QUICK_EXIT" and was_at_max:
                qe_slot_queue.append(trades[idx].exit_time)
        else:
            # Expire stale slots
            evt_time = trades[idx].entry_time
            while qe_slot_queue and (evt_time - qe_slot_queue[0]) > window:
                qe_slot_queue.popleft()
            # Consume one QE-freed slot if available
            if qe_slot_queue:
                qe_slot_queue.popleft()
                qe_enabled_indices.add(idx)
            active.add(idx)

    # ── Split trades into categories ──────────────────────────────────
    qe_trades = [trades[i] for i in sorted(qe_enabled_indices)]
    organic_trades = [
        trades[i] for i in range(len(trades)) if i not in qe_enabled_indices
    ]
    qe_exits = [t for t in trades if t.exit_reason == "QUICK_EXIT"]

    # QE-enabled trades broken down by their own exit reason
    qe_then_qe = [t for t in qe_trades if t.exit_reason == "QUICK_EXIT"]
    qe_then_other = [t for t in qe_trades if t.exit_reason != "QUICK_EXIT"]

    # ── Compute stats ─────────────────────────────────────────────────
    all_s = _group_stats(trades)
    qe_en_s = _group_stats(qe_trades)
    org_s = _group_stats(organic_trades)
    qe_ex_s = _group_stats(qe_exits)
    chain_s = _group_stats(qe_then_qe)
    surv_s = _group_stats(qe_then_other)

    lines = [
        "=" * 70,
        "QUICK EXIT REPLACEMENT TRADE ANALYSIS",
        "=" * 70,
        "",
        f"  Max positions: {max_positions}  |  QE window: {window_minutes} min",
        "",
        "  --- Classification ---",
        f"  Total trades:        {all_s['n']}",
        f"  QE exits:            {qe_ex_s['n']} ({qe_ex_s['n']/max(all_s['n'],1):.1%} of all exits)",
        f"  QE-enabled entries:  {qe_en_s['n']} ({qe_en_s['n']/max(all_s['n'],1):.1%} of all entries)",
        f"  Organic entries:     {org_s['n']} ({org_s['n']/max(all_s['n'],1):.1%} of all entries)",
        "",
        "  QE-enabled = entered a slot freed by QE while positions were at max capacity.",
        "",
        "  --- Performance by Category ---",
        f"  All trades:          {_fmt(all_s)}",
        f"  QE exits themselves: {_fmt(qe_ex_s)}",
        f"  QE-enabled entries:  {_fmt(qe_en_s)}",
        f"  Organic entries:     {_fmt(org_s)}",
        "",
        "  --- QE-Enabled Breakdown by Their Own Exit ---",
        f"  exited via QE (churn):  {_fmt(chain_s)}",
        f"  exited via other:       {_fmt(surv_s)}",
    ]

    # ── Churn chains ──────────────────────────────────────────────────
    if chain_s["n"] > 0 and qe_en_s["n"] > 0:
        lines.extend([
            "",
            "  --- QE Churn Chains ---",
            f"  QE-enabled trades that also got QE'd: {chain_s['n']}"
            f" ({chain_s['n']/qe_en_s['n']:.1%} of QE-enabled)",
            "  These create slot-recycling chains (QE -> entry -> QE -> entry ...)",
        ])

    # ── Entry type split ──────────────────────────────────────────────
    qe_et: dict[str, list] = defaultdict(list)
    org_et: dict[str, list] = defaultdict(list)
    for t in qe_trades:
        key = t.entry_type or t.metadata.get("entry_type", "unknown")
        qe_et[key].append(t.r_multiple)
    for t in organic_trades:
        key = t.entry_type or t.metadata.get("entry_type", "unknown")
        org_et[key].append(t.r_multiple)

    all_types = sorted(set(list(qe_et) + list(org_et)))
    lines.extend([
        "",
        "  --- Entry Type Distribution ---",
        f"  {'Type':<25} {'QE-En N':>8} {'Avg R':>8} {'Org N':>8} {'Avg R':>8}",
    ])
    for et in all_types:
        qr = qe_et.get(et, [])
        orr = org_et.get(et, [])
        qa = f"{np.mean(qr):+.3f}" if qr else "     —"
        oa = f"{np.mean(orr):+.3f}" if orr else "     —"
        lines.append(f"  {et:<25} {len(qr):>8d} {qa:>8} {len(orr):>8d} {oa:>8}")

    # ── Time-of-day split ─────────────────────────────────────────────
    qe_tb: dict[str, list] = defaultdict(list)
    org_tb: dict[str, list] = defaultdict(list)
    for t in qe_trades:
        qe_tb[_time_bucket(t.entry_time)].append(t.r_multiple)
    for t in organic_trades:
        org_tb[_time_bucket(t.entry_time)].append(t.r_multiple)

    all_buckets = sorted(set(list(qe_tb) + list(org_tb)))
    lines.extend([
        "",
        "  --- Entry Time Distribution (30-min buckets) ---",
        f"  {'Time':<8} {'QE-En N':>8} {'Avg R':>8} {'Org N':>8} {'Avg R':>8}",
    ])
    for b in all_buckets:
        qr = qe_tb.get(b, [])
        orr = org_tb.get(b, [])
        qa = f"{np.mean(qr):+.3f}" if qr else "     —"
        oa = f"{np.mean(orr):+.3f}" if orr else "     —"
        lines.append(f"  {b:<8} {len(qr):>8d} {qa:>8} {len(orr):>8d} {oa:>8}")

    # ── Verdict ───────────────────────────────────────────────────────
    lines.extend(["", "  --- Verdict ---"])
    if qe_en_s["n"] == 0:
        lines.append("  No QE-enabled trades (QE never fires at max capacity).")
    elif qe_en_s["total_r"] > 0:
        lines.append(
            f"  QE-enabled trades are NET POSITIVE: {qe_en_s['total_r']:+.1f}R"
        )
        lines.append("  Slot recycling is generating alpha.")
    else:
        lines.append(
            f"  QE-enabled trades are NET NEGATIVE: {qe_en_s['total_r']:+.1f}R"
        )
        if surv_s["n"] > 0 and surv_s["total_r"] > 0:
            lines.append(
                f"  But survivors (non-QE exits) are positive: {surv_s['total_r']:+.1f}R"
            )
            lines.append(
                "  The churn (QE->QE chains) is the problem, not the slot reuse."
            )
        else:
            lines.append("  Slot recycling is destroying value.")
            lines.append(
                "  Consider: cooldown after QE, reduce max_positions, or disable QE."
            )

    return "\n".join(lines)
