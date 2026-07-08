"""Loss streak analysis — consecutive-loss distribution and recovery patterns.

Identifies streak patterns, recovery behavior after N consecutive losses,
optimal cooldown periods, and streak-drawdown correlation.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime


def _get_strategy(trade) -> str:
    """Extract strategy name from trade."""
    return getattr(trade, "entry_class", getattr(trade, "strategy", "unknown"))


def _streaks(trades: list) -> list[dict]:
    """Identify all loss/win streaks.

    Returns list of dicts: {type, length, start_idx, end_idx, total_pnl}.
    """
    if not trades:
        return []
    result = []
    pnl = [getattr(t, "pnl_dollars", 0.0) for t in trades]
    current_type = "loss" if pnl[0] <= 0 else "win"
    start = 0
    total = pnl[0]

    for i in range(1, len(pnl)):
        this_type = "loss" if pnl[i] <= 0 else "win"
        if this_type != current_type:
            result.append({
                "type": current_type, "length": i - start,
                "start_idx": start, "end_idx": i - 1,
                "total_pnl": total,
            })
            current_type = this_type
            start = i
            total = pnl[i]
        else:
            total += pnl[i]

    result.append({
        "type": current_type, "length": len(pnl) - start,
        "start_idx": start, "end_idx": len(pnl) - 1,
        "total_pnl": total,
    })
    return result


def generate_loss_streak_report(trades: list, strategies: dict[str, list] | None = None) -> str:
    """Generate loss streak analysis report.

    Args:
        trades: Combined trade list.
        strategies: Optional dict mapping strategy name -> trade list.

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  LOSS STREAK ANALYSIS")
    lines.append("=" * 72)
    lines.append("")

    if strategies is None:
        # Group by entry_class
        grouped: dict[str, list] = defaultdict(list)
        for t in trades:
            grouped[_get_strategy(t)].append(t)
        strategies = dict(grouped)
        strategies["PORTFOLIO"] = trades

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # ── A. Consecutive loss distribution ──
    lines.append("  A. CONSECUTIVE LOSS DISTRIBUTION")
    lines.append("  " + "-" * 55)

    for name, strat_trades in strategies.items():
        streaks = _streaks(strat_trades)
        loss_streaks = [s for s in streaks if s["type"] == "loss"]
        if not loss_streaks:
            lines.append(f"    {name}: No loss streaks")
            continue

        lengths = [s["length"] for s in loss_streaks]
        max_len = max(lengths)
        dist = defaultdict(int)
        for l in lengths:
            dist[l] += 1

        lines.append(f"\n    {name} (max streak: {max_len})")
        lines.append(f"    {'Length':>6s} {'Count':>5s} {'Freq%':>6s} {'Cumul%':>7s}")
        total_streaks = len(loss_streaks)
        cumul = 0
        for l in range(1, max_len + 1):
            cnt = dist[l]
            cumul += cnt
            lines.append(
                f"    {l:>6d} {cnt:>5d} {cnt/total_streaks*100:>5.1f}% {cumul/total_streaks*100:>6.1f}%"
            )

    # ── B. Recovery after N consecutive losses ──
    lines.append("")
    lines.append("  B. RECOVERY AFTER N CONSECUTIVE LOSSES")
    lines.append("  " + "-" * 55)
    lines.append("    (Performance of next 1-5 trades after streak ends)")
    lines.append("")

    for name, strat_trades in strategies.items():
        if len(strat_trades) < 5:
            continue
        streaks = _streaks(strat_trades)
        pnl_list = [getattr(t, "pnl_dollars", 0.0) for t in strat_trades]
        lines.append(f"    {name}:")
        lines.append(f"    {'After N':>8s} {'NextWR%':>7s} {'Next1PnL':>10s} {'Next3PnL':>10s} {'Next5PnL':>10s} {'Samples':>7s}")

        for streak_len in [1, 2, 3, 4, 5]:
            loss_ends = [s["end_idx"] for s in streaks if s["type"] == "loss" and s["length"] >= streak_len]
            if not loss_ends:
                continue

            next1_pnl, next3_pnl, next5_pnl, next_wins = [], [], [], []
            for end_idx in loss_ends:
                nxt = end_idx + 1
                if nxt < len(pnl_list):
                    next1_pnl.append(pnl_list[nxt])
                    next_wins.append(1 if pnl_list[nxt] > 0 else 0)
                if nxt + 3 <= len(pnl_list):
                    next3_pnl.append(sum(pnl_list[nxt:nxt + 3]))
                if nxt + 5 <= len(pnl_list):
                    next5_pnl.append(sum(pnl_list[nxt:nxt + 5]))

            wr = np.mean(next_wins) * 100 if next_wins else 0
            n1 = np.mean(next1_pnl) if next1_pnl else 0
            n3 = np.mean(next3_pnl) if next3_pnl else 0
            n5 = np.mean(next5_pnl) if next5_pnl else 0
            lines.append(
                f"    {streak_len:>8d} {wr:>6.1f}% ${n1:>+9.0f} ${n3:>+9.0f} ${n5:>+9.0f} {len(loss_ends):>7d}"
            )
        lines.append("")

    # ── C. Optimal cooldown ──
    lines.append("  C. OPTIMAL COOLDOWN PERIOD")
    lines.append("  " + "-" * 55)
    lines.append("    Simulates skipping N trades after a loss streak >= 2:")
    lines.append("")

    for name, strat_trades in strategies.items():
        if len(strat_trades) < 10:
            continue
        pnl_list = [getattr(t, "pnl_dollars", 0.0) for t in strat_trades]
        base_total = sum(pnl_list)

        lines.append(f"    {name} (baseline total: ${base_total:+,.0f}):")
        lines.append(f"    {'Skip N':>6s} {'Adjusted PnL':>14s} {'Trades Skipped':>14s} {'Delta':>10s}")

        best_skip, best_delta = 0, 0.0
        for skip in [1, 2, 3, 5]:
            streaks = _streaks(strat_trades)
            skipped_indices = set()
            for s in streaks:
                if s["type"] == "loss" and s["length"] >= 2:
                    for k in range(1, skip + 1):
                        idx = s["end_idx"] + k
                        if idx < len(pnl_list):
                            skipped_indices.add(idx)

            adj_total = sum(p for i, p in enumerate(pnl_list) if i not in skipped_indices)
            delta = adj_total - base_total
            lines.append(
                f"    {skip:>6d} ${adj_total:>+13,.0f} {len(skipped_indices):>14d} ${delta:>+9,.0f}"
            )
            if delta > best_delta:
                best_delta = delta
                best_skip = skip

        if best_delta > 0:
            lines.append(f"    --> Best: skip {best_skip} trades (+${best_delta:,.0f})")
        else:
            lines.append(f"    --> No cooldown improves results")
        lines.append("")

    # ── D. Streak vs drawdown correlation ──
    lines.append("  D. MAX STREAK vs MAX DRAWDOWN")
    lines.append("  " + "-" * 55)

    for name, strat_trades in strategies.items():
        pnl_list = np.array([getattr(t, "pnl_dollars", 0.0) for t in strat_trades])
        if len(pnl_list) < 5:
            continue
        cum = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = float(np.min(dd))
        max_dd_idx = int(np.argmin(dd))

        streaks = _streaks(strat_trades)
        loss_streaks = [s for s in streaks if s["type"] == "loss"]
        max_streak = max((s["length"] for s in loss_streaks), default=0)

        # Was the max DD caused by a streak?
        dd_streak = None
        for s in loss_streaks:
            if s["start_idx"] <= max_dd_idx <= s["end_idx"] + 2:
                dd_streak = s["length"]
                break

        lines.append(
            f"    {name:<12s}  max_DD=${max_dd:+,.0f}  max_streak={max_streak}  "
            f"DD-streak={'yes('+str(dd_streak)+')' if dd_streak else 'no'}"
        )

    return "\n".join(lines)
