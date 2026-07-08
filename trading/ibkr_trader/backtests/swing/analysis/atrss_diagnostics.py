"""Extended diagnostic reports for ATRSS strategy backtests.

All functions accept list[TradeRecord] (duck-typed) and return str.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np


def _trade_net_pnl(trade) -> float:
    return float(trade.pnl_dollars) - float(getattr(trade, "commission", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# 1. Entry type drill-down
# ---------------------------------------------------------------------------

def atrss_entry_type_drilldown(trades: list) -> str:
    """Per-entry-type (PULLBACK/BREAKOUT/REVERSE) table with detailed metrics."""
    if not trades:
        return "No trades for entry type drilldown."

    lines = ["=== ATRSS Entry Type Drilldown ==="]
    header = (
        f"  {'Type':10s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} "
        f"{'MFE':>6s} {'MAE':>6s} {'Hold':>6s} {'Long':>5s} {'Short':>5s} "
        f"{'Exit':>20s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    entry_types = sorted(set(t.entry_type for t in trades))
    for etype in entry_types:
        ct = [t for t in trades if t.entry_type == etype]
        if not ct:
            continue

        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        mfe = np.mean([t.mfe_r for t in ct])
        mae = np.mean([t.mae_r for t in ct])
        hold = np.mean([t.bars_held for t in ct])
        n_long = sum(1 for t in ct if t.direction == 1)
        n_short = count - n_long

        # Most common exit reason for this type
        exit_counts = Counter(t.exit_reason for t in ct)
        top_exit = exit_counts.most_common(1)[0]
        exit_str = f"{top_exit[0]}({top_exit[1]})"

        lines.append(
            f"  {etype:10s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} "
            f"{mfe:6.2f} {mae:6.2f} {hold:6.1f} {n_long:5d} {n_short:5d} "
            f"{exit_str:>20s}"
        )

    # Summary row
    count = len(trades)
    wr = np.mean([t.r_multiple > 0 for t in trades]) * 100
    avg_r = np.mean([t.r_multiple for t in trades])
    pnl = sum(_trade_net_pnl(t) for t in trades)
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(
        f"  {'ALL':10s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f}"
    )

    # Flag worst entry type
    type_avg_r = {}
    for etype in entry_types:
        ct = [t for t in trades if t.entry_type == etype]
        if ct:
            type_avg_r[etype] = np.mean([t.r_multiple for t in ct])
    if type_avg_r:
        worst = min(type_avg_r, key=type_avg_r.get)
        if type_avg_r[worst] < 0:
            lines.append(f"\n  ** {worst} is the primary drag (avg R = {type_avg_r[worst]:+.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Exit reason analysis
# ---------------------------------------------------------------------------

def atrss_exit_analysis(trades: list) -> str:
    """Detailed exit reason breakdown with per-reason metrics."""
    if not trades:
        return "No trades for exit analysis."

    lines = ["=== ATRSS Exit Reason Analysis ==="]

    exit_reasons = sorted(set(t.exit_reason for t in trades))
    lines.append(f"\n  {'Reason':25s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} "
                  f"{'AvgHold':>7s} {'MFE':>6s} {'MAE':>6s}")
    lines.append("  " + "-" * 80)

    for reason in exit_reasons:
        ct = [t for t in trades if t.exit_reason == reason]
        count = len(ct)
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        pnl = sum(_trade_net_pnl(t) for t in ct)
        hold = np.mean([t.bars_held for t in ct])
        mfe = np.mean([t.mfe_r for t in ct])
        mae = np.mean([t.mae_r for t in ct])
        lines.append(
            f"  {reason:25s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} "
            f"{hold:7.1f} {mfe:6.2f} {mae:6.2f}"
        )

    # Cross-tab: exit reason x entry type
    lines.append(f"\n  Exit reason by entry type:")
    lines.append(f"  {'Reason':25s}", )
    entry_types = sorted(set(t.entry_type for t in trades))
    header2 = f"  {'Reason':25s}"
    for etype in entry_types:
        header2 += f" {etype:>12s}"
    lines.append(header2)
    for reason in exit_reasons:
        row = f"  {reason:25s}"
        for etype in entry_types:
            ct = [t for t in trades if t.exit_reason == reason and t.entry_type == etype]
            row += f" {len(ct):12d}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Bias alignment
# ---------------------------------------------------------------------------

def atrss_bias_alignment(trades: list, result=None) -> str:
    """Direction breakdown and bias day distribution comparison."""
    if not trades:
        return "No trades for bias alignment."

    lines = ["=== ATRSS Bias Alignment ==="]

    # Direction x entry type cross-tab
    lines.append(f"\n  {'Dir':6s} {'Type':12s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s} {'P&L':>10s}")
    lines.append("  " + "-" * 52)

    entry_types = sorted(set(t.entry_type for t in trades))
    for direction, dir_label in [(1, "LONG"), (-1, "SHORT")]:
        for etype in entry_types:
            cell = [t for t in trades
                    if t.direction == direction and t.entry_type == etype]
            if not cell:
                lines.append(f"  {dir_label:6s} {etype:12s} {'0':>6s}")
                continue
            avg_r = np.mean([t.r_multiple for t in cell])
            wr = np.mean([t.r_multiple > 0 for t in cell]) * 100
            pnl = sum(_trade_net_pnl(t) for t in cell)
            lines.append(
                f"  {dir_label:6s} {etype:12s} {len(cell):6d} {avg_r:+7.3f} {wr:5.0f}% {pnl:+10,.0f}"
            )

    # Compare bias time distribution vs trade allocation
    if result is not None:
        total_days = (getattr(result, 'bias_days_long', 0)
                      + getattr(result, 'bias_days_short', 0)
                      + getattr(result, 'bias_days_flat', 0))
        if total_days > 0:
            n_long = sum(1 for t in trades if t.direction == 1)
            n_short = sum(1 for t in trades if t.direction == -1)
            long_pct = n_long / len(trades) * 100
            short_pct = n_short / len(trades) * 100

            bias_long_pct = result.bias_days_long / total_days * 100
            bias_short_pct = result.bias_days_short / total_days * 100
            bias_flat_pct = result.bias_days_flat / total_days * 100

            lines.append(f"\n  Bias time vs trade allocation:")
            lines.append(f"  {'Bias':8s} {'Time%':>7s} {'Trade%':>7s} {'Delta':>7s}")
            lines.append(f"  {'LONG':8s} {bias_long_pct:6.1f}% {long_pct:6.1f}% {long_pct - bias_long_pct:+6.1f}%")
            lines.append(f"  {'SHORT':8s} {bias_short_pct:6.1f}% {short_pct:6.1f}% {short_pct - bias_short_pct:+6.1f}%")
            lines.append(f"  {'FLAT':8s} {bias_flat_pct:6.1f}% {'N/A':>6s}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Stop efficiency
# ---------------------------------------------------------------------------

def atrss_stop_efficiency(trades: list) -> str:
    """Stop distance stats, MFE capture ratio, and loser classification."""
    if not trades:
        return "No trades for stop analysis."

    lines = ["=== ATRSS Stop Efficiency ==="]

    # Stop distance stats
    stop_dists = []
    stop_pcts = []
    for t in trades:
        dist = abs(t.entry_price - t.initial_stop)
        stop_dists.append(dist)
        if t.entry_price != 0:
            stop_pcts.append(dist / t.entry_price * 100)

    lines.append(f"\nStop distance (points):")
    lines.append(f"  Mean: {np.mean(stop_dists):.2f}  Median: {np.median(stop_dists):.2f}")
    lines.append(f"  Min:  {np.min(stop_dists):.2f}  Max: {np.max(stop_dists):.2f}")
    if stop_pcts:
        lines.append(f"\nStop distance (% of price):")
        lines.append(f"  Mean: {np.mean(stop_pcts):.2f}%  Median: {np.median(stop_pcts):.2f}%")

    # MFE-to-realized-R capture ratio for winners
    winners = [t for t in trades if t.r_multiple > 0]
    if winners:
        capture_ratios = []
        for t in winners:
            if t.mfe_r > 0:
                capture_ratios.append(t.r_multiple / t.mfe_r)
        if capture_ratios:
            lines.append(f"\nMFE capture ratio (winners, n={len(winners)}):")
            lines.append(f"  Mean:   {np.mean(capture_ratios):.1%}")
            lines.append(f"  Median: {np.median(capture_ratios):.1%}")
            lines.append(f"  Captures < 50% of MFE: {sum(1 for r in capture_ratios if r < 0.5)}"
                          f" ({sum(1 for r in capture_ratios if r < 0.5)/len(capture_ratios)*100:.0f}%)")

    # Loser classification: "right-then-stopped" vs "immediately wrong"
    losers = [t for t in trades if t.r_multiple <= 0]
    if losers:
        right_then_stopped = [t for t in losers if t.mfe_r >= 0.5]
        immediately_wrong = [t for t in losers if t.mfe_r < 0.5]

        lines.append(f"\nLoser classification (n={len(losers)}):")
        lines.append(f"  Right-then-stopped (MFE >= 0.5R): {len(right_then_stopped)}"
                      f" ({len(right_then_stopped)/len(losers)*100:.0f}%)")
        if right_then_stopped:
            lines.append(f"    Avg MFE: {np.mean([t.mfe_r for t in right_then_stopped]):.2f}R"
                          f"  Avg final R: {np.mean([t.r_multiple for t in right_then_stopped]):+.2f}")
        lines.append(f"  Immediately wrong (MFE < 0.5R):   {len(immediately_wrong)}"
                      f" ({len(immediately_wrong)/len(losers)*100:.0f}%)")
        if immediately_wrong:
            lines.append(f"    Avg MFE: {np.mean([t.mfe_r for t in immediately_wrong]):.2f}R"
                          f"  Avg final R: {np.mean([t.r_multiple for t in immediately_wrong]):+.2f}")

        # MAE analysis for losers
        mae_values = [t.mae_r for t in losers]
        lines.append(f"\n  Loser MAE: mean={np.mean(mae_values):.2f}R  "
                      f"median={np.median(mae_values):.2f}R  max={np.max(mae_values):.2f}R")
        stopped_at_1r = sum(1 for t in losers if -1.1 <= t.r_multiple <= -0.9)
        lines.append(f"  Stopped at ~1R: {stopped_at_1r} ({stopped_at_1r/len(losers)*100:.0f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Time analysis
# ---------------------------------------------------------------------------

def atrss_time_analysis(trades: list) -> str:
    """Entry time patterns: hour, day-of-week, month, year."""
    if not trades:
        return "No trades for time analysis."

    lines = ["=== ATRSS Time Analysis ==="]

    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")

    # By hour of day (ET)
    hour_stats: dict[int, list] = defaultdict(list)
    for t in trades:
        if t.entry_time is not None:
            entry_et = t.entry_time.astimezone(et) if t.entry_time.tzinfo else t.entry_time
            hour_stats[entry_et.hour].append(t.r_multiple)

    if hour_stats:
        lines.append(f"\nEntries by hour (ET):")
        lines.append(f"  {'Hour':>4s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}")
        for hour in sorted(hour_stats):
            rs = hour_stats[hour]
            avg_r = np.mean(rs)
            wr = np.mean([r > 0 for r in rs]) * 100
            lines.append(f"  {hour:4d} {len(rs):6d} {avg_r:+7.3f} {wr:5.0f}%")

    # By day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_stats: dict[int, list] = defaultdict(list)
    for t in trades:
        if t.entry_time is not None:
            entry_et = t.entry_time.astimezone(et) if t.entry_time.tzinfo else t.entry_time
            dow_stats[entry_et.weekday()].append(t.r_multiple)

    if dow_stats:
        lines.append(f"\nEntries by day of week:")
        lines.append(f"  {'Day':>4s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}")
        for dow in sorted(dow_stats):
            rs = dow_stats[dow]
            avg_r = np.mean(rs)
            wr = np.mean([r > 0 for r in rs]) * 100
            lines.append(f"  {dow_names[dow]:>4s} {len(rs):6d} {avg_r:+7.3f} {wr:5.0f}%")

    # By month
    month_stats: dict[int, list] = defaultdict(list)
    for t in trades:
        if t.entry_time is not None:
            month_stats[t.entry_time.month].append(t.r_multiple)

    if month_stats:
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        lines.append(f"\nEntries by month:")
        lines.append(f"  {'Mon':>4s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}")
        for mon in sorted(month_stats):
            rs = month_stats[mon]
            avg_r = np.mean(rs)
            wr = np.mean([r > 0 for r in rs]) * 100
            lines.append(f"  {month_names[mon]:>4s} {len(rs):6d} {avg_r:+7.3f} {wr:5.0f}%")

    # By year
    year_stats: dict[int, list] = defaultdict(list)
    for t in trades:
        if t.entry_time is not None:
            year_stats[t.entry_time.year].append(t.r_multiple)

    if year_stats:
        lines.append(f"\nEntries by year:")
        lines.append(f"  {'Year':>4s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s} {'TotalR':>8s}")
        for year in sorted(year_stats):
            rs = year_stats[year]
            avg_r = np.mean(rs)
            wr = np.mean([r > 0 for r in rs]) * 100
            total_r = sum(rs)
            lines.append(f"  {year:4d} {len(rs):6d} {avg_r:+7.3f} {wr:5.0f}% {total_r:+8.2f}")

    # Flag worst time slots
    worst_items = []
    for label, stats_dict, name_fn in [
        ("hour", hour_stats, lambda k: f"{k}:00 ET"),
        ("day", dow_stats, lambda k: dow_names[k]),
    ]:
        for key, rs in stats_dict.items():
            if len(rs) >= 3 and np.mean(rs) < -0.3:
                worst_items.append((name_fn(key), len(rs), np.mean(rs)))

    if worst_items:
        lines.append(f"\n  ** Systematic loss patterns:")
        for name, count, avg_r in worst_items:
            lines.append(f"     {name}: {count} trades, avg R = {avg_r:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Losing trade detail
# ---------------------------------------------------------------------------

def atrss_losing_trade_detail(trades: list) -> str:
    """Detailed table of every losing trade, sorted worst-first."""
    losers = [t for t in trades if t.r_multiple <= 0]
    if not losers:
        return "=== ATRSS Losing Trade Detail ===\nNo losing trades."

    losers.sort(key=lambda t: t.r_multiple)

    lines = ["=== ATRSS Losing Trade Detail ==="]
    lines.append(f"\nTotal losers: {len(losers)} / {len(trades)}"
                  f" ({len(losers)/len(trades)*100:.0f}%)")
    lines.append(f"Total loss: {sum(_trade_net_pnl(t) for t in losers):+,.0f}")

    lines.append(f"\n  {'#':>3s} {'Entry':19s} {'Exit':19s} {'Type':10s} {'Dir':5s} "
                  f"{'StopDist':>8s} {'R':>7s} {'MFE':>6s} {'MAE':>6s} "
                  f"{'Reason':25s} {'Bars':>4s}")
    lines.append("  " + "-" * 115)

    for i, t in enumerate(losers, 1):
        entry_str = t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "N/A"
        exit_str = t.exit_time.strftime("%Y-%m-%d %H:%M") if t.exit_time else "N/A"
        dir_label = "LONG" if t.direction == 1 else "SHORT"
        stop_dist = abs(t.entry_price - t.initial_stop)
        lines.append(
            f"  {i:3d} {entry_str:19s} {exit_str:19s} {t.entry_type:10s} {dir_label:5s} "
            f"{stop_dist:8.2f} {t.r_multiple:+7.3f} {t.mfe_r:6.2f} {t.mae_r:6.2f} "
            f"{t.exit_reason:25s} {t.bars_held:4d}"
        )

    # Loss clustering detection
    if len(losers) >= 3:
        lines.append(f"\nLoss clustering:")
        loss_dates = []
        for t in losers:
            if t.entry_time is not None:
                loss_dates.append(t.entry_time)
        if loss_dates:
            loss_dates.sort()
            clusters = []
            current_cluster = [loss_dates[0]]
            for dt in loss_dates[1:]:
                if (dt - current_cluster[-1]) <= timedelta(hours=48):
                    current_cluster.append(dt)
                else:
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [dt]
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)

            if clusters:
                for cl in clusters:
                    lines.append(
                        f"  Cluster: {len(cl)} losses between "
                        f"{cl[0].strftime('%Y-%m-%d')} and {cl[-1].strftime('%Y-%m-%d')}"
                    )
            else:
                lines.append("  No significant loss clusters detected (3+ losses within 48h)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Cumulative R curve
# ---------------------------------------------------------------------------

def atrss_r_curve(trades: list) -> str:
    """Cumulative R over time with drawdown analysis."""
    if not trades:
        return "No trades for R curve."

    sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)
    rs = [t.r_multiple for t in sorted_trades]
    cum_r = np.cumsum(rs)

    lines = ["=== ATRSS Cumulative R Curve ==="]

    lines.append(f"\nTotal R: {cum_r[-1]:+.2f} over {len(trades)} trades")
    lines.append(f"Peak R:  {np.max(cum_r):+.2f}")

    # Drawdown from peak
    running_max = np.maximum.accumulate(cum_r)
    drawdowns = cum_r - running_max
    max_dd = np.min(drawdowns)
    max_dd_idx = np.argmin(drawdowns)

    lines.append(f"Max R drawdown: {max_dd:+.2f} (after trade #{max_dd_idx + 1})")

    # Longest drawdown (in # of trades)
    in_dd = drawdowns < 0
    if in_dd.any():
        longest = 0
        current = 0
        for v in in_dd:
            if v:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        lines.append(f"Longest drawdown: {longest} trades")

    # Month-by-month R breakdown
    month_r: dict[str, float] = defaultdict(float)
    month_count: dict[str, int] = defaultdict(int)
    for t in sorted_trades:
        if t.exit_time is not None:
            key = t.exit_time.strftime("%Y-%m")
            month_r[key] += t.r_multiple
            month_count[key] += 1

    if month_r:
        lines.append(f"\nMonth-by-month R:")
        lines.append(f"  {'Month':>7s} {'Trades':>6s} {'R':>8s} {'CumR':>8s}")
        cum = 0.0
        for month in sorted(month_r):
            r = month_r[month]
            cum += r
            lines.append(f"  {month:>7s} {month_count[month]:6d} {r:+8.2f} {cum:+8.2f}")

        monthly_rs = list(month_r.values())
        pos_months = sum(1 for r in monthly_rs if r > 0)
        lines.append(f"\n  Positive months: {pos_months}/{len(monthly_rs)}"
                      f" ({pos_months/len(monthly_rs)*100:.0f}%)")
        lines.append(f"  Best month:  {max(monthly_rs):+.2f}R")
        lines.append(f"  Worst month: {min(monthly_rs):+.2f}R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Streak analysis
# ---------------------------------------------------------------------------

def atrss_streak_analysis(trades: list) -> str:
    """Win/loss streak analysis with recovery patterns."""
    if not trades:
        return "No trades for streak analysis."

    sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)
    outcomes = [1 if t.r_multiple > 0 else 0 for t in sorted_trades]

    lines = ["=== ATRSS Streak Analysis ==="]

    # Compute streaks
    win_streaks = []
    loss_streaks = []
    current_type = outcomes[0]
    current_len = 1

    for o in outcomes[1:]:
        if o == current_type:
            current_len += 1
        else:
            if current_type == 1:
                win_streaks.append(current_len)
            else:
                loss_streaks.append(current_len)
            current_type = o
            current_len = 1
    if current_type == 1:
        win_streaks.append(current_len)
    else:
        loss_streaks.append(current_len)

    lines.append(f"\nWin streaks:")
    if win_streaks:
        lines.append(f"  Max: {max(win_streaks)}  Avg: {np.mean(win_streaks):.1f}  Count: {len(win_streaks)}")
        streak_dist = Counter(win_streaks)
        lines.append(f"  Distribution: {dict(sorted(streak_dist.items()))}")
    else:
        lines.append("  No win streaks")

    lines.append(f"\nLoss streaks:")
    if loss_streaks:
        lines.append(f"  Max: {max(loss_streaks)}  Avg: {np.mean(loss_streaks):.1f}  Count: {len(loss_streaks)}")
        streak_dist = Counter(loss_streaks)
        lines.append(f"  Distribution: {dict(sorted(streak_dist.items()))}")
    else:
        lines.append("  No loss streaks")

    # Post-loss-streak recovery
    if loss_streaks and max(loss_streaks) >= 3:
        lines.append(f"\nPost-loss-streak recovery (after 3+ consecutive losses):")
        recovery_rs = []
        streak_count = 0
        for i, o in enumerate(outcomes):
            if o == 0:
                streak_count += 1
            else:
                if streak_count >= 3:
                    recovery_rs.append(sorted_trades[i].r_multiple)
                streak_count = 0

        if recovery_rs:
            lines.append(f"  Recovery trades: {len(recovery_rs)}")
            lines.append(f"  Win rate after 3+ loss streak: {np.mean([r > 0 for r in recovery_rs])*100:.0f}%")
            lines.append(f"  Avg R of recovery trade: {np.mean(recovery_rs):+.3f}")
        else:
            lines.append("  No recovery trades found after 3+ loss streaks")

    # Overall sequence stats
    total_w = sum(outcomes)
    total_l = len(outcomes) - total_w
    lines.append(f"\nSequence summary:")
    lines.append(f"  Wins: {total_w}  Losses: {total_l}")
    if len(outcomes) >= 10:
        transitions = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i-1])
        expected_transitions = 2 * total_w * total_l / len(outcomes)
        lines.append(f"  Transitions: {transitions} (expected ~{expected_transitions:.0f} if random)")
        if transitions < expected_transitions * 0.7:
            lines.append("  ** Outcomes are clustered: streaks are longer than random")
        elif transitions > expected_transitions * 1.3:
            lines.append("  ** Outcomes alternate more than random: possible mean reversion")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Add-on effectiveness
# ---------------------------------------------------------------------------

def atrss_addon_analysis(trades: list) -> str:
    """Analysis of add-on A/B effectiveness."""
    if not trades:
        return "No trades for add-on analysis."

    lines = ["=== ATRSS Add-on Analysis ==="]

    addon_a = [t for t in trades if t.addon_a_qty > 0]
    addon_b = [t for t in trades if t.addon_b_qty > 0]
    no_addon = [t for t in trades if t.addon_a_qty == 0 and t.addon_b_qty == 0]

    lines.append(f"\n  {'Group':15s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} {'AvgHold':>7s}")
    lines.append("  " + "-" * 55)

    for label, group in [("No add-on", no_addon), ("Add-on A", addon_a), ("Add-on B", addon_b)]:
        if not group:
            lines.append(f"  {label:15s} {'0':>6s}")
            continue
        wr = np.mean([t.r_multiple > 0 for t in group]) * 100
        avg_r = np.mean([t.r_multiple for t in group])
        pnl = sum(_trade_net_pnl(t) for t in group)
        hold = np.mean([t.bars_held for t in group])
        lines.append(f"  {label:15s} {len(group):6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} {hold:7.1f}")

    # Both add-ons triggered
    both = [t for t in trades if t.addon_a_qty > 0 and t.addon_b_qty > 0]
    if both:
        wr = np.mean([t.r_multiple > 0 for t in both]) * 100
        avg_r = np.mean([t.r_multiple for t in both])
        pnl = sum(_trade_net_pnl(t) for t in both)
        lines.append(f"  {'Both A+B':15s} {len(both):6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f}")

    # Add-on triggered vs not: is it a signal of trade quality?
    with_addon = [t for t in trades if t.addon_a_qty > 0 or t.addon_b_qty > 0]
    if with_addon and no_addon:
        lines.append(f"\n  Add-on as quality signal:")
        lines.append(f"    With any add-on: avg R = {np.mean([t.r_multiple for t in with_addon]):+.3f},"
                      f" WR = {np.mean([t.r_multiple > 0 for t in with_addon])*100:.0f}%")
        lines.append(f"    Without add-on:  avg R = {np.mean([t.r_multiple for t in no_addon]):+.3f},"
                      f" WR = {np.mean([t.r_multiple > 0 for t in no_addon])*100:.0f}%")
        delta_r = np.mean([t.r_multiple for t in with_addon]) - np.mean([t.r_multiple for t in no_addon])
        if delta_r > 0.2:
            lines.append(f"    ** Add-on trades outperform by {delta_r:+.3f}R -- add-ons confirm good entries")
        elif delta_r < -0.2:
            lines.append(f"    ** Add-on trades underperform by {delta_r:+.3f}R -- investigate add-on logic")

    # Per-leg breakdown by leg_type (BASE vs ADDON_A vs ADDON_B)
    if hasattr(trades[0], "leg_type"):
        leg_types = sorted(set(t.leg_type for t in trades))
        if len(leg_types) > 1:
            lines.append(f"\n  Per-leg breakdown:")
            lines.append(f"  {'Leg Type':15s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} {'AvgMFE':>7s}")
            lines.append("  " + "-" * 55)
            for lt in leg_types:
                lt_trades = [t for t in trades if t.leg_type == lt]
                if not lt_trades:
                    continue
                wr = np.mean([t.r_multiple > 0 for t in lt_trades]) * 100
                avg_r = np.mean([t.r_multiple for t in lt_trades])
                pnl = sum(_trade_net_pnl(t) for t in lt_trades)
                avg_mfe = np.mean([t.mfe_r for t in lt_trades])
                lines.append(f"  {lt:15s} {len(lt_trades):6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} {avg_mfe:7.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Signal funnel
# ---------------------------------------------------------------------------

def atrss_signal_funnel(funnel, n_trades: int, shadow_rejections: int = 0) -> str:
    """ASCII funnel: total bars down to filled trades with conservation check."""
    if funnel is None:
        return "=== ATRSS Signal Funnel ===\nNo funnel data available."

    f = funnel
    lines = ["=== ATRSS Signal Funnel ==="]

    eligible = (f.total_bars - f.bars_nan - f.bars_warmup
                - f.bars_in_position - f.bars_entry_restricted)
    non_flat = f.bars_regime_range + f.bars_regime_trend + f.bars_regime_strong
    total_signals = f.pullback_signals + f.breakout_signals + f.reverse_signals
    total_rejections = f.rejected_momentum + f.rejected_reentry + f.rejected_sizing

    def _bar(label, count, total, indent=2):
        pct = count / total * 100 if total > 0 else 0
        return f"{' ' * indent}{label:.<40s} {count:>7,d}  ({pct:5.1f}%)"

    lines.append(f"\n  Total hourly bars processed:            {f.total_bars:>7,d}")
    lines.append("")
    lines.append("  --- Filtered out (no signal possible) ---")
    lines.append(_bar("NaN / gap-filled bars", f.bars_nan, f.total_bars))
    lines.append(_bar("Warmup (indicators not ready)", f.bars_warmup, f.total_bars))
    lines.append(_bar("Already in position", f.bars_in_position, f.total_bars))
    lines.append(_bar("Entry time restricted", f.bars_entry_restricted, f.total_bars))
    lines.append("")
    lines.append(f"  Eligible bars (flat, in-window):        {eligible:>7,d}")
    lines.append("")
    lines.append("  --- Eligible bar bias/regime breakdown ---")
    lines.append(_bar("Bias = FLAT (no direction)", f.bars_bias_flat, eligible))
    lines.append(_bar("Regime = RANGE", f.bars_regime_range, eligible))
    lines.append(_bar("Regime = TREND", f.bars_regime_trend, eligible))
    lines.append(_bar("Regime = STRONG_TREND", f.bars_regime_strong, eligible))
    lines.append("")
    lines.append(f"  Bars with confirmed bias (non-FLAT):    {non_flat:>7,d}")
    if getattr(f, 'bars_shorts_disabled', 0) > 0:
        lines.append(f"  (of which shorts disabled).........  {f.bars_shorts_disabled:>6}")
    lines.append("")
    lines.append("  --- Signals generated ---")
    lines.append(_bar("Pullback signals", f.pullback_signals, non_flat))
    lines.append(_bar("Breakout signals", f.breakout_signals, non_flat))
    lines.append(_bar("Reverse signals", f.reverse_signals, non_flat))
    lines.append(f"  Total signals:                          {total_signals:>7,d}")
    lines.append("")
    lines.append("  --- Post-signal rejections ---")
    lines.append(_bar("Momentum filter", f.rejected_momentum, total_signals if total_signals > 0 else 1))
    lines.append(_bar("Re-entry cooldown", f.rejected_reentry, total_signals if total_signals > 0 else 1))
    rejected_quality = getattr(f, 'rejected_quality', 0)
    if rejected_quality > 0:
        lines.append(_bar("Quality gate", rejected_quality, total_signals if total_signals > 0 else 1))
    lines.append(_bar("Sizing (qty=0)", f.rejected_sizing, total_signals if total_signals > 0 else 1))
    if shadow_rejections > 0:
        lines.append(_bar("Shadow rejections (post-signal)", shadow_rejections, total_signals if total_signals > 0 else 1))
    lines.append("")
    lines.append("  --- Orders ---")
    lines.append(f"  Orders submitted:                       {f.orders_submitted:>7,d}")
    lines.append(f"  Orders filled:                          {f.orders_filled:>7,d}")
    lines.append(f"  Orders expired:                         {f.orders_expired:>7,d}")
    lines.append(f"  Orders limit-rejected:                  {f.orders_limit_rejected:>7,d}")
    lines.append(f"  Completed trades:                       {n_trades:>7,d}")

    # Conservation check
    lines.append("")
    lines.append("  --- Conservation check ---")
    filter_sum = (f.bars_nan + f.bars_warmup + f.bars_in_position
                  + f.bars_entry_restricted + f.bars_bias_flat
                  + f.bars_regime_range + f.bars_regime_trend + f.bars_regime_strong)
    if filter_sum == f.total_bars:
        lines.append(f"  OK: filter categories sum to total_bars ({f.total_bars:,d})")
    else:
        delta = f.total_bars - filter_sum
        lines.append(f"  MISMATCH: total_bars={f.total_bars:,d}  sum={filter_sum:,d}  delta={delta:,d}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Regime & time report
# ---------------------------------------------------------------------------

def atrss_regime_time_report(funnel, result=None) -> str:
    """Regime distribution with bias day analysis and trade suppressor flags."""
    if funnel is None:
        return "=== ATRSS Regime & Time Report ===\nNo funnel data available."

    f = funnel
    lines = ["=== ATRSS Regime & Time Report ==="]

    eligible = (f.total_bars - f.bars_nan - f.bars_warmup
                - f.bars_in_position - f.bars_entry_restricted)
    non_flat = f.bars_regime_range + f.bars_regime_trend + f.bars_regime_strong

    lines.append(f"\n  Eligible bars: {eligible:,d}")
    lines.append("")

    # Regime distribution
    lines.append("  Regime distribution (of eligible bars):")
    lines.append(f"  {'Regime':.<25s} {'Bars':>7s} {'Pct':>7s}  Allows")
    lines.append("  " + "-" * 65)
    for label, count, allows in [
        ("FLAT bias", f.bars_bias_flat, "nothing (no confirmed direction)"),
        ("RANGE", f.bars_regime_range, "breakout only (if armed)"),
        ("TREND", f.bars_regime_trend, "pullback + breakout"),
        ("STRONG_TREND", f.bars_regime_strong, "pullback + breakout"),
    ]:
        pct = count / eligible * 100 if eligible > 0 else 0
        lines.append(f"  {label:.<25s} {count:>7,d} {pct:6.1f}%  {allows}")

    # Bias day distribution
    if result is not None:
        total_days = (getattr(result, 'bias_days_long', 0)
                      + getattr(result, 'bias_days_short', 0)
                      + getattr(result, 'bias_days_flat', 0))
        if total_days > 0:
            lines.append(f"\n  Daily bias distribution ({total_days:,d} trading days):")
            for label, count in [
                ("LONG", result.bias_days_long),
                ("SHORT", result.bias_days_short),
                ("FLAT", result.bias_days_flat),
            ]:
                pct = count / total_days * 100
                lines.append(f"    {label:8s} {count:>5d} days ({pct:5.1f}%)")

    # Trade suppressor flags
    lines.append("")
    flags = []
    if eligible > 0:
        flat_pct = f.bars_bias_flat / eligible * 100
        range_pct = f.bars_regime_range / eligible * 100 if eligible > 0 else 0
        if flat_pct > 40:
            flags.append(f"  ** FLAT bias on {flat_pct:.0f}% of eligible bars -- primary trade suppressor")
        if range_pct > 50:
            flags.append(f"  ** RANGE regime on {range_pct:.0f}% of eligible bars -- limits to breakout entries only")
        restricted_pct = f.bars_entry_restricted / f.total_bars * 100 if f.total_bars > 0 else 0
        if restricted_pct > 20:
            flags.append(f"  ** Entry time restriction filtering {restricted_pct:.0f}% of all bars")
        position_pct = f.bars_in_position / f.total_bars * 100 if f.total_bars > 0 else 0
        if position_pct < 5:
            flags.append(f"  ** Position occupancy only {position_pct:.1f}% -- very few trades held")

    if flags:
        lines.append("  Trade suppressor analysis:")
        lines.extend(flags)
    else:
        lines.append("  No major trade suppressors detected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. Position occupancy
# ---------------------------------------------------------------------------

def atrss_position_occupancy(trades: list, funnel=None) -> str:
    """Position occupancy, hold time stats, and trade frequency analysis."""
    lines = ["=== ATRSS Position Occupancy ==="]

    if funnel is not None:
        f = funnel
        active_bars = f.total_bars - f.bars_nan - f.bars_warmup
        if active_bars > 0:
            occ_pct = f.bars_in_position / active_bars * 100
            lines.append(f"\n  Bars in position:   {f.bars_in_position:>7,d} / {active_bars:,d} ({occ_pct:.1f}%)")
        else:
            lines.append("\n  No active bars.")

    if not trades:
        lines.append("  No trades for occupancy analysis.")
        return "\n".join(lines)

    # Hold time stats
    holds = [t.bars_held for t in trades]
    lines.append(f"\n  Hold time (bars):")
    lines.append(f"    Mean:   {np.mean(holds):6.1f}")
    lines.append(f"    Median: {np.median(holds):6.1f}")
    lines.append(f"    Max:    {max(holds):6d}")
    lines.append(f"    Min:    {min(holds):6d}")

    # Gap between trades
    sorted_trades = sorted(trades, key=lambda t: t.entry_time or datetime.min)
    if len(sorted_trades) >= 2:
        gaps_hours = []
        for i in range(1, len(sorted_trades)):
            prev_exit = sorted_trades[i - 1].exit_time
            cur_entry = sorted_trades[i].entry_time
            if prev_exit and cur_entry:
                gap = (cur_entry - prev_exit).total_seconds() / 3600
                gaps_hours.append(gap)

        if gaps_hours:
            lines.append(f"\n  Gap between trades (hours):")
            lines.append(f"    Mean:   {np.mean(gaps_hours):8.1f}")
            lines.append(f"    Median: {np.median(gaps_hours):8.1f}")
            lines.append(f"    Max:    {max(gaps_hours):8.1f}")

            # Longest dry spell
            max_gap_idx = int(np.argmax(gaps_hours))
            if max_gap_idx + 1 < len(sorted_trades):
                t_after = sorted_trades[max_gap_idx + 1]
                t_before = sorted_trades[max_gap_idx]
                start_str = t_before.exit_time.strftime("%Y-%m-%d") if t_before.exit_time else "?"
                end_str = t_after.entry_time.strftime("%Y-%m-%d") if t_after.entry_time else "?"
                lines.append(f"    Longest dry spell: {start_str} to {end_str} ({max(gaps_hours):.0f}h)")

    # Trade frequency
    if len(sorted_trades) >= 2:
        first_entry = sorted_trades[0].entry_time
        last_entry = sorted_trades[-1].entry_time
        if first_entry and last_entry:
            span_days = (last_entry - first_entry).total_seconds() / 86400
            if span_days > 0:
                trades_per_month = len(sorted_trades) / (span_days / 30.44)
                trades_per_week = len(sorted_trades) / (span_days / 7)
                lines.append(f"\n  Trade frequency:")
                lines.append(f"    {len(sorted_trades)} trades over {span_days:.0f} days")
                lines.append(f"    {trades_per_month:.1f} trades/month")
                lines.append(f"    {trades_per_week:.2f} trades/week")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 13. Filter rejection detail
# ---------------------------------------------------------------------------

def atrss_filter_rejection_detail(filter_summary: dict) -> str:
    """Enhanced filter effectiveness report with verdict column."""
    if not filter_summary:
        return "=== ATRSS Filter Rejection Detail ===\nNo shadow/filter data available."

    lines = ["=== ATRSS Filter Rejection Detail ==="]

    lines.append(f"\n  {'Filter':25s} {'Rejected':>8s} {'Filled':>7s} {'AvgR':>7s} "
                  f"{'> 1R':>5s} {'> 2R':>5s} {'Verdict':12s}")
    lines.append("  " + "-" * 80)

    commentary = []
    for name, stats in sorted(filter_summary.items(), key=lambda x: x[1].rejected_count, reverse=True):
        rejected = stats.rejected_count
        filled = stats.filled_count
        avg_r = stats.avg_shadow_r
        pct_1r = stats.pct_above_1r
        pct_2r = stats.pct_above_2r

        # Verdict: classify filter as PROTECTIVE, NEUTRAL, or RESTRICTIVE
        if avg_r < -0.2:
            verdict = "PROTECTIVE"
        elif avg_r > 0.3:
            verdict = "RESTRICTIVE"
        else:
            verdict = "NEUTRAL"

        lines.append(
            f"  {name:25s} {rejected:8d} {filled:7d} {avg_r:+7.3f} "
            f"{pct_1r:4.0f}% {pct_2r:4.0f}% {verdict:12s}"
        )

        if verdict == "RESTRICTIVE" and rejected >= 3:
            commentary.append(
                f"  ** {name}: blocking {rejected} entries with avg shadow R = {avg_r:+.3f} "
                f"-- consider relaxing this filter"
            )

    if commentary:
        lines.append("")
        lines.append("  Over-restrictive filters (blocking profitable signals):")
        lines.extend(commentary)

    # Summary
    total_rejected = sum(s.rejected_count for s in filter_summary.values())
    avg_all_r = np.mean([s.avg_shadow_r for s in filter_summary.values()]) if filter_summary else 0
    lines.append(f"\n  Total shadow rejections: {total_rejected}")
    lines.append(f"  Mean shadow R across all filters: {avg_all_r:+.3f}")
    if avg_all_r > 0.2:
        lines.append("  ** Filters are net-restrictive: blocking more profitable than unprofitable signals")
    elif avg_all_r < -0.2:
        lines.append("  Filters are net-protective: blocking mostly unprofitable signals")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 14. MFE cohort segmentation
# ---------------------------------------------------------------------------

def atrss_mfe_cohort_segmentation(trades: list) -> str:
    """Split trades into developed vs undeveloped by MFE, compare entry features."""
    if not trades:
        return "=== ATRSS MFE Cohort Segmentation ===\nNo trades."

    developed = [t for t in trades if t.mfe_r >= 1.0]
    undeveloped = [t for t in trades if t.mfe_r < 1.0]

    lines = ["=== ATRSS MFE Cohort Segmentation ==="]
    lines.append(f"\n  {'Cohort':15s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'AvgMFE':>7s} {'AvgHold':>7s}")
    lines.append("  " + "-" * 55)

    for label, group in [("Developed", developed), ("Undeveloped", undeveloped)]:
        if not group:
            lines.append(f"  {label:15s} {'0':>6s}")
            continue
        wr = np.mean([t.r_multiple > 0 for t in group]) * 100
        avg_r = np.mean([t.r_multiple for t in group])
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_hold = np.mean([t.bars_held for t in group])
        lines.append(
            f"  {label:15s} {len(group):6d} {wr:5.0f}% {avg_r:+7.3f} {avg_mfe:7.2f} {avg_hold:7.1f}"
        )

    # Entry-time feature comparison
    lines.append(f"\n  Entry feature comparison:")
    lines.append(f"  {'Feature':20s} {'Developed':>10s} {'Undeveloped':>12s} {'Delta':>8s}")
    lines.append("  " + "-" * 55)

    for label, attr, prec in [
        ("ADX", "adx_entry", 1),
        ("Score", "score_entry", 1),
        ("Touch dist (ATR)", "touch_distance_atr", 3),
        ("Quality score", "quality_score", 2),
    ]:
        dev_val = np.mean([getattr(t, attr, 0.0) for t in developed]) if developed else 0
        und_val = np.mean([getattr(t, attr, 0.0) for t in undeveloped]) if undeveloped else 0
        lines.append(
            f"  {label:20s} {dev_val:10.{prec}f} {und_val:12.{prec}f} {dev_val - und_val:+8.{prec}f}"
        )
    # DI agreement (percentage)
    dev_di = np.mean([getattr(t, "di_agrees", False) for t in developed]) if developed else 0
    und_di = np.mean([getattr(t, "di_agrees", False) for t in undeveloped]) if undeveloped else 0
    lines.append(
        f"  {'DI agrees %':20s} {dev_di:10.0%} {und_di:12.0%} {dev_di - und_di:+8.0%}"
    )

    # Regime distribution per cohort
    lines.append(f"\n  Regime distribution:")
    all_regimes = sorted(set(
        getattr(t, "regime_entry", "") for t in trades if getattr(t, "regime_entry", "")
    ))
    if all_regimes:
        lines.append(f"  {'Regime':15s} {'Developed':>10s} {'Undeveloped':>12s}")
        for regime in all_regimes:
            dev_n = sum(1 for t in developed if getattr(t, "regime_entry", "") == regime)
            und_n = sum(1 for t in undeveloped if getattr(t, "regime_entry", "") == regime)
            dev_pct = dev_n / len(developed) * 100 if developed else 0
            und_pct = und_n / len(undeveloped) * 100 if undeveloped else 0
            lines.append(f"  {regime:15s} {dev_pct:9.0f}% {und_pct:11.0f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 15. Breakout arm diagnostic
# ---------------------------------------------------------------------------

def atrss_breakout_arm_diagnostic(funnel) -> str:
    """Breakout arm lifecycle: created, expired, converted."""
    if funnel is None:
        return "=== ATRSS Breakout Arm Diagnostic ===\nNo funnel data."

    created = getattr(funnel, "breakout_arms_created", 0)
    expired = getattr(funnel, "breakout_arms_expired", 0)
    converted = getattr(funnel, "breakout_arms_converted", 0)

    lines = ["=== ATRSS Breakout Arm Diagnostic ==="]
    lines.append(f"\n  Arms created:   {created:6d}")
    lines.append(f"  Arms expired:   {expired:6d}")
    lines.append(f"  Arms converted: {converted:6d}")
    if created > 0:
        conv_rate = converted / created * 100
        exp_rate = expired / created * 100
        lines.append(f"\n  Conversion rate: {conv_rate:.1f}%")
        lines.append(f"  Expiry rate:     {exp_rate:.1f}%")
    else:
        lines.append("\n  No breakout arms created during backtest.")

    if created > 0 and converted == 0:
        lines.append("  ** Zero conversions: breakout pullback condition may be too strict")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 16. Order fill rate diagnostic
# ---------------------------------------------------------------------------

def atrss_order_fill_rate(order_metadata: list[dict]) -> str:
    """Fill/expire/reject analysis with trigger distance breakdown."""
    if not order_metadata:
        return "=== ATRSS Order Fill Rate ===\nNo order metadata available."

    lines = ["=== ATRSS Order Fill Rate ==="]

    filled = [o for o in order_metadata if o.get("status") == "FILLED"]
    expired = [o for o in order_metadata if o.get("status") == "EXPIRED"]
    rejected = [o for o in order_metadata if o.get("status") == "REJECTED"]
    pending = [o for o in order_metadata if o.get("status") not in ("FILLED", "EXPIRED", "REJECTED")]

    total = len(order_metadata)
    lines.append(f"\n  Total orders:    {total:6d}")
    lines.append(f"  Filled:          {len(filled):6d}  ({len(filled)/total*100:5.1f}%)" if total > 0 else "")
    lines.append(f"  Expired:         {len(expired):6d}  ({len(expired)/total*100:5.1f}%)" if total > 0 else "")
    lines.append(f"  Rejected:        {len(rejected):6d}  ({len(rejected)/total*100:5.1f}%)" if total > 0 else "")
    if pending:
        lines.append(f"  Pending/other:   {len(pending):6d}")

    # Trigger distance analysis (ATR units)
    if filled or expired:
        lines.append(f"\n  Trigger distance (ATR units):")
        lines.append(f"  {'Status':10s} {'Count':>6s} {'Mean':>7s} {'Median':>7s} {'Max':>7s}")
        lines.append("  " + "-" * 42)
        for label, group in [("Filled", filled), ("Expired", expired)]:
            dists = [o.get("trigger_dist_atr", 0) for o in group if o.get("trigger_dist_atr") is not None]
            if dists:
                lines.append(
                    f"  {label:10s} {len(dists):6d} {np.mean(dists):7.3f} "
                    f"{np.median(dists):7.3f} {np.max(dists):7.3f}"
                )

    # Submit hour patterns for expired orders
    if expired:
        lines.append(f"\n  Expired orders by submit hour (ET):")
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        hour_counts: dict[int, int] = defaultdict(int)
        for o in expired:
            st = o.get("submit_time")
            if st is not None:
                st_et = st.astimezone(et) if hasattr(st, 'astimezone') else st
                hour_counts[getattr(st_et, 'hour', 0)] += 1
        for hour in sorted(hour_counts):
            lines.append(f"    {hour:2d}:00  {hour_counts[hour]:4d}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 17. Crisis window performance
# ---------------------------------------------------------------------------

CRISIS_WINDOWS = [
    ("2022 Bear", datetime(2022, 1, 3), datetime(2022, 10, 13)),
    ("SVB Crisis", datetime(2023, 3, 8), datetime(2023, 3, 15)),
    ("Aug 2024 Unwind", datetime(2024, 8, 1), datetime(2024, 8, 5)),
    ("Tariff Shock", datetime(2025, 2, 21), datetime(2025, 4, 7)),
    ("Mar 2026 Slow Burn", datetime(2026, 3, 5), datetime(2026, 3, 27)),
]


def _naive(dt: datetime | None) -> datetime | None:
    """Strip timezone info for safe comparison with naive crisis window dates."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def atrss_crisis_window_analysis(trades: list) -> str:
    """Performance during known market stress periods vs normal conditions."""
    if not trades:
        return "=== ATRSS Crisis Window Analysis ===\nNo trades."

    lines = ["=== ATRSS Crisis Window Analysis ==="]
    lines.append(f"  {'Window':25s} {'Dates':25s} {'N':>4s} {'WR':>5s} "
                 f"{'AvgR':>7s} {'TotR':>8s} {'PnL':>10s}")
    lines.append("  " + "-" * 90)

    total_crisis_n = 0
    total_crisis_r = 0.0
    for name, start, end in CRISIS_WINDOWS:
        ct = [t for t in trades
              if (et := _naive(t.entry_time)) is not None and start <= et <= end]
        n = len(ct)
        total_crisis_n += n
        if n == 0:
            lines.append(f"  {name:25s} {str(start.date()) + ' -> ' + str(end.date()):25s} "
                         f"{'--':>4s}")
            continue
        wr = np.mean([t.r_multiple > 0 for t in ct]) * 100
        avg_r = np.mean([t.r_multiple for t in ct])
        tot_r = sum(t.r_multiple for t in ct)
        pnl = sum(_trade_net_pnl(t) for t in ct)
        total_crisis_r += tot_r
        lines.append(f"  {name:25s} {str(start.date()) + ' -> ' + str(end.date()):25s} "
                     f"{n:4d} {wr:4.0f}% {avg_r:+7.3f} {tot_r:+8.2f} ${pnl:+10,.0f}")
        for t in sorted(ct, key=lambda x: x.entry_time or datetime.min):
            d = "L" if t.direction == 1 else "S"
            lines.append(
                f"    {t.symbol:5s} {d} {t.entry_type:10s} "
                f"entry={t.entry_time.strftime('%m-%d %H:%M') if t.entry_time else 'N/A':11s} "
                f"R={t.r_multiple:+.2f} MFE={t.mfe_r:.2f} exit={t.exit_reason}")

    lines.append("")
    non_crisis = [t for t in trades if not any(
        (et := _naive(t.entry_time)) is not None and s <= et <= e
        for _, s, e in CRISIS_WINDOWS)]
    nc_n = len(non_crisis)
    nc_r = np.mean([t.r_multiple for t in non_crisis]) if non_crisis else 0
    lines.append(f"  Crisis:     {total_crisis_n:4d} trades, totR={total_crisis_r:+.2f}")
    lines.append(f"  Non-crisis: {nc_n:4d} trades, avgR={nc_r:+.3f}, "
                 f"totR={sum(t.r_multiple for t in non_crisis):+.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 18. Rolling edge stability
# ---------------------------------------------------------------------------

def atrss_rolling_edge(trades: list, window: int = 30) -> str:
    """Rolling N-trade window metrics to detect edge decay or regime shifts."""
    if len(trades) < window:
        return (f"=== ATRSS Rolling Edge (window={window}) ===\n"
                f"  Insufficient trades ({len(trades)} < {window}).")

    lines = [f"=== ATRSS Rolling Edge Stability (window={window}) ==="]
    lines.append(f"  {'Window':>12s} {'WR':>5s} {'AvgR':>7s} {'PF':>6s} "
                 f"{'TotR':>8s} {'MaxDD_R':>8s}")
    lines.append("  " + "-" * 55)

    roll_wrs: list[float] = []
    roll_avgrs: list[float] = []

    for i in range(0, len(trades) - window + 1, max(1, window // 3)):
        chunk = trades[i:i + window]
        wr = np.mean([t.r_multiple > 0 for t in chunk]) * 100
        avg_r = np.mean([t.r_multiple for t in chunk])
        tot_r = sum(t.r_multiple for t in chunk)
        wins_r = sum(t.r_multiple for t in chunk if t.r_multiple > 0)
        loss_r = abs(sum(t.r_multiple for t in chunk if t.r_multiple < 0))
        pf = wins_r / loss_r if loss_r > 0 else float("inf")

        cum = np.cumsum([t.r_multiple for t in chunk])
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        roll_wrs.append(wr)
        roll_avgrs.append(avg_r)

        label = f"#{i+1}-#{i+window}"
        lines.append(f"  {label:>12s} {wr:4.0f}% {avg_r:+7.3f} {min(pf, 99.99):6.2f} "
                     f"{tot_r:+8.2f} {max_dd:8.2f}")

    lines.append("")
    lines.append(f"  WR range:  {min(roll_wrs):.0f}% - {max(roll_wrs):.0f}%  "
                 f"(std={np.std(roll_wrs):.1f}%)")
    lines.append(f"  AvgR range: {min(roll_avgrs):+.3f} - {max(roll_avgrs):+.3f}  "
                 f"(std={np.std(roll_avgrs):.3f})")

    mid = len(trades) // 2
    fh_avg = np.mean([t.r_multiple for t in trades[:mid]])
    sh_avg = np.mean([t.r_multiple for t in trades[mid:]])
    delta = sh_avg - fh_avg
    verdict = "STABLE" if abs(delta) < 0.15 else ("IMPROVING" if delta > 0 else "DECAYING")
    lines.append(f"\n  Edge trend: first-half avgR={fh_avg:+.3f}, "
                 f"second-half avgR={sh_avg:+.3f}, delta={delta:+.3f} -> {verdict}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 19. Profit concentration (Pareto analysis)
# ---------------------------------------------------------------------------

def atrss_profit_concentration(trades: list) -> str:
    """Pareto analysis: what fraction of trades generates what fraction of profit."""
    if not trades:
        return "=== ATRSS Profit Concentration ===\nNo trades."

    lines = ["=== ATRSS Profit Concentration (Pareto) ==="]

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    total_gross_r = sum(t.r_multiple for t in winners)

    if total_gross_r <= 0:
        lines.append("  No gross R to analyze.")
        return "\n".join(lines)

    by_r_desc = sorted(winners, key=lambda t: t.r_multiple, reverse=True)
    cum = 0.0
    thresholds: dict[int, int | None] = {50: None, 75: None, 90: None}
    for i, t in enumerate(by_r_desc, 1):
        cum += t.r_multiple
        pct = cum / total_gross_r * 100
        for th in thresholds:
            if thresholds[th] is None and pct >= th:
                thresholds[th] = i

    for th, count in thresholds.items():
        if count is not None:
            lines.append(f"  Top {count} trades ({count / len(trades) * 100:.0f}% of all) "
                         f"generate {th}% of gross R")

    big = [t for t in winners if t.r_multiple >= 3.0]
    med = [t for t in winners if 1.0 <= t.r_multiple < 3.0]
    small = [t for t in winners if 0 < t.r_multiple < 1.0]

    lines.append(f"\n  Big winners   (>=3R):  {len(big):3d} trades, "
                 f"totR={sum(t.r_multiple for t in big):+.2f} "
                 f"({sum(t.r_multiple for t in big) / total_gross_r * 100:.0f}% of gross)")
    lines.append(f"  Medium winners (1-3R): {len(med):3d} trades, "
                 f"totR={sum(t.r_multiple for t in med):+.2f} "
                 f"({sum(t.r_multiple for t in med) / total_gross_r * 100:.0f}% of gross)")
    lines.append(f"  Small winners  (<1R):  {len(small):3d} trades, "
                 f"totR={sum(t.r_multiple for t in small):+.2f} "
                 f"({sum(t.r_multiple for t in small) / total_gross_r * 100:.0f}% of gross)")

    total_loss_r = abs(sum(t.r_multiple for t in losers))
    net_r = total_gross_r - total_loss_r
    lines.append(f"\n  Gross win R:  {total_gross_r:+.2f}")
    lines.append(f"  Gross loss R: {-total_loss_r:+.2f}")
    lines.append(f"  Net R:        {net_r:+.2f}")
    if total_loss_r > 0:
        lines.append(f"  Win/Loss ratio: {total_gross_r / total_loss_r:.2f}x")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 20. Right-then-stopped deep dive (exit management failures)
# ---------------------------------------------------------------------------

def atrss_right_then_stopped(trades: list) -> str:
    """Trades that reached good MFE then reversed to a loss -- exit timing failures."""
    if not trades:
        return "=== ATRSS Right-then-Stopped ===\nNo trades."

    rts = [t for t in trades if t.r_multiple <= 0 and t.mfe_r >= 0.5]
    if not rts:
        return "=== ATRSS Right-then-Stopped ===\n  No trades went right then lost."

    lines = ["=== ATRSS Right-then-Stopped Deep Dive ==="]
    total_leaked = sum(t.mfe_r - t.r_multiple for t in rts)
    lines.append(f"  {len(rts)} trades reached MFE >= 0.5R then finished at or below 0R")
    lines.append(f"  Total R leaked (MFE - finalR): {total_leaked:+.2f}R")
    lines.append(f"  Avg MFE before reversal: {np.mean([t.mfe_r for t in rts]):.2f}R")
    lines.append(f"  Avg final R: {np.mean([t.r_multiple for t in rts]):+.3f}")

    by_reason: dict[str, list] = defaultdict(list)
    for t in rts:
        by_reason[t.exit_reason].append(t)
    lines.append(f"\n  By exit reason:")
    lines.append(f"  {'Reason':20s} {'N':>4s} {'AvgMFE':>7s} {'AvgR':>7s} {'Leaked':>8s}")
    lines.append("  " + "-" * 50)
    for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
        rt = by_reason[reason]
        leaked = sum(t.mfe_r - t.r_multiple for t in rt)
        lines.append(f"  {reason:20s} {len(rt):4d} {np.mean([t.mfe_r for t in rt]):7.2f} "
                     f"{np.mean([t.r_multiple for t in rt]):+7.3f} {leaked:+8.2f}")

    lines.append(f"\n  By MFE reached:")
    for lo, hi, label in [(0.5, 1.0, "0.5-1.0R"), (1.0, 2.0, "1.0-2.0R"),
                           (2.0, 3.0, "2.0-3.0R"), (3.0, 99, "3.0R+")]:
        bucket = [t for t in rts if lo <= t.mfe_r < hi]
        if bucket:
            leaked = sum(t.mfe_r - t.r_multiple for t in bucket)
            lines.append(f"    MFE {label:8s}  n={len(bucket):3d}  "
                         f"avgR={np.mean([t.r_multiple for t in bucket]):+.3f}  "
                         f"leaked={leaked:+.2f}R")

    lines.append(f"\n  By hold time:")
    for lo, hi, label in [(1, 13, "1-12 bars"), (13, 37, "13-36 bars"),
                           (37, 73, "37-72 bars"), (73, 999, "72+ bars")]:
        bucket = [t for t in rts if lo <= t.bars_held < hi]
        if bucket:
            lines.append(f"    {label:12s}  n={len(bucket):3d}  "
                         f"avgMFE={np.mean([t.mfe_r for t in bucket]):.2f}R  "
                         f"avgR={np.mean([t.r_multiple for t in bucket]):+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 21. Monthly returns calendar
# ---------------------------------------------------------------------------

def atrss_monthly_returns(trades: list) -> str:
    """Month-by-month R returns grid for seasonality and consistency analysis."""
    if not trades:
        return "=== ATRSS Monthly Returns ===\nNo trades."

    lines = ["=== ATRSS Monthly Returns Calendar ==="]

    month_r: dict[str, float] = defaultdict(float)
    month_n: dict[str, int] = defaultdict(int)
    for t in trades:
        if t.entry_time is None:
            continue
        key = t.entry_time.strftime("%Y-%m")
        month_r[key] += t.r_multiple
        month_n[key] += 1

    if not month_r:
        lines.append("  No timestamped trades.")
        return "\n".join(lines)

    lines.append(f"  {'Month':>7s} {'Trades':>6s} {'R':>8s} {'CumR':>8s} {'AvgR':>7s}")
    lines.append("  " + "-" * 42)
    cum = 0.0
    pos_months = 0
    neg_months = 0
    for month in sorted(month_r):
        r = month_r[month]
        cum += r
        avg = r / month_n[month] if month_n[month] > 0 else 0
        marker = "+" if r > 0 else "-" if r < 0 else " "
        lines.append(f"  {month:>7s} {month_n[month]:6d} {r:+8.2f} {cum:+8.2f} "
                     f"{avg:+7.3f} {marker}")
        if r > 0:
            pos_months += 1
        elif r < 0:
            neg_months += 1

    total_months = pos_months + neg_months
    lines.append("")
    if total_months > 0:
        lines.append(f"  Positive months: {pos_months}/{total_months} "
                     f"({pos_months / total_months * 100:.0f}%)")
    lines.append(f"  Best month:  {max(month_r, key=month_r.get)} "  # type: ignore[arg-type]
                 f"({max(month_r.values()):+.2f}R)")
    lines.append(f"  Worst month: {min(month_r, key=month_r.get)} "  # type: ignore[arg-type]
                 f"({min(month_r.values()):+.2f}R)")

    year_r: dict[int, float] = defaultdict(float)
    year_n: dict[int, int] = defaultdict(int)
    for t in trades:
        if t.entry_time is None:
            continue
        year_r[t.entry_time.year] += t.r_multiple
        year_n[t.entry_time.year] += 1

    lines.append(f"\n  Yearly summary:")
    lines.append(f"  {'Year':>6s} {'Trades':>6s} {'TotR':>8s} {'AvgR':>7s} {'PF':>6s}")
    lines.append("  " + "-" * 38)
    for year in sorted(year_r):
        yt = [t for t in trades if t.entry_time and t.entry_time.year == year]
        tot = year_r[year]
        avg = tot / year_n[year] if year_n[year] > 0 else 0
        wins_r = sum(t.r_multiple for t in yt if t.r_multiple > 0)
        loss_r = abs(sum(t.r_multiple for t in yt if t.r_multiple < 0))
        pf = wins_r / loss_r if loss_r > 0 else float("inf")
        lines.append(f"  {year:6d} {year_n[year]:6d} {tot:+8.2f} {avg:+7.3f} "
                     f"{min(pf, 99.99):6.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 22. ADX edge analysis (ATRSS-specific)
# ---------------------------------------------------------------------------

def atrss_adx_edge_analysis(trades: list) -> str:
    """ADX-at-entry analysis: optimal entry zones, correlation with outcomes."""
    adx_trades = [t for t in trades if t.adx_entry > 0]
    if not adx_trades:
        return "=== ATRSS ADX Edge Analysis ===\nNo ADX data available."

    lines = ["=== ATRSS ADX Edge Analysis ==="]
    adx_vals = np.array([t.adx_entry for t in adx_trades])
    r_vals = np.array([t.r_multiple for t in adx_trades])

    lines.append(f"  ADX at entry: mean={np.mean(adx_vals):.1f}, "
                 f"median={np.median(adx_vals):.1f}, "
                 f"range=[{np.min(adx_vals):.1f}, {np.max(adx_vals):.1f}]")

    if len(adx_vals) >= 5:
        corr = np.corrcoef(adx_vals, r_vals)[0, 1]
        lines.append(f"  Correlation(ADX, R): {corr:+.3f}")

    lines.append(f"\n  {'ADX Range':>12s} {'N':>4s} {'WR':>5s} {'AvgR':>7s} "
                 f"{'TotR':>8s} {'PF':>6s} {'AvgMFE':>7s}")
    lines.append("  " + "-" * 55)
    for lo, hi, label in [(15, 20, "15-20"), (20, 25, "20-25"), (25, 30, "25-30"),
                           (30, 40, "30-40"), (40, 60, "40-60"), (60, 999, "60+")]:
        bucket = [t for t in adx_trades if lo <= t.adx_entry < hi]
        if not bucket:
            continue
        wr = np.mean([t.r_multiple > 0 for t in bucket]) * 100
        avg_r = np.mean([t.r_multiple for t in bucket])
        tot_r = sum(t.r_multiple for t in bucket)
        wins = sum(t.r_multiple for t in bucket if t.r_multiple > 0)
        loss = abs(sum(t.r_multiple for t in bucket if t.r_multiple < 0))
        pf = wins / loss if loss > 0 else float("inf")
        mfe = np.mean([t.mfe_r for t in bucket])
        lines.append(f"  {label:>12s} {len(bucket):4d} {wr:4.0f}% {avg_r:+7.3f} "
                     f"{tot_r:+8.2f} {min(pf, 99.99):6.2f} {mfe:7.2f}")

    lines.append(f"\n  By regime:")
    lines.append(f"  {'Regime':>14s} {'N':>4s} {'AvgADX':>7s} {'WR':>5s} "
                 f"{'AvgR':>7s} {'TotR':>8s}")
    lines.append("  " + "-" * 50)
    regimes = sorted(set(t.regime_entry for t in adx_trades if t.regime_entry))
    for regime in regimes:
        rt = [t for t in adx_trades if t.regime_entry == regime]
        if not rt:
            continue
        lines.append(f"  {regime:>14s} {len(rt):4d} {np.mean([t.adx_entry for t in rt]):7.1f} "
                     f"{np.mean([t.r_multiple > 0 for t in rt]) * 100:4.0f}% "
                     f"{np.mean([t.r_multiple for t in rt]):+7.3f} "
                     f"{sum(t.r_multiple for t in rt):+8.2f}")

    best_bucket = None
    best_avg_r = -999.0
    for lo, hi in [(15, 20), (20, 25), (25, 30), (30, 40), (40, 60)]:
        bucket = [t for t in adx_trades if lo <= t.adx_entry < hi]
        if len(bucket) >= 5:
            avg = float(np.mean([t.r_multiple for t in bucket]))
            if avg > best_avg_r:
                best_avg_r = avg
                best_bucket = f"{lo}-{hi}"
    if best_bucket:
        lines.append(f"\n  Optimal ADX entry zone: {best_bucket} (avgR={best_avg_r:+.3f})")

    return "\n".join(lines)
