"""Extended diagnostic reports for Helix (AKC-Helix) strategy backtests.

All functions accept list[HelixTradeRecord] (duck-typed) and return str.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np


def _trade_net_pnl(trade) -> float:
    return float(trade.pnl_dollars) - float(getattr(trade, "commission", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# 1. Class drill-down
# ---------------------------------------------------------------------------

def helix_class_drilldown(trades: list) -> str:
    """Per-class (A/D) table with detailed metrics."""
    if not trades:
        return "No trades for class drilldown."

    lines = ["=== Helix Class Drilldown ==="]
    header = (
        f"  {'Class':5s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'P&L':>10s} "
        f"{'MFE':>6s} {'MAE':>6s} {'Hold':>6s} {'Long':>5s} {'Short':>5s} "
        f"{'BULL':>5s} {'BEAR':>5s} {'CHOP':>5s} {'DivMag':>7s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for cls in ["A", "D"]:
        ct = [t for t in trades if t.setup_class == cls]
        if not ct:
            lines.append(f"  {cls:5s} {'0':>6s}")
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

        regime_counts = Counter(t.regime_at_entry for t in ct)
        n_bull = regime_counts.get("BULL", 0)
        n_bear = regime_counts.get("BEAR", 0)
        n_chop = regime_counts.get("CHOP", 0)

        div_mag = np.mean([getattr(t, 'div_mag_norm', 0.0) for t in ct])

        lines.append(
            f"  {cls:5s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f} "
            f"{mfe:6.2f} {mae:6.2f} {hold:6.1f} {n_long:5d} {n_short:5d} "
            f"{n_bull:5d} {n_bear:5d} {n_chop:5d} {div_mag:7.3f}"
        )

    # Summary row
    count = len(trades)
    wr = np.mean([t.r_multiple > 0 for t in trades]) * 100
    avg_r = np.mean([t.r_multiple for t in trades])
    pnl = sum(_trade_net_pnl(t) for t in trades)
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(
        f"  {'ALL':5s} {count:6d} {wr:5.0f}% {avg_r:+7.3f} {pnl:+10,.0f}"
    )

    # Flag worst class
    class_avg_r = {}
    for cls in ["A", "D"]:
        ct = [t for t in trades if t.setup_class == cls]
        if ct:
            class_avg_r[cls] = np.mean([t.r_multiple for t in ct])
    if class_avg_r:
        worst = min(class_avg_r, key=class_avg_r.get)
        if class_avg_r[worst] < 0:
            lines.append(f"\n  ** Class {worst} is the primary drag (avg R = {class_avg_r[worst]:+.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Divergence quality (Class A only)
# ---------------------------------------------------------------------------

def helix_divergence_quality(trades: list) -> str:
    """Divergence magnitude analysis for Class A trades."""
    if not trades:
        return "No trades for divergence analysis."

    class_a = [t for t in trades if t.setup_class == "A"]
    if not class_a:
        return "=== Helix Divergence Quality ===\nNo Class A trades."

    lines = ["=== Helix Divergence Quality ==="]

    div_vals = np.array([getattr(t, 'div_mag_norm', 0.0) for t in class_a])
    rs = np.array([t.r_multiple for t in class_a])

    if div_vals.sum() == 0:
        lines.append("\nNo divergence magnitude data recorded.")
        return "\n".join(lines)

    lines.append(f"\nDivergence magnitude distribution (Class A, n={len(class_a)}):")
    for label, val in [
        ("Min", np.min(div_vals)),
        ("25th", np.percentile(div_vals, 25)),
        ("Median", np.median(div_vals)),
        ("75th", np.percentile(div_vals, 75)),
        ("Max", np.max(div_vals)),
        ("Mean", np.mean(div_vals)),
    ]:
        lines.append(f"  {label:8s}: {val:.4f}")

    # Quartile-bucketed analysis
    lines.append(f"\nQuartile analysis:")
    lines.append(f"  {'Quartile':10s} {'Range':>20s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}")
    try:
        edges = np.percentile(div_vals, [0, 25, 50, 75, 100])
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            if i < 3:
                mask = (div_vals >= lo) & (div_vals < hi)
            else:
                mask = (div_vals >= lo) & (div_vals <= hi)
            if mask.sum() == 0:
                continue
            bucket_rs = rs[mask]
            avg_r = np.mean(bucket_rs)
            wr = np.mean(bucket_rs > 0) * 100
            lines.append(
                f"  Q{i+1:1d}        {lo:8.4f} - {hi:8.4f} {mask.sum():6d} {avg_r:+7.3f} {wr:5.0f}%"
            )
    except Exception:
        lines.append("  (insufficient data for quartile analysis)")

    # Correlation
    if len(div_vals) >= 5:
        corr = np.corrcoef(div_vals, rs)[0, 1]
        lines.append(f"\nCorrelation(DivMag, R): {corr:+.3f}")
        if abs(corr) < 0.1:
            lines.append("  -> Weak correlation: magnitude is not predictive of R")
        elif corr > 0.1:
            lines.append("  -> Positive: larger divergence tends to produce better R")
        else:
            lines.append("  -> Negative: larger divergence produces worse R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2b. 4H Regime alignment
# ---------------------------------------------------------------------------

def helix_regime_4h_alignment(trades: list) -> str:
    """Track 4H regime agreement with daily regime at entry."""
    if not trades:
        return "No trades for 4H regime analysis."

    lines = ["=== Helix 4H Regime Alignment ==="]

    regime_4h_vals = [getattr(t, 'regime_4h_at_entry', '') for t in trades]
    has_data = any(v for v in regime_4h_vals)
    if not has_data:
        lines.append("\nNo 4H regime data recorded.")
        return "\n".join(lines)

    # Cross-tab: daily regime x 4H regime
    lines.append(f"\n  {'Daily':8s} {'4H':8s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s}")
    lines.append("  " + "-" * 42)

    for daily_regime in ["BULL", "BEAR", "CHOP"]:
        for regime_4h in ["BULL", "BEAR", "CHOP"]:
            cell = [t for t in trades
                    if t.regime_at_entry == daily_regime
                    and getattr(t, 'regime_4h_at_entry', '') == regime_4h]
            if not cell:
                continue
            avg_r = np.mean([t.r_multiple for t in cell])
            wr = np.mean([t.r_multiple > 0 for t in cell]) * 100
            lines.append(
                f"  {daily_regime:8s} {regime_4h:8s} {len(cell):6d} {avg_r:+7.3f} {wr:5.0f}%"
            )

    # Agreement summary
    agreed = [t for t in trades
              if t.regime_at_entry == getattr(t, 'regime_4h_at_entry', '')]
    disagreed = [t for t in trades
                 if t.regime_at_entry != getattr(t, 'regime_4h_at_entry', '')
                 and getattr(t, 'regime_4h_at_entry', '')]

    if agreed:
        avg_r = np.mean([t.r_multiple for t in agreed])
        wr = np.mean([t.r_multiple > 0 for t in agreed]) * 100
        lines.append(f"\n  Daily-4H AGREED: {len(agreed)} trades, avg R: {avg_r:+.3f}, WR: {wr:.0f}%")
    if disagreed:
        avg_r = np.mean([t.r_multiple for t in disagreed])
        wr = np.mean([t.r_multiple > 0 for t in disagreed]) * 100
        lines.append(f"  Daily-4H DISAGREED: {len(disagreed)} trades, avg R: {avg_r:+.3f}, WR: {wr:.0f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Regime alignment
# ---------------------------------------------------------------------------

def helix_regime_alignment(trades: list, result=None) -> str:
    """Cross-tab of regime x direction with count, avg R, WR."""
    if not trades:
        return "No trades for regime alignment."

    lines = ["=== Helix Regime Alignment ==="]

    # 3x2 cross-tab
    lines.append(f"\n  {'Regime':8s} {'Dir':6s} {'Count':>6s} {'AvgR':>7s} {'WR':>6s} {'P&L':>10s}")
    lines.append("  " + "-" * 50)

    for regime in ["BULL", "BEAR", "CHOP"]:
        for direction, dir_label in [(1, "LONG"), (-1, "SHORT")]:
            cell = [t for t in trades
                    if t.regime_at_entry == regime and t.direction == direction]
            if not cell:
                lines.append(f"  {regime:8s} {dir_label:6s} {'0':>6s}")
                continue
            avg_r = np.mean([t.r_multiple for t in cell])
            wr = np.mean([t.r_multiple > 0 for t in cell]) * 100
            pnl = sum(_trade_net_pnl(t) for t in cell)
            lines.append(
                f"  {regime:8s} {dir_label:6s} {len(cell):6d} {avg_r:+7.3f} {wr:5.0f}% {pnl:+10,.0f}"
            )

    # Highlight counter-regime losses
    counter_regime = []
    for t in trades:
        if t.regime_at_entry == "BULL" and t.direction == -1:
            counter_regime.append(t)
        elif t.regime_at_entry == "BEAR" and t.direction == 1:
            counter_regime.append(t)

    if counter_regime:
        avg_r = np.mean([t.r_multiple for t in counter_regime])
        pnl = sum(_trade_net_pnl(t) for t in counter_regime)
        lines.append(f"\n  Counter-regime trades: {len(counter_regime)}")
        lines.append(f"    Avg R: {avg_r:+.3f}  Total P&L: {pnl:+,.0f}")
        if avg_r < -0.3:
            lines.append("    ** Counter-regime trades are a significant drag")

    # Compare regime time distribution vs trade allocation
    if result is not None:
        total_days = (getattr(result, 'regime_days_bull', 0)
                      + getattr(result, 'regime_days_bear', 0)
                      + getattr(result, 'regime_days_chop', 0))
        if total_days > 0:
            lines.append(f"\n  Regime time vs trade allocation:")
            lines.append(f"  {'Regime':8s} {'Time%':>7s} {'Trade%':>7s} {'Delta':>7s}")
            for regime in ["BULL", "BEAR", "CHOP"]:
                days_attr = f"regime_days_{regime.lower()}"
                days = getattr(result, days_attr, 0)
                time_pct = days / total_days * 100
                trade_ct = sum(1 for t in trades if t.regime_at_entry == regime)
                trade_pct = trade_ct / len(trades) * 100 if trades else 0
                delta = trade_pct - time_pct
                lines.append(
                    f"  {regime:8s} {time_pct:6.1f}% {trade_pct:6.1f}% {delta:+6.1f}%"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Stop efficiency
# ---------------------------------------------------------------------------

def helix_stop_efficiency(trades: list) -> str:
    """Stop distance stats, MFE capture ratio, and loser classification."""
    if not trades:
        return "No trades for stop analysis."

    lines = ["=== Helix Stop Efficiency ==="]

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

def helix_time_analysis(trades: list) -> str:
    """Entry time patterns: hour, day-of-week, month, year."""
    if not trades:
        return "No trades for time analysis."

    lines = ["=== Helix Time Analysis ==="]

    # Convert entry_time to Eastern for hour analysis
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

def helix_losing_trade_detail(trades: list) -> str:
    """Detailed table of every losing trade, sorted worst-first."""
    losers = [t for t in trades if t.r_multiple <= 0]
    if not losers:
        return "=== Helix Losing Trade Detail ===\nNo losing trades."

    losers.sort(key=lambda t: t.r_multiple)

    lines = ["=== Helix Losing Trade Detail ==="]
    lines.append(f"\nTotal losers: {len(losers)} / {len(trades)}"
                  f" ({len(losers)/len(trades)*100:.0f}%)")
    lines.append(f"Total loss: {sum(_trade_net_pnl(t) for t in losers):+,.0f}")

    lines.append(f"\n  {'#':>3s} {'Entry':19s} {'Exit':19s} {'Cls':3s} {'Dir':5s} "
                  f"{'Regime':6s} {'ADX':>5s} {'StopDist':>8s} {'R':>7s} "
                  f"{'MFE':>6s} {'MAE':>6s} {'Reason':15s} {'Bars':>4s}")
    lines.append("  " + "-" * 120)

    for i, t in enumerate(losers, 1):
        entry_str = t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "N/A"
        exit_str = t.exit_time.strftime("%Y-%m-%d %H:%M") if t.exit_time else "N/A"
        dir_label = "LONG" if t.direction == 1 else "SHORT"
        stop_dist = abs(t.entry_price - t.initial_stop)
        adx = getattr(t, 'adx_at_entry', 0.0)
        lines.append(
            f"  {i:3d} {entry_str:19s} {exit_str:19s} {t.setup_class:3s} {dir_label:5s} "
            f"{t.regime_at_entry:6s} {adx:5.1f} {stop_dist:8.2f} {t.r_multiple:+7.3f} "
            f"{t.mfe_r:6.2f} {t.mae_r:6.2f} {t.exit_reason:15s} {t.bars_held:4d}"
        )

    # Loss clustering detection
    if len(losers) >= 3:
        lines.append(f"\nLoss clustering:")
        # Check for date clustering
        loss_dates = []
        for t in losers:
            if t.entry_time is not None:
                loss_dates.append(t.entry_time)
        if loss_dates:
            loss_dates.sort()
            # Find clusters: losses within 48 hours of each other
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

def helix_r_curve(trades: list) -> str:
    """Cumulative R over time with drawdown analysis."""
    if not trades:
        return "No trades for R curve."

    # Sort by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)
    rs = [t.r_multiple for t in sorted_trades]
    cum_r = np.cumsum(rs)

    lines = ["=== Helix Cumulative R Curve ==="]

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

        # Summary
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

def helix_streak_analysis(trades: list) -> str:
    """Win/loss streak analysis with recovery patterns."""
    if not trades:
        return "No trades for streak analysis."

    sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.min)
    outcomes = [1 if t.r_multiple > 0 else 0 for t in sorted_trades]

    lines = ["=== Helix Streak Analysis ==="]

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
    # Final streak
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

    # Post-loss-streak recovery: after a loss streak of 3+, what's the next trade?
    if loss_streaks and max(loss_streaks) >= 3:
        lines.append(f"\nPost-loss-streak recovery (after 3+ consecutive losses):")
        # Walk through outcomes to find recovery points
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
        # Check for serial correlation (are wins/losses clustered more than expected?)
        transitions = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i-1])
        expected_transitions = 2 * total_w * total_l / len(outcomes)
        lines.append(f"  Transitions: {transitions} (expected ~{expected_transitions:.0f} if random)")
        if transitions < expected_transitions * 0.7:
            lines.append("  ** Outcomes are clustered: streaks are longer than random")
        elif transitions > expected_transitions * 1.3:
            lines.append("  ** Outcomes alternate more than random: possible mean reversion")

    return "\n".join(lines)
