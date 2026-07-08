"""Overnight gap analysis — stop vulnerability to session gaps.

NQ futures trade 23h/day with a 1h maintenance break. This module
quantifies gap frequency, magnitude, and impact on open positions.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, time


def overnight_gap_report(trades: list, daily_data: tuple | None = None) -> str:
    """Generate overnight gap vulnerability report.

    Args:
        trades: Trade records with entry_time, exit_time, entry_price,
                initial_stop, direction, r_multiple, bars_held.
        daily_data: Optional tuple of (opens, highs, lows, closes, times)
                   daily bar arrays for gap analysis.
    """
    lines = ["=" * 60]
    lines.append("  OVERNIGHT GAP ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    if not trades and daily_data is None:
        lines.append("  No data to analyze.")
        return "\n".join(lines)

    # ── Gap Frequency & Magnitude ──
    lines.append("  A. GAP FREQUENCY & MAGNITUDE")
    lines.append("  " + "-" * 40)

    gap_magnitudes = []
    gap_pcts = []

    if daily_data is not None:
        opens, highs, lows, closes, times = daily_data
        for i in range(1, len(opens)):
            prev_close = float(closes[i - 1])
            cur_open = float(opens[i])
            if prev_close > 0 and not np.isnan(prev_close) and not np.isnan(cur_open):
                gap = cur_open - prev_close
                gap_pct = gap / prev_close * 100
                gap_magnitudes.append(gap)
                gap_pcts.append(gap_pct)

    if gap_magnitudes:
        gaps = np.array(gap_magnitudes)
        gaps_pct = np.array(gap_pcts)
        abs_gaps = np.abs(gaps)
        abs_pcts = np.abs(gaps_pct)

        lines.append(f"    Total trading days:       {len(gap_magnitudes)}")
        lines.append(f"    Avg absolute gap:         {float(np.mean(abs_gaps)):.2f} pts ({float(np.mean(abs_pcts)):.3f}%)")
        lines.append(f"    Median absolute gap:      {float(np.median(abs_gaps)):.2f} pts")
        lines.append("")
        lines.append("    Gap magnitude distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = float(np.percentile(abs_gaps, p))
            pct_val = float(np.percentile(abs_pcts, p))
            lines.append(f"      P{p:02d}: {val:8.2f} pts ({pct_val:.3f}%)")

        # Gap direction bias
        up_gaps = np.sum(gaps > 0)
        down_gaps = np.sum(gaps < 0)
        flat_gaps = np.sum(gaps == 0)
        lines.append("")
        lines.append(f"    Gap up:    {up_gaps} ({up_gaps/len(gaps)*100:.1f}%)")
        lines.append(f"    Gap down:  {down_gaps} ({down_gaps/len(gaps)*100:.1f}%)")
        lines.append(f"    Flat:      {flat_gaps} ({flat_gaps/len(gaps)*100:.1f}%)")
    else:
        lines.append("    (No daily data provided for gap analysis)")

    # ── Impact on Open Positions ──
    lines.append("")
    lines.append("  B. IMPACT ON OPEN POSITIONS")
    lines.append("  " + "-" * 40)

    # Trades held overnight (bars_held > intraday bars, or check multi-day)
    overnight_trades = [t for t in trades if getattr(t, 'bars_held', 0) > 1]
    lines.append(f"    Trades held overnight:   {len(overnight_trades)} of {len(trades)} ({len(overnight_trades)/len(trades)*100:.1f}%)" if trades else "    No trades")

    if overnight_trades:
        on_r = np.array([getattr(t, 'r_multiple', 0.0) for t in overnight_trades])
        day_trades = [t for t in trades if getattr(t, 'bars_held', 0) <= 1]
        day_r = np.array([getattr(t, 'r_multiple', 0.0) for t in day_trades]) if day_trades else np.array([0.0])

        lines.append(f"    Overnight avg R:         {float(np.mean(on_r)):+.3f}")
        lines.append(f"    Intraday avg R:          {float(np.mean(day_r)):+.3f}")
        delta = float(np.mean(on_r)) - float(np.mean(day_r))
        lines.append(f"    Delta:                   {delta:+.3f}R")

    # ── Stop Vulnerability ──
    lines.append("")
    lines.append("  C. STOP VULNERABILITY")
    lines.append("  " + "-" * 40)

    if gap_magnitudes and trades:
        # Check what % of stops are within P75 gap distance
        p75_gap = float(np.percentile(np.abs(gap_magnitudes), 75))
        p90_gap = float(np.percentile(np.abs(gap_magnitudes), 90))

        stop_distances = []
        for t in trades:
            entry = getattr(t, 'entry_price', 0.0)
            stop = getattr(t, 'initial_stop', 0.0) or getattr(t, 'stop0', 0.0)
            if entry > 0 and stop > 0:
                stop_distances.append(abs(entry - stop))

        if stop_distances:
            sd_arr = np.array(stop_distances)
            within_p75 = float(np.mean(sd_arr <= p75_gap)) * 100
            within_p90 = float(np.mean(sd_arr <= p90_gap)) * 100

            lines.append(f"    P75 gap size:            {p75_gap:.2f} pts")
            lines.append(f"    P90 gap size:            {p90_gap:.2f} pts")
            lines.append(f"    Avg stop distance:       {float(np.mean(sd_arr)):.2f} pts")
            lines.append(f"    Stops within P75 gap:    {within_p75:.1f}%")
            lines.append(f"    Stops within P90 gap:    {within_p90:.1f}%")

            lines.append("")
            if within_p75 > 15:
                verdict = f"VULNERABLE — {within_p75:.0f}% of stops within P75 gap distance"
            elif within_p90 > 30:
                verdict = f"MODERATE — {within_p90:.0f}% of stops within P90 gap distance"
            else:
                verdict = "SAFE — stops are generally wider than typical gaps"
            lines.append(f"    Verdict: {verdict}")
        else:
            lines.append("    (Could not compute stop distances)")
    else:
        lines.append("    (Insufficient data for stop vulnerability analysis)")

    # ── D. ETH→RTH vs RTH→ETH BRIDGE ANALYSIS ──
    lines.append("")
    lines.append("  D. SESSION BRIDGE ANALYSIS")
    lines.append("  " + "-" * 40)

    if daily_data is not None and len(gap_magnitudes) > 0:
        opens, highs, lows, closes, times = daily_data

        # Classify gaps by bridge type using approximate session times
        eth_to_rth_gaps = []   # gap at RTH open (~14:30 UTC / 09:30 ET)
        rth_to_eth_gaps = []   # gap at ETH open (~23:00 UTC / 18:00 ET)

        for i in range(1, len(opens)):
            gap = float(opens[i]) - float(closes[i - 1])
            t = times[i] if i < len(times) else None
            if t is not None and hasattr(t, 'hour'):
                hour = t.hour
                # RTH open is around 14:30 UTC (09:30 ET)
                if 13 <= hour <= 15:
                    eth_to_rth_gaps.append(gap)
                else:
                    rth_to_eth_gaps.append(gap)
            else:
                eth_to_rth_gaps.append(gap)  # default to ETH→RTH for daily bars

        if eth_to_rth_gaps:
            arr = np.array(eth_to_rth_gaps)
            lines.append(f"    ETH->RTH gaps ({len(arr)}):")
            lines.append(f"      Avg: {float(np.mean(arr)):+.2f} pts, Abs avg: {float(np.mean(np.abs(arr))):.2f} pts")
            lines.append(f"      Max up: {float(np.max(arr)):+.2f}, Max down: {float(np.min(arr)):+.2f}")

        if rth_to_eth_gaps:
            arr = np.array(rth_to_eth_gaps)
            lines.append(f"    RTH->ETH gaps ({len(arr)}):")
            lines.append(f"      Avg: {float(np.mean(arr)):+.2f} pts, Abs avg: {float(np.mean(np.abs(arr))):.2f} pts")
            lines.append(f"      Max up: {float(np.max(arr)):+.2f}, Max down: {float(np.min(arr)):+.2f}")
    else:
        lines.append("    (Insufficient data for bridge analysis)")

    # ── E. GAP DIRECTION vs POSITION DIRECTION ──
    lines.append("")
    lines.append("  E. GAP DIRECTION vs POSITION DIRECTION")
    lines.append("  " + "-" * 40)

    if gap_magnitudes and overnight_trades:
        # Build date-to-gap map
        date_gap = {}
        if daily_data is not None:
            opens, highs, lows, closes, times = daily_data
            for i in range(1, len(opens)):
                gap = float(opens[i]) - float(closes[i - 1])
                t = times[i]
                if hasattr(t, 'date'):
                    date_gap[t.date()] = gap
                elif isinstance(t, np.datetime64):
                    try:
                        import pandas as pd
                        date_gap[pd.Timestamp(t).date()] = gap
                    except Exception:
                        pass

        # Match overnight trades to gaps
        aligned = 0
        opposed = 0
        aligned_r = []
        opposed_r = []

        for t in overnight_trades:
            exit_t = getattr(t, 'exit_time', None) or getattr(t, 'entry_time', None)
            direction = getattr(t, 'direction', 0)
            r = getattr(t, 'r_multiple', 0.0)

            if exit_t is None or direction == 0:
                continue

            trade_date = exit_t.date() if hasattr(exit_t, 'date') else None
            if trade_date and trade_date in date_gap:
                gap = date_gap[trade_date]
                if (gap > 0 and direction == 1) or (gap < 0 and direction == -1):
                    aligned += 1
                    aligned_r.append(r)
                elif (gap > 0 and direction == -1) or (gap < 0 and direction == 1):
                    opposed += 1
                    opposed_r.append(r)

        total_matched = aligned + opposed
        if total_matched > 0:
            lines.append(f"    Trades matched to gaps:  {total_matched}")
            lines.append(f"    Gap aligned (favorable): {aligned} ({aligned/total_matched*100:.1f}%)")
            lines.append(f"    Gap opposed (adverse):   {opposed} ({opposed/total_matched*100:.1f}%)")
            if aligned_r:
                lines.append(f"    Aligned avg R:           {float(np.mean(aligned_r)):+.3f}")
            if opposed_r:
                lines.append(f"    Opposed avg R:           {float(np.mean(opposed_r)):+.3f}")

            if opposed_r and aligned_r:
                delta = float(np.mean(aligned_r)) - float(np.mean(opposed_r))
                if delta > 0.3:
                    lines.append(f"    Verdict: FAVORABLE ALIGNMENT (+{delta:.2f}R advantage)")
                elif delta < -0.3:
                    lines.append(f"    Verdict: ADVERSE ALIGNMENT ({delta:+.2f}R penalty)")
                else:
                    lines.append(f"    Verdict: NEUTRAL (delta={delta:+.2f}R)")
        else:
            lines.append("    (Could not match trades to gap data)")
    else:
        lines.append("    (Insufficient data for gap-direction analysis)")

    return "\n".join(lines)
