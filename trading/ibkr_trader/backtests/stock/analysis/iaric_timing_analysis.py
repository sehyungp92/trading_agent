"""IARIC timing analysis -- deep dive into entry timing and windows.

Sections:
1. 30-minute bucket performance -- 13 buckets, n/WR/R/PF each
2. Multiplier calibration -- actual R-per-unit-risk vs current multiplier
3. First-hour alpha -- 9:35-10:30 edge vs rest
4. Midday dead zone -- 12:00-13:30 @0.70x: too generous?
5. Late-day time pressure -- 14:30-15:00: forced EOD flatten %
6. Setup detection timing -- when setups fire, detection time -> quality
7. Open/close block shadow -- setups missed during blocked periods
"""
from __future__ import annotations

from collections import defaultdict
from datetime import time

import numpy as np

from backtests.stock.models import TradeRecord


def _meta(t: TradeRecord, key: str, default=None):
    return t.metadata.get(key, default)


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _group_stats_inline(trades: list[TradeRecord]) -> str:
    if not trades:
        return "n=0"
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    wr = wins / n
    rs = [t.r_multiple for t in trades]
    mean_r = float(np.mean(rs))
    gross_p = sum(r for r in rs if r > 0)
    gross_l = abs(sum(r for r in rs if r < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    return f"n={n}, WR={wr:.0%}, R={mean_r:+.3f}, PF={pf:.2f}, Total={sum(rs):+.2f}"


def _s1_30min_buckets(trades: list[TradeRecord]) -> str:
    """30-minute bucket performance."""
    lines = [_hdr("TIMING-1. 30-Minute Bucket Performance")]

    buckets = []
    start_h, start_m = 9, 30
    while start_h < 16:
        end_m = start_m + 30
        end_h = start_h
        if end_m >= 60:
            end_m -= 60
            end_h += 1
        if end_h >= 16:
            break
        buckets.append((time(start_h, start_m), time(end_h, end_m)))
        start_h, start_m = end_h, end_m

    lines.append(f"  {'Window':<14s} {_group_stats_inline([]):>60s}")
    lines.append("  " + "-" * 74)

    for start, end in buckets:
        group = []
        for t in trades:
            et = t.entry_time
            t_time = time(et.hour, et.minute)
            if start <= t_time < end:
                group.append(t)
        label = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
        lines.append(f"  {label:<14s} {_group_stats_inline(group):>60s}")

    return "\n".join(lines)


def _s2_multiplier_calibration(trades: list[TradeRecord]) -> str:
    """Actual R-per-unit-risk vs current multiplier per window."""
    lines = [_hdr("TIMING-2. Multiplier Calibration")]

    windows = [
        ("09:35-10:30", time(9, 35), time(10, 30), 1.00),
        ("10:30-12:00", time(10, 30), time(12, 0), 0.85),
        ("12:00-13:30", time(12, 0), time(13, 30), 0.70),
        ("13:30-14:30", time(13, 30), time(14, 30), 0.90),
        ("14:30-15:00", time(14, 30), time(15, 0), 0.75),
    ]

    lines.append(f"  {'Window':<14s} {'Mult':>5s} {'N':>5s} {'Actual R':>10s} {'R/Unit':>10s} {'Suggested':>10s}")
    lines.append("  " + "-" * 58)

    for label, start, end, mult in windows:
        group = [t for t in trades if start <= time(t.entry_time.hour, t.entry_time.minute) < end]
        if not group:
            lines.append(f"  {label:<14s} {mult:>5.2f} {0:>5} {'--':>10s} {'--':>10s} {'--':>10s}")
            continue

        mean_r = float(np.mean([t.r_multiple for t in group]))
        # R per unit of risk deployed at this window
        risk_units = [_meta(t, "risk_unit_final", 1.0) or 1.0 for t in group]
        r_per_unit = mean_r / float(np.mean(risk_units)) if np.mean(risk_units) > 0 else 0

        # Suggested multiplier: scale current by actual performance
        all_mean = float(np.mean([t.r_multiple for t in trades])) if trades else 0
        suggested = mult * (mean_r / all_mean) if all_mean != 0 else mult

        lines.append(
            f"  {label:<14s} {mult:>5.2f} {len(group):>5}"
            f" {mean_r:>+10.3f} {r_per_unit:>+10.3f} {suggested:>10.2f}"
        )

    return "\n".join(lines)


def _s3_first_hour_alpha(trades: list[TradeRecord]) -> str:
    """9:35-10:30 edge vs rest."""
    lines = [_hdr("TIMING-3. First-Hour Alpha")]

    first_hour = [t for t in trades if time(9, 35) <= time(t.entry_time.hour, t.entry_time.minute) < time(10, 30)]
    rest = [t for t in trades if time(t.entry_time.hour, t.entry_time.minute) >= time(10, 30)]

    lines.append(f"  First hour (9:35-10:30): {_group_stats_inline(first_hour)}")
    lines.append(f"  Rest of day (10:30+):    {_group_stats_inline(rest)}")

    if first_hour and rest:
        fh_mean = float(np.mean([t.r_multiple for t in first_hour]))
        rest_mean = float(np.mean([t.r_multiple for t in rest]))
        lines.append(f"\n  First hour alpha: {fh_mean - rest_mean:+.3f}R per trade")

    return "\n".join(lines)


def _s4_midday_dead_zone(trades: list[TradeRecord]) -> str:
    """12:00-13:30 @0.70x: too generous?"""
    lines = [_hdr("TIMING-4. Midday Dead Zone (12:00-13:30)")]

    midday = [t for t in trades if time(12, 0) <= time(t.entry_time.hour, t.entry_time.minute) < time(13, 30)]
    non_midday = [t for t in trades if not (time(12, 0) <= time(t.entry_time.hour, t.entry_time.minute) < time(13, 30))]

    lines.append(f"  Midday (12:00-13:30, ×0.70): {_group_stats_inline(midday)}")
    lines.append(f"  Non-midday:                   {_group_stats_inline(non_midday)}")

    if midday:
        midday_mean = float(np.mean([t.r_multiple for t in midday]))
        lines.append(f"\n  Midday mean R: {midday_mean:+.3f}")
        if midday_mean < 0:
            lines.append("  -> Consider lowering multiplier or blocking midday entries")
        elif midday_mean > 0.1:
            lines.append("  -> 0.70x may be too conservative -- midday is profitable")

    return "\n".join(lines)


def _s5_late_day_pressure(trades: list[TradeRecord]) -> str:
    """14:30-15:00: forced EOD flatten %."""
    lines = [_hdr("TIMING-5. Late-Day Time Pressure (14:30-15:00)")]

    late = [t for t in trades if time(14, 30) <= time(t.entry_time.hour, t.entry_time.minute) < time(15, 0)]
    if not late:
        lines.append("  No late-day entries")
        return "\n".join(lines)

    lines.append(f"  Late entries (14:30-15:00): {_group_stats_inline(late)}")

    eod_flatten = [t for t in late if t.exit_reason == "EOD_FLATTEN"]
    lines.append(f"  Forced EOD flatten: {len(eod_flatten)}/{len(late)} ({len(eod_flatten)/len(late):.0%})")

    # Average hold time
    avg_bars = float(np.mean([t.hold_bars for t in late]))
    lines.append(f"  Avg hold: {avg_bars:.0f} bars ({avg_bars*5:.0f}min)")

    return "\n".join(lines)


def _s6_setup_detection_timing(fsm_log: list[dict], trades: list[TradeRecord]) -> str:
    """When setups fire and detection time -> quality."""
    lines = [_hdr("TIMING-6. Setup Detection Timing")]
    if not fsm_log:
        lines.append("  No FSM log data")
        return "\n".join(lines)

    # When do SETUP_DETECTED transitions occur?
    detection_times: list[int] = []  # minutes from 9:30
    for entry in fsm_log:
        if entry["to_state"] == "SETUP_DETECTED":
            ts = entry["timestamp"]
            minutes = ts.hour * 60 + ts.minute - (9 * 60 + 30)
            if 0 <= minutes <= 390:
                detection_times.append(minutes)

    if not detection_times:
        lines.append("  No setup detections in log")
        return "\n".join(lines)

    lines.append(f"  Total setup detections: {len(detection_times)}")
    lines.append(f"  Mean time: {float(np.mean(detection_times)):.0f}min from open")
    lines.append(f"  Median time: {float(np.median(detection_times)):.0f}min from open")

    # Bucket by 30-min windows
    buckets: dict[str, int] = defaultdict(int)
    for m in detection_times:
        h = 9 + (m + 30) // 60
        mn = (m + 30) % 60
        bucket = f"{h:02d}:{(mn // 30) * 30:02d}"
        buckets[bucket] += 1

    lines.append(f"\n  Detection distribution:")
    for bucket in sorted(buckets):
        count = buckets[bucket]
        bar = "█" * min(count, 50)
        lines.append(f"    {bucket}: {count:>4} {bar}")

    return "\n".join(lines)


def _s7_blocked_period_shadow(fsm_log: list[dict], rejection_log: list[dict]) -> str:
    """Setups missed during blocked periods (open block, close block)."""
    lines = [_hdr("TIMING-7. Blocked Period Analysis")]
    if not rejection_log:
        lines.append("  No rejection log data")
        return "\n".join(lines)

    timing_blocked = [r for r in rejection_log if r.get("gate") == "timing_blocked"]
    if not timing_blocked:
        lines.append("  No timing-blocked rejections")
        return "\n".join(lines)

    lines.append(f"  Timing-blocked rejections: {len(timing_blocked)}")

    # By time bucket
    by_bucket: dict[str, int] = defaultdict(int)
    for r in timing_blocked:
        ts = r["timestamp"]
        et_time = time(ts.hour, ts.minute)
        if et_time < time(9, 35):
            by_bucket["pre-open (<9:35)"] += 1
        elif et_time >= time(15, 45):
            by_bucket["close-block (15:45+)"] += 1
        elif et_time >= time(15, 0):
            by_bucket["post-close (15:00-15:45)"] += 1
        else:
            by_bucket["other"] += 1

    for bucket, count in sorted(by_bucket.items(), key=lambda x: -x[1]):
        lines.append(f"    {bucket}: {count}")

    # Setup types missed
    types: dict[str, int] = defaultdict(int)
    for r in timing_blocked:
        types[r.get("setup_type", "?")] += 1
    if types:
        lines.append(f"\n  Setup types missed: {dict(types)}")

    return "\n".join(lines)


def iaric_timing_analysis(
    trades: list[TradeRecord],
    fsm_log: list[dict] | None = None,
    rejection_log: list[dict] | None = None,
) -> str:
    """Generate the full IARIC timing analysis report."""
    if not trades:
        return "No trades to analyze."

    sections = [
        _s1_30min_buckets(trades),
        _s2_multiplier_calibration(trades),
        _s3_first_hour_alpha(trades),
        _s4_midday_dead_zone(trades),
        _s5_late_day_pressure(trades),
    ]

    if fsm_log:
        sections.append(_s6_setup_detection_timing(fsm_log, trades))
        sections.append(_s7_blocked_period_shadow(fsm_log, rejection_log or []))

    return "\n".join(sections)
