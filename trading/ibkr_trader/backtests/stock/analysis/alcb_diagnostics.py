"""ALCB comprehensive diagnostics — 30-section deep analysis.

Provides strategy-specific diagnostic insight into ALCB backtest
results using momentum-era metadata: momentum_score, mfe_r, mae_r,
rvol_at_entry, avwap_at_entry, carry_days, or_high/or_low, and more.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np

from backtests.stock.analysis.alcb_filter_attribution import (
    alcb_filter_attribution_report,
)
from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
from backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics
from backtests.stock.models import TradeRecord


_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(t: TradeRecord, key: str, default=None):
    """Safe metadata access — returns default if metadata empty."""
    return t.metadata.get(key, default)


def _hdr(title: str) -> str:
    """Section header."""
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _group_stats(trades: list[TradeRecord]) -> str:
    """Standard stats block: n, win%, mean R, median R, PF, total R."""
    if not trades:
        return "    (no trades)"
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    wr = wins / n
    rs = [t.r_multiple for t in trades]
    mean_r = float(np.mean(rs))
    median_r = float(np.median(rs))
    total_r = sum(rs)
    gross_p = sum(r for r in rs if r > 0)
    gross_l = abs(sum(r for r in rs if r < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    return (
        f"    n={n}, WR={wr:.1%}, Mean R={mean_r:+.3f}, "
        f"Median R={median_r:+.3f}, PF={pf:.2f}, Total R={total_r:+.2f}"
    )


def _hold_hours(t: TradeRecord) -> float:
    """Compute hold hours from entry/exit times."""
    return (t.exit_time - t.entry_time).total_seconds() / 3600


def _entry_time_et(t: TradeRecord):
    return t.entry_time.astimezone(_ET)


def _has_momentum_metadata(trades: list[TradeRecord]) -> bool:
    """Check if trades have ALCB momentum-era enriched metadata."""
    if not trades:
        return False
    return bool(trades[0].metadata and "momentum_score" in trades[0].metadata)


def _carry_days(t: TradeRecord) -> int:
    """Get carry days from metadata, or compute from entry/exit dates."""
    cd = _meta(t, "carry_days", None)
    if cd is not None:
        return int(cd)
    return (t.exit_time.date() - t.entry_time.date()).days


def _rvol(t: TradeRecord) -> float:
    """Get RVOL at entry from available metadata keys."""
    v = _meta(t, "rvol_at_entry", None)
    if v is not None:
        return float(v)
    v = _meta(t, "breakout_rvol", None)
    if v is not None:
        return float(v)
    return 0.0


def _entry_bar_number(t: TradeRecord) -> int:
    """Compute 5-min bar number from market open (9:30 ET = bar 1)."""
    entry_et = _entry_time_et(t)
    minutes_from_open = (entry_et.hour * 60 + entry_et.minute) - 570
    if minutes_from_open < 0:
        return 0
    return minutes_from_open // 5 + 1


def _normalize_exit(reason: str) -> str:
    """Normalize T1/T2 exit reason naming inconsistency."""
    mapping = {
        "TP1_FULL_EXIT": "TP1_FULL",
        "TP2_FULL_EXIT": "TP2_FULL",
    }
    return mapping.get(reason, reason)


# ---------------------------------------------------------------------------
# Diagnostic sections
# ---------------------------------------------------------------------------


def _s01_overview(trades: list[TradeRecord]) -> str:
    """Section 1: Overview statistics."""
    lines = [_hdr("1. Overview")]
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    rs = [t.r_multiple for t in trades]
    pnls = [t.pnl_net for t in trades]
    total_pnl = sum(pnls)
    mean_r = float(np.mean(rs)) if rs else 0
    median_r = float(np.median(rs)) if rs else 0
    total_r = sum(rs)
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    avg_hold = float(np.mean([_hold_hours(t) for t in trades])) if trades else 0

    lines.append(f"  Trades: {n}")
    lines.append(f"  Win Rate: {wins/n:.1%}" if n else "  Win Rate: N/A")
    lines.append(f"  Mean R: {mean_r:+.3f}  |  Median R: {median_r:+.3f}")
    lines.append(f"  Total R: {total_r:+.2f}  |  Total PnL: ${total_pnl:,.2f}")
    lines.append(f"  Profit Factor: {pf:.2f}")
    lines.append(f"  Avg Hold: {avg_hold:.1f}h")
    return "\n".join(lines)


def _s02_signal_funnel(
    trades: list[TradeRecord],
    daily_selections: dict | None,
    shadow_tracker: ALCBShadowTracker | None,
) -> str:
    """Section 2: Signal funnel (evaluated → entered)."""
    lines = [_hdr("2. Signal Funnel")]
    if shadow_tracker:
        lines.append(shadow_tracker.funnel_report())
    elif daily_selections:
        total_candidates = sum(len(a.tradable) for a in daily_selections.values())
        lines.append(f"  Total tradable candidates evaluated: {total_candidates}")
        lines.append(f"  Trades taken: {len(trades)}")
        if total_candidates > 0:
            lines.append(f"  Conversion: {len(trades)/total_candidates:.1%}")
    else:
        lines.append("  No funnel data available (pass --shadow to enable)")
    return "\n".join(lines)


def _s03_entry_type(trades: list[TradeRecord]) -> str:
    """Section 3: Entry Type A/B/C breakdown."""
    lines = [_hdr("3. Entry Type Breakdown")]
    by_type: dict[str, list[TradeRecord]] = {}
    for t in trades:
        et = t.entry_type or "UNKNOWN"
        by_type.setdefault(et, []).append(t)

    for et in sorted(by_type):
        group = by_type[et]
        lines.append(f"  {et}:")
        lines.append(_group_stats(group))
    return "\n".join(lines)


def _s04_direction(trades: list[TradeRecord]) -> str:
    """Section 4: Long vs Short."""
    lines = [_hdr("4. Direction Breakdown")]
    longs = [t for t in trades if t.direction.value > 0]
    shorts = [t for t in trades if t.direction.value < 0]
    lines.append(f"  LONG:")
    lines.append(_group_stats(longs))
    lines.append(f"  SHORT:")
    lines.append(_group_stats(shorts))
    return "\n".join(lines)


def _s05_momentum_score(trades: list[TradeRecord]) -> str:
    """Section 5: Performance by momentum score — is scoring predictive?"""
    lines = [_hdr("5. Momentum Score Distribution")]
    if not _has_momentum_metadata(trades):
        lines.append("  (no metadata — run with enriched engine)")
        return "\n".join(lines)

    by_score: dict[int, list[TradeRecord]] = {}
    for t in trades:
        sc = _meta(t, "momentum_score", 0) or 0
        by_score.setdefault(int(sc), []).append(t)

    # Show per-score stats
    prev_wr = None
    monotonic = True
    for score in sorted(by_score):
        group = by_score[score]
        n = len(group)
        wr = sum(1 for t in group if t.is_winner) / n if n else 0
        mean_r = float(np.mean([t.r_multiple for t in group]))
        total_r = sum(t.r_multiple for t in group)
        lines.append(
            f"  Score {score}: n={n}, WR={wr:.0%}, "
            f"Mean R={mean_r:+.3f}, Total R={total_r:+.2f}"
        )
        if prev_wr is not None and wr < prev_wr - 0.02:
            monotonic = False
        prev_wr = wr

    lines.append("")
    lines.append(f"  WR monotonic with score: {'YES' if monotonic else 'NO'}")

    # Top vs bottom half
    scores_sorted = sorted(by_score.keys())
    if len(scores_sorted) >= 2:
        mid = scores_sorted[len(scores_sorted) // 2]
        low = [t for t in trades if (_meta(t, "momentum_score", 0) or 0) < mid]
        high = [t for t in trades if (_meta(t, "momentum_score", 0) or 0) >= mid]
        if low and high:
            low_wr = sum(1 for t in low if t.is_winner) / len(low)
            high_wr = sum(1 for t in high if t.is_winner) / len(high)
            lines.append(
                f"  Low scores (<{mid}): WR={low_wr:.0%}, n={len(low)}  |  "
                f"High scores (>={mid}): WR={high_wr:.0%}, n={len(high)}"
            )

    return "\n".join(lines)


def _s06_opening_range(trades: list[TradeRecord]) -> str:
    """Section 6: Opening range quality — OR width and breakout distance."""
    lines = [_hdr("6. Opening Range Quality")]
    if not _has_momentum_metadata(trades):
        lines.append("  (no metadata — run with enriched engine)")
        return "\n".join(lines)

    # Filter trades with OR data
    or_trades = [t for t in trades if _meta(t, "or_high") is not None]
    if not or_trades:
        lines.append("  (no OR data in metadata)")
        return "\n".join(lines)

    # OR width buckets (as % of price)
    lines.append("  By OR Width (% of price):")
    widths = []
    for t in or_trades:
        orh = float(_meta(t, "or_high", 0))
        orl = float(_meta(t, "or_low", 0))
        mid = (orh + orl) / 2 if (orh + orl) > 0 else 1
        widths.append((orh - orl) / mid * 100)

    if widths:
        p33, p66 = np.percentile(widths, [33, 66])
        for label, lo, hi in [
            (f"Tight (<{p33:.1f}%)", 0, p33),
            (f"Normal ({p33:.1f}-{p66:.1f}%)", p33, p66),
            (f"Wide (>{p66:.1f}%)", p66, 999),
        ]:
            group = [
                t for t, w in zip(or_trades, widths) if lo <= w < hi
            ]
            if group:
                lines.append(f"    {label}:")
                lines.append("  " + _group_stats(group))

    # Breakout distance from OR high
    lines.append("")
    lines.append("  Breakout Distance from OR High:")
    dists = []
    for t in or_trades:
        orh = float(_meta(t, "or_high", 0))
        if orh > 0 and t.risk_per_share > 0:
            dists.append((t.entry_price - orh) / t.risk_per_share)
        else:
            dists.append(0.0)

    if dists:
        p50 = float(np.median(dists))
        near = [t for t, d in zip(or_trades, dists) if d <= p50]
        far = [t for t, d in zip(or_trades, dists) if d > p50]
        lines.append(f"    Near OR (<=P50={p50:.2f}R):")
        lines.append("  " + _group_stats(near))
        lines.append(f"    Far from OR (>P50):")
        lines.append("  " + _group_stats(far))

    return "\n".join(lines)


def _s07_rvol_at_entry(trades: list[TradeRecord]) -> str:
    """Section 7: RVOL at entry — does higher relative volume predict edge?"""
    lines = [_hdr("7. RVOL at Entry")]
    if not _has_momentum_metadata(trades):
        lines.append("  (no metadata — run with enriched engine)")
        return "\n".join(lines)

    rvol_trades = [(t, _rvol(t)) for t in trades if _rvol(t) > 0]
    if not rvol_trades:
        lines.append("  (no RVOL data in metadata)")
        return "\n".join(lines)

    buckets = [
        (0.0, 1.5, "1.0-1.5"),
        (1.5, 2.0, "1.5-2.0"),
        (2.0, 3.0, "2.0-3.0"),
        (3.0, 999, "3.0+"),
    ]
    for lo, hi, label in buckets:
        group = [t for t, rv in rvol_trades if lo <= rv < hi]
        if group:
            avg_rv = float(np.mean([rv for t, rv in rvol_trades if lo <= rv < hi]))
            lines.append(f"  RVOL {label} (avg={avg_rv:.2f}):")
            lines.append(_group_stats(group))

    return "\n".join(lines)


def _s08_avwap_distance(trades: list[TradeRecord]) -> str:
    """Section 8: AVWAP distance at entry — premium/discount vs session VWAP."""
    lines = [_hdr("8. AVWAP Distance at Entry")]
    if not _has_momentum_metadata(trades):
        lines.append("  (no metadata — run with enriched engine)")
        return "\n".join(lines)

    avwap_trades = [
        (t, float(_meta(t, "avwap_at_entry", 0)))
        for t in trades
        if _meta(t, "avwap_at_entry", 0) and float(_meta(t, "avwap_at_entry", 0)) > 0
    ]
    if not avwap_trades:
        lines.append("  (no AVWAP data in metadata)")
        return "\n".join(lines)

    # Distance as % of AVWAP
    dists = []
    for t, avwap in avwap_trades:
        pct = (t.entry_price - avwap) / avwap * 100
        dists.append((t, pct))

    lines.append("  Distance = (entry - AVWAP) / AVWAP × 100")
    lines.append("")

    buckets = [
        (-999, -0.5, "Below AVWAP (< -0.5%)"),
        (-0.5, 0.0, "Slight discount (-0.5% to 0%)"),
        (0.0, 0.5, "Slight premium (0% to +0.5%)"),
        (0.5, 1.0, "Premium (+0.5% to +1.0%)"),
        (1.0, 999, "Extended (> +1.0%)"),
    ]
    for lo, hi, label in buckets:
        group = [t for t, d in dists if lo <= d < hi]
        if group:
            avg_d = float(np.mean([d for t, d in dists if lo <= d < hi]))
            lines.append(f"  {label} (avg={avg_d:+.2f}%):")
            lines.append(_group_stats(group))

    return "\n".join(lines)


def _s09_entry_bar_timing(trades: list[TradeRecord]) -> str:
    """Section 9: Entry bar timing — which 5m bar triggers entry? Earlier=better?"""
    lines = [_hdr("9. Entry Bar Timing")]

    by_bar: dict[int, list[TradeRecord]] = {}
    for t in trades:
        bar = _entry_bar_number(t)
        by_bar.setdefault(bar, []).append(t)

    if not by_bar:
        return "\n".join(lines)

    # Group into meaningful windows
    windows = [
        (1, 6, "9:30-10:00 (bars 1-6)"),
        (7, 12, "10:00-10:30 (bars 7-12)"),
        (13, 24, "10:30-11:30 (bars 13-24)"),
        (25, 48, "11:30-13:30 (bars 25-48)"),
        (49, 78, "13:30-16:00 (bars 49-78)"),
    ]

    for lo, hi, label in windows:
        group = [t for bar_n, ts in by_bar.items() for t in ts if lo <= bar_n <= hi]
        if group:
            lines.append(f"  {label}:")
            lines.append(_group_stats(group))

    # Top 5 individual bars by count
    lines.append("")
    lines.append("  Top 5 entry bars:")
    top_bars = sorted(by_bar.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for bar_n, group in top_bars:
        t0 = 570 + (bar_n - 1) * 5
        h, m = divmod(t0, 60)
        wr = sum(1 for t in group if t.is_winner) / len(group)
        mean_r = float(np.mean([t.r_multiple for t in group]))
        lines.append(f"    Bar {bar_n} ({h}:{m:02d}): n={len(group)}, WR={wr:.0%}, Mean R={mean_r:+.3f}")

    return "\n".join(lines)


def _s10_regime_sizing(trades: list[TradeRecord]) -> str:
    """Section 10: Regime sizing impact — Tier A vs B performance and dollar contribution."""
    lines = [_hdr("10. Regime Sizing Impact")]

    by_tier: dict[str, list[TradeRecord]] = {}
    for t in trades:
        tier = t.regime_tier or "UNKNOWN"
        by_tier.setdefault(tier, []).append(t)

    for tier in sorted(by_tier):
        group = by_tier[tier]
        n = len(group)
        dollar_pnl = sum(t.pnl_net for t in group)
        avg_pos_value = float(np.mean([t.quantity * t.entry_price for t in group]))
        lines.append(f"  Tier {tier}:")
        lines.append(_group_stats(group))
        lines.append(f"      Dollar PnL: ${dollar_pnl:,.2f}  |  Avg position: ${avg_pos_value:,.0f}")

    # Sizing-adjusted comparison: what if all tiers had equal sizing?
    if len(by_tier) >= 2:
        lines.append("")
        lines.append("  Sizing-Adjusted (equal $1 exposure):")
        for tier in sorted(by_tier):
            group = by_tier[tier]
            norm_rs = []
            for t in group:
                pos_val = t.quantity * t.entry_price
                if pos_val > 0:
                    norm_rs.append(t.pnl_net / pos_val)
            if norm_rs:
                avg_ret = float(np.mean(norm_rs)) * 100
                lines.append(f"    Tier {tier}: avg return = {avg_ret:+.3f}% per trade (n={len(group)})")

    return "\n".join(lines)


def _s11_carry_analysis(trades: list[TradeRecord]) -> str:
    """Section 11: Carry analysis — overnight hold edge."""
    lines = [_hdr("11. Carry Analysis")]

    intraday = [t for t in trades if _carry_days(t) == 0]
    carries = [t for t in trades if _carry_days(t) > 0]

    lines.append(f"  Intraday (carry_days=0):")
    lines.append(_group_stats(intraday))
    lines.append(f"  Overnight (carry_days>0):")
    lines.append(_group_stats(carries))

    if carries:
        # Break down by carry count
        lines.append("")
        lines.append("  By carry duration:")
        by_cd: dict[int, list[TradeRecord]] = {}
        for t in carries:
            cd = min(_carry_days(t), 5)  # cap display at 5+
            by_cd.setdefault(cd, []).append(t)

        for cd in sorted(by_cd):
            label = f"{cd}d" if cd < 5 else "5d+"
            lines.append(f"    {label}:")
            lines.append("  " + _group_stats(by_cd[cd]))

        # Overnight edge: compare R accrual
        if intraday:
            intra_mean = float(np.mean([t.r_multiple for t in intraday]))
            carry_mean = float(np.mean([t.r_multiple for t in carries]))
            lines.append("")
            lines.append(f"  Overnight edge: {carry_mean - intra_mean:+.3f}R per trade")
            lines.append(f"    Intraday mean: {intra_mean:+.3f}R  |  Carry mean: {carry_mean:+.3f}R")

    return "\n".join(lines)


def _s12_regime_x_entry(trades: list[TradeRecord]) -> str:
    """Section 12: Regime tier × Entry type crosstab."""
    lines = [_hdr("12. Regime × Entry Type")]

    regimes = sorted(set(t.regime_tier or "?" for t in trades))
    entry_types = sorted(set(t.entry_type or "?" for t in trades))

    # Header
    hdr = f"  {'':15s}"
    for et in entry_types:
        hdr += f" {et[:12]:>12s}"
    lines.append(hdr)
    lines.append("  " + "-" * (15 + 13 * len(entry_types)))

    for regime in regimes:
        row = f"  {regime:<15s}"
        for et in entry_types:
            group = [t for t in trades if (t.regime_tier or "?") == regime and (t.entry_type or "?") == et]
            if group:
                mean_r = float(np.mean([t.r_multiple for t in group]))
                row += f" {mean_r:>+8.3f}({len(group):>2})"
            else:
                row += f" {'--':>12s}"
        lines.append(row)

    return "\n".join(lines)


def _s13_exit_reason(trades: list[TradeRecord]) -> str:
    """Section 13: Exit reason deep dive."""
    lines = [_hdr("13. Exit Reason Deep Dive")]

    by_reason: dict[str, list[TradeRecord]] = {}
    for t in trades:
        reason = _normalize_exit(t.exit_reason or "UNKNOWN")
        by_reason.setdefault(reason, []).append(t)

    total = len(trades)
    for reason in sorted(by_reason):
        group = by_reason[reason]
        n = len(group)
        pct = n / total if total > 0 else 0
        avg_hold = float(np.mean([_hold_hours(t) for t in group]))
        lines.append(f"  {reason} ({n}, {pct:.0%}):")
        lines.append(_group_stats(group))
        lines.append(f"      Avg hold: {avg_hold:.1f}h")

        # Stop hit R distribution
        if reason == "STOP_HIT":
            stop_rs = [t.r_multiple for t in group]
            lines.append(f"      R range: [{min(stop_rs):+.2f}, {max(stop_rs):+.2f}]")
            lines.append(f"      Mean stop R: {float(np.mean(stop_rs)):+.3f}")

    return "\n".join(lines)


def _s14_partial_take(trades: list[TradeRecord]) -> str:
    """Section 14: Partial take analysis — does taking partials help?"""
    lines = [_hdr("14. Partial Take Analysis")]
    if not _has_momentum_metadata(trades):
        lines.append("  (no metadata — run with enriched engine)")
        return "\n".join(lines)

    took_partial = [t for t in trades if _meta(t, "partial_taken", False)]
    no_partial = [t for t in trades if not _meta(t, "partial_taken", False)]

    n = len(trades)
    lines.append(f"  Partial taken: {len(took_partial)}/{n} ({len(took_partial)/n:.0%})" if n else "  N/A")
    lines.append("")

    lines.append(f"  With partial:")
    lines.append(_group_stats(took_partial))
    lines.append(f"  Without partial:")
    lines.append(_group_stats(no_partial))

    if took_partial and no_partial:
        # Compare: did partials help?
        partial_mean = float(np.mean([t.r_multiple for t in took_partial]))
        no_partial_mean = float(np.mean([t.r_multiple for t in no_partial]))
        lines.append("")
        lines.append(f"  Partial edge: {partial_mean - no_partial_mean:+.3f}R per trade")

        # MFE comparison for partial vs non-partial
        partial_mfes = [_meta(t, "mfe_r", 0) or 0 for t in took_partial]
        no_partial_mfes = [_meta(t, "mfe_r", 0) or 0 for t in no_partial]
        if any(m > 0 for m in partial_mfes):
            lines.append(
                f"  Avg MFE — partial: {float(np.mean(partial_mfes)):.3f}R  |  "
                f"no partial: {float(np.mean(no_partial_mfes)):.3f}R"
            )

    return "\n".join(lines)


def _s15_mfe_mae(trades: list[TradeRecord]) -> str:
    """Section 15: MFE/MAE analysis (fixed gate check)."""
    lines = [_hdr("15. MFE / MAE Analysis")]
    if not _has_momentum_metadata(trades):
        lines.append("  (no metadata — run with enriched engine)")
        return "\n".join(lines)

    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    if winners:
        mfe_rs = [_meta(t, "mfe_r", 0) or 0 for t in winners]
        actual_rs = [t.r_multiple for t in winners]
        # Capture efficiency: how much of MFE was captured
        efficiencies = [a / m if m > 0 else 0 for a, m in zip(actual_rs, mfe_rs)]
        lines.append(f"  Winners ({len(winners)}):")
        lines.append(f"    Mean MFE: {float(np.mean(mfe_rs)):.3f}R")
        lines.append(f"    Capture efficiency (R/MFE): {float(np.mean(efficiencies)):.1%}")
        lines.append(f"    Mean giveback (MFE-R): {float(np.mean([m-a for m, a in zip(mfe_rs, actual_rs)])):.3f}R")

        # MFE distribution
        if len(mfe_rs) >= 4:
            p25, p50, p75 = np.percentile(mfe_rs, [25, 50, 75])
            lines.append(f"    MFE distribution: P25={p25:.2f}R, P50={p50:.2f}R, P75={p75:.2f}R")

    if losers:
        mae_rs = [_meta(t, "mae_r", 0) or 0 for t in losers]
        mfe_losers = [_meta(t, "mfe_r", 0) or 0 for t in losers]
        lines.append(f"  Losers ({len(losers)}):")
        lines.append(f"    Mean MAE: {float(np.mean(mae_rs)):.3f}R")
        if len(mae_rs) >= 4:
            p25, p50, p75 = np.percentile(mae_rs, [25, 50, 75])
            lines.append(f"    MAE distribution: P25={p25:.2f}R, P50={p50:.2f}R, P75={p75:.2f}R")
        # How many losers had positive MFE first?
        had_mfe = [t for t, mfe in zip(losers, mfe_losers) if mfe > 0.3]
        if had_mfe:
            lines.append(f"    Losers with MFE > 0.3R first: {len(had_mfe)} ({len(had_mfe)/len(losers):.0%})")
            avg_mfe_then_loss = float(np.mean([_meta(t, "mfe_r", 0) or 0 for t in had_mfe]))
            lines.append(f"    Those trades avg MFE: {avg_mfe_then_loss:.3f}R before reversing")

    return "\n".join(lines)


def _s16_eod_hold(trades: list[TradeRecord]) -> str:
    """Section 16: EOD hold quality — what predicts EOD_FLATTEN success?"""
    lines = [_hdr("16. EOD Hold Quality")]

    eod_trades = [t for t in trades if _normalize_exit(t.exit_reason) == "EOD_FLATTEN"]
    if not eod_trades:
        lines.append("  No EOD_FLATTEN exits")
        return "\n".join(lines)

    winners = [t for t in eod_trades if t.is_winner]
    losers = [t for t in eod_trades if not t.is_winner]

    lines.append(f"  EOD flattens: {len(eod_trades)} ({len(eod_trades)/len(trades):.0%} of all)")
    lines.append(_group_stats(eod_trades))
    lines.append("")

    lines.append(f"  EOD Winners:")
    lines.append(_group_stats(winners))
    lines.append(f"  EOD Losers:")
    lines.append(_group_stats(losers))

    if _has_momentum_metadata(trades) and winners and losers:
        # What differentiates EOD winners from losers?
        lines.append("")
        lines.append("  Differentiators (EOD winners vs losers):")

        comparisons = [
            ("Momentum Score", lambda ts: float(np.mean([_meta(t, "momentum_score", 0) or 0 for t in ts]))),
            ("RVOL", lambda ts: float(np.mean([_rvol(t) for t in ts]))),
            ("MFE (R)", lambda ts: float(np.mean([_meta(t, "mfe_r", 0) or 0 for t in ts]))),
            ("Hold Bars", lambda ts: float(np.mean([t.hold_bars for t in ts]))),
            ("Carry Days", lambda ts: float(np.mean([_carry_days(t) for t in ts]))),
        ]

        lines.append(f"    {'Metric':<20s} {'Winners':>10s} {'Losers':>10s}")
        lines.append("    " + "-" * 42)
        for name, fn in comparisons:
            w_val = fn(winners)
            l_val = fn(losers)
            lines.append(f"    {name:<20s} {w_val:>10.2f} {l_val:>10.2f}")

    return "\n".join(lines)


def _s17_stale_exit(trades: list[TradeRecord]) -> str:
    """Section 17: Stale exit analysis."""
    lines = [_hdr("17. Stale Exit Analysis")]
    stales = [t for t in trades if _normalize_exit(t.exit_reason) == "STALE_EXIT"]
    if not stales:
        lines.append("  No stale exits")
        return "\n".join(lines)

    lines.append(f"  Stale exits: {len(stales)} ({len(stales)/len(trades):.1%} of all)")
    lines.append(_group_stats(stales))

    # Entry type mix of stale exits
    by_et: dict[str, int] = {}
    for t in stales:
        et = t.entry_type or "UNKNOWN"
        by_et[et] = by_et.get(et, 0) + 1
    lines.append(f"  Entry types: {by_et}")

    avg_hold = float(np.mean([_hold_hours(t) for t in stales]))
    lines.append(f"  Avg hold: {avg_hold:.1f}h")

    profitable = sum(1 for t in stales if t.is_winner)
    lines.append(f"  Profitable stales: {profitable}/{len(stales)} ({profitable/len(stales):.0%})")

    return "\n".join(lines)


def _s18_hold_duration(trades: list[TradeRecord]) -> str:
    """Section 18: Hold duration analysis."""
    lines = [_hdr("18. Hold Duration")]
    hours = [_hold_hours(t) for t in trades]
    if not hours:
        return "\n".join(lines)

    # Decile table
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    boundaries = np.percentile(hours, percentiles)

    lines.append(f"  {'Pctl':>6s} {'Hours':>8s} {'WR%':>6s} {'Avg R':>7s} {'N':>5s}")
    lines.append("  " + "-" * 36)

    prev_bound = 0
    for pctl, bound in zip(percentiles, boundaries):
        if pctl < 100:
            group = [t for t, h in zip(trades, hours) if prev_bound <= h < bound]
        else:
            group = [t for t, h in zip(trades, hours) if h >= prev_bound]
        if group:
            wr = sum(1 for t in group if t.is_winner) / len(group)
            avg_r = float(np.mean([t.r_multiple for t in group]))
            lines.append(f"  P{pctl:>4} {bound:>8.1f} {wr:>5.0%} {avg_r:>+7.3f} {len(group):>5}")
        prev_bound = bound

    return "\n".join(lines)


def _s19_flow_reversal(trades: list[TradeRecord]) -> str:
    """Section 19: Flow reversal timing — early/mid/late FR cost."""
    lines = [_hdr("19. Flow Reversal Timing")]

    fr_trades = [t for t in trades if _normalize_exit(t.exit_reason) == "FLOW_REVERSAL"]
    if not fr_trades:
        lines.append("  No FLOW_REVERSAL exits")
        return "\n".join(lines)

    lines.append(f"  Flow reversals: {len(fr_trades)} ({len(fr_trades)/len(trades):.0%} of all)")
    lines.append(_group_stats(fr_trades))
    lines.append("")

    # Group by hold_bars timing
    early = [t for t in fr_trades if t.hold_bars <= 6]
    mid = [t for t in fr_trades if 6 < t.hold_bars <= 24]
    late = [t for t in fr_trades if t.hold_bars > 24]

    lines.append(f"  Early FR (<=6 bars, <30min):")
    lines.append(_group_stats(early))
    lines.append(f"  Mid FR (7-24 bars, 30min-2h):")
    lines.append(_group_stats(mid))
    lines.append(f"  Late FR (>24 bars, 2h+):")
    lines.append(_group_stats(late))

    # Cost analysis
    total_fr_r = sum(t.r_multiple for t in fr_trades)
    total_fr_pnl = sum(t.pnl_net for t in fr_trades)
    lines.append("")
    lines.append(f"  Total FR cost: {total_fr_r:+.2f}R (${total_fr_pnl:+,.2f})")

    if early:
        early_cost = sum(t.r_multiple for t in early)
        lines.append(f"  Early FR cost: {early_cost:+.2f}R ({len(early)} trades)")

    # Compare: FR trades that had positive MFE first
    if _has_momentum_metadata(trades):
        had_mfe = [t for t in fr_trades if (_meta(t, "mfe_r", 0) or 0) > 0.3]
        if had_mfe:
            lines.append("")
            lines.append(f"  FR trades with MFE > 0.3R before reversal: {len(had_mfe)}")
            avg_mfe = float(np.mean([_meta(t, "mfe_r", 0) or 0 for t in had_mfe]))
            lines.append(f"    Avg MFE before FR: {avg_mfe:.3f}R (profit left on table)")

    return "\n".join(lines)


def _s20_sector(trades: list[TradeRecord]) -> str:
    """Section 20: Sector breakdown."""
    lines = [_hdr("20. Sector Performance")]
    by_sector: dict[str, list[TradeRecord]] = {}
    for t in trades:
        sec = t.sector or "UNKNOWN"
        by_sector.setdefault(sec, []).append(t)

    sorted_sectors = sorted(by_sector.items(), key=lambda x: sum(t.r_multiple for t in x[1]), reverse=True)
    for sec, group in sorted_sectors:
        lines.append(f"  {sec}:")
        lines.append(_group_stats(group))

    return "\n".join(lines)


def _s21_monthly_pnl(trades: list[TradeRecord]) -> str:
    """Section 21: Monthly P&L table."""
    lines = [_hdr("21. Monthly P&L")]
    by_month: dict[str, list[TradeRecord]] = {}
    for t in trades:
        month = t.exit_time.strftime("%Y-%m")
        by_month.setdefault(month, []).append(t)

    lines.append(f"  {'Month':>8s} {'Trades':>6s} {'WR%':>5s} {'Net R':>8s} {'Cum R':>8s}")
    lines.append("  " + "-" * 40)

    cum_r = 0.0
    for month in sorted(by_month):
        group = by_month[month]
        n = len(group)
        wr = sum(1 for t in group if t.is_winner) / n if n else 0
        net_r = sum(t.r_multiple for t in group)
        cum_r += net_r
        lines.append(f"  {month:>8s} {n:>6} {wr:>4.0%} {net_r:>+8.2f} {cum_r:>+8.2f}")

    return "\n".join(lines)


def _s22_day_of_week(trades: list[TradeRecord]) -> str:
    """Section 22: Day of week analysis."""
    lines = [_hdr("22. Day of Week")]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    by_day: dict[int, list[TradeRecord]] = {}
    for t in trades:
        dow = _entry_time_et(t).weekday()
        by_day.setdefault(dow, []).append(t)

    for i, name in enumerate(days):
        group = by_day.get(i, [])
        if group:
            lines.append(f"  {name}:")
            lines.append(_group_stats(group))

    return "\n".join(lines)


def _s23_time_of_day(trades: list[TradeRecord]) -> str:
    """Section 23: Time of day entry analysis."""
    lines = [_hdr("23. Time of Day")]
    buckets = [
        (10.0, 11.5, "10:00-11:30"),
        (11.5, 13.0, "11:30-13:00"),
        (13.0, 14.5, "13:00-14:30"),
        (14.5, 16.0, "14:30-16:00"),
    ]

    for lo, hi, label in buckets:
        group = [
            t
            for t in trades
            if lo <= _entry_time_et(t).hour + _entry_time_et(t).minute / 60 < hi
        ]
        if group:
            lines.append(f"  {label}:")
            lines.append(_group_stats(group))

    # Catch any outside the expected windows
    outside = [
        t
        for t in trades
        if _entry_time_et(t).hour < 10 or _entry_time_et(t).hour >= 16
    ]
    if outside:
        lines.append(f"  Outside window:")
        lines.append(_group_stats(outside))

    return "\n".join(lines)


def _s24_streak(trades: list[TradeRecord]) -> str:
    """Section 24: Streak analysis."""
    lines = [_hdr("24. Streak Analysis")]
    if not trades:
        return "\n".join(lines)

    # Win/loss sequences
    results = [t.is_winner for t in trades]
    max_win = max_loss = cur_win = cur_loss = 0
    worst_consec_loss_r = 0.0
    cur_loss_r = 0.0

    for i, is_win in enumerate(results):
        if is_win:
            cur_win += 1
            cur_loss = 0
            cur_loss_r = 0.0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            cur_loss_r += trades[i].r_multiple
            max_loss = max(max_loss, cur_loss)
            worst_consec_loss_r = min(worst_consec_loss_r, cur_loss_r)

    lines.append(f"  Max win streak:  {max_win}")
    lines.append(f"  Max loss streak: {max_loss}")
    lines.append(f"  Worst consecutive loss: {worst_consec_loss_r:+.2f}R")

    return "\n".join(lines)


def _s25_rolling_expectancy(trades: list[TradeRecord]) -> str:
    """Section 25: Rolling 20-trade expectancy."""
    lines = [_hdr("25. Rolling Expectancy (20-trade window)")]
    if len(trades) < 20:
        lines.append("  Insufficient trades for rolling analysis")
        return "\n".join(lines)

    rs = [t.r_multiple for t in trades]
    window = 20
    rolling = [float(np.mean(rs[i:i+window])) for i in range(len(rs) - window + 1)]

    lines.append(f"  Start: {rolling[0]:+.3f}R  |  End: {rolling[-1]:+.3f}R")
    lines.append(f"  Min: {min(rolling):+.3f}R  |  Max: {max(rolling):+.3f}R")

    # Trend detection
    first_half = float(np.mean(rolling[:len(rolling)//2]))
    second_half = float(np.mean(rolling[len(rolling)//2:]))
    if second_half > first_half + 0.05:
        trend = "IMPROVING"
    elif second_half < first_half - 0.05:
        trend = "DEGRADING"
    else:
        trend = "STABLE"
    lines.append(f"  Trend: {trend} (1st half: {first_half:+.3f}R, 2nd half: {second_half:+.3f}R)")

    return "\n".join(lines)


def _s26_winner_loser_profiles(trades: list[TradeRecord]) -> str:
    """Section 26: Winner vs Loser profiles with momentum metrics."""
    lines = [_hdr("26. Winner vs Loser Profiles")]
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    if not winners or not losers:
        lines.append("  Need both winners and losers for comparison")
        return "\n".join(lines)

    def _avg_meta(ts: list[TradeRecord], key: str, default=0):
        vals = [_meta(t, key, default) or default for t in ts]
        return float(np.mean(vals)) if vals else 0

    lines.append(f"  {'Metric':<25s} {'Winners':>10s} {'Losers':>10s} {'Delta':>10s}")
    lines.append("  " + "-" * 58)

    metrics: list[tuple[str, object]] = [
        ("Avg R", lambda ts: float(np.mean([t.r_multiple for t in ts]))),
        ("Avg Hold (h)", lambda ts: float(np.mean([_hold_hours(t) for t in ts]))),
        ("Hold Bars", lambda ts: float(np.mean([t.hold_bars for t in ts]))),
    ]

    if _has_momentum_metadata(trades):
        metrics.extend([
            ("Momentum Score", lambda ts: _avg_meta(ts, "momentum_score", 0)),
            ("RVOL", lambda ts: float(np.mean([_rvol(t) for t in ts]))),
            ("MFE (R)", lambda ts: _avg_meta(ts, "mfe_r", 0)),
            ("MAE (R)", lambda ts: _avg_meta(ts, "mae_r", 0)),
            ("Carry Days", lambda ts: float(np.mean([_carry_days(t) for t in ts]))),
        ])

    for name, fn in metrics:
        w_val = fn(winners)
        l_val = fn(losers)
        delta = w_val - l_val
        lines.append(f"  {name:<25s} {w_val:>10.3f} {l_val:>10.3f} {delta:>+10.3f}")

    return "\n".join(lines)


def _s27_drawdown(trades: list[TradeRecord]) -> str:
    """Section 27: Drawdown profile."""
    lines = [_hdr("27. Drawdown Profile")]
    if not trades:
        return "\n".join(lines)

    # Build cumulative R curve
    rs = [t.r_multiple for t in trades]
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak

    max_dd = float(np.min(dd))
    max_dd_idx = int(np.argmin(dd))

    lines.append(f"  Max drawdown: {max_dd:+.2f}R (at trade #{max_dd_idx + 1})")

    # Count DD episodes (consecutive negative DD)
    in_dd = False
    episodes = 0
    max_underwater = 0
    current_underwater = 0
    for d in dd:
        if d < 0:
            if not in_dd:
                in_dd = True
                episodes += 1
                current_underwater = 0
            current_underwater += 1
            max_underwater = max(max_underwater, current_underwater)
        else:
            in_dd = False
            current_underwater = 0

    lines.append(f"  DD episodes: {episodes}")
    lines.append(f"  Max trades underwater: {max_underwater}")

    # Recovery
    if max_dd < 0:
        recovery_trades = 0
        for i in range(max_dd_idx, len(cum)):
            if cum[i] >= peak[max_dd_idx]:
                recovery_trades = i - max_dd_idx
                break
        if recovery_trades > 0:
            lines.append(f"  Recovery from worst DD: {recovery_trades} trades")
        else:
            lines.append(f"  Recovery from worst DD: not recovered")

    return "\n".join(lines)


def _s28_r_vs_dollar(trades: list[TradeRecord]) -> str:
    """Section 28: R vs Dollar disconnect — why PnL > 0 but total R < 0."""
    lines = [_hdr("28. R vs Dollar Disconnect")]

    total_r = sum(t.r_multiple for t in trades)
    total_pnl = sum(t.pnl_net for t in trades)
    n = len(trades)

    lines.append(f"  Total R:   {total_r:+.2f}  (R-expectancy: {total_r/n:+.4f})")
    lines.append(f"  Total PnL: ${total_pnl:+,.2f}  ($-expectancy: ${total_pnl/n:+,.2f})")
    lines.append("")

    # Dollar-weighted expectancy
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    if winners and losers:
        w_avg_pos = float(np.mean([t.quantity * t.entry_price for t in winners]))
        l_avg_pos = float(np.mean([t.quantity * t.entry_price for t in losers]))
        w_avg_pnl = float(np.mean([t.pnl_net for t in winners]))
        l_avg_pnl = float(np.mean([t.pnl_net for t in losers]))
        w_avg_r = float(np.mean([t.r_multiple for t in winners]))
        l_avg_r = float(np.mean([t.r_multiple for t in losers]))

        lines.append(f"  {'':20s} {'Winners':>12s} {'Losers':>12s}")
        lines.append("  " + "-" * 46)
        lines.append(f"  {'Avg Position $':<20s} ${w_avg_pos:>10,.0f} ${l_avg_pos:>10,.0f}")
        lines.append(f"  {'Avg PnL $':<20s} ${w_avg_pnl:>10,.2f} ${l_avg_pnl:>10,.2f}")
        lines.append(f"  {'Avg R':<20s} {w_avg_r:>+12.3f} {l_avg_r:>+12.3f}")
        lines.append(f"  {'Count':<20s} {len(winners):>12} {len(losers):>12}")

        # Explain the gap
        if total_r < 0 and total_pnl > 0:
            lines.append("")
            lines.append("  EXPLANATION: Positive PnL with negative R indicates")
            lines.append("  winners have larger position sizes than losers.")
            size_ratio = w_avg_pos / l_avg_pos if l_avg_pos > 0 else 0
            lines.append(f"  Winner/Loser size ratio: {size_ratio:.2f}x")
        elif total_r > 0 and total_pnl < 0:
            lines.append("")
            lines.append("  EXPLANATION: Positive R with negative PnL indicates")
            lines.append("  losers have larger position sizes than winners.")

    # Risk dollars per trade
    risk_dollars = [_meta(t, "risk_dollars", 0) or 0 for t in trades]
    if any(r > 0 for r in risk_dollars):
        lines.append("")
        lines.append(f"  Avg risk per trade: ${float(np.mean([r for r in risk_dollars if r > 0])):,.2f}")

    return "\n".join(lines)


def _s29_worst_periods(trades: list[TradeRecord]) -> str:
    """Section 29: Worst period autopsy — top 5 worst months dissected."""
    lines = [_hdr("29. Worst Period Autopsy")]

    by_month: dict[str, list[TradeRecord]] = {}
    for t in trades:
        month = t.exit_time.strftime("%Y-%m")
        by_month.setdefault(month, []).append(t)

    if len(by_month) < 2:
        lines.append("  Insufficient months for analysis")
        return "\n".join(lines)

    # Rank by total R, take worst 5
    ranked = sorted(by_month.items(), key=lambda x: sum(t.r_multiple for t in x[1]))
    worst_5 = ranked[:5]

    for month, group in worst_5:
        n = len(group)
        total_r = sum(t.r_multiple for t in group)
        total_pnl = sum(t.pnl_net for t in group)
        wr = sum(1 for t in group if t.is_winner) / n if n else 0

        lines.append(f"  {month}: {total_r:+.2f}R (${total_pnl:+,.2f}), n={n}, WR={wr:.0%}")

        # Entry type mix
        et_counts: dict[str, int] = {}
        for t in group:
            et = t.entry_type or "?"
            et_counts[et] = et_counts.get(et, 0) + 1
        et_str = ", ".join(f"{k}:{v}" for k, v in sorted(et_counts.items(), key=lambda x: -x[1]))
        lines.append(f"    Entry types: {et_str}")

        # Exit type mix
        ex_counts: dict[str, int] = {}
        for t in group:
            ex = _normalize_exit(t.exit_reason or "?")
            ex_counts[ex] = ex_counts.get(ex, 0) + 1
        ex_str = ", ".join(f"{k}:{v}" for k, v in sorted(ex_counts.items(), key=lambda x: -x[1]))
        lines.append(f"    Exit types: {ex_str}")

        # Sector concentration
        sec_counts: dict[str, int] = {}
        for t in group:
            sec = t.sector or "?"
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
        top_sec = sorted(sec_counts.items(), key=lambda x: -x[1])[:3]
        sec_str = ", ".join(f"{k}:{v}" for k, v in top_sec)
        lines.append(f"    Top sectors: {sec_str}")
        lines.append("")

    return "\n".join(lines)


def _s30_intraday_alpha(trades: list[TradeRecord]) -> str:
    """Section 30: Intraday alpha curve — where in the day does alpha accrue?"""
    lines = [_hdr("30. Intraday Alpha Curve")]

    if not trades:
        return "\n".join(lines)

    # Group by hold_bars bucket
    buckets = [
        (1, 6, "1-6 bars (0-30min)"),
        (7, 12, "7-12 bars (30min-1h)"),
        (13, 24, "13-24 bars (1-2h)"),
        (25, 48, "25-48 bars (2-4h)"),
        (49, 78, "49-78 bars (4-6.5h)"),
        (79, 999, "79+ bars (overnight+)"),
    ]

    lines.append(f"  {'Hold Bucket':<25s} {'N':>5s} {'WR%':>5s} {'Mean R':>8s} {'Total R':>8s} {'$/trade':>10s}")
    lines.append("  " + "-" * 65)

    for lo, hi, label in buckets:
        group = [t for t in trades if lo <= t.hold_bars <= hi]
        if group:
            n = len(group)
            wr = sum(1 for t in group if t.is_winner) / n
            mean_r = float(np.mean([t.r_multiple for t in group]))
            total_r = sum(t.r_multiple for t in group)
            avg_pnl = float(np.mean([t.pnl_net for t in group]))
            lines.append(
                f"  {label:<25s} {n:>5} {wr:>4.0%} {mean_r:>+8.3f} {total_r:>+8.2f} ${avg_pnl:>9,.2f}"
            )

    # Identify where bulk of alpha comes from
    short_hold = [t for t in trades if t.hold_bars <= 24]
    long_hold = [t for t in trades if t.hold_bars > 24]
    if short_hold and long_hold:
        short_r = sum(t.r_multiple for t in short_hold)
        long_r = sum(t.r_multiple for t in long_hold)
        lines.append("")
        lines.append(
            f"  Short holds (<=24 bars): {short_r:+.2f}R ({len(short_hold)} trades)  |  "
            f"Long holds (>24 bars): {long_r:+.2f}R ({len(long_hold)} trades)"
        )
        if long_r > short_r:
            lines.append("  → Late-session / overnight holds are the primary alpha source")
        elif short_r > long_r:
            lines.append("  → Early captures are the primary alpha source")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def alcb_full_diagnostic(
    trades: list[TradeRecord],
    shadow_tracker: ALCBShadowTracker | None = None,
    daily_selections: dict | None = None,
) -> str:
    """Generate the full 30-section ALCB diagnostic report.

    Parameters
    ----------
    trades : list[TradeRecord]
        Completed trades from ALCB backtest (with enriched metadata).
    shadow_tracker : ALCBShadowTracker, optional
        Shadow tracker with rejected setup simulations.
    daily_selections : dict, optional
        {date: CandidateArtifact} from engine result for funnel analysis.
    """
    if not trades:
        return "No trades to diagnose."

    sections = [
        _s01_overview(trades),
        _s02_signal_funnel(trades, daily_selections, shadow_tracker),
        _s03_entry_type(trades),
        _s04_direction(trades),
        _s05_momentum_score(trades),
        _s06_opening_range(trades),
        _s07_rvol_at_entry(trades),
        _s08_avwap_distance(trades),
        _s09_entry_bar_timing(trades),
        _s10_regime_sizing(trades),
        _s11_carry_analysis(trades),
        _s12_regime_x_entry(trades),
        _s13_exit_reason(trades),
        _s14_partial_take(trades),
        _s15_mfe_mae(trades),
        _s16_eod_hold(trades),
        _s17_stale_exit(trades),
        _s18_hold_duration(trades),
        _s19_flow_reversal(trades),
        _s20_sector(trades),
        _s21_monthly_pnl(trades),
        _s22_day_of_week(trades),
        _s23_time_of_day(trades),
        _s24_streak(trades),
        _s25_rolling_expectancy(trades),
        _s26_winner_loser_profiles(trades),
        _s27_drawdown(trades),
        _s28_r_vs_dollar(trades),
        _s29_worst_periods(trades),
        _s30_intraday_alpha(trades),
    ]

    # Append filter attribution if shadow tracker available
    if shadow_tracker and shadow_tracker.completed:
        actual_wr = sum(1 for t in trades if t.is_winner) / len(trades) if trades else 0
        sections.append(alcb_filter_attribution_report(shadow_tracker, actual_wr))

    return "\n".join(sections)
