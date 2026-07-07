"""Volatility regime analysis — performance segmented by VIX and ATR regimes.

Evaluates strategy edge persistence across different volatility environments
and quantifies regime-transition impact on P&L.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime

from backtests.momentum.analysis._utils import trade_date


VIX_BUCKETS = [
    ("< 15",  0.0, 15.0),
    ("15-20", 15.0, 20.0),
    ("20-25", 20.0, 25.0),
    ("25-30", 25.0, 30.0),
    ("30+",   30.0, 999.0),
]

ATR_PERCENTILE_BUCKETS = [
    ("Low (0-25)",     0, 25),
    ("Mid (25-50)",   25, 50),
    ("High (50-75)",  50, 75),
    ("Extreme (75+)", 75, 100),
]


def _trade_date(trade) -> str | None:
    """Extract YYYY-MM-DD from entry_time (thin wrapper over shared trade_date)."""
    d = trade_date(trade)
    return d.isoformat() if d else None


def _bucket_vix(v: float) -> str:
    for label, lo, hi in VIX_BUCKETS:
        if lo <= v < hi:
            return label
    return "30+"


def _compute_group_stats(trades: list) -> dict:
    """Return stats dict for a group of trades."""
    if not trades:
        return {"count": 0, "wr": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0, "avg_r": 0.0}
    pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in trades])
    r_vals = np.array([getattr(t, "r_multiple", getattr(t, "pnl_dollars", 0.0)) for t in trades])
    wins = pnl > 0
    return {
        "count": len(trades),
        "wr": float(np.mean(wins)),
        "avg_pnl": float(np.mean(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "avg_r": float(np.mean(r_vals)),
    }


def generate_volatility_regime_report(
    trades: list,
    vix_data: dict[str, float] | None = None,
    daily_bars: list | None = None,
    strategies: dict[str, list] | None = None,
) -> str:
    """Generate volatility regime performance report.

    Args:
        trades: Combined trade list.
        vix_data: Optional dict mapping "YYYY-MM-DD" -> VIX close.
        daily_bars: Optional list of daily bar objects with date, high, low, close
                    for computing ATR percentiles.
        strategies: Optional dict mapping strategy name -> trade list.

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  VOLATILITY REGIME REPORT")
    lines.append("=" * 72)
    lines.append("")

    if strategies is None:
        strategies = {"all": trades}

    if not any(strategies.values()):
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # ── Section A: VIX regime analysis ──
    lines.append("  A. VIX REGIME PERFORMANCE")
    lines.append("  " + "-" * 60)

    if vix_data:
        for strat_name, strat_trades in strategies.items():
            lines.append(f"\n    Strategy: {strat_name}")
            regime_trades: dict[str, list] = defaultdict(list)
            for t in strat_trades:
                dt = _trade_date(t)
                if dt and dt in vix_data:
                    bucket = _bucket_vix(vix_data[dt])
                    regime_trades[bucket].append(t)

            lines.append(f"    {'Regime':<12s} {'Count':>5s} {'WR%':>6s} {'AvgPnL':>10s} {'TotalPnL':>10s} {'AvgR':>7s}")
            lines.append("    " + "-" * 55)
            for label, _, _ in VIX_BUCKETS:
                s = _compute_group_stats(regime_trades[label])
                lines.append(
                    f"    {label:<12s} {s['count']:>5d} {s['wr']*100:>5.1f}% "
                    f"${s['avg_pnl']:>+9.0f} ${s['total_pnl']:>+9.0f} {s['avg_r']:>+6.2f}"
                )
    else:
        lines.append("    [VIX data not provided — skipping VIX analysis]")

    # ── Section B: ATR percentile analysis ──
    lines.append("")
    lines.append("  B. ATR PERCENTILE PERFORMANCE")
    lines.append("  " + "-" * 60)

    if daily_bars:
        # Compute 14-period ATR
        highs = np.array([getattr(b, "high", 0) for b in daily_bars])
        lows = np.array([getattr(b, "low", 0) for b in daily_bars])
        closes = np.array([getattr(b, "close", 0) for b in daily_bars])
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        atr_14 = np.convolve(tr, np.ones(14) / 14, mode="valid")
        bar_dates = [getattr(b, "date", None) for b in daily_bars]
        atr_dates = bar_dates[14:] if len(bar_dates) > 14 else bar_dates
        # Map date -> percentile rank
        atr_pcts = {}
        for i, val in enumerate(atr_14):
            d = atr_dates[i] if i < len(atr_dates) else None
            if d:
                pct = float(np.searchsorted(np.sort(atr_14[:i+1]), val)) / max(1, i + 1) * 100
                dstr = d.strftime("%Y-%m-%d") if isinstance(d, datetime) else str(d)
                atr_pcts[dstr] = pct

        for strat_name, strat_trades in strategies.items():
            lines.append(f"\n    Strategy: {strat_name}")
            pct_trades: dict[str, list] = defaultdict(list)
            for t in strat_trades:
                dt = _trade_date(t)
                if dt and dt in atr_pcts:
                    pct = atr_pcts[dt]
                    for label, lo, hi in ATR_PERCENTILE_BUCKETS:
                        if lo <= pct < hi or (hi == 100 and pct >= 75):
                            pct_trades[label].append(t)
                            break

            lines.append(f"    {'ATR Pctile':<16s} {'Count':>5s} {'WR%':>6s} {'AvgPnL':>10s} {'TotalPnL':>10s}")
            lines.append("    " + "-" * 45)
            for label, _, _ in ATR_PERCENTILE_BUCKETS:
                s = _compute_group_stats(pct_trades[label])
                lines.append(
                    f"    {label:<16s} {s['count']:>5d} {s['wr']*100:>5.1f}% "
                    f"${s['avg_pnl']:>+9.0f} ${s['total_pnl']:>+9.0f}"
                )
    else:
        lines.append("    [Daily bars not provided — skipping ATR analysis]")

    # ── Section C: Regime transition impact ──
    lines.append("")
    lines.append("  C. REGIME TRANSITION IMPACT")
    lines.append("  " + "-" * 60)

    if vix_data:
        sorted_dates = sorted(vix_data.keys())
        transitions: list[tuple[str, str, str]] = []
        for i in range(1, len(sorted_dates)):
            prev_b = _bucket_vix(vix_data[sorted_dates[i - 1]])
            curr_b = _bucket_vix(vix_data[sorted_dates[i]])
            if prev_b != curr_b:
                transitions.append((sorted_dates[i], prev_b, curr_b))

        transition_dates = {d for d, _, _ in transitions}
        trans_trades = [t for t in trades if _trade_date(t) in transition_dates]
        non_trans_trades = [t for t in trades if _trade_date(t) not in transition_dates]

        s_trans = _compute_group_stats(trans_trades)
        s_normal = _compute_group_stats(non_trans_trades)

        lines.append(f"    Regime transitions found:   {len(transitions)}")
        lines.append(f"    Trades on transition days:  {s_trans['count']}  (WR={s_trans['wr']*100:.1f}%, avg=${s_trans['avg_pnl']:+.0f})")
        lines.append(f"    Trades on normal days:      {s_normal['count']}  (WR={s_normal['wr']*100:.1f}%, avg=${s_normal['avg_pnl']:+.0f})")

        if s_trans["count"] > 0 and s_normal["count"] > 0:
            diff = s_trans["avg_pnl"] - s_normal["avg_pnl"]
            lines.append(f"    Transition-day delta:       ${diff:+.0f}/trade")
    else:
        lines.append("    [VIX data not provided — skipping transition analysis]")

    return "\n".join(lines)
