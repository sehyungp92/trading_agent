"""Swing unified weakness report — cross-strategy executive summary.

Synthesizes diagnostics from the active unified swing strategies into a single
actionable report with per-strategy scores and prioritized weaknesses.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict


def _normalize(value: float, baseline: float, ceiling: float) -> float:
    """Normalize a value to [0, 1] range between baseline and ceiling."""
    if ceiling == baseline:
        return 0.5
    raw = (value - baseline) / (ceiling - baseline)
    return max(0.0, min(1.0, raw))


def _score_strategy(trades: list, label: str = "") -> dict:
    """Compute per-strategy weakness score (0-10 scale).

    Scoring:
      0.25 * WR quality (baseline 0.45, ceiling 0.65)
      0.25 * Expectancy R (baseline 0.2, ceiling 1.0)
      0.20 * Rolling stability (baseline -0.1, ceiling 0.1)
      0.15 * DD control (baseline 0.7, ceiling 0.95)
      0.15 * Trade frequency (baseline 1.0/mo, ceiling 5.0/mo)
    """
    if not trades or len(trades) < 5:
        return {
            "label": label, "score": None, "verdict": "INSUFFICIENT",
            "trades": len(trades), "wr": 0, "avg_r": 0, "total_r": 0,
            "stability_slope": 0, "max_dd_r": 0, "trades_per_month": 0,
        }

    r_arr = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
    wr = float(np.mean(r_arr > 0))
    avg_r = float(np.mean(r_arr))
    total_r = float(np.sum(r_arr))

    # Rolling 20-trade expectancy stability
    window = min(20, len(r_arr) // 2)
    if window >= 5:
        rolling = [float(np.mean(r_arr[i:i+window])) for i in range(len(r_arr) - window + 1)]
        if len(rolling) >= 3:
            x = np.arange(len(rolling))
            slope = float(np.polyfit(x, rolling, 1)[0])
        else:
            slope = 0.0
    else:
        slope = 0.0

    # Max drawdown in R
    cum_r = np.cumsum(r_arr)
    peak_r = np.maximum.accumulate(cum_r)
    dd_r = cum_r - peak_r
    max_dd_r = float(np.min(dd_r))

    # Trades per month
    entry_times = [getattr(t, 'entry_time', None) for t in trades]
    valid_times = [t for t in entry_times if t is not None]
    if len(valid_times) >= 2:
        try:
            first = min(valid_times)
            last = max(valid_times)
            if hasattr(first, 'timestamp') and hasattr(last, 'timestamp'):
                months = max(1, (last.timestamp() - first.timestamp()) / (30 * 24 * 3600))
            else:
                import pandas as pd
                months = max(1, (pd.Timestamp(last) - pd.Timestamp(first)).total_seconds() / (30 * 24 * 3600))
            trades_per_month = len(trades) / months
        except Exception:
            trades_per_month = len(trades) / 12.0
    else:
        trades_per_month = len(trades) / 12.0

    # 1 - max_dd normalized (lower DD = better)
    dd_control = 1.0 + max_dd_r / max(abs(total_r), 1.0)  # closer to 1.0 = less DD
    dd_control = max(0.0, min(1.0, dd_control))

    score = (
        0.25 * _normalize(wr, 0.45, 0.65)
        + 0.25 * _normalize(avg_r, 0.2, 1.0)
        + 0.20 * _normalize(slope, -0.1, 0.1)
        + 0.15 * _normalize(dd_control, 0.7, 0.95)
        + 0.15 * _normalize(trades_per_month, 1.0, 5.0)
    ) * 10.0

    if score >= 7.0:
        verdict = "STRONG EDGE"
    elif score >= 5.0:
        verdict = "MODERATE EDGE"
    elif score >= 3.0:
        verdict = "WEAK EDGE"
    else:
        verdict = "NO EDGE"

    return {
        "label": label, "score": round(score, 1), "verdict": verdict,
        "trades": len(trades), "wr": wr, "avg_r": avg_r, "total_r": total_r,
        "stability_slope": slope, "max_dd_r": max_dd_r,
        "trades_per_month": trades_per_month,
    }


def swing_weakness_report(
    atrss_result=None,
    helix_result=None,
    portfolio_result=None,
    overlay_result=None,
    filter_verdicts: dict[str, dict[str, str]] | None = None,
) -> str:
    """Generate unified swing weakness report.

    Each *_result should have a .trades attribute (list of trade records).
    filter_verdicts: {strategy: {filter_name: "KEEP"|"REVIEW"}}
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  SWING WEAKNESS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Gather trades from each strategy
    strategies = {}
    for name, result in [
        ("ATRSS", atrss_result), ("Helix", helix_result),
    ]:
        if result is not None:
            trades = getattr(result, 'trades', [])
            if isinstance(trades, dict):
                # Some results have trades per symbol
                all_trades = []
                for v in trades.values():
                    if isinstance(v, list):
                        all_trades.extend(v)
                trades = all_trades
            strategies[name] = trades
        else:
            strategies[name] = []

    # ── Strategy-Level Verdicts ──
    lines.append("  STRATEGY-LEVEL VERDICTS")
    lines.append("  " + "-" * 50)

    scores = {}
    for name, trades in strategies.items():
        s = _score_strategy(trades, name)
        scores[name] = s
        score_str = f"({s['score']}/10)" if s['score'] is not None else "(N/A)"
        lines.append(
            f"    {name:12s} {s['verdict']:16s} {score_str:>10s}  |  "
            f"{s['trades']:3d} trades"
        )

    # ── Top Weaknesses ──
    lines.append("")
    lines.append("  TOP WEAKNESSES (by estimated impact)")
    lines.append("  " + "-" * 50)

    weaknesses = []

    for name, s in scores.items():
        trades = strategies[name]
        if s['score'] is None and len(trades) < 5:
            weaknesses.append((10.0, f"[{name}] Only {len(trades)} trades — signal generation too restrictive"))
        elif s['score'] is not None:
            if s['wr'] < 0.40:
                weaknesses.append((5.0, f"[{name}] Low win rate ({s['wr']*100:.0f}%) — entry quality issue"))
            if s['avg_r'] < 0.1:
                weaknesses.append((4.0, f"[{name}] Low expectancy ({s['avg_r']:+.3f}R avg) — edge is thin"))
            if s['stability_slope'] < -0.01:
                weaknesses.append((6.0, f"[{name}] Degrading edge (slope: {s['stability_slope']:+.4f}) — recent trades worse"))
            if s['max_dd_r'] < -5.0:
                weaknesses.append((3.0, f"[{name}] Deep drawdown ({s['max_dd_r']:+.1f}R) — risk management concern"))
            if s['trades_per_month'] < 1.0 and s['trades'] > 5:
                weaknesses.append((2.0, f"[{name}] Low frequency ({s['trades_per_month']:.1f}/mo) — few opportunities"))

    # Sort by impact (highest first)
    weaknesses.sort(key=lambda x: -x[0])
    for i, (impact, desc) in enumerate(weaknesses[:8], 1):
        lines.append(f"    {i}. {desc}")

    if not weaknesses:
        lines.append("    No significant weaknesses detected.")

    # ── Filter Verdicts ──
    lines.append("")
    lines.append("  FILTER VERDICTS (shadow-based)")
    lines.append("  " + "-" * 50)

    if filter_verdicts:
        for strat, verdicts in sorted(filter_verdicts.items()):
            parts = [f"{f}={v}" for f, v in sorted(verdicts.items())]
            lines.append(f"    {strat:12s} {', '.join(parts)}")
    else:
        lines.append("    (Filter attribution not available — run with shadow tracking)")

    # ── Portfolio Health ──
    lines.append("")
    lines.append("  PORTFOLIO HEALTH")
    lines.append("  " + "-" * 50)

    if portfolio_result is not None:
        hs = getattr(portfolio_result, 'heat_stats', None)
        if hs:
            lines.append(f"    Heat utilization:     {hs.avg_heat_pct:.1f}% of cap used on average")

        tc = getattr(portfolio_result, 'coordination_tighten_count', 0)
        bc = getattr(portfolio_result, 'coordination_boost_count', 0)
        if tc or bc:
            lines.append(f"    Coordination:         tighten={tc}, boost={bc}")

        op = getattr(portfolio_result, 'overlay_pnl', 0.0)
        if op != 0:
            lines.append(f"    Overlay PnL:          ${op:+,.0f}")
    else:
        lines.append("    (Portfolio result not available)")

    # ── Edge Stability ──
    lines.append("")
    lines.append("  EDGE STABILITY")
    lines.append("  " + "-" * 50)

    for name, s in scores.items():
        if s['score'] is None:
            stability = "INSUFFICIENT DATA"
        elif s['stability_slope'] > 0.01:
            stability = f"IMPROVING (slope: {s['stability_slope']:+.4f})"
        elif s['stability_slope'] < -0.01:
            stability = f"DEGRADING (slope: {s['stability_slope']:+.4f})"
        else:
            stability = "STABLE"
        lines.append(f"    {name:12s} {stability}")

    return "\n".join(lines)
