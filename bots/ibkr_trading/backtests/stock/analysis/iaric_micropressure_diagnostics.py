"""IARIC micropressure diagnostics -- sponsorship, confidence, and flow proxy analysis.

Note: In backtest, micropressure is always PROXY mode with limited signal variety.
This analysis focuses on what data IS available.

Sections:
1. Sponsorship signal breakdown -- STRONG/ACCUMULATE/NEUTRAL/STALE/WEAK -> performance
2. Confidence resolution -- what drives RED/YELLOW/GREEN; RED rejection rate
3. Sponsorship × regime cross-tab -- where confluence is strong/weak
4. Flowproxy UNAVAILABLE impact -- +1 adder cost
5. Signal degradation quantification -- trades with STALE signals vs fresh
6. Proxy mode compensation -- +1 adder adequate?
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _meta(t: TradeRecord, key: str, default=None):
    return t.metadata.get(key, default)


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _group_stats(trades: list[TradeRecord]) -> str:
    if not trades:
        return "    (no trades)"
    n = len(trades)
    wins = sum(1 for t in trades if t.is_winner)
    wr = wins / n
    rs = [t.r_multiple for t in trades]
    mean_r = float(np.mean(rs))
    total_r = sum(rs)
    gross_p = sum(r for r in rs if r > 0)
    gross_l = abs(sum(r for r in rs if r < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf") if gross_p > 0 else 0.0
    return f"    n={n}, WR={wr:.1%}, Mean R={mean_r:+.3f}, PF={pf:.2f}, Total R={total_r:+.2f}"


def _s1_sponsorship_breakdown(trades: list[TradeRecord]) -> str:
    """Sponsorship signal -> performance."""
    lines = [_hdr("MICRO-1. Sponsorship Signal Breakdown")]

    by_spons: dict[str, list[TradeRecord]] = {}
    for t in trades:
        sp = _meta(t, "sponsorship_state", "UNKNOWN") or "UNKNOWN"
        by_spons.setdefault(sp, []).append(t)

    for sp in ["STRONG", "ACCUMULATE", "NEUTRAL", "STALE", "WEAK", "BREAKDOWN", "UNKNOWN"]:
        if sp in by_spons:
            group = by_spons[sp]
            lines.append(f"  {sp} ({len(group)}):")
            lines.append(_group_stats(group))

    return "\n".join(lines)


def _s2_confidence_resolution(trades: list[TradeRecord], rejection_log: list[dict] | None) -> str:
    """What drives RED/YELLOW/GREEN; RED rejection rate."""
    lines = [_hdr("MICRO-2. Confidence Resolution")]

    by_conf: dict[str, list[TradeRecord]] = {}
    for t in trades:
        conf = _meta(t, "confidence", "UNKNOWN") or "UNKNOWN"
        by_conf.setdefault(conf, []).append(t)

    for conf in ["GREEN", "YELLOW", "RED", "UNKNOWN"]:
        if conf in by_conf:
            lines.append(f"  {conf}:")
            lines.append(_group_stats(by_conf[conf]))

    # RED rejections from log
    if rejection_log:
        red_rejections = [r for r in rejection_log if r.get("gate") == "confidence_red"]
        lines.append(f"\n  RED rejections (from log): {len(red_rejections)}")
        if red_rejections:
            # What setup types got RED?
            types: dict[str, int] = defaultdict(int)
            for r in red_rejections:
                types[r.get("setup_type", "?")] += 1
            lines.append(f"    By setup type: {dict(types)}")

    # Confidence transition: what signals drive GREEN vs YELLOW
    lines.append("\n  Confidence drivers (backtest: PROXY mode):")
    lines.append("    GREEN requires: sponsorship=STRONG + micropressure=ACCUMULATE (proxy mode)")
    lines.append("    RED requires: any DISTRIBUTE signal (sponsorship/micropressure/flowproxy)")
    lines.append("    YELLOW: everything else")

    green = by_conf.get("GREEN", [])
    yellow = by_conf.get("YELLOW", [])
    if green and yellow:
        g_mean = float(np.mean([t.r_multiple for t in green]))
        y_mean = float(np.mean([t.r_multiple for t in yellow]))
        lines.append(f"\n  GREEN mean R: {g_mean:+.3f} vs YELLOW mean R: {y_mean:+.3f}")
        lines.append(f"  Confidence premium: {g_mean - y_mean:+.3f}R per trade")

    return "\n".join(lines)


def _s3_sponsorship_x_regime(trades: list[TradeRecord]) -> str:
    """Sponsorship × regime cross-tab."""
    lines = [_hdr("MICRO-3. Sponsorship × Regime")]

    regimes = sorted(set(t.regime_tier or "?" for t in trades))
    spons_states = sorted(set(_meta(t, "sponsorship_state", "?") or "?" for t in trades))

    hdr = f"  {'':12s}"
    for sp in spons_states:
        hdr += f" {sp[:10]:>10s}"
    lines.append(hdr)
    lines.append("  " + "-" * (12 + 11 * len(spons_states)))

    for regime in regimes:
        row = f"  {regime:<12s}"
        for sp in spons_states:
            group = [t for t in trades
                     if (t.regime_tier or "?") == regime
                     and (_meta(t, "sponsorship_state", "?") or "?") == sp]
            if group:
                mean_r = float(np.mean([t.r_multiple for t in group]))
                row += f" {mean_r:>+6.3f}({len(group):>2})"
            else:
                row += f" {'--':>10s}"
        lines.append(row)

    return "\n".join(lines)


def _s4_flowproxy_unavailable(trades: list[TradeRecord]) -> str:
    """Flowproxy UNAVAILABLE impact -- +1 adder cost."""
    lines = [_hdr("MICRO-4. Flowproxy UNAVAILABLE Impact")]

    # Check conviction adders for "flow_unavailable"
    with_flow_adder = [t for t in trades if "flow_unavailable" in (_meta(t, "conviction_adders", []) or [])]
    without_flow_adder = [t for t in trades if "flow_unavailable" not in (_meta(t, "conviction_adders", []) or [])]

    lines.append(f"  With flow_unavailable adder (+1 req):")
    lines.append(_group_stats(with_flow_adder))
    lines.append(f"  Without (flow available):")
    lines.append(_group_stats(without_flow_adder))

    if with_flow_adder and without_flow_adder:
        # Compare acceptance counts
        with_acc = [_meta(t, "acceptance_count", 0) or 0 for t in with_flow_adder]
        without_acc = [_meta(t, "acceptance_count", 0) or 0 for t in without_flow_adder]
        lines.append(f"\n  Avg acceptance count: {float(np.mean(with_acc)):.1f} (flow_unavailable)"
                     f" vs {float(np.mean(without_acc)):.1f} (flow available)")

    return "\n".join(lines)


def _s5_signal_degradation(trades: list[TradeRecord]) -> str:
    """Trades with STALE signals vs fresh."""
    lines = [_hdr("MICRO-5. Signal Degradation")]

    stale = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") == "STALE"]
    fresh = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") in ("STRONG", "ACCUMULATE")]
    neutral = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") in ("NEUTRAL",)]

    lines.append(f"  Fresh signals (STRONG/ACCUMULATE):")
    lines.append(_group_stats(fresh))
    lines.append(f"  Neutral signals:")
    lines.append(_group_stats(neutral))
    lines.append(f"  Stale signals (0.85x penalty):")
    lines.append(_group_stats(stale))

    if fresh and stale:
        fresh_r = float(np.mean([t.r_multiple for t in fresh]))
        stale_r = float(np.mean([t.r_multiple for t in stale]))
        lines.append(f"\n  Degradation: {stale_r - fresh_r:+.3f}R per trade")
        lines.append(f"  Stale penalty (0.85x) {'justified' if stale_r < fresh_r else 'NOT justified -- stale outperforms'}")

    return "\n".join(lines)


def _s6_proxy_mode_compensation(trades: list[TradeRecord]) -> str:
    """Proxy mode +1 adder adequacy."""
    lines = [_hdr("MICRO-6. Proxy Mode Compensation")]

    # In backtest, ALL trades use proxy mode, so compare by acceptance count
    has_proxy_adder = [t for t in trades if "proxy_mode" in (_meta(t, "conviction_adders", []) or [])]
    n = len(trades)

    lines.append(f"  Trades with proxy_mode adder: {len(has_proxy_adder)}/{n}")
    lines.append(f"  (In backtest, all trades use PROXY mode for micropressure)")

    if has_proxy_adder:
        # Split by acceptance count to see if higher acceptance actually helps
        by_acc: dict[int, list[TradeRecord]] = {}
        for t in has_proxy_adder:
            ac = _meta(t, "acceptance_count", 0) or 0
            by_acc.setdefault(ac, []).append(t)

        lines.append(f"\n  Performance by acceptance count (with proxy adder):")
        for ac in sorted(by_acc):
            group = by_acc[ac]
            if group:
                mean_r = float(np.mean([t.r_multiple for t in group]))
                wr = sum(1 for t in group if t.is_winner) / len(group)
                lines.append(f"    Count={ac}: n={len(group)}, WR={wr:.0%}, R={mean_r:+.3f}")

    return "\n".join(lines)


def iaric_micropressure_diagnostics(
    trades: list[TradeRecord],
    rejection_log: list[dict] | None = None,
) -> str:
    """Generate the full IARIC micropressure diagnostics report."""
    if not trades:
        return "No trades to analyze."

    sections = [
        _s1_sponsorship_breakdown(trades),
        _s2_confidence_resolution(trades, rejection_log),
        _s3_sponsorship_x_regime(trades),
        _s4_flowproxy_unavailable(trades),
        _s5_signal_degradation(trades),
        _s6_proxy_mode_compensation(trades),
    ]
    return "\n".join(sections)
