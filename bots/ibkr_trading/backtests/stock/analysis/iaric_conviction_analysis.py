"""IARIC conviction analysis -- 6-factor risk unit decomposition and sizing insights.

Sections:
1. Risk unit factor decomposition -- distribution of each of 6 factors
2. Factor-R correlation -- Spearman of each factor vs R-multiple
3. Conviction bucket performance -- by bucket
4. Oversized vs undersized -- top/bottom 10% risk_unit performance
5. Stale penalty impact -- 0.85x trades vs non-stale, risk-adjusted
6. Confidence multiplier sensitivity -- simulate different multipliers for YELLOW
7. Cross-multiplier interactions -- confidence × location, confidence × regime
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


def _s1_factor_decomposition(trades: list[TradeRecord]) -> str:
    """Distribution of each of the 6 sizing factors."""
    lines = [_hdr("CONV-1. Risk Unit Factor Decomposition")]

    factors = {
        "conviction_multiplier": "Conviction",
        "timing_multiplier": "Timing",
        "regime_risk_multiplier": "Regime",
        "risk_unit_final": "Final Risk Unit",
    }

    for key, label in factors.items():
        vals = [_meta(t, key, None) for t in trades]
        numeric = [v for v in vals if v is not None and isinstance(v, (int, float))]
        if not numeric:
            lines.append(f"  {label}: no data")
            continue
        lines.append(f"  {label}:")
        lines.append(f"    Mean={float(np.mean(numeric)):.3f}, Median={float(np.median(numeric)):.3f}")
        lines.append(f"    Range=[{min(numeric):.3f}, {max(numeric):.3f}]")
        if len(numeric) > 1:
            p25, p75 = np.percentile(numeric, [25, 75])
            lines.append(f"    IQR=[{p25:.3f}, {p75:.3f}]")

    # Categorical factors
    for key, label in [("confidence", "Confidence"), ("location_grade", "Location Grade")]:
        counts: dict[str, int] = defaultdict(int)
        for t in trades:
            v = _meta(t, key, "?") or "?"
            counts[v] += 1
        lines.append(f"  {label}: {dict(sorted(counts.items()))}")

    return "\n".join(lines)


def _s2_factor_r_correlation(trades: list[TradeRecord]) -> str:
    """Spearman correlation of each factor vs R-multiple."""
    lines = [_hdr("CONV-2. Factor-R Correlation")]

    rs = np.array([t.r_multiple for t in trades])
    if len(rs) < 10:
        lines.append("  Insufficient trades for correlation analysis")
        return "\n".join(lines)

    factors = [
        ("conviction_multiplier", "Conviction"),
        ("timing_multiplier", "Timing"),
        ("regime_risk_multiplier", "Regime"),
        ("risk_unit_final", "Final Risk Unit"),
        ("drop_from_hod_pct", "Drop from HOD"),
        ("acceptance_count", "Acceptance Count"),
    ]

    lines.append(f"  {'Factor':<20s} {'Spearman ρ':>12s} {'Direction':>12s}")
    lines.append("  " + "-" * 48)

    for key, label in factors:
        vals = [_meta(t, key, None) for t in trades]
        numeric = [v if v is not None and isinstance(v, (int, float)) else np.nan for v in vals]
        x = np.array(numeric)
        mask = ~np.isnan(x)
        if mask.sum() < 10:
            lines.append(f"  {label:<20s} {'--':>12s} {'--':>12s}")
            continue

        # Spearman rank correlation
        from scipy.stats import spearmanr
        try:
            rho, pval = spearmanr(x[mask], rs[mask])
            direction = "positive" if rho > 0.05 else "negative" if rho < -0.05 else "neutral"
            sig = "*" if pval < 0.05 else ""
            lines.append(f"  {label:<20s} {rho:>+11.3f}{sig} {direction:>12s}")
        except Exception:
            lines.append(f"  {label:<20s} {'error':>12s} {'--':>12s}")

    lines.append("\n  * = p < 0.05")
    return "\n".join(lines)


def _s3_conviction_bucket(trades: list[TradeRecord]) -> str:
    """Performance by conviction bucket."""
    lines = [_hdr("CONV-3. Conviction Bucket Performance")]

    by_bucket: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        # Use conviction multiplier ranges as buckets
        cm = _meta(t, "conviction_multiplier", 1.0) or 1.0
        if cm < 0.6:
            bucket = "Low (<0.6)"
        elif cm < 0.8:
            bucket = "Med-Low (0.6-0.8)"
        elif cm < 1.0:
            bucket = "Medium (0.8-1.0)"
        elif cm < 1.2:
            bucket = "Med-High (1.0-1.2)"
        else:
            bucket = "High (>1.2)"
        by_bucket[bucket].append(t)

    for bucket in ["Low (<0.6)", "Med-Low (0.6-0.8)", "Medium (0.8-1.0)",
                    "Med-High (1.0-1.2)", "High (>1.2)"]:
        if bucket in by_bucket:
            lines.append(f"  {bucket}:")
            lines.append(_group_stats(by_bucket[bucket]))

    return "\n".join(lines)


def _s4_oversized_undersized(trades: list[TradeRecord]) -> str:
    """Top/bottom 10% risk_unit performance."""
    lines = [_hdr("CONV-4. Oversized vs Undersized")]

    risk_units = [(t, _meta(t, "risk_unit_final", 1.0) or 1.0) for t in trades]
    if len(risk_units) < 20:
        lines.append("  Insufficient trades for percentile analysis")
        return "\n".join(lines)

    risk_units.sort(key=lambda x: x[1])
    n = len(risk_units)
    bottom_10 = [t for t, _ in risk_units[:n // 10]]
    top_10 = [t for t, _ in risk_units[-(n // 10):]]
    middle = [t for t, _ in risk_units[n // 10:-(n // 10)]]

    bottom_ru = [ru for _, ru in risk_units[:n // 10]]
    top_ru = [ru for _, ru in risk_units[-(n // 10):]]

    lines.append(f"  Bottom 10% (risk_unit ≤ {max(bottom_ru):.3f}):")
    lines.append(_group_stats(bottom_10))
    lines.append(f"  Middle 80%:")
    lines.append(_group_stats(middle))
    lines.append(f"  Top 10% (risk_unit ≥ {min(top_ru):.3f}):")
    lines.append(_group_stats(top_10))

    return "\n".join(lines)


def _s5_stale_penalty_impact(trades: list[TradeRecord]) -> str:
    """0.85x trades vs non-stale, risk-adjusted."""
    lines = [_hdr("CONV-5. Stale Penalty Impact")]

    stale = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") == "STALE"]
    non_stale = [t for t in trades if (_meta(t, "sponsorship_state", "") or "") != "STALE"]

    lines.append(f"  Stale (0.85x penalty): {len(stale)} trades")
    lines.append(_group_stats(stale))
    lines.append(f"  Non-stale: {len(non_stale)} trades")
    lines.append(_group_stats(non_stale))

    if stale and non_stale:
        # Risk-adjusted comparison: R per unit of risk
        stale_ru = [_meta(t, "risk_unit_final", 1.0) or 1.0 for t in stale]
        non_stale_ru = [_meta(t, "risk_unit_final", 1.0) or 1.0 for t in non_stale]
        stale_r_per_ru = [t.r_multiple / ru for t, ru in zip(stale, stale_ru)]
        non_stale_r_per_ru = [t.r_multiple / ru for t, ru in zip(non_stale, non_stale_ru)]

        lines.append(f"\n  R per risk unit:")
        lines.append(f"    Stale: {float(np.mean(stale_r_per_ru)):+.3f}")
        lines.append(f"    Non-stale: {float(np.mean(non_stale_r_per_ru)):+.3f}")

    return "\n".join(lines)


def _s6_confidence_multiplier_sensitivity(trades: list[TradeRecord]) -> str:
    """Simulate different multipliers for YELLOW."""
    lines = [_hdr("CONV-6. Confidence Multiplier Sensitivity")]

    yellow = [t for t in trades if (_meta(t, "confidence", "") or "") == "YELLOW"]
    green = [t for t in trades if (_meta(t, "confidence", "") or "") == "GREEN"]

    if not yellow:
        lines.append("  No YELLOW trades to analyze")
        return "\n".join(lines)

    yellow_total_r = sum(t.r_multiple for t in yellow)
    green_total_r = sum(t.r_multiple for t in green)

    lines.append(f"  Current: YELLOW={len(yellow)} trades, GREEN={len(green)} trades")
    lines.append(f"  YELLOW total R: {yellow_total_r:+.2f}  |  GREEN total R: {green_total_r:+.2f}")

    # Simulate different YELLOW multipliers
    lines.append(f"\n  YELLOW multiplier sensitivity (current: 0.65x):")
    lines.append(f"  {'Mult':>6s} {'Adj Total R':>12s} {'Delta vs 0.65':>14s}")
    lines.append("  " + "-" * 36)

    current_mult = 0.65
    for mult in [0.50, 0.65, 0.80, 1.00]:
        # Approximate: scale R by ratio of new mult to current
        scale = mult / current_mult
        adj_r = sum(t.r_multiple * scale for t in yellow)
        total = adj_r + green_total_r
        delta = adj_r - yellow_total_r
        lines.append(f"  {mult:>6.2f} {total:>+12.2f} {delta:>+14.2f}")

    return "\n".join(lines)


def _s7_cross_multiplier_interactions(trades: list[TradeRecord]) -> str:
    """Confidence × location and confidence × regime interactions."""
    lines = [_hdr("CONV-7. Cross-Multiplier Interactions")]

    # Confidence × Location
    lines.append("  Confidence × Location Grade:")
    confs = sorted(set(_meta(t, "confidence", "?") or "?" for t in trades))
    grades = sorted(set(_meta(t, "location_grade", "?") or "?" for t in trades))

    hdr = f"  {'':8s}"
    for g in grades:
        hdr += f" {g:>10s}"
    lines.append(hdr)
    lines.append("  " + "-" * (8 + 11 * len(grades)))

    for conf in confs:
        row = f"  {conf:<8s}"
        for g in grades:
            group = [t for t in trades
                     if (_meta(t, "confidence", "?") or "?") == conf
                     and (_meta(t, "location_grade", "?") or "?") == g]
            if group:
                mean_r = float(np.mean([t.r_multiple for t in group]))
                row += f" {mean_r:>+6.3f}({len(group):>2})"
            else:
                row += f" {'--':>10s}"
        lines.append(row)

    # Confidence × Regime
    lines.append(f"\n  Confidence × Regime:")
    regimes = sorted(set(t.regime_tier or "?" for t in trades))

    hdr = f"  {'':8s}"
    for r in regimes:
        hdr += f" {r:>10s}"
    lines.append(hdr)
    lines.append("  " + "-" * (8 + 11 * len(regimes)))

    for conf in confs:
        row = f"  {conf:<8s}"
        for r in regimes:
            group = [t for t in trades
                     if (_meta(t, "confidence", "?") or "?") == conf
                     and (t.regime_tier or "?") == r]
            if group:
                mean_r = float(np.mean([t.r_multiple for t in group]))
                row += f" {mean_r:>+6.3f}({len(group):>2})"
            else:
                row += f" {'--':>10s}"
        lines.append(row)

    return "\n".join(lines)


def iaric_conviction_analysis(trades: list[TradeRecord]) -> str:
    """Generate the full IARIC conviction analysis report."""
    if not trades:
        return "No trades to analyze."

    sections = [
        _s1_factor_decomposition(trades),
        _s2_factor_r_correlation(trades),
        _s3_conviction_bucket(trades),
        _s4_oversized_undersized(trades),
        _s5_stale_penalty_impact(trades),
        _s6_confidence_multiplier_sensitivity(trades),
        _s7_cross_multiplier_interactions(trades),
    ]
    return "\n".join(sections)
