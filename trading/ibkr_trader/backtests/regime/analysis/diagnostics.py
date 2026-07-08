"""Regime portfolio diagnostics — 12-section comprehensive analysis.

Analyzes regime distribution, per-regime performance, allocations, crisis overlay,
leverage, confidence calibration, drawdowns, and benchmark comparison for the
MR-AWQ meta-allocator.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from regime.config import REGIMES

REGIME_NAMES = {
    "G": "Recovery",
    "R": "Reflation",
    "S": "Infl Hedge",
    "D": "Defensive",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dominant_regime_series(signals: pd.DataFrame) -> pd.Series:
    """Weekly dominant regime (G/R/S/D) from posterior probabilities."""
    regime_cols = [f"P_{r}" for r in REGIMES]
    available = [c for c in regime_cols if c in signals.columns]
    if not available:
        return pd.Series(dtype=str)
    return signals[available].idxmax(axis=1).str.replace("P_", "")


def _align_regime_to_daily(
    signals: pd.DataFrame, daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Forward-fill weekly dominant regime to daily frequency."""
    weekly_dom = _dominant_regime_series(signals)
    return weekly_dom.reindex(daily_index, method="ffill")


def _weekly_returns(daily_returns: pd.Series) -> pd.Series:
    """Aggregate daily returns to weekly (Friday-ending)."""
    return (1 + daily_returns).resample("W-FRI").prod() - 1


def _find_drawdown_episodes(
    equity_curve: pd.Series, top_n: int = 5,
) -> list[dict]:
    """Find top N drawdown episodes by depth."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    is_at_peak = drawdown >= -1e-10
    if is_at_peak.all():
        return []

    peak_groups = is_at_peak.cumsum()
    episodes: list[dict] = []

    for _, grp_dd in drawdown[~is_at_peak].groupby(peak_groups[~is_at_peak]):
        if grp_dd.empty:
            continue

        trough_date = grp_dd.idxmin()
        depth = float(grp_dd.min())
        dd_start = grp_dd.index[0]

        # Peak date: date of maximum equity before this drawdown
        pre_dd = equity_curve.loc[:dd_start]
        peak_date = pre_dd.idxmax() if len(pre_dd) > 0 else dd_start

        # Duration: trading days in drawdown
        duration = len(grp_dd)

        # Recovery: trading days from trough back to previous peak level
        peak_eq = running_max.loc[dd_start]
        post_trough = equity_curve.loc[trough_date:]
        recovered = post_trough >= peak_eq * 0.9999
        if recovered.any():
            recovery_date = post_trough.index[recovered.values][0]
            recovery_days = max(
                0, len(equity_curve.loc[trough_date:recovery_date]) - 1,
            )
        else:
            recovery_days = -1

        episodes.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "depth": depth,
            "duration": duration,
            "recovery_days": recovery_days,
        })

    episodes.sort(key=lambda x: x["depth"])
    return episodes[:top_n]


# ---------------------------------------------------------------------------
# HMM State Feature Means
# ---------------------------------------------------------------------------


def _section_hmm_feature_means(
    hmm_state_means: Optional[pd.DataFrame] = None,
) -> str:
    """Display per-state HMM feature means (z-scored) for economic interpretability."""
    if hmm_state_means is None or hmm_state_means.empty:
        return ""

    lines = ["  2b. HMM STATE FEATURE MEANS (z-scored)"]
    lines.append("  " + "-" * 40)

    features = hmm_state_means.columns.tolist()
    feat_display = [f[:14] for f in features]

    hdr = f"    {'Regime':<18s}" + "".join(f" {f:>14s}" for f in feat_display)
    lines.append(hdr)
    lines.append(f"    {'─' * (18 + 15 * len(features))}")

    for r in REGIMES:
        if r not in hmm_state_means.index:
            continue
        name = f"{REGIME_NAMES[r]} ({r})"
        row = f"    {name:<18s}"
        for feat in features:
            val = float(hmm_state_means.loc[r, feat])
            row += f" {val:>+13.3f}"
        lines.append(row)

    lines.append("")
    lines.append("    Note: Values are z-scored (rolling window). "
                 "Positive = above historical mean.")

    # Highlight dominant feature per state
    lines.append("")
    lines.append("    Dominant features per state:")
    for r in REGIMES:
        if r not in hmm_state_means.index:
            continue
        means = hmm_state_means.loc[r]
        top_pos = means.nlargest(2)
        top_neg = means.nsmallest(2)
        pos_str = ", ".join(f"{n}={v:+.2f}" for n, v in top_pos.items() if v > 0.1)
        neg_str = ", ".join(f"{n}={v:+.2f}" for n, v in top_neg.items() if v < -0.1)
        name = f"{REGIME_NAMES[r]} ({r})"
        parts = []
        if pos_str:
            parts.append(f"HIGH: {pos_str}")
        if neg_str:
            parts.append(f"LOW: {neg_str}")
        lines.append(f"      {name:<18s}  {' | '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def generate_regime_diagnostics_report(
    signals: pd.DataFrame,
    result,
    benchmark,
    score,
    sim_cfg,
    walk_forward_results: Optional[list[dict]] = None,
    hmm_state_means: Optional[pd.DataFrame] = None,
) -> str:
    """Generate 12-section regime diagnostics report.

    Args:
        signals: Weekly signal DataFrame from run_signal_engine.
        result: PortfolioResult from simulate_portfolio.
        benchmark: PortfolioResult from simulate_benchmark_60_40.
        score: CompositeScore from composite_score.
        sim_cfg: RegimeBacktestConfig.
        walk_forward_results: Optional list of fold result dicts.
        hmm_state_means: Optional DataFrame of per-state feature means (z-scored).
    """
    header = [
        "=" * 70,
        "  REGIME PORTFOLIO DIAGNOSTICS",
        "=" * 70,
        "",
    ]

    sections = [
        _section_portfolio_overview(result, score, sim_cfg),
        _section_regime_distribution(signals),
        _section_hmm_feature_means(hmm_state_means),
        _section_per_regime_performance(signals, result),
        _section_per_regime_allocations(signals),
        _section_regime_probability_heatmap(signals),
        _section_crisis_overlay(signals),
        _section_crisis_decomposition(signals),
        _section_ensemble_disagreement(signals),
        _section_leverage_analysis(signals, result),
        _section_confidence_calibration(signals, result),
        _section_drawdown_episodes(result, signals),
        _section_yearly_returns(result, signals),
        _section_benchmark_comparison(result, benchmark),
        _section_walk_forward_summary(walk_forward_results),
        _section_scanner_analysis(signals),
    ]

    parts = ["\n".join(header)]
    parts.extend(s for s in sections if s)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 13. Leading Indicator Scanner
# ---------------------------------------------------------------------------


def _section_scanner_analysis(signals: pd.DataFrame) -> str:
    """Section 13: Leading Indicator Scanner diagnostics."""
    if "shift_prob" not in signals.columns:
        return ""

    lines = ["  13. LEADING INDICATOR SCANNER"]
    lines.append("  " + "-" * 40)

    sp = signals["shift_prob"]
    n_weeks = len(sp)
    lines.append(f"    Mean shift_prob:   {sp.mean():.4f}")
    lines.append(f"    Max shift_prob:    {sp.max():.4f}")
    lines.append(f"    Std shift_prob:    {sp.std():.4f}")

    # Weeks above threshold
    above_06 = int((sp > 0.6).sum())
    above_07 = int((sp > 0.7).sum())
    above_08 = int((sp > 0.8).sum())
    lines.append("")
    lines.append("    Activation frequency:")
    lines.append(f"      shift_prob > 0.6:  {above_06:>4d} weeks ({above_06/n_weeks*100:.1f}%)")
    lines.append(f"      shift_prob > 0.7:  {above_07:>4d} weeks ({above_07/n_weeks*100:.1f}%)")
    lines.append(f"      shift_prob > 0.8:  {above_08:>4d} weeks ({above_08/n_weeks*100:.1f}%)")

    # Direction counts
    if "shift_dir" in signals.columns:
        dir_counts = signals["shift_dir"].value_counts()
        lines.append("")
        lines.append("    Direction distribution:")
        for d in ["risk_off", "risk_on", "neutral"]:
            count = int(dir_counts.get(d, 0))
            lines.append(f"      {d:<12s}  {count:>4d} weeks ({count/n_weeks*100:.1f}%)")

    # Dominant indicator frequency
    if "dominant_indicator" in signals.columns:
        ind_counts = signals["dominant_indicator"].value_counts()
        lines.append("")
        lines.append("    Dominant indicator frequency:")
        for ind, count in ind_counts.items():
            lines.append(f"      {ind:<26s}  {count:>4d} ({count/n_weeks*100:.1f}%)")

    # Leverage comparison when scanner active vs inactive
    lev_col = "final_leverage" if "final_leverage" in signals.columns else "L"
    if lev_col in signals.columns:
        active = sp > 0.6
        if active.any() and (~active).any():
            avg_L_active = float(signals.loc[active, lev_col].mean())
            avg_L_inactive = float(signals.loc[~active, lev_col].mean())
            lines.append("")
            lines.append(f"    Avg leverage (scanner active, p>0.6):  {avg_L_active:.4f}")
            lines.append(f"    Avg leverage (scanner inactive):       {avg_L_inactive:.4f}")
            if avg_L_inactive > 1e-6:
                reduction = (1 - avg_L_active / avg_L_inactive) * 100
                lines.append(f"    Scanner leverage reduction:            {reduction:.1f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Portfolio Overview
# ---------------------------------------------------------------------------


def _section_portfolio_overview(result, score, sim_cfg) -> str:
    m = result.metrics
    lines = ["  1. PORTFOLIO OVERVIEW"]
    lines.append("  " + "-" * 40)

    lines.append(f"    Initial equity:    ${sim_cfg.initial_equity:>12,.0f}")
    lines.append(f"    Final equity:      ${float(result.equity_curve.iloc[-1]):>12,.0f}")
    lines.append(f"    Total return:      {m.total_return:>12.1%}")
    lines.append(f"    CAGR:              {m.cagr:>12.2%}")
    lines.append(f"    Sharpe:            {m.sharpe:>12.3f}")
    lines.append(f"    Sortino:           {m.sortino:>12.3f}")
    lines.append(f"    Calmar:            {m.calmar:>12.3f}")
    lines.append(f"    Max drawdown:      {m.max_drawdown_pct:>12.1%}")
    lines.append(f"    Max DD duration:   {m.max_drawdown_duration:>12d} days")
    lines.append(f"    Avg annual TO:     {m.avg_annual_turnover:>12.2f}")
    lines.append(f"    Rebalances:        {m.n_rebalances:>12d}")

    lines.append("")
    lines.append("    Composite Score Breakdown:")
    lines.append(f"      Sharpe  (w=0.25):  {score.sharpe_component:.4f}")
    lines.append(f"      Calmar  (w=0.25):  {score.calmar_component:.4f}")
    lines.append(f"      Inv DD  (w=0.20):  {score.inv_dd_component:.4f}")
    lines.append(f"      CAGR    (w=0.15):  {score.cagr_component:.4f}")
    lines.append(f"      Sortino (w=0.15):  {score.sortino_component:.4f}")
    lines.append(f"      TOTAL:             {score.total:.4f}")
    if score.rejected:
        lines.append(f"      REJECTED: {score.reject_reason}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Regime Distribution
# ---------------------------------------------------------------------------


def _section_regime_distribution(signals: pd.DataFrame) -> str:
    if signals.empty:
        return ""

    lines = ["  2. REGIME DISTRIBUTION"]
    lines.append("  " + "-" * 40)

    dom = _dominant_regime_series(signals)
    n_weeks = len(dom)

    lines.append(
        f"    {'Regime':<14s} {'Weeks':>6s} {'%':>7s} "
        f"{'AvgP(dom)':>10s} {'AvgP(all)':>10s}"
    )
    lines.append(f"    {'─' * 47}")

    for r in REGIMES:
        mask = dom == r
        count = int(mask.sum())
        pct = count / n_weeks * 100 if n_weeks > 0 else 0
        col = f"P_{r}"
        avg_p_all = float(signals[col].mean()) if col in signals.columns else 0
        avg_p_dom = (
            float(signals.loc[mask, col].mean())
            if mask.any() and col in signals.columns
            else 0
        )
        name = f"{REGIME_NAMES[r][:11]} ({r})"
        lines.append(
            f"    {name:<14s} {count:>6d} {pct:>6.1f}% "
            f"{avg_p_dom:>10.3f} {avg_p_all:>10.3f}"
        )

    # Transitions
    transitions = max(0, int((dom != dom.shift()).sum()) - 1)
    transition_rate = transitions / n_weeks if n_weeks > 1 else 0

    # Average spell duration
    spell_groups = (dom != dom.shift()).cumsum()
    spell_lengths = dom.groupby(spell_groups).size()
    avg_spell = float(spell_lengths.mean()) if len(spell_lengths) > 0 else 0

    lines.append("")
    lines.append(f"    Total weeks:       {n_weeks}")
    lines.append(f"    Transitions:       {transitions}")
    lines.append(f"    Transition rate:   {transition_rate:.3f} /week")
    lines.append(f"    Avg spell length:  {avg_spell:.1f} weeks")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Per-Regime Performance
# ---------------------------------------------------------------------------


def _section_per_regime_performance(signals: pd.DataFrame, result) -> str:
    if signals.empty:
        return ""

    lines = ["  3. PER-REGIME PERFORMANCE"]
    lines.append("  " + "-" * 40)

    daily_ret = result.daily_returns
    daily_regime = _align_regime_to_daily(signals, daily_ret.index)
    valid = daily_regime.notna()
    daily_ret = daily_ret[valid]
    daily_regime = daily_regime[valid]

    if daily_ret.empty:
        lines.append("    No valid data after regime alignment.")
        return "\n".join(lines)

    total_cum = float((1 + daily_ret).prod() - 1)

    lines.append(
        f"    {'Regime':<14s} {'AnnRet':>8s} {'AnnVol':>8s} "
        f"{'Sharpe':>8s} {'MaxDD':>8s} {'Contrib':>8s}"
    )
    lines.append(f"    {'─' * 54}")

    for r in REGIMES:
        mask = daily_regime == r
        name = f"{REGIME_NAMES[r][:11]} ({r})"
        if not mask.any():
            lines.append(
                f"    {name:<14s} {'—':>8s} {'—':>8s} "
                f"{'—':>8s} {'—':>8s} {'—':>8s}"
            )
            continue

        ret_r = daily_ret[mask]
        n_days = len(ret_r)

        cum_ret = float((1 + ret_r).prod() - 1)
        ann_factor = 252.0 / n_days if n_days >= 5 else 0
        ann_ret = (
            (1 + cum_ret) ** ann_factor - 1 if ann_factor > 0 else cum_ret
        )
        ann_vol = float(ret_r.std() * np.sqrt(252)) if n_days > 1 else 0
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0

        eq_r = (1 + ret_r).cumprod()
        max_dd = float(-(eq_r / eq_r.cummax() - 1).min()) if len(eq_r) > 0 else 0

        contrib = cum_ret / total_cum * 100 if abs(total_cum) > 1e-10 else 0

        lines.append(
            f"    {name:<14s} {ann_ret:>+7.1%} {ann_vol:>7.1%} "
            f"{sharpe:>+8.2f} {max_dd:>7.1%} {contrib:>+7.1f}%"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Per-Regime Allocations
# ---------------------------------------------------------------------------


def _section_per_regime_allocations(signals: pd.DataFrame) -> str:
    if signals.empty:
        return ""

    lines = ["  4. PER-REGIME ALLOCATIONS"]
    lines.append("  " + "-" * 40)

    pi_cols = sorted(c for c in signals.columns if c.startswith("pi_"))
    if not pi_cols:
        lines.append("    No allocation columns found.")
        return "\n".join(lines)

    sleeves = [c.replace("pi_", "") for c in pi_cols]
    dom = _dominant_regime_series(signals)

    hdr = f"    {'Regime':<14s}" + "".join(f" {s[:6]:>7s}" for s in sleeves)
    lines.append(hdr)
    lines.append(f"    {'─' * (14 + 8 * len(sleeves))}")

    for r in REGIMES:
        mask = dom == r
        name = f"{REGIME_NAMES[r][:11]} ({r})"
        if not mask.any():
            row = f"    {name:<14s}" + "".join(f" {'—':>7s}" for _ in sleeves)
            lines.append(row)
            continue

        avgs = signals.loc[mask, pi_cols].mean()
        row = f"    {name:<14s}" + "".join(
            f" {avgs[c]:>7.3f}" for c in pi_cols
        )
        lines.append(row)

    lines.append(f"    {'─' * (14 + 8 * len(sleeves))}")
    overall = signals[pi_cols].mean()
    row = f"    {'Overall':<14s}" + "".join(
        f" {overall[c]:>7.3f}" for c in pi_cols
    )
    lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Regime Probability Heatmap
# ---------------------------------------------------------------------------


def _section_regime_probability_heatmap(signals: pd.DataFrame) -> str:
    if signals.empty:
        return ""

    regime_cols = [f"P_{r}" for r in REGIMES]
    available = [c for c in regime_cols if c in signals.columns]
    if not available:
        return ""

    lines = ["  5. REGIME PROBABILITY HEATMAP (yearly avg)"]
    lines.append("  " + "-" * 40)

    yearly = signals[available].groupby(signals.index.year).mean()

    hdr = f"    {'Year':<6s}" + "".join(f"   {r:>5s}" for r in REGIMES) + "  Dominant"
    lines.append(hdr)
    lines.append(f"    {'─' * (6 + 8 * len(REGIMES) + 10)}")

    for year in yearly.index:
        vals = [float(yearly.loc[year, f"P_{r}"]) for r in REGIMES]
        dominant = REGIMES[int(np.argmax(vals))]
        row = f"    {year:<6d}"
        for i, r in enumerate(REGIMES):
            marker = "*" if r == dominant else " "
            row += f"  {vals[i]:.3f}{marker}"
        row += f"  {dominant}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Crisis Overlay
# ---------------------------------------------------------------------------


def _section_crisis_overlay(signals: pd.DataFrame) -> str:
    if signals.empty or "p_crisis" not in signals.columns:
        return ""

    lines = ["  6. CRISIS OVERLAY"]
    lines.append("  " + "-" * 40)

    pc = signals["p_crisis"]
    lines.append(f"    Mean p_crisis:     {pc.mean():.4f}")
    lines.append(f"    Median:            {pc.median():.4f}")
    lines.append(f"    Std:               {pc.std():.4f}")
    lines.append(f"    Max:               {pc.max():.4f}")
    if "crisis_severity" in signals.columns:
        severity_counts = signals["crisis_severity"].value_counts()
        lines.append("")
        lines.append("    Severity distribution:")
        for level in ["none", "elevated", "acute"]:
            count = int(severity_counts.get(level, 0))
            lines.append(f"      {level:<9s} {count:>4d} weeks ({count / len(pc) * 100:.1f}%)")

    lines.append("")
    lines.append("    Activation frequency:")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        count = int((pc > thresh).sum())
        pct = count / len(pc) * 100
        lines.append(f"      p_crisis > {thresh:.1f}:  {count:>4d} weeks ({pct:.1f}%)")

    # Leverage during crisis vs normal
    lev_col = "final_leverage" if "final_leverage" in signals.columns else "L"
    if lev_col in signals.columns:
        high_crisis = pc > 0.5
        if high_crisis.any() and (~high_crisis).any():
            avg_L_crisis = float(signals.loc[high_crisis, lev_col].mean())
            avg_L_normal = float(signals.loc[~high_crisis, lev_col].mean())
            lines.append("")
            lines.append(f"    Avg leverage (crisis p>0.5):  {avg_L_crisis:.3f}")
            lines.append(f"    Avg leverage (normal):        {avg_L_normal:.3f}")
            if avg_L_normal > 1e-6:
                reduction = (1 - avg_L_crisis / avg_L_normal) * 100
                lines.append(f"    Leverage reduction:           {reduction:.1f}%")

    # Top 5 crisis spikes
    top5 = pc.nlargest(5)
    if len(top5) > 0:
        lines.append("")
        lines.append("    Top 5 crisis spikes:")
        for date, val in top5.items():
            lines.append(f"      {date.strftime('%Y-%m-%d')}:  {val:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6b. Crisis Decomposition
# ---------------------------------------------------------------------------


def _section_crisis_decomposition(signals: pd.DataFrame) -> str:
    if signals.empty or "crisis_composite" not in signals.columns:
        return ""

    lines = ["  6b. CRISIS CHANNEL DECOMPOSITION"]
    lines.append("  " + "-" * 40)

    component_pairs = [
        ("vix", "crisis_vix_component"),
        ("spread", "crisis_spread_component"),
        ("realized_vol", "crisis_vol_component"),
        ("spy_tlt_corr", "crisis_corr_component"),
        ("legacy_pairwise", "crisis_legacy_corr_component"),
    ]

    for label, col in component_pairs:
        if col not in signals.columns:
            continue
        series = signals[col]
        lines.append(
            f"    {label:<14s} mean={series.mean():>7.4f}  "
            f"max={series.max():>7.4f}"
        )

    if "crisis_corr_trigger" in signals.columns:
        trigger = signals["crisis_corr_trigger"]
        lines.append("")
        lines.append(
            f"    Positive SPY/TLT corr weeks: {(trigger > 0).sum():>4d} "
            f"({(trigger > 0).mean() * 100:.1f}%)"
        )

    top5 = signals["crisis_composite"].nlargest(5)
    if len(top5) > 0:
        lines.append("")
        lines.append("    Top 5 composite readings:")
        for date, val in top5.items():
            lines.append(f"      {date.strftime('%Y-%m-%d')}:  {val:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6c. Ensemble Disagreement
# ---------------------------------------------------------------------------


def _section_ensemble_disagreement(signals: pd.DataFrame) -> str:
    if signals.empty or "consensus_ratio" not in signals.columns:
        return ""

    lines = ["  6c. ENSEMBLE DISAGREEMENT"]
    lines.append("  " + "-" * 40)

    consensus = signals["consensus_ratio"]
    lines.append(f"    Mean consensus:    {consensus.mean():.4f}")
    lines.append(f"    Min consensus:     {consensus.min():.4f}")
    lines.append(f"    Mean trend (4w):   {signals['consensus_trend_4w'].mean():+.4f}")
    lines.append(f"    Mean disagreement: {signals['avg_disagreement'].mean():.4f}")

    if "uncertainty_level" in signals.columns:
        counts = signals["uncertainty_level"].value_counts()
        lines.append("")
        lines.append("    Uncertainty distribution:")
        for level in ["low", "moderate", "high"]:
            count = int(counts.get(level, 0))
            lines.append(
                f"      {level:<9s} {count:>4d} weeks ({count / len(signals) * 100:.1f}%)"
            )

    if "minority_regime" in signals.columns:
        minority_counts = signals["minority_regime"].value_counts()
        lines.append("")
        lines.append("    Minority regime frequency:")
        for regime, count in minority_counts.items():
            lines.append(f"      {regime:<9s} {count:>4d}")

    if "disagree_std_G" in signals.columns:
        lines.append("")
        lines.append("    Per-regime posterior std:")
        for regime in REGIMES:
            col = f"disagree_std_{regime}"
            lines.append(f"      {regime}: {signals[col].mean():.4f}")

    if "disagreement_leverage_adj" in signals.columns:
        lines.append("")
        lines.append(f"    Mean disagreement adj: {signals['disagreement_leverage_adj'].mean():.4f}")
    if "disagreement_warning" in signals.columns:
        warning_count = int(signals["disagreement_warning"].sum())
        lines.append(f"    Warning weeks:         {warning_count:>4d} ({warning_count / len(signals) * 100:.1f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Leverage Analysis
# ---------------------------------------------------------------------------


def _section_leverage_analysis(signals: pd.DataFrame, result) -> str:
    lev_col = "final_leverage" if "final_leverage" in signals.columns else "L"
    if signals.empty or lev_col not in signals.columns:
        return ""

    lines = ["  7. LEVERAGE ANALYSIS"]
    lines.append("  " + "-" * 40)

    lev = signals[lev_col]
    lines.append(f"    Mean:       {lev.mean():.4f}")
    lines.append(f"    Median:     {lev.median():.4f}")
    lines.append(f"    Std:        {lev.std():.4f}")
    lines.append(f"    Min:        {lev.min():.4f}")
    lines.append(f"    Max:        {lev.max():.4f}")

    lines.append("")
    lines.append("    Percentile distribution:")
    for pct in [10, 25, 50, 75, 90]:
        val = float(np.percentile(lev.dropna(), pct))
        lines.append(f"      P{pct:<3d}  {val:.4f}")

    component_cols = [
        ("HMM base", "hmm_leverage"),
        ("Crisis adj", "crisis_leverage_adj"),
        ("Scanner adj", "scanner_leverage_adj"),
        ("Disagree adj", "disagreement_leverage_adj"),
    ]
    available_components = [item for item in component_cols if item[1] in signals.columns]
    if available_components:
        lines.append("")
        lines.append("    Mean leverage components:")
        for label, col in available_components:
            lines.append(f"      {label:<11s} {signals[col].mean():.4f}")

    # DD ladder band analysis (from equity curve)
    eq = result.equity_curve
    running_max = eq.cummax()
    dd_pct = (eq - running_max) / running_max
    n_days = len(dd_pct)

    dd_bands = [
        ("0% to -8%", 0.001, -0.08),
        ("-8% to -12%", -0.08, -0.12),
        ("-12% to -16%", -0.12, -0.16),
        ("-16% to -20%", -0.16, -0.20),
        ("Below -20%", -0.20, -1.0),
    ]

    lines.append("")
    lines.append("    DD ladder band counts (trading days):")
    for label, upper, lower in dd_bands:
        count = int(((dd_pct <= upper) & (dd_pct > lower)).sum())
        pct_val = count / n_days * 100 if n_days > 0 else 0
        lines.append(f"      {label:<16s}  {count:>5d} days ({pct_val:>5.1f}%)")

    # Per-regime leverage
    dom = _dominant_regime_series(signals)
    lines.append("")
    lines.append("    Per-regime average leverage:")
    for r in REGIMES:
        mask = dom == r
        if mask.any():
            avg_L = float(signals.loc[mask, "L"].mean())
            lines.append(f"      {REGIME_NAMES[r][:11]} ({r}):  {avg_L:.4f}")

    # L vs weekly return correlation
    weekly_ret = _weekly_returns(result.daily_returns)
    common_idx = lev.index.intersection(weekly_ret.index)
    if len(common_idx) > 10:
        corr = float(lev.reindex(common_idx).corr(weekly_ret.reindex(common_idx)))
        lines.append("")
        lines.append(f"    Leverage vs weekly return corr: {corr:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Confidence Calibration
# ---------------------------------------------------------------------------


def _section_confidence_calibration(signals: pd.DataFrame, result) -> str:
    if signals.empty or "Conf" not in signals.columns:
        return ""

    lines = ["  8. CONFIDENCE CALIBRATION"]
    lines.append("  " + "-" * 40)

    conf = signals["Conf"]
    lines.append(f"    Mean:       {conf.mean():.4f}")
    lines.append(f"    Median:     {conf.median():.4f}")
    lines.append(f"    Std:        {conf.std():.4f}")
    lines.append(f"    Min:        {conf.min():.4f}")
    lines.append(f"    Max:        {conf.max():.4f}")

    if "posterior_conf" in signals.columns:
        lines.append(f"    Posterior-only mean:  {signals['posterior_conf'].mean():.4f}")
    if "disagreement_conf" in signals.columns and signals["disagreement_conf"].notna().any():
        lines.append(f"    Disagreement mean:    {signals['disagreement_conf'].dropna().mean():.4f}")

    # Distribution buckets
    lines.append("")
    lines.append("    Distribution:")
    edges = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        count = int(((conf >= lo) & (conf < hi)).sum())
        pct = count / len(conf) * 100 if len(conf) > 0 else 0
        label = f"[{lo:.1f}, 1.0]" if hi >= 1.01 else f"[{lo:.1f}, {hi:.1f})"
        lines.append(f"      {label:<12s}  {count:>4d} weeks ({pct:>5.1f}%)")

    # Performance by quartile
    weekly_ret = _weekly_returns(result.daily_returns)
    common_idx = conf.index.intersection(weekly_ret.index)
    if len(common_idx) > 20:
        df = pd.DataFrame({
            "conf": conf.reindex(common_idx),
            "ret": weekly_ret.reindex(common_idx),
        }).dropna()

        if len(df) > 20:
            try:
                quartiles = pd.qcut(
                    df["conf"], 4,
                    labels=["Q1(low)", "Q2", "Q3", "Q4(high)"],
                    duplicates="drop",
                )
            except ValueError:
                return "\n".join(lines)

            lines.append("")
            lines.append(
                f"    {'Quartile':<12s} {'AvgConf':>8s} {'AnnRet':>8s} "
                f"{'Sharpe':>8s} {'Weeks':>6s}"
            )
            lines.append(f"    {'─' * 42}")

            for q_label in sorted(quartiles.unique()):
                q_mask = quartiles == q_label
                q_ret = df.loc[q_mask, "ret"]
                q_conf = df.loc[q_mask, "conf"]
                n_wk = len(q_ret)
                avg_conf = float(q_conf.mean())
                ann_ret = float((1 + q_ret.mean()) ** 52 - 1)
                vol = float(q_ret.std() * np.sqrt(52)) if n_wk > 1 else 0
                sharpe = ann_ret / vol if vol > 1e-10 else 0
                lines.append(
                    f"    {str(q_label):<12s} {avg_conf:>8.3f} {ann_ret:>+7.1%} "
                    f"{sharpe:>+8.2f} {n_wk:>6d}"
                )

            corr = float(df["conf"].corr(df["ret"]))
            lines.append("")
            lines.append(f"    Confidence vs weekly return corr: {corr:+.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Drawdown Episodes
# ---------------------------------------------------------------------------


def _section_drawdown_episodes(result, signals: pd.DataFrame) -> str:
    episodes = _find_drawdown_episodes(result.equity_curve, top_n=5)
    if not episodes:
        return ""

    lines = ["  9. DRAWDOWN EPISODES (Top 5)"]
    lines.append("  " + "-" * 40)

    daily_regime = _align_regime_to_daily(signals, result.equity_curve.index)

    lines.append(
        f"    {'#':<3s} {'Peak':>12s} {'Trough':>12s} "
        f"{'Depth':>8s} {'Days':>6s} {'Recov':>6s} {'Regime':<5s}"
    )
    lines.append(f"    {'─' * 53}")

    for i, ep in enumerate(episodes, 1):
        peak_str = ep["peak_date"].strftime("%Y-%m-%d")
        trough_str = ep["trough_date"].strftime("%Y-%m-%d")
        depth_str = f"{ep['depth']:.1%}"
        dur_str = str(ep["duration"])
        recov_str = str(ep["recovery_days"]) if ep["recovery_days"] >= 0 else "N/R"

        # Dominant regime at trough
        td = ep["trough_date"]
        if td in daily_regime.index:
            regime = str(daily_regime.loc[td])
        elif len(daily_regime) > 0:
            idx = daily_regime.index.get_indexer([td], method="nearest")
            regime = str(daily_regime.iloc[idx[0]]) if idx[0] >= 0 else "?"
        else:
            regime = "?"

        lines.append(
            f"    {i:<3d} {peak_str:>12s} {trough_str:>12s} {depth_str:>8s} "
            f"{dur_str:>6s} {recov_str:>6s} {regime:<5s}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Yearly Returns
# ---------------------------------------------------------------------------


def _section_yearly_returns(result, signals: pd.DataFrame) -> str:
    daily_ret = result.daily_returns
    eq = result.equity_curve
    if daily_ret.empty:
        return ""

    lines = ["  10. YEARLY RETURNS"]
    lines.append("  " + "-" * 40)

    dom = _dominant_regime_series(signals)
    years = sorted(daily_ret.index.year.unique())

    lines.append(
        f"    {'Year':<6s} {'Return':>8s} {'MaxDD':>8s} "
        f"{'EndEquity':>12s}  Regime Mix"
    )
    lines.append(f"    {'─' * 60}")

    for year in years:
        yr_mask = daily_ret.index.year == year
        yr_ret = daily_ret[yr_mask]
        yr_return = float((1 + yr_ret).prod() - 1)

        yr_eq = eq[yr_mask]
        yr_peak = yr_eq.cummax()
        yr_dd = (yr_eq - yr_peak) / yr_peak
        yr_max_dd = float(-yr_dd.min()) if len(yr_dd) > 0 else 0
        end_eq = float(yr_eq.iloc[-1]) if len(yr_eq) > 0 else 0

        yr_dom = dom[dom.index.year == year]
        counts = yr_dom.value_counts()
        mix_parts = [
            f"{r}:{counts.get(r, 0)}" for r in REGIMES if counts.get(r, 0) > 0
        ]
        mix_str = " ".join(mix_parts) if mix_parts else "—"

        lines.append(
            f"    {year:<6d} {yr_return:>+7.1%} {yr_max_dd:>7.1%} "
            f"${end_eq:>11,.0f}  {mix_str}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Benchmark Comparison
# ---------------------------------------------------------------------------


def _section_benchmark_comparison(result, benchmark) -> str:
    if benchmark is None:
        return ""

    lines = ["  11. BENCHMARK COMPARISON (60/40 SPY/TLT)"]
    lines.append("  " + "-" * 40)

    rm = result.metrics
    bm = benchmark.metrics

    rows = [
        ("Total Return", f"{rm.total_return:.1%}", f"{bm.total_return:.1%}",
         f"{rm.total_return - bm.total_return:+.1%}"),
        ("CAGR", f"{rm.cagr:.2%}", f"{bm.cagr:.2%}",
         f"{rm.cagr - bm.cagr:+.2%}"),
        ("Sharpe", f"{rm.sharpe:.3f}", f"{bm.sharpe:.3f}",
         f"{rm.sharpe - bm.sharpe:+.3f}"),
        ("Sortino", f"{rm.sortino:.3f}", f"{bm.sortino:.3f}",
         f"{rm.sortino - bm.sortino:+.3f}"),
        ("Calmar", f"{rm.calmar:.3f}", f"{bm.calmar:.3f}",
         f"{rm.calmar - bm.calmar:+.3f}"),
        ("Max DD", f"{rm.max_drawdown_pct:.1%}", f"{bm.max_drawdown_pct:.1%}",
         f"{rm.max_drawdown_pct - bm.max_drawdown_pct:+.1%}"),
        ("Max DD Duration", f"{rm.max_drawdown_duration}d",
         f"{bm.max_drawdown_duration}d",
         f"{rm.max_drawdown_duration - bm.max_drawdown_duration:+d}d"),
        ("Avg Annual TO", f"{rm.avg_annual_turnover:.2f}",
         f"{bm.avg_annual_turnover:.2f}",
         f"{rm.avg_annual_turnover - bm.avg_annual_turnover:+.2f}"),
    ]

    lines.append(f"    {'Metric':<20s} {'Regime':>10s} {'60/40':>10s} {'Delta':>10s}")
    lines.append(f"    {'─' * 50}")
    for label, rv, bv, dv in rows:
        lines.append(f"    {label:<20s} {rv:>10s} {bv:>10s} {dv:>10s}")

    # Correlation and tracking error
    r_daily = result.daily_returns
    b_daily = benchmark.daily_returns
    common = r_daily.index.intersection(b_daily.index)

    if len(common) > 20:
        r_a = r_daily.reindex(common)
        b_a = b_daily.reindex(common)

        corr = float(r_a.corr(b_a))
        tracking_diff = r_a - b_a
        te = float(tracking_diff.std() * np.sqrt(252))
        ir = float(tracking_diff.mean() * 252 / te) if te > 1e-10 else 0

        lines.append("")
        lines.append(f"    Daily return correlation:  {corr:+.3f}")
        lines.append(f"    Tracking error (ann):      {te:.3f}")
        lines.append(f"    Information ratio:         {ir:+.3f}")

    # Yearly side-by-side
    years = sorted(r_daily.index.year.unique())
    if len(years) > 1:
        lines.append("")
        lines.append(f"    {'Year':<6s} {'Regime':>8s} {'60/40':>8s} {'Excess':>8s}")
        lines.append(f"    {'─' * 30}")

        for year in years:
            yr_r = float((1 + r_daily[r_daily.index.year == year]).prod() - 1)
            yr_b = float((1 + b_daily[b_daily.index.year == year]).prod() - 1)
            lines.append(
                f"    {year:<6d} {yr_r:>+7.1%} {yr_b:>+7.1%} "
                f"{yr_r - yr_b:>+7.1%}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. Walk-Forward Summary
# ---------------------------------------------------------------------------


def _section_walk_forward_summary(
    walk_forward_results: Optional[list[dict]],
) -> str:
    if not walk_forward_results:
        return ""

    lines = ["  12. WALK-FORWARD SUMMARY"]
    lines.append("  " + "-" * 40)

    lines.append(
        f"    {'Fold':<5s} {'Train':<22s} {'Test':<22s} "
        f"{'IS':>7s} {'OOS':>7s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s}"
    )
    lines.append(f"    {'─' * 79}")

    is_scores: list[float] = []
    oos_scores: list[float] = []

    for f in walk_forward_results:
        fold = f.get("fold", "?")
        train = f.get("train", "?")
        test = f.get("test", "?")
        is_s = f.get("is_score", 0)
        oos_s = f.get("oos_score", 0)
        sharpe = f.get("oos_sharpe", 0)
        cagr = f.get("oos_cagr", 0)
        max_dd = f.get("oos_max_dd", 0)

        is_scores.append(is_s)
        oos_scores.append(oos_s)

        lines.append(
            f"    {fold:<5} {train:<22s} {test:<22s} "
            f"{is_s:>7.4f} {oos_s:>7.4f} {sharpe:>+8.3f} "
            f"{cagr:>+7.2%} {max_dd:>7.1%}"
        )

    # Summary stats
    avg_is = float(np.mean(is_scores)) if is_scores else 0
    avg_oos = float(np.mean(oos_scores)) if oos_scores else 0
    stability = avg_oos / avg_is if avg_is > 0 else 0
    positive_oos = sum(1 for s in oos_scores if s > 0)

    lines.append("")
    lines.append(f"    Avg IS score:   {avg_is:.4f}")
    lines.append(f"    Avg OOS score:  {avg_oos:.4f}")
    lines.append(f"    OOS/IS ratio:   {stability:.3f}")
    lines.append(f"    Positive OOS:   {positive_oos}/{len(oos_scores)}")

    if stability >= 0.70:
        verdict = "STABLE"
    elif stability >= 0.40:
        verdict = "MARGINAL"
    else:
        verdict = "OVERFITTING"
    lines.append(f"    Verdict:        {verdict}")

    # OOS regime dominance per fold
    has_regime = any("oos_regime_dist" in f for f in walk_forward_results)
    if has_regime:
        lines.append("")
        lines.append("    OOS regime dominance per fold:")
        for f in walk_forward_results:
            dist = f.get("oos_regime_dist", {})
            if dist:
                parts = [
                    f"{r}:{dist[r]:.0%}" for r in REGIMES if r in dist
                ]
                lines.append(
                    f"      Fold {f.get('fold', '?')}: {' '.join(parts)}"
                )

    return "\n".join(lines)
