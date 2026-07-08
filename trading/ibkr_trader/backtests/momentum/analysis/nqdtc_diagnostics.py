"""NQDTC v2.0 strategy-specific diagnostic reports."""
from __future__ import annotations

from collections import Counter

import numpy as np


def nqdtc_full_diagnostic(
    trades: list,
    signal_events: list | None = None,
    equity_curve: np.ndarray | None = None,
    initial_equity: float = 100_000.0,
    point_value: float = 2.0,
) -> str:
    """Generate comprehensive NQDTC diagnostic report."""
    sections = []
    sections.append(_signal_funnel(trades, signal_events))
    sections.append(_entry_subtype_breakdown(trades))
    sections.append(_session_breakdown(trades))
    sections.append(_regime_breakdown(trades))
    sections.append(_chop_mode_breakdown(trades))
    sections.append(_box_analysis(trades))
    sections.append(_exit_tier_analysis(trades))
    sections.append(_exit_reason_breakdown(trades))
    # --- Entry/Exit diagnostics ---
    sections.append(_mfe_capture_analysis(trades))
    sections.append(_loser_classification(trades))
    sections.append(_be_transition_analysis(trades))
    sections.append(_quality_mult_calibration(trades))
    sections.append(_displacement_analysis(trades))
    sections.append(_score_sensitivity(trades))
    sections.append(_regime_subtype_crosstab(trades))
    sections.append(_rolling_expectancy(trades))
    sections.append(_monthly_pnl(trades))
    sections.append(_hourly_performance(trades))
    sections.append(_day_of_week(trades))
    sections.append(_stale_deep_dive(trades))
    sections.append(_streak_analysis(trades))
    # --- Weakness diagnostics ---
    sections.append(_per_breakout_attribution(trades, signal_events))
    sections.append(_expiry_decay_analysis(trades))
    sections.append(_stop_distance_analysis(trades, point_value))
    sections.append(_trade_autocorrelation(trades))
    sections.append(_r_per_bar_efficiency(trades))
    sections.append(_winner_loser_entry_profile(trades))
    sections.append(_post_tp1_runner_deep_dive(trades))
    sections.append(_equity_reconciliation(trades, equity_curve, initial_equity, point_value))
    # --- Structural diagnostics ---
    sections.append(_drawdown_episode_anatomy(trades, initial_equity, point_value))
    sections.append(_direction_asymmetry(trades))
    sections.append(_volatility_bucketed_performance(trades))
    sections.append(_trade_clustering(trades))
    return "\n\n".join(s for s in sections if s)


def _signal_funnel(trades: list, signal_events: list | None) -> str:
    """Signal funnel: evaluations -> qualified -> entries -> filled."""
    lines = ["=== NQDTC Signal Funnel ==="]
    if signal_events:
        total_eval = len(signal_events)
        passed = sum(1 for e in signal_events if e.passed_all)
        blocked = total_eval - passed
        lines.append(f"  30m breakout evaluations:  {total_eval}")
        lines.append(f"  Qualification passed:      {passed}")
        lines.append(f"  Gates blocked:             {blocked}")

        # Block reason distribution
        reasons = Counter(e.first_block_reason for e in signal_events if not e.passed_all)
        if reasons:
            lines.append("  Block reason distribution:")
            for reason, count in reasons.most_common():
                pct = count / blocked * 100 if blocked > 0 else 0
                lines.append(f"    {reason:28s} {count:5d}  ({pct:5.1f}%)")
    lines.append(f"  Trades completed:          {len(trades)}")
    return "\n".join(lines)


def _entry_subtype_breakdown(trades: list) -> str:
    """Entry subtype breakdown (A1/A2/B/C performance)."""
    lines = ["=== Entry Subtype Breakdown ==="]
    by_subtype: dict[str, list] = {}
    for t in trades:
        st = t.entry_subtype
        by_subtype.setdefault(st, []).append(t)

    header = f"  {'Subtype':20s} {'N':>5s} {'WinR':>5s} {'AvgR':>7s} {'AvgPnL':>9s} {'AvgBars':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for subtype in sorted(by_subtype):
        group = by_subtype[subtype]
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        bars = [t.bars_held_30m for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        avg_bars = np.mean(bars) if bars else 0
        lines.append(
            f"  {subtype:20s} {n:5d} {win_rate:4.0f}% {avg_r:+7.3f} "
            f"${avg_pnl:+8.0f} {avg_bars:6.1f}"
        )

    return "\n".join(lines)


def _session_breakdown(trades: list) -> str:
    """Session breakdown (ETH vs RTH)."""
    lines = ["=== Session Breakdown ==="]
    by_session: dict[str, list] = {}
    for t in trades:
        by_session.setdefault(t.session, []).append(t)

    header = f"  {'Session':10s} {'N':>5s} {'WinR':>5s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for session in sorted(by_session):
        group = by_session[session]
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        total_pnl = sum(pnls)
        lines.append(
            f"  {session:10s} {n:5d} {win_rate:4.0f}% {avg_r:+7.3f} ${total_pnl:+9.0f}"
        )

    return "\n".join(lines)


def _regime_breakdown(trades: list) -> str:
    """Regime breakdown (ALIGNED/NEUTRAL/CAUTION/RANGE/COUNTER)."""
    lines = ["=== Composite Regime Breakdown ==="]
    by_regime: dict[str, list] = {}
    for t in trades:
        by_regime.setdefault(t.composite_regime, []).append(t)

    header = f"  {'Regime':12s} {'N':>5s} {'WinR':>5s} {'AvgR':>7s} {'TotalPnL':>10s} {'TP1%':>5s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for regime in ["Aligned", "Neutral", "Caution", "Range", "Counter"]:
        group = by_regime.get(regime, [])
        if not group:
            continue
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        total_pnl = sum(pnls)
        tp1_pct = sum(1 for t in group if t.tp1_hit) / n * 100 if n > 0 else 0
        lines.append(
            f"  {regime:12s} {n:5d} {win_rate:4.0f}% {avg_r:+7.3f} "
            f"${total_pnl:+9.0f} {tp1_pct:4.0f}%"
        )

    return "\n".join(lines)


def _chop_mode_breakdown(trades: list) -> str:
    """Chop mode breakdown (NORMAL/DEGRADED/HALT)."""
    lines = ["=== Chop Mode Breakdown ==="]
    by_chop: dict[str, list] = {}
    for t in trades:
        by_chop.setdefault(t.chop_mode, []).append(t)

    for mode in ["NORMAL", "DEGRADED", "HALT"]:
        group = by_chop.get(mode, [])
        if not group:
            continue
        n = len(group)
        rs = [t.r_multiple for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        lines.append(f"  {mode:12s}  N={n:4d}  WinR={win_rate:4.0f}%  AvgR={avg_r:+.3f}")

    return "\n".join(lines)


def _box_analysis(trades: list) -> str:
    """Box analysis: width distribution, adaptive L, lifetime."""
    lines = ["=== Box Analysis ==="]
    widths = [t.box_width for t in trades if t.box_width > 0]
    ls = [t.adaptive_L for t in trades if t.adaptive_L > 0]

    if widths:
        arr = np.array(widths)
        lines.append(f"  Box width: mean={np.mean(arr):.2f}  median={np.median(arr):.2f}  "
                      f"min={np.min(arr):.2f}  max={np.max(arr):.2f}")
    if ls:
        l_counter = Counter(ls)
        lines.append("  Adaptive L usage:")
        for l_val, cnt in sorted(l_counter.items()):
            lines.append(f"    L={l_val}: {cnt} trades ({cnt/len(ls)*100:.0f}%)")

    return "\n".join(lines)


def _exit_tier_analysis(trades: list) -> str:
    """Exit tier analysis with TP hit rates."""
    lines = ["=== Exit Tier Analysis ==="]
    by_tier: dict[str, list] = {}
    for t in trades:
        by_tier.setdefault(t.exit_tier, []).append(t)

    for tier in ["Aligned", "Neutral", "Caution"]:
        group = by_tier.get(tier, [])
        if not group:
            continue
        n = len(group)
        tp1 = sum(1 for t in group if t.tp1_hit) / n * 100 if n > 0 else 0
        tp2 = sum(1 for t in group if t.tp2_hit) / n * 100 if n > 0 else 0
        tp3 = sum(1 for t in group if t.tp3_hit) / n * 100 if n > 0 else 0
        avg_r = np.mean([t.r_multiple for t in group]) if group else 0
        lines.append(
            f"  {tier:10s}  N={n:4d}  AvgR={avg_r:+.3f}  "
            f"TP1={tp1:.0f}%  TP2={tp2:.0f}%  TP3={tp3:.0f}%"
        )

    return "\n".join(lines)


def _exit_reason_breakdown(trades: list) -> str:
    """Exit reason distribution."""
    lines = ["=== Exit Reason Breakdown ==="]
    reasons = Counter(t.exit_reason for t in trades)
    for reason, count in reasons.most_common():
        group = [t for t in trades if t.exit_reason == reason]
        avg_r = np.mean([t.r_multiple for t in group]) if group else 0
        lines.append(f"  {reason:20s}  {count:5d}  AvgR={avg_r:+.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Priority 1 — Entry vs Exit Diagnosis
# ---------------------------------------------------------------------------

def _mfe_capture_analysis(trades: list) -> str:
    """MFE capture analysis: exit efficiency and milestone hit rates."""
    if not trades:
        return "=== MFE Capture Analysis ===\n  No trades."

    lines = ["=== MFE Capture Analysis ==="]

    winners = [t for t in trades if t.r_multiple > 0 and t.mfe_r > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    n = len(trades)

    # Milestone hit rates
    hit_1r = sum(1 for t in trades if t.mfe_r >= 1.0)
    hit_2r = sum(1 for t in trades if t.mfe_r >= 2.0)
    hit_3r = sum(1 for t in trades if t.mfe_r >= 3.0)
    lines.append(f"  Trades reaching +1.0R MFE: {hit_1r:4d}/{n} ({100 * hit_1r / n:.0f}%)")
    lines.append(f"  Trades reaching +2.0R MFE: {hit_2r:4d}/{n} ({100 * hit_2r / n:.0f}%)")
    lines.append(f"  Trades reaching +3.0R MFE: {hit_3r:4d}/{n} ({100 * hit_3r / n:.0f}%)")

    # Winner capture ratio (R / MFE)
    if winners:
        captures = [t.r_multiple / t.mfe_r for t in winners]
        lines.append(f"\n  Winner capture ratio (R/MFE): mean={np.mean(captures):.2f}  "
                     f"median={np.median(captures):.2f}")

        # By entry subtype
        lines.append(f"\n  Capture by entry subtype (winners):")
        header = f"    {'Subtype':20s} {'N':>5s} {'Capture':>8s} {'AvgMFE':>7s} {'AvgR':>7s}"
        lines.append(header)
        for st in sorted(set(t.entry_subtype for t in winners)):
            ct = [t for t in winners if t.entry_subtype == st]
            caps = [t.r_multiple / t.mfe_r for t in ct]
            lines.append(
                f"    {st:20s} {len(ct):5d} {np.mean(caps):8.2f} "
                f"{np.mean([t.mfe_r for t in ct]):7.2f} {np.mean([t.r_multiple for t in ct]):+7.3f}"
            )

        # By exit reason
        lines.append(f"\n  Capture by exit reason (winners):")
        header = f"    {'Reason':20s} {'N':>5s} {'Capture':>8s} {'AvgMFE':>7s}"
        lines.append(header)
        for reason in sorted(set(t.exit_reason for t in winners)):
            ct = [t for t in winners if t.exit_reason == reason]
            caps = [t.r_multiple / t.mfe_r for t in ct]
            lines.append(
                f"    {reason:20s} {len(ct):5d} {np.mean(caps):8.2f} "
                f"{np.mean([t.mfe_r for t in ct]):7.2f}"
            )

    # Loser MFE profile
    if losers:
        loser_mfes = [t.mfe_r for t in losers]
        lines.append(f"\n  Loser MFE profile:")
        lines.append(f"    Avg MFE:    {np.mean(loser_mfes):.2f}R")
        lines.append(f"    Median MFE: {np.median(loser_mfes):.2f}R")

    return "\n".join(lines)


def _loser_classification(trades: list) -> str:
    """Classify losers: right-then-stopped vs immediately wrong."""
    if not trades:
        return "=== Loser Classification ===\n  No trades."

    losers = [t for t in trades if t.r_multiple <= 0]
    if not losers:
        return "=== Loser Classification ===\n  No losing trades."

    lines = ["=== Loser Classification ==="]
    lines.append(f"  Total losers: {len(losers)}")

    right_then_stopped = [t for t in losers if t.mfe_r >= 0.5]
    immediately_wrong = [t for t in losers if t.mfe_r < 0.5]

    for label, cohort in [("Right-then-stopped (MFE >= 0.5R)", right_then_stopped),
                          ("Immediately wrong  (MFE <  0.5R)", immediately_wrong)]:
        if not cohort:
            lines.append(f"  {label}: 0 trades")
            continue
        cn = len(cohort)
        pct = 100 * cn / len(losers)
        avg_mfe = np.mean([t.mfe_r for t in cohort])
        avg_r = np.mean([t.r_multiple for t in cohort])
        lines.append(f"  {label}")
        lines.append(f"    Count: {cn} ({pct:.0f}%)  avgMFE={avg_mfe:.2f}  avgR={avg_r:+.3f}")

        # By entry subtype
        for st in sorted(set(t.entry_subtype for t in cohort)):
            sub = [t for t in cohort if t.entry_subtype == st]
            lines.append(f"      {st:20s} N={len(sub):3d}  avgR={np.mean([t.r_multiple for t in sub]):+.3f}")

    # Right-then-stopped sub-split: profit-funded before stopping?
    if right_then_stopped:
        tp1_funded = [t for t in right_then_stopped if t.tp1_hit]
        not_funded = [t for t in right_then_stopped if not t.tp1_hit]
        lines.append(f"\n  Right-then-stopped sub-split:")
        lines.append(f"    TP1 hit (profit-funded):     {len(tp1_funded):3d}  "
                     f"avgR={np.mean([t.r_multiple for t in tp1_funded]):+.3f}" if tp1_funded
                     else f"    TP1 hit (profit-funded):       0")
        lines.append(f"    TP1 NOT hit:                 {len(not_funded):3d}  "
                     f"avgR={np.mean([t.r_multiple for t in not_funded]):+.3f}" if not_funded
                     else f"    TP1 NOT hit:                   0")

    # Verdict
    if len(right_then_stopped) > 0.6 * len(losers):
        lines.append("\n  ** EXIT MANAGEMENT is the primary drag (>60% right-then-stopped)")
    elif len(immediately_wrong) > 0.6 * len(losers):
        lines.append("\n  ** ENTRY TIMING is the primary drag (>60% immediately wrong)")
    else:
        lines.append("\n  ** Mixed profile — no single dominant drag")

    return "\n".join(lines)


def _be_transition_analysis(trades: list) -> str:
    """TP1 breakeven transition: hit rate and subsequent outcomes."""
    if not trades:
        return "=== BE Transition Analysis ===\n  No trades."

    lines = ["=== BE Transition Analysis ==="]
    n = len(trades)

    tp1_trades = [t for t in trades if t.tp1_hit]
    non_tp1 = [t for t in trades if not t.tp1_hit]

    lines.append(f"  TP1 hit rate: {len(tp1_trades)}/{n} ({100 * len(tp1_trades) / n:.0f}%)")

    # TP1 hit rate by exit tier
    lines.append(f"\n  TP1 hit rate by exit tier:")
    for tier in ["Aligned", "Neutral", "Caution"]:
        tier_trades = [t for t in trades if t.exit_tier == tier]
        if not tier_trades:
            continue
        tp1_in_tier = sum(1 for t in tier_trades if t.tp1_hit)
        lines.append(f"    {tier:10s} {tp1_in_tier}/{len(tier_trades)} "
                     f"({100 * tp1_in_tier / len(tier_trades):.0f}%)")

    # Post-TP1 outcomes
    if tp1_trades:
        be_stopped = [t for t in tp1_trades if abs(t.r_multiple) < 0.3]
        reached_tp2 = [t for t in tp1_trades if t.tp2_hit]
        reached_tp3 = [t for t in tp1_trades if t.tp3_hit]

        lines.append(f"\n  Post-TP1 outcomes ({len(tp1_trades)} trades):")
        lines.append(f"    Avg final R:           {np.mean([t.r_multiple for t in tp1_trades]):+.3f}")
        lines.append(f"    Stopped near BE (|R|<0.3): {len(be_stopped)} "
                     f"({100 * len(be_stopped) / len(tp1_trades):.0f}%)")
        lines.append(f"    Reached TP2:           {len(reached_tp2)} "
                     f"({100 * len(reached_tp2) / len(tp1_trades):.0f}%)")
        lines.append(f"    Reached TP3:           {len(reached_tp3)} "
                     f"({100 * len(reached_tp3) / len(tp1_trades):.0f}%)")

        # Exit reason distribution for TP1 trades
        lines.append(f"\n    Exit reasons after TP1:")
        reasons = Counter(t.exit_reason for t in tp1_trades)
        for reason, count in reasons.most_common():
            ct = [t for t in tp1_trades if t.exit_reason == reason]
            lines.append(f"      {reason:20s} {count:4d}  avgR={np.mean([t.r_multiple for t in ct]):+.3f}")

    # Non-TP1 trades
    if non_tp1:
        lines.append(f"\n  Non-TP1 trades ({len(non_tp1)}):")
        lines.append(f"    Avg R:   {np.mean([t.r_multiple for t in non_tp1]):+.3f}")
        lines.append(f"    Avg MFE: {np.mean([t.mfe_r for t in non_tp1]):.2f}R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Priority 2 — Quality/Sizing Calibration
# ---------------------------------------------------------------------------

def _quality_mult_calibration(trades: list) -> str:
    """Quality multiplier calibration: quartile breakdown, component decomposition."""
    if not trades:
        return "=== Quality Mult Calibration ===\n  No trades."

    from strategies.momentum.nqdtc.config import CHOP_SIZE_MULT, REGIME_MULT

    lines = ["=== Quality Mult Calibration ==="]

    # --- Quartile breakdown by quality_mult ---
    qm = np.array([t.quality_mult for t in trades])
    rs = np.array([t.r_multiple for t in trades])

    if len(qm) < 4:
        lines.append("  Insufficient trades for quartile analysis.")
        return "\n".join(lines)

    q25, q50, q75 = np.percentile(qm, [25, 50, 75])
    lines.append(f"  quality_mult distribution: min={np.min(qm):.3f}  Q1={q25:.3f}  "
                 f"median={q50:.3f}  Q3={q75:.3f}  max={np.max(qm):.3f}")

    header = f"  {'Quartile':14s} {'Range':18s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s} {'AvgMFE':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    bounds = [(-np.inf, q25), (q25, q50), (q50, q75), (q75, np.inf)]
    labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    for label, (lo, hi) in zip(labels, bounds):
        if hi == np.inf:
            cohort = [t for t in trades if t.quality_mult >= lo]
        else:
            cohort = [t for t in trades if lo <= t.quality_mult < hi]
        if not cohort:
            continue
        cn = len(cohort)
        wr = sum(1 for t in cohort if t.r_multiple > 0) / cn * 100
        avg_r = np.mean([t.r_multiple for t in cohort])
        pnl = sum(t.pnl_dollars for t in cohort)
        avg_mfe = np.mean([t.mfe_r for t in cohort])
        rng = f"[{lo:.3f}, {hi:.3f})" if hi != np.inf else f"[{lo:.3f}, max]"
        lines.append(
            f"  {label:14s} {rng:18s} {cn:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f} {avg_mfe:7.2f}"
        )

    # --- Component decomposition ---
    lines.append(f"\n  Component decomposition:")
    regime_mults = []
    chop_mults = []
    disp_mults = []
    for t in trades:
        rm = REGIME_MULT.get(t.composite_regime, 0.60)
        cm = CHOP_SIZE_MULT if t.chop_mode == "DEGRADED" else 1.0
        dm = 1.0  # DISP_MULT removed in v2.1
        regime_mults.append(rm)
        chop_mults.append(cm)
        disp_mults.append(dm)

    regime_arr = np.array(regime_mults)
    chop_arr = np.array(chop_mults)
    disp_arr = np.array(disp_mults)

    lines.append(f"    {'Component':14s} {'Mean':>7s} {'Median':>7s} {'Corr(R)':>8s}")
    for name, arr in [("regime_mult", regime_arr), ("chop_mult", chop_arr), ("disp_mult", disp_arr)]:
        if len(set(arr)) > 1:
            corr = np.corrcoef(arr, rs)[0, 1]
        else:
            corr = 0.0
        lines.append(f"    {name:14s} {np.mean(arr):7.3f} {np.median(arr):7.3f} {corr:+8.3f}")

    # --- Weighted vs equal-weight E[R] ---
    weighted_r = np.array([t.r_multiple * t.quality_mult for t in trades])
    eq_r = np.mean(rs)
    wt_r = np.sum(weighted_r) / np.sum(qm) if np.sum(qm) > 0 else 0.0
    delta = wt_r - eq_r
    verdict = "HELPS" if delta > 0.01 else "HURTS" if delta < -0.01 else "NEUTRAL"
    lines.append(f"\n  Weighted E[R] (sum(R*qm)/sum(qm)): {wt_r:+.3f}")
    lines.append(f"  Equal-weight E[R]:                  {eq_r:+.3f}")
    lines.append(f"  Delta:                              {delta:+.3f} ({verdict})")

    # --- Expiry mult impact ---
    high_exp = [t for t in trades if t.expiry_mult >= 0.8]
    low_exp = [t for t in trades if t.expiry_mult < 0.8]
    if high_exp and low_exp:
        lines.append(f"\n  Expiry mult impact:")
        lines.append(f"    expiry_mult >= 0.8: N={len(high_exp):4d}  "
                     f"avgR={np.mean([t.r_multiple for t in high_exp]):+.3f}")
        lines.append(f"    expiry_mult <  0.8: N={len(low_exp):4d}  "
                     f"avgR={np.mean([t.r_multiple for t in low_exp]):+.3f}")

    return "\n".join(lines)


def _displacement_analysis(trades: list) -> str:
    """Displacement at entry: distribution and quartile performance."""
    if not trades:
        return "=== Displacement Analysis ===\n  No trades."

    disps = np.array([t.displacement_at_entry for t in trades])
    if len(disps) < 4:
        return "=== Displacement Analysis ===\n  Insufficient trades."

    lines = ["=== Displacement Analysis ==="]

    q25, q50, q75 = np.percentile(disps, [25, 50, 75])
    lines.append(f"  displacement distribution: min={np.min(disps):.2f}  Q25={q25:.2f}  "
                 f"median={q50:.2f}  Q75={q75:.2f}  max={np.max(disps):.2f}")

    header = f"  {'Quartile':14s} {'Range':18s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s} {'AvgMFE':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    bounds = [(-np.inf, q25), (q25, q50), (q50, q75), (q75, np.inf)]
    labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
    for label, (lo, hi) in zip(labels, bounds):
        if hi == np.inf:
            cohort = [t for t in trades if t.displacement_at_entry >= lo]
        else:
            cohort = [t for t in trades if lo <= t.displacement_at_entry < hi]
        if not cohort:
            continue
        cn = len(cohort)
        wr = sum(1 for t in cohort if t.r_multiple > 0) / cn * 100
        avg_r = np.mean([t.r_multiple for t in cohort])
        pnl = sum(t.pnl_dollars for t in cohort)
        avg_mfe = np.mean([t.mfe_r for t in cohort])
        rng = f"[{lo:.2f}, {hi:.2f})" if hi != np.inf else f"[{lo:.2f}, max]"
        lines.append(
            f"  {label:14s} {rng:18s} {cn:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f} {avg_mfe:7.2f}"
        )

    # Correlation
    rs = np.array([t.r_multiple for t in trades])
    if len(rs) > 2:
        corr = np.corrcoef(disps, rs)[0, 1]
        lines.append(f"\n  Correlation (displacement vs R): {corr:+.3f}")

    # Regime-stratified view
    regimes = sorted(set(t.composite_regime for t in trades))
    if len(regimes) > 1:
        lines.append(f"\n  Displacement quartile performance by regime:")
        for regime in regimes:
            rtrades = [t for t in trades if t.composite_regime == regime]
            if len(rtrades) < 4:
                continue
            r_disps = np.array([t.displacement_at_entry for t in rtrades])
            r_q50 = np.median(r_disps)
            above = [t for t in rtrades if t.displacement_at_entry >= r_q50]
            below = [t for t in rtrades if t.displacement_at_entry < r_q50]
            lines.append(
                f"    {regime:12s}  above-med N={len(above):3d} avgR={np.mean([t.r_multiple for t in above]):+.3f}  "
                f"below-med N={len(below):3d} avgR={np.mean([t.r_multiple for t in below]):+.3f}"
            )

    return "\n".join(lines)


def _score_sensitivity(trades: list) -> str:
    """Score at entry: distribution and performance by band."""
    if not trades:
        return "=== Score Sensitivity ===\n  No trades."

    lines = ["=== Score Sensitivity ==="]

    # Bands
    bands = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 99.0)]
    band_labels = ["[2.0,2.5)", "[2.5,3.0)", "[3.0,3.5)", "[3.5,4.0)", "[4.0+)"]

    for mode_label, mode_val in [("NORMAL", "NORMAL"), ("DEGRADED", "DEGRADED")]:
        mode_trades = [t for t in trades if t.chop_mode == mode_val]
        if not mode_trades:
            continue

        lines.append(f"\n  {mode_label} mode (N={len(mode_trades)}):")
        header = f"    {'Band':12s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s} {'AvgMFE':>7s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))

        for bl, (lo, hi) in zip(band_labels, bands):
            cohort = [t for t in mode_trades if lo <= t.score_at_entry < hi]
            if not cohort:
                continue
            cn = len(cohort)
            wr = sum(1 for t in cohort if t.r_multiple > 0) / cn * 100
            avg_r = np.mean([t.r_multiple for t in cohort])
            pnl = sum(t.pnl_dollars for t in cohort)
            avg_mfe = np.mean([t.mfe_r for t in cohort])
            lines.append(
                f"    {bl:12s} {cn:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f} {avg_mfe:7.2f}"
            )

        # Correlation for this mode
        scores = np.array([t.score_at_entry for t in mode_trades])
        rs = np.array([t.r_multiple for t in mode_trades])
        if len(rs) > 2 and len(set(scores)) > 1:
            corr = np.corrcoef(scores, rs)[0, 1]
            lines.append(f"    Correlation (score vs R): {corr:+.3f}")

        # Threshold hint
        lowest_band = [t for t in mode_trades if bands[0][0] <= t.score_at_entry < bands[0][1]]
        if lowest_band and np.mean([t.r_multiple for t in lowest_band]) < 0:
            lines.append(f"    ** HINT: lowest band avgR is negative — consider raising threshold")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Priority 3 — Interaction Effects
# ---------------------------------------------------------------------------

def _regime_subtype_crosstab(trades: list) -> str:
    """Regime x entry subtype cross-tab with restrict/value flags."""
    if not trades:
        return "=== Regime x Subtype Cross-Tab ===\n  No trades."

    lines = ["=== Regime x Subtype Cross-Tab ==="]

    subtypes = sorted(set(t.entry_subtype for t in trades))
    regimes = ["Aligned", "Neutral", "Caution", "Range", "Counter"]
    regimes = [r for r in regimes if any(t.composite_regime == r for t in trades)]

    # Header
    r_header = "  ".join(f"{r:>18s}" for r in regimes)
    lines.append(f"  {'Subtype':20s}  {r_header}")
    lines.append("  " + "-" * (22 + 20 * len(regimes)))

    flags: list[str] = []

    for st in subtypes:
        cells = []
        for regime in regimes:
            ct = [t for t in trades if t.entry_subtype == st and t.composite_regime == regime]
            if not ct:
                cells.append(f"{'--':>18s}")
            else:
                cn = len(ct)
                wr = sum(1 for t in ct if t.r_multiple > 0) / cn * 100
                avg_r = np.mean([t.r_multiple for t in ct])
                cells.append(f"{cn:3d} {wr:3.0f}% {avg_r:+.2f}")

                # Flag cells
                if cn >= 5 and avg_r < -0.2:
                    flags.append(f"  ** RESTRICT candidate: {st} x {regime} (N={cn}, avgR={avg_r:+.3f})")
                elif cn >= 5 and avg_r > 0.3:
                    flags.append(f"  ** HIGH VALUE: {st} x {regime} (N={cn}, avgR={avg_r:+.3f})")
        lines.append(f"  {st:20s}  {'  '.join(cells)}")

    # Avg quality_mult per cell
    lines.append(f"\n  Avg quality_mult per cell:")
    for st in subtypes:
        cells = []
        for regime in regimes:
            ct = [t for t in trades if t.entry_subtype == st and t.composite_regime == regime]
            if not ct:
                cells.append(f"{'--':>18s}")
            else:
                avg_qm = np.mean([t.quality_mult for t in ct])
                cells.append(f"{avg_qm:18.3f}")
        lines.append(f"  {st:20s}  {'  '.join(cells)}")

    if flags:
        lines.append("")
        lines.extend(flags)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Priority 4 — Temporal Patterns
# ---------------------------------------------------------------------------

def _rolling_expectancy(trades: list, window: int = 40) -> str:
    """Rolling N-trade expectancy to detect performance drift."""
    if not trades or len(trades) < window:
        return f"=== Rolling Expectancy (window={window}) ===\n  Insufficient trades (need >= {window})."

    lines = [f"=== Rolling Expectancy (window={window}) ==="]
    sorted_trades = sorted(
        [t for t in trades if t.entry_time],
        key=lambda t: t.entry_time,
    )
    if len(sorted_trades) < window:
        lines.append(f"  Insufficient dated trades (need >= {window}).")
        return "\n".join(lines)

    rs = [t.r_multiple for t in sorted_trades]
    rolling = []
    for i in range(len(rs) - window + 1):
        rolling.append(np.mean(rs[i:i + window]))

    rolling = np.array(rolling)
    lines.append(f"  Windows computed: {len(rolling)}")
    lines.append(f"  Rolling E[R]:  min={np.min(rolling):+.3f}  max={np.max(rolling):+.3f}  "
                 f"current={rolling[-1]:+.3f}")

    neg_windows = int(np.sum(rolling < 0))
    lines.append(f"  Negative-expectancy windows: {neg_windows} ({100 * neg_windows / len(rolling):.0f}%)")

    # Trend: compare first half to second half
    mid = len(rolling) // 2
    first_half = np.mean(rolling[:mid])
    second_half = np.mean(rolling[mid:])
    delta = second_half - first_half
    trend = "IMPROVING" if delta > 0.05 else "DEGRADING" if delta < -0.05 else "STABLE"
    lines.append(f"  Trend: {trend} (1st half={first_half:+.3f}, 2nd half={second_half:+.3f}, "
                 f"delta={delta:+.3f})")

    # Worst rolling window with dates
    worst_idx = int(np.argmin(rolling))
    worst_start = sorted_trades[worst_idx].entry_time
    worst_end = sorted_trades[worst_idx + window - 1].entry_time
    lines.append(f"  Worst window: E[R]={rolling[worst_idx]:+.3f}  "
                 f"({worst_start.strftime('%Y-%m-%d')} to {worst_end.strftime('%Y-%m-%d')})")

    return "\n".join(lines)


def _monthly_pnl(trades: list) -> str:
    """Monthly P&L breakdown."""
    if not trades:
        return "=== Monthly P&L ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Monthly P&L ===\n  No dated trades."

    lines = ["=== Monthly P&L ==="]
    header = f"  {'Month':10s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'NetPnL':>10s} {'CumPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_month: dict[str, list] = {}
    for t in dated:
        key = t.entry_time.strftime("%Y-%m")
        by_month.setdefault(key, []).append(t)

    cum_pnl = 0.0
    for month in sorted(by_month):
        group = by_month[month]
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        net = sum(t.pnl_dollars for t in group)
        cum_pnl += net
        lines.append(
            f"  {month:10s} {n:5d} {wr:5.0f}% {avg_r:+7.3f} ${net:+9.0f} ${cum_pnl:+9.0f}"
        )

    # Winning/losing months
    monthly_pnls = [sum(t.pnl_dollars for t in by_month[m]) for m in sorted(by_month)]
    win_months = sum(1 for p in monthly_pnls if p > 0)
    total_months = len(monthly_pnls)
    lines.append(f"\n  Winning months: {win_months}/{total_months} "
                 f"({100 * win_months / total_months:.0f}%)")

    if monthly_pnls:
        best_idx = int(np.argmax(monthly_pnls))
        worst_idx = int(np.argmin(monthly_pnls))
        months_sorted = sorted(by_month)
        lines.append(f"  Best month:  {months_sorted[best_idx]} (${monthly_pnls[best_idx]:+,.0f})")
        lines.append(f"  Worst month: {months_sorted[worst_idx]} (${monthly_pnls[worst_idx]:+,.0f})")

    return "\n".join(lines)


def _hourly_performance(trades: list) -> str:
    """Performance by hour of entry (ET)."""
    if not trades:
        return "=== Hourly Performance ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Hourly Performance ===\n  No dated trades."

    lines = ["=== Hourly Performance (ET) ==="]

    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        lines.append("  zoneinfo not available; skipping.")
        return "\n".join(lines)

    header = f"  {'Hour':6s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_hour: dict[int, list] = {}
    for t in dated:
        try:
            h = t.entry_time.astimezone(et).hour
        except Exception:
            h = t.entry_time.hour
        by_hour.setdefault(h, []).append(t)

    for hour in sorted(by_hour):
        group = by_hour[hour]
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        pnl = sum(t.pnl_dollars for t in group)
        lines.append(f"  {hour:02d}:00 {n:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f}")

    return "\n".join(lines)


def _day_of_week(trades: list) -> str:
    """Performance by day of week."""
    if not trades:
        return "=== Day-of-Week Analysis ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Day-of-Week Analysis ===\n  No dated trades."

    lines = ["=== Day-of-Week Analysis ==="]
    header = f"  {'Day':10s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day: dict[int, list] = {}
    for t in dated:
        dow = t.entry_time.weekday()
        by_day.setdefault(dow, []).append(t)

    for dow in sorted(by_day):
        group = by_day[dow]
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        pnl = sum(t.pnl_dollars for t in group)
        lines.append(f"  {day_names[dow]:10s} {n:5d} {wr:5.0f}% {avg_r:+7.3f} ${pnl:+9.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Priority 5 — Supporting Diagnostics
# ---------------------------------------------------------------------------

def _stale_deep_dive(trades: list) -> str:
    """Detailed analysis of stale exits."""
    if not trades:
        return "=== Stale Exit Deep-Dive ===\n  No trades."

    stale = [t for t in trades if t.exit_reason == "STALE"]
    non_stale = [t for t in trades if t.exit_reason != "STALE"]

    lines = ["=== Stale Exit Deep-Dive ==="]
    lines.append(f"  Stale exits: {len(stale)}/{len(trades)} ({100 * len(stale) / len(trades):.0f}%)")

    if not stale:
        lines.append("  No stale exits.")
        return "\n".join(lines)

    lines.append(f"\n  Stale trade profile:")
    lines.append(f"    Avg R:        {np.mean([t.r_multiple for t in stale]):+.3f}")
    lines.append(f"    Avg MFE:      {np.mean([t.mfe_r for t in stale]):.2f}R")
    lines.append(f"    Avg MAE:      {np.mean([t.mae_r for t in stale]):.2f}R")
    lines.append(f"    Avg bars held: {np.mean([t.bars_held_30m for t in stale]):.1f}")

    # Were stale trades ever right?
    stale_right = [t for t in stale if t.mfe_r >= 0.5]
    lines.append(f"    Had MFE >= 0.5R: {len(stale_right)}/{len(stale)} "
                 f"({100 * len(stale_right) / len(stale):.0f}%)")

    # Stale trades where tp1_hit was True (profit-funded then went stale)
    stale_tp1 = [t for t in stale if t.tp1_hit]
    if stale_tp1:
        lines.append(f"    TP1 hit then stale (chandelier issue): {len(stale_tp1)} trades  "
                     f"avgR={np.mean([t.r_multiple for t in stale_tp1]):+.3f}")

    # Breakdown by chop_mode
    lines.append(f"\n  Stale exits by chop mode:")
    for mode in ["NORMAL", "DEGRADED"]:
        ct = [t for t in stale if t.chop_mode == mode]
        if not ct:
            continue
        lines.append(f"    {mode:12s} N={len(ct):3d}  avgR={np.mean([t.r_multiple for t in ct]):+.3f}  "
                     f"avgMFE={np.mean([t.mfe_r for t in ct]):.2f}  "
                     f"avgBars={np.mean([t.bars_held_30m for t in ct]):.1f}")

    # Stale vs non-stale comparison
    if non_stale:
        lines.append(f"\n  Stale vs non-stale comparison:")
        lines.append(f"    Stale:     N={len(stale):4d}  avgR={np.mean([t.r_multiple for t in stale]):+.3f}  "
                     f"avgMFE={np.mean([t.mfe_r for t in stale]):.2f}")
        lines.append(f"    Non-stale: N={len(non_stale):4d}  avgR={np.mean([t.r_multiple for t in non_stale]):+.3f}  "
                     f"avgMFE={np.mean([t.mfe_r for t in non_stale]):.2f}")

    return "\n".join(lines)


def _streak_analysis(trades: list) -> str:
    """Win/loss streak analysis."""
    if not trades:
        return "=== Streak Analysis ===\n  No trades."

    lines = ["=== Streak Analysis ==="]
    max_win = max_loss = cur_win = cur_loss = 0
    worst_loss_streak_pnl = 0.0
    cur_loss_pnl = 0.0

    for t in trades:
        if t.r_multiple > 0:
            cur_win += 1
            cur_loss = 0
            cur_loss_pnl = 0.0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            cur_loss_pnl += t.pnl_dollars
            max_loss = max(max_loss, cur_loss)
            worst_loss_streak_pnl = min(worst_loss_streak_pnl, cur_loss_pnl)

    lines.append(f"  Max win streak:  {max_win}")
    lines.append(f"  Max loss streak: {max_loss}")
    lines.append(f"  Worst consecutive-loss P&L: ${worst_loss_streak_pnl:+,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Weakness diagnostics — actionable sections for further improvement
# ---------------------------------------------------------------------------

def _per_breakout_attribution(trades: list, signal_events: list | None) -> str:
    """Per-breakout P&L attribution — reveals concentration risk.

    With only 2 breakouts generating 213 trades, this shows whether
    profitability is balanced or concentrated in one event.
    """
    if not trades:
        return "=== Per-Breakout Attribution ===\n  No trades."

    dated = sorted(
        [t for t in trades if t.entry_time],
        key=lambda t: t.entry_time,
    )
    if not dated:
        return "=== Per-Breakout Attribution ===\n  No dated trades."

    lines = ["=== Per-Breakout Attribution ==="]

    # Identify breakout boundaries from signal_events (passed_all=True)
    breakout_starts = []
    if signal_events:
        for e in sorted(signal_events, key=lambda e: e.timestamp):
            if e.passed_all:
                breakout_starts.append(e.timestamp)

    if not breakout_starts:
        # Fallback: detect gaps > 48h between trades as breakout boundaries
        breakout_starts = [dated[0].entry_time]
        for i in range(1, len(dated)):
            gap = (dated[i].entry_time - dated[i - 1].exit_time).total_seconds()
            if gap > 48 * 3600:  # 48 hour gap
                breakout_starts.append(dated[i].entry_time)
        lines.append("  (Breakout boundaries inferred from trade gaps > 48h)")

    # Assign trades to breakouts (each trade belongs to the latest breakout before it)
    breakout_groups: dict[int, list] = {}
    for t in dated:
        bo_idx = 0
        for i, bs in enumerate(breakout_starts):
            if t.entry_time >= bs:
                bo_idx = i
        breakout_groups.setdefault(bo_idx, []).append(t)

    lines.append(f"  Breakout events identified: {len(breakout_starts)}")

    header = (f"  {'Breakout':10s} {'Start':>12s} {'N':>5s} {'WR':>5s} "
              f"{'AvgR':>7s} {'TotalPnL':>10s} {'AvgMFE':>7s} {'AvgMAE':>7s} {'Days':>5s}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    total_pnls = []
    for idx in sorted(breakout_groups):
        group = breakout_groups[idx]
        n = len(group)
        rs = [t.r_multiple for t in group]
        wr = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs)
        pnl = sum(t.pnl_dollars for t in group)
        total_pnls.append(pnl)
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_mae = np.mean([t.mae_r for t in group])
        start = group[0].entry_time.strftime("%Y-%m-%d")
        end = group[-1].exit_time
        days = (end - group[0].entry_time).total_seconds() / 86400 if end else 0
        lines.append(
            f"  BO-{idx + 1:02d}     {start:>12s} {n:5d} {wr:4.0f}% "
            f"{avg_r:+7.3f} ${pnl:+9.0f} {avg_mfe:7.2f} {avg_mae:7.2f} {days:5.0f}"
        )

    # Concentration metrics
    if len(total_pnls) >= 2:
        total = sum(total_pnls)
        if total > 0:
            max_pct = max(total_pnls) / total * 100
            lines.append(f"\n  Concentration: largest breakout = {max_pct:.0f}% of total P&L")
            if max_pct > 70:
                lines.append("  ** WARNING: >70% of profit from single breakout — high concentration risk")
        # Per-breakout subtypes
        lines.append(f"\n  Subtype mix per breakout:")
        for idx in sorted(breakout_groups):
            group = breakout_groups[idx]
            subtypes = Counter(t.entry_subtype for t in group)
            st_str = "  ".join(f"{st}={cnt}" for st, cnt in subtypes.most_common())
            lines.append(f"    BO-{idx + 1:02d}: {st_str}")

    return "\n".join(lines)


def _expiry_decay_analysis(trades: list) -> str:
    """Expiry mult decay: do late entries in a breakout cycle still work?

    expiry_mult decays from 1.0 toward DECAY_FLOOR (0.30) as the breakout
    ages. This section bins trades by expiry_mult to reveal if early entries
    are systematically better, which would inform tighter expiry or faster
    decay.
    """
    if not trades:
        return "=== Expiry Decay Analysis ===\n  No trades."

    lines = ["=== Expiry Decay Analysis ==="]

    exp_vals = [t.expiry_mult for t in trades]
    lines.append(
        f"  expiry_mult: min={min(exp_vals):.2f}  median={np.median(exp_vals):.2f}  "
        f"max={max(exp_vals):.2f}"
    )

    # Bin into ranges
    bins = [(0.0, 0.40, "Stale [0.0-0.4)"), (0.40, 0.60, "Fading [0.4-0.6)"),
            (0.60, 0.80, "Fresh [0.6-0.8)"), (0.80, 1.01, "Hot [0.8-1.0]")]

    header = f"  {'Band':18s} {'N':>5s} {'WR':>5s} {'AvgR':>7s} {'TotalPnL':>10s} {'AvgMFE':>7s} {'AvgBars':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for lo, hi, label in bins:
        cohort = [t for t in trades if lo <= t.expiry_mult < hi]
        if not cohort:
            continue
        cn = len(cohort)
        rs = [t.r_multiple for t in cohort]
        wr = sum(1 for r in rs if r > 0) / cn * 100
        avg_r = np.mean(rs)
        pnl = sum(t.pnl_dollars for t in cohort)
        avg_mfe = np.mean([t.mfe_r for t in cohort])
        avg_bars = np.mean([t.bars_held_30m for t in cohort])
        lines.append(
            f"  {label:18s} {cn:5d} {wr:4.0f}% {avg_r:+7.3f} "
            f"${pnl:+9.0f} {avg_mfe:7.2f} {avg_bars:6.1f}"
        )

    # Correlation
    rs = np.array([t.r_multiple for t in trades])
    exp = np.array(exp_vals)
    if len(set(exp)) > 1 and len(rs) > 2:
        corr = np.corrcoef(exp, rs)[0, 1]
        lines.append(f"\n  Correlation (expiry_mult vs R): {corr:+.3f}")
        if corr < -0.1:
            lines.append("  ** Late entries perform WORSE — consider faster decay or tighter expiry")
        elif corr > 0.1:
            lines.append("  ** Late entries perform BETTER — continuation has momentum advantage")

    # Expiry by entry subtype
    lines.append(f"\n  Avg expiry_mult by subtype:")
    for st in sorted(set(t.entry_subtype for t in trades)):
        sub = [t for t in trades if t.entry_subtype == st]
        lines.append(f"    {st:20s} N={len(sub):4d}  avg_exp={np.mean([t.expiry_mult for t in sub]):.3f}")

    return "\n".join(lines)


def _stop_distance_analysis(trades: list, point_value: float = 2.0) -> str:
    """Initial stop distance distribution and outcome correlation.

    Reveals whether stops are too tight (many immediate stops) or too wide
    (excessive risk per trade). Bins by stop width in points and ATR-relative.
    """
    if not trades:
        return "=== Stop Distance Analysis ===\n  No trades."

    lines = ["=== Stop Distance Analysis ==="]

    # Stop distance in points
    dists = [abs(t.entry_price - t.initial_stop) for t in trades]
    dists_arr = np.array(dists)
    risk_dollars = [d * point_value * t.qty for d, t in zip(dists, trades)]

    lines.append(f"  Stop distance (points): "
                 f"mean={np.mean(dists_arr):.2f}  median={np.median(dists_arr):.2f}  "
                 f"min={np.min(dists_arr):.2f}  max={np.max(dists_arr):.2f}")
    lines.append(f"  Risk per trade ($): "
                 f"mean=${np.mean(risk_dollars):,.0f}  median=${np.median(risk_dollars):,.0f}")

    # Quintile analysis by stop distance
    if len(dists_arr) >= 10:
        q20, q40, q60, q80 = np.percentile(dists_arr, [20, 40, 60, 80])
        bins = [
            (0, q20, "Tight (0-P20)"),
            (q20, q40, "Narrow (P20-P40)"),
            (q40, q60, "Medium (P40-P60)"),
            (q60, q80, "Wide (P60-P80)"),
            (q80, np.inf, "Very wide (P80+)"),
        ]

        header = (f"  {'Band':20s} {'Range':>14s} {'N':>5s} {'WR':>5s} "
                  f"{'AvgR':>7s} {'TotalPnL':>10s} {'AvgMFE':>7s}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for lo, hi, label in bins:
            cohort = [(t, d) for t, d in zip(trades, dists) if lo <= d < hi]
            if not cohort:
                continue
            ct = [t for t, _ in cohort]
            cn = len(ct)
            rs = [t.r_multiple for t in ct]
            wr = sum(1 for r in rs if r > 0) / cn * 100
            avg_r = np.mean(rs)
            pnl = sum(t.pnl_dollars for t in ct)
            avg_mfe = np.mean([t.mfe_r for t in ct])
            rng = f"{lo:.1f}-{hi:.1f}" if hi != np.inf else f"{lo:.1f}+"
            lines.append(
                f"  {label:20s} {rng:>14s} {cn:5d} {wr:4.0f}% "
                f"{avg_r:+7.3f} ${pnl:+9.0f} {avg_mfe:7.2f}"
            )

    # Correlation: stop distance vs R
    rs = np.array([t.r_multiple for t in trades])
    if len(set(dists_arr)) > 1:
        corr = np.corrcoef(dists_arr, rs)[0, 1]
        lines.append(f"\n  Correlation (stop_dist vs R): {corr:+.3f}")
        if corr > 0.1:
            lines.append("  ** Wider stops correlate with better R — tight stops may be prematurely stopping out")
        elif corr < -0.1:
            lines.append("  ** Tighter stops correlate with better R — wide stops trap losing trades")

    # Stop distance by subtype
    lines.append(f"\n  Stop distance by entry subtype:")
    for st in sorted(set(t.entry_subtype for t in trades)):
        sub = [(t, d) for t, d in zip(trades, dists) if t.entry_subtype == st]
        sub_dists = [d for _, d in sub]
        sub_r = [t.r_multiple for t, _ in sub]
        lines.append(
            f"    {st:20s} N={len(sub):4d}  "
            f"mean_dist={np.mean(sub_dists):6.1f}pts  avgR={np.mean(sub_r):+.3f}"
        )

    # "Immediately stopped" rate: trades where MAE > 0.8R (stop nearly hit immediately)
    tight_stops = [t for t in trades if t.mae_r > 0.8 and t.r_multiple <= 0]
    if trades:
        lines.append(f"\n  Immediately stopped (MAE > 0.8R, loser): {len(tight_stops)}/{len(trades)} "
                     f"({100 * len(tight_stops) / len(trades):.0f}%)")

    return "\n".join(lines)


def _trade_autocorrelation(trades: list) -> str:
    """Consecutive trade outcome correlation — loss clustering.

    If losses cluster (positive autocorrelation), adaptive cooldown or
    reduced sizing after N consecutive losses could improve risk-adjusted returns.
    """
    if not trades or len(trades) < 20:
        return "=== Trade Autocorrelation ===\n  Insufficient trades (need >= 20)."

    dated = sorted(
        [t for t in trades if t.entry_time],
        key=lambda t: t.entry_time,
    )
    if len(dated) < 20:
        return "=== Trade Autocorrelation ===\n  Insufficient dated trades."

    lines = ["=== Trade Autocorrelation ==="]

    rs = np.array([t.r_multiple for t in dated])
    n = len(rs)

    # Lag-1 autocorrelation of R multiples
    if n > 2:
        lag1 = np.corrcoef(rs[:-1], rs[1:])[0, 1]
        lines.append(f"  Lag-1 R autocorrelation: {lag1:+.3f}")
        if lag1 > 0.15:
            lines.append("  ** Significant positive clustering — losses follow losses")
            lines.append("     Consider: adaptive cooldown or size reduction after consecutive losses")
        elif lag1 < -0.15:
            lines.append("  ** Mean-reverting — losses tend to be followed by wins")
        else:
            lines.append("  Outcomes appear independent (|corr| < 0.15)")

    # Win/loss transition matrix
    wins = rs > 0
    ww = sum(1 for i in range(n - 1) if wins[i] and wins[i + 1])
    wl = sum(1 for i in range(n - 1) if wins[i] and not wins[i + 1])
    lw = sum(1 for i in range(n - 1) if not wins[i] and wins[i + 1])
    ll = sum(1 for i in range(n - 1) if not wins[i] and not wins[i + 1])

    lines.append(f"\n  Transition matrix (N={n - 1} transitions):")
    lines.append(f"                 Next Win    Next Loss")
    lines.append(f"    After Win:   {ww:5d} ({100 * ww / max(1, ww + wl):.0f}%)   "
                 f"{wl:5d} ({100 * wl / max(1, ww + wl):.0f}%)")
    lines.append(f"    After Loss:  {lw:5d} ({100 * lw / max(1, lw + ll):.0f}%)   "
                 f"{ll:5d} ({100 * ll / max(1, lw + ll):.0f}%)")

    # Performance after N consecutive losses
    lines.append(f"\n  Performance after consecutive losses:")
    header = f"    {'After':12s} {'NextN':>5s} {'NextWR':>7s} {'NextAvgR':>9s}"
    lines.append(header)

    for streak_len in [1, 2, 3, 5]:
        # Find trades that follow exactly streak_len consecutive losses
        next_trades = []
        consec = 0
        for i in range(n):
            if rs[i] <= 0:
                consec += 1
            else:
                consec = 0
            if i > 0 and consec == 0:
                # This trade is a win that broke a loss streak
                pass
            if i >= streak_len:
                # Check if previous streak_len trades were all losses
                prev_all_loss = all(rs[i - j - 1] <= 0 for j in range(streak_len))
                if prev_all_loss:
                    next_trades.append(dated[i])

        if next_trades:
            nr = [t.r_multiple for t in next_trades]
            nw = sum(1 for r in nr if r > 0) / len(nr) * 100
            lines.append(
                f"    {streak_len} losses:   {len(next_trades):5d} {nw:6.0f}% {np.mean(nr):+9.3f}"
            )

    return "\n".join(lines)


def _r_per_bar_efficiency(trades: list) -> str:
    """R earned per 30m bar held — identifies dead time and holding inefficiency.

    Trades with high R but also high bars_held may be spending too long
    in dead zones. Trades with high R/bar are the most capital-efficient.
    """
    if not trades:
        return "=== R/Bar Efficiency ===\n  No trades."

    lines = ["=== R/Bar Efficiency ==="]

    # Filter trades with positive hold time
    valid = [t for t in trades if t.bars_held_30m > 0]
    if not valid:
        lines.append("  No trades with positive hold time.")
        return "\n".join(lines)

    r_per_bar = [t.r_multiple / t.bars_held_30m for t in valid]
    rpb = np.array(r_per_bar)

    lines.append(f"  R/bar: mean={np.mean(rpb):+.4f}  median={np.median(rpb):+.4f}  "
                 f"std={np.std(rpb):.4f}")

    # By winner/loser
    winners = [t for t in valid if t.r_multiple > 0]
    losers = [t for t in valid if t.r_multiple <= 0]
    if winners:
        w_rpb = [t.r_multiple / t.bars_held_30m for t in winners]
        lines.append(f"  Winners:  R/bar mean={np.mean(w_rpb):+.4f}  "
                     f"avg_hold={np.mean([t.bars_held_30m for t in winners]):.1f} bars")
    if losers:
        l_rpb = [t.r_multiple / t.bars_held_30m for t in losers]
        lines.append(f"  Losers:   R/bar mean={np.mean(l_rpb):+.4f}  "
                     f"avg_hold={np.mean([t.bars_held_30m for t in losers]):.1f} bars")

    # Hold time buckets with R/bar
    buckets = [(1, 2, "1-2 bars"), (3, 4, "3-4 bars"), (5, 8, "5-8 bars"),
               (9, 16, "9-16 bars"), (17, 999, "17+ bars")]

    header = f"  {'Hold':12s} {'N':>5s} {'WR':>5s} {'AvgR':>7s} {'R/bar':>7s} {'TotalPnL':>10s}"
    lines.append(f"\n{header}")
    lines.append("  " + "-" * (len(header) - 2))

    for lo, hi, label in buckets:
        cohort = [t for t in valid if lo <= t.bars_held_30m <= hi]
        if not cohort:
            continue
        cn = len(cohort)
        rs = [t.r_multiple for t in cohort]
        wr = sum(1 for r in rs if r > 0) / cn * 100
        avg_r = np.mean(rs)
        avg_rpb = np.mean([t.r_multiple / t.bars_held_30m for t in cohort])
        pnl = sum(t.pnl_dollars for t in cohort)
        lines.append(
            f"  {label:12s} {cn:5d} {wr:4.0f}% {avg_r:+7.3f} {avg_rpb:+7.4f} ${pnl:+9.0f}"
        )

    # Optimal hold time hint
    if len(valid) >= 20:
        # Find hold time that maximizes cumulative R
        max_hold = max(t.bars_held_30m for t in valid)
        best_cum_r = -999
        best_cutoff = max_hold
        for cutoff in range(1, min(max_hold + 1, 50)):
            # Hypothetical: exit all trades at this bar if still open
            cum_r = sum(min(t.r_multiple, t.mfe_r) if t.bars_held_30m > cutoff else t.r_multiple
                        for t in valid)
            # Simplified: just sum R of trades held <= cutoff vs longer
            short_trades = [t for t in valid if t.bars_held_30m <= cutoff]
            if short_trades and sum(t.r_multiple for t in short_trades) > best_cum_r:
                best_cum_r = sum(t.r_multiple for t in short_trades)
                best_cutoff = cutoff

        lines.append(f"\n  Best cumulative R comes from trades held <= {best_cutoff} bars "
                     f"(cumR={best_cum_r:+.1f})")

    return "\n".join(lines)


def _winner_loser_entry_profile(trades: list) -> str:
    """Winner vs loser entry context comparison.

    Identifies which entry conditions (chop mode, expiry, displacement,
    score, hour) differentiate winners from losers. Useful for finding
    filterable signals.
    """
    if not trades:
        return "=== Winner/Loser Entry Profile ===\n  No trades."

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    if not winners or not losers:
        return "=== Winner/Loser Entry Profile ===\n  Need both winners and losers."

    lines = ["=== Winner/Loser Entry Profile ==="]
    lines.append(f"  Winners: {len(winners)}  Losers: {len(losers)}")

    # Compare context metrics
    metrics = [
        ("quality_mult", lambda t: t.quality_mult),
        ("expiry_mult", lambda t: t.expiry_mult),
        ("score_at_entry", lambda t: t.score_at_entry),
        ("displacement", lambda t: t.displacement_at_entry),
        ("rvol_at_entry", lambda t: t.rvol_at_entry),
        ("disp_norm", lambda t: t.disp_norm_at_entry),
        ("box_width", lambda t: t.box_width),
        ("bars_held_30m", lambda t: t.bars_held_30m),
    ]

    header = f"  {'Metric':18s} {'Win mean':>10s} {'Loss mean':>10s} {'Delta':>8s} {'Signal':>8s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name, getter in metrics:
        w_vals = [getter(t) for t in winners]
        l_vals = [getter(t) for t in losers]
        w_mean = np.mean(w_vals)
        l_mean = np.mean(l_vals)
        delta = w_mean - l_mean
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(w_vals) + np.var(l_vals)) / 2) if len(w_vals) > 1 and len(l_vals) > 1 else 1.0
        d = delta / pooled_std if pooled_std > 0 else 0.0
        signal = "STRONG" if abs(d) > 0.5 else "WEAK" if abs(d) > 0.2 else ""
        lines.append(
            f"  {name:18s} {w_mean:10.3f} {l_mean:10.3f} {delta:+8.3f} {signal:>8s}"
        )

    # Chop mode distribution: winners vs losers
    lines.append(f"\n  Chop mode distribution:")
    for mode in ["NORMAL", "DEGRADED", "HALT"]:
        w_n = sum(1 for t in winners if t.chop_mode == mode)
        l_n = sum(1 for t in losers if t.chop_mode == mode)
        w_pct = 100 * w_n / len(winners)
        l_pct = 100 * l_n / len(losers)
        lines.append(f"    {mode:12s}  Winners: {w_n:3d} ({w_pct:4.0f}%)  "
                     f"Losers: {l_n:3d} ({l_pct:4.0f}%)")

    # Entry hour distribution (if available)
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        w_hours = [t.entry_time.astimezone(et).hour for t in winners if t.entry_time]
        l_hours = [t.entry_time.astimezone(et).hour for t in losers if t.entry_time]
        if w_hours and l_hours:
            lines.append(f"\n  Entry hour (ET):")
            all_hours = sorted(set(w_hours + l_hours))
            for h in all_hours:
                wh = sum(1 for x in w_hours if x == h)
                lh = sum(1 for x in l_hours if x == h)
                wr = 100 * wh / (wh + lh) if (wh + lh) > 0 else 0
                lines.append(f"    {h:02d}:00  wins={wh:3d}  losses={lh:3d}  WR={wr:.0f}%")
    except Exception:
        pass

    return "\n".join(lines)


def _post_tp1_runner_deep_dive(trades: list) -> str:
    """Deep analysis of post-TP1 runner behavior.

    57% of trades that hit TP1 stop near breakeven. This section
    quantifies the runner leak and identifies improvement vectors.
    """
    if not trades:
        return "=== Post-TP1 Runner Analysis ===\n  No trades."

    tp1_trades = [t for t in trades if t.tp1_hit]
    if not tp1_trades:
        return "=== Post-TP1 Runner Analysis ===\n  No TP1 trades."

    lines = ["=== Post-TP1 Runner Analysis ==="]
    n = len(tp1_trades)
    lines.append(f"  Total TP1 trades: {n}")

    # Classify outcomes
    big_winners = [t for t in tp1_trades if t.r_multiple >= 2.0]
    small_winners = [t for t in tp1_trades if 0.3 <= t.r_multiple < 2.0]
    be_zone = [t for t in tp1_trades if -0.3 < t.r_multiple < 0.3]
    small_losers = [t for t in tp1_trades if t.r_multiple <= -0.3]

    lines.append(f"\n  Outcome distribution after TP1:")
    for label, group in [("Big winner (R >= +2.0)", big_winners),
                         ("Small winner (+0.3 to +2.0)", small_winners),
                         ("BE zone (|R| < 0.3)", be_zone),
                         ("Net loser (R <= -0.3)", small_losers)]:
        pct = 100 * len(group) / n
        avg_r = np.mean([t.r_multiple for t in group]) if group else 0
        lines.append(f"    {label:30s}: {len(group):3d} ({pct:4.0f}%)  avgR={avg_r:+.3f}")

    # Quantify the BE leak
    if be_zone:
        be_r = [t.r_multiple for t in be_zone]
        be_mfe = [t.mfe_r for t in be_zone]
        lines.append(f"\n  BE-zone analysis ({len(be_zone)} trades):")
        lines.append(f"    Avg final R:  {np.mean(be_r):+.3f}")
        lines.append(f"    Avg MFE:      {np.mean(be_mfe):.2f}R")
        lines.append(f"    Avg bars held: {np.mean([t.bars_held_30m for t in be_zone]):.1f}")

        # What portion of MFE was captured in BE-zone trades?
        be_capture = [t.r_multiple / t.mfe_r if t.mfe_r > 0 else 0 for t in be_zone]
        lines.append(f"    Capture ratio: {np.mean(be_capture):.2f} (of MFE)")
        lines.append(f"    ** These trades saw +MFE then gave it all back to BE stop")

    # TP2 conversion rate from TP1
    tp2_from_tp1 = [t for t in tp1_trades if t.tp2_hit]
    lines.append(f"\n  TP1 -> TP2 conversion: {len(tp2_from_tp1)}/{n} ({100 * len(tp2_from_tp1) / n:.0f}%)")
    if tp2_from_tp1:
        lines.append(f"    TP2 trades avgR: {np.mean([t.r_multiple for t in tp2_from_tp1]):+.3f}")

    # Exit tier impact on post-TP1
    lines.append(f"\n  Post-TP1 by exit tier:")
    for tier in ["Aligned", "Neutral", "Caution"]:
        ct = [t for t in tp1_trades if t.exit_tier == tier]
        if not ct:
            continue
        be_ct = sum(1 for t in ct if abs(t.r_multiple) < 0.3)
        lines.append(f"    {tier:10s}: N={len(ct):3d}  avgR={np.mean([t.r_multiple for t in ct]):+.3f}  "
                     f"BE-zone={100 * be_ct / len(ct):.0f}%")

    # Hypothetical: what if BE stop was tighter (entry + 0.1R)?
    # vs looser (entry - 0.2R)? Use MFE/MAE to estimate
    lines.append(f"\n  Hypothetical BE buffer sensitivity (TP1 trades):")
    for be_r_threshold in [0.1, 0.2, 0.3, 0.5]:
        # Trades that would survive with this BE buffer
        survivors = [t for t in tp1_trades if t.r_multiple >= be_r_threshold or t.mfe_r >= be_r_threshold * 2]
        stopped = [t for t in tp1_trades if t not in survivors]
        lines.append(f"    BE at +{be_r_threshold:.1f}R: {len(stopped)} stopped early, "
                     f"{len(survivors)} survive, net avgR={np.mean([t.r_multiple for t in tp1_trades]):+.3f}")

    return "\n".join(lines)


def _equity_reconciliation(
    trades: list,
    equity_curve: np.ndarray | None,
    initial_equity: float,
    point_value: float,
) -> str:
    """Reconcile trade record P&L vs actual equity curve.

    Recorded R uses initial_stop_price for the risk denominator (fixed).
    Remaining R gap vs simple entry-exit is due to TP partial fills:
    recorded R correctly accounts for qty closed at TP prices, while the
    simple (exit - entry) / risk calculation assumes all qty at final exit.
    """
    if not trades:
        return "=== Equity Reconciliation ===\n  No trades."

    lines = ["=== Equity Reconciliation ==="]

    # Sum of trade record P&L
    recorded_pnl = sum(t.pnl_dollars for t in trades)

    # Recompute P&L from entry/exit prices (ground truth per trade)
    recomputed_pnl = 0.0
    for t in trades:
        if t.direction == 1:  # LONG
            trade_pnl = (t.exit_price - t.entry_price) * point_value * t.qty
        else:  # SHORT
            trade_pnl = (t.entry_price - t.exit_price) * point_value * t.qty
        recomputed_pnl += trade_pnl

    # Equity curve delta
    equity_delta = None
    if equity_curve is not None and len(equity_curve) >= 2:
        equity_delta = float(equity_curve[-1]) - initial_equity

    lines.append(f"  Recorded trade P&L:     ${recorded_pnl:+,.2f}")
    lines.append(f"  Recomputed (entry-exit): ${recomputed_pnl:+,.2f}")
    if equity_delta is not None:
        lines.append(f"  Equity curve delta:     ${equity_delta:+,.2f}")

    # Gaps
    record_vs_recompute = recorded_pnl - recomputed_pnl
    lines.append(f"\n  Record vs recomputed gap: ${record_vs_recompute:+,.2f}")
    if equity_delta is not None:
        recompute_vs_equity = recomputed_pnl - equity_delta
        lines.append(f"  Recomputed vs equity gap: ${recompute_vs_equity:+,.2f} (commissions + rounding)")

    # Per-trade R-multiple accuracy
    lines.append(f"\n  R-multiple accuracy check:")
    r_gaps = []
    for t in trades:
        initial_r_pts = abs(t.entry_price - t.initial_stop)
        if initial_r_pts > 0 and t.qty > 0:
            initial_r_dollars = initial_r_pts * point_value * t.qty
            if t.direction == 1:
                actual_pnl = (t.exit_price - t.entry_price) * point_value * t.qty
            else:
                actual_pnl = (t.entry_price - t.exit_price) * point_value * t.qty
            true_r = actual_pnl / initial_r_dollars
            r_gaps.append(t.r_multiple - true_r)

    if r_gaps:
        gaps = np.array(r_gaps)
        lines.append(f"    R gap (recorded - true): mean={np.mean(gaps):+.3f}  "
                     f"median={np.median(gaps):+.3f}  max_abs={np.max(np.abs(gaps)):.3f}")
        large_gaps = sum(1 for g in gaps if abs(g) > 0.1)
        lines.append(f"    Trades with |R gap| > 0.1: {large_gaps}/{len(gaps)} "
                     f"({100 * large_gaps / len(gaps):.0f}%)")
        if np.mean(gaps) > 0.05:
            lines.append("    ** Recorded R > simple entry-exit R")
            lines.append("       Expected: TP partial fills closed at better prices than final exit")
            lines.append("       (R accounts for TP qty, simple calc does not)")
        elif np.mean(gaps) < -0.05:
            lines.append("    ** Recorded R < simple entry-exit R")
            lines.append("       Expected: TP partials exited early, missing additional upside")
            lines.append("       (R accounts for TP qty at TP prices, simple calc assumes all at final exit)")

    # Trust hierarchy
    lines.append(f"\n  Trust hierarchy:")
    lines.append(f"    1. Equity curve (tracks actual fills) -- GROUND TRUTH")
    lines.append(f"    2. Recorded P&L (uses initial_stop, includes TP partials)")
    lines.append(f"    3. Simple entry-exit P&L -- ignores TP partial fills")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structural diagnostics (added for optimized-config deep analysis)
# ---------------------------------------------------------------------------


def _drawdown_episode_anatomy(
    trades: list,
    initial_equity: float = 100_000.0,
    point_value: float = 2.0,
) -> str:
    """Decompose the top 5 drawdown episodes into contributing trades.

    For each episode: date range, contributing trades count, regime/session
    mix, entry subtypes involved, worst single trade, and recovery time.
    Critical for live deployment confidence -- answers "what caused the DDs?"
    """
    if len(trades) < 5:
        return "=== Drawdown Episode Anatomy ===\n  Insufficient trades."

    dated = sorted(
        [t for t in trades if t.entry_time and t.exit_time],
        key=lambda t: t.entry_time,
    )
    if len(dated) < 5:
        return "=== Drawdown Episode Anatomy ===\n  Insufficient dated trades."

    lines = ["=== Drawdown Episode Anatomy ==="]

    # Build cumulative P&L curve from trade sequence
    cum_pnl = np.zeros(len(dated) + 1)
    for i, t in enumerate(dated):
        cum_pnl[i + 1] = cum_pnl[i] + t.pnl_dollars

    # Identify drawdown episodes: peak -> trough -> recovery
    episodes = []
    peak_idx = 0
    peak_val = cum_pnl[0]
    in_dd = False
    trough_idx = 0
    trough_val = peak_val

    for i in range(1, len(cum_pnl)):
        if cum_pnl[i] > peak_val:
            if in_dd:
                # Recovery -- record episode
                episodes.append({
                    "peak_idx": peak_idx,
                    "trough_idx": trough_idx,
                    "recovery_idx": i,
                    "dd_dollars": trough_val - peak_val,
                    "dd_pct": (trough_val - peak_val) / (initial_equity + peak_val) * 100
                    if (initial_equity + peak_val) > 0 else 0,
                })
                in_dd = False
            peak_idx = i
            peak_val = cum_pnl[i]
            trough_idx = i
            trough_val = peak_val
        elif cum_pnl[i] < trough_val:
            in_dd = True
            trough_idx = i
            trough_val = cum_pnl[i]

    # Capture any still-open drawdown at end of data
    if in_dd:
        episodes.append({
            "peak_idx": peak_idx,
            "trough_idx": trough_idx,
            "recovery_idx": None,
            "dd_dollars": trough_val - peak_val,
            "dd_pct": (trough_val - peak_val) / (initial_equity + peak_val) * 100
            if (initial_equity + peak_val) > 0 else 0,
        })

    if not episodes:
        lines.append("  No drawdown episodes detected (monotonically increasing equity).")
        return "\n".join(lines)

    # Sort by severity
    episodes.sort(key=lambda e: e["dd_dollars"])
    top = episodes[:5]

    lines.append(f"  Total DD episodes: {len(episodes)}")
    lines.append(f"  Analyzing top {len(top)} by severity:\n")

    for rank, ep in enumerate(top, 1):
        pi = ep["peak_idx"]
        ti = ep["trough_idx"]
        ri = ep["recovery_idx"]

        # Trades in the drawdown phase (peak -> trough)
        dd_trades = dated[pi:ti]
        if not dd_trades:
            continue

        start_date = dd_trades[0].entry_time.strftime("%Y-%m-%d %H:%M") if dd_trades[0].entry_time else "?"
        end_date = dd_trades[-1].exit_time.strftime("%Y-%m-%d %H:%M") if dd_trades[-1].exit_time else "?"

        lines.append(f"  --- Episode #{rank}: ${ep['dd_dollars']:+,.0f} ({ep['dd_pct']:.2f}%) ---")
        lines.append(f"    Period: {start_date} to {end_date}")
        lines.append(f"    Trades in DD: {len(dd_trades)}")

        # Recovery info
        if ri is not None:
            recovery_trades = dated[ti:ri]
            if recovery_trades and dd_trades:
                rec_start = dd_trades[-1].exit_time
                rec_end = recovery_trades[-1].exit_time
                if rec_start and rec_end:
                    rec_days = (rec_end - rec_start).total_seconds() / 86400
                    lines.append(f"    Recovery: {len(recovery_trades)} trades, {rec_days:.1f} calendar days")
        else:
            lines.append(f"    Recovery: STILL OPEN (has not recovered)")

        # Worst single trade
        worst = min(dd_trades, key=lambda t: t.pnl_dollars)
        lines.append(f"    Worst trade: ${worst.pnl_dollars:+,.0f} (R={worst.r_multiple:+.2f}, "
                     f"{worst.entry_subtype}, {worst.session})")

        # Regime mix
        regimes = Counter(t.composite_regime for t in dd_trades)
        regime_str = ", ".join(f"{r}={c}" for r, c in regimes.most_common(3))
        lines.append(f"    Regimes: {regime_str}")

        # Session mix
        sessions = Counter(t.session for t in dd_trades)
        sess_str = ", ".join(f"{s}={c}" for s, c in sessions.most_common(3))
        lines.append(f"    Sessions: {sess_str}")

        # Entry subtype mix
        subtypes = Counter(t.entry_subtype for t in dd_trades)
        sub_str = ", ".join(f"{s}={c}" for s, c in subtypes.most_common())
        lines.append(f"    Entry subtypes: {sub_str}")

        # Direction mix
        longs = sum(1 for t in dd_trades if t.direction == 1)
        shorts = len(dd_trades) - longs
        lines.append(f"    Direction: {longs}L / {shorts}S")

        # Win rate during DD
        wr = sum(1 for t in dd_trades if t.r_multiple > 0) / len(dd_trades) * 100
        avg_r = np.mean([t.r_multiple for t in dd_trades])
        lines.append(f"    WR during DD: {wr:.0f}%  Avg R: {avg_r:+.3f}")
        lines.append("")

    # Cross-episode patterns
    all_dd_trades = []
    for ep in top:
        all_dd_trades.extend(dated[ep["peak_idx"]:ep["trough_idx"]])

    if all_dd_trades:
        lines.append("  Cross-Episode Patterns (top 5 DDs):")
        # Most dangerous regime
        regime_pnl: dict[str, float] = {}
        for t in all_dd_trades:
            regime_pnl.setdefault(t.composite_regime, 0.0)
            regime_pnl[t.composite_regime] += t.pnl_dollars
        worst_regime = min(regime_pnl, key=regime_pnl.get)
        lines.append(f"    Most dangerous regime: {worst_regime} "
                     f"(${regime_pnl[worst_regime]:+,.0f} across DD episodes)")

        # Most dangerous session
        sess_pnl: dict[str, float] = {}
        for t in all_dd_trades:
            sess_pnl.setdefault(t.session, 0.0)
            sess_pnl[t.session] += t.pnl_dollars
        worst_sess = min(sess_pnl, key=sess_pnl.get)
        lines.append(f"    Most dangerous session: {worst_sess} "
                     f"(${sess_pnl[worst_sess]:+,.0f} across DD episodes)")

        # Most dangerous subtype
        sub_pnl: dict[str, float] = {}
        for t in all_dd_trades:
            sub_pnl.setdefault(t.entry_subtype, 0.0)
            sub_pnl[t.entry_subtype] += t.pnl_dollars
        worst_sub = min(sub_pnl, key=sub_pnl.get)
        lines.append(f"    Most dangerous subtype: {worst_sub} "
                     f"(${sub_pnl[worst_sub]:+,.0f} across DD episodes)")

    return "\n".join(lines)


def _direction_asymmetry(trades: list) -> str:
    """Comprehensive Long vs Short breakdown across all dimensions.

    Reveals structural directional edge. For a box breakout strategy,
    asymmetry could indicate trend bias, mean-reversion tendency,
    or session-specific directional patterns.
    """
    if not trades:
        return "=== Direction Asymmetry ===\n  No trades."

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction != 1]

    if not longs or not shorts:
        return "=== Direction Asymmetry ===\n  Need both long and short trades."

    lines = ["=== Direction Asymmetry ==="]

    # Overall comparison
    def _dir_stats(label: str, group: list) -> list[str]:
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        total_pnl = sum(t.pnl_dollars for t in group)
        pf_w = sum(t.pnl_dollars for t in group if t.pnl_dollars > 0)
        pf_l = abs(sum(t.pnl_dollars for t in group if t.pnl_dollars < 0))
        pf = pf_w / pf_l if pf_l > 0 else float("inf")
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_mae = np.mean([t.mae_r for t in group])
        avg_hold = np.mean([t.bars_held_30m for t in group])
        return [
            f"  {label}:",
            f"    N={n}  WR={wr:.1f}%  AvgR={avg_r:+.3f}  PnL=${total_pnl:+,.0f}  PF={pf:.2f}",
            f"    AvgMFE={avg_mfe:.2f}R  AvgMAE={avg_mae:.2f}R  AvgHold={avg_hold:.1f} bars",
        ]

    lines.extend(_dir_stats("LONG", longs))
    lines.extend(_dir_stats("SHORT", shorts))

    # Edge delta
    l_avg = np.mean([t.r_multiple for t in longs])
    s_avg = np.mean([t.r_multiple for t in shorts])
    edge = l_avg - s_avg
    lines.append(f"\n  Directional edge (L - S): {edge:+.3f}R")
    if abs(edge) > 0.15:
        stronger = "LONG" if edge > 0 else "SHORT"
        lines.append(f"  ** Significant {stronger} bias detected")

    # By regime
    lines.append(f"\n  Direction x Regime:")
    regimes = sorted(set(t.composite_regime for t in trades))
    header = f"    {'Regime':20s} {'L_N':>5s} {'L_WR':>5s} {'L_AvgR':>7s} {'S_N':>5s} {'S_WR':>5s} {'S_AvgR':>7s} {'Edge':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for reg in regimes:
        rl = [t for t in longs if t.composite_regime == reg]
        rs = [t for t in shorts if t.composite_regime == reg]
        if not rl and not rs:
            continue
        l_wr = sum(1 for t in rl if t.r_multiple > 0) / len(rl) * 100 if rl else 0
        s_wr = sum(1 for t in rs if t.r_multiple > 0) / len(rs) * 100 if rs else 0
        l_r = np.mean([t.r_multiple for t in rl]) if rl else 0
        s_r = np.mean([t.r_multiple for t in rs]) if rs else 0
        lines.append(
            f"    {reg:20s} {len(rl):5d} {l_wr:4.0f}% {l_r:+7.3f} "
            f"{len(rs):5d} {s_wr:4.0f}% {s_r:+7.3f} {l_r - s_r:+7.3f}"
        )

    # By session
    lines.append(f"\n  Direction x Session:")
    sessions = sorted(set(t.session for t in trades))
    header = f"    {'Session':20s} {'L_N':>5s} {'L_WR':>5s} {'L_AvgR':>7s} {'S_N':>5s} {'S_WR':>5s} {'S_AvgR':>7s} {'Edge':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for sess in sessions:
        sl = [t for t in longs if t.session == sess]
        ss_trades = [t for t in shorts if t.session == sess]
        if not sl and not ss_trades:
            continue
        l_wr = sum(1 for t in sl if t.r_multiple > 0) / len(sl) * 100 if sl else 0
        s_wr = sum(1 for t in ss_trades if t.r_multiple > 0) / len(ss_trades) * 100 if ss_trades else 0
        l_r = np.mean([t.r_multiple for t in sl]) if sl else 0
        s_r = np.mean([t.r_multiple for t in ss_trades]) if ss_trades else 0
        lines.append(
            f"    {sess:20s} {len(sl):5d} {l_wr:4.0f}% {l_r:+7.3f} "
            f"{len(ss_trades):5d} {s_wr:4.0f}% {s_r:+7.3f} {l_r - s_r:+7.3f}"
        )

    # By entry subtype
    lines.append(f"\n  Direction x Entry Subtype:")
    subtypes = sorted(set(t.entry_subtype for t in trades))
    header = f"    {'Subtype':20s} {'L_N':>5s} {'L_AvgR':>7s} {'L_PnL':>9s} {'S_N':>5s} {'S_AvgR':>7s} {'S_PnL':>9s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for st in subtypes:
        stl = [t for t in longs if t.entry_subtype == st]
        sts = [t for t in shorts if t.entry_subtype == st]
        l_r = np.mean([t.r_multiple for t in stl]) if stl else 0
        s_r = np.mean([t.r_multiple for t in sts]) if sts else 0
        l_pnl = sum(t.pnl_dollars for t in stl)
        s_pnl = sum(t.pnl_dollars for t in sts)
        lines.append(
            f"    {st:20s} {len(stl):5d} {l_r:+7.3f} ${l_pnl:+8,.0f} "
            f"{len(sts):5d} {s_r:+7.3f} ${s_pnl:+8,.0f}"
        )

    # TP hit rates by direction
    lines.append(f"\n  TP Hit Rates by Direction:")
    for label, group in [("LONG", longs), ("SHORT", shorts)]:
        tp1 = sum(1 for t in group if t.tp1_hit) / len(group) * 100
        tp2 = sum(1 for t in group if t.tp2_hit) / len(group) * 100
        tp3 = sum(1 for t in group if t.tp3_hit) / len(group) * 100
        lines.append(f"    {label}: TP1={tp1:.0f}%  TP2={tp2:.0f}%  TP3={tp3:.0f}%")

    return "\n".join(lines)


def _volatility_bucketed_performance(trades: list) -> str:
    """Performance by box_width quintiles (volatility proxy).

    Box width is the primary indicator of market volatility context at
    entry. Reveals whether the strategy works better in tight consolidation
    (small boxes) or volatile expansion (wide boxes). Actionable: could
    filter or resize by box width bucket.
    """
    if not trades:
        return "=== Volatility-Bucketed Performance ===\n  No trades."

    valid = [t for t in trades if t.box_width > 0]
    if len(valid) < 10:
        return "=== Volatility-Bucketed Performance ===\n  Insufficient trades with box_width."

    lines = ["=== Volatility-Bucketed Performance ==="]

    widths = np.array([t.box_width for t in valid])
    lines.append(f"  Box width (points): mean={np.mean(widths):.2f}  "
                 f"median={np.median(widths):.2f}  min={np.min(widths):.2f}  "
                 f"max={np.max(widths):.2f}")

    # Quintile boundaries
    pcts = [0, 20, 40, 60, 80, 100]
    boundaries = np.percentile(widths, pcts)

    lines.append(f"\n  Quintile Performance:")
    header = (
        f"  {'Bucket':22s} {'Range':>14s} {'N':>5s} {'WR':>5s} "
        f"{'AvgR':>7s} {'PF':>6s} {'PnL':>10s} {'AvgMFE':>7s} {'AvgMAE':>7s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    bucket_labels = ["Q1 Tightest", "Q2 Narrow", "Q3 Medium", "Q4 Wide", "Q5 Widest"]
    for i in range(5):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i < 4:
            bucket = [t for t in valid if lo <= t.box_width < hi]
        else:
            bucket = [t for t in valid if lo <= t.box_width <= hi]

        if not bucket:
            continue

        n = len(bucket)
        wr = sum(1 for t in bucket if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in bucket])
        gross_w = sum(t.pnl_dollars for t in bucket if t.pnl_dollars > 0)
        gross_l = abs(sum(t.pnl_dollars for t in bucket if t.pnl_dollars < 0))
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        total_pnl = sum(t.pnl_dollars for t in bucket)
        avg_mfe = np.mean([t.mfe_r for t in bucket])
        avg_mae = np.mean([t.mae_r for t in bucket])
        rng = f"{lo:.1f}-{hi:.1f}"

        lines.append(
            f"  {bucket_labels[i]:22s} {rng:>14s} {n:5d} {wr:4.0f}% "
            f"{avg_r:+7.3f} {pf:6.2f} ${total_pnl:+9,.0f} {avg_mfe:7.2f} {avg_mae:7.2f}"
        )

    # Correlation: box_width vs R
    rs = np.array([t.r_multiple for t in valid])
    if len(set(widths)) > 1:
        corr = np.corrcoef(widths, rs)[0, 1]
        lines.append(f"\n  Correlation (box_width vs R): {corr:+.3f}")
        if corr > 0.1:
            lines.append("  ** Wider boxes correlate with better R -- strategy likes volatility")
        elif corr < -0.1:
            lines.append("  ** Tighter boxes correlate with better R -- strategy prefers consolidation")
        else:
            lines.append("  No significant correlation -- performance is volatility-neutral")

    # Best/worst quintile impact
    lines.append(f"\n  Box Width Impact Analysis:")
    q_results = []
    for i in range(5):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i < 4:
            bucket = [t for t in valid if lo <= t.box_width < hi]
        else:
            bucket = [t for t in valid if lo <= t.box_width <= hi]
        if bucket:
            q_results.append((bucket_labels[i], sum(t.pnl_dollars for t in bucket), len(bucket)))

    if q_results:
        best = max(q_results, key=lambda x: x[1])
        worst = min(q_results, key=lambda x: x[1])
        lines.append(f"    Best quintile:  {best[0]} (${best[1]:+,.0f}, N={best[2]})")
        lines.append(f"    Worst quintile: {worst[0]} (${worst[1]:+,.0f}, N={worst[2]})")
        if worst[1] < 0:
            lines.append(f"    Filtering out {worst[0]} would remove "
                         f"${worst[1]:,.0f} drag ({worst[2]} trades)")

    # Entry subtype distribution per quintile
    lines.append(f"\n  Entry Subtype Mix by Volatility:")
    subtypes = sorted(set(t.entry_subtype for t in valid))
    header = f"  {'Bucket':22s} " + " ".join(f"{st:>12s}" for st in subtypes)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i in range(5):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i < 4:
            bucket = [t for t in valid if lo <= t.box_width < hi]
        else:
            bucket = [t for t in valid if lo <= t.box_width <= hi]
        if not bucket:
            continue
        row = f"  {bucket_labels[i]:22s} "
        for st in subtypes:
            cnt = sum(1 for t in bucket if t.entry_subtype == st)
            pct = cnt / len(bucket) * 100
            row += f"{pct:11.0f}% "
        lines.append(row)

    return "\n".join(lines)


def _trade_clustering(trades: list) -> str:
    """Inter-trade gap analysis, burst detection, and opportunity density.

    Reveals capital utilization patterns: long droughts vs bursts of
    activity. Important for portfolio allocation and psychological
    preparation. Also identifies whether clustered trades perform
    differently from isolated ones.
    """
    if not trades:
        return "=== Trade Clustering ===\n  No trades."

    dated = sorted(
        [t for t in trades if t.entry_time],
        key=lambda t: t.entry_time,
    )
    if len(dated) < 10:
        return "=== Trade Clustering ===\n  Insufficient dated trades."

    lines = ["=== Trade Clustering ==="]

    # Inter-trade gaps (hours between consecutive entries)
    gaps_hours = []
    for i in range(1, len(dated)):
        delta = (dated[i].entry_time - dated[i - 1].entry_time).total_seconds() / 3600
        gaps_hours.append(delta)

    gaps = np.array(gaps_hours)
    lines.append(f"  Inter-trade gaps (hours):")
    lines.append(f"    mean={np.mean(gaps):.1f}  median={np.median(gaps):.1f}  "
                 f"min={np.min(gaps):.1f}  max={np.max(gaps):.1f}")

    # Gap distribution
    gap_buckets = [
        (0, 2, "Same session (<2h)"),
        (2, 8, "Same day (2-8h)"),
        (8, 24, "Next session (8-24h)"),
        (24, 72, "1-3 days"),
        (72, 168, "3-7 days"),
        (168, float("inf"), "1+ week drought"),
    ]

    lines.append(f"\n  Gap Distribution:")
    header = f"    {'Bucket':24s} {'N':>5s} {'%':>6s} {'NextR':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for lo, hi, label in gap_buckets:
        indices = [i for i, g in enumerate(gaps_hours) if lo <= g < hi]
        if not indices:
            continue
        pct = len(indices) / len(gaps_hours) * 100
        # Performance of trades AFTER each gap type
        next_trades = [dated[i + 1] for i in indices if i + 1 < len(dated)]
        avg_r = np.mean([t.r_multiple for t in next_trades]) if next_trades else 0
        lines.append(f"    {label:24s} {len(indices):5d} {pct:5.1f}% {avg_r:+7.3f}")

    # Burst detection: 3+ trades within 4 hours
    burst_threshold_hours = 4.0
    min_burst_size = 3
    bursts = []
    current_burst = [dated[0]]

    for i in range(1, len(dated)):
        delta_h = (dated[i].entry_time - current_burst[-1].entry_time).total_seconds() / 3600
        if delta_h <= burst_threshold_hours:
            current_burst.append(dated[i])
        else:
            if len(current_burst) >= min_burst_size:
                bursts.append(list(current_burst))
            current_burst = [dated[i]]
    if len(current_burst) >= min_burst_size:
        bursts.append(list(current_burst))

    lines.append(f"\n  Burst Detection ({min_burst_size}+ trades within {burst_threshold_hours:.0f}h):")
    lines.append(f"    Total bursts: {len(bursts)}")
    burst_trades = [t for b in bursts for t in b]
    isolated = [t for t in dated if t not in burst_trades]

    if burst_trades:
        b_wr = sum(1 for t in burst_trades if t.r_multiple > 0) / len(burst_trades) * 100
        b_avg = np.mean([t.r_multiple for t in burst_trades])
        b_pnl = sum(t.pnl_dollars for t in burst_trades)
        lines.append(f"    Burst trades: {len(burst_trades)} "
                     f"(WR={b_wr:.0f}%, AvgR={b_avg:+.3f}, PnL=${b_pnl:+,.0f})")

    if isolated:
        i_wr = sum(1 for t in isolated if t.r_multiple > 0) / len(isolated) * 100
        i_avg = np.mean([t.r_multiple for t in isolated])
        i_pnl = sum(t.pnl_dollars for t in isolated)
        lines.append(f"    Isolated trades: {len(isolated)} "
                     f"(WR={i_wr:.0f}%, AvgR={i_avg:+.3f}, PnL=${i_pnl:+,.0f})")

    if burst_trades and isolated:
        edge = np.mean([t.r_multiple for t in burst_trades]) - np.mean([t.r_multiple for t in isolated])
        lines.append(f"    Burst vs isolated edge: {edge:+.3f}R")
        if edge < -0.1:
            lines.append("    ** Clustered trades underperform -- consider cooldown between entries")
        elif edge > 0.1:
            lines.append("    ** Clustered trades outperform -- momentum carries through bursts")

    # Biggest bursts detail
    if bursts:
        bursts.sort(key=lambda b: -len(b))
        lines.append(f"\n  Largest Bursts:")
        for i, burst in enumerate(bursts[:5], 1):
            start = burst[0].entry_time.strftime("%Y-%m-%d %H:%M") if burst[0].entry_time else "?"
            pnl = sum(t.pnl_dollars for t in burst)
            wr = sum(1 for t in burst if t.r_multiple > 0) / len(burst) * 100
            lines.append(f"    #{i}: {start}  {len(burst)} trades  "
                         f"WR={wr:.0f}%  PnL=${pnl:+,.0f}")

    # Opportunity density by month
    if dated[0].entry_time and dated[-1].entry_time:
        lines.append(f"\n  Monthly Opportunity Density:")
        from collections import defaultdict
        monthly: dict[str, list] = defaultdict(list)
        for t in dated:
            key = t.entry_time.strftime("%Y-%m")
            monthly[key].append(t)

        header = f"    {'Month':>8s} {'Trades':>6s} {'AvgR':>7s} {'PnL':>10s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for month in sorted(monthly):
            mt = monthly[month]
            avg_r = np.mean([t.r_multiple for t in mt])
            pnl = sum(t.pnl_dollars for t in mt)
            lines.append(f"    {month:>8s} {len(mt):6d} {avg_r:+7.3f} ${pnl:+9,.0f}")

        # Trades per month stats
        counts = [len(v) for v in monthly.values()]
        lines.append(f"\n    Trades/month: mean={np.mean(counts):.1f}  "
                     f"min={min(counts)}  max={max(counts)}  std={np.std(counts):.1f}")

    return "\n".join(lines)
