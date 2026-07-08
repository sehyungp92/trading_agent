"""VdubusNQ v4.0 strategy-specific diagnostic reports."""
from __future__ import annotations

from collections import Counter

import numpy as np


def vdubus_full_diagnostic(
    trades: list,
    signal_events: list | None = None,
    equity_curve: np.ndarray | None = None,
    time_series: np.ndarray | None = None,
) -> str:
    """Generate comprehensive VdubusNQ diagnostic report."""
    sections = []
    # --- Existing 9 sections ---
    sections.append(_signal_funnel(trades, signal_events))
    sections.append(_entry_type_breakdown(trades))
    sections.append(_session_breakdown(trades))
    sections.append(_regime_breakdown(trades))
    sections.append(_decision_gate_analysis(trades))
    sections.append(_position_lifecycle(trades))
    sections.append(_exit_reason_breakdown(trades))
    sections.append(_flip_entry_analysis(trades))
    sections.append(_predator_analysis(trades))
    # --- New 13 sections ---
    # Tier 1 — Directly Actionable
    sections.append(_mfe_mae_analysis(trades))
    sections.append(_r_multiple_distribution(trades))
    sections.append(_direction_breakdown(trades))
    sections.append(_streak_analysis(trades))
    sections.append(_drawdown_profile(equity_curve, time_series))
    # Tier 2 — Pattern Discovery
    sections.append(_monthly_pnl(trades))
    sections.append(_day_of_week(trades))
    sections.append(_hourly_performance(trades))
    sections.append(_cross_tab_breakdowns(trades))
    sections.append(_stop_distance_analysis(trades))
    # Tier 3 — Refinement
    sections.append(_entry_efficiency(trades))
    sections.append(_hold_time_optimization(trades))
    sections.append(_trade_autocorrelation(trades))
    # Tier 4 — Weakness Discovery
    sections.append(_winner_giveback_analysis(trades))
    sections.append(_loser_autopsy(trades))
    sections.append(_vwap_distance_analysis(trades))
    sections.append(_pnl_development_curve(trades))
    sections.append(_exit_reason_x_subwindow(trades))
    sections.append(_rolling_stability(trades))
    sections.append(_mae_recovery_analysis(trades))
    # Tier 5 — Structural Deep-Dives
    sections.append(_stale_exit_deep_dive(trades))
    sections.append(_overnight_risk_analysis(trades))
    sections.append(_early_kill_audit(trades))
    sections.append(_r_per_bar_efficiency(trades))
    sections.append(_class_mult_calibration(trades))
    return "\n\n".join(s for s in sections if s)


def _signal_funnel(trades: list, signal_events: list | None) -> str:
    """Signal funnel: evaluations -> regime pass -> signal -> entries -> filled."""
    lines = ["=== VdubusNQ Signal Funnel ==="]
    if signal_events:
        total_eval = len(signal_events)
        passed = sum(1 for e in signal_events if e.passed_all)
        blocked = total_eval - passed
        lines.append(f"  15m evaluations:          {total_eval}")
        lines.append(f"  All gates passed:         {passed}")
        lines.append(f"  Gates blocked:            {blocked}")

        reasons = Counter(e.first_block_reason for e in signal_events if not e.passed_all)
        if reasons:
            lines.append("  Block reason distribution:")
            for reason, count in reasons.most_common():
                pct = count / blocked * 100 if blocked > 0 else 0
                lines.append(f"    {reason:28s} {count:5d}  ({pct:5.1f}%)")
    lines.append(f"  Trades completed:         {len(trades)}")
    return "\n".join(lines)


def _entry_type_breakdown(trades: list) -> str:
    """Entry type breakdown: Type A vs Type B."""
    lines = ["=== Entry Type Breakdown ==="]
    by_type: dict[str, list] = {}
    for t in trades:
        by_type.setdefault(t.entry_type, []).append(t)

    header = f"  {'Type':12s} {'N':>5s} {'WinR':>5s} {'AvgR':>7s} {'AvgPnL':>9s} {'AvgBars':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for etype in sorted(by_type):
        group = by_type[etype]
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        bars = [t.bars_held_15m for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        avg_bars = np.mean(bars) if bars else 0
        lines.append(
            f"  {etype:12s} {n:5d} {win_rate:4.0f}% {avg_r:+7.3f} "
            f"${avg_pnl:+8.0f} {avg_bars:6.1f}"
        )
    return "\n".join(lines)


def _session_breakdown(trades: list) -> str:
    """Session/sub-window breakdown: OPEN vs CORE vs CLOSE vs EVENING."""
    lines = ["=== Session / Sub-Window Breakdown ==="]
    by_sub: dict[str, list] = {}
    for t in trades:
        by_sub.setdefault(t.sub_window, []).append(t)

    header = f"  {'SubWindow':10s} {'N':>5s} {'WinR':>5s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]:
        group = by_sub.get(sw, [])
        if not group:
            continue
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        total_pnl = sum(pnls)
        lines.append(
            f"  {sw:10s} {n:5d} {win_rate:4.0f}% {avg_r:+7.3f} ${total_pnl:+9.0f}"
        )
    return "\n".join(lines)


def _regime_breakdown(trades: list) -> str:
    """Regime breakdown by daily_trend x vol_state."""
    lines = ["=== Regime Breakdown (DailyTrend x VolState) ==="]
    by_regime: dict[str, list] = {}
    for t in trades:
        key = f"DT={t.daily_trend:+d}/{t.vol_state}"
        by_regime.setdefault(key, []).append(t)

    header = f"  {'Regime':20s} {'N':>5s} {'WinR':>5s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for regime in sorted(by_regime):
        group = by_regime[regime]
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100 if n > 0 else 0
        avg_r = np.mean(rs) if rs else 0
        total_pnl = sum(pnls)
        lines.append(
            f"  {regime:20s} {n:5d} {win_rate:4.0f}% {avg_r:+7.3f} ${total_pnl:+9.0f}"
        )
    return "\n".join(lines)


def _decision_gate_analysis(trades: list) -> str:
    """Decision gate analysis: HOLD vs FLATTEN counts, R at gate, overnight outcomes."""
    lines = ["=== Decision Gate Analysis ==="]
    holds = [t for t in trades if t.decision_gate_action == "HOLD"]
    flattens = [t for t in trades if t.decision_gate_action == "FLATTEN"]
    no_gate = [t for t in trades if not t.decision_gate_action]

    lines.append(f"  HOLD:     {len(holds):4d}  AvgR={np.mean([t.r_multiple for t in holds]):+.3f}" if holds else "  HOLD:        0")
    lines.append(f"  FLATTEN:  {len(flattens):4d}  AvgR={np.mean([t.r_multiple for t in flattens]):+.3f}" if flattens else "  FLATTEN:     0")
    lines.append(f"  No gate:  {len(no_gate):4d}")

    if holds:
        overnight_rs = [t.r_multiple for t in holds]
        lines.append(f"  Overnight holds: avg final R={np.mean(overnight_rs):+.3f}  "
                      f"sessions={np.mean([t.overnight_sessions for t in holds]):.1f}")
    return "\n".join(lines)


def _position_lifecycle(trades: list) -> str:
    """Position lifecycle: ACTIVE_RISK -> ACTIVE_FREE -> SWING_HOLD transition rates."""
    lines = ["=== Position Lifecycle ==="]
    by_stage: dict[str, list] = {}
    for t in trades:
        by_stage.setdefault(t.stage_at_exit, []).append(t)

    for stage in ["ACTIVE_RISK", "ACTIVE_FREE", "SWING_HOLD"]:
        group = by_stage.get(stage, [])
        if not group:
            continue
        n = len(group)
        pct = n / len(trades) * 100 if trades else 0
        avg_r = np.mean([t.r_multiple for t in group])
        lines.append(f"  {stage:15s}  N={n:4d} ({pct:4.0f}%)  AvgR={avg_r:+.3f}")
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


def _flip_entry_analysis(trades: list) -> str:
    """Flip entry analysis."""
    lines = ["=== Flip Entry Analysis ==="]
    flips = [t for t in trades if t.is_flip]
    normal = [t for t in trades if not t.is_flip]
    lines.append(f"  Normal entries: {len(normal)}")
    lines.append(f"  Flip entries:   {len(flips)}")
    if flips:
        avg_r = np.mean([t.r_multiple for t in flips])
        win_rate = sum(1 for t in flips if t.r_multiple > 0) / len(flips) * 100
        lines.append(f"    Flip WinR={win_rate:.0f}%  AvgR={avg_r:+.3f}")
    return "\n".join(lines)


def _predator_analysis(trades: list) -> str:
    """Predator overlay impact: trades with/without predator class_mult."""
    lines = ["=== Predator Overlay Impact ==="]
    from strategies.momentum.vdub.config import CLASS_MULT_PREDATOR, CLASS_MULT_NOPRED
    pred = [t for t in trades if t.class_mult >= CLASS_MULT_PREDATOR - 0.01]
    nopred = [t for t in trades if abs(t.class_mult - CLASS_MULT_NOPRED) < 0.01]

    lines.append(f"  With predator:    N={len(pred):4d}")
    if pred:
        avg_r = np.mean([t.r_multiple for t in pred])
        lines.append(f"    AvgR={avg_r:+.3f}  WinR={sum(1 for t in pred if t.r_multiple > 0)/len(pred)*100:.0f}%")

    lines.append(f"  Without predator: N={len(nopred):4d}")
    if nopred:
        avg_r = np.mean([t.r_multiple for t in nopred])
        lines.append(f"    AvgR={avg_r:+.3f}  WinR={sum(1 for t in nopred if t.r_multiple > 0)/len(nopred)*100:.0f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 1 — Directly Actionable (Exit/Entry/Stop Optimization)
# ---------------------------------------------------------------------------

def _mfe_mae_analysis(trades: list) -> str:
    """MFE/MAE analysis: edge ratio, capture ratio, loser/winner profiles."""
    if not trades:
        return "=== MFE / MAE Analysis ===\n  No trades."

    lines = ["=== MFE / MAE Analysis ==="]

    mfe_arr = np.array([t.mfe_r for t in trades])
    mae_arr = np.array([t.mae_r for t in trades])

    # Overall distribution
    lines.append("  MFE (R) distribution:")
    lines.append(f"    Mean={np.mean(mfe_arr):.3f}  Median={np.median(mfe_arr):.3f}  "
                 f"P25={np.percentile(mfe_arr, 25):.3f}  P75={np.percentile(mfe_arr, 75):.3f}  "
                 f"P90={np.percentile(mfe_arr, 90):.3f}")
    lines.append("  MAE (R) distribution:")
    lines.append(f"    Mean={np.mean(mae_arr):.3f}  Median={np.median(mae_arr):.3f}  "
                 f"P25={np.percentile(mae_arr, 25):.3f}  P75={np.percentile(mae_arr, 75):.3f}  "
                 f"P90={np.percentile(mae_arr, 90):.3f}")

    # Edge ratio
    avg_mae = np.mean(mae_arr)
    edge_ratio = np.mean(mfe_arr) / abs(avg_mae) if abs(avg_mae) > 0 else 0.0
    lines.append(f"\n  Edge ratio (avgMFE / |avgMAE|): {edge_ratio:.2f}")

    # MFE of losers — exit timing diagnosis
    losers = [t for t in trades if t.r_multiple <= 0]
    if losers:
        loser_mfes = np.array([t.mfe_r for t in losers])
        loser_went_profit = sum(1 for m in loser_mfes if m >= 0.5)
        pct_went_profit = 100 * loser_went_profit / len(losers)
        lines.append(f"\n  MFE of losers:")
        lines.append(f"    Mean={np.mean(loser_mfes):.3f}  Median={np.median(loser_mfes):.3f}")
        lines.append(f"    Losers reaching >= 0.5R MFE: {loser_went_profit}/{len(losers)} ({pct_went_profit:.0f}%)")
        if pct_went_profit > 50:
            lines.append("    ** >50% of losers went profitable first — EXIT TIMING issue, not entry quality")

    # MAE of winners
    winners = [t for t in trades if t.r_multiple > 0]
    if winners:
        winner_maes = np.array([t.mae_r for t in winners])
        lines.append(f"\n  MAE of winners (heat absorbed):")
        lines.append(f"    Mean={np.mean(winner_maes):.3f}  Median={np.median(winner_maes):.3f}")

        # Winner capture ratio
        captures = [t.r_multiple / t.mfe_r for t in winners if t.mfe_r > 0]
        if captures:
            lines.append(f"\n  Winner capture ratio (R / MFE):")
            lines.append(f"    Mean={np.mean(captures):.2f}  Median={np.median(captures):.2f}")

    # Bucketed by entry_type
    lines.append(f"\n  By entry_type:")
    header = f"    {'Type':12s} {'N':>5s} {'AvgMFE':>7s} {'AvgMAE':>7s} {'EdgeR':>6s}"
    lines.append(header)
    by_type: dict[str, list] = {}
    for t in trades:
        by_type.setdefault(t.entry_type, []).append(t)
    for etype in sorted(by_type):
        group = by_type[etype]
        n = len(group)
        am = np.mean([t.mfe_r for t in group])
        aa = np.mean([t.mae_r for t in group])
        er = am / abs(aa) if abs(aa) > 0 else 0.0
        lines.append(f"    {etype:12s} {n:5d} {am:+7.3f} {aa:+7.3f} {er:6.2f}")

    # Bucketed by regime
    lines.append(f"\n  By regime (DT/VolState):")
    header = f"    {'Regime':20s} {'N':>5s} {'AvgMFE':>7s} {'AvgMAE':>7s} {'EdgeR':>6s}"
    lines.append(header)
    by_regime: dict[str, list] = {}
    for t in trades:
        key = f"DT={t.daily_trend:+d}/{t.vol_state}"
        by_regime.setdefault(key, []).append(t)
    for regime in sorted(by_regime):
        group = by_regime[regime]
        n = len(group)
        am = np.mean([t.mfe_r for t in group])
        aa = np.mean([t.mae_r for t in group])
        er = am / abs(aa) if abs(aa) > 0 else 0.0
        lines.append(f"    {regime:20s} {n:5d} {am:+7.3f} {aa:+7.3f} {er:6.2f}")

    # Bucketed by sub_window
    lines.append(f"\n  By sub_window:")
    header = f"    {'SubWindow':10s} {'N':>5s} {'AvgMFE':>7s} {'AvgMAE':>7s} {'EdgeR':>6s}"
    lines.append(header)
    by_sw: dict[str, list] = {}
    for t in trades:
        by_sw.setdefault(t.sub_window, []).append(t)
    for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]:
        group = by_sw.get(sw, [])
        if not group:
            continue
        n = len(group)
        am = np.mean([t.mfe_r for t in group])
        aa = np.mean([t.mae_r for t in group])
        er = am / abs(aa) if abs(aa) > 0 else 0.0
        lines.append(f"    {sw:10s} {n:5d} {am:+7.3f} {aa:+7.3f} {er:6.2f}")

    return "\n".join(lines)


def _r_multiple_distribution(trades: list) -> str:
    """R-multiple distribution and cumulative thresholds."""
    if not trades:
        return "=== R-Multiple Distribution ===\n  No trades."

    lines = ["=== R-Multiple Distribution ==="]
    rs = np.array([t.r_multiple for t in trades])
    n = len(rs)

    lines.append(f"  Mean:   {np.mean(rs):+.3f}")
    lines.append(f"  Median: {np.median(rs):+.3f}")
    lines.append(f"  Std:    {np.std(rs):.3f}")

    # Positive skew indicator
    if np.mean(rs) > np.median(rs):
        lines.append("  Skew:   POSITIVE (mean > median — letting winners run)")
    elif np.mean(rs) < np.median(rs):
        lines.append("  Skew:   NEGATIVE (mean < median — fat left tail)")
    else:
        lines.append("  Skew:   SYMMETRIC")

    # Cumulative counts at thresholds
    lines.append(f"\n  Cumulative distribution:")
    thresholds = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]
    for thresh in thresholds:
        count = int(np.sum(rs >= thresh))
        pct = 100 * count / n
        lines.append(f"    >= {thresh:+.0f}R:  {count:5d} ({pct:5.1f}%)")

    return "\n".join(lines)


def _direction_breakdown(trades: list) -> str:
    """Long vs Short performance breakdown."""
    if not trades:
        return "=== Direction Breakdown ===\n  No trades."

    lines = ["=== Direction Breakdown ==="]
    header = f"  {'Direction':10s} {'N':>5s} {'WinR':>6s} {'AvgR':>7s} {'PF':>6s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, dir_val in [("Long", 1), ("Short", -1)]:
        group = [t for t in trades if t.direction == dir_val]
        if not group:
            continue
        n = len(group)
        rs = [t.r_multiple for t in group]
        pnls = [t.pnl_dollars for t in group]
        win_rate = sum(1 for r in rs if r > 0) / n * 100
        avg_r = np.mean(rs)
        total_pnl = sum(pnls)
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        pf_str = f"{pf:6.2f}" if pf < 100 else "   inf"
        lines.append(
            f"  {label:10s} {n:5d} {win_rate:5.0f}% {avg_r:+7.3f} {pf_str} ${total_pnl:+9.0f}"
        )

    return "\n".join(lines)


def _streak_analysis(trades: list) -> str:
    """Win/loss streak analysis."""
    if not trades:
        return "=== Streak Analysis ===\n  No trades."

    lines = ["=== Streak Analysis ==="]
    max_win = max_loss = cur_win = cur_loss = 0
    worst_loss_streak_pnl = 0.0
    worst_loss_streak_r = 0.0
    cur_loss_pnl = 0.0
    cur_loss_r = 0.0
    win_streaks: list[int] = []
    loss_streaks: list[int] = []

    for t in trades:
        if t.r_multiple > 0:
            cur_win += 1
            if cur_loss > 0:
                loss_streaks.append(cur_loss)
            cur_loss = 0
            cur_loss_pnl = 0.0
            cur_loss_r = 0.0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_loss_pnl += t.pnl_dollars
            cur_loss_r += t.r_multiple
            if cur_win > 0:
                win_streaks.append(cur_win)
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
            worst_loss_streak_pnl = min(worst_loss_streak_pnl, cur_loss_pnl)
            worst_loss_streak_r = min(worst_loss_streak_r, cur_loss_r)

    # Capture final streak
    if cur_win > 0:
        win_streaks.append(cur_win)
    if cur_loss > 0:
        loss_streaks.append(cur_loss)

    lines.append(f"  Max consecutive wins:   {max_win}")
    lines.append(f"  Max consecutive losses: {max_loss}")
    lines.append(f"  Worst consecutive-loss P&L: ${worst_loss_streak_pnl:+,.0f}")
    lines.append(f"  Worst consecutive-loss R:   {worst_loss_streak_r:+.2f}R")

    avg_win_streak = np.mean(win_streaks) if win_streaks else 0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
    lines.append(f"  Avg win streak length:  {avg_win_streak:.1f}")
    lines.append(f"  Avg loss streak length: {avg_loss_streak:.1f}")

    return "\n".join(lines)


def _drawdown_profile(
    equity_curve: np.ndarray | None,
    time_series: np.ndarray | None,
) -> str:
    """Drawdown profile: max DD, episode counts, time underwater."""
    if equity_curve is None or len(equity_curve) == 0:
        return "=== Drawdown Profile ===\n  No equity curve data available."

    lines = ["=== Drawdown Profile ==="]
    eq = np.asarray(equity_curve, dtype=float)

    # Running peak and drawdown series
    peak = np.maximum.accumulate(eq)
    dd_pct = np.where(peak > 0, (eq - peak) / peak * 100, 0.0)

    # Max drawdown
    max_dd_idx = int(np.argmin(dd_pct))
    max_dd = dd_pct[max_dd_idx]
    # Find the peak before the max DD trough
    peak_idx = int(np.argmax(eq[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    dd_duration_bars = max_dd_idx - peak_idx

    lines.append(f"  Max drawdown: {max_dd:.2f}%")

    if time_series is not None and len(time_series) > max(max_dd_idx, peak_idx):
        ts = time_series
        peak_date = str(ts[peak_idx])[:10]
        trough_date = str(ts[max_dd_idx])[:10]
        lines.append(f"    Peak:   bar {peak_idx} ({peak_date})")
        lines.append(f"    Trough: bar {max_dd_idx} ({trough_date})")
        lines.append(f"    Duration: {dd_duration_bars} bars (~{dd_duration_bars * 15 / 60:.0f} hours)")
    else:
        lines.append(f"    Duration: {dd_duration_bars} bars")

    # Count drawdown episodes exceeding thresholds
    lines.append(f"\n  Drawdown episode counts:")
    for thresh in [-2.0, -5.0, -10.0, -15.0]:
        # Count distinct episodes: transition from above to below threshold
        in_dd = dd_pct < thresh
        episodes = 0
        was_in = False
        for v in in_dd:
            if v and not was_in:
                episodes += 1
            was_in = bool(v)
        lines.append(f"    Exceeding {thresh:+.0f}%: {episodes} episodes")

    # % of time underwater
    underwater = np.sum(eq < peak)
    pct_underwater = 100 * underwater / len(eq) if len(eq) > 0 else 0
    lines.append(f"\n  Time underwater (below HWM): {pct_underwater:.1f}% of bars")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 2 — Pattern Discovery (Regime/Timing Optimization)
# ---------------------------------------------------------------------------

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
                 f"({100 * win_months / total_months:.0f}%)" if total_months > 0 else "")

    if monthly_pnls:
        best_idx = int(np.argmax(monthly_pnls))
        worst_idx = int(np.argmin(monthly_pnls))
        months_sorted = sorted(by_month)
        lines.append(f"  Best month:  {months_sorted[best_idx]} (${monthly_pnls[best_idx]:+,.0f})")
        lines.append(f"  Worst month: {months_sorted[worst_idx]} (${monthly_pnls[worst_idx]:+,.0f})")

    return "\n".join(lines)


def _day_of_week(trades: list) -> str:
    """Performance by day of week (ET)."""
    if not trades:
        return "=== Day-of-Week Analysis ===\n  No trades."

    dated = [t for t in trades if t.entry_time]
    if not dated:
        return "=== Day-of-Week Analysis ===\n  No dated trades."

    lines = ["=== Day-of-Week Analysis ==="]

    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        et = None

    header = f"  {'Day':10s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day: dict[int, list] = {}
    for t in dated:
        try:
            dow = t.entry_time.astimezone(et).weekday() if et else t.entry_time.weekday()
        except Exception:
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
        et = None

    header = f"  {'Hour':6s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'TotalPnL':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_hour: dict[int, list] = {}
    for t in dated:
        try:
            h = t.entry_time.astimezone(et).hour if et else t.entry_time.hour
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


def _cross_tab_breakdowns(trades: list) -> str:
    """Cross-tab breakdowns: entry_type x regime, entry_type x sub_window, direction x regime."""
    if not trades:
        return "=== Cross-Tab Breakdowns ===\n  No trades."

    lines = ["=== Cross-Tab Breakdowns ==="]

    def _cell_stats(group: list) -> tuple[int, float, float]:
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100 if n > 0 else 0
        avg_r = float(np.mean([t.r_multiple for t in group])) if group else 0
        return n, wr, avg_r

    def _flag(n: int, avg_r: float) -> str:
        if n >= 5 and avg_r < -0.2:
            return " !WEAK"
        if n >= 5 and avg_r > 0.3:
            return " *STRONG"
        return ""

    # --- Cross-tab 1: Entry Type x Regime (DT/VolState) ---
    lines.append("\n  1) Entry Type x Regime (DT/VolState)")
    entry_types = sorted(set(t.entry_type for t in trades))
    regimes = sorted(set(f"DT={t.daily_trend:+d}/{t.vol_state}" for t in trades))

    r_header = "  ".join(f"{r:>20s}" for r in regimes)
    lines.append(f"    {'Type':12s}  {r_header}")
    lines.append("    " + "-" * (14 + 22 * len(regimes)))

    flags: list[str] = []
    for etype in entry_types:
        cells = []
        for regime in regimes:
            group = [t for t in trades if t.entry_type == etype
                     and f"DT={t.daily_trend:+d}/{t.vol_state}" == regime]
            if not group:
                cells.append(f"{'--':>20s}")
            else:
                n, wr, avg_r = _cell_stats(group)
                tag = _flag(n, avg_r)
                cells.append(f"{n:3d} {wr:3.0f}% {avg_r:+.2f}{tag:>7s}")
                if tag.strip():
                    flags.append(f"    {tag.strip()}: {etype} x {regime} (N={n}, avgR={avg_r:+.3f})")
        lines.append(f"    {etype:12s}  {'  '.join(cells)}")

    # --- Cross-tab 2: Entry Type x Sub-Window ---
    lines.append(f"\n  2) Entry Type x Sub-Window")
    sub_windows = [sw for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]
                   if any(t.sub_window == sw for t in trades)]

    r_header = "  ".join(f"{sw:>20s}" for sw in sub_windows)
    lines.append(f"    {'Type':12s}  {r_header}")
    lines.append("    " + "-" * (14 + 22 * len(sub_windows)))

    for etype in entry_types:
        cells = []
        for sw in sub_windows:
            group = [t for t in trades if t.entry_type == etype and t.sub_window == sw]
            if not group:
                cells.append(f"{'--':>20s}")
            else:
                n, wr, avg_r = _cell_stats(group)
                tag = _flag(n, avg_r)
                cells.append(f"{n:3d} {wr:3.0f}% {avg_r:+.2f}{tag:>7s}")
                if tag.strip():
                    flags.append(f"    {tag.strip()}: {etype} x {sw} (N={n}, avgR={avg_r:+.3f})")
        lines.append(f"    {etype:12s}  {'  '.join(cells)}")

    # --- Cross-tab 3: Direction x Regime ---
    lines.append(f"\n  3) Direction x Regime (DT/VolState)")
    directions = [("Long", 1), ("Short", -1)]

    r_header = "  ".join(f"{r:>20s}" for r in regimes)
    lines.append(f"    {'Dir':10s}  {r_header}")
    lines.append("    " + "-" * (12 + 22 * len(regimes)))

    for label, dir_val in directions:
        cells = []
        for regime in regimes:
            group = [t for t in trades if t.direction == dir_val
                     and f"DT={t.daily_trend:+d}/{t.vol_state}" == regime]
            if not group:
                cells.append(f"{'--':>20s}")
            else:
                n, wr, avg_r = _cell_stats(group)
                tag = _flag(n, avg_r)
                cells.append(f"{n:3d} {wr:3.0f}% {avg_r:+.2f}{tag:>7s}")
                if tag.strip():
                    flags.append(f"    {tag.strip()}: {label} x {regime} (N={n}, avgR={avg_r:+.3f})")
        lines.append(f"    {label:10s}  {'  '.join(cells)}")

    if flags:
        lines.append("")
        lines.extend(flags)

    return "\n".join(lines)


def _stop_distance_analysis(trades: list) -> str:
    """Stop distance analysis: distribution, stop-hit stats, exit proximity."""
    if not trades:
        return "=== Stop Distance Analysis ===\n  No trades."

    lines = ["=== Stop Distance Analysis ==="]

    # Stop distance in points
    stop_dists = []
    for t in trades:
        dist = abs(t.entry_price - t.initial_stop)
        stop_dists.append(dist)
    arr = np.array(stop_dists)

    lines.append(f"  Stop distance (points):")
    lines.append(f"    Mean={np.mean(arr):.2f}  Median={np.median(arr):.2f}  "
                 f"P25={np.percentile(arr, 25):.2f}  P75={np.percentile(arr, 75):.2f}  "
                 f"Max={np.max(arr):.2f}")

    # % that hit protective stop
    stop_trades = [t for t in trades if t.exit_reason == "STOP"]
    non_stop = [t for t in trades if t.exit_reason != "STOP"]
    pct_stop = 100 * len(stop_trades) / len(trades)
    lines.append(f"\n  Protective stop hit: {len(stop_trades)}/{len(trades)} ({pct_stop:.0f}%)")

    # For stop-hit trades: avg R at exit
    if stop_trades:
        avg_r_stop = np.mean([t.r_multiple for t in stop_trades])
        lines.append(f"    Avg R at stop exit: {avg_r_stop:+.3f} (ideal ~ -1.0R)")

    # For non-stop exits: avg distance from initial stop at exit
    if non_stop:
        dists_from_stop = []
        for t in non_stop:
            if t.direction == 1:  # Long
                dist = t.exit_price - t.initial_stop
            else:  # Short
                dist = t.initial_stop - t.exit_price
            dists_from_stop.append(dist)
        arr_ns = np.array(dists_from_stop)
        lines.append(f"\n  Non-stop exits ({len(non_stop)} trades):")
        lines.append(f"    Avg distance from initial stop at exit: {np.mean(arr_ns):+.2f} pts")
        lines.append(f"    Median: {np.median(arr_ns):+.2f} pts")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 3 — Refinement
# ---------------------------------------------------------------------------

def _entry_efficiency(trades: list) -> str:
    """Entry efficiency: slippage analysis (signal_entry_price vs entry_price)."""
    if not trades:
        return "=== Entry Efficiency ===\n  No trades."

    # Check if signal_entry_price is populated
    has_signal = any(t.signal_entry_price != 0.0 for t in trades)
    if not has_signal:
        return ("=== Entry Efficiency ===\n"
                "  signal_entry_price not populated — enable in VdubusTradeRecord to see slippage analysis.")

    lines = ["=== Entry Efficiency ==="]

    slippages_pts = []
    slippages_frac = []
    for t in trades:
        if t.signal_entry_price == 0.0:
            continue
        # Adverse slippage: positive means we paid more (long) or received less (short)
        slip = (t.entry_price - t.signal_entry_price) * t.direction
        slippages_pts.append(slip)
        risk = abs(t.entry_price - t.initial_stop)
        if risk > 0:
            slippages_frac.append(slip / risk)

    if not slippages_pts:
        lines.append("  No valid slippage data.")
        return "\n".join(lines)

    arr_pts = np.array(slippages_pts)
    arr_ticks = arr_pts / 0.25
    arr_frac = np.array(slippages_frac) if slippages_frac else np.array([0.0])

    lines.append(f"  Adverse slippage (positive = worse fill):")
    lines.append(f"    Mean:   {np.mean(arr_pts):+.2f} pts  ({np.mean(arr_ticks):+.1f} ticks)  "
                 f"{np.mean(arr_frac) * 100:+.1f}% of initial risk")
    lines.append(f"    Median: {np.median(arr_pts):+.2f} pts  ({np.median(arr_ticks):+.1f} ticks)")

    # Bucketed by sub_window
    lines.append(f"\n  By sub_window:")
    header = f"    {'SubWindow':10s} {'N':>5s} {'MeanSlip':>10s} {'MeanTicks':>10s}"
    lines.append(header)
    by_sw: dict[str, list[float]] = {}
    sw_map: dict[str, list] = {}
    for t in trades:
        if t.signal_entry_price == 0.0:
            continue
        slip = (t.entry_price - t.signal_entry_price) * t.direction
        by_sw.setdefault(t.sub_window, []).append(slip)

    for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]:
        slips = by_sw.get(sw, [])
        if not slips:
            continue
        ms = np.mean(slips)
        mt = ms / 0.25
        lines.append(f"    {sw:10s} {len(slips):5d} {ms:+10.2f} {mt:+10.1f}")

    return "\n".join(lines)


def _hold_time_optimization(trades: list) -> str:
    """Hold time optimization: performance by bars_held_15m buckets."""
    if not trades:
        return "=== Hold Time Optimization ===\n  No trades."

    lines = ["=== Hold Time Optimization ==="]

    buckets = [
        ("1-4", 1, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33-64", 33, 64),
        ("65+", 65, 999999),
    ]

    header = f"  {'Bars':8s} {'N':>5s} {'WR':>6s} {'AvgR':>7s} {'AvgMFE':>7s} {'AvgMAE':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    best_bucket = ""
    best_avg_r = -999.0

    for label, lo, hi in buckets:
        group = [t for t in trades if lo <= t.bars_held_15m <= hi]
        if not group:
            continue
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = float(np.mean([t.r_multiple for t in group]))
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_mae = np.mean([t.mae_r for t in group])
        marker = ""
        if avg_r > best_avg_r:
            best_avg_r = avg_r
            best_bucket = label
        lines.append(
            f"  {label:8s} {n:5d} {wr:5.0f}% {avg_r:+7.3f} {avg_mfe:+7.3f} {avg_mae:+7.3f}"
        )

    if best_bucket:
        lines.append(f"\n  ** Optimal hold bucket: {best_bucket} bars (AvgR={best_avg_r:+.3f})")

    # Avg hold time for winners vs losers
    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    if winners:
        lines.append(f"  Avg hold (winners): {np.mean([t.bars_held_15m for t in winners]):.1f} bars")
    if losers:
        lines.append(f"  Avg hold (losers):  {np.mean([t.bars_held_15m for t in losers]):.1f} bars")

    return "\n".join(lines)


def _trade_autocorrelation(trades: list) -> str:
    """Trade autocorrelation: lag-1 of R-multiples and binary win/loss."""
    if not trades or len(trades) < 4:
        return "=== Trade Autocorrelation ===\n  Insufficient trades (need >= 4)."

    lines = ["=== Trade Autocorrelation ==="]

    rs = np.array([t.r_multiple for t in trades])

    # Lag-1 autocorrelation of R-multiples
    r_mean = np.mean(rs)
    diffs = rs - r_mean
    var = np.sum(diffs ** 2)
    if var > 0:
        r_autocorr = float(np.sum(diffs[:-1] * diffs[1:]) / var)
    else:
        r_autocorr = 0.0

    # Lag-1 autocorrelation of binary win/loss
    wins = (rs > 0).astype(float)
    w_mean = np.mean(wins)
    w_diffs = wins - w_mean
    w_var = np.sum(w_diffs ** 2)
    if w_var > 0:
        w_autocorr = float(np.sum(w_diffs[:-1] * w_diffs[1:]) / w_var)
    else:
        w_autocorr = 0.0

    lines.append(f"  Lag-1 autocorrelation (R-multiples): {r_autocorr:+.3f}")
    lines.append(f"  Lag-1 autocorrelation (win/loss):    {w_autocorr:+.3f}")

    # Interpretation
    if r_autocorr > 0.15:
        interp = "CLUSTERING (wins follow wins, losses follow losses)"
    elif r_autocorr < -0.15:
        interp = "ALTERNATING (wins tend to follow losses and vice versa)"
    else:
        interp = "INDEPENDENT (no significant serial dependency)"
    lines.append(f"  Interpretation: {interp}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 4 — Weakness Discovery
# ---------------------------------------------------------------------------

def _winner_giveback_analysis(trades: list) -> str:
    """Analyse how much profit winners surrender from MFE to exit.

    Directly measures trailing stop efficiency. A capture ratio of 0.55 means
    45% of peak profit is given back — this section breaks down WHERE and WHY.
    """
    winners = [t for t in trades if t.r_multiple > 0 and t.mfe_r > 0]
    if not winners:
        return "=== Winner Give-Back Analysis ===\n  No winners."

    lines = ["=== Winner Give-Back Analysis ==="]

    # Overall give-back
    givebacks_r = [t.mfe_r - t.r_multiple for t in winners]
    givebacks_pct = [(t.mfe_r - t.r_multiple) / t.mfe_r * 100
                     for t in winners if t.mfe_r > 0]
    captures = [t.r_multiple / t.mfe_r for t in winners if t.mfe_r > 0]

    gb = np.array(givebacks_r)
    cap = np.array(captures)
    gb_pct = np.array(givebacks_pct)

    lines.append(f"  Winners: {len(winners)}")
    lines.append(f"  Give-back (MFE - exit R):")
    lines.append(f"    Mean={np.mean(gb):.3f}R  Median={np.median(gb):.3f}R  "
                 f"P75={np.percentile(gb, 75):.3f}R  P90={np.percentile(gb, 90):.3f}R")
    lines.append(f"  Give-back as % of MFE:")
    lines.append(f"    Mean={np.mean(gb_pct):.1f}%  Median={np.median(gb_pct):.1f}%")
    lines.append(f"  Capture ratio (exit R / MFE):")
    lines.append(f"    Mean={np.mean(cap):.2f}  Median={np.median(cap):.2f}")

    # Give-back by exit reason
    lines.append(f"\n  Give-back by exit reason:")
    header = f"    {'ExitReason':18s} {'N':>4s} {'AvgGB_R':>8s} {'AvgCapt':>8s} {'AvgMFE':>8s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    by_exit: dict[str, list] = {}
    for t in winners:
        by_exit.setdefault(t.exit_reason, []).append(t)
    for reason in sorted(by_exit, key=lambda r: -np.mean([t.mfe_r - t.r_multiple for t in by_exit[r]])):
        group = by_exit[reason]
        n = len(group)
        avg_gb = np.mean([t.mfe_r - t.r_multiple for t in group])
        avg_cap = np.mean([t.r_multiple / t.mfe_r for t in group if t.mfe_r > 0])
        avg_mfe = np.mean([t.mfe_r for t in group])
        lines.append(f"    {reason:18s} {n:4d} {avg_gb:+8.3f} {avg_cap:8.2f} {avg_mfe:+8.3f}")

    # Give-back by sub_window
    lines.append(f"\n  Give-back by sub_window:")
    header = f"    {'SubWindow':10s} {'N':>4s} {'AvgGB_R':>8s} {'AvgCapt':>8s} {'AvgMFE':>8s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    by_sw: dict[str, list] = {}
    for t in winners:
        by_sw.setdefault(t.sub_window, []).append(t)
    for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]:
        group = by_sw.get(sw, [])
        if not group:
            continue
        n = len(group)
        avg_gb = np.mean([t.mfe_r - t.r_multiple for t in group])
        avg_cap = np.mean([t.r_multiple / t.mfe_r for t in group if t.mfe_r > 0])
        avg_mfe = np.mean([t.mfe_r for t in group])
        lines.append(f"    {sw:10s} {n:4d} {avg_gb:+8.3f} {avg_cap:8.2f} {avg_mfe:+8.3f}")

    # Worst give-backs (biggest R left on table)
    worst = sorted(winners, key=lambda t: -(t.mfe_r - t.r_multiple))[:5]
    if worst:
        lines.append(f"\n  Top 5 worst give-backs:")
        for t in worst:
            gb_r = t.mfe_r - t.r_multiple
            entry_str = t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "?"
            lines.append(f"    {entry_str}  MFE={t.mfe_r:+.2f}R  Exit={t.r_multiple:+.2f}R  "
                         f"GaveBack={gb_r:.2f}R  via {t.exit_reason}  [{t.sub_window}]")

    return "\n".join(lines)


def _loser_autopsy(trades: list) -> str:
    """Deep dive into losers: fast deaths (1-4 bars) vs slow deaths (5+ bars).

    Fast deaths = entry quality issue (signal fired without structural support).
    Slow deaths = exit timing issue (trade had time, failed to develop).
    """
    losers = [t for t in trades if t.r_multiple <= 0]
    if not losers:
        return "=== Loser Autopsy ===\n  No losers."

    lines = ["=== Loser Autopsy ==="]
    lines.append(f"  Total losers: {len(losers)}")

    fast = [t for t in losers if t.bars_held_15m <= 4]
    slow = [t for t in losers if t.bars_held_15m > 4]

    for label, group in [("Fast deaths (1-4 bars)", fast), ("Slow deaths (5+ bars)", slow)]:
        if not group:
            lines.append(f"\n  {label}: 0 trades")
            continue
        n = len(group)
        pct = n / len(losers) * 100
        avg_r = np.mean([t.r_multiple for t in group])
        total_pnl = sum(t.pnl_dollars for t in group)
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_mae = np.mean([t.mae_r for t in group])
        reached_profit = sum(1 for t in group if t.mfe_r >= 0.3)

        lines.append(f"\n  {label}: {n} trades ({pct:.0f}% of losers)")
        lines.append(f"    AvgR={avg_r:+.3f}  TotalPnL=${total_pnl:+,.0f}")
        lines.append(f"    AvgMFE={avg_mfe:+.3f}R  AvgMAE={avg_mae:+.3f}R")
        lines.append(f"    Reached >=0.3R profit: {reached_profit}/{n} ({reached_profit/n*100:.0f}%)")

        # Sub-window distribution
        sw_counts: dict[str, int] = {}
        for t in group:
            sw_counts[t.sub_window] = sw_counts.get(t.sub_window, 0) + 1
        sw_str = "  ".join(f"{sw}={c}" for sw, c in sorted(sw_counts.items(), key=lambda x: -x[1]))
        lines.append(f"    Sub-windows: {sw_str}")

        # Exit reason distribution
        exit_counts: dict[str, int] = {}
        for t in group:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
        exit_str = "  ".join(f"{r}={c}" for r, c in sorted(exit_counts.items(), key=lambda x: -x[1]))
        lines.append(f"    Exit reasons: {exit_str}")

    # Diagnosis
    lines.append(f"\n  Diagnosis:")
    if fast:
        fast_pct = len(fast) / len(losers) * 100
        if fast_pct > 50:
            lines.append(f"    ** {fast_pct:.0f}% of losers die within 4 bars — ENTRY QUALITY issue")
            lines.append(f"       Signal fires without sufficient structural support.")
        fast_mfe_low = sum(1 for t in fast if t.mfe_r < 0.2) / len(fast) * 100 if fast else 0
        if fast_mfe_low > 70:
            lines.append(f"    ** {fast_mfe_low:.0f}% of fast deaths never reach 0.2R — trades have no edge at entry")
    if slow:
        slow_recovered = sum(1 for t in slow if t.mfe_r >= 0.5) / len(slow) * 100 if slow else 0
        if slow_recovered > 30:
            lines.append(f"    ** {slow_recovered:.0f}% of slow deaths reached 0.5R+ before failing — EXIT TIMING issue")
            lines.append(f"       Consider earlier profit protection or tighter trailing for these cases.")

    return "\n".join(lines)


def _vwap_distance_analysis(trades: list) -> str:
    """Analyse entry distance from VWAP and its relationship to trade outcome.

    Closer entries should have better win rates (genuine pullback structure).
    """
    valid = [t for t in trades if t.vwap_used_at_entry > 0 and t.entry_price > 0]
    if not valid:
        return "=== VWAP Distance at Entry ===\n  No trades with VWAP data."

    lines = ["=== VWAP Distance at Entry ==="]

    # Compute distance in points and as fraction of stop distance
    dists_pts = []
    dists_r_frac = []
    for t in valid:
        if t.direction == 1:  # Long
            dist = t.entry_price - t.vwap_used_at_entry
        else:
            dist = t.vwap_used_at_entry - t.entry_price
        dists_pts.append(dist)
        r_pts = abs(t.entry_price - t.initial_stop)
        if r_pts > 0:
            dists_r_frac.append(dist / r_pts)

    arr_pts = np.array(dists_pts)
    arr_frac = np.array(dists_r_frac) if dists_r_frac else np.array([0.0])

    lines.append(f"  Distance from VWAP at entry (points):")
    lines.append(f"    Mean={np.mean(arr_pts):.1f}  Median={np.median(arr_pts):.1f}  "
                 f"P25={np.percentile(arr_pts, 25):.1f}  P75={np.percentile(arr_pts, 75):.1f}")
    lines.append(f"  Distance as fraction of R:")
    lines.append(f"    Mean={np.mean(arr_frac):.2f}R  Median={np.median(arr_frac):.2f}R")

    # Bucket by distance and show performance
    lines.append(f"\n  Performance by VWAP distance (fraction of R):")
    header = f"    {'Distance':12s} {'N':>4s} {'WinR':>6s} {'AvgR':>7s} {'AvgMFE':>7s} {'EdgeR':>6s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    buckets = [
        ("0.0-0.2R", 0.0, 0.2),
        ("0.2-0.4R", 0.2, 0.4),
        ("0.4-0.6R", 0.4, 0.6),
        ("0.6-1.0R", 0.6, 1.0),
        ("1.0R+", 1.0, 999),
    ]

    for label, lo, hi in buckets:
        group = [t for t, d in zip(valid, dists_r_frac) if lo <= d < hi]
        if not group:
            continue
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        avg_mfe = np.mean([t.mfe_r for t in group])
        avg_mae = np.mean([t.mae_r for t in group])
        er = avg_mfe / abs(avg_mae) if abs(avg_mae) > 0 else 0
        lines.append(f"    {label:12s} {n:4d} {wr:5.0f}% {avg_r:+7.3f} {avg_mfe:+7.3f} {er:6.2f}")

    # Correlation
    rs = np.array([t.r_multiple for t in valid])
    if len(arr_frac) > 3 and len(rs) == len(arr_frac):
        corr = float(np.corrcoef(arr_frac, rs)[0, 1])
        lines.append(f"\n  Correlation (VWAP distance vs R-multiple): {corr:+.3f}")
        if corr < -0.15:
            lines.append(f"    ** Negative correlation — closer entries perform better (tighten VWAP cap?)")
        elif corr > 0.15:
            lines.append(f"    ** Positive correlation — further entries perform better (loosen VWAP cap?)")
        else:
            lines.append(f"    No significant relationship between distance and outcome.")

    return "\n".join(lines)


def _pnl_development_curve(trades: list) -> str:
    """Average R-trajectory by bars since entry.

    Shows when trades reach peak profitability and when they start giving back.
    Identifies optimal exit timing.
    """
    if not trades:
        return "=== P&L Development Curve ===\n  No trades."

    lines = ["=== P&L Development Curve ==="]
    lines.append("  (Approximated from MFE/MAE timing and hold duration)")

    # We don't have bar-by-bar P&L, but we can profile using
    # hold duration vs final R and MFE to infer development
    # Group by hold time and show the progression
    max_bars = max(t.bars_held_15m for t in trades)
    checkpoints = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    checkpoints = [c for c in checkpoints if c <= max_bars + 1]

    # For each checkpoint, look at trades still alive at that bar
    # and their eventual outcome
    lines.append(f"\n  Outcome by minimum hold duration:")
    header = f"    {'MinBars':>8s} {'Alive':>6s} {'WinR':>6s} {'AvgR':>7s} {'AvgMFE':>7s} {'AvgMAE':>7s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for cp in checkpoints:
        alive = [t for t in trades if t.bars_held_15m >= cp]
        if not alive:
            continue
        n = len(alive)
        wr = sum(1 for t in alive if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in alive])
        avg_mfe = np.mean([t.mfe_r for t in alive])
        avg_mae = np.mean([t.mae_r for t in alive])
        lines.append(f"    {cp:>8d} {n:6d} {wr:5.0f}% {avg_r:+7.3f} {avg_mfe:+7.3f} {avg_mae:+7.3f}")

    # Trades that exited at each checkpoint range
    lines.append(f"\n  Exit timing distribution (when trades close):")
    header = f"    {'ExitBars':>10s} {'N':>4s} {'WinR':>6s} {'AvgR':>7s} {'Verdict':>10s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    exit_buckets = [
        ("1-2", 1, 2), ("3-4", 3, 4), ("5-8", 5, 8),
        ("9-16", 9, 16), ("17-32", 17, 32), ("33+", 33, 999999),
    ]
    for label, lo, hi in exit_buckets:
        group = [t for t in trades if lo <= t.bars_held_15m <= hi]
        if not group:
            continue
        n = len(group)
        wr = sum(1 for t in group if t.r_multiple > 0) / n * 100
        avg_r = np.mean([t.r_multiple for t in group])
        if avg_r < -0.3 and n >= 5:
            verdict = "DRAG"
        elif avg_r > 0.5 and wr > 50:
            verdict = "SWEET SPOT"
        else:
            verdict = ""
        lines.append(f"    {label:>10s} {n:4d} {wr:5.0f}% {avg_r:+7.3f} {verdict:>10s}")

    return "\n".join(lines)


def _exit_reason_x_subwindow(trades: list) -> str:
    """Cross-tab of exit reason x sub-window.

    Reveals which exit mechanisms dominate in each window and whether
    certain exits are producing systematic drag in specific windows.
    """
    if not trades:
        return "=== Exit Reason x Sub-Window ===\n  No trades."

    lines = ["=== Exit Reason x Sub-Window ==="]

    exit_reasons = sorted(set(t.exit_reason for t in trades))
    sub_windows = [sw for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]
                   if any(t.sub_window == sw for t in trades)]

    # Table header
    sw_header = "  ".join(f"{sw:>18s}" for sw in sub_windows)
    lines.append(f"    {'ExitReason':16s}  {sw_header}")
    lines.append("    " + "-" * (18 + 20 * len(sub_windows)))

    flags: list[str] = []
    for reason in exit_reasons:
        cells = []
        for sw in sub_windows:
            group = [t for t in trades if t.exit_reason == reason and t.sub_window == sw]
            if not group:
                cells.append(f"{'--':>18s}")
            else:
                n = len(group)
                avg_r = np.mean([t.r_multiple for t in group])
                total = sum(t.pnl_dollars for t in group)
                cells.append(f"{n:3d} {avg_r:+.2f}R ${total:+7.0f}")
                if n >= 3 and avg_r < -0.3:
                    flags.append(f"    !DRAG: {reason} in {sw} (N={n}, AvgR={avg_r:+.3f}, ${total:+,.0f})")
        lines.append(f"    {reason:16s}  {'  '.join(cells)}")

    if flags:
        lines.append(f"\n  Identified drags:")
        lines.extend(flags)

    return "\n".join(lines)


def _rolling_stability(trades: list) -> str:
    """Rolling 20-trade window analysis to detect regime degradation.

    Shows whether the strategy is improving, degrading, or stable over time.
    """
    n = len(trades)
    if n < 25:
        return "=== Rolling Performance Stability ===\n  Insufficient trades (need >= 25)."

    lines = ["=== Rolling Performance Stability ==="]
    window = 20

    rs = np.array([t.r_multiple for t in trades])
    pnls = np.array([t.pnl_dollars for t in trades])

    rolling_wr: list[float] = []
    rolling_avg_r: list[float] = []
    rolling_pf: list[float] = []
    rolling_pnl: list[float] = []

    for i in range(window, n + 1):
        chunk_r = rs[i - window:i]
        chunk_pnl = pnls[i - window:i]
        wr = np.mean(chunk_r > 0) * 100
        avg_r = float(np.mean(chunk_r))
        wins_pnl = float(np.sum(chunk_pnl[chunk_pnl > 0]))
        loss_pnl = float(abs(np.sum(chunk_pnl[chunk_pnl < 0])))
        pf = wins_pnl / loss_pnl if loss_pnl > 0 else 99.0
        total = float(np.sum(chunk_pnl))
        rolling_wr.append(wr)
        rolling_avg_r.append(avg_r)
        rolling_pf.append(min(pf, 99.0))
        rolling_pnl.append(total)

    rwr = np.array(rolling_wr)
    rar = np.array(rolling_avg_r)
    rpf = np.array(rolling_pf)
    rpnl = np.array(rolling_pnl)

    lines.append(f"  Window: {window} trades")
    lines.append(f"  Periods: {len(rolling_wr)}")

    lines.append(f"\n  Rolling Win Rate:")
    lines.append(f"    Min={np.min(rwr):.0f}%  Max={np.max(rwr):.0f}%  "
                 f"Current={rwr[-1]:.0f}%  Avg={np.mean(rwr):.0f}%")

    lines.append(f"  Rolling Avg R:")
    lines.append(f"    Min={np.min(rar):+.3f}  Max={np.max(rar):+.3f}  "
                 f"Current={rar[-1]:+.3f}  Avg={np.mean(rar):+.3f}")

    lines.append(f"  Rolling Profit Factor:")
    lines.append(f"    Min={np.min(rpf):.2f}  Max={np.max(rpf):.2f}  "
                 f"Current={rpf[-1]:.2f}  Avg={np.mean(rpf):.2f}")

    lines.append(f"  Rolling 20-Trade P&L:")
    lines.append(f"    Min=${np.min(rpnl):+,.0f}  Max=${np.max(rpnl):+,.0f}  "
                 f"Current=${rpnl[-1]:+,.0f}")

    # Trend detection: compare first half to second half
    mid = len(rolling_avg_r) // 2
    if mid > 0:
        first_half = np.mean(rar[:mid])
        second_half = np.mean(rar[mid:])
        delta = second_half - first_half

        lines.append(f"\n  Trend detection:")
        lines.append(f"    First half avg R:  {first_half:+.3f}")
        lines.append(f"    Second half avg R: {second_half:+.3f}")
        lines.append(f"    Delta:             {delta:+.3f}")
        if delta < -0.3:
            lines.append(f"    ** DEGRADING — strategy edge may be weakening")
        elif delta > 0.3:
            lines.append(f"    ** IMPROVING — recent trades outperforming")
        else:
            lines.append(f"    STABLE — no significant trend")

    # Worst rolling period
    worst_idx = int(np.argmin(rpnl))
    worst_start = worst_idx
    worst_end = worst_idx + window - 1
    if worst_start < n and worst_end < n:
        start_time = trades[worst_start].entry_time
        end_time = trades[min(worst_end, n - 1)].entry_time
        start_str = start_time.strftime("%Y-%m-%d") if start_time else "?"
        end_str = end_time.strftime("%Y-%m-%d") if end_time else "?"
        lines.append(f"\n  Worst 20-trade window:")
        lines.append(f"    Trades {worst_start + 1}-{worst_end + 1}  ({start_str} to {end_str})")
        lines.append(f"    P&L=${rpnl[worst_idx]:+,.0f}  WR={rwr[worst_idx]:.0f}%  "
                     f"AvgR={rar[worst_idx]:+.3f}")

    return "\n".join(lines)


def _mae_recovery_analysis(trades: list) -> str:
    """Analyse adverse excursion recovery rates.

    For trades that went X-R adverse, what fraction recovered to profit?
    Shows whether stop placement is working or if stops are too tight/loose.
    """
    if not trades:
        return "=== MAE Recovery Analysis ===\n  No trades."

    lines = ["=== MAE Recovery Analysis ==="]

    mae_thresholds = [0.25, 0.50, 0.75, 1.0]
    lines.append(f"  Trades reaching MAE threshold — recovery rate:")
    header = f"    {'MAE >=':>8s} {'N':>5s} {'Recovered':>10s} {'RecovR%':>8s} {'AvgR_rec':>9s} {'AvgR_die':>9s}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for thresh in mae_thresholds:
        reached = [t for t in trades if t.mae_r >= thresh]
        if not reached:
            continue
        recovered = [t for t in reached if t.r_multiple > 0]
        died = [t for t in reached if t.r_multiple <= 0]
        n = len(reached)
        n_rec = len(recovered)
        rec_pct = n_rec / n * 100

        avg_r_rec = np.mean([t.r_multiple for t in recovered]) if recovered else 0.0
        avg_r_die = np.mean([t.r_multiple for t in died]) if died else 0.0

        lines.append(f"    {thresh:>7.2f}R {n:5d} {n_rec:5d}/{n:<4d} {rec_pct:7.1f}% "
                     f"{avg_r_rec:+9.3f} {avg_r_die:+9.3f}")

    # Heat absorbed by winners vs losers
    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]

    if winners and losers:
        lines.append(f"\n  Heat absorbed comparison:")
        lines.append(f"    Winners (N={len(winners)}): "
                     f"AvgMAE={np.mean([t.mae_r for t in winners]):.3f}R  "
                     f"MedianMAE={np.median([t.mae_r for t in winners]):.3f}R")
        lines.append(f"    Losers  (N={len(losers)}):  "
                     f"AvgMAE={np.mean([t.mae_r for t in losers]):.3f}R  "
                     f"MedianMAE={np.median([t.mae_r for t in losers]):.3f}R")

        # Winners that absorbed > 0.5R MAE
        heavy_heat_wins = [t for t in winners if t.mae_r > 0.5]
        if heavy_heat_wins:
            lines.append(f"\n  Winners absorbing >0.5R heat: {len(heavy_heat_wins)}/{len(winners)} "
                         f"({len(heavy_heat_wins)/len(winners)*100:.0f}%)")
            lines.append(f"    These trades needed wide stops to survive.")
            lines.append(f"    AvgR={np.mean([t.r_multiple for t in heavy_heat_wins]):+.3f}  "
                         f"AvgMFE={np.mean([t.mfe_r for t in heavy_heat_wins]):+.3f}")

    # Stop tightness diagnostic
    stop_exits = [t for t in trades if t.exit_reason == "STOP"]
    if stop_exits:
        avg_stop_r = np.mean([t.r_multiple for t in stop_exits])
        lines.append(f"\n  Stop exit diagnostic:")
        lines.append(f"    Stop exits: {len(stop_exits)} trades")
        lines.append(f"    Avg R at stop: {avg_stop_r:+.3f}")
        if avg_stop_r > -0.5:
            lines.append(f"    ** Stops triggering well above -1.0R — trailing stop is working")
            lines.append(f"       Average stop loss is only {abs(avg_stop_r):.1f}R (vs -1.0R max)")
        elif avg_stop_r < -0.9:
            lines.append(f"    ** Most stops hitting near -1.0R — initial stop too tight, no trail benefit")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 5 — Structural Deep-Dives
# ---------------------------------------------------------------------------


def _stale_exit_deep_dive(trades: list) -> str:
    """Analyse STALE exits: sub-window clustering, MFE before stale, R-at-exit
    distribution, and comparison vs non-stale exits."""
    stale = [t for t in trades if t.exit_reason == "STALE"]
    non_stale = [t for t in trades if t.exit_reason != "STALE"]
    if not stale:
        return ""

    lines = ["=== Stale Exit Deep Dive ==="]
    pct = len(stale) / len(trades) * 100
    lines.append(f"  Stale exits: {len(stale)}/{len(trades)} ({pct:.0f}% of all trades)")

    # Sub-window distribution
    sw_counts = Counter(t.sub_window for t in stale)
    lines.append("\n  Sub-window clustering:")
    for sw in ["OPEN", "CORE", "CLOSE", "EVENING"]:
        n = sw_counts.get(sw, 0)
        sw_all = [t for t in stale if t.sub_window == sw]
        if n == 0:
            continue
        avg_r = np.mean([t.r_multiple for t in sw_all])
        avg_mfe = np.mean([t.mfe_r for t in sw_all])
        lines.append(f"    {sw:<10} {n:>3} trades  AvgR={avg_r:+.3f}  AvgMFE={avg_mfe:+.3f}")

    # MFE before going stale
    stale_mfes = np.array([t.mfe_r for t in stale])
    lines.append(f"\n  MFE reached before stale exit:")
    lines.append(f"    Mean={np.mean(stale_mfes):+.3f}R  Median={np.median(stale_mfes):+.3f}R  "
                 f"P75={np.percentile(stale_mfes, 75):+.3f}R  P90={np.percentile(stale_mfes, 90):+.3f}R")

    # How many stale trades reached meaningful MFE
    reached_05 = sum(1 for m in stale_mfes if m >= 0.5)
    reached_10 = sum(1 for m in stale_mfes if m >= 1.0)
    lines.append(f"    Reached >=0.5R MFE: {reached_05}/{len(stale)} ({reached_05/len(stale)*100:.0f}%)")
    lines.append(f"    Reached >=1.0R MFE: {reached_10}/{len(stale)} ({reached_10/len(stale)*100:.0f}%)")

    # R-at-exit distribution
    stale_rs = np.array([t.r_multiple for t in stale])
    winners = sum(1 for r in stale_rs if r > 0)
    losers = sum(1 for r in stale_rs if r <= 0)
    lines.append(f"\n  R-at-exit distribution:")
    lines.append(f"    Winners: {winners}  Losers: {losers}  WR: {winners/len(stale)*100:.0f}%")
    lines.append(f"    Mean={np.mean(stale_rs):+.3f}R  Median={np.median(stale_rs):+.3f}R")

    # Stale hold times
    stale_holds = np.array([t.bars_held_15m for t in stale])
    lines.append(f"\n  Hold time (15m bars):")
    lines.append(f"    Mean={np.mean(stale_holds):.1f}  Median={np.median(stale_holds):.0f}  "
                 f"P25={np.percentile(stale_holds, 25):.0f}  P75={np.percentile(stale_holds, 75):.0f}")

    # Compare stale vs non-stale
    if non_stale:
        ns_avg_r = np.mean([t.r_multiple for t in non_stale])
        ns_wr = sum(1 for t in non_stale if t.r_multiple > 0) / len(non_stale)
        lines.append(f"\n  Stale vs non-stale comparison:")
        lines.append(f"    {'':12} {'Stale':>10} {'Non-stale':>10}")
        lines.append(f"    {'AvgR':12} {np.mean(stale_rs):>+10.3f} {ns_avg_r:>+10.3f}")
        lines.append(f"    {'WinRate':12} {winners/len(stale)*100:>9.0f}% {ns_wr*100:>9.0f}%")
        lines.append(f"    {'AvgMFE':12} {np.mean(stale_mfes):>+10.3f} "
                     f"{np.mean([t.mfe_r for t in non_stale]):>+10.3f}")

    # Direction breakdown for stale
    long_stale = [t for t in stale if t.direction == 1]
    short_stale = [t for t in stale if t.direction == -1]
    if long_stale:
        lr = np.mean([t.r_multiple for t in long_stale])
        lines.append(f"\n  Stale by direction:")
        lines.append(f"    Long:  {len(long_stale)} trades  AvgR={lr:+.3f}")
    if short_stale:
        sr = np.mean([t.r_multiple for t in short_stale])
        lines.append(f"    Short: {len(short_stale)} trades  AvgR={sr:+.3f}")

    # Interpretation
    lines.append(f"\n  ** Interpretation:")
    if pct > 40:
        lines.append(f"     {pct:.0f}% stale rate is very high -- strategy enters many low-conviction trades")
        lines.append(f"     that drift sideways. Consider tighter entry filters or adaptive stale timers.")
    if len(stale) > 0 and reached_05 / len(stale) > 0.3:
        lines.append(f"     {reached_05/len(stale)*100:.0f}% of stale trades reached 0.5R MFE -- partial-take")
        lines.append(f"     or MFE-triggered floor stop could capture some of this dead alpha.")

    return "\n".join(lines)


def _overnight_risk_analysis(trades: list) -> str:
    """Analyse overnight/multi-session hold risk for futures positions."""
    if not trades or not hasattr(trades[0], 'overnight_sessions'):
        return ""

    single = [t for t in trades if t.overnight_sessions <= 1]
    multi = [t for t in trades if t.overnight_sessions > 1]

    if not multi:
        lines = ["=== Overnight Risk Analysis ==="]
        lines.append(f"  No multi-session trades detected ({len(single)} single-session only).")
        return "\n".join(lines)

    lines = ["=== Overnight Risk Analysis ==="]
    lines.append(f"  Single-session: {len(single)} trades  Multi-session: {len(multi)} trades")

    # Performance comparison
    for label, group in [("Single-session", single), ("Multi-session", multi)]:
        if not group:
            continue
        rs = np.array([t.r_multiple for t in group])
        mfes = np.array([t.mfe_r for t in group])
        maes = np.array([t.mae_r for t in group])
        wr = sum(1 for r in rs if r > 0) / len(rs)
        pnl = sum(t.pnl_dollars for t in group)
        avg_r = np.mean(rs)
        lines.append(f"\n  {label} ({len(group)} trades):")
        lines.append(f"    WR={wr:.0%}  AvgR={avg_r:+.3f}  PnL=${pnl:+,.0f}")
        lines.append(f"    AvgMFE={np.mean(mfes):+.3f}  AvgMAE={np.mean(maes):+.3f}")
        lines.append(f"    AvgHold={np.mean([t.bars_held_15m for t in group]):.1f} bars")

    # Session-count distribution for multi
    sess_dist = Counter(t.overnight_sessions for t in multi)
    lines.append(f"\n  Overnight sessions distribution:")
    for sess_n in sorted(sess_dist):
        grp = [t for t in multi if t.overnight_sessions == sess_n]
        avg_r = np.mean([t.r_multiple for t in grp])
        lines.append(f"    {sess_n} sessions: {sess_dist[sess_n]} trades  AvgR={avg_r:+.3f}")

    # Gap risk: MAE of multi-session trades
    multi_maes = np.array([t.mae_r for t in multi])
    lines.append(f"\n  Overnight MAE (gap risk proxy):")
    lines.append(f"    Mean={np.mean(multi_maes):+.3f}R  Max={np.max(multi_maes):+.3f}R")
    heavy_mae = sum(1 for m in multi_maes if m > 0.7)
    if heavy_mae:
        lines.append(f"    Trades with MAE > 0.7R: {heavy_mae}/{len(multi)} "
                     f"({heavy_mae/len(multi)*100:.0f}%)")

    # Sub-window origin of multi-session trades
    sw_counts = Counter(t.sub_window for t in multi)
    lines.append(f"\n  Multi-session entry window:")
    for sw, n in sw_counts.most_common():
        grp = [t for t in multi if t.sub_window == sw]
        avg_r = np.mean([t.r_multiple for t in grp])
        lines.append(f"    {sw:<10} {n:>3} trades  AvgR={avg_r:+.3f}")

    # Interpretation
    multi_avg = np.mean([t.r_multiple for t in multi])
    single_avg = np.mean([t.r_multiple for t in single]) if single else 0
    lines.append(f"\n  ** Interpretation:")
    if multi_avg > single_avg + 0.3:
        lines.append(f"     Multi-session trades outperform by {multi_avg - single_avg:+.3f}R.")
        lines.append(f"     Holding winners overnight is adding significant alpha.")
    elif multi_avg < single_avg - 0.3:
        lines.append(f"     Multi-session trades underperform by {single_avg - multi_avg:+.3f}R.")
        lines.append(f"     Overnight holds are destroying value -- consider EOD flatten or tighter trail.")
    else:
        lines.append(f"     Overnight risk is neutral ({multi_avg - single_avg:+.3f}R delta).")

    return "\n".join(lines)


def _early_kill_audit(trades: list) -> str:
    """Audit EARLY_KILL exits: are they protecting capital or cutting potential winners?"""
    early_kills = [t for t in trades if t.exit_reason == "EARLY_KILL"]
    if not early_kills:
        return ""

    lines = ["=== Early Kill Effectiveness Audit ==="]
    lines.append(f"  Early kills: {len(early_kills)}/{len(trades)} "
                 f"({len(early_kills)/len(trades)*100:.0f}% of trades)")

    # Basic stats
    rs = np.array([t.r_multiple for t in early_kills])
    mfes = np.array([t.mfe_r for t in early_kills])
    maes = np.array([t.mae_r for t in early_kills])
    lines.append(f"\n  Performance:")
    lines.append(f"    AvgR={np.mean(rs):+.3f}  MedianR={np.median(rs):+.3f}")
    lines.append(f"    Total PnL: ${sum(t.pnl_dollars for t in early_kills):+,.0f}")

    # MFE reached before early kill
    lines.append(f"\n  MFE before kill:")
    lines.append(f"    Mean={np.mean(mfes):+.3f}R  Median={np.median(mfes):+.3f}R  "
                 f"Max={np.max(mfes):+.3f}R")
    had_profit = sum(1 for m in mfes if m >= 0.3)
    lines.append(f"    Reached >=0.3R MFE before kill: {had_profit}/{len(early_kills)} "
                 f"({had_profit/len(early_kills)*100:.0f}%)")

    # MAE at kill (how deep in the hole)
    lines.append(f"\n  MAE at kill (drawdown absorbed):")
    lines.append(f"    Mean={np.mean(maes):+.3f}R  Median={np.median(maes):+.3f}R")

    # Sub-window breakdown
    sw_counts = Counter(t.sub_window for t in early_kills)
    lines.append(f"\n  Entry window of killed trades:")
    for sw, n in sw_counts.most_common():
        grp = [t for t in early_kills if t.sub_window == sw]
        avg_r = np.mean([t.r_multiple for t in grp])
        lines.append(f"    {sw:<10} {n:>3} trades  AvgR={avg_r:+.3f}")

    # Hold time distribution (early kills should be very short)
    holds = np.array([t.bars_held_15m for t in early_kills])
    lines.append(f"\n  Hold time (15m bars):")
    lines.append(f"    Mean={np.mean(holds):.1f}  Median={np.median(holds):.0f}  "
                 f"Max={np.max(holds)}")

    # Direction
    long_ek = [t for t in early_kills if t.direction == 1]
    short_ek = [t for t in early_kills if t.direction == -1]
    lines.append(f"\n  Direction:")
    if long_ek:
        lines.append(f"    Long:  {len(long_ek)} kills  AvgR={np.mean([t.r_multiple for t in long_ek]):+.3f}")
    if short_ek:
        lines.append(f"    Short: {len(short_ek)} kills  AvgR={np.mean([t.r_multiple for t in short_ek]):+.3f}")

    # Counterfactual: compare vs stale exit average
    stale_trades = [t for t in trades if t.exit_reason == "STALE"]
    if stale_trades:
        avg_stale_r = np.mean([t.r_multiple for t in stale_trades])
        avg_ek_r = np.mean(rs)
        saved = avg_stale_r - avg_ek_r
        lines.append(f"\n  Counterfactual vs stale exit:")
        lines.append(f"    Avg R of stale exits: {avg_stale_r:+.3f}")
        lines.append(f"    Avg R of early kills: {avg_ek_r:+.3f}")
        lines.append(f"    Early kill {'saves' if avg_ek_r > avg_stale_r else 'costs'} "
                     f"{abs(saved):.3f}R per trade vs letting them go stale")

    # Interpretation
    lines.append(f"\n  ** Interpretation:")
    avg_r_val = np.mean(rs)
    if avg_r_val < -0.3 and had_profit / len(early_kills) < 0.2:
        lines.append(f"     Early kill is working: catching fast-dying trades that never showed profit.")
        lines.append(f"     Only {had_profit/len(early_kills)*100:.0f}% even reached 0.3R MFE.")
    elif had_profit / len(early_kills) > 0.3:
        lines.append(f"     WARNING: {had_profit/len(early_kills)*100:.0f}% of killed trades reached 0.3R MFE.")
        lines.append(f"     Early kill may be too aggressive -- some of these had potential.")
    else:
        lines.append(f"     Early kill is marginally effective (avg R={avg_r_val:+.3f}).")

    return "\n".join(lines)


def _r_per_bar_efficiency(trades: list) -> str:
    """R earned per bar held -- identifies optimal holding duration and
    marginal value of additional hold time."""
    if not trades:
        return ""

    lines = ["=== R-per-Bar Efficiency ==="]

    buckets = [
        ("1-2", 1, 2),
        ("3-4", 3, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33-64", 33, 64),
        ("65+", 65, 9999),
    ]

    lines.append(f"  {'Bars':<8} {'N':>4} {'AvgR':>8} {'R/Bar':>8} {'TotalR':>9} {'CumR':>9}")
    lines.append(f"  {'-'*52}")

    cum_r = 0.0
    bucket_data = []
    for label, lo, hi in buckets:
        grp = [t for t in trades if lo <= t.bars_held_15m <= hi]
        if not grp:
            continue
        avg_r = np.mean([t.r_multiple for t in grp])
        total_r = sum(t.r_multiple for t in grp)
        avg_bars = np.mean([t.bars_held_15m for t in grp])
        r_per_bar = avg_r / avg_bars if avg_bars > 0 else 0
        cum_r += total_r
        lines.append(f"  {label:<8} {len(grp):>4} {avg_r:>+8.3f} {r_per_bar:>+8.4f} "
                     f"{total_r:>+9.1f} {cum_r:>+9.1f}")
        bucket_data.append((label, len(grp), avg_r, r_per_bar, total_r))

    # Best R/bar bucket
    if bucket_data:
        best = max(bucket_data, key=lambda x: x[3])
        worst = min(bucket_data, key=lambda x: x[3])
        lines.append(f"\n  Best R/bar:  {best[0]} ({best[3]:+.4f} R/bar)")
        lines.append(f"  Worst R/bar: {worst[0]} ({worst[3]:+.4f} R/bar)")

    # Marginal value: what does each additional hold bar add?
    lines.append(f"\n  Marginal hold analysis:")
    short_holds = [t for t in trades if t.bars_held_15m <= 4]
    medium_holds = [t for t in trades if 5 <= t.bars_held_15m <= 16]
    long_holds = [t for t in trades if t.bars_held_15m > 16]

    for label, grp in [("Short (1-4)", short_holds), ("Medium (5-16)", medium_holds),
                       ("Long (17+)", long_holds)]:
        if not grp:
            continue
        avg_r = np.mean([t.r_multiple for t in grp])
        total_pnl = sum(t.pnl_dollars for t in grp)
        wr = sum(1 for t in grp if t.r_multiple > 0) / len(grp)
        lines.append(f"    {label:<16} {len(grp):>3} trades  WR={wr:.0%}  AvgR={avg_r:+.3f}  "
                     f"PnL=${total_pnl:+,.0f}")

    # Total R from short holds (opportunity cost)
    if short_holds:
        short_total_r = sum(t.r_multiple for t in short_holds)
        short_total_pnl = sum(t.pnl_dollars for t in short_holds)
        lines.append(f"\n  Short-hold drag:")
        lines.append(f"    1-4 bar trades contribute {short_total_r:+.1f}R (${short_total_pnl:+,.0f})")
        if short_total_r < 0:
            lines.append(f"    ** Eliminating these would improve net R by {abs(short_total_r):.1f}R")

    # Interpretation
    lines.append(f"\n  ** Interpretation:")
    if short_holds and np.mean([t.r_multiple for t in short_holds]) < -0.3:
        lines.append(f"     Short holds (1-4 bars) are a significant drag.")
        lines.append(f"     Either the entry signal is wrong or the position needs more time to develop.")
        lines.append(f"     Consider minimum hold time or entry filter tightening.")
    if long_holds and np.mean([t.r_multiple for t in long_holds]) > 1.0:
        lines.append(f"     Long holds (17+ bars) are where the alpha lives.")
        lines.append(f"     Strategy's edge comes from holding winners -- protect this at all costs.")

    return "\n".join(lines)


def _class_mult_calibration(trades: list) -> str:
    """Evaluate class_mult (quality/sizing multiplier) calibration -- do higher-class
    trades actually outperform?"""
    if not trades or not hasattr(trades[0], 'class_mult'):
        return ""

    # Group by class_mult value
    mult_arr = np.array([t.class_mult for t in trades])
    mults = sorted(set(mult_arr))
    if len(mults) <= 1:
        return ""  # no variation to analyse

    lines = ["=== Class Multiplier Calibration ==="]
    lines.append(f"  Distinct class_mult values: {len(mults)}")
    lines.append(f"  Range: {min(mults):.2f} to {max(mults):.2f}")

    # Performance by class_mult bucket
    if len(mults) <= 5:
        bucket_labels = [(f"{m:.2f}", m, m) for m in mults]
    else:
        p25, p50, p75 = np.percentile(mult_arr, [25, 50, 75])
        bucket_labels = [
            (f"Low (<={p25:.2f})", 0, p25),
            (f"Mid ({p25:.2f}-{p75:.2f})", p25, p75),
            (f"High (>{p75:.2f})", p75, 999),
        ]

    lines.append(f"\n  {'ClassMult':<22} {'N':>4} {'WR':>6} {'AvgR':>8} {'AvgMFE':>8} {'PnL':>10}")
    lines.append(f"  {'-'*62}")

    for label, lo, hi in bucket_labels:
        if lo == hi:
            grp = [t for t in trades if t.class_mult == lo]
        else:
            grp = [t for t in trades if lo < t.class_mult <= hi]
            if lo == 0:
                grp = [t for t in trades if t.class_mult <= hi]
        if not grp:
            continue
        wr = sum(1 for t in grp if t.r_multiple > 0) / len(grp)
        avg_r = np.mean([t.r_multiple for t in grp])
        avg_mfe = np.mean([t.mfe_r for t in grp])
        pnl = sum(t.pnl_dollars for t in grp)
        lines.append(f"  {label:<22} {len(grp):>4} {wr:>5.0%} {avg_r:>+8.3f} "
                     f"{avg_mfe:>+8.3f} ${pnl:>+9,.0f}")

    # Correlation between class_mult and R
    r_arr = np.array([t.r_multiple for t in trades])
    if len(mult_arr) > 10 and np.std(mult_arr) > 0:
        corr = np.corrcoef(mult_arr, r_arr)[0, 1]
        lines.append(f"\n  Correlation (class_mult vs R): {corr:+.3f}")
        if abs(corr) < 0.05:
            lines.append(f"    ** Near-zero correlation -- class_mult is NOT predictive of trade quality")
        elif corr > 0.1:
            lines.append(f"    ** Positive correlation -- higher class trades do outperform")
        elif corr < -0.1:
            lines.append(f"    ** NEGATIVE correlation -- higher class trades underperform (!)")

    # Interpretation
    lines.append(f"\n  ** Interpretation:")
    high_grp = [t for t in trades if t.class_mult >= np.percentile(mult_arr, 75)]
    low_grp = [t for t in trades if t.class_mult <= np.percentile(mult_arr, 25)]
    if high_grp and low_grp:
        high_r = np.mean([t.r_multiple for t in high_grp])
        low_r = np.mean([t.r_multiple for t in low_grp])
        if high_r > low_r + 0.1:
            lines.append(f"     Class multiplier is well-calibrated: high-class trades outperform by "
                         f"{high_r - low_r:+.3f}R.")
        elif low_r > high_r + 0.1:
            lines.append(f"     Class multiplier is MIS-calibrated: low-class trades actually outperform by "
                         f"{low_r - high_r:+.3f}R.")
            lines.append(f"     Consider inverting or removing the class_mult sizing adjustment.")
        else:
            lines.append(f"     Class multiplier has minimal impact on outcomes ({high_r - low_r:+.3f}R delta).")

    return "\n".join(lines)
