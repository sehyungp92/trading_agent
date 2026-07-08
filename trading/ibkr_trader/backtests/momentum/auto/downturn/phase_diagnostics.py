"""Downturn 8 diagnostic modules (D1-D8).

D1: Regime accuracy during correction windows (always)
D2: Signal quality per-engine (always)
D3: Exit efficiency per trade (Phase 2+)
D4: Drawdown attribution (Phase 2+)
D5: Hold time & conviction (Phase 2+)
D6: Phase delta (always)
D7: Correction-window detail (Phase 3+)
D8: Per-engine exit & MFE analysis (Phase 2+)
"""
from __future__ import annotations

import logging
from io import StringIO

from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics
from backtests.momentum.auto.downturn.phase_gates import check_phase_gate
from strategies.momentum.downturn.bt_models import (
    CompositeRegime,
    DownturnTradeRecord,
    EngineTag,
)

logger = logging.getLogger(__name__)


def generate_phase_diagnostics(
    phase: int,
    metrics: DownturnMetrics,
    greedy_result: dict | None,
    state_dict: dict | None,
    all_trades: list[DownturnTradeRecord] | None = None,
    force_all_modules: bool = False,
) -> str:
    """Generate full diagnostic report for a downturn phase."""
    buf = StringIO()
    buf.write(f"{'=' * 70}\n")
    buf.write(f"DOWNTURN DOMINATOR PHASE {phase} DIAGNOSTICS\n")
    buf.write(f"{'=' * 70}\n\n")

    _write_d1_regime_accuracy(buf, metrics, all_trades)
    _write_d2_signal_quality(buf, metrics, all_trades)

    if phase >= 2 or force_all_modules:
        _write_d3_exit_efficiency(buf, all_trades)

    if phase >= 2 or force_all_modules:
        _write_d4_drawdown_attribution(buf, all_trades, metrics)

    if phase >= 2 or force_all_modules:
        _write_d5_hold_conviction(buf, all_trades, metrics)

    _write_d6_phase_delta(buf, phase, metrics, state_dict)

    if phase >= 3 or force_all_modules:
        _write_d7_correction_detail(buf, all_trades, metrics)

    if phase >= 2 or force_all_modules:
        _write_d8_engine_exit_mfe(buf, all_trades)

    _write_gate_assessment(buf, phase, metrics, greedy_result)

    return buf.getvalue()


def get_diagnostic_gaps(phase: int, metrics: DownturnMetrics) -> list[str]:
    """Identify weakness areas without diagnostic coverage."""
    gaps = []

    # Check if exit analysis would help but isn't enabled yet
    if phase < 2 and metrics.exit_efficiency < 0.20:
        gaps.append("Exit efficiency is low but D3 not enabled until Phase 2")

    # Check if correction detail would help
    if phase < 3 and metrics.correction_pnl_pct < 5.0:
        gaps.append("Correction-window PnL is low but D7 is not enabled until Phase 3")

    # Engine-specific gaps
    if metrics.reversal_trades > 0 and metrics.reversal_avg_r < -0.5:
        gaps.append("Reversal engine has negative avg R — needs per-engine exit analysis")
    if metrics.breakdown_trades > 0 and metrics.breakdown_avg_r < -0.5:
        gaps.append("Breakdown engine has negative avg R — needs per-engine exit analysis")
    if metrics.fade_trades > 0 and metrics.fade_avg_r < -0.5:
        gaps.append("Fade engine has negative avg R — needs per-engine exit analysis")

    # Trade frequency gaps
    if metrics.total_trades < 40:
        gaps.append("Too few trades for statistical significance — need signal gate analysis")
    if metrics.reversal_trades == 0:
        gaps.append("Reversal engine dead (0 trades) — pivot detection or gate configuration issue")

    # TP hit rate gaps
    tp = metrics.tp_hit_rates or {}
    if tp.get("tp2", 0) == 0 and tp.get("tp3", 0) == 0 and metrics.total_trades > 10:
        gaps.append("TP2/TP3 never hit — exit schedule may be unreachable")

    # Hold time gaps (D5 now available at Phase 2)
    if phase < 2 and metrics.exit_efficiency < 0.15:
        gaps.append("Poor exit efficiency but hold time analysis (D5) not enabled until Phase 2")

    return gaps


# ---------------------------------------------------------------------------
# D1: Regime accuracy
# ---------------------------------------------------------------------------

def _write_d1_regime_accuracy(
    buf: StringIO, metrics: DownturnMetrics,
    trades: list[DownturnTradeRecord] | None,
) -> None:
    buf.write("--- D1: Regime Accuracy ---\n")
    if not trades:
        buf.write("  No trades for analysis.\n\n")
        return

    # Count trades by regime
    regime_counts: dict[str, int] = {}
    regime_pnl: dict[str, float] = {}
    for t in trades:
        r = t.composite_regime_at_entry.value
        regime_counts[r] = regime_counts.get(r, 0) + 1
        regime_pnl[r] = regime_pnl.get(r, 0.0) + t.pnl

    for regime in sorted(regime_counts.keys()):
        count = regime_counts[regime]
        pnl = regime_pnl[regime]
        avg_pnl = pnl / count if count > 0 else 0
        buf.write(f"  {regime:20s}  trades={count:4d}  PnL=${pnl:>10,.0f}  avg=${avg_pnl:>8,.0f}\n")

    # Correction window accuracy
    corr_trades = [t for t in trades if t.in_correction_window]
    non_corr = [t for t in trades if not t.in_correction_window]
    buf.write(f"\n  Correction trades:     {len(corr_trades)}")
    if corr_trades:
        corr_pnl = sum(t.pnl for t in corr_trades)
        buf.write(f"  PnL=${corr_pnl:>10,.0f}")
    buf.write(f"\n  Non-correction trades: {len(non_corr)}")
    if non_corr:
        nc_pnl = sum(t.pnl for t in non_corr)
        buf.write(f"  PnL=${nc_pnl:>10,.0f}")
    buf.write("\n\n")


# ---------------------------------------------------------------------------
# D2: Signal quality
# ---------------------------------------------------------------------------

def _write_d2_signal_quality(
    buf: StringIO, metrics: DownturnMetrics,
    trades: list[DownturnTradeRecord] | None,
) -> None:
    buf.write("--- D2: Signal Quality ---\n")
    buf.write(f"  Signal→entry ratio:  {metrics.signal_to_entry_ratio:.3f}\n")

    for tag, name, n_trades, wr, avg_r in [
        (EngineTag.REVERSAL, "Reversal", metrics.reversal_trades, metrics.reversal_wr, metrics.reversal_avg_r),
        (EngineTag.BREAKDOWN, "Breakdown", metrics.breakdown_trades, metrics.breakdown_wr, metrics.breakdown_avg_r),
        (EngineTag.FADE, "Fade", metrics.fade_trades, metrics.fade_wr, metrics.fade_avg_r),
    ]:
        buf.write(f"  {name:12s}  n={n_trades:4d}  WR={wr:.1%}  avgR={avg_r:+.3f}\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D3: Exit efficiency
# ---------------------------------------------------------------------------

def _write_d3_exit_efficiency(buf: StringIO, trades: list[DownturnTradeRecord] | None) -> None:
    import numpy as np

    buf.write("--- D3: Exit Efficiency ---\n")
    if not trades:
        buf.write("  No trades.\n\n")
        return

    # Global by exit type
    exit_counts: dict[str, list[float]] = {}
    for t in trades:
        exit_counts.setdefault(t.exit_type, []).append(t.r_multiple)

    for exit_type in sorted(exit_counts.keys()):
        rs = exit_counts[exit_type]
        avg_r = float(np.mean(rs))
        buf.write(f"  {exit_type:20s}  n={len(rs):4d}  avgR={avg_r:+.3f}\n")

    # Per-engine exit breakdown
    buf.write("\n  Per-engine exit breakdown:\n")
    for tag in [EngineTag.REVERSAL, EngineTag.BREAKDOWN, EngineTag.FADE]:
        eng = [t for t in trades if t.engine_tag == tag]
        if not eng:
            continue
        buf.write(f"    {tag.value}:\n")
        eng_exits: dict[str, list[float]] = {}
        for t in eng:
            eng_exits.setdefault(t.exit_type, []).append(t.r_multiple)
        for exit_type in sorted(eng_exits.keys()):
            rs = eng_exits[exit_type]
            avg_r = float(np.mean(rs))
            buf.write(f"      {exit_type:18s}  n={len(rs):3d}  avgR={avg_r:+.3f}\n")

    # MFE capture: positive realized R divided by available favorable excursion.
    total_mfe = sum(t.mfe for t in trades if t.mfe > 0)
    captured_r = sum(max(0.0, t.r_multiple) for t in trades)
    if total_mfe > 0:
        low_mfe = [t for t in trades if t.mfe < 0.5]
        low_mfe_pnl = sum(t.pnl for t in low_mfe)
        buf.write(f"\n  Positive MFE capture: {captured_r / total_mfe:.2f}\n")
        buf.write(f"  Low-MFE trades (<0.5R): {len(low_mfe)}/{len(trades)} ({len(low_mfe) / len(trades):.0%})")
        buf.write(f"  PnL=${low_mfe_pnl:+,.0f}\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D4: Drawdown attribution
# ---------------------------------------------------------------------------

def _write_d4_drawdown_attribution(
    buf: StringIO, trades: list[DownturnTradeRecord] | None,
    metrics: DownturnMetrics,
) -> None:
    buf.write("--- D4: Drawdown Attribution ---\n")
    buf.write(f"  Max DD: {metrics.max_dd_pct:.2%}\n")

    if not trades:
        buf.write("  No trades.\n\n")
        return

    # Worst 5 trades
    sorted_trades = sorted(trades, key=lambda t: t.pnl)[:5]
    buf.write("  Worst 5 trades:\n")
    for i, t in enumerate(sorted_trades):
        buf.write(
            f"    {i+1}. {t.engine_tag.value:10s} PnL=${t.pnl:>8,.0f}"
            f"  R={t.r_multiple:+.2f}  {t.exit_type}\n"
        )

    # Attribution by engine
    for tag in [EngineTag.REVERSAL, EngineTag.BREAKDOWN, EngineTag.FADE]:
        eng = [t for t in trades if t.engine_tag == tag]
        losses = [t for t in eng if t.pnl < 0]
        total_loss = sum(t.pnl for t in losses)
        buf.write(f"  {tag.value:12s} total loss: ${total_loss:>10,.0f} ({len(losses)} losing trades)\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D5: Hold time & conviction
# ---------------------------------------------------------------------------

def _write_d5_hold_conviction(
    buf: StringIO, trades: list[DownturnTradeRecord] | None,
    metrics: DownturnMetrics,
) -> None:
    buf.write("--- D5: Hold Time & Conviction ---\n")
    if not trades:
        buf.write("  No trades.\n\n")
        return

    import numpy as np
    for tag in [EngineTag.REVERSAL, EngineTag.BREAKDOWN, EngineTag.FADE]:
        eng = [t for t in trades if t.engine_tag == tag]
        if not eng:
            continue
        holds = [t.hold_bars for t in eng]
        rs = [t.r_multiple for t in eng]
        buf.write(
            f"  {tag.value:12s}  avgHold={float(np.mean(holds)):5.1f}  "
            f"medR={float(np.median(rs)):+.2f}  "
            f"maxR={max(rs):+.2f}  minR={min(rs):+.2f}\n"
        )
    buf.write("\n")


# ---------------------------------------------------------------------------
# D6: Phase delta
# ---------------------------------------------------------------------------

def _write_d6_phase_delta(
    buf: StringIO, phase: int, metrics: DownturnMetrics,
    state_dict: dict | None,
) -> None:
    buf.write("--- D6: Phase Delta ---\n")
    if not state_dict:
        buf.write("  No prior phase data for comparison.\n\n")
        return

    # Compare with previous phase
    prev_phase = phase - 1
    prev_result = state_dict.get("phase_results", {}).get(str(prev_phase))
    if not prev_result:
        buf.write("  No prior phase for delta.\n\n")
        return

    prev_metrics = prev_result.get("final_metrics", {})
    for key in ["total_trades", "profit_factor", "max_dd_pct", "calmar",
                "sharpe", "net_return_pct", "correction_pnl_pct",
                "exit_efficiency", "signal_to_entry_ratio"]:
        prev = prev_metrics.get(key, prev_metrics.get("correction_alpha_pct", 0) if key == "correction_pnl_pct" else 0)
        curr = getattr(metrics, key, 0)
        delta = curr - prev
        arrow = "+" if delta >= 0 else ""
        buf.write(f"  {key:25s}  {prev:>10.3f} → {curr:>10.3f}  ({arrow}{delta:.3f})\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D7: Correction-window detail
# ---------------------------------------------------------------------------

def _write_d7_correction_detail(
    buf: StringIO, trades: list[DownturnTradeRecord] | None,
    metrics: DownturnMetrics,
) -> None:
    buf.write("--- D7: Correction-Window Detail ---\n")
    buf.write(f"  Correction PnL: {metrics.correction_pnl_pct:.2f}%\n")
    buf.write(f"  Bear capture ratio: {metrics.bear_capture_ratio:.1%}\n")

    if not trades:
        buf.write("  No trades.\n\n")
        return

    corr_trades = [t for t in trades if t.in_correction_window]
    if not corr_trades:
        buf.write("  No correction-window trades.\n\n")
        return

    # Per-engine breakdown during corrections
    for tag in [EngineTag.REVERSAL, EngineTag.BREAKDOWN, EngineTag.FADE]:
        eng = [t for t in corr_trades if t.engine_tag == tag]
        if eng:
            pnl = sum(t.pnl for t in eng)
            wr = sum(1 for t in eng if t.pnl > 0) / len(eng)
            buf.write(f"  {tag.value:12s}  n={len(eng):3d}  PnL=${pnl:>8,.0f}  WR={wr:.1%}\n")
    buf.write("\n")


# ---------------------------------------------------------------------------
# D8: Per-engine exit & MFE analysis
# ---------------------------------------------------------------------------

def _write_d8_engine_exit_mfe(
    buf: StringIO, trades: list[DownturnTradeRecord] | None,
) -> None:
    """Per-engine performance dashboard + MFE distribution to assess TP reachability."""
    import numpy as np

    buf.write("--- D8: Per-Engine Exit & MFE Analysis ---\n")
    if not trades:
        buf.write("  No trades.\n\n")
        return

    for tag in [EngineTag.REVERSAL, EngineTag.BREAKDOWN, EngineTag.FADE]:
        eng = [t for t in trades if t.engine_tag == tag]
        if not eng:
            continue

        wins = [t for t in eng if t.pnl > 0]
        losses = [t for t in eng if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in eng)
        rs = [t.r_multiple for t in eng]
        holds = [t.hold_bars for t in eng]
        mfes = [t.mfe for t in eng if t.mfe > 0]

        buf.write(f"\n  {tag.value.upper()} ({len(eng)} trades):\n")
        buf.write(f"    WR={len(wins)}/{len(eng)} ({len(wins)/len(eng):.0%})  "
                  f"PnL=${total_pnl:>+,.0f}  avgR={float(np.mean(rs)):+.2f}  "
                  f"medR={float(np.median(rs)):+.2f}\n")
        buf.write(f"    Hold: avg={float(np.mean(holds)):.0f} bars  "
                  f"med={float(np.median(holds)):.0f}  "
                  f"max={max(holds)}\n")

        # MFE distribution — key for TP reachability
        if mfes:
            mfe_arr = np.array(mfes)
            pcts = [25, 50, 75, 90, 95]
            quantiles = np.percentile(mfe_arr, pcts)
            buf.write(f"    MFE distribution (R):  ")
            buf.write("  ".join(f"p{p}={q:.2f}" for p, q in zip(pcts, quantiles)))
            buf.write(f"\n    MFE: avg={float(np.mean(mfe_arr)):.2f}  "
                      f"max={float(np.max(mfe_arr)):.2f}\n")

            # TP reachability: what % of trades reached various R levels
            for r_level in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                reached = sum(1 for m in mfes if m >= r_level)
                pct = reached / len(mfes) * 100
                if pct > 0:
                    buf.write(f"    Reached {r_level:>4.1f}R: {reached:3d}/{len(mfes)} ({pct:.0f}%)\n")
        else:
            buf.write(f"    MFE: no data\n")

        # MAE distribution — shows risk characteristics
        maes = [t.mae for t in eng if t.mae < 0]
        if maes:
            mae_arr = np.array(maes)
            buf.write(f"    MAE: avg={float(np.mean(mae_arr)):.2f}  "
                      f"worst={float(np.min(mae_arr)):.2f}\n")

        # Positive MFE capture per engine
        eng_total_mfe = sum(t.mfe for t in eng if t.mfe > 0)
        eng_captured_r = sum(max(0.0, t.r_multiple) for t in eng)
        if eng_total_mfe > 0:
            low_mfe = [t for t in eng if t.mfe < 0.5]
            buf.write(f"    Positive MFE capture: {eng_captured_r / eng_total_mfe:.2f}\n")
            buf.write(f"    Low-MFE trades (<0.5R): {len(low_mfe)}/{len(eng)} ({len(low_mfe) / len(eng):.0%})\n")

    buf.write("\n")


# ---------------------------------------------------------------------------
# Gate assessment
# ---------------------------------------------------------------------------

def _write_gate_assessment(
    buf: StringIO, phase: int, metrics: DownturnMetrics,
    greedy_result: dict | None,
) -> None:
    buf.write("--- Gate Assessment ---\n")
    gate = check_phase_gate(phase, metrics, greedy_result)
    buf.write(f"  Gate passed: {gate.passed}\n")
    for c in gate.criteria:
        status = "PASS" if c.passed else "FAIL"
        buf.write(f"  [{status}] {c.name}: {c.actual:.3f} (target: {c.target:.3f})\n")
    if gate.failure_category:
        buf.write(f"  Failure category: {gate.failure_category}\n")
    for r in gate.recommendations:
        buf.write(f"  → {r}\n")
    buf.write("\n")
