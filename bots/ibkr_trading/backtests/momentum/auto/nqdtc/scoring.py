"""NQDTC composite scoring -- 7 immutable components for NQDTC search.

Components (BASE_WEIGHTS):
  returns       (22%): raw + largest-winner-robust log-scaled net return
  pf            (12%): profit factor quality, 1.0 at PF=2.1
  expectancy    (14%): average R per trade, 1.0 at +0.55R
  frequency     (18%): trade count throughput, 1.0 at 155 trades
  risk          (10%): drawdown + Calmar blend
  exit_capture  (16%): winner MFE capture + TP1/TP2 conversion blend
  stability     ( 8%): Sharpe + Sortino blend

Hard rejects (configurable per-phase): min_trades, max_dd_pct, min_pf,
min_avg_r, min_capture, min_net_return_pct, min_robust_net_return_pct,
max_largest_win_pnl_share.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _span_days(timestamps) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    delta = timestamps[-1] - timestamps[0]
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds()) / 86400.0
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s")) / 86400.0
    if isinstance(delta, (int, float, np.integer, np.floating)):
        return float(delta) / 86400.0
    return 0.0


@dataclass
class NQDTCMetrics:
    """NQDTC-specific performance metrics for scoring and diagnostics."""

    # Core performance
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    net_return_pct: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    avg_r: float = 0.0

    # NQDTC-specific (from trade records)
    capture_ratio: float = 0.0       # mean(R/MFE) for winners with MFE > 0
    burst_trade_pct: float = 0.0     # fraction of trades in bursts (3+ within 4h)
    eth_short_wr: float = 0.0        # ETH shorts win rate
    eth_short_trades: int = 0        # ETH shorts count
    range_regime_pct: float = 0.0    # fraction of trades in Range regime
    tp1_hit_rate: float = 0.0        # TP1 fill rate
    tp2_hit_rate: float = 0.0        # TP2 fill rate
    avg_hold_hours: float = 0.0      # average hold duration in hours
    avg_winner_r: float = 0.0        # average R for winners
    avg_loser_r: float = 0.0         # average R for losers
    avg_mfe_r: float = 0.0           # average MFE in R for all trades
    robust_net_return_pct: float = 0.0       # net return excluding largest net winner
    largest_win_return_pct: float = 0.0      # largest net winner as pct of initial equity
    largest_win_pnl_share: float = 0.0       # largest net winner / total net profit
    largest_winner_r: float = 0.0            # largest winning trade in R


@dataclass(frozen=True)
class NQDTCCompositeScore:
    """Frozen 7-component composite score for NQDTC."""

    returns: float = 0.0
    pf: float = 0.0
    expectancy: float = 0.0
    frequency: float = 0.0
    risk: float = 0.0
    exit_capture: float = 0.0
    stability: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


BASE_WEIGHTS = {
    "returns": 0.22,
    "pf": 0.12,
    "expectancy": 0.14,
    "frequency": 0.18,
    "risk": 0.10,
    "exit_capture": 0.16,
    "stability": 0.08,
}


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(
    metrics: NQDTCMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> NQDTCCompositeScore:
    """Compute the immutable 7-component composite score for NQDTC."""
    w = dict(BASE_WEIGHTS)
    if weight_overrides:
        w.update(weight_overrides)

    # Configurable hard rejects
    hr = hard_rejects or {"min_trades": 80, "max_dd_pct": 0.35, "min_pf": 1.00}
    if metrics.total_trades < hr.get("min_trades", 15):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"too_few_trades ({metrics.total_trades})",
        )
    if metrics.max_dd_pct > hr.get("max_dd_pct", 0.35):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"max_dd_exceeded ({metrics.max_dd_pct:.2%})",
        )
    if metrics.profit_factor < hr.get("min_pf", 0.80):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"low_pf ({metrics.profit_factor:.2f})",
        )
    if metrics.avg_r < hr.get("min_avg_r", -999.0):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"low_avg_r ({metrics.avg_r:.3f})",
        )
    if metrics.capture_ratio < hr.get("min_capture", -999.0):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"low_capture ({metrics.capture_ratio:.3f})",
        )
    if metrics.net_return_pct < hr.get("min_net_return_pct", -999.0):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"low_return ({metrics.net_return_pct:.1f}%)",
        )
    if metrics.robust_net_return_pct < hr.get("min_robust_net_return_pct", -999.0):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"low_robust_return ({metrics.robust_net_return_pct:.1f}%)",
        )
    if metrics.largest_win_pnl_share > hr.get("max_largest_win_pnl_share", 999.0):
        return NQDTCCompositeScore(
            rejected=True, reject_reason=f"outlier_concentration ({metrics.largest_win_pnl_share:.1%})",
        )

    # --- Components (round-3 calibrated scales) ---

    # Returns: blend raw return with largest-winner-robust return so a single
    # outsized trade cannot dominate the expected-return objective.
    raw_return_c = _clip01(math.log(1 + max(metrics.net_return_pct, 0) / 100) / math.log(4.0))
    robust_return_c = _clip01(math.log(1 + max(metrics.robust_net_return_pct, 0) / 100) / math.log(3.2))
    returns_c = (0.60 * raw_return_c) + (0.40 * robust_return_c)

    # Profit factor: 0 at PF=1.20, 1.0 at PF=2.10.
    pf_c = _clip01((metrics.profit_factor - 1.20) / 0.90)

    # Expectancy: 0 at +0.10R, 1.0 at +0.55R.
    expectancy_c = _clip01((metrics.avg_r - 0.10) / 0.45)

    # Frequency: baseline should stay in play while genuine extra fills matter.
    frequency_c = _clip01((metrics.total_trades - 75.0) / 80.0)

    # Risk: combine drawdown containment and return-per-drawdown efficiency.
    dd_c = _clip01(1 - metrics.max_dd_pct / 0.28)
    calmar_c = _clip01(metrics.calmar / 9.0)
    risk_c = (0.50 * dd_c) + (0.50 * calmar_c)

    # Exit capture: the structural weakness is leaving winner MFE on the table.
    capture_c = _clip01((metrics.capture_ratio - 0.28) / 0.34)
    tp1_c = _clip01(metrics.tp1_hit_rate / 0.65)
    tp2_c = _clip01(metrics.tp2_hit_rate / 0.18)
    exit_capture_c = (0.60 * capture_c) + (0.25 * tp1_c) + (0.15 * tp2_c)

    # Stability: keep a smaller reward for path quality after hard risk gates.
    sharpe_c = _clip01(metrics.sharpe / 2.0)
    sortino_c = _clip01(metrics.sortino / 6.0)
    stability_c = (0.50 * sharpe_c) + (0.50 * sortino_c)

    total = (
        w["returns"] * returns_c
        + w["pf"] * pf_c
        + w["expectancy"] * expectancy_c
        + w["frequency"] * frequency_c
        + w["risk"] * risk_c
        + w["exit_capture"] * exit_capture_c
        + w["stability"] * stability_c
    )

    return NQDTCCompositeScore(
        returns=returns_c,
        pf=pf_c,
        expectancy=expectancy_c,
        frequency=frequency_c,
        risk=risk_c,
        exit_capture=exit_capture_c,
        stability=stability_c,
        total=total,
    )


def extract_nqdtc_metrics(
    trades: list,
    equity_curve: list[float],
    timestamps: list,
    initial_equity: float,
) -> NQDTCMetrics:
    """Extract NQDTCMetrics from trade records and equity curve."""
    if not trades:
        return NQDTCMetrics()

    total = len(trades)
    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    win_count = len(winners)
    win_rate = win_count / total if total > 0 else 0.0

    net_pnls = [float(t.pnl_dollars) - float(getattr(t, "commission", 0.0) or 0.0) for t in trades]
    gross_profit = sum(pnl for pnl in net_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in net_pnls if pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    avg_r = sum(t.r_multiple for t in trades) / total if total > 0 else 0.0
    avg_winner_r = sum(t.r_multiple for t in winners) / win_count if win_count else 0.0
    avg_loser_r = sum(t.r_multiple for t in losers) / len(losers) if losers else 0.0
    avg_mfe_r = sum(t.mfe_r for t in trades) / total if total > 0 else 0.0
    largest_winner_r = max((t.r_multiple for t in winners), default=0.0)

    # Equity curve stats
    eq = equity_curve if len(equity_curve) > 0 else [initial_equity]
    peak = initial_equity
    max_dd = 0.0
    for e in eq:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    final_eq = eq[-1] if len(eq) > 0 else initial_equity
    net_return_pct = ((final_eq - initial_equity) / initial_equity) * 100.0
    net_profit_dollars = final_eq - initial_equity
    largest_win_pnl = max((pnl for pnl in net_pnls if pnl > 0), default=0.0)
    largest_win_return_pct = (largest_win_pnl / initial_equity) * 100.0 if initial_equity > 0 else 0.0
    robust_net_return_pct = net_return_pct - largest_win_return_pct
    largest_win_pnl_share = (
        largest_win_pnl / net_profit_dollars
        if net_profit_dollars > 0 and largest_win_pnl > 0
        else 0.0
    )

    calmar = (net_return_pct / 100.0) / max_dd if max_dd > 0 else 99.0

    # Sharpe/Sortino from trade R-multiples
    r_vals = [t.r_multiple for t in trades]
    mean_r = avg_r
    if len(r_vals) >= 2:
        std_r = float(np.std(r_vals, ddof=1))
        # Annualized: mean_r * trades_per_year / (std_r * sqrt(trades_per_year))
        if timestamps is not None and len(timestamps) >= 2:
            span_days = _span_days(timestamps)
            trades_per_year = total / (span_days / 365.25) if span_days > 0 else total
        else:
            trades_per_year = total
        sharpe = (mean_r * trades_per_year) / (std_r * trades_per_year ** 0.5) if std_r > 0 else 0.0
        downside = [r for r in r_vals if r < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else std_r
        sortino = (mean_r * trades_per_year) / (downside_std * trades_per_year ** 0.5) if downside_std > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    # Capture ratio: mean(R/MFE) for winners with MFE > 0
    capture_vals = [t.r_multiple / t.mfe_r for t in winners if t.mfe_r > 0.01]
    capture_ratio = sum(capture_vals) / len(capture_vals) if capture_vals else 0.0

    # Burst detection: 3+ trades within 4h -- O(n log n) via bisect
    entry_times = sorted(t.entry_time for t in trades)
    burst_count = 0
    if entry_times:
        from bisect import bisect_right
        from datetime import timedelta
        _4h = timedelta(seconds=14400)
        for i, t_i in enumerate(entry_times):
            if bisect_right(entry_times, t_i + _4h) - i >= 3:
                burst_count += 1
    burst_trade_pct = burst_count / total if total > 0 else 0.0

    # ETH short stats
    eth_shorts = [t for t in trades if t.session == "ETH" and t.direction == -1]
    eth_short_trades = len(eth_shorts)
    eth_short_wins = sum(1 for t in eth_shorts if t.r_multiple > 0)
    eth_short_wr = eth_short_wins / eth_short_trades if eth_short_trades > 0 else 0.0

    # Range regime fraction
    range_trades = sum(1 for t in trades if t.composite_regime == "Range")
    range_regime_pct = range_trades / total if total > 0 else 0.0

    # TP hit rates (use NQDTCTradeRecord bool fields, not exit_reason string)
    tp1_hits = sum(1 for t in trades if getattr(t, 'tp1_hit', False))
    tp2_hits = sum(1 for t in trades if getattr(t, 'tp2_hit', False))
    tp1_hit_rate = tp1_hits / total if total > 0 else 0.0
    tp2_hit_rate = tp2_hits / total if total > 0 else 0.0

    # Average hold hours
    hold_hours = []
    for t in trades:
        if hasattr(t, 'entry_time') and hasattr(t, 'exit_time') and t.exit_time:
            dt = (t.exit_time - t.entry_time).total_seconds() / 3600.0
            hold_hours.append(dt)
    avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

    return NQDTCMetrics(
        total_trades=total,
        win_rate=win_rate,
        profit_factor=pf,
        max_dd_pct=max_dd,
        net_return_pct=net_return_pct,
        calmar=calmar,
        sharpe=sharpe,
        sortino=sortino,
        avg_r=avg_r,
        capture_ratio=capture_ratio,
        burst_trade_pct=burst_trade_pct,
        eth_short_wr=eth_short_wr,
        eth_short_trades=eth_short_trades,
        range_regime_pct=range_regime_pct,
        tp1_hit_rate=tp1_hit_rate,
        tp2_hit_rate=tp2_hit_rate,
        avg_hold_hours=avg_hold_hours,
        avg_winner_r=avg_winner_r,
        avg_loser_r=avg_loser_r,
        avg_mfe_r=avg_mfe_r,
        robust_net_return_pct=robust_net_return_pct,
        largest_win_return_pct=largest_win_return_pct,
        largest_win_pnl_share=largest_win_pnl_share,
        largest_winner_r=largest_winner_r,
    )
