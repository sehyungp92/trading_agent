"""Downturn Dominator diagnostics — metrics computation + report generation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from io import StringIO

import numpy as np

from backtests.momentum.data.preprocessing import NumpyBars
from strategies.momentum.downturn.bt_models import (
    CorrectionWindow,
    DownturnResult,
    DownturnTradeRecord,
    EngineTag,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class DownturnMetrics:
    """Comprehensive metrics for scoring + diagnostics."""

    # Standard
    total_trades: int = 0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    calmar: float = 0.0
    sharpe: float = 0.0
    net_return_pct: float = 0.0
    win_rate: float = 0.0

    # Per-engine breakdown
    reversal_trades: int = 0
    reversal_wr: float = 0.0
    reversal_avg_r: float = 0.0
    breakdown_trades: int = 0
    breakdown_wr: float = 0.0
    breakdown_avg_r: float = 0.0
    fade_trades: int = 0
    fade_wr: float = 0.0
    fade_avg_r: float = 0.0
    momentum_trades: int = 0
    momentum_wr: float = 0.0
    momentum_avg_r: float = 0.0

    # Hold duration (5m bars)
    median_hold_5m: float = 0.0

    # Downturn-specific
    correction_pnl_pct: float = 0.0
    bear_regime_pnl: float = 0.0
    signal_to_entry_ratio: float = 0.0
    regime_detection_latency: float = 0.0  # deferred: requires per-window peak tracking
    exit_efficiency: float = 0.0
    bear_capture_ratio: float = 0.0
    correction_capture_ratio: float = 0.0  # PnL / available NQ move during correction windows
    sortino: float = 0.0  # annualized Sortino ratio from trade R-multiples

    # Correction coverage
    correction_coverage: float = 0.0  # fraction of eligible correction windows with >= 1 trade

    # Exit quality
    exit_quality: float = 0.0  # mean(actual_r * mfe); rewards magnitude and capture
    tp_hit_rates: dict[str, float] = field(default_factory=dict)
    avg_mfe_capture: float = 0.0
    low_mfe_trade_rate: float = 0.0  # fraction of trades that never reached +0.5R MFE
    low_mfe_loss_pct: float = 0.0  # low-MFE PnL as % of initial equity

    @property
    def correction_alpha_pct(self) -> float:
        """Backward-compatible alias for older payloads."""
        return self.correction_pnl_pct

    @correction_alpha_pct.setter
    def correction_alpha_pct(self, value: float) -> None:
        self.correction_pnl_pct = value


def compute_downturn_metrics(
    result: DownturnResult,
    daily_bars: NumpyBars,
    point_value: float = 2.0,
) -> DownturnMetrics:
    """Compute all downturn metrics from a backtest result."""
    trades = result.trades
    m = DownturnMetrics()

    if not trades:
        return m

    m.total_trades = len(trades)

    # Win rate
    wins = [t for t in trades if t.pnl > 0]
    m.win_rate = len(wins) / len(trades) if trades else 0.0

    # PnL
    total_pnl = sum(t.pnl for t in trades)
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Net return
    initial_eq = result.equity_curve[0] if len(result.equity_curve) > 0 else 100_000.0
    final_eq = initial_eq + total_pnl
    m.net_return_pct = (final_eq / initial_eq - 1) * 100 if initial_eq > 0 else 0.0

    # Max drawdown
    eq = result.equity_curve
    if len(eq) > 0:
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        dd[peak == 0] = 0
        m.max_dd_pct = float(np.max(dd))

    # Calmar
    n_days = len(daily_bars.closes)
    years = max(n_days / 252, 0.5)
    annual_ret = m.net_return_pct / years
    m.calmar = annual_ret / (m.max_dd_pct * 100) if m.max_dd_pct > 0.001 else 0.0

    # Sharpe and Sortino (per-trade, annualized by trades-per-year)
    if len(trades) >= 2:
        r_multiples = np.array([t.r_multiple for t in trades])
        mean_r = float(np.mean(r_multiples))
        std_r = float(np.std(r_multiples))
        trades_per_year = len(trades) / max(0.5, years)
        ann_factor = math.sqrt(trades_per_year)
        m.sharpe = (mean_r / std_r * ann_factor) if std_r > 0 else 0.0
        # Sortino: downside deviation only
        downside = r_multiples[r_multiples < 0]
        downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0
        m.sortino = (mean_r / downside_dev * ann_factor) if downside_dev > 0 else 0.0

    # Per-engine
    for tag, attr_prefix in [
        (EngineTag.REVERSAL, "reversal"),
        (EngineTag.BREAKDOWN, "breakdown"),
        (EngineTag.FADE, "fade"),
    ]:
        eng_trades = [t for t in trades if t.engine_tag == tag]
        setattr(m, f"{attr_prefix}_trades", len(eng_trades))
        if eng_trades:
            eng_wins = [t for t in eng_trades if t.pnl > 0]
            setattr(m, f"{attr_prefix}_wr", len(eng_wins) / len(eng_trades))
            setattr(m, f"{attr_prefix}_avg_r", float(np.mean([t.r_multiple for t in eng_trades])))

    # Momentum impulse (subset of fade, identified by signal_class)
    mom_trades = [t for t in trades if getattr(t, "signal_class", "") == "momentum_impulse"]
    m.momentum_trades = len(mom_trades)
    if mom_trades:
        mom_wins = [t for t in mom_trades if t.pnl > 0]
        m.momentum_wr = len(mom_wins) / len(mom_trades)
        m.momentum_avg_r = float(np.mean([t.r_multiple for t in mom_trades]))

    # Median hold duration (5m bars)
    hold_bars = [getattr(t, "hold_bars_5m", 0) for t in trades if getattr(t, "hold_bars_5m", 0) > 0]
    m.median_hold_5m = float(np.median(hold_bars)) if hold_bars else 0.0

    # Correction-window PnL scaled by initial equity
    correction_pnl = sum(t.pnl for t in trades if t.in_correction_window)
    m.correction_pnl_pct = (correction_pnl / initial_eq * 100) if initial_eq > 0 else 0.0

    # Correction capture ratio: PnL / available NQ move during each window
    if result.correction_windows:
        window_captures = []
        for w in result.correction_windows:
            w_trades = [
                t for t in trades
                if t.in_correction_window and t.entry_time
                and w.start_date <= t.entry_time <= w.end_date
            ]
            if not w_trades:
                continue
            w_pnl = sum(t.pnl for t in w_trades)
            # Estimate peak from max entry price (shorts enter near highs)
            peak_est = max(abs(t.entry_price) for t in w_trades)
            nq_move_points = peak_est * (w.peak_to_trough_pct / 100)
            avg_qty = float(np.mean([abs(t.qty) for t in w_trades]))
            available = nq_move_points * point_value * avg_qty
            if available > 0:
                window_captures.append(max(0, w_pnl) / available)
        m.correction_capture_ratio = float(np.mean(window_captures)) if window_captures else 0.0

    # Correction coverage: fraction of *tradeable* correction windows (>= 2 days) with >= 1 trade
    if result.correction_windows:
        min_days = 2  # Only count windows >= 2 trading days
        eligible_windows = [
            (i, w) for i, w in enumerate(result.correction_windows)
            if (w.end_date - w.start_date).days >= min_days
        ]
        if eligible_windows:
            covered = set()
            for t in trades:
                if t.in_correction_window and t.entry_time:
                    for i, w in eligible_windows:
                        if w.start_date <= t.entry_time <= w.end_date:
                            covered.add(i)
                            break
            m.correction_coverage = len(covered) / len(eligible_windows)
        else:
            m.correction_coverage = 0.0

    # Bear regime PnL
    from strategies.momentum.downturn.bt_models import CompositeRegime
    bear_regimes = {CompositeRegime.ALIGNED_BEAR, CompositeRegime.EMERGING_BEAR}
    m.bear_regime_pnl = sum(t.pnl for t in trades if t.composite_regime_at_entry in bear_regimes)

    # Signal to entry ratio
    total_signals = (
        result.reversal_counters.signals_detected
        + result.breakdown_counters.signals_detected
        + result.fade_counters.signals_detected
    )
    total_entries = (
        result.reversal_counters.entries_filled
        + result.breakdown_counters.entries_filled
        + result.fade_counters.entries_filled
    )
    m.signal_to_entry_ratio = total_entries / total_signals if total_signals > 0 else 0.0

    # Exit efficiency: total-based capture ratio (sum of gains / sum of MFE)
    total_captured = sum(max(0, t_rec.r_multiple) for t_rec in trades)
    total_mfe = sum(t_rec.mfe for t_rec in trades if t_rec.mfe > 0)
    m.avg_mfe_capture = total_captured / total_mfe if total_mfe > 0 else 0.0
    low_mfe_trades = [t_rec for t_rec in trades if t_rec.mfe < 0.5]
    m.low_mfe_trade_rate = len(low_mfe_trades) / len(trades) if trades else 0.0
    low_mfe_pnl = sum(t_rec.pnl for t_rec in low_mfe_trades)
    m.low_mfe_loss_pct = (min(0.0, low_mfe_pnl) / initial_eq * 100) if initial_eq > 0 else 0.0
    # Exit quality for diagnostics (mean of r * mfe products)
    r_mfe_products = [t_rec.r_multiple * t_rec.mfe for t_rec in trades if t_rec.mfe > 0]
    m.exit_quality = float(np.mean(r_mfe_products)) if r_mfe_products else 0.0
    m.exit_efficiency = m.avg_mfe_capture

    # TP hit rates by engine
    tp_counts: dict[str, dict[str, int]] = {}
    for tag in [EngineTag.REVERSAL, EngineTag.BREAKDOWN, EngineTag.FADE]:
        eng_trades = [t_rec for t_rec in trades if t_rec.engine_tag == tag]
        total_eng = len(eng_trades)
        if total_eng > 0:
            for tp_name in ["tp1", "tp2", "tp3"]:
                hits = sum(1 for t_rec in eng_trades if t_rec.exit_type == tp_name)
                m.tp_hit_rates[f"{tag.value}_{tp_name}"] = hits / total_eng

    # Bear capture ratio: proportional PnL / available move for windows >= 2%
    bear_captures = []
    for w in result.correction_windows:
        if w.peak_to_trough_pct >= 2.0:
            w_trades = [
                t_rec for t_rec in trades
                if t_rec.entry_time and w.start_date <= t_rec.entry_time <= w.end_date
            ]
            if not w_trades:
                bear_captures.append(0.0)
                continue
            w_pnl = sum(t_rec.pnl for t_rec in w_trades)
            peak_est = max(abs(t_rec.entry_price) for t_rec in w_trades)
            nq_move_points = peak_est * (w.peak_to_trough_pct / 100)
            avg_qty = float(np.mean([abs(t_rec.qty) for t_rec in w_trades]))
            available = nq_move_points * point_value * avg_qty
            bear_captures.append(max(0, w_pnl) / available if available > 0 else 0.0)
    m.bear_capture_ratio = float(np.mean(bear_captures)) if bear_captures else 0.0

    return m


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_downturn_report(metrics: DownturnMetrics) -> str:
    """Generate formatted diagnostic report."""
    buf = StringIO()
    m = metrics

    buf.write("=" * 70 + "\n")
    buf.write("DOWNTURN DOMINATOR BACKTEST REPORT\n")
    buf.write("=" * 70 + "\n\n")

    # Summary
    buf.write("--- Summary ---\n")
    buf.write(f"  Total trades:     {m.total_trades}\n")
    buf.write(f"  Win rate:         {m.win_rate:.1%}\n")
    buf.write(f"  Profit factor:    {m.profit_factor:.2f}\n")
    buf.write(f"  Net return:       {m.net_return_pct:.1f}%\n")
    buf.write(f"  Max drawdown:     {m.max_dd_pct:.2%}\n")
    buf.write(f"  Calmar:           {m.calmar:.2f}\n")
    buf.write(f"  Sharpe:           {m.sharpe:.2f}\n")
    buf.write(f"  Sortino:          {m.sortino:.2f}\n\n")

    # Per-engine
    buf.write("--- Per-Engine Breakdown ---\n")
    for name, trades, wr, avg_r in [
        ("Reversal", m.reversal_trades, m.reversal_wr, m.reversal_avg_r),
        ("Breakdown", m.breakdown_trades, m.breakdown_wr, m.breakdown_avg_r),
        ("Fade", m.fade_trades, m.fade_wr, m.fade_avg_r),
        ("Momentum", m.momentum_trades, m.momentum_wr, m.momentum_avg_r),
    ]:
        buf.write(f"  {name:12s}  trades={trades:4d}  WR={wr:.1%}  avgR={avg_r:+.2f}\n")
    buf.write("\n")

    # Correction-window PnL
    buf.write("--- Correction PnL ---\n")
    buf.write(f"  Correction PnL:      {m.correction_pnl_pct:.2f}%\n")
    buf.write(f"  Bear regime PnL:     ${m.bear_regime_pnl:,.0f}\n")
    buf.write(f"  Bear capture ratio:  {m.bear_capture_ratio:.3f}\n")
    buf.write(f"  Corr capture ratio:  {m.correction_capture_ratio:.3f}\n\n")

    # Signal quality
    buf.write("--- Signal Quality ---\n")
    buf.write(f"  Signal-to-entry ratio:  {m.signal_to_entry_ratio:.2f}\n")
    buf.write(f"  Exit efficiency:     {m.exit_efficiency:.2f}\n")
    buf.write(f"  Avg MFE capture:     {m.avg_mfe_capture:.2f}\n")
    buf.write(f"  Low-MFE trade rate:  {m.low_mfe_trade_rate:.1%}\n")
    buf.write(f"  Low-MFE loss:        {m.low_mfe_loss_pct:.2f}%\n\n")

    # TP hit rates
    if m.tp_hit_rates:
        buf.write("--- TP Hit Rates ---\n")
        for k, v in sorted(m.tp_hit_rates.items()):
            buf.write(f"  {k:25s}  {v:.1%}\n")
        buf.write("\n")

    return buf.getvalue()
