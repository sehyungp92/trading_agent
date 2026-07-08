"""Performance, behavior, and diagnostic reports.

Shared utilities (format_summary, print_summary, buy_and_hold_report)
plus strategy-specific report functions for NQDTC.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from backtests.momentum.analysis.metrics import BuyAndHoldMetrics, PerformanceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def buy_and_hold_report(
    bh: BuyAndHoldMetrics,
    strategy_metrics: PerformanceMetrics,
) -> str:
    """Side-by-side comparison of buy-and-hold vs strategy."""
    lines = [
        f"=== Buy & Hold vs Strategy: {bh.symbol} ===",
        f"{'':20s} {'Buy & Hold':>14s} {'Strategy':>14s}",
        f"{'-'*50}",
        f"{'Total Return':20s} {bh.total_return_pct:>+13.1f}% {strategy_metrics.net_profit / max(strategy_metrics.total_trades, 1) * strategy_metrics.total_trades:>+13,.2f}$",
        f"{'CAGR':20s} {bh.cagr:>13.1%} {strategy_metrics.cagr:>13.1%}",
        f"{'Max Drawdown':20s} {bh.max_drawdown_pct:>13.1%} {strategy_metrics.max_drawdown_pct:>13.1%}",
        f"{'Start Price':20s} {bh.start_price:>14.2f} {'':>14s}",
        f"{'End Price':20s} {bh.end_price:>14.2f} {'':>14s}",
    ]
    return "\n".join(lines)


def format_summary(metrics: PerformanceMetrics) -> str:
    """Return a compact one-line summary string."""
    if metrics.total_trades > 0:
        return (
            f"Trades={metrics.total_trades}  "
            f"WR={metrics.win_rate:.0%}  "
            f"PF={metrics.profit_factor:.2f}  "
            f"E[R]={metrics.expectancy:+.3f}  "
            f"CAGR={metrics.cagr:.1%}  "
            f"Sharpe={metrics.sharpe:.2f}  "
            f"MaxDD={metrics.max_drawdown_pct:.1%}  "
            f"$/mo={metrics.net_profit / max(metrics.trades_per_month * 12 / metrics.total_trades, 1):.0f}"
        )
    return "No trades"


def print_summary(metrics: PerformanceMetrics) -> None:
    """Print a compact one-line summary to console."""
    print(format_summary(metrics))


def nqdtc_performance_report(symbol: str, metrics: PerformanceMetrics) -> str:
    """Performance summary for an NQDTC v2.0 backtest."""
    lines = [
        f"=== NQDTC v2.0 Performance Report: {symbol} ===",
        f"Total trades:       {metrics.total_trades}",
        f"Win rate:           {metrics.win_rate:.1%}",
        f"Profit factor:      {metrics.profit_factor:.2f}",
        f"Expectancy (R):     {metrics.expectancy:+.3f}",
        f"Expectancy ($):     {metrics.expectancy_dollar:+,.2f}",
        f"Net profit:         ${metrics.net_profit:+,.2f}",
        f"CAGR:               {metrics.cagr:.1%}",
        f"Sharpe:             {metrics.sharpe:.2f}",
        f"Sortino:            {metrics.sortino:.2f}",
        f"Calmar:             {metrics.calmar:.2f}",
        f"Max drawdown:       {metrics.max_drawdown_pct:.1%} (${metrics.max_drawdown_dollar:,.2f})",
        f"Avg hold (30m bars):{metrics.avg_hold_hours:.1f}",
        f"Trades/month:       {metrics.trades_per_month:.1f}",
        f"Total commissions:  ${metrics.total_commissions:,.2f}",
        f"Tail loss (5%):     ${metrics.tail_loss_pct:,.2f}  ({metrics.tail_loss_r:+.2f}R)",
    ]
    return "\n".join(lines)


def nqdtc_behavior_report(trades: list) -> str:
    """NQDTC behavior report: entry subtype, exit, session, regime, TP rates.

    Accepts list[NQDTCTradeRecord] (duck-typed).
    """
    if not trades:
        return "No trades to analyze."

    lines = ["=== NQDTC Behavior Report ==="]

    # Entry subtype breakdown
    subtype_counts = Counter(t.entry_subtype for t in trades)
    lines.append("\nEntry subtype breakdown:")
    for subtype, count in subtype_counts.most_common():
        pct = count / len(trades) * 100
        st_trades = [t for t in trades if t.entry_subtype == subtype]
        avg_r = np.mean([t.r_multiple for t in st_trades]) if st_trades else 0
        wr = np.mean([t.r_multiple > 0 for t in st_trades]) * 100 if st_trades else 0
        lines.append(f"  {subtype:20s}: {count:4d} ({pct:5.1f}%)  avg R: {avg_r:+.3f}  WR: {wr:.0f}%")

    # Exit reason breakdown
    exit_reasons = Counter(t.exit_reason for t in trades)
    lines.append("\nExit reasons:")
    for reason, count in exit_reasons.most_common():
        pct = count / len(trades) * 100
        avg_r = np.mean([t.r_multiple for t in trades if t.exit_reason == reason])
        lines.append(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)  avg R: {avg_r:+.3f}")

    # Direction breakdown
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]
    lines.append(f"\nLong trades:  {len(long_trades)}")
    if long_trades:
        lines.append(f"  Avg R: {np.mean([t.r_multiple for t in long_trades]):+.3f}  "
                      f"Win rate: {np.mean([t.r_multiple > 0 for t in long_trades]):.1%}")
    lines.append(f"Short trades: {len(short_trades)}")
    if short_trades:
        lines.append(f"  Avg R: {np.mean([t.r_multiple for t in short_trades]):+.3f}  "
                      f"Win rate: {np.mean([t.r_multiple > 0 for t in short_trades]):.1%}")

    # Session breakdown
    session_counts = Counter(t.session for t in trades)
    lines.append("\nSession breakdown:")
    for sess, count in session_counts.most_common():
        s_trades = [t for t in trades if t.session == sess]
        avg_r = np.mean([t.r_multiple for t in s_trades]) if s_trades else 0
        lines.append(f"  {sess:6s}: {count:4d}  avg R: {avg_r:+.3f}")

    # TP hit rates
    tp1 = sum(1 for t in trades if t.tp1_hit)
    tp2 = sum(1 for t in trades if t.tp2_hit)
    tp3 = sum(1 for t in trades if t.tp3_hit)
    n = len(trades)
    lines.append(f"\nTP hit rates:")
    lines.append(f"  TP1: {tp1} ({100 * tp1 / n:.0f}%)")
    lines.append(f"  TP2: {tp2} ({100 * tp2 / n:.0f}%)")
    lines.append(f"  TP3: {tp3} ({100 * tp3 / n:.0f}%)")

    # R-multiple distribution
    r_values = [t.r_multiple for t in trades]
    lines.append(f"\nR-multiple distribution:")
    lines.append(f"  Mean:   {np.mean(r_values):+.3f}")
    lines.append(f"  Median: {np.median(r_values):+.3f}")
    lines.append(f"  Std:    {np.std(r_values):.3f}")
    for thresh in [-2, -1, 0, 1, 2, 3, 5]:
        count = sum(1 for r in r_values if r >= thresh)
        lines.append(f"  >= {thresh:+d}R: {count} ({100 * count / n:.1f}%)")

    # MFE/MAE
    lines.append(f"\nMFE (R): mean={np.mean([t.mfe_r for t in trades]):.3f}  "
                  f"median={np.median([t.mfe_r for t in trades]):.3f}")
    lines.append(f"MAE (R): mean={np.mean([t.mae_r for t in trades]):.3f}  "
                  f"median={np.median([t.mae_r for t in trades]):.3f}")

    # Hold time
    hold = [t.bars_held_30m for t in trades]
    lines.append(f"\nHold time (30m bars):")
    lines.append(f"  Mean: {np.mean(hold):.1f}  Median: {np.median(hold):.1f}  Max: {np.max(hold)}")

    return "\n".join(lines)


def nqdtc_diagnostic_report(result) -> str:
    """NQDTC diagnostic: signal pipeline, regime distribution.

    Accepts NQDTCSymbolResult (duck-typed).
    """
    lines = [f"=== NQDTC Diagnostic Report: {result.symbol} ==="]

    lines.append(f"\nSignal pipeline:")
    lines.append(f"  Breakouts evaluated: {result.breakouts_evaluated}")
    lines.append(f"  Breakouts qualified: {result.breakouts_qualified}")
    lines.append(f"  Entries placed:      {result.entries_placed}")
    lines.append(f"  Entries filled:      {result.entries_filled}")
    lines.append(f"  Gates blocked:       {result.gates_blocked}")

    if result.breakouts_evaluated > 0:
        qual_rate = result.breakouts_qualified / result.breakouts_evaluated * 100
        lines.append(f"  Qualification rate:  {qual_rate:.1f}%")
    if result.entries_placed > 0:
        fill_rate = result.entries_filled / result.entries_placed * 100
        lines.append(f"  Entry fill rate:     {fill_rate:.1f}%")

    # Regime at entry breakdown (from trades)
    if result.trades:
        regime_counts = Counter(t.composite_regime for t in result.trades)
        lines.append(f"\nComposite regime at entry:")
        for regime, count in regime_counts.most_common():
            lines.append(f"  {regime}: {count}")

        # Exit tier breakdown
        tier_counts = Counter(t.exit_tier for t in result.trades)
        lines.append(f"\nExit tier at entry:")
        for tier, count in tier_counts.most_common():
            lines.append(f"  {tier}: {count}")

    return "\n".join(lines)
