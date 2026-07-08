"""Performance, behavior, and filter effectiveness reports."""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from backtests.swing.analysis.metrics import BuyAndHoldMetrics, PerformanceMetrics
from backtests.swing.analysis.shadow_tracker import ShadowTracker
from backtests.swing.engine.backtest_engine import SymbolResult, TradeRecord

logger = logging.getLogger(__name__)


def _fmt_pct(value: float) -> str:
    """Format percentage with adaptive precision (more decimals for tiny values)."""
    abs_val = abs(value)
    if abs_val >= 0.01:        # >= 1%
        return f"{value:.1%}"
    elif abs_val >= 0.001:     # >= 0.1%
        return f"{value:.2%}"
    else:                      # < 0.1%
        return f"{value:.3%}"


def _format_dd(metrics: PerformanceMetrics) -> str:
    """Format max drawdown line, preferring R-based DD when available."""
    if metrics.max_r_dd > 0:
        return f"Max drawdown:       {metrics.max_r_dd:.2f}R (${metrics.max_drawdown_dollar:,.2f})"
    return f"Max drawdown:       {_fmt_pct(metrics.max_drawdown_pct)} (${metrics.max_drawdown_dollar:,.2f})"


def performance_report(result: SymbolResult, metrics: PerformanceMetrics) -> str:
    """Generate a performance summary report."""
    lines = [
        f"=== Performance Report: {result.symbol} ===",
        f"Total trades:       {metrics.total_trades}",
        f"Win rate:           {metrics.win_rate:.1%}",
        f"Profit factor:      {metrics.profit_factor:.2f}",
        f"Expectancy (R):     {metrics.expectancy:+.3f}",
        f"Expectancy ($):     {metrics.expectancy_dollar:+,.2f}",
        f"Net profit:         ${metrics.net_profit:+,.2f}",
        f"CAGR:               {_fmt_pct(metrics.cagr)}",
        f"Sharpe:             {metrics.sharpe:.2f}",
        f"Sortino:            {metrics.sortino:.2f}",
        f"Calmar:             {metrics.calmar:.2f}",
        _format_dd(metrics),
        f"Avg hold (hours):   {metrics.avg_hold_hours:.1f}",
        f"Trades/month:       {metrics.trades_per_month:.1f}",
        f"Total commissions:  ${metrics.total_commissions:,.2f}",
        f"Tail loss (5%):     ${metrics.tail_loss_pct:,.2f}  ({metrics.tail_loss_r:+.2f}R)",
    ]
    if metrics.per_instrument_trades_per_month:
        lines.append("Per-instrument trades/month:")
        for sym, tpm in sorted(metrics.per_instrument_trades_per_month.items()):
            lines.append(f"  {sym}: {tpm:.1f}")
    return "\n".join(lines)

def behavior_report(trades: list[TradeRecord]) -> str:
    """Generate a behavior analysis report."""
    if not trades:
        return "No trades to analyze."

    lines = ["=== Behavior Report ==="]

    # Entry type breakdown
    entry_types = Counter(t.entry_type for t in trades)
    lines.append("\nEntry types:")
    for etype, count in entry_types.most_common():
        pct = count / len(trades) * 100
        avg_r = np.mean([t.r_multiple for t in trades if t.entry_type == etype])
        lines.append(f"  {etype:12s}: {count:4d} ({pct:5.1f}%)  avg R: {avg_r:+.3f}")

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
        avg_r_long = np.mean([t.r_multiple for t in long_trades])
        lines.append(f"  Avg R: {avg_r_long:+.3f}  Win rate: {np.mean([t.r_multiple > 0 for t in long_trades]):.1%}")
    lines.append(f"Short trades: {len(short_trades)}")
    if short_trades:
        avg_r_short = np.mean([t.r_multiple for t in short_trades])
        lines.append(f"  Avg R: {avg_r_short:+.3f}  Win rate: {np.mean([t.r_multiple > 0 for t in short_trades]):.1%}")

    # Hold time distribution
    hold_hours = [t.bars_held for t in trades]
    lines.append(f"\nHold time (bars):")
    lines.append(f"  Mean:   {np.mean(hold_hours):.1f}")
    lines.append(f"  Median: {np.median(hold_hours):.1f}")
    lines.append(f"  Min:    {np.min(hold_hours)}")
    lines.append(f"  Max:    {np.max(hold_hours)}")

    # MFE/MAE analysis
    mfe_values = [t.mfe_r for t in trades]
    mae_values = [t.mae_r for t in trades]
    lines.append(f"\nMFE (R-multiples):")
    lines.append(f"  Mean: {np.mean(mfe_values):.3f}  Median: {np.median(mfe_values):.3f}")
    lines.append(f"MAE (R-multiples):")
    lines.append(f"  Mean: {np.mean(mae_values):.3f}  Median: {np.median(mae_values):.3f}")

    # Add-on stats
    addon_a = [t for t in trades if t.addon_a_qty > 0]
    addon_b = [t for t in trades if t.addon_b_qty > 0]
    lines.append(f"\nTrades with Add-on A: {len(addon_a)}")
    lines.append(f"Trades with Add-on B: {len(addon_b)}")

    return "\n".join(lines)
def diagnostic_report(result: SymbolResult) -> str:
    """Short-side diagnostic: bias distribution, long/short breakdown, losing short detail."""
    trades = result.trades
    if not trades:
        return "No trades to diagnose."

    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]
    lines = [f"=== Diagnostic Report: {result.symbol} ==="]

    # --- Bias day distribution ---
    total_days = result.bias_days_long + result.bias_days_short + result.bias_days_flat
    lines.append("\nDaily confirmed-bias distribution:")
    if total_days > 0:
        lines.append(f"  LONG:  {result.bias_days_long:4d} ({100*result.bias_days_long/total_days:5.1f}%)")
        lines.append(f"  SHORT: {result.bias_days_short:4d} ({100*result.bias_days_short/total_days:5.1f}%)")
        lines.append(f"  FLAT:  {result.bias_days_flat:4d} ({100*result.bias_days_flat/total_days:5.1f}%)")
        if result.bias_days_long > 0:
            lines.append(f"  LONG:SHORT ratio: {result.bias_days_long/max(result.bias_days_short,1):.1f}:1")
    else:
        lines.append("  (no daily bars processed)")

    # --- Long/short trade breakdown ---
    n = len(trades)
    lines.append(f"\nTrade direction breakdown:")
    lines.append(f"  Total:  {n}")
    lines.append(f"  Long:   {len(long_trades)} ({100*len(long_trades)/n:.1f}%)")
    lines.append(f"  Short:  {len(short_trades)} ({100*len(short_trades)/n:.1f}%)")

    if long_trades:
        long_wins = sum(1 for t in long_trades if t.r_multiple > 0)
        long_avg_r = np.mean([t.r_multiple for t in long_trades])
        lines.append(f"\n  Long WR:    {100*long_wins/len(long_trades):.1f}%")
        lines.append(f"  Long avg R: {long_avg_r:+.3f}")
    if short_trades:
        short_wins = sum(1 for t in short_trades if t.r_multiple > 0)
        short_avg_r = np.mean([t.r_multiple for t in short_trades])
        lines.append(f"  Short WR:    {100*short_wins/len(short_trades):.1f}%")
        lines.append(f"  Short avg R: {short_avg_r:+.3f}")

    # --- Losing short trade detail ---
    losing_shorts = [t for t in short_trades if t.r_multiple <= 0]
    if losing_shorts:
        lines.append(f"\nLosing short trades ({len(losing_shorts)}):")
        for t in losing_shorts:
            expected_r = abs(t.initial_stop - t.entry_price)
            lines.append(
                f"  {t.entry_time} | entry={t.entry_price:.2f} stop={t.initial_stop:.2f} "
                f"exit={t.exit_price:.2f} | R={t.r_multiple:+.2f} MFE={t.mfe_r:.2f}R "
                f"MAE={t.mae_r:.2f}R | {t.exit_reason} | hold={t.bars_held}h "
                f"| risk_pts={expected_r:.2f}"
            )

    return "\n".join(lines)


def filter_effectiveness_report(tracker: ShadowTracker) -> str:
    """Generate filter effectiveness report from shadow tracking."""
    summary = tracker.get_filter_summary()
    if not summary:
        return "No shadow trades recorded."

    lines = ["=== Filter Effectiveness Report ===", ""]
    lines.append(
        f"{'Filter':<25s} {'Rejected':>8s} {'Filled':>7s} "
        f"{'Avg R':>7s} {'>1R':>6s} {'>2R':>6s} "
        f"{'Missed$':>10s} {'Avoided$':>10s}"
    )
    lines.append("-" * 90)

    for name, stats in sorted(summary.items()):
        lines.append(
            f"{name:<25s} {stats.rejected_count:>8d} {stats.filled_count:>7d} "
            f"{stats.avg_shadow_r:>+7.3f} {stats.pct_above_1r:>5.1f}% {stats.pct_above_2r:>5.1f}% "
            f"{stats.net_missed_expectancy:>+10.1f} {stats.net_avoided_loss:>10.1f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Negative avg R + high avoided$  ->  filter is PROTECTIVE (keep)")
    lines.append("  Positive avg R + high missed$   ->  filter is OVERLY RESTRICTIVE (investigate)")

    return "\n".join(lines)


def buy_and_hold_report(
    bh: BuyAndHoldMetrics,
    strategy_metrics: PerformanceMetrics,
) -> str:
    """Side-by-side comparison of buy-and-hold vs strategy (same capital basis)."""
    sizing_note = f"(qty={bh.qty}, mult={bh.multiplier})" if bh.qty > 1 or bh.multiplier != 1.0 else ""
    lines = [
        f"=== Buy & Hold vs Strategy: {bh.symbol} {sizing_note} ===",
        f"{'':20s} {'Buy & Hold':>14s} {'Strategy':>14s}",
        f"{'-'*50}",
        f"{'Net Profit ($)':20s} {bh.net_profit:>+14,.2f} {strategy_metrics.net_profit:>+14,.2f}",
        f"{'CAGR':20s} {bh.cagr:>13.1%} {_fmt_pct(strategy_metrics.cagr):>14s}",
        f"{'Max Drawdown %':20s} {bh.max_drawdown_pct:>13.1%} {_fmt_pct(strategy_metrics.max_drawdown_pct):>14s}",
        f"{'Max Drawdown ($)':20s} {bh.max_drawdown_dollar:>14,.2f} {strategy_metrics.max_drawdown_dollar:>14,.2f}",
        f"{'Price Return':20s} {bh.total_return_pct:>+13.1f}% {'':>14s}",
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
            f"CAGR={_fmt_pct(metrics.cagr)}  "
            f"Sharpe={metrics.sharpe:.2f}  "
            f"MaxDD={_fmt_pct(metrics.max_drawdown_pct)}  "
            f"$/mo={metrics.net_profit / max(metrics.trades_per_month * 12 / metrics.total_trades, 1):.0f}"
        )
    return "No trades"


def print_summary(metrics: PerformanceMetrics) -> None:
    """Print a compact one-line summary to console."""
    print(format_summary(metrics))


# ---------------------------------------------------------------------------
# Helix-specific reports
# ---------------------------------------------------------------------------

def helix_performance_report(symbol: str, metrics: PerformanceMetrics) -> str:
    """Generate a performance summary for a Helix symbol result.

    Accepts symbol + metrics directly (decoupled from HelixSymbolResult).
    """
    lines = [
        f"=== Helix Performance Report: {symbol} ===",
        f"Total trades:       {metrics.total_trades}",
        f"Win rate:           {metrics.win_rate:.1%}",
        f"Profit factor:      {metrics.profit_factor:.2f}",
        f"Expectancy (R):     {metrics.expectancy:+.3f}",
        f"Expectancy ($):     {metrics.expectancy_dollar:+,.2f}",
        f"Net profit:         ${metrics.net_profit:+,.2f}",
        f"CAGR:               {_fmt_pct(metrics.cagr)}",
        f"Sharpe:             {metrics.sharpe:.2f}",
        f"Sortino:            {metrics.sortino:.2f}",
        f"Calmar:             {metrics.calmar:.2f}",
        _format_dd(metrics),
        f"Avg hold (hours):   {metrics.avg_hold_hours:.1f}",
        f"Trades/month:       {metrics.trades_per_month:.1f}",
        f"Total commissions:  ${metrics.total_commissions:,.2f}",
        f"Tail loss (5%):     ${metrics.tail_loss_pct:,.2f}  ({metrics.tail_loss_r:+.2f}R)",
    ]
    if metrics.per_instrument_trades_per_month:
        lines.append("Per-instrument trades/month:")
        for sym, tpm in sorted(metrics.per_instrument_trades_per_month.items()):
            lines.append(f"  {sym}: {tpm:.1f}")
    return "\n".join(lines)


def helix_behavior_report(trades: list) -> str:
    """Helix behavior report: setup class breakdown, partials, R distribution, add-ons.

    Accepts list[HelixTradeRecord] (imported lazily to avoid circular imports).
    """
    if not trades:
        return "No trades to analyze."

    lines = ["=== Helix Behavior Report ==="]

    # Setup class breakdown
    from collections import Counter
    class_counts = Counter(t.setup_class for t in trades)
    lines.append("\nSetup class breakdown:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = count / len(trades) * 100 if trades else 0
        cls_trades = [t for t in trades if t.setup_class == cls]
        avg_r = np.mean([t.r_multiple for t in cls_trades]) if cls_trades else 0
        wr = np.mean([t.r_multiple > 0 for t in cls_trades]) * 100 if cls_trades else 0
        lines.append(f"  Class {cls}: {count:4d} ({pct:5.1f}%)  avg R: {avg_r:+.3f}  WR: {wr:.0f}%")

    # Origin TF breakdown
    tf_counts = Counter(t.origin_tf for t in trades)
    lines.append("\nOrigin timeframe:")
    for tf in ["4H", "1H"]:
        count = tf_counts.get(tf, 0)
        pct = count / len(trades) * 100 if trades else 0
        lines.append(f"  {tf}: {count:4d} ({pct:5.1f}%)")

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
        avg_r_long = np.mean([t.r_multiple for t in long_trades])
        lines.append(f"  Avg R: {avg_r_long:+.3f}  Win rate: {np.mean([t.r_multiple > 0 for t in long_trades]):.1%}")
    lines.append(f"Short trades: {len(short_trades)}")
    if short_trades:
        avg_r_short = np.mean([t.r_multiple for t in short_trades])
        lines.append(f"  Avg R: {avg_r_short:+.3f}  Win rate: {np.mean([t.r_multiple > 0 for t in short_trades]):.1%}")

    # Partial exit stats
    partial_1 = [t for t in trades if t.qty_partial_1 > 0]
    partial_2 = [t for t in trades if t.qty_partial_2 > 0]
    lines.append(f"\nPartial exits:")
    lines.append(f"  +2.5R partial (50%): {len(partial_1)} trades  ({100*len(partial_1)/max(len(trades),1):.1f}%)")
    lines.append(f"  +5R partial (25%):   {len(partial_2)} trades  ({100*len(partial_2)/max(len(trades),1):.1f}%)")

    # Add-on stats
    add_trades = [t for t in trades if t.add_on_qty > 0]
    lines.append(f"\nAdd-on entries: {len(add_trades)} ({100*len(add_trades)/max(len(trades),1):.1f}%)")

    # R-multiple distribution
    r_values = [t.r_multiple for t in trades]
    lines.append(f"\nR-multiple distribution:")
    lines.append(f"  Mean:   {np.mean(r_values):+.3f}")
    lines.append(f"  Median: {np.median(r_values):+.3f}")
    lines.append(f"  Std:    {np.std(r_values):.3f}")
    for thresh in [-2, -1, 0, 1, 2, 3, 5]:
        count = sum(1 for r in r_values if r >= thresh)
        lines.append(f"  >= {thresh:+d}R: {count} ({100*count/len(trades):.1f}%)")

    # MFE/MAE
    mfe_values = [t.mfe_r for t in trades]
    mae_values = [t.mae_r for t in trades]
    lines.append(f"\nMFE (R): mean={np.mean(mfe_values):.3f}  median={np.median(mfe_values):.3f}")
    lines.append(f"MAE (R): mean={np.mean(mae_values):.3f}  median={np.median(mae_values):.3f}")

    # Hold time
    hold_hours = [t.bars_held for t in trades]
    lines.append(f"\nHold time (1H bars):")
    lines.append(f"  Mean: {np.mean(hold_hours):.1f}  Median: {np.median(hold_hours):.1f}  Max: {np.max(hold_hours)}")

    return "\n".join(lines)


def helix_diagnostic_report(result) -> str:
    """Helix diagnostic: regime distribution, setup detection/fill rates.

    Accepts HelixSymbolResult (duck-typed to avoid circular imports).
    """
    lines = [f"=== Helix Diagnostic Report: {result.symbol} ==="]

    # Regime distribution
    total_days = result.regime_days_bull + result.regime_days_bear + result.regime_days_chop
    lines.append("\nDaily regime distribution:")
    if total_days > 0:
        lines.append(f"  BULL: {result.regime_days_bull:4d} ({100*result.regime_days_bull/total_days:5.1f}%)")
        lines.append(f"  BEAR: {result.regime_days_bear:4d} ({100*result.regime_days_bear/total_days:5.1f}%)")
        lines.append(f"  CHOP: {result.regime_days_chop:4d} ({100*result.regime_days_chop/total_days:5.1f}%)")
    else:
        lines.append("  (no daily bars processed)")

    # Setup pipeline funnel
    lines.append(f"\nSetup pipeline:")
    lines.append(f"  Detected: {result.setups_detected}")
    lines.append(f"  Armed:    {result.setups_armed}")
    lines.append(f"  Filled:   {result.setups_filled}")
    lines.append(f"  Expired:  {result.setups_expired}")

    if result.setups_detected > 0:
        arm_rate = result.setups_armed / result.setups_detected * 100
        fill_rate = result.setups_filled / result.setups_detected * 100
        lines.append(f"  Arm rate:  {arm_rate:.1f}%")
        lines.append(f"  Fill rate: {fill_rate:.1f}%")

    # Regime at entry breakdown (from trades)
    if result.trades:
        regime_at_entry = Counter(t.regime_at_entry for t in result.trades)
        lines.append(f"\nRegime at entry:")
        for regime, count in regime_at_entry.most_common():
            label = regime if regime else "unknown"
            lines.append(f"  {label}: {count}")

    return "\n".join(lines)
