"""Investigation 3: Regime-filtered buy-and-hold benchmark.

Compares:
1. Full buy-and-hold returns
2. Regime-filtered B&H: long only when trend_dir == LONG
3. ATRSS strategy returns on the same period

This tells us whether the pullback entry timing adds value,
or whether regime classification alone is the entire edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategies.swing.atrss.models import DailyState, Direction
from backtests.swing.data.preprocessing import NumpyBars


@dataclass
class RegimeBenchmarkResult:
    """Results for one symbol's regime benchmark comparison."""
    symbol: str = ""
    # Full buy-and-hold
    bh_total_return: float = 0.0
    bh_total_pct: float = 0.0
    bh_days: int = 0
    # Regime-filtered B&H (long when LONG, cash otherwise)
    regime_bh_total_return: float = 0.0
    regime_bh_total_pct: float = 0.0
    regime_bh_days_long: int = 0
    regime_bh_days_total: int = 0
    regime_bh_pct_time_invested: float = 0.0
    # Regime-filtered B&H (long when LONG, short when SHORT)
    regime_ls_total_return: float = 0.0
    regime_ls_total_pct: float = 0.0
    # ATRSS strategy
    strategy_total_return: float = 0.0
    strategy_total_pct: float = 0.0
    strategy_trades: int = 0
    # Derived
    regime_vs_bh_pct: float = 0.0  # regime B&H return / full B&H return
    strategy_vs_regime_pct: float = 0.0  # strategy / regime B&H
    entry_timing_value: float = 0.0  # strategy return - regime B&H return


def compute_regime_benchmark(
    symbol: str,
    daily: NumpyBars,
    daily_states: dict[int, DailyState],
    trades: list,
    initial_price: float | None = None,
) -> RegimeBenchmarkResult:
    """Compute regime-filtered buy-and-hold benchmark.

    Parameters
    ----------
    symbol : str
        Symbol name.
    daily : NumpyBars
        Daily OHLCV data.
    daily_states : dict mapping daily bar index -> DailyState
        Daily state computed during the backtest (contains trend_dir).
    trades : list of TradeRecord
        Completed strategy trades.
    initial_price : float, optional
        Starting price for return calculation. Defaults to first daily close.
    """
    result = RegimeBenchmarkResult(symbol=symbol)

    closes = daily.closes
    n = len(closes)
    if n < 2:
        return result

    if initial_price is None:
        initial_price = closes[0]

    # --- Full buy-and-hold ---
    result.bh_total_return = closes[-1] - closes[0]
    result.bh_total_pct = result.bh_total_return / closes[0] * 100
    result.bh_days = n

    # --- Regime-filtered B&H ---
    # For each day, compute daily return, then include it only if trend_dir == LONG
    regime_bh_pnl = 0.0
    regime_ls_pnl = 0.0
    days_long = 0
    days_short = 0

    for i in range(1, n):
        daily_return = closes[i] - closes[i - 1]
        d = daily_states.get(i)
        if d is None:
            # Try i-1 (state computed from prior bar)
            d = daily_states.get(i - 1)

        if d is not None:
            if d.trend_dir == Direction.LONG:
                regime_bh_pnl += daily_return
                regime_ls_pnl += daily_return
                days_long += 1
            elif d.trend_dir == Direction.SHORT:
                regime_ls_pnl -= daily_return  # short position
                days_short += 1
        # FLAT days: no position in regime B&H

    result.regime_bh_total_return = regime_bh_pnl
    result.regime_bh_total_pct = regime_bh_pnl / closes[0] * 100 if closes[0] > 0 else 0
    result.regime_bh_days_long = days_long
    result.regime_bh_days_total = n - 1  # daily returns = n-1
    result.regime_bh_pct_time_invested = days_long / (n - 1) * 100 if n > 1 else 0

    result.regime_ls_total_return = regime_ls_pnl
    result.regime_ls_total_pct = regime_ls_pnl / closes[0] * 100 if closes[0] > 0 else 0

    # --- ATRSS strategy results ---
    strategy_pnl = sum(t.pnl_dollars for t in trades)
    result.strategy_total_return = strategy_pnl
    result.strategy_total_pct = strategy_pnl / closes[0] * 100 if closes[0] > 0 else 0
    result.strategy_trades = len(trades)

    # --- Derived metrics ---
    if result.bh_total_return != 0:
        result.regime_vs_bh_pct = result.regime_bh_total_return / abs(result.bh_total_return) * 100
    if result.regime_bh_total_return != 0:
        result.strategy_vs_regime_pct = result.strategy_total_return / abs(result.regime_bh_total_return) * 100
    result.entry_timing_value = result.strategy_total_return - result.regime_bh_total_return

    return result


def format_regime_benchmark_report(
    results: dict[str, RegimeBenchmarkResult],
) -> str:
    """Format regime benchmark results as a printable report.

    Parameters
    ----------
    results : dict mapping symbol -> RegimeBenchmarkResult
    """
    lines = ["=" * 90, "INVESTIGATION 3: REGIME-FILTERED BUY-AND-HOLD BENCHMARK", "=" * 90]
    lines.append("")
    lines.append("Compares three approaches over the same period:")
    lines.append("  1. Full Buy-and-Hold (always long)")
    lines.append("  2. Regime B&H Long-Only (long when trend_dir == LONG, cash otherwise)")
    lines.append("  3. Regime B&H Long/Short (long when LONG, short when SHORT)")
    lines.append("  4. ATRSS Strategy (actual pullback entries + exits)")

    for symbol, r in results.items():
        lines.append(f"\n--- {symbol} ({r.bh_days} daily bars) ---")
        lines.append(f"{'Approach':<30} {'Total Return':>14} {'Return %':>10} {'Notes':>30}")
        lines.append("-" * 90)
        lines.append(
            f"{'Full Buy-and-Hold':<30} {r.bh_total_return:>+14.2f} "
            f"{r.bh_total_pct:>+9.2f}% {'100% invested':>30}"
        )
        lines.append(
            f"{'Regime B&H (Long-Only)':<30} {r.regime_bh_total_return:>+14.2f} "
            f"{r.regime_bh_total_pct:>+9.2f}% "
            f"{f'{r.regime_bh_pct_time_invested:.0f}% time invested ({r.regime_bh_days_long}d)':>30}"
        )
        lines.append(
            f"{'Regime B&H (Long/Short)':<30} {r.regime_ls_total_return:>+14.2f} "
            f"{r.regime_ls_total_pct:>+9.2f}% {'':>30}"
        )
        lines.append(
            f"{'ATRSS Strategy':<30} {r.strategy_total_return:>+14.2f} "
            f"{r.strategy_total_pct:>+9.2f}% "
            f"{f'{r.strategy_trades} trades':>30}"
        )

        lines.append("")
        lines.append("  Analysis:")
        lines.append(f"    Regime B&H captures {r.regime_vs_bh_pct:.1f}% of buy-and-hold returns")
        if r.regime_bh_total_return > 0:
            lines.append(f"    ATRSS captures {r.strategy_vs_regime_pct:.1f}% of regime B&H returns")
        lines.append(f"    Entry timing value: {r.entry_timing_value:+.2f} "
                     f"(strategy - regime B&H)")
        if r.entry_timing_value < 0:
            lines.append(f"    >> Pullback entry timing is DESTROYING value vs simple regime B&H")
        elif r.entry_timing_value > 0:
            lines.append(f"    >> Pullback entry timing is ADDING value vs simple regime B&H")
        else:
            lines.append(f"    >> Pullback entry timing adds no incremental value")

    return "\n".join(lines)
