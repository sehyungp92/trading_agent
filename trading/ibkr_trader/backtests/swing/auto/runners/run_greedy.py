"""Run greedy forward selection for swing portfolio optimization.

Usage:
    cd trading
    PYTHONUNBUFFERED=1 python -u backtests/swing/auto/run_greedy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Install swing aliases before any backtest imports
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data
from backtests.swing.auto.greedy_optimize import (
    PORTFOLIO_CANDIDATES,
    run_greedy,
    save_result,
)

EQUITY = 10_000.0
DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "swing" / "auto" / "output"


def main():
    print("Loading unified portfolio data...")
    t0 = time.time()
    config = UnifiedBacktestConfig(initial_equity=EQUITY, data_dir=DATA_DIR)
    data = load_unified_data(config)
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    result = run_greedy(
        data=data,
        candidates=PORTFOLIO_CANDIDATES,
        initial_equity=EQUITY,
        data_dir=DATA_DIR,
        max_workers=3,
        verbose=True,
    )

    save_result(result, OUTPUT_DIR / "greedy_portfolio_optimal.json")

    # Also run the final config once more and print detailed output
    print("\n\nRunning final config for detailed output...")
    from backtests.swing.auto.config_mutator import mutate_unified_config
    from backtests.swing.engine.unified_portfolio_engine import run_unified

    final_config = UnifiedBacktestConfig(initial_equity=EQUITY)
    if result.final_mutations:
        final_config = mutate_unified_config(final_config, result.final_mutations)

    final_result = run_unified(data, final_config)

    # Print per-strategy breakdown
    print("\nPer-Strategy Breakdown:")
    print(f"{'Strategy':<20} {'Trades':>6} {'Win%':>6} {'PnL':>12}")
    print("-" * 50)
    for attr, name in [
        ('atrss_trades', 'ATRSS'),
        ('helix_trades', 'AKC_HELIX'),
        ('tpc_trades', 'TPC'),
    ]:
        trades = getattr(final_result, attr, [])
        if not trades:
            print(f"{name:<20} {'0':>6}")
            continue
        pnls = [getattr(t, 'pnl_dollars', getattr(t, 'pnl', 0)) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        wr = wins / len(trades) * 100 if trades else 0
        print(f"{name:<20} {len(trades):>6} {wr:>5.1f}% ${total_pnl:>+10,.2f}")

    # Overlay stats
    overlay_pnl = getattr(final_result, 'overlay_pnl', 0)
    print(f"\nOverlay PnL: ${overlay_pnl:>+10,.2f}" if overlay_pnl else "")

    eq = final_result.combined_equity
    if len(eq) > 0:
        final_eq = float(eq[-1])
        total_pnl = final_eq - EQUITY
        total_ret = total_pnl / EQUITY * 100
        print(f"\nFinal Equity: ${final_eq:,.2f}")
        print(f"Total PnL:    ${total_pnl:+,.2f}")
        print(f"Total Return: {total_ret:+.1f}%")


if __name__ == "__main__":
    main()
