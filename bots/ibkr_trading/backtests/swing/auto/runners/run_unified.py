"""Unified swing portfolio backtest runner.

Usage:
    python -m backtest.run_unified [--equity 10000] [--data-dir backtest/data/raw]
                                   [--no-coordination]
    python -m backtest.run_unified --preset baseline --equity 10000
    python -m backtest.run_unified --run-all --equity 10000
    python -m backtest.run_unified --run-sweep
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from backtests.swing.config_unified import PRESETS, UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import (
    UnifiedPortfolioResult,
    load_unified_data,
    print_unified_report,
    run_unified,
)


def _compute_sharpe(equity: np.ndarray) -> float:
    """Annualized Sharpe from hourly equity curve."""
    if len(equity) < 2:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 7))


def _compute_max_dd_pct(equity: np.ndarray) -> float:
    """Max drawdown as a percentage (negative value)."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return float(np.min(dd))


def print_comparison_report(
    results: dict[str, tuple[UnifiedBacktestConfig, UnifiedPortfolioResult]],
    equity: float,
) -> None:
    """Print a side-by-side comparison table across presets."""
    print()
    print("=" * 110)
    print(f"PORTFOLIO OPTIMIZATION COMPARISON (${equity:,.0f})")
    print("=" * 110)
    hdr = (
        f"{'Preset':<22} {'Trades':>7} {'ATRSS':>6} {'Helix':>6} {'TPC':>6} "
        f"{'PnL':>10} {'OvlyPnL':>9} {'Sharpe':>7} {'MaxDD':>7} {'Blocked':>8}"
    )
    print(hdr)
    print("-" * 110)

    best_sharpe = ("", -999.0)
    best_pnl = ("", -1e18)
    best_dd = ("", -1e18)  # least negative = best

    for name, (cfg, res) in results.items():
        sr = res.strategy_results
        atrss_n = sr.get("ATRSS")
        helix_n = sr.get("AKC_HELIX")
        tpc_n = sr.get("TPC")
        total_trades = sum(s.total_trades for s in sr.values())
        total_pnl = sum(s.total_pnl for s in sr.values()) + res.overlay_pnl
        total_blocked = sum(s.entries_blocked_by_heat for s in sr.values())
        sharpe = _compute_sharpe(res.combined_equity)
        max_dd = _compute_max_dd_pct(res.combined_equity)

        ovly_pnl = res.overlay_pnl

        print(
            f"{name:<22} {total_trades:>7} "
            f"{atrss_n.total_trades if atrss_n else 0:>6} "
            f"{helix_n.total_trades if helix_n else 0:>6} "
            f"{tpc_n.total_trades if tpc_n else 0:>6} "
            f"{'${:>,.0f}'.format(total_pnl):>10} "
            f"{'${:>,.0f}'.format(ovly_pnl):>9} "
            f"{sharpe:>7.2f} "
            f"{max_dd:>6.1f}% "
            f"{total_blocked:>8}"
        )

        if sharpe > best_sharpe[1]:
            best_sharpe = (name, sharpe)
        if total_pnl > best_pnl[1]:
            best_pnl = (name, total_pnl)
        if max_dd > best_dd[1]:
            best_dd = (name, max_dd)

    print("-" * 110)
    print(f"  Best Sharpe:  {best_sharpe[0]} ({best_sharpe[1]:.2f})")
    print(f"  Best PnL:     {best_pnl[0]} (${best_pnl[1]:,.0f})")
    print(f"  Lowest MaxDD: {best_dd[0]} ({best_dd[1]:.1f}%)")
    print("=" * 110)


def _run_preset(
    name: str,
    equity: float,
    data_dir: Path,
) -> tuple[UnifiedBacktestConfig, UnifiedPortfolioResult]:
    """Build config from preset, load data, and run backtest."""
    factory = PRESETS[name]
    config = factory(equity)
    config.data_dir = data_dir

    data = load_unified_data(config)
    t0 = time.perf_counter()
    result = run_unified(data, config)
    elapsed = time.perf_counter() - t0

    total_trades = sum(s.total_trades for s in result.strategy_results.values())
    total_pnl = sum(s.total_pnl for s in result.strategy_results.values()) + result.overlay_pnl
    print(f"  {name:<22} ${equity:>10,.0f}  {total_trades:>4} trades  "
          f"PnL=${total_pnl:>+10,.0f}  ({elapsed:.1f}s)")
    return config, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unified 3-strategy portfolio backtest",
    )
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="Initial equity (default: $10,000)")
    parser.add_argument("--data-dir", type=str, default="backtest/data/raw",
                        help="Path to parquet data directory")
    parser.add_argument("--no-coordination", action="store_true",
                        help="Disable cross-strategy coordination rules")
    parser.add_argument("--fixed-qty", type=int, default=None,
                        help="Fixed position size (default: None = risk-based)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Run a single named preset")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all presets at the specified equity")
    parser.add_argument("--run-sweep", action="store_true",
                        help="Run all presets at both $10K and $100K")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_dir = Path(args.data_dir)

    # --run-sweep: all presets at $10K and $100K
    if args.run_sweep:
        for equity in [10_000.0, 100_000.0]:
            print(f"\n{'='*60}")
            print(f"SWEEP: equity=${equity:,.0f}")
            print(f"{'='*60}")
            all_results: dict[str, tuple[UnifiedBacktestConfig, UnifiedPortfolioResult]] = {}
            for name in PRESETS:
                cfg, res = _run_preset(name, equity, data_dir)
                all_results[name] = (cfg, res)
            print_comparison_report(all_results, equity)
        return

    # --run-all: all presets at specified equity
    if args.run_all:
        equity = args.equity
        print(f"\nRunning all presets at ${equity:,.0f}...")
        all_results = {}
        for name in PRESETS:
            cfg, res = _run_preset(name, equity, data_dir)
            all_results[name] = (cfg, res)
        print_comparison_report(all_results, equity)
        return

    # --preset NAME: single preset
    if args.preset:
        equity = args.equity
        print(f"\nRunning preset '{args.preset}' at ${equity:,.0f}...")
        cfg, res = _run_preset(args.preset, equity, data_dir)
        print_unified_report(res, cfg)
        return

    # Default: legacy mode (manual config)
    fixed_qty = args.fixed_qty
    if fixed_qty is None:
        # Legacy default was 10 for backward compat when no preset flags used
        fixed_qty = 10

    config = UnifiedBacktestConfig(
        initial_equity=args.equity,
        data_dir=data_dir,
        enable_atrss_helix_tighten=not args.no_coordination,
        enable_atrss_helix_size_boost=not args.no_coordination,
        fixed_qty=fixed_qty,
    )

    print(f"Loading data from {config.data_dir} ...")
    data = load_unified_data(config)

    print(f"Running unified backtest (equity=${config.initial_equity:,.0f}) ...")
    t0 = time.perf_counter()
    result = run_unified(data, config)
    elapsed = time.perf_counter() - t0

    print_unified_report(result, config)
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
