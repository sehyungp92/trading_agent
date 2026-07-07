"""Multi-overlay parameter sweep to optimize multi_overlay vs optimized_v1.

Sweeps 6 dimensions one-at-a-time (best-of-each), then runs a combined
validation with all winners applied simultaneously.

Usage:
    python -m backtest.run_overlay_sweep --phase all
    python -m backtest.run_overlay_sweep --phase weights
    python -m backtest.run_overlay_sweep --phase threshold
    python -m backtest.run_overlay_sweep --phase rsi
    python -m backtest.run_overlay_sweep --phase macd
    python -m backtest.run_overlay_sweep --phase adaptive
    python -m backtest.run_overlay_sweep --phase sym_weights
    python -m backtest.run_overlay_sweep --phase combined
"""
from __future__ import annotations

import argparse
import copy
import logging
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from backtests.swing.config_unified import UnifiedBacktestConfig, PRESETS
from backtests.swing.engine.unified_portfolio_engine import (
    UnifiedPortfolioData,
    UnifiedPortfolioResult,
    load_unified_data,
    run_unified,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers (same as run_capital_tilt.py)
# ---------------------------------------------------------------------------

def _sharpe(eq: np.ndarray) -> float:
    if len(eq) < 2:
        return 0.0
    ret = np.diff(eq) / eq[:-1]
    if len(ret) < 2 or np.std(ret) == 0:
        return 0.0
    return float(np.mean(ret) / np.std(ret) * np.sqrt(252 * 7))


def _max_dd_pct(eq: np.ndarray) -> float:
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / peak * 100))


def _calmar(eq: np.ndarray, init_eq: float) -> float:
    dd = _max_dd_pct(eq)
    ret = (eq[-1] - init_eq) / init_eq * 100 if len(eq) > 0 else 0.0
    return ret / abs(dd) if dd != 0 else 0.0


def _total_pnl(result: UnifiedPortfolioResult) -> float:
    return sum(s.total_pnl for s in result.strategy_results.values()) + result.overlay_pnl


# ---------------------------------------------------------------------------
# Base config builder
# ---------------------------------------------------------------------------

def _base_multi(equity: float) -> UnifiedBacktestConfig:
    """Start from current multi_overlay preset."""
    return PRESETS["multi_overlay"](equity)


# ---------------------------------------------------------------------------
# Run engine
# ---------------------------------------------------------------------------

def _run_one(
    label: str,
    cfg: UnifiedBacktestConfig,
    data: UnifiedPortfolioData,
) -> tuple[str, UnifiedBacktestConfig, UnifiedPortfolioResult]:
    t0 = time.perf_counter()
    result = run_unified(data, cfg)
    elapsed = time.perf_counter() - t0
    pnl = _total_pnl(result)
    sharpe = _sharpe(result.combined_equity)
    dd = _max_dd_pct(result.combined_equity)
    print(f"  {label:<40} PnL=${pnl:>+10,.0f}  Sharpe={sharpe:.2f}  DD={dd:.1f}%  ({elapsed:.0f}s)")
    return label, cfg, result


# ---------------------------------------------------------------------------
# Generic dimension sweeper
# ---------------------------------------------------------------------------

def _sweep_dimension(
    title: str,
    configs: list[tuple[str, UnifiedBacktestConfig]],
    data: UnifiedPortfolioData,
    equity: float,
) -> tuple[str, UnifiedBacktestConfig, list[tuple[str, float, float, float, float]]]:
    """Run a set of configs, print comparison table, return best by Sharpe."""
    print(f"\n{'='*80}")
    print(f"{title} ({len(configs)} variants)")
    print(f"{'='*80}")

    rows: list[tuple[str, float, float, float, float]] = []
    best_cfg: UnifiedBacktestConfig | None = None
    best_label = ""

    for label, cfg in configs:
        _, _, res = _run_one(label, cfg, data)
        pnl = _total_pnl(res)
        sharpe = _sharpe(res.combined_equity)
        dd = _max_dd_pct(res.combined_equity)
        calmar = _calmar(res.combined_equity, equity)
        rows.append((label, pnl, sharpe, dd, calmar))

    # Print comparison table
    print(f"\n  {'Label':<40} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print(f"  {'-'*75}")
    best_row = max(rows, key=lambda r: r[2])
    for label, pnl, sharpe, dd, calmar in rows:
        marker = " <-- best" if label == best_row[0] else ""
        print(f"  {label:<40} ${pnl:>9,.0f} {sharpe:>7.2f} {dd:>6.1f}% {calmar:>7.2f}{marker}")
    print(f"  {'='*75}\n")

    # Find matching config
    best_label = best_row[0]
    best_cfg = next(c for l, c in configs if l == best_label)
    return best_label, best_cfg, rows


# ---------------------------------------------------------------------------
# Individual dimension sweeps
# ---------------------------------------------------------------------------

def sweep_score_weights(
    equity: float, data: UnifiedPortfolioData,
) -> tuple[float, float, float]:
    """Sweep EMA/RSI/MACD score weights."""
    weight_sets = [
        (0.40, 0.30, 0.30),  # current
        (0.50, 0.25, 0.25),
        (0.60, 0.20, 0.20),
        (0.70, 0.15, 0.15),
    ]
    configs = []
    for w in weight_sets:
        label = f"w_ema{w[0]:.0%}_rsi{w[1]:.0%}_macd{w[2]:.0%}"
        cfg = _base_multi(equity)
        cfg.overlay_score_weights = w
        configs.append((label, cfg))

    best_label, best_cfg, _ = _sweep_dimension(
        "SCORE WEIGHTS SWEEP", configs, data, equity,
    )
    return best_cfg.overlay_score_weights


def sweep_entry_threshold(
    equity: float, data: UnifiedPortfolioData,
) -> float:
    """Sweep entry score threshold."""
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.60]
    configs = []
    for t in thresholds:
        label = f"entry_thresh_{t:.2f}"
        cfg = _base_multi(equity)
        cfg.overlay_entry_score_min = t
        configs.append((label, cfg))

    best_label, best_cfg, _ = _sweep_dimension(
        "ENTRY THRESHOLD SWEEP", configs, data, equity,
    )
    return best_cfg.overlay_entry_score_min


def sweep_rsi_bull_min(
    equity: float, data: UnifiedPortfolioData,
) -> float:
    """Sweep RSI bull_min threshold."""
    values = [25, 30, 35, 40]
    configs = []
    for v in values:
        label = f"rsi_bull_min_{v}"
        cfg = _base_multi(equity)
        cfg.overlay_rsi_bull_min = float(v)
        configs.append((label, cfg))

    best_label, best_cfg, _ = _sweep_dimension(
        "RSI BULL_MIN SWEEP", configs, data, equity,
    )
    return best_cfg.overlay_rsi_bull_min


def sweep_macd_pos_falling(
    equity: float, data: UnifiedPortfolioData,
) -> tuple[float, float, float, float]:
    """Sweep MACD positive-but-falling score."""
    values = [0.6, 0.7, 0.8, 0.9]
    configs = []
    for v in values:
        label = f"macd_pf_{v:.1f}"
        cfg = _base_multi(equity)
        cfg.overlay_macd_scores = (1.0, v, 0.0, 0.3)
        configs.append((label, cfg))

    best_label, best_cfg, _ = _sweep_dimension(
        "MACD POS-FALLING SCORE SWEEP", configs, data, equity,
    )
    return best_cfg.overlay_macd_scores


def sweep_adaptive_min_alloc(
    equity: float, data: UnifiedPortfolioData,
) -> float:
    """Sweep adaptive sizing minimum allocation."""
    values = [0.30, 0.50, 0.70, 1.00]
    configs = []
    for v in values:
        label = f"min_alloc_{v:.0%}"
        cfg = _base_multi(equity)
        cfg.overlay_min_alloc_pct = v
        configs.append((label, cfg))

    best_label, best_cfg, _ = _sweep_dimension(
        "ADAPTIVE MIN ALLOC SWEEP", configs, data, equity,
    )
    return best_cfg.overlay_min_alloc_pct


def sweep_symbol_weights(
    equity: float, data: UnifiedPortfolioData,
) -> dict[str, float]:
    """Sweep per-symbol overlay weight allocation."""
    weight_sets = [
        {"QQQ": 0.50, "GLD": 0.50},  # current
        {"QQQ": 0.55, "GLD": 0.45},
        {"QQQ": 0.45, "GLD": 0.55},
        {"QQQ": 0.60, "GLD": 0.40},
    ]
    configs = []
    for ws in weight_sets:
        syms = "/".join(f"{k}{v:.0%}" for k, v in ws.items())
        label = f"sym_{syms}"
        cfg = _base_multi(equity)
        cfg.overlay_weights = ws
        cfg.overlay_symbols = list(ws.keys())
        configs.append((label, cfg))

    best_label, best_cfg, _ = _sweep_dimension(
        "SYMBOL WEIGHTS SWEEP", configs, data, equity,
    )
    return best_cfg.overlay_weights


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------

def run_combined_validation(
    equity: float,
    data: UnifiedPortfolioData,
    best_weights: tuple[float, float, float],
    best_threshold: float,
    best_rsi_bull_min: float,
    best_macd_scores: tuple[float, float, float, float],
    best_min_alloc: float,
    best_sym_weights: dict[str, float],
) -> None:
    """Run combined best params vs optimized_v1 and current multi_overlay."""
    print(f"\n{'='*80}")
    print("COMBINED VALIDATION - best-of-each vs baselines")
    print(f"{'='*80}")
    print(f"  Score weights:    {best_weights}")
    print(f"  Entry threshold:  {best_threshold}")
    print(f"  RSI bull_min:     {best_rsi_bull_min}")
    print(f"  MACD scores:      {best_macd_scores}")
    print(f"  Min alloc:        {best_min_alloc}")
    print(f"  Symbol weights:   {best_sym_weights}")
    print()

    # 1. optimized_v1 baseline
    ov1_cfg = PRESETS["optimized_v1"](equity)
    _, _, ov1_res = _run_one("optimized_v1", ov1_cfg, data)

    # 2. current multi_overlay
    mo_cfg = PRESETS["multi_overlay"](equity)
    _, _, mo_res = _run_one("multi_overlay_current", mo_cfg, data)

    # 3. combined best
    best_cfg = _base_multi(equity)
    best_cfg.overlay_score_weights = best_weights
    best_cfg.overlay_entry_score_min = best_threshold
    best_cfg.overlay_rsi_bull_min = best_rsi_bull_min
    best_cfg.overlay_macd_scores = best_macd_scores
    best_cfg.overlay_min_alloc_pct = best_min_alloc
    best_cfg.overlay_weights = best_sym_weights
    best_cfg.overlay_symbols = list(best_sym_weights.keys())
    _, _, best_res = _run_one("multi_overlay_optimized", best_cfg, data)

    # Comparison table
    configs_results = [
        ("optimized_v1", ov1_res, ov1_cfg),
        ("multi_overlay_current", mo_res, mo_cfg),
        ("multi_overlay_optimized", best_res, best_cfg),
    ]
    print(f"\n  {'Config':<30} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'OvlPnL':>10}")
    print(f"  {'-'*75}")
    for label, res, cfg in configs_results:
        pnl = _total_pnl(res)
        sharpe = _sharpe(res.combined_equity)
        dd = _max_dd_pct(res.combined_equity)
        calmar = _calmar(res.combined_equity, cfg.initial_equity)
        print(f"  {label:<30} ${pnl:>9,.0f} {sharpe:>7.2f} {dd:>6.1f}% {calmar:>7.2f} ${res.overlay_pnl:>9,.0f}")

    # Per-symbol overlay PnL for the optimized variant
    if best_res.overlay_per_symbol_pnl:
        print(f"\n  Per-symbol overlay PnL (optimized):")
        for sym, pnl in sorted(best_res.overlay_per_symbol_pnl.items()):
            pct = pnl / equity * 100
            print(f"    {sym:<6} ${pnl:>+9,.2f} ({pct:+.1f}%)")

    ov1_sharpe = _sharpe(ov1_res.combined_equity)
    best_sharpe = _sharpe(best_res.combined_equity)
    if best_sharpe > ov1_sharpe:
        print(f"\n  RESULT: multi_overlay_optimized BEATS optimized_v1 by {best_sharpe - ov1_sharpe:+.2f} Sharpe")
    else:
        print(f"\n  RESULT: multi_overlay_optimized TRAILS optimized_v1 by {best_sharpe - ov1_sharpe:.2f} Sharpe")

    print(f"{'='*80}\n")

    # Print recommended config update
    print("  Recommended make_multi_overlay() update:")
    print(f"    overlay_score_weights={best_weights},")
    print(f"    overlay_ema_spread_norm={best_cfg.overlay_ema_spread_norm},")
    print(f"    overlay_entry_score_min={best_threshold},")
    print(f"    overlay_rsi_bull_min={best_rsi_bull_min},")
    print(f"    overlay_macd_scores={best_macd_scores},")
    print(f"    overlay_min_alloc_pct={best_min_alloc},")
    print(f"    overlay_weights={best_sym_weights},")
    print(f"    overlay_symbols={list(best_sym_weights.keys())},")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-overlay parameter sweep")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--data-dir", type=str, default="backtest/data/raw")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["weights", "threshold", "rsi", "macd",
                                 "adaptive", "sym_weights", "combined", "all"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load data once using the unified multi-overlay configuration
    cfg = _base_multi(args.equity)
    cfg.data_dir = Path(args.data_dir)
    print("Loading data...")
    data = load_unified_data(cfg)
    print("Data loaded.\n")

    # Defaults (current multi_overlay values)
    best_weights = (0.40, 0.30, 0.30)
    best_threshold = 0.60
    best_rsi_bull_min = 40.0
    best_macd_scores = (1.0, 0.6, 0.0, 0.3)
    best_min_alloc = 0.30
    best_sym_weights = {"QQQ": 0.50, "GLD": 0.50}

    if args.phase in ("weights", "all"):
        best_weights = sweep_score_weights(args.equity, data)

    if args.phase in ("threshold", "all"):
        best_threshold = sweep_entry_threshold(args.equity, data)

    if args.phase in ("rsi", "all"):
        best_rsi_bull_min = sweep_rsi_bull_min(args.equity, data)

    if args.phase in ("macd", "all"):
        best_macd_scores = sweep_macd_pos_falling(args.equity, data)

    if args.phase in ("adaptive", "all"):
        best_min_alloc = sweep_adaptive_min_alloc(args.equity, data)

    if args.phase in ("sym_weights", "all"):
        best_sym_weights = sweep_symbol_weights(args.equity, data)

    if args.phase in ("combined", "all"):
        run_combined_validation(
            args.equity, data,
            best_weights, best_threshold, best_rsi_bull_min,
            best_macd_scores, best_min_alloc, best_sym_weights,
        )


if __name__ == "__main__":
    main()
