"""Per-symbol capital tilt optimization.

Sweeps overlay weights (QQQ/GLD ratio) and per-symbol active risk multipliers
to find the best capital allocation. Data is loaded once and reused.

Usage:
    python -m backtest.run_capital_tilt --phase diagnostic
    python -m backtest.run_capital_tilt --phase overlay
    python -m backtest.run_capital_tilt --phase active
    python -m backtest.run_capital_tilt --phase combined
    python -m backtest.run_capital_tilt --phase all
"""
from __future__ import annotations

import argparse
import logging
import time
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
# Metrics helpers
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


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _base_config(equity: float) -> UnifiedBacktestConfig:
    """Build optimized_v1 as the baseline for all sweeps."""
    return PRESETS["optimized_v1"](equity)


def _make_overlay_configs(
    equity: float,
    qqq_weights: list[float],
) -> list[tuple[str, UnifiedBacktestConfig]]:
    configs = []
    for qw in qqq_weights:
        gw = round(1.0 - qw, 2)
        label = f"ovl_q{qw:.0%}_g{gw:.0%}"
        cfg = _base_config(equity)
        cfg.overlay_weights = {"QQQ": qw, "GLD": gw}
        configs.append((label, cfg))
    return configs


def _make_active_configs(
    equity: float,
    best_overlay_weights: dict[str, float] | None,
) -> list[tuple[str, UnifiedBacktestConfig]]:
    multipliers = [0.75, 1.0, 1.25, 1.5]
    configs = []

    # ATRSS: sweep one dimension at a time
    for m in multipliers:
        if m == 1.0:
            continue
        label = f"atrss_qqq{m:.2f}"
        cfg = _base_config(equity)
        cfg.overlay_weights = best_overlay_weights
        cfg.symbol_risk_multipliers = {f"ATRSS:QQQ": m}
        configs.append((label, cfg))

        label = f"atrss_gld{m:.2f}"
        cfg = _base_config(equity)
        cfg.overlay_weights = best_overlay_weights
        cfg.symbol_risk_multipliers = {f"ATRSS:GLD": m}
        configs.append((label, cfg))

    # Helix: sweep one dimension at a time
    for sym in ["QQQ", "GLD"]:
        for m in multipliers:
            if m == 1.0:
                continue
            label = f"helix_{sym.lower()}{m:.2f}"
            cfg = _base_config(equity)
            cfg.overlay_weights = best_overlay_weights
            cfg.symbol_risk_multipliers = {f"AKC_HELIX:{sym}": m}
            configs.append((label, cfg))

    return configs


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
    total_pnl = sum(s.total_pnl for s in result.strategy_results.values()) + result.overlay_pnl
    sharpe = _sharpe(result.combined_equity)
    dd = _max_dd_pct(result.combined_equity)
    print(f"  {label:<30} PnL=${total_pnl:>+10,.0f}  Sharpe={sharpe:.2f}  DD={dd:.1f}%  ({elapsed:.0f}s)")
    return label, cfg, result


# ---------------------------------------------------------------------------
# Diagnostic: per-symbol PnL breakdown
# ---------------------------------------------------------------------------

def run_diagnostic(equity: float, data: UnifiedPortfolioData) -> None:
    print(f"\n{'='*70}")
    print(f"PER-SYMBOL PnL DIAGNOSTIC (optimized_v1, ${equity:,.0f})")
    print(f"{'='*70}")

    cfg = _base_config(equity)
    _, _, result = _run_one("optimized_v1_baseline", cfg, data)

    # Active strategies
    for label, trades in [
        ("ATRSS", result.atrss_trades),
        ("AKC_HELIX", result.helix_trades),
    ]:
        by_sym: dict[str, list] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)
        print(f"\n  {label}:")
        for sym, sym_trades in sorted(by_sym.items()):
            pnl = sum(t.pnl_dollars for t in sym_trades)
            wins = sum(1 for t in sym_trades if t.pnl_dollars > 0)
            n = len(sym_trades)
            wr = wins / n * 100 if n > 0 else 0
            avg_r = sum(t.r_multiple for t in sym_trades) / n if n > 0 else 0
            print(f"    {sym:<6} {n:>4} trades  WR={wr:5.1f}%  PnL=${pnl:>+9,.2f}  AvgR={avg_r:+.2f}")

    # Overlay per-symbol
    if result.overlay_per_symbol_pnl:
        print(f"\n  OVERLAY:")
        for sym, pnl in sorted(result.overlay_per_symbol_pnl.items()):
            pct = pnl / equity * 100
            print(f"    {sym:<6} PnL=${pnl:>+9,.2f}  ({pct:+.1f}%)")
        total_ovl = sum(result.overlay_per_symbol_pnl.values())
        print(f"    {'TOTAL':<6} PnL=${total_ovl:>+9,.2f}  (aggregate: ${result.overlay_pnl:>+9,.2f})")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Overlay weight sweep
# ---------------------------------------------------------------------------

def run_overlay_sweep(
    equity: float,
    data: UnifiedPortfolioData,
) -> dict[str, float] | None:
    qqq_weights = [round(0.30 + i * 0.05, 2) for i in range(9)]  # 0.30..0.70
    configs = _make_overlay_configs(equity, qqq_weights)

    # Add equal-weight baseline (overlay_weights=None)
    baseline_cfg = _base_config(equity)
    configs.insert(0, ("equal_weight", baseline_cfg))

    print(f"\n{'='*70}")
    print(f"OVERLAY WEIGHT SWEEP ({len(configs)} configs)")
    print(f"{'='*70}")

    results: list[tuple[str, float, float, float, float]] = []
    for label, cfg in configs:
        _, _, res = _run_one(label, cfg, data)
        total_pnl = sum(s.total_pnl for s in res.strategy_results.values()) + res.overlay_pnl
        sharpe = _sharpe(res.combined_equity)
        dd = _max_dd_pct(res.combined_equity)
        calmar = _calmar(res.combined_equity, equity)
        results.append((label, total_pnl, sharpe, dd, calmar))

    # Print comparison table
    print(f"\n{'Label':<30} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("-" * 65)
    best_sharpe_row = max(results, key=lambda r: r[2])
    for label, pnl, sharpe, dd, calmar in results:
        marker = " <-- best" if label == best_sharpe_row[0] else ""
        print(f"{label:<30} ${pnl:>9,.0f} {sharpe:>7.2f} {dd:>6.1f}% {calmar:>7.2f}{marker}")
    print(f"{'='*70}\n")

    # Return best weights (by Sharpe)
    baseline_sharpe = next(r[2] for r in results if r[0] == "equal_weight")
    if best_sharpe_row[2] > baseline_sharpe + 0.05:
        best_label = best_sharpe_row[0]
        # Parse weights from label
        if best_label == "equal_weight":
            return None
        qw = float(best_label.split("_q")[1].split("_g")[0].replace("%", "")) / 100
        return {"QQQ": qw, "GLD": round(1.0 - qw, 2)}
    else:
        print("  No significant improvement over equal-weight. Keeping equal-weight.\n")
        return None


# ---------------------------------------------------------------------------
# Active risk multiplier sweep
# ---------------------------------------------------------------------------

def run_active_sweep(
    equity: float,
    data: UnifiedPortfolioData,
    best_overlay_weights: dict[str, float] | None,
) -> dict[str, float]:
    configs = _make_active_configs(equity, best_overlay_weights)

    # Add baseline (no multipliers)
    baseline_cfg = _base_config(equity)
    baseline_cfg.overlay_weights = best_overlay_weights
    configs.insert(0, ("active_baseline", baseline_cfg))

    print(f"\n{'='*70}")
    print(f"ACTIVE RISK MULTIPLIER SWEEP ({len(configs)} configs)")
    print(f"{'='*70}")

    results: list[tuple[str, float, float, float, float]] = []
    for label, cfg in configs:
        _, _, res = _run_one(label, cfg, data)
        total_pnl = sum(s.total_pnl for s in res.strategy_results.values()) + res.overlay_pnl
        sharpe = _sharpe(res.combined_equity)
        dd = _max_dd_pct(res.combined_equity)
        calmar = _calmar(res.combined_equity, equity)
        results.append((label, total_pnl, sharpe, dd, calmar))

    # Print comparison table
    print(f"\n{'Label':<30} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}")
    print("-" * 65)
    best_sharpe_row = max(results, key=lambda r: r[2])
    for label, pnl, sharpe, dd, calmar in results:
        marker = " <-- best" if label == best_sharpe_row[0] else ""
        print(f"{label:<30} ${pnl:>9,.0f} {sharpe:>7.2f} {dd:>6.1f}% {calmar:>7.2f}{marker}")
    print(f"{'='*70}\n")

    # Return best multipliers (by Sharpe)
    baseline_sharpe = next(r[2] for r in results if r[0] == "active_baseline")
    if best_sharpe_row[2] > baseline_sharpe + 0.05 and best_sharpe_row[0] != "active_baseline":
        best_label = best_sharpe_row[0]
        # Reconstruct the multiplier from label
        cfg = next(c for l, c in configs if l == best_label)
        return dict(cfg.symbol_risk_multipliers)
    else:
        print("  No significant improvement. Keeping default risk allocation.\n")
        return {}


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------

def run_combined(
    equity: float,
    data: UnifiedPortfolioData,
    best_overlay_weights: dict[str, float] | None,
    best_multipliers: dict[str, float],
) -> None:
    print(f"\n{'='*70}")
    print("COMBINED VALIDATION")
    print(f"{'='*70}")
    print(f"  Overlay weights: {best_overlay_weights or 'equal-weight'}")
    print(f"  Risk multipliers: {best_multipliers or 'none'}")

    # Baseline (optimized_v1)
    baseline_cfg = _base_config(equity)
    _, _, baseline_res = _run_one("optimized_v1", baseline_cfg, data)

    # Combined best
    combined_cfg = _base_config(equity)
    combined_cfg.overlay_weights = best_overlay_weights
    combined_cfg.symbol_risk_multipliers = best_multipliers
    _, _, combined_res = _run_one("combined_best", combined_cfg, data)

    # Compare
    b_pnl = sum(s.total_pnl for s in baseline_res.strategy_results.values()) + baseline_res.overlay_pnl
    c_pnl = sum(s.total_pnl for s in combined_res.strategy_results.values()) + combined_res.overlay_pnl
    b_sh = _sharpe(baseline_res.combined_equity)
    c_sh = _sharpe(combined_res.combined_equity)
    b_dd = _max_dd_pct(baseline_res.combined_equity)
    c_dd = _max_dd_pct(combined_res.combined_equity)

    print(f"\n  {'Metric':<20} {'optimized_v1':>14} {'combined':>14} {'delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'PnL':<20} ${b_pnl:>13,.0f} ${c_pnl:>13,.0f} ${c_pnl-b_pnl:>+9,.0f}")
    print(f"  {'Sharpe':<20} {b_sh:>14.2f} {c_sh:>14.2f} {c_sh-b_sh:>+10.2f}")
    print(f"  {'Max DD':<20} {b_dd:>13.1f}% {c_dd:>13.1f}% {c_dd-b_dd:>+9.1f}pp")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-symbol capital tilt optimization")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--data-dir", type=str, default="backtest/data/raw")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["diagnostic", "overlay", "active", "combined", "all"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load data once
    cfg = _base_config(args.equity)
    cfg.data_dir = Path(args.data_dir)
    print("Loading data...")
    data = load_unified_data(cfg)

    best_overlay: dict[str, float] | None = None
    best_mults: dict[str, float] = {}

    if args.phase in ("diagnostic", "all"):
        run_diagnostic(args.equity, data)

    if args.phase in ("overlay", "all"):
        best_overlay = run_overlay_sweep(args.equity, data)

    if args.phase in ("active", "all"):
        best_mults = run_active_sweep(args.equity, data, best_overlay)

    if args.phase in ("combined", "all"):
        run_combined(args.equity, data, best_overlay, best_mults)


if __name__ == "__main__":
    main()
