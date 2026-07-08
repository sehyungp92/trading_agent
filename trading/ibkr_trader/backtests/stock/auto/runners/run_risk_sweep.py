"""IARIC Risk Parameter Sweep -- Post-V4R1.

Tests risk allocation parameters on top of V4R1's optimized signal/exit config
to find the right aggressiveness level.

Sweep design (layered, not full grid):
  Layer 1: base_risk_fraction (5 levels) -- per-trade risk
  Layer 2: intraday_leverage (4 levels) -- buying power cap
  Layer 3: thursday_mult (3 levels) -- day-of-week sizing
  Layer 4: Combined top picks from layers 1-3

Total: ~20 runs on $10K, 2024-01-01 to 2026-03-01.

Usage::

    python -m backtests.stock.auto.runners.run_risk_sweep
    python -m backtests.stock.auto.runners.run_risk_sweep --quick   # layers 1+2 only
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

DATA_DIR = Path("backtests/stock/data/raw")
OUTPUT_DIR = Path("backtests/stock/auto/iaric/output_risk_sweep")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0

# ---------------------------------------------------------------------------
# V4R1 final mutations (58 cumulative from phase_state.json)
# ---------------------------------------------------------------------------
V4R1_BASE: dict = {
    "max_per_sector": 2,
    "param_overrides.pb_atr_stop_mult": 1.0,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_carry_close_pct_min": 0.0,
    "param_overrides.pb_carry_mfe_gate_r": 0.0,
    "param_overrides.pb_cdd_max": 5,
    "param_overrides.pb_cdd_min": 3,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_delayed_confirm_after_bar": 3,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_delayed_confirm_min_daily_signal_score": 35.0,
    "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.0,
    "param_overrides.pb_delayed_confirm_score_min": 46.0,
    "param_overrides.pb_delayed_confirm_stale_exit_bars": 0,
    "param_overrides.pb_delayed_confirm_vwap_fail_cpr_max": -1.0,
    "param_overrides.pb_execution_mode": "intraday_hybrid",
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_friday_mult": 1.0,
    "param_overrides.pb_max_hold_days": 2,
    "param_overrides.pb_max_positions": 10,
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_min_candidates_day_hard_gate": False,
    "param_overrides.pb_open_scored_carry_close_pct_min": 0.0,
    "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.0,
    "param_overrides.pb_open_scored_enabled": True,
    "param_overrides.pb_open_scored_flow_reversal_lookback": 2,
    "param_overrides.pb_open_scored_max_hold_days": 2,
    "param_overrides.pb_opening_reclaim_enabled": False,
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.pb_thursday_mult": 0.5,
    "param_overrides.pb_v2_afternoon_retest_enabled": True,
    "param_overrides.pb_v2_allow_secular": True,
    "param_overrides.pb_v2_carry_overnight_stop_atr": 1.0,
    "param_overrides.pb_v2_ema_reversion_exit": True,
    "param_overrides.pb_v2_ema_reversion_min_r": 0.03,
    "param_overrides.pb_v2_enabled": True,
    "param_overrides.pb_v2_flatten_loss_r": -0.5,
    "param_overrides.pb_v2_flow_grace_days": 2,
    "param_overrides.pb_v2_mfe_stage1_stop_r": -0.1,
    "param_overrides.pb_v2_mfe_stage1_trigger": 0.3,
    "param_overrides.pb_v2_mfe_stage2_trigger": 0.6,
    "param_overrides.pb_v2_mfe_stage3_trail_atr": 0.75,
    "param_overrides.pb_v2_mfe_stage3_trigger": 1.25,
    "param_overrides.pb_v2_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_max_slots": 4,
    "param_overrides.pb_v2_open_scored_min_score": 45.0,
    "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
    "param_overrides.pb_v2_partial_profit_trigger_r": 0.3,
    "param_overrides.pb_v2_rsi_exit_open_scored": 60.0,
    "param_overrides.pb_v2_secular_sizing_mult": 0.65,
    "param_overrides.pb_v2_signal_floor": 75.0,
    "param_overrides.pb_v2_stale_bars": 4,
    "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
    "param_overrides.pb_v2_vwap_bounce_enabled": True,
    "param_overrides.pb_wednesday_mult": 1.0,
}


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

@dataclass
class SweepCase:
    name: str
    layer: str
    overrides: dict  # on top of V4R1_BASE


def build_sweep_cases(*, quick: bool = False) -> list[SweepCase]:
    cases: list[SweepCase] = []

    # Baseline (V4R1 as-is)
    cases.append(SweepCase("baseline_v4r1", "0_baseline", {}))

    # Layer 1: base_risk_fraction
    for brf in [0.0090, 0.0100, 0.0120, 0.0150]:
        cases.append(SweepCase(
            f"brf_{brf:.4f}",
            "1_risk_fraction",
            {"param_overrides.base_risk_fraction": brf},
        ))

    # Layer 2: intraday_leverage
    for lev in [2.5, 3.0, 4.0]:
        cases.append(SweepCase(
            f"lev_{lev:.1f}x",
            "2_leverage",
            {"param_overrides.intraday_leverage": lev},
        ))

    if quick:
        return cases

    # Layer 3: thursday_mult (with default risk)
    for thu in [0.75, 1.0]:
        cases.append(SweepCase(
            f"thu_{thu:.2f}",
            "3_thursday",
            {"param_overrides.pb_thursday_mult": thu},
        ))

    # Layer 4: Combined profiles
    combos = [
        ("moderate", 0.0100, 2.5, 0.50),
        ("lean_aggressive", 0.0100, 3.0, 0.75),
        ("aggressive", 0.0120, 3.0, 0.75),
        ("full_aggressive", 0.0120, 4.0, 1.0),
        ("max_aggressive", 0.0150, 4.0, 1.0),
    ]
    for label, brf, lev, thu in combos:
        cases.append(SweepCase(
            f"combo_{label}",
            "4_combined",
            {
                "param_overrides.base_risk_fraction": brf,
                "param_overrides.intraday_leverage": lev,
                "param_overrides.pb_thursday_mult": thu,
            },
        ))

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_single(replay, mutations: dict) -> dict:
    """Run one backtest, return metrics dict."""
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.auto.scoring import extract_metrics, composite_score, compute_r_multiples, IARIC_NORM
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

    config = mutate_iaric_config(
        IARICBacktestConfig(
            start_date=START_DATE,
            end_date=END_DATE,
            initial_equity=INITIAL_EQUITY,
            tier=3,
            data_dir=DATA_DIR,
        ),
        mutations,
    )
    result = IARICPullbackEngine(config, replay).run()
    perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, INITIAL_EQUITY)
    r_mult = compute_r_multiples(result.trades)
    score = composite_score(perf, INITIAL_EQUITY, r_multiples=r_mult, norm=IARIC_NORM)

    # Compute peak notional from equity curve
    eq = np.array(result.equity_curve)
    max_eq = float(np.max(eq)) if len(eq) > 0 else INITIAL_EQUITY

    return {
        "total_trades": perf.total_trades,
        "win_rate": perf.win_rate,
        "net_profit": perf.net_profit,
        "cagr": perf.cagr,
        "profit_factor": perf.profit_factor,
        "sharpe": perf.sharpe,
        "sortino": perf.sortino,
        "calmar": perf.calmar,
        "max_dd_pct": perf.max_drawdown_pct,
        "max_dd_dollar": perf.max_drawdown_dollar,
        "avg_hold_hours": perf.avg_hold_hours,
        "trades_per_month": perf.trades_per_month,
        "peak_equity": max_eq,
        "composite_score": score.total if not score.rejected else 0.0,
        "score_rejected": score.rejected,
    }


def format_table(results: list[dict]) -> str:
    """Format results as aligned ASCII table."""
    headers = [
        "Name", "Layer", "Trades", "WR%", "Net$", "CAGR%",
        "PF", "Sharpe", "Sortino", "Calmar", "MaxDD%", "MaxDD$",
        "Score", "Time",
    ]
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append([
            r["name"][:25],
            r["layer"][:15],
            f"{m['total_trades']}",
            f"{m['win_rate']*100:.1f}",
            f"{m['net_profit']:.0f}",
            f"{m['cagr']*100:.1f}",
            f"{m['profit_factor']:.2f}",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['calmar']:.1f}",
            f"{m['max_dd_pct']*100:.2f}",
            f"{m['max_dd_dollar']:.0f}",
            f"{m['composite_score']:.3f}",
            f"{r['elapsed_s']:.0f}s",
        ])

    # Compute column widths
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  "

    lines = []
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append(sep.join(val.rjust(w) if i >= 2 else val.ljust(w) for i, (val, w) in enumerate(zip(row, widths))))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run layers 1+2 only (~8 runs)")
    args = parser.parse_args()

    from backtests.stock.engine.research_replay import ResearchReplayEngine

    print("=" * 80)
    print("IARIC V4R1 Risk Parameter Sweep")
    print(f"Period: {START_DATE} to {END_DATE} | Equity: ${INITIAL_EQUITY:,.0f}")
    print("=" * 80)

    # Load data once
    print("\nLoading bar data...")
    t0 = time.time()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    cases = build_sweep_cases(quick=args.quick)
    print(f"Running {len(cases)} sweep configurations...\n")

    results = []
    for i, case in enumerate(cases, 1):
        mutations = {**V4R1_BASE, **case.overrides}
        risk_desc = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in case.overrides.items()) or "V4R1 defaults"
        print(f"[{i}/{len(cases)}] {case.name} ({risk_desc})...", end=" ", flush=True)

        t1 = time.time()
        try:
            metrics = run_single(replay, mutations)
            elapsed = time.time() - t1
            print(
                f"OK  trades={metrics['total_trades']}  "
                f"net=${metrics['net_profit']:.0f}  "
                f"dd={metrics['max_dd_pct']*100:.2f}%  "
                f"sharpe={metrics['sharpe']:.2f}  "
                f"calmar={metrics['calmar']:.1f}  "
                f"({elapsed:.0f}s)"
            )
            results.append({
                "name": case.name,
                "layer": case.layer,
                "overrides": case.overrides,
                "metrics": metrics,
                "elapsed_s": elapsed,
            })
        except Exception as exc:
            elapsed = time.time() - t1
            print(f"FAIL ({elapsed:.0f}s): {exc}")
            results.append({
                "name": case.name,
                "layer": case.layer,
                "overrides": case.overrides,
                "metrics": {k: 0 for k in [
                    "total_trades", "win_rate", "net_profit", "cagr", "profit_factor",
                    "sharpe", "sortino", "calmar", "max_dd_pct", "max_dd_dollar",
                    "avg_hold_hours", "trades_per_month", "peak_equity", "composite_score",
                    "score_rejected",
                ]},
                "elapsed_s": elapsed,
                "error": str(exc),
            })

    # Results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80 + "\n")
    print(format_table(results))

    # Layer analysis
    print("\n\n" + "=" * 80)
    print("LAYER ANALYSIS")
    print("=" * 80)

    baseline = next((r for r in results if r["name"] == "baseline_v4r1"), None)
    if baseline:
        bm = baseline["metrics"]
        print(f"\nBaseline (V4R1): net=${bm['net_profit']:.0f}, dd={bm['max_dd_pct']*100:.2f}%, "
              f"calmar={bm['calmar']:.1f}, sharpe={bm['sharpe']:.2f}")

        for layer_name in sorted(set(r["layer"] for r in results if r["layer"] != "0_baseline")):
            layer_results = [r for r in results if r["layer"] == layer_name]
            print(f"\n--- {layer_name} ---")
            best = max(layer_results, key=lambda r: r["metrics"]["composite_score"])
            for r in layer_results:
                m = r["metrics"]
                delta_profit = m["net_profit"] - bm["net_profit"]
                delta_dd = m["max_dd_pct"] - bm["max_dd_pct"]
                marker = " <-- BEST" if r is best else ""
                print(
                    f"  {r['name']:<25s}  net=${m['net_profit']:>7.0f} ({delta_profit:+.0f})  "
                    f"dd={m['max_dd_pct']*100:>5.2f}% ({delta_dd*100:+.2f}pp)  "
                    f"calmar={m['calmar']:>6.1f}  sharpe={m['sharpe']:.2f}  "
                    f"score={m['composite_score']:.3f}{marker}"
                )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "risk_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_equity": INITIAL_EQUITY,
            "total_cases": len(cases),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Save table to text file
    table_path = OUTPUT_DIR / "risk_sweep_table.txt"
    with open(table_path, "w") as f:
        f.write(f"IARIC V4R1 Risk Parameter Sweep\n")
        f.write(f"Period: {START_DATE} to {END_DATE} | Equity: ${INITIAL_EQUITY:,.0f}\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")
        f.write(format_table(results))
    print(f"Table saved to {table_path}")

    total_time = sum(r["elapsed_s"] for r in results)
    print(f"\nTotal sweep time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
