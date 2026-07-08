"""Full swing auto pipeline: experiments -> greedy optimization -> diagnostics -> v3 comparison.

Usage:
    cd trading
    python -u backtests/swing/auto/run_full_pipeline.py
"""
from __future__ import annotations

import sys
import time
import json
import re
import traceback
import numpy as np
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Install swing aliases before any backtest imports
from backtests.swing.auto.harness import SwingAutoHarness
from backtests.swing.auto.experiments import build_experiment_queue
from backtests.swing.auto.greedy_optimize import run_greedy, save_result
from backtests.swing.auto.config_mutator import mutate_unified_config
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data, run_unified

EQUITY = 10_000.0
DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "swing" / "auto" / "output"
V3_PATH = ROOT / "backtests" / "swing" / "output" / "portfolio_optimized_v3.txt"


def convert_experiment_to_unified_mutation(strategy: str, mutations: dict) -> dict:
    """Convert strategy-level experiment mutations to unified config mutations."""
    unified = {}
    for key, value in mutations.items():
        if strategy == "portfolio":
            # Portfolio experiments already use unified-level keys
            unified[key] = value
        else:
        # ATRSS, Helix, and TPC have flags/param_overrides/slippage-style routing.
            if key.startswith("flags."):
                unified[f"{strategy}_{key}"] = value
            elif key.startswith("param_overrides."):
                param = key.split(".", 1)[1]
                unified[f"{strategy}_param.{param}"] = value
            elif key.startswith("slippage."):
                unified[f"{strategy}_{key}"] = value
            else:
                unified[f"{strategy}.{key}"] = value
    return unified


def parse_v3(path: Path) -> dict:
    """Parse V3 baseline file for comparison metrics."""
    text = path.read_text()
    v3: dict = {}
    m = re.search(r"Final Equity:\s+\$([\d,.]+)", text)
    if m:
        v3["final_equity"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Total Return:\s+\+([\d.]+)%", text)
    if m:
        v3["total_return_pct"] = float(m.group(1))
    m = re.search(r"Total PnL:\s+\$\+([\d,.]+)", text)
    if m:
        v3["total_pnl"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Max Drawdown:\s+-([\d.]+)%", text)
    if m:
        v3["max_dd_pct"] = -float(m.group(1))
    m = re.search(r"Sharpe Ratio:\s+([\d.]+)", text)
    if m:
        v3["sharpe"] = float(m.group(1))
    m = re.search(r"Total Trades:\s+(\d+)", text)
    if m:
        v3["total_trades"] = int(m.group(1))

    # Per-strategy from the breakdown table (regex handles $ sign + comma-separated numbers)
    v3["strategies"] = {}
    for line in text.splitlines():
        for sname in ["ATRSS", "AKC_HELIX", "TPC"]:
            if line.strip().startswith(sname):
                m = re.match(
                    rf"\s*{sname}\s+(\d+)\s+([\d.]+)%\s+\$\s*([\d,.]+)",
                    line,
                )
                if m:
                    v3["strategies"][sname] = {
                        "trades": int(m.group(1)),
                        "win_rate": float(m.group(2)),
                        "pnl": float(m.group(3).replace(",", "")),
                    }
    return v3


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    # ================================================================
    # PHASE 1: Full auto pipeline (all experiments)
    # ================================================================
    print("=" * 70, flush=True)
    print("PHASE 1: Running full auto pipeline", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    harness = SwingAutoHarness(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        initial_equity=EQUITY,
    )
    harness.run_all(
        strategy_filter="all",
        skip_robustness=True,
        resume=True,
    )
    phase1_time = time.time() - t0
    print(f"\nPhase 1 complete in {phase1_time:.0f}s ({phase1_time / 60:.1f}m)", flush=True)

    # ================================================================
    # PHASE 2: Build greedy candidates from positive-impact experiments
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 2: Building greedy candidates from positive-impact experiments", flush=True)
    print("=" * 70, flush=True)

    results = harness.tracker.load_all()
    experiments_map = {e.id: e for e in build_experiment_queue("all")}

    # Collect all experiments with positive delta (any status except CRASH/UNWIRED)
    positive = []
    seen_mutations = set()
    for r in results:
        if r.delta_pct <= 0 or r.status in ("CRASH", "UNWIRED"):
            continue
        exp = experiments_map.get(r.experiment_id)
        if not exp or not exp.mutations:
            continue
        try:
            unified_muts = convert_experiment_to_unified_mutation(exp.strategy, exp.mutations)
        except Exception as e:
            print(f"  WARNING: Could not convert {r.experiment_id}: {e}", flush=True)
            continue

        # Deduplicate by mutation content
        muts_key = json.dumps(unified_muts, sort_keys=True)
        if muts_key in seen_mutations:
            continue
        seen_mutations.add(muts_key)

        positive.append({
            "id": r.experiment_id,
            "strategy": exp.strategy,
            "delta_pct": r.delta_pct,
            "score": r.experiment_score,
            "status": r.status,
            "mutations": unified_muts,
        })

    positive.sort(key=lambda x: x["delta_pct"], reverse=True)
    print(f"\nFound {len(positive)} unique positive-impact experiments:", flush=True)
    for p in positive[:40]:
        print(f"  {p['id']:<45} delta={p['delta_pct']:+.2%}  [{p['status']}]  {p['mutations']}", flush=True)
    if len(positive) > 40:
        print(f"  ... and {len(positive) - 40} more", flush=True)

    # Build greedy candidates
    candidates = [(p["id"], p["mutations"]) for p in positive]
    print(f"\nBuilt {len(candidates)} greedy candidates", flush=True)

    # Save candidates reference
    with open(OUTPUT_DIR / "greedy_candidates.json", "w") as f:
        json.dump(
            [{"name": n, "mutations": m, "delta_pct": p["delta_pct"]}
             for (n, m), p in zip(candidates, positive)],
            f, indent=2,
        )

    if not candidates:
        print("\nNo positive-impact experiments found. Exiting.", flush=True)
        return

    # ================================================================
    # PHASE 3: Greedy forward selection
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 3: Greedy forward selection ({len(candidates)} candidates)", flush=True)
    print("=" * 70, flush=True)
    t2 = time.time()

    config = UnifiedBacktestConfig(initial_equity=EQUITY, data_dir=DATA_DIR)
    data = load_unified_data(config)

    result = run_greedy(
        data=data,
        candidates=candidates,
        initial_equity=EQUITY,
        data_dir=DATA_DIR,
        max_workers=3,
        verbose=True,
    )

    save_result(result, OUTPUT_DIR / "greedy_portfolio_optimal.json")
    phase3_time = time.time() - t2
    print(f"\nGreedy optimization complete in {phase3_time:.0f}s ({phase3_time / 60:.1f}m)", flush=True)
    print(f"Kept features: {result.kept_features}", flush=True)
    print(f"Final score: {result.final_score:.4f} (base: {result.base_score:.4f})", flush=True)

    # ================================================================
    # PHASE 4: Full diagnostics of optimal configuration
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 4: Full diagnostics of optimal configuration", flush=True)
    print("=" * 70, flush=True)

    final_config = UnifiedBacktestConfig(initial_equity=EQUITY, data_dir=DATA_DIR)
    if result.final_mutations:
        final_config = mutate_unified_config(final_config, result.final_mutations)

    final_result = run_unified(data, final_config)
    eq = np.array(final_result.combined_equity, dtype=float)
    final_eq = float(eq[-1])
    total_pnl = final_eq - EQUITY

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd_pct = float(np.min(dd)) * 100
    max_dd_dollars = float(np.min(eq - peak))

    # Sharpe ratio (resample to daily returns for proper annualization)
    timestamps = final_result.combined_timestamps
    if timestamps is not None and len(timestamps) > 1:
        import pandas as pd
        eq_series = pd.Series(eq, index=pd.to_datetime(timestamps))
        daily_eq = eq_series.resample("D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    else:
        returns = np.diff(eq) / eq[:-1]
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0

    # Profit Factor
    all_trades = []
    strat_list = [
        ('atrss_trades', 'ATRSS'),
        ('helix_trades', 'AKC_HELIX'),
        ('tpc_trades', 'TPC'),
    ]
    strategies_data = []
    for attr, name in strat_list:
        trades = getattr(final_result, attr, []) or []
        if not trades:
            strategies_data.append((name, 0, 0.0, 0.0, 0.0))
            continue
        pnls = []
        for t in trades:
            p = getattr(t, 'pnl_dollars', None)
            if p is None:
                p = getattr(t, 'pnl', 0)
            pnls.append(float(p))
        all_trades.extend(pnls)
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        wr = wins / len(trades) * 100 if trades else 0
        total_r = 0.0
        for t, p in zip(trades, pnls):
            r = getattr(t, 'initial_risk_dollars', None)
            if r is None:
                r = getattr(t, 'risk_dollars', 1.0)
            r = float(r) if r else 1.0
            if r > 0:
                total_r += p / r
        strategies_data.append((name, len(trades), wr, total, total_r))

    total_trades = sum(s[1] for s in strategies_data)
    gross_wins = sum(p for p in all_trades if p > 0)
    gross_losses = abs(sum(p for p in all_trades if p < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    overlay_pnl = getattr(final_result, 'overlay_pnl', 0.0)
    if overlay_pnl is None:
        overlay_pnl = 0.0
    active_pnl = total_pnl - overlay_pnl

    # Calmar ratio
    calmar = (total_pnl / EQUITY) / abs(max_dd_pct / 100) if max_dd_pct != 0 else 0.0

    # ================================================================
    # Build diagnostics text
    # ================================================================
    diag = []
    diag.append("")
    diag.append("=" * 70)
    diag.append("UNIFIED SWING PORTFOLIO BACKTEST RESULTS (GREEDY OPTIMAL)")
    diag.append("=" * 70)
    diag.append(f"Initial Equity:  ${EQUITY:,.2f}")
    diag.append(f"Final Equity:    ${final_eq:,.2f}")
    diag.append(f"Total Return:    {total_pnl / EQUITY * 100:+.2f}%")
    diag.append(f"Total PnL:       ${total_pnl:+,.2f}")
    diag.append(f"Max Drawdown:    {max_dd_pct:.2f}% (${max_dd_dollars:+,.2f})")
    diag.append(f"Sharpe Ratio:    {sharpe:.2f}")
    diag.append(f"Profit Factor:   {profit_factor:.2f}")
    diag.append(f"Calmar Ratio:    {calmar:.2f}")
    diag.append(f"Total Trades:    {total_trades}")
    diag.append(f"Timeline:        {len(eq)} bars")
    diag.append("")

    if overlay_pnl:
        diag.append("Idle-Capital Overlay")
        diag.append(f"  Overlay PnL:         ${overlay_pnl:+,.2f} ({overlay_pnl / EQUITY * 100:+.1f}%)")
        diag.append(f"  Active Strategy PnL: ${active_pnl:+,.2f} ({active_pnl / EQUITY * 100:+.1f}%)")
        diag.append("")

    diag.append("Greedy Optimization Results")
    diag.append(f"  Base Score:    {result.base_score:.4f}")
    diag.append(f"  Final Score:   {result.final_score:.4f}")
    delta_score = (result.final_score - result.base_score) / result.base_score * 100 if result.base_score else 0
    diag.append(f"  Improvement:   {delta_score:+.1f}%")
    diag.append(f"  Rounds:        {len(result.rounds)}")
    diag.append(f"  Kept Features: {len(result.kept_features)}")
    for feat in result.kept_features:
        diag.append(f"    - {feat}")
    diag.append(f"  Final Mutations:")
    for k, v in (result.final_mutations or {}).items():
        diag.append(f"    {k}: {v}")
    diag.append("")

    diag.append("Per-Strategy Breakdown")
    diag.append(f"{'Strategy':<25} {'Trades':>6} {'Win%':>6} {'PnL':>12} {'Total R':>8}")
    diag.append("-" * 60)
    for name, cnt, wr, pnl, tot_r in strategies_data:
        if cnt > 0:
            diag.append(f"{name:<25} {cnt:>6} {wr:>5.1f}% ${pnl:>+10,.2f} {tot_r:>+7.1f}R")
        else:
            diag.append(f"{name:<25} {'0':>6}")
    diag.append("")

    # Round-by-round details
    diag.append("Greedy Optimization Round-by-Round")
    diag.append("-" * 70)
    for r in result.rounds:
        status = "KEPT" if r.kept else "STOP"
        diag.append(
            f"  Round {r.round_num}: {r.best_name:<40} "
            f"score={r.best_score:.4f} delta={r.best_delta_pct:+.2%} [{status}]"
        )
    diag.append("")

    # All-experiments summary
    diag.append("Full Experiment Pipeline Summary")
    diag.append("-" * 70)
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    for st, cnt in sorted(status_counts.items()):
        diag.append(f"  {st:<15} {cnt:>4}")
    diag.append(f"  {'TOTAL':<15} {len(results):>4}")
    diag.append("")

    # Top experiment results (sorted by delta)
    sorted_results = sorted(results, key=lambda r: r.delta_pct, reverse=True)
    diag.append("Top 20 Experiment Results (by delta)")
    diag.append(f"{'Experiment':<45} {'Strategy':>10} {'Delta':>8} {'Status':>15}")
    diag.append("-" * 80)
    for r in sorted_results[:20]:
        diag.append(f"{r.experiment_id:<45} {r.strategy:>10} {r.delta_pct:>+7.2%} {r.status:>15}")
    diag.append("")

    # ================================================================
    # V3 Comparison
    # ================================================================
    diag.append("=" * 70)
    diag.append("COMPARISON vs portfolio_optimized_v3")
    diag.append("=" * 70)

    try:
        v3 = parse_v3(V3_PATH)

        v3_eq = v3.get("final_equity", 0)
        v3_ret = v3.get("total_return_pct", 0)
        v3_pnl = v3.get("total_pnl", 0)
        v3_dd = v3.get("max_dd_pct", 0)
        v3_sha = v3.get("sharpe", 0)
        v3_tr = v3.get("total_trades", 0)
        opt_ret = total_pnl / EQUITY * 100

        diag.append(f"{'Metric':<25} {'V3':>15} {'Greedy Opt':>15} {'Delta':>12}")
        diag.append("-" * 70)
        diag.append(f"{'Final Equity':<25} ${v3_eq:>13,.2f} ${final_eq:>13,.2f} ${final_eq - v3_eq:>+10,.2f}")
        diag.append(f"{'Total Return':<25} {v3_ret:>14.2f}% {opt_ret:>14.2f}% {opt_ret - v3_ret:>+11.2f}%")
        diag.append(f"{'Total PnL':<25} ${v3_pnl:>13,.2f} ${total_pnl:>13,.2f} ${total_pnl - v3_pnl:>+10,.2f}")
        diag.append(f"{'Max Drawdown':<25} {v3_dd:>14.2f}% {max_dd_pct:>14.2f}% {max_dd_pct - v3_dd:>+11.2f}%")
        diag.append(f"{'Sharpe Ratio':<25} {v3_sha:>15.2f} {sharpe:>15.2f} {sharpe - v3_sha:>+12.2f}")
        diag.append(f"{'Total Trades':<25} {v3_tr:>15} {total_trades:>15} {total_trades - v3_tr:>+12}")

        # Per-strategy comparison
        diag.append("")
        diag.append("Per-Strategy Comparison")
        diag.append(f"{'Strategy':<25} {'V3 Tr':>6} {'Opt Tr':>6} {'V3 PnL':>12} {'Opt PnL':>12} {'Delta PnL':>12}")
        diag.append("-" * 75)
        for name, cnt, wr, pnl, tot_r in strategies_data:
            v3s = v3.get("strategies", {}).get(name, {})
            v3t = v3s.get("trades", 0)
            v3p = v3s.get("pnl", 0)
            diag.append(
                f"{name:<25} {v3t:>6} {cnt:>6} "
                f"${v3p:>+10,.2f} ${pnl:>+10,.2f} ${pnl - v3p:>+10,.2f}"
            )
        if overlay_pnl or True:
            v3_overlay = 17395.56  # from V3 file
            diag.append(
                f"{'OVERLAY':<25} {'':>6} {'':>6} "
                f"${v3_overlay:>+10,.2f} ${overlay_pnl:>+10,.2f} ${overlay_pnl - v3_overlay:>+10,.2f}"
            )
    except Exception as e:
        diag.append(f"Could not parse V3 file: {e}")
        traceback.print_exc()

    diag.append("")
    total_time = time.time() - overall_t0
    diag.append(f"Pipeline Timing:")
    diag.append(f"  Phase 1 (experiments): {phase1_time:.0f}s ({phase1_time / 60:.1f}m)")
    diag.append(f"  Phase 3 (greedy):      {phase3_time:.0f}s ({phase3_time / 60:.1f}m)")
    diag.append(f"  Total:                 {total_time:.0f}s ({total_time / 60:.1f}m)")
    diag.append("=" * 70)

    diagnostics_text = "\n".join(diag)
    print(diagnostics_text, flush=True)

    # Save diagnostics
    diag_path = OUTPUT_DIR / "greedy_optimal_diagnostics.txt"
    with open(diag_path, "w") as f:
        f.write(diagnostics_text)
    print(f"\nDiagnostics saved to: {diag_path}", flush=True)

    # Save experiment summary
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    summary = {
        "total_experiments": len(results),
        "positive_impact": len(positive),
        "status_counts": status_counts,
        "top_experiments": [
            {"id": r.experiment_id, "strategy": r.strategy,
             "delta_pct": r.delta_pct, "status": r.status}
            for r in sorted_results[:30]
        ],
        "greedy_result": {
            "base_score": result.base_score,
            "final_score": result.final_score,
            "kept_features": result.kept_features,
            "final_mutations": result.final_mutations,
            "rounds": len(result.rounds),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Experiment summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
