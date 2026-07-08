"""IARIC Pullback V2 Baseline -- Establish initial performance with default V2 parameters.

Runs two baselines:
  1. Daily engine  (pb_execution_mode="daily")        -- signal model quality
  2. Hybrid engine (pb_execution_mode="intraday_hybrid") -- full 4-route system

Both use pb_v2_enabled=True with all V2 defaults from StrategySettings.
Full diagnostics saved to output directory.

Usage::

    python -m backtests.stock.auto.runners.run_v2_baseline
    python -m backtests.stock.auto.runners.run_v2_baseline --mode daily
    python -m backtests.stock.auto.runners.run_v2_baseline --mode hybrid
    python -m backtests.stock.auto.runners.run_v2_baseline --start-date 2024-01-01 --end-date 2026-03-01
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Defaults ──────────────────────────────────────────────────���───────────────

DATA_DIR = Path("backtests/stock/data/raw")
DEFAULT_OUTPUT_DIR = Path("backtests/stock/auto/iaric/output_v2_baseline")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0


# ── V2 Baseline Mutations ────────────────────────────────────────────────────

V2_COMMON: dict[str, object] = {
    # Master switch
    "param_overrides.pb_v2_enabled": True,
    # F75 baseline defaults (validated via daily parameter sweep)
    "param_overrides.pb_v2_signal_floor": 75.0,
    "param_overrides.pb_v2_flow_grace_days": 2,
    # Infrastructure -- keep legacy routes enabled so V2 can use them
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_carry_enabled": True,
    # Wider funnel -- let V2 scoring handle quality
    "param_overrides.pb_min_candidates_day": 12,
    "param_overrides.pb_entry_rank_max": 999,
    "param_overrides.pb_entry_rank_pct_max": 100.0,
    "param_overrides.pb_min_candidates_day_hard_gate": False,
    "param_overrides.pb_backtest_intraday_universe_only": True,
}

V2_DAILY_MUTATIONS: dict[str, object] = {
    **V2_COMMON,
    "param_overrides.pb_execution_mode": "daily",
}

V2_HYBRID_MUTATIONS: dict[str, object] = {
    **V2_COMMON,
    "param_overrides.pb_execution_mode": "intraday_hybrid",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_baseline(
    mutations: dict[str, object],
    label: str,
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    """Run a single baseline and return metrics + diagnostics."""
    from backtests.stock.analysis.iaric_pullback_diagnostics import (
        compute_pullback_diagnostic_snapshot,
        pullback_full_diagnostic,
    )
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.auto.iaric.phase_scoring import (
        enrich_phase_score_metrics,
        merge_pullback_metrics,
    )
    from backtests.stock.auto.scoring import extract_metrics
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    print(f"\n{'=' * 72}")
    print(f"  V2 Baseline: {label}")
    print(f"  Date range: {start_date} -> {end_date}")
    print(f"  Equity: ${INITIAL_EQUITY:,.0f}")
    print(f"{'=' * 72}\n")

    # Build config with mutations
    base_config = IARICBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=INITIAL_EQUITY,
        tier=3,
        data_dir=DATA_DIR,
    )
    config = mutate_iaric_config(base_config, mutations)

    # Load data
    t0 = _time.monotonic()
    replay = ResearchReplayEngine(data_dir=DATA_DIR)
    replay.load_all_data()
    print(f"  Data loaded in {_time.monotonic() - t0:.1f}s")

    # Run engine
    t0 = _time.monotonic()
    engine = IARICPullbackEngine(config, replay, collect_diagnostics=True)
    result = engine.run()
    elapsed = _time.monotonic() - t0
    print(f"  Engine completed in {elapsed:.1f}s -- {len(result.trades)} trades")

    # Extract metrics
    perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, INITIAL_EQUITY)
    metrics = enrich_phase_score_metrics(
        merge_pullback_metrics(
            perf,
            result.trades,
            candidate_ledger=result.candidate_ledger,
            selection_attribution=result.selection_attribution,
        )
    )

    # Print key metrics
    print(f"\n  ── Key Metrics ──")
    print(f"  Trades:         {metrics.get('total_trades', 0):.0f}")
    print(f"  Net profit:     ${metrics.get('net_profit', 0):,.2f}")
    print(f"  Profit factor:  {metrics.get('profit_factor', 0):.2f}")
    print(f"  Avg R:          {metrics.get('avg_r', 0):.3f}")
    print(f"  Win rate:       {metrics.get('win_rate', 0):.1%}")
    print(f"  Max DD:         {metrics.get('max_drawdown_pct', 0):.2%}")
    print(f"  Sharpe:         {metrics.get('sharpe', 0):.2f}")
    print(f"  Expected TR:    {metrics.get('expected_total_r', 0):.1f}")

    # Full diagnostics
    diag_text = pullback_full_diagnostic(
        result.trades,
        replay=replay,
        daily_selections=result.daily_selections,
        candidate_ledger=result.candidate_ledger,
        funnel_counters=getattr(result, "funnel_counters", None),
        rejection_log=getattr(result, "rejection_log", None),
        shadow_outcomes=getattr(result, "shadow_outcomes", None),
        selection_attribution=result.selection_attribution,
        fsm_log=getattr(result, "fsm_log", None),
    )

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.lower().replace(" ", "_")

    (output_dir / f"{safe_label}_diagnostics.txt").write_text(
        diag_text, encoding="utf-8"
    )
    (output_dir / f"{safe_label}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / f"{safe_label}_mutations.json").write_text(
        json.dumps(mutations, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Save trade-level detail for analysis
    trade_records = []
    for t in result.trades:
        rec = {
            "symbol": t.symbol,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "r_multiple": t.r_multiple,
            "pnl": t.pnl,
            "exit_reason": t.exit_reason,
            "hold_bars": t.hold_bars,
            "sector": t.sector,
        }
        if t.metadata:
            for k in ("route_family", "trigger_types", "trigger_tier", "trend_tier",
                       "daily_signal_score", "v2_sizing_mult", "mfe_stage"):
                if k in t.metadata:
                    rec[k] = t.metadata[k]
        trade_records.append(rec)

    (output_dir / f"{safe_label}_trades.json").write_text(
        json.dumps(trade_records, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Diagnostic snapshot for programmatic analysis
    snap = compute_pullback_diagnostic_snapshot(
        result.trades,
        metrics=metrics,
        replay=replay,
        daily_selections=result.daily_selections,
        candidate_ledger=result.candidate_ledger,
    )
    (output_dir / f"{safe_label}_snapshot.json").write_text(
        json.dumps(snap, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\n  Artifacts saved to {output_dir}/")
    print(f"    - {safe_label}_diagnostics.txt")
    print(f"    - {safe_label}_metrics.json")
    print(f"    - {safe_label}_trades.json")
    print(f"    - {safe_label}_snapshot.json")

    return {
        "label": label,
        "metrics": metrics,
        "diagnostics": diag_text,
        "trades": result.trades,
    }


def _print_comparison(daily_metrics: dict, hybrid_metrics: dict | None) -> None:
    """Print side-by-side comparison of daily vs hybrid baselines."""
    print(f"\n{'=' * 72}")
    print("  V2 BASELINE COMPARISON")
    print(f"{'=' * 72}")

    headers = ["Metric", "Daily", "Hybrid"] if hybrid_metrics else ["Metric", "Daily"]
    rows = [
        ("Trades", "total_trades", ".0f"),
        ("Net Profit", "net_profit", ",.2f"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Avg R", "avg_r", ".3f"),
        ("Win Rate", "win_rate", ".1%"),
        ("Max DD", "max_drawdown_pct", ".2%"),
        ("Sharpe", "sharpe", ".2f"),
        ("Expected TR", "expected_total_r", ".1f"),
        ("Managed Exit %", "managed_exit_share", ".1%"),
        ("EOD Flatten %", "eod_flatten_share", ".1%"),
    ]

    col_w = 18
    header_line = f"  {'Metric':<22}" + "".join(f"{h:>{col_w}}" for h in headers[1:])
    print(header_line)
    print(f"  {'-' * 22}" + "-" * col_w * (len(headers) - 1))

    for label, key, fmt in rows:
        d_val = daily_metrics.get(key, 0)
        line = f"  {label:<22}{d_val:{col_w}{fmt}}"
        if hybrid_metrics:
            h_val = hybrid_metrics.get(key, 0)
            line += f"{h_val:{col_w}{fmt}}"
        print(line)

    # Plan targets
    print(f"\n  ── Plan Targets ──")
    print(f"  Trades:       150-250")
    print(f"  Avg R:        >= 0.30")
    print(f"  PF:           >= 2.00")
    print(f"  EOD Flatten:  < 10%")
    print(f"  Managed Exit: > 50%")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="both", choices=["daily", "hybrid", "both"],
                        help="Which engine(s) to run")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    # Write run manifest
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "round": "V2_baseline",
        "mode": args.mode,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_equity": INITIAL_EQUITY,
        "v2_common_mutations": V2_COMMON,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    daily_result = None
    hybrid_result = None

    if args.mode in ("daily", "both"):
        daily_result = _run_baseline(
            V2_DAILY_MUTATIONS,
            "V2 Daily",
            output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    if args.mode in ("hybrid", "both"):
        hybrid_result = _run_baseline(
            V2_HYBRID_MUTATIONS,
            "V2 Hybrid",
            output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    # Comparison
    if daily_result:
        _print_comparison(
            daily_result["metrics"],
            hybrid_result["metrics"] if hybrid_result else None,
        )

    print(f"\n  All artifacts saved to: {output_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()
