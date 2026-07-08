"""Standalone runner for IARIC T3 pullback diagnostics.

Usage::

    python -m backtests.stock.analysis.run_pullback_diagnostics
    python -m backtests.stock.analysis.run_pullback_diagnostics --report-file output.txt
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="IARIC T3 Pullback Diagnostics")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--data-dir", default="backtests/stock/data/raw")
    parser.add_argument("--execution-mode", default="intraday_hybrid", choices=["daily", "intraday_hybrid"])
    parser.add_argument("--report-file", type=str, default=None)
    args = parser.parse_args()

    from backtests.stock.analysis.iaric_pullback_diagnostics import (
        pullback_full_diagnostic,
    )
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    config = IARICBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=3,
        data_dir=data_dir,
        param_overrides={"pb_execution_mode": args.execution_mode},
    )

    print(f"Running IARIC Tier 3 pullback backtest ({args.execution_mode})...")
    engine = IARICPullbackEngine(config, replay, collect_diagnostics=True)
    result = engine.run()
    print(f"Completed: {len(result.trades)} trades\n")

    diag = pullback_full_diagnostic(
        result.trades,
        replay=replay,
        daily_selections=result.daily_selections,
        candidate_ledger=result.candidate_ledger,
        funnel_counters=result.funnel_counters,
        rejection_log=result.rejection_log,
        shadow_outcomes=result.shadow_outcomes,
        selection_attribution=result.selection_attribution,
        fsm_log=result.fsm_log,
    )
    print(diag)

    if args.report_file:
        Path(args.report_file).write_text(diag, encoding="utf-8")
        print(f"\nDiagnostics saved to {args.report_file}")


if __name__ == "__main__":
    main()
