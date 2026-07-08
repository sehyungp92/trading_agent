"""CLI interface for the stock backtest framework.

Usage::

    python -m backtests.stock download --timeframes 1d
    python -m backtests.stock run --strategy alcb --start 2024-01-01
    python -m backtests.stock run --strategy iaric --tier 3
    python -m backtests.stock portfolio
    python -m backtests.stock auto --strategy all
    python -m backtests.stock auto --strategy alcb --skip-robustness
    python -m backtests.stock auto --experiments abl_alcb_stale_exit abl_alcb_regime_gate
    python -m backtests.stock auto --resume
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows to avoid cp949 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace) -> None:
    """Download historical bar data."""
    from backtests.stock.data.downloader import download_stock_universe

    timeframes = args.timeframes.split(",") if args.timeframes else ["1d"]
    output_dir = Path(args.data_dir)

    async def _run():
        return await download_stock_universe(
            timeframes=timeframes,
            duration=args.duration,
            output_dir=output_dir,
            host=args.host,
            port=args.port,
            skip_existing=not args.force,
        )

    result = asyncio.run(_run())
    for tf, symbols in result.items():
        print(f"  {tf}: {len(symbols)} symbols downloaded")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single-strategy backtest."""
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    if args.strategy == "alcb":
        _run_alcb(args, replay)
    elif args.strategy == "iaric":
        _run_iaric(args, replay)
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        sys.exit(1)


def _run_alcb(args: argparse.Namespace, replay) -> None:
    from backtests.stock.analysis.reports import full_report
    from backtests.stock.config_alcb import ALCBBacktestConfig

    # Parse --param key=value overrides
    param_overrides: dict = {}
    for p in getattr(args, "param", []):
        k, _, v = p.partition("=")
        if not v:
            print(f"Invalid --param format: {p} (expected key=value)", file=sys.stderr)
            sys.exit(1)
        # Auto-cast numeric values
        try:
            v_parsed: object = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                v_parsed = v
        param_overrides[k] = v_parsed

    config = ALCBBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=args.tier,
        data_dir=Path(args.data_dir),
        verbose=args.verbose,
        param_overrides=param_overrides,
    )

    # Shadow tracker (Tier 2 only)
    shadow_tracker = None
    if getattr(args, "shadow", False) and args.tier == 2:
        from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
        shadow_tracker = ALCBShadowTracker()

    from backtests.stock.engine.alcb_engine import ALCBIntradayEngine

    engine = ALCBIntradayEngine(config, replay)
    if shadow_tracker:
        engine.shadow_tracker = shadow_tracker
    result = engine.run()
    report = full_report(
        result.trades, result.equity_curve, result.timestamps,
        config.initial_equity, strategy="ALCB Intraday",
        daily_selections=result.daily_selections,
    )
    print(report)

    # Deep diagnostics
    if getattr(args, "diagnostics", False):
        from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic

        diag = alcb_full_diagnostic(
            result.trades,
            shadow_tracker=shadow_tracker,
            daily_selections=result.daily_selections,
        )
        if args.report_file:
            Path(args.report_file).write_text(diag, encoding="utf-8")
            print(f"  Diagnostics saved to {args.report_file}")
        print(diag)

    if args.save_charts:
        _save_charts(result, f"ALCB_Tier{args.tier}", args.output_dir)


def _run_iaric(args: argparse.Namespace, replay) -> None:
    from backtests.stock.analysis.reports import full_report
    from backtests.stock.config_iaric import IARICBacktestConfig

    # Parse --param key=value overrides
    param_overrides: dict = {}
    for p in getattr(args, "param", []):
        k, _, v = p.partition("=")
        if not v:
            print(f"Invalid --param format: {p} (expected key=value)", file=sys.stderr)
            sys.exit(1)
        try:
            v_parsed: object = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                v_parsed = v
        param_overrides[k] = v_parsed

    config = IARICBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=args.tier,
        data_dir=Path(args.data_dir),
        verbose=args.verbose,
        param_overrides=param_overrides,
    )

    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

    engine = IARICPullbackEngine(config, replay)
    result = engine.run()
    report = full_report(
        result.trades, result.equity_curve, result.timestamps,
        config.initial_equity, strategy="IARIC Pullback",
        daily_selections=result.daily_selections,
    )
    print(report)

    # Deep diagnostics
    if getattr(args, "diagnostics", False):
        from backtests.stock.analysis.iaric_pullback_diagnostics import (
            pullback_full_diagnostic,
        )

        diag = pullback_full_diagnostic(
            result.trades,
            replay=replay,
            daily_selections=result.daily_selections,
            candidate_ledger=getattr(result, "candidate_ledger", None),
            funnel_counters=getattr(result, "funnel_counters", None),
            rejection_log=getattr(result, "rejection_log", None),
            shadow_outcomes=getattr(result, "shadow_outcomes", None),
            selection_attribution=getattr(result, "selection_attribution", None),
            fsm_log=getattr(result, "fsm_log", None),
        )
        print(diag)
        if args.report_file:
            Path(args.report_file).write_text(diag, encoding="utf-8")
            print(f"  Diagnostics saved to {args.report_file}")

    if args.save_charts:
        _save_charts(result, f"IARIC_Tier{args.tier}", args.output_dir)


def _save_charts(result, prefix: str, output_dir: str) -> None:
    from backtests.stock.analysis.charts import (
        plot_equity_curve,
        plot_monthly_returns,
        plot_sector_attribution,
        plot_trade_distribution,
    )

    out = Path(output_dir)
    plot_equity_curve(result.equity_curve, result.timestamps, f"{prefix} Equity Curve", out / f"{prefix}_equity.png")
    plot_trade_distribution(result.trades, f"{prefix} Trade Distribution", out / f"{prefix}_distribution.png")
    plot_monthly_returns(result.trades, f"{prefix} Monthly Returns", out / f"{prefix}_monthly.png")
    plot_sector_attribution(result.trades, f"{prefix} Sector Attribution", out / f"{prefix}_sectors.png")
    print(f"  Charts saved to {out}/")


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Run portfolio backtest (both strategies)."""
    from backtests.stock.analysis.reports import full_report
    from backtests.stock.config_portfolio import PortfolioBacktestConfig
    from backtests.stock.engine.portfolio_engine import StockPortfolioEngine
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    pf_config = PortfolioBacktestConfig(
        data_dir=data_dir,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        tier=args.tier,
        verbose=args.verbose,
    )
    alcb_config, iaric_config = pf_config.build_strategy_configs()

    # Run individual strategies
    print("Running ALCB...")
    from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
    alcb_result = ALCBIntradayEngine(alcb_config, replay).run()

    print("Running IARIC...")
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine
    iaric_result = IARICPullbackEngine(iaric_config, replay).run()

    # Portfolio merge
    print("Merging with portfolio rules...")
    engine = StockPortfolioEngine(pf_config)
    pf_result = engine.run(alcb_result.trades, iaric_result.trades)

    # Reports
    print("\n" + "=" * 60)
    print("ALCB individual:")
    print(full_report(
        alcb_result.trades, alcb_result.equity_curve, alcb_result.timestamps,
        pf_config.initial_equity, strategy=f"ALCB Tier {args.tier}",
    ))
    print("\n" + "=" * 60)
    print("IARIC individual:")
    print(full_report(
        iaric_result.trades, iaric_result.equity_curve, iaric_result.timestamps,
        pf_config.initial_equity, strategy=f"IARIC Tier {args.tier}",
    ))
    print("\n" + "=" * 60)
    print("Portfolio combined:")
    print(full_report(
        pf_result.trades, pf_result.equity_curve, pf_result.timestamps,
        pf_config.initial_equity, strategy="Stock Family Portfolio",
    ))
    print(f"  Blocked trades: {len(pf_result.blocked_trades)}")


def cmd_auto(args: argparse.Namespace) -> None:
    """Run automated experiment harness."""
    from backtests.stock.auto.harness import AutoBacktestHarness

    harness = AutoBacktestHarness(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        verbose=args.verbose,
    )

    harness.run_all(
        strategy_filter=args.strategy,
        experiment_ids=args.experiments,
        skip_robustness=args.skip_robustness,
        resume=args.resume,
    )


def cmd_greedy(args: argparse.Namespace) -> None:
    """Run greedy forward selection for optimal config."""
    from backtests.stock.auto.greedy_optimize import (
        run_greedy,
        save_result,
    )
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    data_dir = Path(args.data_dir)
    replay = ResearchReplayEngine(data_dir=data_dir)
    print("Loading bar data...")
    replay.load_all_data()

    if args.strategy == "iaric":
        from backtests.stock.auto.greedy_optimize import (
            IARIC_T3_P7_BASE_MUTATIONS,
            IARIC_T3_P7_CANDIDATES,
        )
        base_mutations = IARIC_T3_P7_BASE_MUTATIONS
        candidates = IARIC_T3_P7_CANDIDATES
        tier = 3
    else:
        print(f"Greedy selection not yet configured for {args.strategy}", file=sys.stderr)
        print("  (ALCB phased optimization now lives under backtests.stock.auto.alcb)", file=sys.stderr)
        sys.exit(1)

    result = run_greedy(
        replay=replay,
        strategy=args.strategy,
        tier=tier,
        base_mutations=base_mutations,
        candidates=candidates,
        initial_equity=args.equity,
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data_dir,
    )

    output_path = Path(args.output_dir) / f"greedy_optimal_{args.strategy}_t{tier}.json"
    save_result(result, output_path)


def _build_pullback_phase_runner(
    args: argparse.Namespace,
    *,
    profile: str | None = None,
    output_dir: Path | None = None,
):
    from backtests.shared.auto.phase_runner import PhaseRunner
    from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

    resolved_profile = str(profile or getattr(args, "profile", "mainline")).lower()
    plugin = IARICPullbackPlugin(
        data_dir=Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        profile=resolved_profile,
    )
    return PhaseRunner(
        plugin=plugin,
        output_dir=Path(output_dir) if output_dir is not None else Path(args.output_dir),
        round_name=f"iaric_{resolved_profile}",
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 0),
    )


def _split_dual_track_workers(total_workers: int | None) -> tuple[int | None, int | None]:
    if total_workers is None:
        return None, None
    workers = max(int(total_workers), 1)
    if workers <= 1:
        return 1, 1
    mainline = max(1, (workers + 1) // 2)
    aggressive = max(1, workers // 2)
    return mainline, aggressive


def _phase_auto_command(
    args: argparse.Namespace,
    *,
    profile: str,
    output_dir: Path,
    max_workers: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "backtests.stock.cli",
        "phase-auto",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--equity",
        str(args.equity),
        "--profile",
        str(profile),
        "--max-rounds",
        str(getattr(args, "max_rounds", 24)),
        "--min-delta",
        str(getattr(args, "min_delta", 0.001)),
        "--max-retries",
        str(getattr(args, "max_retries", 0)),
    ]
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    return command


def _phase_mutations_through(state, phase: int) -> dict:
    phase_result = state.phase_results.get(phase, {})
    if phase_result.get("final_mutations"):
        return dict(phase_result["final_mutations"])

    mutations: dict = {}
    for phase_num in sorted(state.phase_results):
        if phase_num > phase:
            break
        mutations.update(state.phase_results[phase_num].get("new_mutations", {}))
    return mutations


def cmd_phase_run(args: argparse.Namespace) -> None:
    runner = _build_pullback_phase_runner(args)
    state = runner.load_state()
    missing = [phase for phase in range(1, args.phase) if phase not in state.completed_phases]
    if missing:
        print(
            f"Cannot run phase {args.phase} yet. Missing earlier phases: {missing}. "
            "Run phase-auto or complete prior phases first.",
            file=sys.stderr,
        )
        sys.exit(1)

    state = runner.run_phase(args.phase, state)
    result = state.phase_results.get(args.phase, {})
    print(f"IARIC pullback phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_pullback_phase_runner(args)
    state = runner.run_all_phases()
    print(f"IARIC pullback phased auto-optimization complete ({runner.plugin.profile}).")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")

def cmd_phase_auto_dual(args: argparse.Namespace) -> None:
    from backtests.stock.auto.iaric.plugin import select_pullback_branch

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    mainline_output = output_root / "mainline"
    aggressive_output = output_root / "aggressive"
    mainline_workers, aggressive_workers = _split_dual_track_workers(getattr(args, "max_workers", None))

    launches = {
        "mainline": {
            "command": _phase_auto_command(
                args,
                profile="mainline",
                output_dir=mainline_output,
                max_workers=mainline_workers,
            ),
            "stdout_path": log_dir / "mainline.stdout.log",
            "stderr_path": log_dir / "mainline.stderr.log",
        },
        "aggressive": {
            "command": _phase_auto_command(
                args,
                profile="aggressive",
                output_dir=aggressive_output,
                max_workers=aggressive_workers,
            ),
            "stdout_path": log_dir / "aggressive.stdout.log",
            "stderr_path": log_dir / "aggressive.stderr.log",
        },
    }

    processes: dict[str, subprocess.Popen] = {}
    handles = []
    try:
        for profile, launch in launches.items():
            stdout_handle = open(launch["stdout_path"], "w", encoding="utf-8")
            stderr_handle = open(launch["stderr_path"], "w", encoding="utf-8")
            handles.extend([stdout_handle, stderr_handle])
            processes[profile] = subprocess.Popen(
                launch["command"],
                cwd=str(Path.cwd()),
                stdout=stdout_handle,
                stderr=stderr_handle,
            )

        while True:
            statuses = {profile: proc.poll() for profile, proc in processes.items()}
            if all(code is not None for code in statuses.values()):
                break
            failed = {profile: code for profile, code in statuses.items() if code not in (None, 0)}
            if failed:
                for profile, proc in processes.items():
                    if statuses[profile] is None:
                        proc.terminate()
                break
            time.sleep(2.0)
    finally:
        for handle in handles:
            handle.close()

    exit_codes = {profile: proc.wait() for profile, proc in processes.items()}
    failed = {profile: code for profile, code in exit_codes.items() if code != 0}
    if failed:
        for profile, code in failed.items():
            print(
                f"{profile} branch failed with exit code {code}. "
                f"See {launches[profile]['stdout_path']} and {launches[profile]['stderr_path']}.",
                file=sys.stderr,
            )
        sys.exit(next(iter(failed.values())))

    mainline_runner = _build_pullback_phase_runner(args, profile="mainline", output_dir=mainline_output)
    aggressive_runner = _build_pullback_phase_runner(args, profile="aggressive", output_dir=aggressive_output)
    mainline_state = mainline_runner.load_state()
    aggressive_state = aggressive_runner.load_state()
    mainline_metrics = mainline_runner.plugin.compute_final_metrics(mainline_state.cumulative_mutations)
    aggressive_metrics = aggressive_runner.plugin.compute_final_metrics(aggressive_state.cumulative_mutations)
    selection = select_pullback_branch(mainline_metrics, aggressive_metrics)

    summary_lines = [
        "IARIC pullback dual-track summary",
        f"mainline: avg_r={mainline_metrics['avg_r']:+.3f}, trades={int(mainline_metrics['total_trades'])}, pf={mainline_metrics['profit_factor']:.2f}, dd={mainline_metrics['max_drawdown_pct']:.1%}, exp_total_r={mainline_metrics['expected_total_r']:+.2f}",
        f"aggressive: avg_r={aggressive_metrics['avg_r']:+.3f}, trades={int(aggressive_metrics['total_trades'])}, pf={aggressive_metrics['profit_factor']:.2f}, dd={aggressive_metrics['max_drawdown_pct']:.1%}, exp_total_r={aggressive_metrics['expected_total_r']:+.2f}",
        f"selected: {selection['selected_profile']}",
        f"reason: {selection['reason']}",
        f"mainline_logs: {launches['mainline']['stdout_path']} | {launches['mainline']['stderr_path']}",
        f"aggressive_logs: {launches['aggressive']['stdout_path']} | {launches['aggressive']['stderr_path']}",
    ]
    summary_path = output_root / "dual_track_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("IARIC pullback dual-track auto-optimization complete.")
    print(f"Selected profile: {selection['selected_profile']}")
    print(selection["reason"])
    print(f"Summary: {summary_path}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    from backtests.shared.auto.phase_gates import evaluate_gate
    from backtests.shared.auto.phase_state import save_phase_state

    runner = _build_pullback_phase_runner(args)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Phase {args.phase} has not been completed yet.", file=sys.stderr)
        sys.exit(1)

    phase_mutations = _phase_mutations_through(state, args.phase)
    metrics = runner.plugin.compute_final_metrics(phase_mutations)
    spec = runner.plugin.get_phase_spec(args.phase, state)
    gate = evaluate_gate(spec.gate_criteria_fn(metrics))
    state.record_gate(args.phase, {
        "passed": gate.passed,
        "criteria": [criterion.__dict__ for criterion in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    })
    save_phase_state(state, runner.state_path)

    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for criterion in gate.criteria:
        marker = "[PASS]" if criterion.passed else "[FAIL]"
        print(f"  {marker} {criterion.name}: {criterion.actual:.4f} (target {criterion.target:.4f})")
    if not gate.passed:
        print(f"Failure category: {gate.failure_category}")
        for recommendation in gate.recommendations:
            print(f"  - {recommendation}")


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    diag_path = Path(args.output_dir) / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.", file=sys.stderr)
        sys.exit(1)
    print(diag_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-backtest",
        description="Stock family backtesting framework (ALCB + IARIC)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download historical bar data")
    dl.add_argument("--timeframes", default="1d", help="Comma-separated timeframes (default: 1d)")
    dl.add_argument("--duration", default="5 Y", help="IBKR duration string")
    dl.add_argument("--data-dir", default="backtests/stock/data/raw")
    dl.add_argument("--host", default="127.0.0.1")
    dl.add_argument("--port", type=int, default=7496)
    dl.add_argument("--force", action="store_true", help="Re-download existing files")

    # run
    run = sub.add_parser("run", help="Run a backtest")
    run.add_argument("--strategy", choices=["alcb", "iaric"], required=True)
    run.add_argument("--tier", type=int, default=2, choices=[2, 3])
    run.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    run.add_argument("--end", default="2026-03-01", help="End date (YYYY-MM-DD)")
    run.add_argument("--equity", type=float, default=10_000.0)
    run.add_argument("--data-dir", default="backtests/stock/data/raw")
    run.add_argument("--save-charts", action="store_true", help="Save chart PNGs")
    run.add_argument("--output-dir", default="backtests/stock/output")
    run.add_argument("--diagnostics", action="store_true",
                     help="Run deep 27-section diagnostics report")
    run.add_argument("--shadow", action="store_true",
                     help="Enable shadow tracker for rejected setup analysis")
    run.add_argument("--report-file", type=str, default=None,
                     help="Write diagnostics report to file")
    run.add_argument("--param", action="append", default=[],
                     help="Override param: key=value (e.g. --param opening_range_bars=3)")

    # portfolio
    pf = sub.add_parser("portfolio", help="Run portfolio backtest (both strategies)")
    pf.add_argument("--tier", type=int, default=2)
    pf.add_argument("--start", default="2024-01-01")
    pf.add_argument("--end", default="2026-03-01")
    pf.add_argument("--equity", type=float, default=10_000.0)
    pf.add_argument("--data-dir", default="backtests/stock/data/raw")

    # auto
    auto = sub.add_parser("auto", help="Run automated experiment harness")
    auto.add_argument("--strategy", choices=["alcb", "iaric", "all"], default="all")
    auto.add_argument("--experiments", nargs="*", help="Specific experiment IDs to run")
    auto.add_argument("--skip-robustness", action="store_true",
                       help="Skip robustness checks for fast ablation scan")
    auto.add_argument("--resume", action="store_true", help="Skip completed experiments")
    auto.add_argument("--output-dir", default="backtests/stock/auto/output")
    auto.add_argument("--start", default="2024-01-01")
    auto.add_argument("--end", default="2026-03-01")
    auto.add_argument("--equity", type=float, default=10_000.0)
    auto.add_argument("--data-dir", default="backtests/stock/data/raw")

    # greedy
    gr = sub.add_parser("greedy", help="Greedy forward selection for optimal config")
    gr.add_argument("--strategy", choices=["alcb", "iaric"], required=True)
    gr.add_argument("--tier", type=int, default=3, choices=[2, 3])
    gr.add_argument("--data-dir", default="backtests/stock/data/raw")
    gr.add_argument("--output-dir", default="backtests/stock/auto/output")
    gr.add_argument("--start", default="2024-01-01")
    gr.add_argument("--end", default="2026-03-01")
    gr.add_argument("--equity", type=float, default=10_000.0)

    def add_phase_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/stock/data/raw")
        command.add_argument("--output-dir", default="backtests/stock/auto/iaric/output")
        command.add_argument("--start", default="2024-01-01")
        command.add_argument("--end", default="2026-03-01")
        command.add_argument("--equity", type=float, default=10_000.0)
        command.add_argument("--profile", choices=["mainline", "aggressive"], default="mainline")

    phase_run = sub.add_parser("phase-run", help="Run a single IARIC Tier 3 pullback phase")
    add_phase_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    phase_run.add_argument("--max-rounds", type=int, default=24)
    phase_run.add_argument("--max-workers", type=int, default=4)
    phase_run.add_argument("--min-delta", type=float, default=0.001)

    phase_auto = sub.add_parser("phase-auto", help="Run all IARIC Tier 3 pullback phases")
    add_phase_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=24)
    phase_auto.add_argument("--max-workers", type=int, default=4)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=0)

    phase_auto_dual = sub.add_parser("phase-auto-dual", help="Run mainline and aggressive pullback branches")
    phase_auto_dual.add_argument("--data-dir", default="backtests/stock/data/raw")
    phase_auto_dual.add_argument("--output-dir", default="backtests/stock/auto/iaric/output_dual")
    phase_auto_dual.add_argument("--start", default="2024-01-01")
    phase_auto_dual.add_argument("--end", default="2026-03-01")
    phase_auto_dual.add_argument("--equity", type=float, default=10_000.0)
    phase_auto_dual.add_argument("--max-rounds", type=int, default=24)
    phase_auto_dual.add_argument("--max-workers", type=int, default=4)
    phase_auto_dual.add_argument("--min-delta", type=float, default=0.001)
    phase_auto_dual.add_argument("--max-retries", type=int, default=0)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed IARIC pullback phase gate")
    add_phase_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])

    phase_diag = sub.add_parser("phase-diagnostics", help="Print saved IARIC pullback phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    phase_diag.add_argument("--output-dir", default="backtests/stock/auto/iaric/output")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "download":
        cmd_download(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "portfolio":
        cmd_portfolio(args)
    elif args.command == "auto":
        cmd_auto(args)
    elif args.command == "greedy":
        cmd_greedy(args)
    elif args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-auto-dual":
        cmd_phase_auto_dual(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
