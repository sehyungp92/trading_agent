"""CLI for regime backtesting and optimization.

Usage:
    python -m backtests.regime.cli download
    python -m backtests.regime.cli seed-refresh
    python -m backtests.regime.cli seed-validate --require-manifest
    python -m backtests.regime.cli run [--diagnostics]
    python -m backtests.regime.cli optimize [--max-rounds N] [--max-workers N]
    python -m backtests.regime.cli walk-forward [--test-years 2]
    python -m backtests.regime.cli phase-run --phase N [--preset recommended_full_stack]
    python -m backtests.regime.cli phase-auto [--preset recommended_full_stack]
    python -m backtests.regime.cli phase-gate --phase N
    python -m backtests.regime.cli phase-diagnostics --phase N
    python -m backtests.regime.cli historical-validate [--preset recommended_full_stack]
    python -m backtests.regime.cli scanner-validate [--preset recommended_full_stack] [--diagnostics]
    python -m backtests.regime.cli calibration-sweep [--preset r3_reference]
    python -m backtests.regime.cli validate-2022 [--preset recommended_full_stack]
    python -m backtests.regime.cli step9-optimize [--max-workers 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure project root on path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.regime.auto.presets import (
    DEFAULT_REFERENCE_PRESET,
    DEFAULT_RESEARCH_PRESET,
    R7_RESEARCH_PRESET,
    R8_RESEARCH_PRESET,
    R9_RESEARCH_PRESET,
    STEP9_RESEARCH_PRESET,
    get_research_preset,
    preset_choices,
)

logger = logging.getLogger(__name__)
DEFAULT_PHASE_OUTPUT_DIR = Path("backtests/regime/auto/output")
DEFAULT_STEP9_OUTPUT_DIR = DEFAULT_PHASE_OUTPUT_DIR / "step9_r6"
DEFAULT_R7_OUTPUT_DIR = DEFAULT_PHASE_OUTPUT_DIR / "r7_overlay_recal"
DEFAULT_R8_OUTPUT_DIR = DEFAULT_PHASE_OUTPUT_DIR / "r8_two_model"
DEFAULT_R9_OUTPUT_DIR = DEFAULT_PHASE_OUTPUT_DIR / "r9_budget"


def _setup_phase_logging(phase: int, output_dir: Path) -> Path:
    """Configure file-based logging for a phase run.

    Writes to ``output_dir/phase_{N}.log`` with timestamps and module names.
    Returns the log file path.
    """
    log_path = output_dir / f"phase_{phase}.log"
    regime_logger = logging.getLogger("backtests.regime")
    regime_logger.setLevel(logging.DEBUG)

    # Remove stale file handlers from previous invocations
    for h in regime_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            regime_logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    regime_logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("Phase %d logging initialized -> %s", phase, log_path)
    logger.info("=" * 70)
    return log_path


def _extract_mutations_dict(data: dict) -> dict:
    """Normalize supported JSON payload shapes into a mutations dict."""
    if "cumulative_mutations" in data:
        return data["cumulative_mutations"]
    if "accepted_mutations" in data:
        return data["accepted_mutations"]
    return data


def _load_mutations_json(path: Path) -> dict:
    """Load a mutations JSON file into a plain dict."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _extract_mutations_dict(data)


def _resolve_mutation_source(
    explicit_path: str | None,
    preset_name: str | None,
    default_preset: str,
) -> tuple[dict, dict[str, str | None]]:
    """Resolve mutations from explicit JSON or a version-controlled preset."""
    requested_preset = preset_name or default_preset
    if explicit_path:
        path = Path(explicit_path)
        return _load_mutations_json(path), {
            "type": "json",
            "label": str(path),
            "preset_name": None,
            "requested_preset": requested_preset,
        }

    return get_research_preset(requested_preset), {
        "type": "preset",
        "label": requested_preset,
        "preset_name": requested_preset,
        "requested_preset": requested_preset,
    }


def _print_mutation_source(label: str, source: dict[str, str | None]) -> None:
    """Print a human-readable config source summary."""
    if source["type"] == "json":
        override = ""
        requested_preset = source.get("requested_preset")
        if requested_preset:
            override = f" (overrides preset '{requested_preset}')"
        print(f"{label}: {source['label']}{override}")
        return

    print(f"{label}: preset '{source['label']}'")


def _resolve_phase_output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory used by phase-oriented commands."""
    raw = getattr(args, "output_dir", None)
    return Path(raw) if raw else DEFAULT_PHASE_OUTPUT_DIR


def _resolve_phase_sequence(args: argparse.Namespace) -> tuple[int, ...]:
    """Return the ordered phase sequence for a phase-auto style command."""
    phases = getattr(args, "phase_sequence", None)
    if phases is None:
        return (1, 2, 3, 4)
    return tuple(int(phase) for phase in phases)


def _resolve_phase_max_rounds(args: argparse.Namespace, phase: int) -> int:
    """Return the max_rounds override for a specific phase, if configured."""
    phase_max_rounds = getattr(args, "phase_max_rounds", None) or {}
    return int(phase_max_rounds.get(phase, args.max_rounds))


def add_candidate_profile_arg(parser: argparse.ArgumentParser, default: str = "default") -> None:
    """Add a candidate-profile selector to a phase-oriented parser."""
    from backtests.regime.auto.phase_candidates import candidate_profile_choices

    parser.add_argument(
        "--candidate-profile",
        choices=candidate_profile_choices(),
        default=default,
        help=f"Candidate profile to use (default: {default})",
    )


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    """Download and cache all data from FRED + yfinance."""
    from backtests.regime.data.downloader import build_all_data

    data_dir = Path(args.data_dir)
    print(f"Downloading data to {data_dir}...")
    macro_df, market_df, strat_ret_df = build_all_data(data_dir)

    print(f"\n=== Data Download Complete ===")
    print(f"\nmacro_df:")
    print(f"  Columns: {list(macro_df.columns)}")
    print(f"  Date range: {macro_df.index.min().date()} to {macro_df.index.max().date()}")
    print(f"  Rows: {len(macro_df):,}")
    print(f"  NaN: {macro_df.isna().sum().to_dict()}")

    print(f"\nmarket_df:")
    print(f"  Columns: {list(market_df.columns)}")
    print(f"  Date range: {market_df.index.min().date()} to {market_df.index.max().date()}")
    print(f"  Rows: {len(market_df):,}")
    print(f"  NaN: {market_df.isna().sum().to_dict()}")

    print(f"\nstrat_ret_df:")
    print(f"  Columns: {list(strat_ret_df.columns)}")
    print(f"  Date range: {strat_ret_df.index.min().date()} to {strat_ret_df.index.max().date()}")
    print(f"  Rows: {len(strat_ret_df):,}")
    print(f"  NaN: {strat_ret_df.isna().sum().to_dict()}")

    print(f"\nFiles saved to: {data_dir}")
    print(f"Manifest saved to: {data_dir / 'regime_seed_manifest.json'}")


def cmd_seed_refresh(args: argparse.Namespace) -> None:
    """Refresh deployment seed parquets and write a manifest."""
    from backtests.regime.data.downloader import (
        build_all_data,
        write_manifest_for_cached_data,
    )
    from regime.seed_manifest import validate_seed_data_dir

    data_dir = Path(args.data_dir)
    if args.manifest_only:
        print(f"Writing manifest for cached seed data in {data_dir}...")
        manifest = write_manifest_for_cached_data(data_dir)
    else:
        print(f"Refreshing regime seed data in {data_dir}...")
        print("Sources: FRED/ALFRED for macro/market data, yfinance for ETF seed prices.")
        build_all_data(data_dir)
        manifest = json.loads(
            (data_dir / "regime_seed_manifest.json").read_text(encoding="utf-8")
        )

    ok, status, _ = validate_seed_data_dir(
        data_dir,
        require_manifest=True,
        validate_hashes=True,
    )
    print("\n=== Regime Seed Refresh ===")
    print(f"  Data dir: {data_dir}")
    print(f"  Manifest: {data_dir / 'regime_seed_manifest.json'}")
    print(f"  data_as_of: {manifest.get('data_as_of')}")
    print(f"  row_counts: {manifest.get('row_counts')}")
    print(f"  validation: {'OK' if ok else 'FAIL'} ({status})")
    if not ok:
        raise SystemExit(1)


def cmd_seed_validate(args: argparse.Namespace) -> None:
    """Validate deployment seed parquets and manifest."""
    from regime.seed_manifest import build_seed_manifest, validate_seed_data_dir

    data_dir = Path(args.data_dir)
    ok, status, manifest = validate_seed_data_dir(
        data_dir,
        require_manifest=args.require_manifest,
        validate_hashes=not args.skip_hashes,
    )
    current = build_seed_manifest(
        data_dir,
        generated_by="backtests.regime.cli.seed-validate",
        source_versions=(manifest or {}).get("source_versions", {}),
    )
    print("\n=== Regime Seed Validation ===")
    print(f"  Data dir: {data_dir}")
    print(f"  Status: {'OK' if ok else 'FAIL'} ({status})")
    print(f"  data_as_of: {current.get('data_as_of')}")
    print(f"  row_counts: {current.get('row_counts')}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single backtest with default or custom MetaConfig."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.scoring import composite_score
    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data
    from backtests.regime.engine.portfolio_sim import (
        simulate_benchmark_60_40,
        simulate_portfolio,
    )

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    # Load mutations if provided
    mutations = {}
    if args.mutations_json:
        with open(args.mutations_json) as f:
            data = json.load(f)
        if "accepted_mutations" in data:
            mutations = data["accepted_mutations"]
        else:
            mutations = data
        print(f"Loaded {len(mutations)} mutations from {args.mutations_json}")

    # Build config
    from backtests.regime.auto.config_mutator import mutate_meta_config
    cfg = mutate_meta_config(MetaConfig(), mutations) if mutations else MetaConfig()

    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    print(f"Running regime backtest...")
    print(f"  Equity: ${args.equity:,.0f}")
    print(f"  Cost: {args.cost_bps} bps")
    print(f"  Data: {strat_ret_df.index.min().date()} to {strat_ret_df.index.max().date()}")

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    # Persist latest RegimeContext for live runtime consumption
    try:
        from regime.persistence import save_regime_context
        from regime.context import RegimeContext
        row = signals.iloc[-1]
        sleeves = [c.replace("w_", "") for c in signals.columns if c.startswith("w_")]
        ctx = RegimeContext(
            regime=row["mode_regime"],
            regime_confidence=float(row["Conf"]),
            stress_level=float(row.get("stress_level", 0.0)),
            stress_onset=bool(row.get("stress_onset", False)),
            shift_velocity=float(row.get("stress_velocity", row.get("shift_velocity", 0.0))),
            suggested_leverage_mult=float(row["L"]),
            regime_allocations={s: float(row[f"w_{s}"]) for s in sleeves},
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        save_regime_context(ctx)
        print(f"Regime context saved: {ctx.regime} (confidence={ctx.regime_confidence:.3f})")
    except Exception as exc:
        print(f"Warning: Failed to save regime context (non-fatal): {exc}")

    result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
    score = composite_score(result.metrics)

    _print_metrics(result, score)

    if args.diagnostics:
        from backtests.regime.analysis.diagnostics import (
            generate_regime_diagnostics_report,
        )

        benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)
        report = generate_regime_diagnostics_report(
            signals=signals,
            result=result,
            benchmark=benchmark,
            score=score,
            sim_cfg=sim_cfg,
        )
        print(report.encode("ascii", errors="replace").decode("ascii"))

        output_dir = Path("backtests/regime/auto/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "diagnostics_latest.txt"
        output_path.write_text(report, encoding="utf-8")
        print(f"\nDiagnostics saved to {output_path}")


def _print_metrics(result, score) -> None:
    """Print portfolio metrics and composite score."""
    m = result.metrics
    print(f"\n=== Portfolio Metrics ===")
    print(f"  Total return:     {m.total_return:>10.1%}")
    print(f"  CAGR:             {m.cagr:>10.2%}")
    print(f"  Sharpe:           {m.sharpe:>10.3f}")
    print(f"  Sortino:          {m.sortino:>10.3f}")
    print(f"  Calmar:           {m.calmar:>10.3f}")
    print(f"  Max drawdown:     {m.max_drawdown_pct:>10.1%}")
    print(f"  Max DD duration:  {m.max_drawdown_duration:>10d} days")
    print(f"  Avg annual TO:    {m.avg_annual_turnover:>10.2f}")
    print(f"  Rebalances:       {m.n_rebalances:>10d}")

    print(f"\n=== Composite Score ===")
    print(f"  Sharpe component:  {score.sharpe_component:.4f}  (w=0.25)")
    print(f"  Calmar component:  {score.calmar_component:.4f}  (w=0.25)")
    print(f"  Inv DD component:  {score.inv_dd_component:.4f}  (w=0.20)")
    print(f"  CAGR component:    {score.cagr_component:.4f}  (w=0.15)")
    print(f"  Sortino component: {score.sortino_component:.4f}  (w=0.15)")
    print(f"  TOTAL:             {score.total:.4f}")
    if score.rejected:
        print(f"  REJECTED: {score.reject_reason}")


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

def cmd_optimize(args: argparse.Namespace) -> None:
    """Run greedy forward selection."""
    import json as _json

    from backtests.regime.auto.candidates import get_all_candidates
    from backtests.regime.auto.greedy_optimize import (
        run_greedy,
        save_greedy_result,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path("backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "greedy_optimal.json"

    candidates = get_all_candidates()

    # Optimization uses daily rebalance + zero costs by default to measure
    # pure prediction quality. Production deployment decides cadence separately.
    base_mutations = {"rebalance_freq": "B"}
    if args.resume:
        resume_path = output_path if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = _json.load(f)
            base_mutations = {"rebalance_freq": "B", **prev.get("accepted_mutations", {})}
            accepted_ids = {r["candidate_id"] for r in prev.get("rounds", [])}
            before = len(candidates)
            candidates = [(n, m) for n, m in candidates if n not in accepted_ids]
            print(f"Resuming from {resume_path.name}: "
                  f"base score={prev.get('final_score', 0):.4f}, "
                  f"{len(accepted_ids)} already accepted, "
                  f"{before - len(candidates)} candidates removed")
        else:
            print(f"No previous result at {resume_path}, starting fresh")

    print(f"Loaded {len(candidates)} candidates")

    result = run_greedy(
        candidates=candidates,
        data_dir=data_dir,
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        prune_threshold=args.prune_threshold,
        base_mutations=base_mutations,
        verbose=True,
    )

    save_greedy_result(result, output_path)


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run expanding-window walk-forward validation.

    Two modes:
      1. Re-optimize per fold (default): run greedy optimizer on each training fold,
         then evaluate on OOS window. Tests optimizer generalization.
      2. Fixed-config (--mutations-json): apply a fixed config to all folds, run the
         signal engine ONCE, slice IS/OOS windows. Tests config robustness.
    """
    import pandas as pd

    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.auto.scoring import composite_score
    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data
    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    test_years = args.test_years
    fixed_mode = args.mutations_json is not None

    # Load fixed mutations if provided
    fixed_mutations = {}
    if fixed_mode:
        with open(args.mutations_json) as f:
            data = json.load(f)
        if "cumulative_mutations" in data:
            fixed_mutations = data["cumulative_mutations"]
        elif "accepted_mutations" in data:
            fixed_mutations = data["accepted_mutations"]
        else:
            fixed_mutations = data
        print(f"Fixed-config mode: {len(fixed_mutations)} mutations from {args.mutations_json}")

    # Build folds dynamically: expanding train, fixed-size test window
    data_start_year = macro_df.index.min().year
    data_end_year = macro_df.index.max().year
    # First test window starts after a minimum 4-year training period
    first_test_start = data_start_year + 5
    folds = []
    test_start_yr = first_test_start
    while test_start_yr + test_years - 1 <= data_end_year:
        train_end_yr = test_start_yr - 1
        test_end_yr = test_start_yr + test_years - 1
        folds.append((
            f"{data_start_year}-01-01",
            f"{train_end_yr}-12-31",
            f"{test_start_yr}-01-01",
            f"{test_end_yr}-12-31",
        ))
        test_start_yr += test_years
    if not folds:
        print("Not enough data for walk-forward validation.")
        return

    mode_label = "Fixed-Config" if fixed_mode else "Re-Optimize"
    print(f"=== Walk-Forward Validation ({mode_label}) ===")
    print(f"  Folds: {len(folds)}")
    print(f"  Test window: {test_years} years")

    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    if fixed_mode:
        fold_results, signals = _walk_forward_fixed(
            args, folds, fixed_mutations, macro_df, market_df,
            strat_ret_df, sim_cfg,
        )
    else:
        fold_results, signals = _walk_forward_reoptimize(
            args, folds, macro_df, market_df, strat_ret_df, sim_cfg, data_dir,
        )

    # Summary
    _print_walk_forward_summary(fold_results, fixed_mode)

    # Save results
    output_dir = Path("backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_fixed" if fixed_mode else ""
    output_path = output_dir / f"walk_forward_results{suffix}.json"
    with open(output_path, "w") as fp:
        json.dump(fold_results, fp, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    # Full diagnostics if requested
    if getattr(args, "diagnostics", False) and fold_results and signals is not None:
        from backtests.regime.analysis.diagnostics import (
            generate_regime_diagnostics_report,
        )
        from backtests.regime.engine.portfolio_sim import (
            simulate_benchmark_60_40,
        )

        full_result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
        full_score = composite_score(full_result.metrics)
        benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)

        report = generate_regime_diagnostics_report(
            signals=signals,
            result=full_result,
            benchmark=benchmark,
            score=full_score,
            sim_cfg=sim_cfg,
            walk_forward_results=fold_results,
        )
        print(report.encode("ascii", errors="replace").decode("ascii"))

        diag_path = output_dir / f"diagnostics_walk_forward{suffix}.txt"
        diag_path.write_text(report, encoding="utf-8")
        print(f"\nDiagnostics saved to {diag_path}")


def _walk_forward_fixed(
    args,
    folds: list[tuple[str, str, str, str]],
    mutations: dict,
    macro_df,
    market_df,
    strat_ret_df,
    sim_cfg,
) -> tuple[list[dict], object]:
    """Fixed-config walk-forward: run engine once, slice IS/OOS per fold."""
    import numpy as np
    import pandas as pd

    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.auto.scoring import composite_score
    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    cfg = mutate_meta_config(MetaConfig(), mutations)

    print(f"\nRunning signal engine with fixed config...")
    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )
    print(f"  Signals: {len(signals)} weeks ({signals.index.min().date()} to {signals.index.max().date()})")

    fold_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        ts_train_start = pd.Timestamp(train_start)
        ts_train_end = pd.Timestamp(train_end)
        ts_test_start = pd.Timestamp(test_start)
        ts_test_end = pd.Timestamp(test_end)

        print(f"\n--- Fold {i + 1}: Train {train_start[:4]}-{train_end[:4]}, "
              f"Test {test_start[:4]}-{test_end[:4]} ---")

        # IS metrics (training period)
        is_result = simulate_portfolio(
            signals, strat_ret_df, sim_cfg,
            start_date=ts_train_start,
            end_date=ts_train_end,
        )
        is_score = composite_score(is_result.metrics)

        # OOS metrics (test period)
        oos_result = simulate_portfolio(
            signals, strat_ret_df, sim_cfg,
            start_date=ts_test_start,
            end_date=ts_test_end,
        )
        oos_score = composite_score(oos_result.metrics)

        # OOS regime distribution
        oos_signals = signals.loc[ts_test_start:ts_test_end]
        oos_regime_dist = {}
        if not oos_signals.empty:
            _rcols = ["P_G", "P_R", "P_S", "P_D"]
            _dom = oos_signals[_rcols].idxmax(axis=1).str.replace("P_", "")
            oos_regime_dist = _dom.value_counts(normalize=True).to_dict()

        # Per-year OOS breakdown
        oos_yearly = _compute_yearly_metrics(
            signals, strat_ret_df, sim_cfg, ts_test_start, ts_test_end,
        )

        # Crisis analysis for OOS window
        crisis_analysis = _analyze_crisis_periods(
            signals, strat_ret_df, sim_cfg, ts_test_start, ts_test_end,
        )

        is_sharpe = is_result.metrics.sharpe
        oos_sharpe = oos_result.metrics.sharpe
        ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0.0

        print(f"  IS:  Sharpe={is_sharpe:.3f}  CAGR={is_result.metrics.cagr:.2%}  "
              f"DD={is_result.metrics.max_drawdown_pct:.1%}  Score={is_score.total:.4f}")
        print(f"  OOS: Sharpe={oos_sharpe:.3f}  CAGR={oos_result.metrics.cagr:.2%}  "
              f"DD={oos_result.metrics.max_drawdown_pct:.1%}  Score={oos_score.total:.4f}")
        print(f"  OOS/IS Sharpe: {ratio:.3f}")

        if oos_yearly:
            print(f"  OOS yearly:")
            for yr_info in oos_yearly:
                print(f"    {yr_info['year']}: Ret={yr_info['return']:.1%}  "
                      f"DD={yr_info['max_dd']:.1%}")

        if crisis_analysis:
            print(f"  Crisis periods in OOS:")
            for ca in crisis_analysis:
                print(f"    {ca['name']}: dominant={ca['dominant_regime']}  "
                      f"p_defensive={ca['p_defensive']:.3f}  "
                      f"DD={ca['max_dd']:.1%}")

        fold_results.append({
            "fold": i + 1,
            "train": f"{train_start}-{train_end}",
            "test": f"{test_start}-{test_end}",
            "is_score": is_score.total,
            "is_sharpe": is_sharpe,
            "is_cagr": is_result.metrics.cagr,
            "is_max_dd": is_result.metrics.max_drawdown_pct,
            "oos_score": oos_score.total,
            "oos_sharpe": oos_sharpe,
            "oos_cagr": oos_result.metrics.cagr,
            "oos_max_dd": oos_result.metrics.max_drawdown_pct,
            "oos_sortino": oos_result.metrics.sortino,
            "oos_calmar": oos_result.metrics.calmar,
            "oos_is_sharpe_ratio": ratio,
            "n_mutations": len(mutations),
            "mutations": list(mutations.keys()),
            "oos_regime_dist": oos_regime_dist,
            "oos_yearly": oos_yearly,
            "crisis_analysis": crisis_analysis,
        })

    return fold_results, signals


def _walk_forward_reoptimize(
    args,
    folds: list[tuple[str, str, str, str]],
    macro_df,
    market_df,
    strat_ret_df,
    sim_cfg,
    data_dir: Path,
) -> tuple[list[dict], object]:
    """Original walk-forward: re-optimize on each training fold."""
    import pandas as pd

    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.candidates import get_all_candidates
    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.auto.greedy_optimize import run_greedy
    from backtests.regime.auto.scoring import composite_score
    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    fold_results = []
    candidates = get_all_candidates()
    signals = None

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n--- Fold {i + 1}: Train {train_start}-{train_end}, "
              f"Test {test_start}-{test_end} ---")

        ts_train_end = pd.Timestamp(train_end)
        ts_test_start = pd.Timestamp(test_start)
        ts_test_end = pd.Timestamp(test_end)

        # Filter data for training period
        train_macro = macro_df.loc[:ts_train_end]
        train_market = market_df.loc[:ts_train_end]
        train_strat = strat_ret_df.loc[:ts_train_end]

        # Save filtered training data temporarily for greedy
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_macro.to_parquet(tmp_path / "macro_df.parquet")
            train_market.to_parquet(tmp_path / "market_df.parquet")
            train_strat.to_parquet(tmp_path / "strat_ret_df.parquet")

            # Run greedy on training period
            train_result = run_greedy(
                candidates=list(candidates),
                data_dir=tmp_path,
                initial_equity=args.equity,
                rebalance_cost_bps=args.cost_bps,
                max_workers=args.max_workers,
                max_rounds=args.max_rounds,
                min_delta=args.min_delta,
                verbose=False,
            )

        is_score = train_result.final_score
        optimal_muts = train_result.accepted_mutations

        print(f"  IS score: {is_score:.4f} ({len(train_result.rounds)} mutations accepted)")

        # Evaluate on test period (using full data but scoring only test window)
        cfg = mutate_meta_config(MetaConfig(), optimal_muts) if optimal_muts else MetaConfig()

        signals = run_signal_engine(
            macro_df=macro_df,
            strat_ret_df=strat_ret_df,
            market_df=market_df,
            growth_feature="GROWTH",
            inflation_feature="INFLATION",
            cfg=cfg,
        )

        oos_result = simulate_portfolio(
            signals, strat_ret_df, sim_cfg,
            start_date=ts_test_start,
            end_date=ts_test_end,
        )
        oos_score = composite_score(oos_result.metrics)

        # OOS regime distribution
        oos_signals = signals.loc[ts_test_start:ts_test_end]
        if not oos_signals.empty:
            _rcols = ["P_G", "P_R", "P_S", "P_D"]
            _dom = oos_signals[_rcols].idxmax(axis=1).str.replace("P_", "")
            oos_regime_dist = _dom.value_counts(normalize=True).to_dict()
        else:
            oos_regime_dist = {}

        print(f"  OOS score: {oos_score.total:.4f}")
        print(f"  OOS Sharpe: {oos_result.metrics.sharpe:.3f}")
        print(f"  OOS CAGR: {oos_result.metrics.cagr:.2%}")
        print(f"  OOS Max DD: {oos_result.metrics.max_drawdown_pct:.1%}")

        fold_results.append({
            "fold": i + 1,
            "train": f"{train_start}-{train_end}",
            "test": f"{test_start}-{test_end}",
            "is_score": is_score,
            "oos_score": oos_score.total,
            "oos_sharpe": oos_result.metrics.sharpe,
            "oos_cagr": oos_result.metrics.cagr,
            "oos_max_dd": oos_result.metrics.max_drawdown_pct,
            "n_mutations": len(train_result.rounds),
            "mutations": list(optimal_muts.keys()),
            "oos_regime_dist": oos_regime_dist,
        })

    return fold_results, signals


def _compute_yearly_metrics(
    signals, strat_ret_df, sim_cfg, start: "pd.Timestamp", end: "pd.Timestamp",
) -> list[dict]:
    """Compute per-year return and max drawdown within an OOS window."""
    import pandas as pd

    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    results = []
    year = start.year
    while year <= end.year:
        yr_start = pd.Timestamp(f"{year}-01-01")
        yr_end = pd.Timestamp(f"{year}-12-31")
        # Clip to actual OOS window
        yr_start = max(yr_start, start)
        yr_end = min(yr_end, end)
        try:
            yr_result = simulate_portfolio(
                signals, strat_ret_df, sim_cfg,
                start_date=yr_start, end_date=yr_end,
            )
            results.append({
                "year": year,
                "return": yr_result.metrics.total_return,
                "max_dd": yr_result.metrics.max_drawdown_pct,
                "sharpe": yr_result.metrics.sharpe,
            })
        except Exception:
            pass
        year += 1
    return results


def _analyze_crisis_periods(
    signals, strat_ret_df, sim_cfg,
    oos_start: "pd.Timestamp", oos_end: "pd.Timestamp",
) -> list[dict]:
    """Check regime behavior during known crisis periods that fall in the OOS window."""
    import numpy as np
    import pandas as pd

    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    crisis_periods = {
        "GFC": ("2008-09-01", "2009-03-01", "D"),
        "Euro Crisis": ("2011-07-01", "2011-12-31", "D"),
        "COVID": ("2020-02-15", "2020-04-15", "D"),
        "2022 Inflation": ("2022-01-01", "2022-10-01", "S"),
        "2025 Tariff": ("2025-03-01", "2025-05-31", "D"),
    }

    results = []
    for name, (cs, ce, expected) in crisis_periods.items():
        cs_ts = pd.Timestamp(cs)
        ce_ts = pd.Timestamp(ce)
        # Check if crisis overlaps with OOS window
        if ce_ts < oos_start or cs_ts > oos_end:
            continue
        # Clip to OOS window
        cs_ts = max(cs_ts, oos_start)
        ce_ts = min(ce_ts, oos_end)

        crisis_signals = signals.loc[cs_ts:ce_ts]
        if crisis_signals.empty:
            continue

        _rcols = ["P_G", "P_R", "P_S", "P_D"]
        _labels = ["G", "R", "S", "D"]

        # Average posteriors during crisis
        avg_posteriors = {
            _labels[j]: float(crisis_signals[_rcols[j]].mean())
            for j in range(4)
        }
        dominant = max(avg_posteriors, key=avg_posteriors.get)

        # Probability of expected + defensive regimes
        p_defensive = avg_posteriors.get("D", 0.0) + avg_posteriors.get("S", 0.0)

        # Drawdown during crisis
        try:
            crisis_result = simulate_portfolio(
                signals, strat_ret_df, sim_cfg,
                start_date=cs_ts, end_date=ce_ts,
            )
            crisis_dd = crisis_result.metrics.max_drawdown_pct
            crisis_ret = crisis_result.metrics.total_return
        except Exception:
            crisis_dd = float("nan")
            crisis_ret = float("nan")

        results.append({
            "name": name,
            "period": f"{cs_ts.date()} to {ce_ts.date()}",
            "expected_regime": expected,
            "dominant_regime": dominant,
            "correct_regime": dominant == expected,
            "avg_posteriors": avg_posteriors,
            "p_defensive": p_defensive,
            "max_dd": crisis_dd,
            "return": crisis_ret,
        })

    return results


def _print_walk_forward_summary(fold_results: list[dict], fixed_mode: bool) -> None:
    """Print walk-forward summary with acceptance criteria check."""
    import numpy as np

    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'=' * 60}")

    is_key = "is_sharpe" if fixed_mode else "is_score"
    oos_key = "oos_sharpe" if fixed_mode else "oos_score"

    is_vals = [f[is_key] for f in fold_results if is_key in f]
    oos_sharpes = [f["oos_sharpe"] for f in fold_results]
    oos_scores = [f["oos_score"] for f in fold_results]
    oos_dds = [f["oos_max_dd"] for f in fold_results]

    if fixed_mode and is_vals:
        avg_is_sharpe = np.mean(is_vals)
        avg_oos_sharpe = np.mean(oos_sharpes)
        oos_is_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0.0

        print(f"\n  Avg IS Sharpe:   {avg_is_sharpe:.3f}")
        print(f"  Avg OOS Sharpe:  {avg_oos_sharpe:.3f}")
        print(f"  OOS/IS Sharpe:   {oos_is_ratio:.3f}")
    else:
        is_scores = [f.get("is_score", 0) for f in fold_results]
        avg_is = np.mean(is_scores)
        avg_oos = np.mean(oos_scores)
        stability = avg_oos / avg_is if avg_is > 0 else 0
        print(f"\n  Avg IS score:  {avg_is:.4f}")
        print(f"  Avg OOS score: {avg_oos:.4f}")
        print(f"  OOS/IS ratio:  {stability:.3f}")

    print(f"  Avg OOS Score:   {np.mean(oos_scores):.4f}")
    print(f"  Positive OOS:    {sum(1 for s in oos_scores if s > 0)}/{len(oos_scores)}")

    print(f"\n  Per-fold:")
    for f in fold_results:
        line = (f"    Fold {f['fold']}: "
                f"OOS Sharpe={f['oos_sharpe']:.3f}  "
                f"CAGR={f.get('oos_cagr', 0):.2%}  "
                f"DD={f['oos_max_dd']:.1%}")
        if "oos_is_sharpe_ratio" in f:
            line += f"  OOS/IS={f['oos_is_sharpe_ratio']:.3f}"
        print(line)

    # Acceptance criteria (from assessment doc)
    if fixed_mode:
        print(f"\n  {'=' * 50}")
        print(f"  ACCEPTANCE CRITERIA")
        print(f"  {'=' * 50}")

        # 1. OOS/IS Sharpe ratio >= 0.60
        per_fold_ratios = [f.get("oos_is_sharpe_ratio", 0) for f in fold_results]
        avg_ratio = np.mean(per_fold_ratios) if per_fold_ratios else 0
        pass_ratio = avg_ratio >= 0.60
        verdict_ratio = "PASS" if pass_ratio else "FAIL"
        print(f"\n  1. OOS/IS Sharpe ratio >= 0.60")
        print(f"     Avg OOS/IS: {avg_ratio:.3f}  [{verdict_ratio}]")
        for f in fold_results:
            r = f.get("oos_is_sharpe_ratio", 0)
            flag = "ok" if r >= 0.60 else "FAIL"
            print(f"       Fold {f['fold']}: {r:.3f} [{flag}]")

        # 2. No OOS year with >20% MaxDD
        worst_yearly_dd = 0.0
        all_yearly = []
        for f in fold_results:
            for yr in f.get("oos_yearly", []):
                all_yearly.append(yr)
                worst_yearly_dd = max(worst_yearly_dd, yr["max_dd"])
        pass_dd = worst_yearly_dd <= 0.20
        verdict_dd = "PASS" if pass_dd else "FAIL"
        print(f"\n  2. No OOS year with >20% MaxDD")
        print(f"     Worst yearly DD: {worst_yearly_dd:.1%}  [{verdict_dd}]")
        for yr in sorted(all_yearly, key=lambda x: -x["max_dd"])[:5]:
            flag = "ok" if yr["max_dd"] <= 0.20 else "FAIL"
            print(f"       {yr['year']}: DD={yr['max_dd']:.1%} Ret={yr['return']:.1%} [{flag}]")

        # 3. OOS crisis periods show defensive positioning
        all_crises = []
        for f in fold_results:
            all_crises.extend(f.get("crisis_analysis", []))
        if all_crises:
            n_correct = sum(1 for c in all_crises if c["p_defensive"] >= 0.40)
            pass_crisis = n_correct == len(all_crises)
            verdict_crisis = "PASS" if pass_crisis else "FAIL"
            print(f"\n  3. OOS crisis periods show defensive positioning (p_defensive >= 0.40)")
            print(f"     {n_correct}/{len(all_crises)} crises with defensive posture  [{verdict_crisis}]")
            for c in all_crises:
                flag = "ok" if c["p_defensive"] >= 0.40 else "FAIL"
                print(f"       {c['name']} ({c['period']}): "
                      f"dominant={c['dominant_regime']} "
                      f"p_def={c['p_defensive']:.3f} "
                      f"DD={c['max_dd']:.1%} [{flag}]")
        else:
            print(f"\n  3. No crisis periods in OOS windows")
            pass_crisis = True

        # Overall verdict
        all_pass = pass_ratio and pass_dd and pass_crisis
        overall = "STABLE -- R3 is a valid baseline" if all_pass else "UNSTABLE -- review failures"
        print(f"\n  {'=' * 50}")
        print(f"  VERDICT: {overall}")
        print(f"  {'=' * 50}")


# ---------------------------------------------------------------------------
# phase-run
# ---------------------------------------------------------------------------

def _run_phase_core(args: argparse.Namespace, phase: int,
                    extra_candidates: list | None = None):
    """Core phase execution logic. Returns (greedy_result, analysis, state).

    Shared by ``cmd_phase_run`` (single-phase) and ``cmd_phase_auto`` (loop).
    """
    from dataclasses import asdict as _asdict

    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.auto.greedy_optimize import (
        run_greedy,
        save_greedy_result,
    )
    from backtests.regime.auto.phase_analyzer import (
        analyze_phase,
        save_phase_analysis,
    )
    from backtests.regime.auto.phase_candidates import get_phase_candidates
    from backtests.regime.auto.phase_gates import check_phase_gate
    from backtests.regime.auto.phase_scoring import compute_regime_stats
    from backtests.regime.auto.phase_state import (
        PhaseState,
        load_phase_state,
        save_phase_state,
    )
    from backtests.regime.auto.scoring import composite_score
    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data
    from backtests.regime.engine.portfolio_sim import (
        simulate_benchmark_60_40,
        simulate_portfolio,
    )

    data_dir = Path(args.data_dir)
    output_dir = _resolve_phase_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "phase_state.json"

    # Set up file logging
    log_path = _setup_phase_logging(phase, output_dir)
    print(f"Logging to {log_path}")

    # Load or create phase state
    baseline_source = None
    if state_path.exists():
        state = load_phase_state(state_path)
        if state.completed_phases or state.cumulative_mutations:
            logger.info("Loaded phase state: completed=%s, cumulative_mutations=%s",
                         state.completed_phases, state.cumulative_mutations)
            print(f"Loaded phase state: completed phases {state.completed_phases}")
            print("Using saved phase state mutations (preset applies only to fresh state)")
        else:
            state.cumulative_mutations, baseline_source = _resolve_mutation_source(
                getattr(args, "mutations_json", None),
                getattr(args, "preset", None),
                DEFAULT_RESEARCH_PRESET,
            )
            logger.info("Initialized empty phase state from %s", baseline_source["label"])
            _print_mutation_source("Initialized empty phase state from", baseline_source)
    else:
        baseline_mutations, baseline_source = _resolve_mutation_source(
            getattr(args, "mutations_json", None),
            getattr(args, "preset", None),
            DEFAULT_RESEARCH_PRESET,
        )
        state = PhaseState(cumulative_mutations=baseline_mutations)
        logger.info("Starting fresh phase state from %s", baseline_source["label"])
        _print_mutation_source("Starting fresh phase state from", baseline_source)

    # Get cumulative mutations from prior phases
    base_muts = dict(state.cumulative_mutations)
    base_muts["rebalance_freq"] = "W-FRI"  # match production rebalance frequency
    phase_max_rounds = _resolve_phase_max_rounds(args, phase)

    # Get candidates for this phase (Phase 4 gets prior mutations for adaptive narrowing)
    prior_diag = {"cumulative_mutations": base_muts} if phase == 4 else None
    candidates = get_phase_candidates(
        phase,
        prior_diagnostics=prior_diag,
        profile=getattr(args, "candidate_profile", "default"),
    )

    # Merge any extra experiments from prior analysis recommendations
    if extra_candidates:
        existing_names = {c[0] for c in candidates}
        n_before = len(candidates)
        for name, muts in extra_candidates:
            if name not in existing_names:
                candidates.append((name, muts))
                existing_names.add(name)
        n_added = len(candidates) - n_before
        if n_added:
            logger.info("Added %d extra experiments from prior analysis", n_added)

    logger.info("Phase %d: %d candidates", phase, len(candidates))
    print(f"Phase {phase}: {len(candidates)} candidates")

    # Run greedy with phase-aware scoring
    try:
        result = run_greedy(
            candidates=candidates,
            data_dir=data_dir,
            initial_equity=args.equity,
            rebalance_cost_bps=args.cost_bps,
            max_workers=args.max_workers,
            max_rounds=phase_max_rounds,
            min_delta=args.min_delta,
            prune_threshold=args.prune_threshold,
            base_mutations=base_muts,
            verbose=True,
            phase=phase,
            candidate_timeout=getattr(args, 'candidate_timeout', 600.0),
        )
    except Exception:
        logger.exception("Phase %d greedy optimization CRASHED", phase)
        raise

    # Save greedy result
    result_path = output_dir / f"phase_{phase}_result.json"
    save_greedy_result(result, result_path)

    # Update phase state
    new_muts = {k: v for k, v in result.accepted_mutations.items()
                if k != "rebalance_freq"}
    state.advance_phase(phase, new_muts, {
        "baseline_score": result.baseline_score,
        "final_score": result.final_score,
        "final_metrics": result.final_metrics,
        "n_rounds": len(result.rounds),
        "max_rounds": phase_max_rounds,
        "rounds": [r.candidate_id for r in result.rounds],
        "elapsed_seconds": result.elapsed_seconds,
    })
    save_phase_state(state, state_path)

    # --- Diagnostics, gate check, and analysis ---
    # Wrapped so a crash here doesn't lose the already-saved greedy result.
    try:
        logger.info("Running phase %d diagnostics...", phase)
        print(f"\n--- Running phase {phase} diagnostics ---")
        macro_df, market_df, strat_ret_df = load_cached_data(data_dir)
        cfg = mutate_meta_config(MetaConfig(), {**state.cumulative_mutations, "rebalance_freq": "B"})
        sim_cfg = RegimeBacktestConfig(
            initial_equity=args.equity,
            rebalance_cost_bps=args.cost_bps,
            data_dir=data_dir,
        )

        signals = run_signal_engine(
            macro_df=macro_df,
            strat_ret_df=strat_ret_df,
            market_df=market_df,
            growth_feature="GROWTH",
            inflation_feature="INFLATION",
            cfg=cfg,
        )

        sim_result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
        score = composite_score(sim_result.metrics)
        regime_stats = compute_regime_stats(signals, L_max=cfg.L_max)

        # Full diagnostics report
        from backtests.regime.analysis.diagnostics import (
            generate_regime_diagnostics_report,
        )
        benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)
        report = generate_regime_diagnostics_report(
            signals=signals, result=sim_result, benchmark=benchmark,
            score=score, sim_cfg=sim_cfg,
        )

        # Phase-specific diagnostics
        from backtests.regime.auto.phase_diagnostics import (
            generate_phase_diagnostics,
        )
        phase_report = generate_phase_diagnostics(
            phase=phase, regime_stats=regime_stats,
            metrics=sim_result.metrics, greedy_result=result, state=state,
        )

        full_report = report + "\n\n" + phase_report
        diag_path = output_dir / f"phase_{phase}_diagnostics.txt"
        diag_path.write_text(full_report, encoding="utf-8")
        logger.info("Diagnostics saved to %s", diag_path)
        print(full_report.encode("ascii", errors="replace").decode("ascii"))
        print(f"\nDiagnostics saved to {diag_path}")

        # Gate check
        logger.info("Running phase %d gate check...", phase)
        greedy_data = state.phase_results.get(phase, {})
        gate = check_phase_gate(phase, sim_result.metrics, regime_stats, greedy_data)
        gate_passed = gate.passed

        state.record_gate(phase, {
            "passed": gate.passed,
            "criteria": [_asdict(c) for c in gate.criteria],
            "failure_category": gate.failure_category,
            "recommendations": gate.recommendations,
        })
        save_phase_state(state, state_path)

        gate_status = "PASSED" if gate.passed else "FAILED"
        logger.info("Phase %d gate: %s", phase, gate_status)
        print(f"\n=== Phase {phase} Gate: {gate_status} ===")
        for c in gate.criteria:
            mark = "[PASS]" if c.passed else "[FAIL]"
            logger.info("  %s %s: %.4f (target: %.4f)",
                        mark, c.name, c.actual, c.target)
            print(f"  {mark} {c.name}: {c.actual:.4f} (target: {c.target:.4f})")
        if not gate.passed:
            print(f"  Failure: {gate.failure_category}")
            for r in gate.recommendations:
                print(f"    - {r}".encode("ascii", errors="replace").decode("ascii"))

        # Post-phase analysis
        logger.info("Running post-phase analysis...")
        analysis = analyze_phase(
            phase=phase, greedy_result=result, regime_stats=regime_stats,
            metrics=sim_result.metrics, state=state, gate_passed=gate_passed,
        )
        analysis_path = output_dir / f"phase_{phase}_analysis.json"
        save_phase_analysis(analysis, analysis_path)
        print(analysis.report.encode("ascii", errors="replace").decode("ascii"))
        print(f"Analysis saved to {analysis_path}")

        # Regime stats summary
        print(f"\n=== Regime Stats ===")
        print(f"  Active regimes: {regime_stats['n_active_regimes']}")
        print(f"  Regime entropy: {regime_stats['regime_entropy']:.4f}")
        print(f"  Transition rate: {regime_stats['transition_rate']:.4f}")
        print(f"  Distribution: {regime_stats['dominant_dist']}")
        print(f"  Crisis response: {regime_stats['crisis_response']:.4f}")

    except Exception:
        logger.exception("Phase %d diagnostics/analysis CRASHED "
                         "(greedy result already saved to %s)", phase, result_path)
        # Return a minimal analysis so phase-auto can still proceed
        from backtests.regime.auto.phase_analyzer import PhaseAnalysis
        analysis = PhaseAnalysis(
            phase=phase, goal_progress={}, strengths=[], weaknesses=[],
            scoring_assessment="UNKNOWN (diagnostics crashed)",
            suggested_experiments=[], recommendation="proceed",
            recommendation_reason="Diagnostics crashed; proceeding with saved result.",
            report="(diagnostics crashed - see log for traceback)",
        )
        print(f"\nWARNING: Diagnostics crashed. Greedy result saved to {result_path}.")
        print("Check the log file for the full traceback.")

    logger.info("Phase %d complete. Recommendation: %s",
                phase, analysis.recommendation)

    return result, analysis, state


def cmd_phase_run(args: argparse.Namespace) -> None:
    """Run a single phase of the multi-phase optimization."""
    _run_phase_core(args, args.phase)


# ---------------------------------------------------------------------------
# phase-auto
# ---------------------------------------------------------------------------

def cmd_phase_auto(args: argparse.Namespace) -> None:
    """Run all remaining phases with analysis-driven orchestration.

    After each phase, the analyzer evaluates results and recommends:
    - ``proceed``: Move to the next phase.
    - ``rerun``: Retry the phase with the same candidates (max 1 retry).
    - ``expand_and_proceed``: Add suggested experiments to the next phase's
      candidate pool and continue.

    All decisions and their reasoning are logged to the phase log files.
    """
    from backtests.regime.auto.phase_state import (
        PhaseState,
        load_phase_state,
    )

    output_dir = _resolve_phase_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "phase_state.json"
    phase_sequence = _resolve_phase_sequence(args)
    allow_extra_experiments = getattr(args, "allow_extra_experiments", True)

    # Determine starting phase
    if state_path.exists():
        state = load_phase_state(state_path)
        remaining_phases = [
            phase for phase in phase_sequence if phase not in state.completed_phases
        ]
        if not remaining_phases:
            print("Phase-auto already complete for configured phase sequence.")
            print(f"  Completed phases: {state.completed_phases}")
            return
        start_phase = remaining_phases[0]
        print(f"Resuming from phase {start_phase} "
              f"(completed: {state.completed_phases})")
    else:
        start_phase = 1
        remaining_phases = list(phase_sequence)
        baseline_source = _resolve_mutation_source(
            getattr(args, "mutations_json", None),
            getattr(args, "preset", None),
            DEFAULT_RESEARCH_PRESET,
        )[1]
        _print_mutation_source("Starting fresh phase-auto from", baseline_source)

    if not state_path.exists():
        remaining_phases = [phase for phase in phase_sequence if phase >= start_phase]
    if not allow_extra_experiments:
        print("Strict mode: analyzer suggested experiments are disabled")

    max_retries = args.max_retries
    extra_experiments: list[tuple[str, dict]] = []

    # On resume, load suggested experiments from the last completed phase's analysis
    if allow_extra_experiments and state_path.exists() and start_phase > 1:
        prev_analysis_path = output_dir / f"phase_{start_phase - 1}_analysis.json"
        if prev_analysis_path.exists():
            import json as _json
            with open(prev_analysis_path, encoding="utf-8") as _f:
                prev_analysis = _json.load(_f)
            extra_experiments = [
                (e["name"], e["mutations"])
                for e in prev_analysis.get("suggested_experiments", [])
            ]
            if extra_experiments:
                print(f"Loaded {len(extra_experiments)} suggested experiments "
                      f"from Phase {start_phase - 1} analysis")
                logger.info("Loaded %d experiments from phase_%d_analysis.json",
                            len(extra_experiments), start_phase - 1)

    for phase in remaining_phases:
        for attempt in range(max_retries + 1):
            attempt_label = f" (retry {attempt})" if attempt > 0 else ""
            print(f"\n{'='*60}")
            print(f"  PHASE {phase}{attempt_label}")
            print(f"{'='*60}\n")

            result, analysis, state = _run_phase_core(
                args, phase,
                extra_candidates=extra_experiments if extra_experiments else None,
            )
            extra_experiments = []  # consumed

            if analysis.recommendation in ("proceed", "expand_and_proceed"):
                if allow_extra_experiments:
                    extra_experiments = list(analysis.suggested_experiments)
                if extra_experiments:
                    print(f"\n>> Phase {phase}: Adding {len(extra_experiments)} "
                           f"suggested experiments to Phase {phase + 1}")
                    logger.info("Phase %d: carrying %d suggested experiments "
                                "to Phase %d", phase, len(extra_experiments),
                                phase + 1)
                print(f"\n>> Phase {phase}: PROCEEDING to next phase")
                break

            elif analysis.recommendation == "rerun" and attempt < max_retries:
                if allow_extra_experiments:
                    extra_experiments = list(analysis.suggested_experiments)
                print(f"\n>> Phase {phase}: RERUNNING (attempt {attempt + 2})")
                if extra_experiments:
                    print(f"   Adding {len(extra_experiments)} suggested "
                          f"experiments to retry")
                logger.info("Phase %d: rerunning with %d extra experiments "
                            "(reason: %s)", phase, len(extra_experiments),
                            analysis.recommendation_reason)
            else:
                print(f"\n>> Phase {phase}: Exhausted retries, proceeding anyway")
                logger.warning("Phase %d: exhausted %d retries, proceeding",
                               phase, max_retries)
                break

    print(f"\n{'='*60}")
    print(f"  AUTO-OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Completed phases: {state.completed_phases}")
    print(f"  Final mutations: {state.cumulative_mutations}")
    logger.info("Auto-optimization complete. Phases=%s, mutations=%s",
                state.completed_phases, state.cumulative_mutations)


def cmd_step9_optimize(args: argparse.Namespace) -> None:
    """Run the assessment's Step 9 R6 optimization in an isolated output dir."""
    if not getattr(args, "output_dir", None):
        args.output_dir = str(DEFAULT_STEP9_OUTPUT_DIR)
    args.preset = STEP9_RESEARCH_PRESET
    args.candidate_profile = "step9_r6"
    args.mutations_json = None
    args.allow_extra_experiments = False
    args.phase_sequence = (1, 2, 3)
    args.phase_max_rounds = {
        1: args.max_rounds,
        2: 1,
        3: 1,
    }

    print("Launching assessment Step 9 R6 optimization")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Baseline preset: {STEP9_RESEARCH_PRESET}")
    print(f"  Candidate profile: {args.candidate_profile}")
    print(f"  Max workers: {args.max_workers}")
    print("  Strict mode: no carried-forward experiments, phases 1-3 only")
    print("  Phase rounds: phase1=greedy, phase2=1, phase3=1")

    cmd_phase_auto(args)


def cmd_r7_optimize(args: argparse.Namespace) -> None:
    """Run the R7 overlay recalibration optimization."""
    if not getattr(args, "output_dir", None):
        args.output_dir = str(DEFAULT_R7_OUTPUT_DIR)
    args.preset = R7_RESEARCH_PRESET
    args.candidate_profile = "r7_overlay"
    args.mutations_json = None
    args.allow_extra_experiments = False
    args.phase_sequence = (1, 2, 3)
    args.phase_max_rounds = {
        1: args.max_rounds,
        2: 1,
        3: 1,
    }

    print("Launching R7 overlay recalibration optimization")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Baseline preset: {R7_RESEARCH_PRESET}")
    print(f"  Candidate profile: {args.candidate_profile}")
    print(f"  Max workers: {args.max_workers}")
    print("  Strict mode: no carried-forward experiments, phases 1-3 only")
    print("  Phase rounds: phase1=greedy, phase2=1, phase3=1")

    cmd_phase_auto(args)


def cmd_r8_optimize(args: argparse.Namespace) -> None:
    """Run the R8 two-model architecture optimization."""
    if not getattr(args, "output_dir", None):
        args.output_dir = str(DEFAULT_R8_OUTPUT_DIR)
    args.preset = R8_RESEARCH_PRESET
    args.candidate_profile = "r8_stress"
    args.mutations_json = None
    args.allow_extra_experiments = False
    args.phase_sequence = (1, 2, 3, 4)
    args.phase_max_rounds = {
        1: args.max_rounds,
        2: args.max_rounds,
        3: 1,
        4: 1,
    }

    print("Launching R8 two-model architecture optimization")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Baseline preset: {R8_RESEARCH_PRESET}")
    print(f"  Candidate profile: {args.candidate_profile}")
    print(f"  Max workers: {args.max_workers}")
    print("  Phase rounds: phase1+2=greedy, phase3+4=1 round")

    cmd_phase_auto(args)


def cmd_r9_budget_optimize(args: argparse.Namespace) -> None:
    """Run the R9 budget-only optimization."""
    if not getattr(args, "output_dir", None):
        args.output_dir = str(DEFAULT_R9_OUTPUT_DIR)
    args.preset = R9_RESEARCH_PRESET
    args.candidate_profile = "r9_budget"
    args.mutations_json = None
    args.allow_extra_experiments = False
    args.phase_sequence = (5,)
    args.phase_max_rounds = {5: args.max_rounds}

    print("Launching R9 budget-only optimization")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Baseline preset: {R9_RESEARCH_PRESET}")
    print(f"  Candidate profile: {args.candidate_profile}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Max rounds: {args.max_rounds}")
    print("  Phase: 5 (budget allocation only, HMM cached)")

    cmd_phase_auto(args)


# ---------------------------------------------------------------------------
# phase-gate
# ---------------------------------------------------------------------------

def cmd_phase_gate(args: argparse.Namespace) -> None:
    """Check the success gate for a completed phase."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.auto.phase_gates import check_phase_gate
    from backtests.regime.auto.phase_scoring import compute_regime_stats
    from backtests.regime.auto.phase_state import (
        load_phase_state,
        save_phase_state,
    )
    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data
    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    phase = args.phase
    data_dir = Path(args.data_dir)
    output_dir = _resolve_phase_output_dir(args)
    state_path = output_dir / "phase_state.json"

    if not state_path.exists():
        print("No phase state found. Run phase-run first.")
        return

    state = load_phase_state(state_path)
    if phase not in state.completed_phases:
        print(f"Phase {phase} not completed yet. Completed: {state.completed_phases}")
        return

    # Run evaluation with cumulative mutations
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)
    cfg = mutate_meta_config(MetaConfig(), state.cumulative_mutations)
    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    sim_result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
    regime_stats = compute_regime_stats(signals, L_max=cfg.L_max)

    # Check gate
    greedy_data = state.phase_results.get(phase, {})
    gate = check_phase_gate(phase, sim_result.metrics, regime_stats, greedy_data)

    # Record gate result
    from dataclasses import asdict
    state.record_gate(phase, {
        "passed": gate.passed,
        "criteria": [asdict(c) for c in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": gate.recommendations,
    })
    save_phase_state(state, state_path)

    # Print results
    status = "PASSED" if gate.passed else "FAILED"
    print(f"\n=== Phase {phase} Gate: {status} ===")
    for c in gate.criteria:
        mark = "[PASS]" if c.passed else "[FAIL]"
        print(f"  {mark} {c.name}: {c.actual:.4f} (target: {c.target:.4f})")

    if not gate.passed:
        print(f"\n  Failure category: {gate.failure_category}")
        print(f"  Recommendations:")
        for r in gate.recommendations:
            print(f"    - {r}")


# ---------------------------------------------------------------------------
# phase-diagnostics
# ---------------------------------------------------------------------------

def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    """Print diagnostics for a completed phase."""
    output_dir = _resolve_phase_output_dir(args)
    diag_path = output_dir / f"phase_{args.phase}_diagnostics.txt"

    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}. Run phase-run first.")
        return

    print(diag_path.read_text())


# ---------------------------------------------------------------------------
# historical-validate
# ---------------------------------------------------------------------------

def cmd_historical_validate(args: argparse.Namespace) -> None:
    """Run historical validation against known regime timeline."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.auto.historical_validation import (
        compute_historical_alignment,
        compute_transition_latency,
    )
    from backtests.regime.data.downloader import load_cached_data

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    mutations, source = _resolve_mutation_source(
        args.mutations_json,
        getattr(args, "preset", None),
        DEFAULT_RESEARCH_PRESET,
    )
    _print_mutation_source("Historical validation config", source)

    cfg = mutate_meta_config(MetaConfig(), mutations)

    print("Running signal engine...")
    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    alignment = compute_historical_alignment(signals)
    latency = compute_transition_latency(signals)

    print(f"\n=== Historical Validation ===")
    print(f"  Alignment score:  {alignment['overall']:.4f} (target: >0.5)")
    print(f"  Per-period:")
    for name, score in alignment.get("per_period", {}).items():
        print(f"    {name}: {score:.4f}")

    print(f"\n  Transition latency:")
    for name, weeks in latency.items():
        target = "<8 weeks"
        print(f"    {name}: {weeks:.1f} weeks ({target})")


# ---------------------------------------------------------------------------
# scanner-validate
# ---------------------------------------------------------------------------

def cmd_scanner_validate(args: argparse.Namespace) -> None:
    """Run leading indicator scanner validation against acceptance criteria."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.analysis.scanner_validation import validate_scanner
    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.data.downloader import load_cached_data

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    mutations, source = _resolve_mutation_source(
        args.mutations_json,
        getattr(args, "preset", None),
        DEFAULT_RESEARCH_PRESET,
    )
    _print_mutation_source("Scanner validation config", source)

    # Force scanner enabled
    mutations["scanner_enabled"] = True

    cfg = mutate_meta_config(MetaConfig(), mutations)

    print("Running signal engine with scanner enabled...")
    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )

    # Validate
    results = validate_scanner(signals, threshold=cfg.scanner_threshold)

    print(f"\n{'=' * 60}")
    print(f"  LEADING INDICATOR SCANNER VALIDATION")
    print(f"{'=' * 60}")
    print(f"\n  Transitions analyzed:    {results['transitions_analyzed']}")
    print(f"  Total risk-off alerts:   {results['total_risk_off_alerts']}")

    print(f"\n  Acceptance Criteria:")
    for name, crit in results["criteria"].items():
        status = "PASS" if crit["passed"] else "FAIL"
        print(f"    [{status}] {name}: {crit['actual']} (target: {crit['target']})")

    if results["jan_2022_first_week"]:
        print(f"\n  Jan 2022 first risk-off flag: {results['jan_2022_first_week']}")

    if results.get("lead_times"):
        print(f"\n  Per-transition lead times (weeks):")
        for i, lt in enumerate(results["lead_times"], 1):
            print(f"    Transition {i}: {lt:.1f} weeks")

    print(f"\n  VERDICT: {results['verdict']}")

    # Full diagnostics if requested
    if args.diagnostics:
        from backtests.regime.analysis.diagnostics import (
            generate_regime_diagnostics_report,
        )
        from backtests.regime.auto.scoring import composite_score
        from backtests.regime.config import RegimeBacktestConfig
        from backtests.regime.engine.portfolio_sim import (
            simulate_benchmark_60_40,
            simulate_portfolio,
        )

        sim_cfg = RegimeBacktestConfig(
            initial_equity=args.equity,
            rebalance_cost_bps=args.cost_bps,
            data_dir=data_dir,
        )
        result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
        score = composite_score(result.metrics)
        benchmark = simulate_benchmark_60_40(strat_ret_df, sim_cfg)

        report = generate_regime_diagnostics_report(
            signals=signals,
            result=result,
            benchmark=benchmark,
            score=score,
            sim_cfg=sim_cfg,
        )
        print(report.encode("ascii", errors="replace").decode("ascii"))

        output_dir = Path("backtests/regime/auto/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "scanner_diagnostics.txt"
        output_path.write_text(report, encoding="utf-8")
        print(f"\nDiagnostics saved to {output_path}")


# ---------------------------------------------------------------------------
# calibration-sweep
# ---------------------------------------------------------------------------


def cmd_calibration_sweep(args: argparse.Namespace) -> None:
    """Run the Step 6 posterior calibration sweep against the R3 reference."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.analysis.assessment_validation import (
        calibration_candidate_passes,
        summarize_calibration_candidate,
    )
    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data
    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)

    base_mutations, source = _resolve_mutation_source(
        args.mutations_json,
        getattr(args, "preset", None),
        DEFAULT_REFERENCE_PRESET,
    )
    _print_mutation_source("Calibration sweep baseline", source)
    base_mutations = {**base_mutations, "scanner_enabled": False}

    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    base_cfg = mutate_meta_config(MetaConfig(), base_mutations) if base_mutations else MetaConfig()
    base_signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=base_cfg,
    )
    base_result = simulate_portfolio(base_signals, strat_ret_df, sim_cfg)
    baseline_summary = summarize_calibration_candidate(base_signals, base_result)
    baseline_summary["name"] = "baseline_r3"
    baseline_summary["temperature"] = float(base_cfg.posterior_temperature)
    baseline_summary["ema_alpha"] = float(base_cfg.posterior_ema_alpha)
    baseline_summary["passed"] = calibration_candidate_passes(baseline_summary)

    candidates = [
        ("temp=1.2, ema=0.8", {"posterior_temperature": 1.2, "posterior_ema_alpha": 0.8}),
        ("temp=1.5, ema=0.7", {"posterior_temperature": 1.5, "posterior_ema_alpha": 0.7}),
        ("temp=1.5, ema=0.8", {"posterior_temperature": 1.5, "posterior_ema_alpha": 0.8}),
    ]

    summaries = []
    for name, overrides in candidates:
        mutations = {
            **base_mutations,
            **overrides,
            "posterior_smoothing_eps": 0.01,
        }
        cfg = mutate_meta_config(MetaConfig(), mutations)
        signals = run_signal_engine(
            macro_df=macro_df,
            strat_ret_df=strat_ret_df,
            market_df=market_df,
            growth_feature="GROWTH",
            inflation_feature="INFLATION",
            cfg=cfg,
        )
        result = simulate_portfolio(signals, strat_ret_df, sim_cfg)
        summary = summarize_calibration_candidate(signals, result)
        summary["name"] = name
        summary["temperature"] = overrides["posterior_temperature"]
        summary["ema_alpha"] = overrides["posterior_ema_alpha"]
        summary["passed"] = calibration_candidate_passes(summary)
        summaries.append(summary)

    passing = [summary for summary in summaries if summary["passed"]]
    if passing:
        winner = max(passing, key=lambda item: item["sharpe"])
        recommendation = f"Adopt {winner['name']} (highest Sharpe among passing configs)"
    else:
        winner = baseline_summary
        recommendation = "Keep the unmodified R3 reference (no candidate passed all thresholds)"

    print("\n=== Calibration Sweep ===")
    print(f"Recommendation: {recommendation}")
    print(
        f"{'Candidate':<20s} {'AvgP(dom)':>10s} {'SPYRange':>10s} "
        f"{'Sharpe':>8s} {'MaxDD':>8s} {'2022':>6s} {'PASS':>6s}"
    )
    print("-" * 76)
    for summary in [baseline_summary, *summaries]:
        print(
            f"{summary['name']:<20s} {summary['avg_p_dom']:>10.3f} "
            f"{summary['spy_allocation_range_bp']:>9.0f}bp "
            f"{summary['sharpe']:>8.3f} {summary['max_drawdown_pct']:>7.1%} "
            f"{summary['regime_2022']:>6s} {str(summary['passed']):>6s}"
        )

    output_dir = Path("backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_sweep.json"
    output = {
        "baseline_source": source,
        "baseline": baseline_summary,
        "candidates": summaries,
        "winner": winner,
        "recommendation": recommendation,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


# ---------------------------------------------------------------------------
# validate-2022
# ---------------------------------------------------------------------------


def cmd_validate_2022(args: argparse.Namespace) -> None:
    """Run the Step 7 targeted 2022 validation scenarios."""
    from regime.config import MetaConfig
    from regime.engine import run_signal_engine

    from backtests.regime.analysis.assessment_validation import (
        summarize_2022_validation,
        validate_step7_outcome,
    )
    from backtests.regime.auto.config_mutator import mutate_meta_config
    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data
    from backtests.regime.engine.portfolio_sim import simulate_portfolio

    data_dir = Path(args.data_dir)
    macro_df, market_df, strat_ret_df = load_cached_data(data_dir)
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2023-06-30")
    full_mutations, full_source = _resolve_mutation_source(
        args.mutations_json,
        getattr(args, "preset", None),
        DEFAULT_RESEARCH_PRESET,
    )
    r3_mutations, r3_source = _resolve_mutation_source(
        args.r3_mutations_json,
        getattr(args, "r3_preset", None),
        DEFAULT_REFERENCE_PRESET,
    )
    _print_mutation_source("Validate-2022 full stack", full_source)
    _print_mutation_source("Validate-2022 R3 reference", r3_source)

    sim_cfg = RegimeBacktestConfig(
        initial_equity=args.equity,
        rebalance_cost_bps=args.cost_bps,
        data_dir=data_dir,
    )

    scenarios = {
        "full_stack": {
            **full_mutations,
            "scanner_enabled": True,
            "crisis_leverage_enabled": True,
        },
        "scanner_off": {
            **full_mutations,
            "scanner_enabled": False,
            "crisis_leverage_enabled": True,
        },
        "crisis_off": {
            **full_mutations,
            "scanner_enabled": True,
            "crisis_leverage_enabled": False,
        },
        "r3_reference": {
            **r3_mutations,
            "scanner_enabled": False,
            "crisis_leverage_enabled": True,
        },
    }

    summaries = {}
    for name, mutations in scenarios.items():
        cfg = mutate_meta_config(MetaConfig(), mutations) if mutations else MetaConfig()
        signals = run_signal_engine(
            macro_df=macro_df,
            strat_ret_df=strat_ret_df,
            market_df=market_df,
            growth_feature="GROWTH",
            inflation_feature="INFLATION",
            cfg=cfg,
        )
        result = simulate_portfolio(
            signals,
            strat_ret_df,
            sim_cfg,
            start_date=start,
            end_date=end,
        )
        summaries[name] = summarize_2022_validation(
            signals,
            result,
            scanner_threshold=cfg.scanner_threshold,
        )

    verdict = validate_step7_outcome(
        summaries["full_stack"],
        summaries["scanner_off"],
        summaries["crisis_off"],
        summaries["r3_reference"],
    )

    print("\n=== Validate 2022 ===")
    print(
        f"{'Scenario':<14s} {'FirstAlert':>12s} {'pCrisis':>8s} "
        f"{'PeakFeb':>8s} {'Ret2022':>9s} {'DD2022':>8s} {'FullDD':>8s}"
    )
    print("-" * 82)
    for name, summary in summaries.items():
        first_alert = summary["first_jan_alert"] or "-"
        print(
            f"{name:<14s} {first_alert:>12s} {summary['p_crisis_feb25']:>8.3f} "
            f"{summary['p_crisis_peak_feb']:>8.3f} {summary['return_2022']:>+8.1%} "
            f"{summary['max_dd_2022']:>7.1%} {summary['max_dd_full_window']:>7.1%}"
        )

    print("\nAcceptance:")
    print(f"  Jan alert in target window: {verdict['alert_ok']}")
    print(f"  p_crisis >= 0.5 by 2022-02-25: {verdict['crisis_ok']}")
    print(f"  Full stack beats both ablations on 2022 max DD: {verdict['dd_2022_ok']}")
    print(f"  Full stack beats both ablations on full-window max DD: {verdict['dd_full_window_ok']}")
    print(f"  Full stack beats R3 on 2022 return: {verdict['return_ok']}")
    print(f"  PASS: {verdict['passed']}")

    print("\nDeltas vs comparisons:")
    print(
        "  vs scanner_off: "
        f"FullDD {summaries['full_stack']['max_dd_full_window'] - summaries['scanner_off']['max_dd_full_window']:+.2%}, "
        f"Ret2022 {summaries['full_stack']['return_2022'] - summaries['scanner_off']['return_2022']:+.2%}"
    )
    print(
        "  vs crisis_off: "
        f"FullDD {summaries['full_stack']['max_dd_full_window'] - summaries['crisis_off']['max_dd_full_window']:+.2%}, "
        f"Ret2022 {summaries['full_stack']['return_2022'] - summaries['crisis_off']['return_2022']:+.2%}"
    )
    print(
        "  vs r3_reference: "
        f"FullDD {summaries['full_stack']['max_dd_full_window'] - summaries['r3_reference']['max_dd_full_window']:+.2%}, "
        f"Ret2022 {summaries['full_stack']['return_2022'] - summaries['r3_reference']['return_2022']:+.2%}"
    )

    output_dir = Path("backtests/regime/auto/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validate_2022.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "sources": {
                    "full_stack": full_source,
                    "r3_reference": r3_source,
                },
                "scenarios": summaries,
                "verdict": verdict,
                "deltas": {
                    "vs_scanner_off": {
                        "max_dd_2022": summaries["full_stack"]["max_dd_2022"]
                        - summaries["scanner_off"]["max_dd_2022"],
                        "max_dd_full_window": summaries["full_stack"]["max_dd_full_window"]
                        - summaries["scanner_off"]["max_dd_full_window"],
                        "return_2022": summaries["full_stack"]["return_2022"]
                        - summaries["scanner_off"]["return_2022"],
                    },
                    "vs_crisis_off": {
                        "max_dd_2022": summaries["full_stack"]["max_dd_2022"]
                        - summaries["crisis_off"]["max_dd_2022"],
                        "max_dd_full_window": summaries["full_stack"]["max_dd_full_window"]
                        - summaries["crisis_off"]["max_dd_full_window"],
                        "return_2022": summaries["full_stack"]["return_2022"]
                        - summaries["crisis_off"]["return_2022"],
                    },
                    "vs_r3_reference": {
                        "max_dd_2022": summaries["full_stack"]["max_dd_2022"]
                        - summaries["r3_reference"]["max_dd_2022"],
                        "max_dd_full_window": summaries["full_stack"]["max_dd_full_window"]
                        - summaries["r3_reference"]["max_dd_full_window"],
                        "return_2022": summaries["full_stack"]["return_2022"]
                        - summaries["r3_reference"]["return_2022"],
                    },
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {output_path}")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="regime-backtest",
        description="Regime Prediction Backtesting & Optimization",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Shared args
    def add_common(p):
        p.add_argument("--data-dir", default="backtests/regime/data/raw",
                       help="Path to data directory")
        p.add_argument("--equity", type=float, default=100_000.0,
                       help="Initial equity (default: 100,000)")
        p.add_argument("--cost-bps", type=float, default=5.0,
                       help="Rebalance cost in bps (default: 5)")

    def add_preset_arg(
        p,
        default_preset: str,
        help_label: str = "research",
        override_flag: str | None = "--mutations-json",
    ) -> None:
        suffix = (
            f"; ignored when {override_flag} is passed"
            if override_flag
            else ""
        )
        p.add_argument(
            "--preset",
            choices=preset_choices(),
            default=None,
            help=(
                f"Named {help_label} preset "
                f"(default: {default_preset}{suffix})"
            ),
        )

    def add_output_dir_arg(p, default_dir: Path = DEFAULT_PHASE_OUTPUT_DIR) -> None:
        p.add_argument(
            "--output-dir",
            default=str(default_dir),
            help=f"Output directory for phase state/results (default: {default_dir})",
        )

    # download
    dl = sub.add_parser("download", help="Download data from FRED + yfinance")
    add_common(dl)

    seed_refresh = sub.add_parser(
        "seed-refresh",
        help="Refresh deployment seed parquets and write manifest",
    )
    add_common(seed_refresh)
    seed_refresh.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only fingerprint and validate existing parquet files",
    )

    seed_validate = sub.add_parser(
        "seed-validate",
        help="Validate deployment seed parquets and manifest",
    )
    add_common(seed_validate)
    seed_validate.add_argument(
        "--require-manifest",
        action="store_true",
        help="Fail when regime_seed_manifest.json is missing",
    )
    seed_validate.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip manifest sha256 comparisons",
    )

    # run
    run = sub.add_parser("run", help="Run single backtest")
    add_common(run)
    run.add_argument("--diagnostics", action="store_true",
                     help="Print extended diagnostics")
    run.add_argument("--mutations-json", default=None,
                     help="Path to mutations JSON (e.g., greedy_optimal.json)")

    # optimize
    opt = sub.add_parser("optimize", help="Greedy forward selection")
    add_common(opt)
    opt.set_defaults(cost_bps=0.0)  # zero costs: measure prediction quality, not trading profit
    opt.add_argument("--max-rounds", type=int, default=50,
                     help="Max greedy rounds (default: 50)")
    opt.add_argument("--max-workers", type=int, default=None,
                     help="Parallel workers (default: cpu_count - 1)")
    opt.add_argument("--min-delta", type=float, default=0.001,
                     help="Min score improvement to accept (default: 0.001)")
    opt.add_argument("--prune-threshold", type=float, default=-0.10,
                     help="Prune candidates with delta below this; also prunes failures (default: -0.10)")
    opt.add_argument("--resume", nargs="?", const="auto", default=None,
                     help="Resume from previous result (default: greedy_optimal.json, or pass path)")

    # walk-forward
    wf = sub.add_parser("walk-forward", help="Walk-forward validation")
    add_common(wf)
    wf.add_argument("--test-years", type=int, default=2,
                    help="Test window in years (default: 2)")
    wf.add_argument("--max-rounds", type=int, default=50,
                    help="Max greedy rounds per fold (default: 50)")
    wf.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers (default: cpu_count - 1)")
    wf.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    wf.add_argument("--diagnostics", action="store_true",
                    help="Generate full diagnostics report with WF summary")
    wf.add_argument("--mutations-json", default=None,
                    help="Path to fixed mutations JSON -- skip per-fold optimization, "
                         "validate this config across folds (preset export or raw mutations)")

    # phase-run
    pr = sub.add_parser("phase-run", help="Run single phase of multi-phase optimization")
    add_common(pr)
    pr.set_defaults(cost_bps=0.0)
    pr.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                    help="Phase number (1-4)")
    pr.add_argument("--max-rounds", type=int, default=20,
                    help="Max greedy rounds (default: 20)")
    pr.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers")
    pr.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    pr.add_argument("--prune-threshold", type=float, default=-0.10,
                    help="Prune candidates below this delta (default: -0.10)")
    pr.add_argument("--candidate-timeout", type=float, default=600.0,
                    help="Per-candidate timeout in seconds (default: 600)")
    pr.add_argument("--mutations-json", default=None,
                    help="Path to baseline mutations JSON (overrides --preset on fresh starts)")
    add_preset_arg(pr, DEFAULT_RESEARCH_PRESET, help_label="phase baseline", override_flag=None)
    add_candidate_profile_arg(pr)
    add_output_dir_arg(pr)

    # phase-auto
    pa = sub.add_parser("phase-auto",
                        help="Run all remaining phases with analysis-driven orchestration")
    add_common(pa)
    pa.set_defaults(cost_bps=0.0)
    pa.add_argument("--max-rounds", type=int, default=20,
                    help="Max greedy rounds per phase (default: 20)")
    pa.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers")
    pa.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    pa.add_argument("--prune-threshold", type=float, default=-0.10,
                    help="Prune candidates below this delta (default: -0.10)")
    pa.add_argument("--candidate-timeout", type=float, default=600.0,
                    help="Per-candidate timeout in seconds (default: 600)")
    pa.add_argument("--max-retries", type=int, default=1,
                    help="Max retries per phase if analysis recommends rerun (default: 1)")
    pa.add_argument("--mutations-json", default=None,
                    help="Path to baseline mutations JSON (overrides --preset on fresh starts)")
    add_preset_arg(pa, DEFAULT_RESEARCH_PRESET, help_label="phase baseline", override_flag=None)
    add_candidate_profile_arg(pa)
    add_output_dir_arg(pa)

    # step9-optimize
    s9 = sub.add_parser(
        "step9-optimize",
        help="Run the assessment Step 9 R6 optimization in an isolated output dir",
    )
    add_common(s9)
    s9.set_defaults(cost_bps=0.0)
    s9.add_argument("--max-rounds", type=int, default=20,
                    help="Max greedy rounds per phase (default: 20)")
    s9.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers")
    s9.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    s9.add_argument("--prune-threshold", type=float, default=-0.10,
                    help="Prune candidates below this delta (default: -0.10)")
    s9.add_argument("--candidate-timeout", type=float, default=600.0,
                    help="Per-candidate timeout in seconds (default: 600)")
    s9.add_argument("--max-retries", type=int, default=0,
                    help="Max retries per phase if analysis recommends rerun (default: 0)")
    add_output_dir_arg(s9, DEFAULT_STEP9_OUTPUT_DIR)

    # r7-optimize
    r7 = sub.add_parser(
        "r7-optimize",
        help="Run the R7 overlay recalibration optimization",
    )
    add_common(r7)
    r7.set_defaults(cost_bps=0.0)
    r7.add_argument("--max-rounds", type=int, default=20,
                    help="Max greedy rounds per phase (default: 20)")
    r7.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers")
    r7.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    r7.add_argument("--prune-threshold", type=float, default=-0.10,
                    help="Prune candidates below this delta (default: -0.10)")
    r7.add_argument("--candidate-timeout", type=float, default=600.0,
                    help="Per-candidate timeout in seconds (default: 600)")
    r7.add_argument("--max-retries", type=int, default=0,
                    help="Max retries per phase if analysis recommends rerun (default: 0)")
    add_output_dir_arg(r7, DEFAULT_R7_OUTPUT_DIR)

    # r8-optimize
    r8 = sub.add_parser(
        "r8-optimize",
        help="Run the R8 two-model architecture optimization",
    )
    add_common(r8)
    r8.set_defaults(cost_bps=0.0)
    r8.add_argument("--max-rounds", type=int, default=20)
    r8.add_argument("--max-workers", type=int, default=None)
    r8.add_argument("--min-delta", type=float, default=0.001)
    r8.add_argument("--prune-threshold", type=float, default=-0.10)
    r8.add_argument("--candidate-timeout", type=float, default=600.0)
    r8.add_argument("--max-retries", type=int, default=0)
    add_output_dir_arg(r8, DEFAULT_R8_OUTPUT_DIR)

    # r9-budget-optimize
    r9 = sub.add_parser(
        "r9-budget-optimize",
        help="Run the R9 budget-only allocation optimization",
    )
    add_common(r9)
    r9.set_defaults(cost_bps=0.0)
    r9.add_argument("--max-rounds", type=int, default=8,
                    help="Max greedy rounds (default: 8)")
    r9.add_argument("--max-workers", type=int, default=None,
                    help="Parallel workers")
    r9.add_argument("--min-delta", type=float, default=0.001,
                    help="Min score improvement (default: 0.001)")
    r9.add_argument("--prune-threshold", type=float, default=-0.10,
                    help="Prune candidates below this delta (default: -0.10)")
    r9.add_argument("--candidate-timeout", type=float, default=600.0,
                    help="Per-candidate timeout in seconds (default: 600)")
    r9.add_argument("--max-retries", type=int, default=0,
                    help="Max retries per phase if analysis recommends rerun (default: 0)")
    add_output_dir_arg(r9, DEFAULT_R9_OUTPUT_DIR)

    # phase-gate
    pg = sub.add_parser("phase-gate", help="Check phase success gate")
    add_common(pg)
    pg.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help="Phase number (1-5)")
    add_output_dir_arg(pg)

    # phase-diagnostics
    pd_cmd = sub.add_parser("phase-diagnostics", help="Print phase diagnostics")
    pd_cmd.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Phase number (1-5)")
    add_output_dir_arg(pd_cmd)

    # historical-validate
    hv = sub.add_parser("historical-validate", help="Historical regime validation")
    add_common(hv)
    hv.add_argument("--mutations-json", default=None,
                    help="Path to mutations JSON (overrides --preset)")
    add_preset_arg(hv, DEFAULT_RESEARCH_PRESET)

    # scanner-validate
    sv = sub.add_parser("scanner-validate", help="Validate leading indicator scanner")
    add_common(sv)
    sv.add_argument("--mutations-json", default=None,
                    help="Path to mutations JSON (overrides --preset)")
    sv.add_argument("--diagnostics", action="store_true",
                    help="Print full diagnostics report with scanner section")
    add_preset_arg(sv, DEFAULT_RESEARCH_PRESET)

    # calibration-sweep
    cs = sub.add_parser("calibration-sweep", help="Run the Step 6 calibration sweep")
    add_common(cs)
    cs.add_argument("--mutations-json", default=None,
                    help="Path to base mutations JSON (overrides --preset)")
    add_preset_arg(cs, DEFAULT_REFERENCE_PRESET, help_label="calibration baseline")

    # validate-2022
    v22 = sub.add_parser("validate-2022", help="Run the Step 7 targeted 2022 validation")
    add_common(v22)
    v22.add_argument("--mutations-json", default=None,
                     help="Path to full-stack mutations JSON (overrides --preset)")
    v22.add_argument("--r3-mutations-json", default=None,
                     help="Path to R3 reference mutations JSON (overrides --r3-preset)")
    add_preset_arg(v22, DEFAULT_RESEARCH_PRESET, help_label="validate-2022 full-stack")
    v22.add_argument(
        "--r3-preset",
        choices=preset_choices(),
        default=None,
        help=(
            f"Named R3 comparison preset "
            f"(default: {DEFAULT_REFERENCE_PRESET}; ignored when --r3-mutations-json is passed)"
        ),
    )

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "seed-refresh":
        cmd_seed_refresh(args)
    elif args.command == "seed-validate":
        cmd_seed_validate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "walk-forward":
        cmd_walk_forward(args)
    elif args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "step9-optimize":
        cmd_step9_optimize(args)
    elif args.command == "r7-optimize":
        cmd_r7_optimize(args)
    elif args.command == "r8-optimize":
        cmd_r8_optimize(args)
    elif args.command == "r9-budget-optimize":
        cmd_r9_budget_optimize(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    elif args.command == "historical-validate":
        cmd_historical_validate(args)
    elif args.command == "scanner-validate":
        cmd_scanner_validate(args)
    elif args.command == "calibration-sweep":
        cmd_calibration_sweep(args)
    elif args.command == "validate-2022":
        cmd_validate_2022(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
