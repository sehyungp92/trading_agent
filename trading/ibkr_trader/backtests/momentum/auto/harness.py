"""Core auto-backtesting harness for momentum strategies.

Loads data once, caches baselines, runs all experiments, and generates reports.
Reuses existing CLI data loaders for NQDTC/Vdubus engines.
Supports parallel experiment execution via multiprocessing.Pool.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from backtests.momentum.auto.config_mutator import (
    extract_passthrough_mutations,
    mutate_nqdtc_config,
    mutate_portfolio_config,
    mutate_vdubus_config,
)
from backtests.momentum.auto.experiments import Experiment, build_experiment_queue
from backtests.momentum.auto.report import generate_report
from backtests.momentum.auto.results_tracker import ExperimentResult, ResultsTracker
from backtests.momentum.auto.robustness import run_robustness
from backtests.momentum.auto.scoring import CompositeScore, composite_score, extract_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 parallel execution — worker-process-local state and functions
# ---------------------------------------------------------------------------
# Mirrors greedy_optimize.py pattern: module-level dict populated once per
# worker by Pool(initializer=...), reused across all tasks in that worker.
# Each worker loads bar data independently (~6-10s startup cost, amortised
# across all experiments assigned to it).

_p1: dict = {}  # Populated by _init_phase1_worker in each child process


def _init_phase1_worker(data_dir_str: str, equity: float) -> None:
    """Initialize a Phase 1 worker. Loads all bar data once per process."""
    from backtests.momentum.cli import (
        _load_nqdtc_data, _load_vdubus_data,
    )
    data_dir = Path(data_dir_str)
    _p1.update({
        "nqdtc": _load_nqdtc_data("NQ", data_dir),
        "vdubus": _load_vdubus_data("NQ", data_dir),
        "equity": equity,
        "data_dir": data_dir_str,
        "trade_cache": {},  # strategy -> default trades, lazy for portfolio reuse
    })


def _p1_default_config(strategy: str):
    """Build default backtest config in worker."""
    eq = _p1["equity"]
    dd = Path(_p1["data_dir"])
    if strategy == "nqdtc":
        from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
        return NQDTCBacktestConfig(
            symbols=["MNQ"], initial_equity=eq, data_dir=dd, fixed_qty=10,
        )
    elif strategy == "vdubus":
        from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
        return VdubusBacktestConfig(
            initial_equity=eq, data_dir=dd, fixed_qty=10,
            flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _p1_apply_mutations(strategy: str, config, mutations: dict):
    """Apply config mutations in worker."""
    from backtests.momentum.auto.config_mutator import (
        mutate_nqdtc_config, mutate_vdubus_config,
    )
    if strategy == "nqdtc":
        return mutate_nqdtc_config(config, mutations)
    elif strategy == "vdubus":
        return mutate_vdubus_config(config, mutations)
    raise ValueError(f"Unknown strategy: {strategy}")


def _p1_run_engine(strategy: str, config) -> tuple[list, np.ndarray, np.ndarray]:
    """Run backtest engine in worker process using worker-local data."""
    d = _p1[strategy]
    if strategy == "nqdtc":
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        engine = NQDTCEngine(symbol="MNQ", bt_config=config)
        r = engine.run(d["five_min_bars"], d["thirty_min"], d["hourly"], d["four_hour"],
                       d["daily"], d["thirty_min_idx_map"], d["hourly_idx_map"],
                       d["four_hour_idx_map"], d["daily_idx_map"],
                       daily_es=d.get("daily_es"), daily_es_idx_map=d.get("daily_es_idx_map"))
        return r.trades, r.equity_curve, r.timestamps
    elif strategy == "vdubus":
        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        engine = VdubusEngine(symbol="NQ", bt_config=config)
        r = engine.run(d["bars_15m"], d.get("bars_5m"), d["hourly"], d["daily_es"],
                       d["hourly_idx_map"], d["daily_es_idx_map"], d.get("five_to_15_idx_map"))
        return r.trades, r.equity_curve, r.time_series
    raise ValueError(f"Unknown strategy: {strategy}")


def _p1_default_trades(strategy: str) -> list:
    """Get/cache default engine trades in worker (for portfolio experiments).

    First call runs the default engine; subsequent calls return the cached list.
    This avoids re-running all 3 engines for every portfolio experiment that
    only changes portfolio-level config (heat cap, stops, etc.).
    """
    cache = _p1["trade_cache"]
    if strategy not in cache:
        config = _p1_default_config(strategy)
        trades, _, _ = _p1_run_engine(strategy, config)
        cache[strategy] = trades
    return cache[strategy]


def _p1_decide(score_total: float, score_rejected: bool, baseline_total: float,
               is_ablation: bool, robust: bool) -> str:
    """Compute experiment status (mirrors ResultsTracker.decide logic)."""
    if score_rejected:
        return "DISCARD"
    delta = (score_total - baseline_total) / baseline_total if baseline_total > 0 else 0.0
    if is_ablation and abs(delta) < 1e-10:
        return "UNWIRED"
    if delta >= 0.05:
        return "APPROVE" if robust else "TEST_FURTHER"
    if delta >= 0.02:
        return "TEST_FURTHER"
    return "DISCARD"


def _run_experiment_worker(task: dict) -> dict:
    """Execute one experiment in a worker process.

    Args:
        task: Serialisable dict with experiment definition + baseline info.

    Returns:
        Dict matching ExperimentResult fields plus an 'error' key.
    """
    from backtests.momentum.auto.scoring import composite_score, extract_metrics

    exp_id = task["id"]
    strategy = task["strategy"]
    mutations = task["mutations"]
    baseline_total = task["baseline_total"]
    equity = _p1["equity"]
    is_ablation = task["type"] == "ABLATION"

    try:
        if strategy == "portfolio":
            return _run_portfolio_in_worker(task)

        # Strategy-level experiment
        config = _p1_default_config(strategy)
        config = _p1_apply_mutations(strategy, config, mutations)
        trades, eq, ts = _p1_run_engine(strategy, config)

        metrics = extract_metrics(trades, eq, ts, equity)
        score = composite_score(metrics, equity, strategy=strategy, equity_curve=eq)
        delta_pct = (score.total - baseline_total) / baseline_total if baseline_total > 0 else 0.0

        # Robustness (when enabled and experiment looks promising)
        robust = False
        if not task["skip_robustness"] and not score.rejected and delta_pct >= 0.02:
            from backtests.momentum.auto.robustness import run_robustness as _run_rob

            exp_obj = Experiment(
                id=exp_id, type=task["type"], strategy=strategy,
                description=task["description"], hypothesis=task.get("hypothesis", ""),
                priority=task.get("priority", 1), mutations=mutations,
            )

            def run_fn(muts):
                cfg = _p1_default_config(strategy)
                cfg = _p1_apply_mutations(strategy, cfg, muts)
                t, e, s = _p1_run_engine(strategy, cfg)
                return t, e, s, equity

            report = _run_rob(exp_obj, trades, eq, ts, equity, score.total, run_fn)
            robust = report.passes_all

        status = _p1_decide(score.total, score.rejected, baseline_total, is_ablation, robust)
        return {
            "experiment_id": exp_id, "strategy": strategy, "type": task["type"],
            "baseline_score": baseline_total, "experiment_score": score.total,
            "delta_pct": delta_pct, "robust": robust, "status": status,
            "description": task["description"], "error": None,
        }
    except Exception:
        import traceback as _tb
        return {
            "experiment_id": exp_id, "strategy": strategy, "type": task["type"],
            "baseline_score": baseline_total, "experiment_score": 0.0,
            "delta_pct": 0.0, "robust": False, "status": "CRASH",
            "description": task["description"], "error": _tb.format_exc(),
        }


def _run_portfolio_in_worker(task: dict) -> dict:
    """Execute a portfolio experiment in a worker process.

    Reuses per-worker cached default engine trades when no per-strategy
    passthrough mutations are present — avoids redundant engine runs.
    """
    from backtests.momentum.auto.config_mutator import (
        extract_passthrough_mutations, mutate_portfolio_config,
    )
    from backtests.momentum.auto.scoring import composite_score, extract_metrics
    from backtests.momentum.config_portfolio import PortfolioBacktestConfig
    from backtests.momentum.engine.portfolio_engine import PortfolioBacktester

    mutations = task["mutations"]
    baseline_total = task["baseline_total"]

    # Build portfolio config
    if "preset" in mutations:
        from backtests.momentum.config_portfolio import PRESETS
        preset_name = mutations["preset"]
        if preset_name not in PRESETS:
            return {
                "experiment_id": task["id"], "strategy": "portfolio",
                "type": task["type"], "baseline_score": baseline_total,
                "experiment_score": 0.0, "delta_pct": 0.0, "robust": False,
                "status": "CRASH", "description": task["description"],
                "error": f"Unknown preset: {preset_name}",
            }
        portfolio_cfg = PortfolioBacktestConfig(portfolio=PRESETS[preset_name]())
    else:
        from libs.oms.config.portfolio_config import make_10k_v6_config
        portfolio_cfg = PortfolioBacktestConfig(portfolio=make_10k_v6_config())
        portfolio_cfg = mutate_portfolio_config(portfolio_cfg, mutations)

    # Get per-strategy trades — reuse cached defaults when no passthrough muts
    nqdtc_muts = extract_passthrough_mutations(mutations, "nqdtc")
    vdubus_muts = extract_passthrough_mutations(mutations, "vdubus")

    def _get_trades(strat, muts, run_flag):
        if not run_flag or _p1.get(strat) is None:
            return []
        if muts:
            cfg = _p1_default_config(strat)
            cfg = _p1_apply_mutations(strat, cfg, muts)
            t, _, _ = _p1_run_engine(strat, cfg)
            return t
        return _p1_default_trades(strat)

    nqdtc_trades = _get_trades("nqdtc", nqdtc_muts, portfolio_cfg.run_nqdtc)
    vdubus_trades = _get_trades("vdubus", vdubus_muts, portfolio_cfg.run_vdubus)

    backtester = PortfolioBacktester(portfolio_cfg)
    result = backtester.run(nqdtc_trades=nqdtc_trades, vdubus_trades=vdubus_trades)

    eq = result.equity_curve
    ts = (np.array([dt.timestamp() for dt in result.equity_timestamps])
          if result.equity_timestamps else np.array([]))
    init_eq = portfolio_cfg.portfolio.initial_equity

    metrics = extract_metrics(result.trades, eq, ts, init_eq)
    score = composite_score(metrics, init_eq, strategy="portfolio", equity_curve=eq)
    delta_pct = (score.total - baseline_total) / baseline_total if baseline_total > 0 else 0.0
    status = _p1_decide(score.total, score.rejected, baseline_total, False, False)

    return {
        "experiment_id": task["id"], "strategy": "portfolio", "type": task["type"],
        "baseline_score": baseline_total, "experiment_score": score.total,
        "delta_pct": delta_pct, "robust": False, "status": status,
        "description": task["description"], "error": None,
    }


# ---------------------------------------------------------------------------
# Main harness class
# ---------------------------------------------------------------------------

class MomentumAutoHarness:
    """Automated experiment runner for the momentum strategy family."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        initial_equity: float = 10_000.0,
        verbose: bool = False,
        max_workers: int = 1,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.initial_equity = initial_equity
        self.verbose = verbose
        self.max_workers = max_workers

        self.tracker = ResultsTracker(output_dir)

        # Data caches (loaded once in main process for baselines)
        self._nqdtc_data: dict | None = None
        self._vdubus_data: dict | None = None

        # Baseline caches
        self._baselines: dict[str, CompositeScore] = {}
        self._baseline_trades: dict[str, list] = {}
        self._portfolio_baseline: CompositeScore | None = None

        # Default configs
        self._default_configs: dict[str, object] = {}
        self._experiment_queue_cache: dict[str, list[Experiment]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        strategy_filter: str = "all",
        experiment_ids: list[str] | None = None,
        skip_robustness: bool = False,
        resume: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Run the full experiment pipeline.

        Args:
            strategy_filter: "nqdtc", "vdubus", "portfolio", or "all"
            experiment_ids: Specific experiment IDs to run (None = all)
            skip_robustness: Skip robustness checks for faster scan
            resume: Skip experiments already in results.tsv
            max_workers: Override instance-level worker count (None = use self.max_workers)
        """
        n_workers = max_workers if max_workers is not None else self.max_workers

        # 1. Load data ONCE (main process needs it for baselines)
        print("Loading bar data (this may take a moment)...")
        t0 = time.time()
        self._load_data(strategy_filter)
        print(f"Data loaded in {time.time() - t0:.1f}s")

        # 2. Build experiment queue
        if strategy_filter not in self._experiment_queue_cache:
            self._experiment_queue_cache[strategy_filter] = build_experiment_queue(strategy_filter)
        experiments = list(self._experiment_queue_cache[strategy_filter])
        if experiment_ids:
            id_set = set(experiment_ids)
            experiments = [e for e in experiments if e.id in id_set]

        if not experiments:
            print("No experiments to run.")
            self._generate_report()
            return

        # Skip completed if resuming
        completed = self.tracker.completed_ids() if resume else set()
        pending = [e for e in experiments if e.id not in completed]
        if resume and (len(experiments) - len(pending)) > 0:
            print(f"Resuming: skipping {len(experiments) - len(pending)} completed experiments")

        # 3. Compute baselines (always in main process)
        print("\nComputing baselines...")
        strategies_needed = set()
        for e in pending:
            if e.strategy == "portfolio":
                strategies_needed.update(("nqdtc", "vdubus"))
            else:
                strategies_needed.add(e.strategy)

        failed_baselines: set[str] = set()
        for strategy in sorted(strategies_needed):
            if not self._run_baseline(strategy):
                failed_baselines.add(strategy)
                print(f"  WARNING: {strategy} baseline FAILED")

        # Eagerly compute portfolio baseline if any portfolio experiments pending
        has_portfolio = any(e.strategy == "portfolio" for e in pending)
        if has_portfolio and self._portfolio_baseline is None:
            self._run_portfolio_baseline()

        # Filter out experiments whose baselines failed
        all_strategy_baselines_failed = {"nqdtc", "vdubus"}.issubset(failed_baselines)
        if failed_baselines:
            pending = [
                e for e in pending
                if e.strategy not in failed_baselines
                and not (e.strategy == "portfolio" and all_strategy_baselines_failed)
            ]

        if not pending:
            print("No experiments to run after filtering.")
            self._generate_report()
            return

        # 4. Run experiments
        total = len(pending)
        print(f"\nRunning {total} experiments ({len(completed)} already done)...")

        if n_workers > 1:
            self._run_experiments_parallel(pending, skip_robustness, n_workers)
        else:
            self._run_experiments_sequential(pending, skip_robustness)

        # 5. Generate report
        self._generate_report()

    # ------------------------------------------------------------------
    # Sequential experiment execution (single process)
    # ------------------------------------------------------------------

    def _run_experiments_sequential(
        self, pending: list[Experiment], skip_robustness: bool,
    ) -> None:
        """Run experiments one at a time in the main process."""
        total = len(pending)
        cumulative_time = 0.0
        for i, experiment in enumerate(pending, 1):
            t1 = time.time()
            try:
                if experiment.strategy == "portfolio":
                    self._run_portfolio_experiment(experiment, skip_robustness)
                else:
                    self._run_experiment(experiment, skip_robustness)
            except Exception:
                logger.error("Experiment %s crashed (mutations=%s):\n%s",
                             experiment.id, experiment.mutations, traceback.format_exc())
                self.tracker.record(ExperimentResult(
                    experiment_id=experiment.id,
                    strategy=experiment.strategy,
                    type=experiment.type,
                    baseline_score=self._baselines.get(experiment.strategy, CompositeScore(0, 0, 0, 0, 0)).total,
                    experiment_score=0.0,
                    delta_pct=0.0,
                    robust=False,
                    status="CRASH",
                    description=experiment.description,
                ))

            elapsed = time.time() - t1
            cumulative_time += elapsed
            avg_per_exp = cumulative_time / i
            eta_s = avg_per_exp * (total - i)
            eta_str = f"{eta_s / 60:.0f}m" if eta_s >= 60 else f"{eta_s:.0f}s"
            print(f"  [{i}/{total}] {experiment.id} ({elapsed:.1f}s) ETA: {eta_str}")

    # ------------------------------------------------------------------
    # Parallel experiment execution (multiprocessing)
    # ------------------------------------------------------------------

    def _run_experiments_parallel(
        self, pending: list[Experiment], skip_robustness: bool, n_workers: int,
    ) -> None:
        """Run experiments across a multiprocessing.Pool.

        Each worker loads bar data independently (once), then processes
        experiments assigned to it. Portfolio experiments reuse per-worker
        cached default engine results for strategies without passthrough
        mutations — avoiding redundant engine runs.
        """
        baseline_totals: dict[str, float] = {s: sc.total for s, sc in self._baselines.items()}
        if self._portfolio_baseline:
            baseline_totals["portfolio"] = self._portfolio_baseline.total

        tasks = []
        for e in pending:
            bl = baseline_totals.get(e.strategy, 0.0)
            tasks.append({
                "id": e.id, "strategy": e.strategy, "type": e.type,
                "mutations": e.mutations, "description": e.description,
                "hypothesis": e.hypothesis, "priority": e.priority,
                "skip_robustness": skip_robustness, "baseline_total": bl,
            })

        total = len(tasks)
        print(f"  Spawning {n_workers} workers (each loads data independently)...")

        pool = mp.Pool(
            processes=n_workers,
            initializer=_init_phase1_worker,
            initargs=(str(self.data_dir), self.initial_equity),
        )

        try:
            t_start = time.time()
            for i, result in enumerate(pool.imap_unordered(_run_experiment_worker, tasks), 1):
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = avg * (total - i)
                eta_str = f"{eta / 60:.0f}m" if eta >= 60 else f"{eta:.0f}s"

                if result["error"]:
                    logger.error("Experiment %s crashed:\n%s",
                                 result["experiment_id"], result["error"])

                delta_s = f"Δ={result['delta_pct']:+.1%}" if result["status"] != "CRASH" else "CRASH"
                print(f"  [{i}/{total}] {result['experiment_id']} "
                      f"({delta_s}) ETA: {eta_str}")

                self.tracker.record(ExperimentResult(
                    experiment_id=result["experiment_id"],
                    strategy=result["strategy"],
                    type=result["type"],
                    baseline_score=result["baseline_score"],
                    experiment_score=result["experiment_score"],
                    delta_pct=result["delta_pct"],
                    robust=result["robust"],
                    status=result["status"],
                    description=result["description"],
                ))
        finally:
            pool.close()
            pool.join()

    # ------------------------------------------------------------------
    # Data loading — reuse CLI loaders
    # ------------------------------------------------------------------

    def _load_data(self, strategy_filter: str) -> None:
        """Load bar data using existing CLI data loaders."""
        from backtests.momentum.cli import (
            _load_nqdtc_data,
            _load_vdubus_data,
        )

        _all = strategy_filter == "all"

        if (_all or strategy_filter in ("nqdtc", "portfolio")) and self._nqdtc_data is None:
            self._nqdtc_data = _load_nqdtc_data("NQ", self.data_dir)

        if (_all or strategy_filter in ("vdubus", "portfolio")) and self._vdubus_data is None:
            self._vdubus_data = _load_vdubus_data("NQ", self.data_dir)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _run_baseline(self, strategy: str) -> bool:
        """Compute and cache baseline score for a strategy."""
        if strategy in self._baselines:
            return True

        try:
            config = self._get_default_config(strategy)
            trades, eq, ts = self._run_engine(strategy, config)

            if not trades:
                logger.warning("Baseline %s produced no trades", strategy)
                return False

            metrics = extract_metrics(trades, eq, ts, self.initial_equity)
            score = composite_score(
                metrics, self.initial_equity,
                strategy=strategy, equity_curve=eq,
            )

            self._baselines[strategy] = score
            self._baseline_trades[strategy] = trades
            rejected = " REJECTED" if score.rejected else ""
            print(f"  {strategy} baseline: score={score.total:.4f}{rejected}, "
                  f"trades={metrics.total_trades}, PF={metrics.profit_factor:.2f}, "
                  f"DD={metrics.max_drawdown_pct:.1%}, net=${metrics.net_profit:,.0f}")
            return not score.rejected

        except Exception:
            logger.error("Baseline %s failed:\n%s", strategy, traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Experiment runners
    # ------------------------------------------------------------------

    def _run_experiment(self, experiment: Experiment, skip_robustness: bool) -> None:
        """Run a single strategy-level experiment."""
        strategy = experiment.strategy
        baseline = self._baselines[strategy]

        # Build mutated config
        config = self._get_default_config(strategy)
        config = self._apply_mutations(strategy, config, experiment.mutations)

        # Run engine
        trades, eq, ts = self._run_engine(strategy, config)

        # Score
        metrics = extract_metrics(trades, eq, ts, self.initial_equity)
        score = composite_score(
            metrics, self.initial_equity,
            strategy=strategy, equity_curve=eq,
        )

        # Delta
        if baseline.total == 0:
            delta_pct = 0.0
        else:
            delta_pct = (score.total - baseline.total) / baseline.total

        # Robustness
        robustness_report = None
        if not skip_robustness and not score.rejected and delta_pct >= 0.02:

            def run_fn(muts):
                cfg = self._get_default_config(strategy)
                cfg = self._apply_mutations(strategy, cfg, muts)
                t, e, s = self._run_engine(strategy, cfg)
                return t, e, s, self.initial_equity

            robustness_report = run_robustness(
                experiment, trades, eq, ts, self.initial_equity,
                score.total, run_fn,
            )

        # Route and record
        status = self.tracker.decide(baseline, score, robustness_report,
                                     is_ablation=(experiment.type == "ABLATION"))

        self.tracker.record(ExperimentResult(
            experiment_id=experiment.id,
            strategy=strategy,
            type=experiment.type,
            baseline_score=baseline.total,
            experiment_score=score.total,
            delta_pct=delta_pct,
            robust=robustness_report.passes_all if robustness_report else False,
            status=status,
            description=experiment.description,
        ))

    def _run_portfolio_experiment(
        self,
        experiment: Experiment,
        skip_robustness: bool,
    ) -> None:
        """Run a portfolio-level experiment using post-hoc PortfolioBacktester.

        Reuses cached baseline trades when no per-strategy passthrough
        mutations are present — avoids re-running engines needlessly.
        """
        from backtests.momentum.config_portfolio import PortfolioBacktestConfig
        from backtests.momentum.engine.portfolio_engine import PortfolioBacktester

        # Check for preset experiments
        if "preset" in experiment.mutations:
            from backtests.momentum.config_portfolio import PRESETS
            preset_name = experiment.mutations["preset"]
            if preset_name in PRESETS:
                portfolio_cfg = PortfolioBacktestConfig(portfolio=PRESETS[preset_name]())
            else:
                logger.warning("Unknown preset %s", preset_name)
                return
        else:
            portfolio_cfg = PortfolioBacktestConfig(
                portfolio=self._get_default_portfolio_config(),
            )
            portfolio_cfg = mutate_portfolio_config(portfolio_cfg, experiment.mutations)

        # Extract per-strategy mutations from portfolio experiment
        nqdtc_muts = extract_passthrough_mutations(experiment.mutations, "nqdtc")
        vdubus_muts = extract_passthrough_mutations(experiment.mutations, "vdubus")

        # Run individual engines — reuse cached baseline trades when possible
        nqdtc_trades = []
        vdubus_trades = []

        if portfolio_cfg.run_nqdtc and self._nqdtc_data is not None:
            if nqdtc_muts:
                cfg = self._get_default_config("nqdtc")
                cfg = self._apply_mutations("nqdtc", cfg, nqdtc_muts)
                nqdtc_trades, _, _ = self._run_engine("nqdtc", cfg)
            elif "nqdtc" in self._baseline_trades:
                nqdtc_trades = self._baseline_trades["nqdtc"]
            else:
                nqdtc_trades, _, _ = self._run_engine("nqdtc", self._get_default_config("nqdtc"))

        if portfolio_cfg.run_vdubus and self._vdubus_data is not None:
            if vdubus_muts:
                cfg = self._get_default_config("vdubus")
                cfg = self._apply_mutations("vdubus", cfg, vdubus_muts)
                vdubus_trades, _, _ = self._run_engine("vdubus", cfg)
            elif "vdubus" in self._baseline_trades:
                vdubus_trades = self._baseline_trades["vdubus"]
            else:
                vdubus_trades, _, _ = self._run_engine("vdubus", self._get_default_config("vdubus"))

        # Post-hoc portfolio simulation
        backtester = PortfolioBacktester(portfolio_cfg)
        result = backtester.run(nqdtc_trades=nqdtc_trades, vdubus_trades=vdubus_trades)

        # Score from portfolio result
        all_trades = result.trades
        eq = result.equity_curve
        ts = np.array([dt.timestamp() for dt in result.equity_timestamps]) if result.equity_timestamps else np.array([])
        init_eq = portfolio_cfg.portfolio.initial_equity

        metrics = extract_metrics(all_trades, eq, ts, init_eq)
        score = composite_score(metrics, init_eq, strategy="portfolio", equity_curve=eq)

        # Compute baseline if needed
        if self._portfolio_baseline is None:
            self._run_portfolio_baseline()

        baseline = self._portfolio_baseline or CompositeScore(0, 0, 0, 0, 0)

        # Delta
        if baseline.total == 0:
            delta_pct = 0.0
        else:
            delta_pct = (score.total - baseline.total) / baseline.total

        # Robustness (skip for portfolio — too expensive)
        robustness_report = None

        status = self.tracker.decide(baseline, score, robustness_report,
                                     is_ablation=False)

        self.tracker.record(ExperimentResult(
            experiment_id=experiment.id,
            strategy="portfolio",
            type=experiment.type,
            baseline_score=baseline.total,
            experiment_score=score.total,
            delta_pct=delta_pct,
            robust=False,
            status=status,
            description=experiment.description,
        ))

    def _run_portfolio_baseline(self) -> None:
        """Compute portfolio baseline using default configs."""
        from backtests.momentum.config_portfolio import PortfolioBacktestConfig
        from backtests.momentum.engine.portfolio_engine import PortfolioBacktester

        portfolio_cfg = PortfolioBacktestConfig(
            portfolio=self._get_default_portfolio_config(),
        )

        nqdtc_trades = self._baseline_trades.get("nqdtc", [])
        vdubus_trades = self._baseline_trades.get("vdubus", [])

        backtester = PortfolioBacktester(portfolio_cfg)
        result = backtester.run(nqdtc_trades=nqdtc_trades, vdubus_trades=vdubus_trades)

        eq = result.equity_curve
        ts = np.array([dt.timestamp() for dt in result.equity_timestamps]) if result.equity_timestamps else np.array([])
        init_eq = portfolio_cfg.portfolio.initial_equity

        metrics = extract_metrics(result.trades, eq, ts, init_eq)
        self._portfolio_baseline = composite_score(
            metrics, init_eq, strategy="portfolio", equity_curve=eq,
        )
        print(f"  portfolio baseline: score={self._portfolio_baseline.total:.4f}, "
              f"trades={metrics.total_trades}, PF={metrics.profit_factor:.2f}")

    # ------------------------------------------------------------------
    # Engine dispatch — exact signatures from cli.py
    # ------------------------------------------------------------------

    def _run_engine(
        self,
        strategy: str,
        config,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """Dispatch to the correct engine and return (trades, equity_curve, timestamps)."""
        if strategy == "nqdtc":
            return self._run_nqdtc(config)
        elif strategy == "vdubus":
            return self._run_vdubus(config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _run_nqdtc(self, config) -> tuple[list, np.ndarray, np.ndarray]:
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

        d = self._nqdtc_data
        engine = NQDTCEngine(symbol="MNQ", bt_config=config)
        result = engine.run(
            d["five_min_bars"], d["thirty_min"], d["hourly"], d["four_hour"], d["daily"],
            d["thirty_min_idx_map"], d["hourly_idx_map"], d["four_hour_idx_map"], d["daily_idx_map"],
            daily_es=d.get("daily_es"), daily_es_idx_map=d.get("daily_es_idx_map"),
        )
        return result.trades, result.equity_curve, result.timestamps

    def _run_vdubus(self, config) -> tuple[list, np.ndarray, np.ndarray]:
        from backtests.momentum.engine.vdubus_engine import VdubusEngine

        d = self._vdubus_data
        engine = VdubusEngine(symbol="NQ", bt_config=config)
        result = engine.run(
            d["bars_15m"], d.get("bars_5m"), d["hourly"], d["daily_es"],
            d["hourly_idx_map"], d["daily_es_idx_map"], d.get("five_to_15_idx_map"),
        )
        # NOTE: Vdubus uses .time_series not .timestamps
        return result.trades, result.equity_curve, result.time_series

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_default_config(self, strategy: str):
        """Get or create the default config for a strategy."""
        if strategy not in self._default_configs:
            if strategy == "nqdtc":
                from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
                self._default_configs[strategy] = NQDTCBacktestConfig(
                    symbols=["MNQ"],
                    initial_equity=self.initial_equity,
                    data_dir=self.data_dir,
                    fixed_qty=10,
                )
            elif strategy == "vdubus":
                from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
                self._default_configs[strategy] = VdubusBacktestConfig(
                    initial_equity=self.initial_equity,
                    data_dir=self.data_dir,
                    fixed_qty=10,
                    flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        # Return a fresh copy via replace to avoid state leakage
        return replace(self._default_configs[strategy])

    def _get_default_portfolio_config(self):
        """Get default PortfolioConfig for portfolio experiments."""
        from libs.oms.config.portfolio_config import make_10k_v6_config
        return make_10k_v6_config()

    def _apply_mutations(self, strategy: str, config, mutations: dict):
        """Apply mutations to a strategy config."""
        if strategy == "nqdtc":
            return mutate_nqdtc_config(config, mutations)
        elif strategy == "vdubus":
            return mutate_vdubus_config(config, mutations)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _generate_report(self) -> None:
        """Generate markdown report and write to output dir."""
        results = self.tracker.load_all()
        all_experiments = build_experiment_queue("all")
        report_text = generate_report(results, self._baselines, all_experiments)
        report_path = self.output_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport written to: {report_path}")
