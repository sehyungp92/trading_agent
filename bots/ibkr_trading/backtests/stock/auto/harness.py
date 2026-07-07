"""Core orchestrator for auto backtesting.

Loads data once, runs all experiments in priority order,
computes composite scores, checks robustness, and writes results.
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import numpy as np

from backtests.stock.auto.config_mutator import (
    mutate_alcb_config,
    mutate_iaric_config,
    mutate_portfolio_config,
)
from backtests.stock.auto.experiments import Experiment, build_experiment_queue
from backtests.stock.auto.report import generate_report
from backtests.stock.auto.results_tracker import ExperimentResult, ResultsTracker
from backtests.stock.auto.robustness import run_robustness
from backtests.stock.auto.scoring import (
    CompositeScore, composite_score, compute_r_multiples, extract_metrics,
)
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.config_portfolio import PortfolioBacktestConfig
from backtests.stock.models import TradeRecord

logger = logging.getLogger(__name__)


class AutoBacktestHarness:
    """Orchestrates automated experiment runs."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        start_date: str = "2024-01-01",
        end_date: str = "2026-03-01",
        initial_equity: float = 10_000.0,
        verbose: bool = False,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.verbose = verbose

        self.replay = None
        self.tracker = ResultsTracker(output_dir)

        # Baseline scores keyed by (strategy, tier)
        self._baselines: dict[tuple[str, int], CompositeScore] = {}
        self._baseline_trades: dict[tuple[str, int], list[TradeRecord]] = {}
        # Cached portfolio baseline keyed by tier
        self._portfolio_baselines: dict[int, CompositeScore] = {}

    def run_all(
        self,
        strategy_filter: str = "all",
        experiment_ids: list[str] | None = None,
        skip_robustness: bool = False,
        resume: bool = False,
    ) -> None:
        """Run the full experiment pipeline.

        Args:
            strategy_filter: "alcb", "iaric", or "all"
            experiment_ids: Specific experiment IDs to run (None = all)
            skip_robustness: Skip robustness checks for faster ablation scan
            resume: Skip experiments already in results.tsv
        """
        from backtests.stock.engine.research_replay import ResearchReplayEngine

        # 1. Load data ONCE
        print("Loading bar data (this may take a moment)...")
        t0 = time.time()
        self.replay = ResearchReplayEngine(data_dir=self.data_dir)
        self.replay.load_all_data()
        print(f"Data loaded in {time.time() - t0:.1f}s")

        # 2. Build experiment queue
        experiments = build_experiment_queue(strategy_filter)
        if experiment_ids:
            experiments = [e for e in experiments if e.id in set(experiment_ids)]

        if not experiments:
            print("No experiments to run.")
            return

        # Skip completed if resuming
        completed = self.tracker.completed_ids() if resume else set()
        pending = [e for e in experiments if e.id not in completed]
        if resume and len(completed) > 0:
            print(f"Resuming: {len(completed)} already done, {len(pending)} remaining")

        if not pending:
            print("All experiments already completed.")
            self._generate_report()
            return

        # 3. Run baselines for needed (strategy, tier) combos
        needed_keys = {(e.strategy, e.tier) for e in pending if e.strategy != "portfolio"}
        failed_baselines: set[tuple[str, int]] = set()
        for strategy, tier in sorted(needed_keys):
            if (strategy, tier) not in self._baselines:
                if not self._run_baseline(strategy, tier):
                    failed_baselines.add((strategy, tier))

        # Filter out experiments whose baselines crashed (not just rejected)
        crashed_baselines = {
            key for key in failed_baselines
            if key not in self._baselines  # crashed = never stored
        }
        if crashed_baselines:
            pending = [
                e for e in pending
                if e.strategy == "portfolio"
                or (e.strategy, e.tier) not in crashed_baselines
            ]

        if not pending:
            print("All experiments skipped (baselines failed or already completed).")
            self._generate_report()
            return

        # 4. Run experiments in priority order
        total = len(pending)
        for i, experiment in enumerate(pending, 1):
            print(f"\n[{i}/{total}] {experiment.id}: {experiment.description}")
            t_start = time.time()

            try:
                self._run_experiment(experiment, skip_robustness)
            except Exception:
                logger.error("Experiment %s crashed:\n%s",
                             experiment.id, traceback.format_exc())
                baseline_key = (experiment.strategy, experiment.tier)
                baseline_score = self._baselines.get(baseline_key, CompositeScore(0, 0, 0, 0, 0))
                self.tracker.record(ExperimentResult(
                    experiment_id=experiment.id,
                    strategy=experiment.strategy,
                    tier=experiment.tier,
                    type=experiment.type,
                    baseline_score=baseline_score.total,
                    experiment_score=0.0,
                    delta_pct=0.0,
                    robust=False,
                    status="CRASH",
                    description=experiment.description,
                ))

            elapsed = time.time() - t_start
            print(f"  Completed in {elapsed:.1f}s")

        # 5. Portfolio integration (if running "all" and we have approved changes)
        if strategy_filter == "all" and not experiment_ids:
            self._portfolio_integration()

        # 6. Generate report
        self._generate_report()

    def _run_baseline(self, strategy: str, tier: int) -> bool:
        """Run default config to establish baseline score.

        Returns True if baseline is valid (not rejected), False otherwise.
        """
        print(f"  Running baseline: {strategy} tier {tier}...")
        try:
            config = self._make_default_config(strategy, tier)
            trades, eq, ts = self._run_engine(strategy, tier, config)
        except Exception:
            logger.error("Baseline %s T%d crashed:\n%s",
                         strategy, tier, traceback.format_exc())
            print(f"  BASELINE CRASH: {strategy} T{tier} -- skipping all experiments for this combo")
            return False

        metrics = extract_metrics(trades, eq, ts, self.initial_equity)
        r_mult = compute_r_multiples(trades)
        score = composite_score(metrics, self.initial_equity, r_multiples=r_mult)

        if score.rejected:
            print(f"  BASELINE REJECTED: {strategy} T{tier} -- {score.reject_reason}")
            print(f"  Experiments will run individually (may overcome baseline limitations)")

        self._baselines[(strategy, tier)] = score
        self._baseline_trades[(strategy, tier)] = trades

        if not score.rejected:
            print(f"  Baseline {strategy} T{tier}: score={score.total:.4f} "
                  f"(trades={metrics.total_trades}, PF={metrics.profit_factor:.2f}, "
                  f"DD={metrics.max_drawdown_pct:.1%})")
        return not score.rejected

    def _run_experiment(self, experiment: Experiment, skip_robustness: bool) -> None:
        """Run a single experiment and record results."""
        if experiment.type == "PORTFOLIO":
            self._run_portfolio_experiment(experiment)
            return

        baseline_key = (experiment.strategy, experiment.tier)
        baseline = self._baselines[baseline_key]

        # Mutate config and run
        config = self._make_mutated_config(
            experiment.strategy, experiment.tier, experiment.mutations,
        )
        trades, eq, ts = self._run_engine(
            experiment.strategy, experiment.tier, config,
        )

        metrics = extract_metrics(trades, eq, ts, self.initial_equity)
        r_mult = compute_r_multiples(trades)
        score = composite_score(metrics, self.initial_equity, r_multiples=r_mult)

        # Compute delta
        if baseline.total == 0:
            delta_pct = 0.0
        else:
            delta_pct = (score.total - baseline.total) / baseline.total

        # Check for unwired ablation flags
        is_ablation = experiment.type == "ABLATION"
        if is_ablation and abs(delta_pct) < 1e-10:
            status = "UNWIRED"
            robust = False
            print(f"  → UNWIRED (delta=0, flag not checked in engine)")
        elif score.rejected:
            status = "DISCARD"
            robust = False
            print(f"  → DISCARD (rejected: {score.reject_reason})")
        else:
            # Robustness checks (if delta warrants)
            robustness = None
            if not skip_robustness and delta_pct >= 0.02:
                print(f"  Running robustness checks...")

                def run_fn(mutations):
                    cfg = self._make_mutated_config(
                        experiment.strategy, experiment.tier, mutations,
                    )
                    t, e, s = self._run_engine(
                        experiment.strategy, experiment.tier, cfg,
                    )
                    return t, e, s, self.initial_equity

                robustness = run_robustness(
                    experiment, score, trades, run_fn,
                    replay=self.replay,
                )

            status = self.tracker.decide(baseline, score, robustness, is_ablation)
            robust = robustness.passes_all if robustness else False

            status_msg = f"  → {status} (delta={delta_pct:+.2%}, score={score.total:.4f})"
            if robustness:
                status_msg += f" robust={robust}"
            print(status_msg)

        self.tracker.record(ExperimentResult(
            experiment_id=experiment.id,
            strategy=experiment.strategy,
            tier=experiment.tier,
            type=experiment.type,
            baseline_score=baseline.total,
            experiment_score=score.total,
            delta_pct=delta_pct,
            robust=robust,
            status=status,
            description=experiment.description,
        ))

    def _run_portfolio_experiment(self, experiment: Experiment) -> None:
        """Run a portfolio-level experiment."""
        from backtests.stock.engine.portfolio_engine import StockPortfolioEngine

        # Get baseline individual trades
        alcb_trades = self._baseline_trades.get(("alcb", experiment.tier), [])
        iaric_trades = self._baseline_trades.get(("iaric", experiment.tier), [])

        if not alcb_trades and not iaric_trades:
            print("  Skipping portfolio experiment -- no baseline trades available")
            self.tracker.record(ExperimentResult(
                experiment_id=experiment.id,
                strategy="portfolio",
                tier=experiment.tier,
                type="PORTFOLIO",
                baseline_score=0.0,
                experiment_score=0.0,
                delta_pct=0.0,
                robust=False,
                status="DISCARD",
                description=experiment.description,
            ))
            return

        # Build default portfolio config
        base_pf = PortfolioBacktestConfig(
            data_dir=self.data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=self.initial_equity,
            tier=experiment.tier,
        )

        # Use cached portfolio baseline or compute once per tier
        if experiment.tier in self._portfolio_baselines:
            base_score = self._portfolio_baselines[experiment.tier]
        else:
            engine = StockPortfolioEngine(base_pf)
            base_result = engine.run(alcb_trades, iaric_trades)
            base_metrics = extract_metrics(
                base_result.trades, base_result.equity_curve,
                base_result.timestamps, self.initial_equity,
            )
            base_r = compute_r_multiples(base_result.trades)
            base_score = composite_score(base_metrics, self.initial_equity, r_multiples=base_r)
            self._portfolio_baselines[experiment.tier] = base_score

        # Mutated portfolio
        mutated_pf = mutate_portfolio_config(base_pf, experiment.mutations)
        engine = StockPortfolioEngine(mutated_pf)
        mut_result = engine.run(alcb_trades, iaric_trades)
        mut_metrics = extract_metrics(
            mut_result.trades, mut_result.equity_curve,
            mut_result.timestamps, self.initial_equity,
        )
        mut_r = compute_r_multiples(mut_result.trades)
        mut_score = composite_score(mut_metrics, self.initial_equity, r_multiples=mut_r)

        if base_score.total == 0:
            delta_pct = 0.0
        else:
            delta_pct = (mut_score.total - base_score.total) / base_score.total

        status = self.tracker.decide(base_score, mut_score, None)
        print(f"  → {status} (delta={delta_pct:+.2%})")

        self.tracker.record(ExperimentResult(
            experiment_id=experiment.id,
            strategy="portfolio",
            tier=experiment.tier,
            type="PORTFOLIO",
            baseline_score=base_score.total,
            experiment_score=mut_score.total,
            delta_pct=delta_pct,
            robust=False,
            status=status,
            description=experiment.description,
        ))

    def _portfolio_integration(self) -> None:
        """Merge all APPROVED changes and test via StockPortfolioEngine."""
        from backtests.stock.engine.portfolio_engine import StockPortfolioEngine

        results = self.tracker.load_all()
        approved = [r for r in results if r.status == "APPROVE"]

        if not approved:
            print("\nNo APPROVED experiments -- skipping portfolio integration.")
            return

        print(f"\nPortfolio integration with {len(approved)} APPROVED changes...")
        for r in approved:
            print(f"  APPROVED: {r.experiment_id} (delta={r.delta_pct:+.2%})")

        # Rebuild experiment queue to get mutations for approved IDs
        experiments = build_experiment_queue()
        exp_by_id = {e.id: e for e in experiments}

        # Merge mutations per strategy
        alcb_mutations: dict = {}
        iaric_mutations: dict = {}
        for r in approved:
            exp = exp_by_id.get(r.experiment_id)
            if not exp:
                continue
            if exp.strategy == "alcb":
                alcb_mutations.update(exp.mutations)
            elif exp.strategy == "iaric":
                iaric_mutations.update(exp.mutations)

        if not alcb_mutations and not iaric_mutations:
            print("  No strategy-level mutations to integrate.")
            return

        # Run individual strategies with merged APPROVED mutations
        merged_trades: dict[str, list[TradeRecord]] = {}
        for strategy, mutations in [("alcb", alcb_mutations), ("iaric", iaric_mutations)]:
            if not mutations:
                # Use baseline trades if no mutations for this strategy
                for tier in [1, 2]:
                    key = (strategy, tier)
                    if key in self._baseline_trades:
                        merged_trades[strategy] = self._baseline_trades[key]
                        break
                continue

            # Find best tier from baselines
            tier = 2 if (strategy, 2) in self._baselines else 1
            try:
                config = self._make_mutated_config(strategy, tier, mutations)
                trades, eq, ts = self._run_engine(strategy, tier, config)
                merged_trades[strategy] = trades
                metrics = extract_metrics(trades, eq, ts, self.initial_equity)
                r_m = compute_r_multiples(trades)
                score = composite_score(metrics, self.initial_equity, r_multiples=r_m)
                print(f"  Merged {strategy.upper()}: score={score.total:.4f}, "
                      f"trades={metrics.total_trades}")
            except Exception:
                logger.error("Portfolio integration %s run failed:\n%s",
                             strategy, traceback.format_exc())
                print(f"  CRASH: merged {strategy.upper()} run failed")
                # Fall back to baseline trades
                for tier in [1, 2]:
                    key = (strategy, tier)
                    if key in self._baseline_trades:
                        merged_trades[strategy] = self._baseline_trades[key]
                        break

        alcb_trades = merged_trades.get("alcb", [])
        iaric_trades = merged_trades.get("iaric", [])

        if not alcb_trades and not iaric_trades:
            print("  No trades available for portfolio integration.")
            return

        # Run portfolio engine with merged trades
        pf_config = PortfolioBacktestConfig(
            data_dir=self.data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=self.initial_equity,
        )

        try:
            engine = StockPortfolioEngine(pf_config)
            pf_result = engine.run(alcb_trades, iaric_trades)
            pf_metrics = extract_metrics(
                pf_result.trades, pf_result.equity_curve,
                pf_result.timestamps, self.initial_equity,
            )
            pf_r = compute_r_multiples(pf_result.trades)
            pf_score = composite_score(pf_metrics, self.initial_equity, r_multiples=pf_r)

            # Compare against baseline portfolio
            base_alcb = next(
                (self._baseline_trades[k] for k in self._baseline_trades if k[0] == "alcb"),
                [],
            )
            base_iaric = next(
                (self._baseline_trades[k] for k in self._baseline_trades if k[0] == "iaric"),
                [],
            )
            base_engine = StockPortfolioEngine(pf_config)
            base_pf_result = base_engine.run(base_alcb, base_iaric)
            base_pf_metrics = extract_metrics(
                base_pf_result.trades, base_pf_result.equity_curve,
                base_pf_result.timestamps, self.initial_equity,
            )
            base_pf_r = compute_r_multiples(base_pf_result.trades)
            base_pf_score = composite_score(base_pf_metrics, self.initial_equity, r_multiples=base_pf_r)

            if base_pf_score.total == 0:
                delta_pct = 0.0
            else:
                delta_pct = (pf_score.total - base_pf_score.total) / base_pf_score.total

            print(f"  Portfolio integration: score={pf_score.total:.4f}, "
                  f"delta={delta_pct:+.2%} vs baseline portfolio")

            self.tracker.record(ExperimentResult(
                experiment_id="portfolio_integration",
                strategy="portfolio",
                tier=0,
                type="PORTFOLIO",
                baseline_score=base_pf_score.total,
                experiment_score=pf_score.total,
                delta_pct=delta_pct,
                robust=False,
                status="APPROVE" if delta_pct >= 0 else "DISCARD",
                description=f"Merged {len(approved)} APPROVED changes",
            ))
        except Exception:
            logger.error("Portfolio integration failed:\n%s", traceback.format_exc())
            print("  CRASH: portfolio integration failed")

    def _run_engine(
        self,
        strategy: str,
        tier: int,
        config,
    ) -> tuple[list[TradeRecord], np.ndarray, np.ndarray]:
        """Run the correct engine and return (trades, equity_curve, timestamps)."""
        if strategy == "alcb":
            from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
            engine = ALCBIntradayEngine(config, self.replay)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result = engine.run()
        return result.trades, result.equity_curve, result.timestamps

    def _make_default_config(self, strategy: str, tier: int):
        """Build a default config for the given strategy and tier.

        If optimal_config.json exists for the strategy, loads it as the
        baseline (ablation flags + param_overrides).  Otherwise falls back
        to hardcoded $10K-tuned defaults.
        """
        if strategy == "alcb":
            base = ALCBBacktestConfig(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_equity=self.initial_equity,
                tier=tier,
                data_dir=self.data_dir,
                verbose=self.verbose,
                param_overrides={
                    "base_risk_fraction": 0.015,
                    "min_adv_usd": 5_000_000.0,
                    "heat_cap_r": 10.0,
                    "min_containment": 0.70,
                    "max_squeeze_metric": 1.30,
                    "breakout_tolerance_pct": 0.10,
                },
            )
            optimal_path = self.output_dir / "optimal_config.json"
            if optimal_path.exists():
                import json
                with open(optimal_path) as f:
                    optimal = json.load(f)
                mutations: dict = {}
                for k, v in optimal.get("ablation", {}).items():
                    mutations[f"ablation.{k}"] = v
                for k, v in optimal.get("param_overrides", {}).items():
                    mutations[f"param_overrides.{k}"] = v
                base = mutate_alcb_config(base, mutations)
            return base
        elif strategy == "iaric":
            return IARICBacktestConfig(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_equity=self.initial_equity,
                tier=tier,
                data_dir=self.data_dir,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _make_mutated_config(self, strategy: str, tier: int, mutations: dict):
        """Build a mutated config from default + mutations."""
        base = self._make_default_config(strategy, tier)
        if strategy == "alcb":
            return mutate_alcb_config(base, mutations)
        elif strategy == "iaric":
            return mutate_iaric_config(base, mutations)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_report(self) -> None:
        """Generate markdown report from results."""
        results = self.tracker.load_all()
        if not results:
            print("No results to report.")
            return

        report_path = self.output_dir / "report.md"
        report = generate_report(results, self._baselines)
        report_path.write_text(report)
        print(f"\nReport written to {report_path}")
