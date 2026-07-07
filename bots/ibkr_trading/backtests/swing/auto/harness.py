"""Core orchestrator for auto swing backtesting.

Loads data once, runs all experiments in priority order,
computes composite scores, checks robustness, and writes results.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from backtests.swing.auto.config_mutator import (
    mutate_atrss_config,
    mutate_helix_config,
    mutate_unified_config,
)
from backtests.swing.auto.experiments import Experiment, build_experiment_queue
from backtests.swing.auto.report import generate_report
from backtests.swing.auto.results_tracker import ExperimentResult, ResultsTracker
from backtests.swing.auto.robustness import run_robustness
from backtests.swing.auto.scoring import CompositeScore, composite_score, extract_metrics

logger = logging.getLogger(__name__)


class SwingAutoHarness:
    """Orchestrates automated experiment runs for active swing strategies."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        initial_equity: float = 100_000.0,
        verbose: bool = False,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.initial_equity = initial_equity
        self.verbose = verbose

        self.tracker = ResultsTracker(output_dir)

        # Cached data (loaded once)
        self._atrss_data = None   # PortfolioData
        self._helix_data = None   # HelixPortfolioData
        self._unified_data = None  # UnifiedPortfolioData

        # Baseline scores keyed by strategy name
        self._baselines: dict[str, CompositeScore] = {}
        self._baseline_trades: dict[str, list] = {}
        self._portfolio_baseline: CompositeScore | None = None

        # Caches for speed
        self._default_configs: dict[str, object] = {}
        self._experiment_queue_cache: dict[str, list[Experiment]] = {}

    def run_all(
        self,
        strategy_filter: str = "all",
        experiment_ids: list[str] | None = None,
        skip_robustness: bool = False,
        resume: bool = False,
    ) -> None:
        """Run the full experiment pipeline.

        Args:
            strategy_filter: "atrss", "helix", "portfolio", or "all"
            experiment_ids: Specific experiment IDs to run (None = all)
            skip_robustness: Skip robustness checks for faster ablation scan
            resume: Skip experiments already in results.tsv
        """
        # 1. Load data ONCE
        print("Loading bar data (this may take a moment)...")
        t0 = time.time()
        self._load_data(strategy_filter)
        print(f"Data loaded in {time.time() - t0:.1f}s")

        # 2. Build experiment queue (cached for reuse in _portfolio_integration)
        if strategy_filter not in self._experiment_queue_cache:
            self._experiment_queue_cache[strategy_filter] = build_experiment_queue(strategy_filter)
        experiments = list(self._experiment_queue_cache[strategy_filter])
        if experiment_ids:
            id_set = set(experiment_ids)
            experiments = [e for e in experiments if e.id in id_set]

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

        # 3. Run baselines for needed strategies
        needed_strategies = {e.strategy for e in pending if e.strategy != "portfolio"}
        failed_baselines: set[str] = set()
        for strategy in sorted(needed_strategies):
            if strategy not in self._baselines:
                if not self._run_baseline(strategy):
                    failed_baselines.add(strategy)

        # Filter out experiments whose baselines failed
        if failed_baselines:
            pending = [
                e for e in pending
                if e.strategy == "portfolio"
                or e.strategy not in failed_baselines
            ]

        if not pending:
            print("All experiments skipped (baselines failed or already completed).")
            self._generate_report()
            return

        # 4. Run experiments in priority order
        total = len(pending)
        elapsed_total = 0.0
        for i, experiment in enumerate(pending, 1):
            if i > 1 and elapsed_total > 0:
                avg_per = elapsed_total / (i - 1)
                remaining = avg_per * (total - i + 1)
                eta_min = remaining / 60
                print(f"\n[{i}/{total}] {experiment.id}: {experiment.description}  "
                      f"(ETA: {eta_min:.0f}m)")
            else:
                print(f"\n[{i}/{total}] {experiment.id}: {experiment.description}")
            t_start = time.time()

            try:
                self._run_experiment(experiment, skip_robustness)
            except Exception:
                logger.error("Experiment %s crashed:\n%s",
                             experiment.id, traceback.format_exc())
                baseline_score = self._baselines.get(
                    experiment.strategy, CompositeScore(0, 0, 0, 0, 0),
                )
                self.tracker.record(ExperimentResult(
                    experiment_id=experiment.id,
                    strategy=experiment.strategy,
                    type=experiment.type,
                    baseline_score=baseline_score.total,
                    experiment_score=0.0,
                    delta_pct=0.0,
                    robust=False,
                    status="CRASH",
                    description=experiment.description,
                ))

            elapsed = time.time() - t_start
            elapsed_total += elapsed
            print(f"  Completed in {elapsed:.1f}s")

        # 5. Portfolio integration (if running "all" and we have approved changes)
        if strategy_filter == "all" and not experiment_ids:
            self._portfolio_integration()

        # 6. Generate report
        self._generate_report()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_data(self, strategy_filter: str) -> None:
        """Load bar data for needed strategies."""
        from backtests.swing.cli import _load_data, _load_helix_data
        from backtests.swing.data.preprocessing import build_numpy_arrays, normalize_timezone
        from backtests.swing.data.cache import load_bars

        _all = strategy_filter == "all"

        def _timed_load(label, fn):
            t = time.time()
            result = fn()
            print(f"  {label}: {time.time() - t:.1f}s")
            return result

        # ATRSS data (daily + hourly)
        if _all or strategy_filter == "atrss":
            atrss_symbols = self._get_atrss_symbols()
            self._atrss_data = _timed_load(
                f"ATRSS ({','.join(atrss_symbols)})",
                lambda: _load_data(atrss_symbols, self.data_dir),
            )

        # Helix data (daily + hourly + 4H)
        if _all or strategy_filter == "helix":
            helix_symbols = self._get_helix_symbols()
            self._helix_data = _timed_load(
                f"Helix ({','.join(helix_symbols)})",
                lambda: _load_helix_data(helix_symbols, self.data_dir),
            )

        # Unified data (for portfolio experiments)
        if _all or strategy_filter == "portfolio":
            # Portfolio needs all per-strategy data loaded first
            if not _all:
                if self._atrss_data is None:
                    self._atrss_data = _load_data(self._get_atrss_symbols(), self.data_dir)
                if self._helix_data is None:
                    self._helix_data = _load_helix_data(self._get_helix_symbols(), self.data_dir)
            self._load_unified_data()

    def _load_unified_data(self) -> None:
        """Load UnifiedPortfolioData for portfolio-level experiments."""
        from backtests.swing.config_unified import UnifiedBacktestConfig
        from backtests.swing.engine.unified_portfolio_engine import (
            UnifiedPortfolioData,
            load_unified_data,
        )
        try:
            config = UnifiedBacktestConfig(
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
            )
            self._unified_data = load_unified_data(config)
        except Exception:
            logger.warning("load_unified_data failed, assembling from per-strategy data")
            self._unified_data = UnifiedPortfolioData()
            if self._atrss_data:
                self._unified_data.atrss_hourly = getattr(self._atrss_data, 'hourly', {})
                self._unified_data.atrss_daily_idx_maps = getattr(self._atrss_data, 'daily_idx_maps', {})
            if self._helix_data:
                self._unified_data.hourly = getattr(self._helix_data, 'hourly', {})
                self._unified_data.four_hour = getattr(self._helix_data, 'four_hour', {})
                self._unified_data.daily_idx_maps = getattr(self._helix_data, 'daily_idx_maps', {})
                self._unified_data.four_hour_idx_maps = getattr(self._helix_data, 'four_hour_idx_maps', {})
            self._unified_data.daily = {
                **getattr(self._atrss_data, 'daily', {}),
                **getattr(self._helix_data, 'daily', {}),
            }

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _get_atrss_symbols() -> list[str]:
        try:
            from strategies.swing.atrss.config import SYMBOLS
            return list(SYMBOLS)
        except ImportError:
            return ["QQQ", "GLD"]

    @staticmethod
    def _get_helix_symbols() -> list[str]:
        try:
            from strategies.swing.akc_helix.config import SYMBOLS
            return list(SYMBOLS)
        except ImportError:
            return ["QQQ", "GLD"]

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _run_baseline(self, strategy: str) -> bool:
        """Run default config to establish baseline score.

        Returns True if baseline is valid, False otherwise.
        """
        print(f"  Running baseline: {strategy}...")
        try:
            config = self._get_default_config(strategy)
            trades, eq, ts = self._run_engine(strategy, config)
        except Exception:
            logger.error("Baseline %s crashed:\n%s",
                         strategy, traceback.format_exc())
            print(f"  BASELINE CRASH: {strategy} — skipping all experiments")
            return False

        metrics = extract_metrics(trades, eq, ts, self.initial_equity)
        score = composite_score(metrics, self.initial_equity, strategy=strategy)

        self._baselines[strategy] = score
        self._baseline_trades[strategy] = trades

        status = "OK" if not score.rejected else f"REJECTED ({score.reject_reason})"
        print(f"  Baseline {strategy}: score={score.total:.4f} "
              f"trades={metrics.total_trades} PF={metrics.profit_factor:.2f} "
              f"DD={metrics.max_drawdown_pct:.1%} [{status}]")

        return not score.rejected

    # ------------------------------------------------------------------
    # Experiment execution
    # ------------------------------------------------------------------

    def _run_experiment(self, experiment: Experiment, skip_robustness: bool) -> None:
        """Run a single experiment, score it, and record results."""
        if experiment.strategy == "portfolio":
            self._run_portfolio_experiment(experiment, skip_robustness)
            return

        baseline = self._baselines.get(experiment.strategy)
        if baseline is None:
            logger.warning("No baseline for %s — skipping %s",
                           experiment.strategy, experiment.id)
            return

        config = self._make_mutated_config(experiment.strategy, experiment.mutations)
        trades, eq, ts = self._run_engine(experiment.strategy, config)

        metrics = extract_metrics(trades, eq, ts, self.initial_equity)
        score = composite_score(metrics, self.initial_equity, strategy=experiment.strategy)

        delta_pct = (
            (score.total - baseline.total) / baseline.total
            if baseline.total > 0 else 0.0
        )
        is_ablation = experiment.type == "ABLATION"

        # Robustness checks (only when delta is promising and not skipped)
        robustness = None
        if (not skip_robustness
                and not score.rejected
                and delta_pct >= 0.02
                and not (is_ablation and abs(delta_pct) < 1e-10)):
            print(f"  Running robustness checks...")

            def run_fn(mutations):
                cfg = self._make_mutated_config(experiment.strategy, mutations)
                t, e, ts_ = self._run_engine(experiment.strategy, cfg)
                return t, e, ts_, self.initial_equity

            robustness = run_robustness(
                experiment, trades, eq, ts,
                self.initial_equity, score.total, run_fn,
            )

        status = self.tracker.decide(baseline, score, robustness, is_ablation)
        robust = robustness.passes_all if robustness else False

        detail = score.reject_reason if score.rejected else f"score={score.total:.4f}"
        print(f"  → {status} (delta={delta_pct:+.2%}, {detail})")

        self.tracker.record(ExperimentResult(
            experiment_id=experiment.id,
            strategy=experiment.strategy,
            type=experiment.type,
            baseline_score=baseline.total,
            experiment_score=score.total,
            delta_pct=delta_pct,
            robust=robust,
            status=status,
            description=experiment.description,
        ))

    def _run_portfolio_experiment(self, experiment: Experiment, skip_robustness: bool) -> None:
        """Run a portfolio-level experiment using run_unified."""
        from backtests.swing.config_unified import UnifiedBacktestConfig
        from backtests.swing.engine.unified_portfolio_engine import run_unified

        if self._unified_data is None:
            logger.warning("No unified data — skipping %s", experiment.id)
            return

        # Compute portfolio baseline if not cached
        if self._portfolio_baseline is None:
            base_config = UnifiedBacktestConfig(initial_equity=self.initial_equity)
            base_result = run_unified(self._unified_data, base_config)
            all_trades = self._collect_unified_trades(base_result)
            base_eq = base_result.combined_equity
            base_ts = base_result.combined_timestamps
            base_metrics = extract_metrics(all_trades, base_eq, base_ts, self.initial_equity)
            self._portfolio_baseline = composite_score(base_metrics, self.initial_equity, equity_curve=base_eq)
            print(f"  Portfolio baseline: score={self._portfolio_baseline.total:.4f}")

        baseline = self._portfolio_baseline

        # Run mutated portfolio
        base_config = UnifiedBacktestConfig(initial_equity=self.initial_equity)
        config = mutate_unified_config(base_config, experiment.mutations)
        result = run_unified(self._unified_data, config)
        all_trades = self._collect_unified_trades(result)
        eq = result.combined_equity
        ts = result.combined_timestamps

        metrics = extract_metrics(all_trades, eq, ts, self.initial_equity)
        score = composite_score(metrics, self.initial_equity, equity_curve=eq)

        if baseline.total == 0:
            delta_pct = 0.0
        else:
            delta_pct = (score.total - baseline.total) / baseline.total

        if score.rejected:
            status = "DISCARD"
            robust = False
        else:
            # Run robustness checks for promising portfolio experiments
            robustness = None
            if not skip_robustness and delta_pct >= 0.02:
                print(f"  Running robustness checks...")

                def run_fn(mutations):
                    cfg = mutate_unified_config(base_config, mutations)
                    r = run_unified(self._unified_data, cfg)
                    t_all = self._collect_unified_trades(r)
                    return t_all, r.combined_equity, r.combined_timestamps, self.initial_equity

                robustness = run_robustness(
                    experiment, all_trades, eq, ts,
                    self.initial_equity, score.total, run_fn,
                )

            status = self.tracker.decide(baseline, score, robustness, False)
            robust = robustness.passes_all if robustness else False

        print(f"  → {status} (delta={delta_pct:+.2%}, score={score.total:.4f})")

        self.tracker.record(ExperimentResult(
            experiment_id=experiment.id,
            strategy="portfolio",
            type=experiment.type,
            baseline_score=baseline.total,
            experiment_score=score.total,
            delta_pct=delta_pct,
            robust=robust,
            status=status,
            description=experiment.description,
        ))

    # ------------------------------------------------------------------
    # Engine dispatch
    # ------------------------------------------------------------------

    def _run_engine(
        self,
        strategy: str,
        config,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """Run the correct engine and return (trades, equity_curve, timestamps)."""
        if strategy == "atrss":
            return self._run_atrss(config)
        elif strategy == "helix":
            return self._run_helix(config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _run_atrss(self, config) -> tuple[list, np.ndarray, np.ndarray]:
        """Run ATRSS per symbol, merge results."""
        from backtests.swing.engine.backtest_engine import BacktestEngine
        try:
            from strategies.swing.atrss.config import SYMBOL_CONFIGS
        except ImportError:
            from backtests.swing.config import _default_atrss_symbols
            SYMBOL_CONFIGS = {}

        all_trades = []
        equity_curves = []
        timestamps_all = []

        for sym in config.symbols:
            if self._atrss_data is None:
                continue
            hourly = getattr(self._atrss_data, 'hourly', {}).get(sym)
            daily = getattr(self._atrss_data, 'daily', {}).get(sym)
            idx_map = getattr(self._atrss_data, 'daily_idx_maps', {}).get(sym)
            if hourly is None or daily is None or idx_map is None:
                continue

            cfg = SYMBOL_CONFIGS.get(sym)
            if cfg is None:
                continue

            engine = BacktestEngine(
                symbol=sym, cfg=cfg, bt_config=config,
                point_value=cfg.multiplier,
            )
            result = engine.run(daily, hourly, idx_map)
            all_trades.extend(result.trades)
            if len(result.equity_curve) > 0:
                equity_curves.append(result.equity_curve)
                timestamps_all.append(result.timestamps)

        eq, ts = self._merge_equity(equity_curves, timestamps_all)
        return all_trades, eq, ts

    def _run_helix(self, config) -> tuple[list, np.ndarray, np.ndarray]:
        """Run Helix per symbol, merge results."""
        from backtests.swing.engine.helix_engine import HelixEngine
        try:
            from strategies.swing.akc_helix.config import SYMBOL_CONFIGS
        except ImportError:
            SYMBOL_CONFIGS = {}

        all_trades = []
        equity_curves = []
        timestamps_all = []

        for sym in config.symbols:
            if self._helix_data is None:
                continue
            hourly = getattr(self._helix_data, 'hourly', {}).get(sym)
            daily = getattr(self._helix_data, 'daily', {}).get(sym)
            four_hour = getattr(self._helix_data, 'four_hour', {}).get(sym)
            d_idx = getattr(self._helix_data, 'daily_idx_maps', {}).get(sym)
            fh_idx = getattr(self._helix_data, 'four_hour_idx_maps', {}).get(sym)
            if any(x is None for x in [hourly, daily, four_hour, d_idx, fh_idx]):
                continue

            cfg = SYMBOL_CONFIGS.get(sym)
            if cfg is None:
                continue

            engine = HelixEngine(
                symbol=sym, cfg=cfg, bt_config=config,
                point_value=cfg.multiplier,
            )
            result = engine.run(daily, hourly, four_hour, d_idx, fh_idx)
            all_trades.extend(result.trades)
            if len(result.equity_curve) > 0:
                equity_curves.append(result.equity_curve)
                timestamps_all.append(result.timestamps)

        eq, ts = self._merge_equity(equity_curves, timestamps_all)
        return all_trades, eq, ts

    # ------------------------------------------------------------------
    # Config factories
    # ------------------------------------------------------------------

    def _make_default_config(self, strategy: str):
        """Build the default (baseline) config for a strategy."""
        if strategy == "atrss":
            from backtests.swing.config import BacktestConfig
            return BacktestConfig(
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
                symbols=self._get_atrss_symbols(),
            )
        elif strategy == "helix":
            from backtests.swing.config_helix import HelixBacktestConfig
            return HelixBacktestConfig(
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
                symbols=self._get_helix_symbols(),
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_default_config(self, strategy: str):
        """Cached default config — avoids re-importing and re-constructing each experiment."""
        if strategy not in self._default_configs:
            self._default_configs[strategy] = self._make_default_config(strategy)
        return self._default_configs[strategy]

    def _make_mutated_config(self, strategy: str, mutations: dict):
        """Build a mutated config from default + mutations."""
        base = self._get_default_config(strategy)
        if strategy == "atrss":
            return mutate_atrss_config(base, mutations)
        elif strategy == "helix":
            return mutate_helix_config(base, mutations)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    # Portfolio integration
    # ------------------------------------------------------------------

    def _portfolio_integration(self) -> None:
        """Merge APPROVED individual mutations and test the combined effect."""
        results = self.tracker.load_all()
        approved = [r for r in results if r.status == "APPROVE"]

        if not approved:
            print("\nNo APPROVE results — skipping portfolio integration.")
            return

        print(f"\n{'='*60}")
        print(f"PORTFOLIO INTEGRATION: merging {len(approved)} approved changes")
        print(f"{'='*60}")

        # Build lookup (use cache if available)
        if "all" not in self._experiment_queue_cache:
            self._experiment_queue_cache["all"] = build_experiment_queue("all")
        exp_by_id = {e.id: e for e in self._experiment_queue_cache["all"]}

        # Merge mutations per strategy
        merged_mutations: dict[str, dict] = {}
        for r in approved:
            exp = exp_by_id.get(r.experiment_id)
            if not exp or exp.strategy == "portfolio":
                continue
            merged_mutations.setdefault(exp.strategy, {}).update(exp.mutations)

        if not merged_mutations:
            print("  No individual strategy mutations to merge.")
            return

        for strategy, mutations in merged_mutations.items():
            print(f"  {strategy}: {len(mutations)} merged mutations")

        # Run each strategy with merged mutations
        for strategy, mutations in merged_mutations.items():
            try:
                config = self._make_mutated_config(strategy, mutations)
                trades, eq, ts = self._run_engine(strategy, config)
                metrics = extract_metrics(trades, eq, ts, self.initial_equity)
                score = composite_score(metrics, self.initial_equity, strategy=strategy)
                baseline = self._baselines.get(strategy, CompositeScore(0, 0, 0, 0, 0))

                if baseline.total > 0:
                    delta = (score.total - baseline.total) / baseline.total
                else:
                    delta = 0.0

                print(f"  {strategy} merged: score={score.total:.4f} "
                      f"delta={delta:+.2%} (baseline={baseline.total:.4f})")

                self.tracker.record(ExperimentResult(
                    experiment_id=f"integration_{strategy}",
                    strategy=strategy,
                    type="INTEGRATION",
                    baseline_score=baseline.total,
                    experiment_score=score.total,
                    delta_pct=delta,
                    robust=False,
                    status="APPROVE" if delta >= 0.05 else "TEST_FURTHER",
                    description=f"Merged {len(mutations)} approved mutations",
                ))
            except Exception:
                logger.error("Integration %s crashed:\n%s",
                             strategy, traceback.format_exc())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_equity(
        self,
        equity_curves: list[np.ndarray],
        timestamps_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge multiple per-symbol equity curves into a combined curve."""
        if not equity_curves:
            return np.array([]), np.array([])

        if len(equity_curves) == 1:
            return equity_curves[0], timestamps_list[0]

        # Use the longest curve's timestamps as the base
        longest_idx = max(range(len(equity_curves)), key=lambda i: len(equity_curves[i]))
        base_ts = timestamps_list[longest_idx]

        # Sum PnL deltas from each curve, anchored to shared initial equity
        combined = np.zeros(len(base_ts))
        for eq in equity_curves:
            if len(eq) > 0:
                pnl = eq - eq[0]  # PnL relative to start
                if len(pnl) < len(combined):
                    padded = np.full(len(combined), pnl[-1])
                    padded[:len(pnl)] = pnl
                    combined += padded
                else:
                    combined += pnl[:len(combined)]

        combined += self.initial_equity
        return combined, base_ts

    @staticmethod
    def _collect_unified_trades(result) -> list:
        """Collect trades from a UnifiedPortfolioResult."""
        all_trades = []
        # Direct trade lists on the result object
        for attr in ('atrss_trades', 'helix_trades'):
            trades = getattr(result, attr, [])
            if isinstance(trades, list):
                all_trades.extend(trades)
        # Also collect from strategy_results dict if present
        strategy_results = getattr(result, 'strategy_results', {})
        if isinstance(strategy_results, dict) and not all_trades:
            for sr in strategy_results.values():
                all_trades.extend(getattr(sr, 'trades', []))
        return all_trades

    def _generate_report(self) -> None:
        """Generate markdown report and write to output dir."""
        results = self.tracker.load_all()
        # Collect all experiments for config recommendations
        all_experiments = build_experiment_queue("all")
        report_text = generate_report(results, self._baselines, all_experiments)
        report_path = self.output_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport written to: {report_path}")
