"""Optimization runner: LHS coarse search + Optuna TPE refinement."""
from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from backtests.swing.analysis.metrics import compute_metrics
from backtests.swing.config import BacktestConfig
from backtests.swing.engine.portfolio_engine import PortfolioData, run_independent
from backtests.swing.optimization.objective import composite_objective
from backtests.swing.optimization.param_space import (
    PARAM_SPACE,
    ParamRange,
    latin_hypercube_sample,
    params_to_overrides,
)
from strategies.swing.atrss.config import SYMBOL_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a single optimization trial."""

    params: dict[str, float]
    score: float
    total_trades: int = 0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    profit_factor: float = 0.0
    trades_per_month: float = 0.0


@dataclass
class OptimizationResult:
    """Result of the full optimization run."""

    best_params: dict[str, float] = field(default_factory=dict)
    best_score: float = -1.0
    coarse_results: list[TrialResult] = field(default_factory=list)
    refine_results: list[TrialResult] = field(default_factory=list)
    all_sorted: list[TrialResult] = field(default_factory=list)


def _evaluate_single(
    params: dict[str, float],
    data: PortfolioData,
    base_config: BacktestConfig,
) -> TrialResult:
    """Evaluate one parameter set. Called in worker processes."""
    from dataclasses import replace

    overrides = params_to_overrides(params)
    config = BacktestConfig(
        symbols=base_config.symbols,
        start_date=base_config.start_date,
        end_date=base_config.end_date,
        initial_equity=base_config.initial_equity,
        slippage=base_config.slippage,
        flags=base_config.flags,
        param_overrides=overrides,
        data_dir=base_config.data_dir,
        track_shadows=False,
        warmup_daily=base_config.warmup_daily,
        warmup_hourly=base_config.warmup_hourly,
    )

    result = run_independent(data, config)

    # Aggregate trades across symbols
    all_pnls = []
    all_risks = []
    all_holds = []
    all_comms = []
    all_syms = []
    for sr in result.symbol_results.values():
        for t in sr.trades:
            all_pnls.append(t.pnl_dollars)
            sym_mult = SYMBOL_CONFIGS[t.symbol].multiplier if t.symbol in SYMBOL_CONFIGS else 1.0
            all_risks.append(abs(t.entry_price - t.initial_stop) * sym_mult * t.qty)
            all_holds.append(t.bars_held)
            all_comms.append(t.commission)
            all_syms.append(t.symbol)

    if not all_pnls:
        return TrialResult(params=params, score=-1.0)

    metrics = compute_metrics(
        trade_pnls=np.array(all_pnls),
        trade_risks=np.array(all_risks),
        trade_hold_hours=np.array(all_holds),
        trade_commissions=np.array(all_comms),
        equity_curve=result.combined_equity,
        timestamps=result.combined_timestamps,
        initial_equity=config.initial_equity,
        trade_symbols=all_syms,
    )

    score = composite_objective(metrics)

    return TrialResult(
        params=params,
        score=score,
        total_trades=metrics.total_trades,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe,
        max_dd=metrics.max_drawdown_pct,
        profit_factor=metrics.profit_factor,
        trades_per_month=metrics.trades_per_month,
    )


class OptimizationRunner:
    """Two-stage optimizer: LHS coarse → Optuna TPE refinement."""

    def __init__(
        self,
        base_config: BacktestConfig,
        data: PortfolioData,
        n_coarse: int = 1000,
        n_refine: int = 300,
        n_jobs: int = -1,
        seed: int = 42,
    ):
        self.base_config = base_config
        self.data = data
        self.n_coarse = n_coarse
        self.n_refine = n_refine
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count() - 1)
        self.seed = seed

    def run(self) -> OptimizationResult:
        """Execute the full two-stage optimization."""
        logger.info(
            "Starting optimization: %d coarse + %d refine, %d workers",
            self.n_coarse, self.n_refine, self.n_jobs,
        )

        # Stage A: LHS coarse search
        coarse_results = self._stage_coarse()
        logger.info(
            "Coarse search complete: %d/%d valid",
            sum(1 for r in coarse_results if r.score > 0), len(coarse_results),
        )

        # Sort by score, take top 50
        coarse_sorted = sorted(coarse_results, key=lambda r: r.score, reverse=True)
        top_50 = [r for r in coarse_sorted[:50] if r.score > 0]

        if not top_50:
            logger.warning("No valid coarse results, skipping refinement")
            return OptimizationResult(coarse_results=coarse_results)

        # Stage B: Optuna TPE refinement
        refine_results = self._stage_refine(top_50)
        logger.info("Refinement complete: %d trials", len(refine_results))

        # Combine and sort all results
        all_results = coarse_results + refine_results
        all_sorted = sorted(all_results, key=lambda r: r.score, reverse=True)

        best = all_sorted[0] if all_sorted else TrialResult(params={}, score=-1.0)

        return OptimizationResult(
            best_params=best.params,
            best_score=best.score,
            coarse_results=coarse_results,
            refine_results=refine_results,
            all_sorted=all_sorted[:100],
        )

    def _stage_coarse(self) -> list[TrialResult]:
        """Stage A: Latin Hypercube Sampling coarse search.

        Uses ProcessPoolExecutor for parallel evaluation when n_jobs > 1.
        """
        samples = latin_hypercube_sample(PARAM_SPACE, self.n_coarse, self.seed)

        if self.n_jobs <= 1:
            # Sequential fallback
            results = []
            for i, sample in enumerate(samples):
                if (i + 1) % 100 == 0:
                    logger.info("Coarse trial %d/%d", i + 1, self.n_coarse)
                try:
                    result = _evaluate_single(sample, self.data, self.base_config)
                    results.append(result)
                except Exception:
                    logger.exception("Error in coarse trial %d", i)
                    results.append(TrialResult(params=sample, score=-1.0))
            return results

        # Parallel execution
        results: list[TrialResult] = [TrialResult(params={}, score=-1.0)] * len(samples)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_idx = {
                executor.submit(_evaluate_single, sample, self.data, self.base_config): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                if completed % 100 == 0:
                    logger.info("Coarse trial %d/%d completed", completed, self.n_coarse)
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception("Error in coarse trial %d", idx)
                    results[idx] = TrialResult(params=samples[idx], score=-1.0)

        return results

    def _stage_refine(self, top_results: list[TrialResult]) -> list[TrialResult]:
        """Stage B: Optuna TPE refinement around top coarse results.

        Captures full TrialResult metrics (not just score) via trial user attrs.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, skipping refinement stage")
            return []

        study = optuna.create_study(direction="maximize")

        # Seed study with top coarse results
        for tr in top_results:
            try:
                study.enqueue_trial(tr.params)
            except Exception:
                pass

        # Side dict to capture full metrics (trial.number -> TrialResult)
        trial_details: dict[int, TrialResult] = {}

        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}
            for p in PARAM_SPACE:
                if p.is_int:
                    params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), step=int(p.step) if p.step else 1)
                elif p.step > 0:
                    params[p.name] = trial.suggest_float(p.name, p.low, p.high, step=p.step)
                else:
                    params[p.name] = trial.suggest_float(p.name, p.low, p.high)

            try:
                result = _evaluate_single(params, self.data, self.base_config)
                trial_details[trial.number] = result
                return result.score
            except Exception:
                return -1.0

        study.optimize(optuna_objective, n_trials=self.n_refine)

        # Convert study results with full metrics
        refine_results = []
        for trial in study.trials:
            if trial.value is not None:
                detail = trial_details.get(trial.number)
                if detail is not None:
                    refine_results.append(detail)
                else:
                    refine_results.append(TrialResult(
                        params=trial.params,
                        score=trial.value,
                    ))

        return refine_results
