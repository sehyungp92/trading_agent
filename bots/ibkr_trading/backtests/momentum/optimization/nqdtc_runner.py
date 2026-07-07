"""NQDTC optimization runner: LHS coarse search + Optuna TPE refinement.

Mirrors ApexOptimizationRunner but uses NQDTC types, 5m data, and parameter space.
Reuses TrialResult, OptimizationResult, composite_objective.
"""
from __future__ import annotations

import logging
import multiprocessing

import numpy as np

from backtests.momentum.analysis.metrics import compute_metrics
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.optimization.nqdtc_param_space import (
    NQDTC_PARAM_SPACE,
    nqdtc_params_to_overrides,
)
from backtests.momentum.optimization.objective import composite_objective
from backtests.momentum.optimization.param_space import latin_hypercube_sample
from backtests.momentum.optimization.runner import OptimizationResult, TrialResult

logger = logging.getLogger(__name__)


def _evaluate_single(
    params: dict[str, float],
    nqdtc_data: dict,
    base_config: NQDTCBacktestConfig,
) -> TrialResult:
    """Evaluate one NQDTC parameter set."""
    overrides = nqdtc_params_to_overrides(params)
    config = NQDTCBacktestConfig(
        symbols=base_config.symbols,
        start_date=base_config.start_date,
        end_date=base_config.end_date,
        initial_equity=base_config.initial_equity,
        slippage=base_config.slippage,
        flags=base_config.flags,
        param_overrides=overrides,
        data_dir=base_config.data_dir,
        track_signals=False,
        track_shadows=False,
        warmup_daily=base_config.warmup_daily,
        warmup_30m=base_config.warmup_30m,
        warmup_1h=base_config.warmup_1h,
        warmup_4h=base_config.warmup_4h,
        warmup_5m=base_config.warmup_5m,
    )

    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

    engine = NQDTCEngine(config.symbols[0], config)
    result = engine.run(
        nqdtc_data["five_min_bars"],
        nqdtc_data["thirty_min"],
        nqdtc_data["hourly"],
        nqdtc_data["four_hour"],
        nqdtc_data["daily"],
        nqdtc_data["thirty_min_idx_map"],
        nqdtc_data["hourly_idx_map"],
        nqdtc_data["four_hour_idx_map"],
        nqdtc_data["daily_idx_map"],
        daily_es=nqdtc_data.get("daily_es"),
        daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
    )

    if not result.trades:
        return TrialResult(params=params, score=-1.0)

    pnls = np.array([t.pnl_dollars for t in result.trades])
    risks = np.array([
        abs(t.entry_price - t.initial_stop) * base_config.point_value
        for t in result.trades
    ])
    holds = np.array([t.bars_held_30m for t in result.trades])
    comms = np.array([t.commission for t in result.trades])

    metrics = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=holds,
        trade_commissions=comms,
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
        initial_equity=config.initial_equity,
    )

    score = composite_objective(metrics, min_trades_per_month=1.0, min_total_trades=50)

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


class NQDTCOptimizationRunner:
    """Two-stage optimizer for NQDTC v2.0 strategy."""

    def __init__(
        self,
        base_config: NQDTCBacktestConfig,
        nqdtc_data: dict,
        n_coarse: int = 500,
        n_refine: int = 200,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.base_config = base_config
        self.nqdtc_data = nqdtc_data
        self.n_coarse = n_coarse
        self.n_refine = n_refine
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count() - 1)
        self.seed = seed

    def run(self) -> OptimizationResult:
        """Execute the full two-stage optimization."""
        logger.info(
            "Starting NQDTC optimization: %d coarse + %d refine, %d workers",
            self.n_coarse, self.n_refine, self.n_jobs,
        )

        coarse_results = self._stage_coarse()
        logger.info(
            "Coarse search complete: %d/%d valid",
            sum(1 for r in coarse_results if r.score > 0), len(coarse_results),
        )

        coarse_sorted = sorted(coarse_results, key=lambda r: r.score, reverse=True)
        top_50 = [r for r in coarse_sorted[:50] if r.score > 0]

        if not top_50:
            logger.warning("No valid coarse results, skipping refinement")
            return OptimizationResult(coarse_results=coarse_results)

        refine_results = self._stage_refine(top_50)
        logger.info("Refinement complete: %d trials", len(refine_results))

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
        """Stage A: Latin Hypercube Sampling coarse search."""
        samples = latin_hypercube_sample(NQDTC_PARAM_SPACE, self.n_coarse, self.seed)

        results = []
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                logger.info("Coarse trial %d/%d", i + 1, self.n_coarse)
            try:
                result = _evaluate_single(sample, self.nqdtc_data, self.base_config)
                results.append(result)
            except Exception:
                logger.exception("Error in coarse trial %d", i)
                results.append(TrialResult(params=sample, score=-1.0))
        return results

    def _stage_refine(self, top_results: list[TrialResult]) -> list[TrialResult]:
        """Stage B: Optuna TPE refinement around top coarse results."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed, skipping refinement stage")
            return []

        study = optuna.create_study(direction="maximize")

        for tr in top_results:
            try:
                study.enqueue_trial(tr.params)
            except Exception:
                pass

        trial_details: dict[int, TrialResult] = {}

        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}
            for p in NQDTC_PARAM_SPACE:
                if p.is_int:
                    params[p.name] = trial.suggest_int(
                        p.name, int(p.low), int(p.high),
                        step=int(p.step) if p.step else 1,
                    )
                elif p.step > 0:
                    params[p.name] = trial.suggest_float(
                        p.name, p.low, p.high, step=p.step,
                    )
                else:
                    params[p.name] = trial.suggest_float(p.name, p.low, p.high)

            try:
                result = _evaluate_single(params, self.nqdtc_data, self.base_config)
                trial_details[trial.number] = result
                return result.score
            except Exception:
                return -1.0

        study.optimize(optuna_objective, n_trials=self.n_refine)

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
