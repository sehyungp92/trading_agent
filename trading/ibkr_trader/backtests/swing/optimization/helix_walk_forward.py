"""Helix walk-forward validation with expanding training windows.

Mirrors WalkForwardValidator but uses Helix types.
Reuses WalkForwardFold, RobustnessThresholds, WalkForwardResult.
"""
from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from backtests.swing.analysis.metrics import compute_metrics
from backtests.swing.config_helix import HelixBacktestConfig
from backtests.swing.data.preprocessing import (
    NumpyBars,
    align_4h_to_hourly,
    align_daily_to_hourly,
    resample_1h_to_4h,
)
from backtests.swing.engine.helix_portfolio_engine import (
    HelixPortfolioData,
    run_helix_independent,
)
from backtests.swing.optimization.helix_runner import HelixOptimizationRunner
from backtests.swing.optimization.objective import composite_objective
from backtests.swing.optimization.walk_forward import (
    RobustnessThresholds,
    WalkForwardFold,
    WalkForwardResult,
)
from strategies.swing.akc_helix.config import SYMBOL_CONFIGS

logger = logging.getLogger(__name__)


class HelixWalkForwardValidator:
    """Expanding-window walk-forward validation for Helix strategy."""

    def __init__(
        self,
        data: HelixPortfolioData,
        base_config: HelixBacktestConfig,
        test_window_months: int = 12,
        min_train_months: int = 24,
        purge_days: int = 30,
        n_coarse: int = 500,
        n_refine: int = 150,
        robustness: RobustnessThresholds | None = None,
    ):
        self.data = data
        self.base_config = base_config
        self.test_window_months = test_window_months
        self.min_train_months = min_train_months
        self.purge_days = purge_days
        self.n_coarse = n_coarse
        self.n_refine = n_refine
        self.robustness = robustness or RobustnessThresholds()

    def run(self) -> WalkForwardResult:
        """Execute walk-forward validation."""
        folds = self._generate_folds()
        if not folds:
            logger.warning("No valid folds generated")
            return WalkForwardResult()

        logger.info("Walk-forward: %d folds", len(folds))

        results = []
        for fold in folds:
            logger.info(
                "Fold %d: train %s-%s, test %s-%s",
                fold.fold_id,
                fold.train_start.date(), fold.train_end.date(),
                fold.test_start.date(), fold.test_end.date(),
            )
            result = self._run_fold(fold)
            results.append(result)

        return self._aggregate(results)

    def _generate_folds(self) -> list[WalkForwardFold]:
        """Generate expanding-window fold boundaries."""
        first_sym = next(iter(self.data.hourly), None)
        if first_sym is None:
            return []

        times = self.data.hourly[first_sym].times
        data_start = pd.Timestamp(times[0]).to_pydatetime().replace(tzinfo=timezone.utc)
        data_end = pd.Timestamp(times[-1]).to_pydatetime().replace(tzinfo=timezone.utc)

        folds = []
        fold_id = 0
        test_end = data_end

        while True:
            test_start = test_end - pd.DateOffset(months=self.test_window_months)
            train_end = test_start - pd.DateOffset(days=self.purge_days)
            train_start = data_start

            train_months = (train_end - train_start).days / 30.44
            if train_months < self.min_train_months:
                break

            folds.append(WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start.to_pydatetime() if hasattr(train_start, 'to_pydatetime') else train_start,
                train_end=train_end.to_pydatetime() if hasattr(train_end, 'to_pydatetime') else train_end,
                test_start=test_start.to_pydatetime() if hasattr(test_start, 'to_pydatetime') else test_start,
                test_end=test_end.to_pydatetime() if hasattr(test_end, 'to_pydatetime') else test_end,
            ))

            test_end = test_start
            fold_id += 1

        folds.reverse()
        for i, f in enumerate(folds):
            f.fold_id = i

        return folds

    def _run_fold(self, fold: WalkForwardFold) -> WalkForwardFold:
        """Run optimization on train period, evaluate on test period."""
        train_data = self._slice_data(fold.train_start, fold.train_end)
        test_data = self._slice_data(fold.test_start, fold.test_end)

        train_config = HelixBacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            data_dir=self.base_config.data_dir,
            track_shadows=False,
            warmup_daily=self.base_config.warmup_daily,
            warmup_hourly=self.base_config.warmup_hourly,
            warmup_4h=self.base_config.warmup_4h,
        )

        optimizer = HelixOptimizationRunner(
            base_config=train_config,
            data=train_data,
            n_coarse=self.n_coarse,
            n_refine=self.n_refine,
        )
        opt_result = optimizer.run()
        fold.best_params = opt_result.best_params
        fold.train_score = opt_result.best_score

        test_config = HelixBacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            param_overrides=opt_result.best_params,
            data_dir=self.base_config.data_dir,
            track_shadows=False,
            warmup_daily=self.base_config.warmup_daily,
            warmup_hourly=self.base_config.warmup_hourly,
            warmup_4h=self.base_config.warmup_4h,
        )

        test_result = run_helix_independent(test_data, test_config)

        all_pnls = []
        all_risks = []
        all_holds = []
        all_comms = []
        for sr in test_result.symbol_results.values():
            for t in sr.trades:
                all_pnls.append(t.pnl_dollars)
                sym_mult = SYMBOL_CONFIGS[t.symbol].multiplier if t.symbol in SYMBOL_CONFIGS else 1.0
                all_risks.append(abs(t.entry_price - t.initial_stop) * sym_mult * t.qty)
                all_holds.append(t.bars_held)
                all_comms.append(t.commission)

        if all_pnls:
            test_metrics = compute_metrics(
                trade_pnls=np.array(all_pnls),
                trade_risks=np.array(all_risks),
                trade_hold_hours=np.array(all_holds),
                trade_commissions=np.array(all_comms),
                equity_curve=test_result.combined_equity,
                timestamps=test_result.combined_timestamps,
                initial_equity=test_config.initial_equity,
            )
            fold.test_metrics = test_metrics
            fold.test_score = composite_objective(test_metrics)
            fold.test_trades = test_metrics.total_trades

        return fold

    def _slice_data(self, start, end) -> HelixPortfolioData:
        """Slice HelixPortfolioData to a time range with 4H recomputation."""
        sliced = HelixPortfolioData()

        start_ts = np.datetime64(start, 'ns')
        end_ts = np.datetime64(end, 'ns')

        for sym in self.data.hourly:
            h = self.data.hourly[sym]
            mask = (h.times >= start_ts) & (h.times <= end_ts)
            if not mask.any():
                continue

            sliced.hourly[sym] = NumpyBars(
                opens=h.opens[mask],
                highs=h.highs[mask],
                lows=h.lows[mask],
                closes=h.closes[mask],
                volumes=h.volumes[mask],
                times=h.times[mask],
            )

        for sym in self.data.daily:
            d = self.data.daily[sym]
            mask = (d.times >= start_ts) & (d.times <= end_ts)
            if not mask.any():
                continue

            sliced.daily[sym] = NumpyBars(
                opens=d.opens[mask],
                highs=d.highs[mask],
                lows=d.lows[mask],
                closes=d.closes[mask],
                volumes=d.volumes[mask],
                times=d.times[mask],
            )

        # Recompute alignment maps and 4H data for sliced ranges
        for sym in sliced.hourly:
            if sym not in sliced.daily:
                continue

            h_times = pd.DatetimeIndex(sliced.hourly[sym].times)
            d_times = pd.DatetimeIndex(sliced.daily[sym].times)
            h_df = pd.DataFrame(index=h_times)
            d_df = pd.DataFrame(index=d_times)
            sliced.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)

            # Resample 1H → 4H for sliced data
            h_df_full = pd.DataFrame({
                "open": sliced.hourly[sym].opens,
                "high": sliced.hourly[sym].highs,
                "low": sliced.hourly[sym].lows,
                "close": sliced.hourly[sym].closes,
                "volume": sliced.hourly[sym].volumes,
            }, index=h_times)

            from backtests.swing.data.preprocessing import build_numpy_arrays
            four_hour_df = resample_1h_to_4h(h_df_full)
            sliced.four_hour[sym] = build_numpy_arrays(four_hour_df)
            sliced.four_hour_idx_maps[sym] = align_4h_to_hourly(h_df_full, four_hour_df)

        return sliced

    def _aggregate(self, folds: list[WalkForwardFold]) -> WalkForwardResult:
        """Compute summary statistics across folds."""
        test_scores = [f.test_score for f in folds if f.test_score != 0]
        train_scores = [f.train_score for f in folds if f.train_score > 0]

        avg_test = float(np.mean(test_scores)) if test_scores else 0
        pct_positive = sum(1 for s in test_scores if s > 0) / len(test_scores) * 100 if test_scores else 0

        test_sharpes = [f.test_metrics.sharpe for f in folds if f.test_metrics]
        avg_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0

        if train_scores and test_scores:
            avg_train = float(np.mean(train_scores))
            degradation = avg_test / avg_train if avg_train > 0 else 0
        else:
            degradation = 0

        r = self.robustness
        failures: list[str] = []

        if pct_positive < r.min_pct_positive_folds:
            failures.append(
                f"positive_folds={pct_positive:.0f}% < {r.min_pct_positive_folds:.0f}%"
            )

        for f in folds:
            if f.test_metrics and f.test_metrics.max_drawdown_pct > r.max_drawdown_pct:
                failures.append(
                    f"fold_{f.fold_id}_dd={f.test_metrics.max_drawdown_pct:.1%} > {r.max_drawdown_pct:.1%}"
                )
                break

        for f in folds:
            if f.test_metrics and f.test_metrics.trades_per_month < r.min_trades_per_month_oos:
                failures.append(
                    f"fold_{f.fold_id}_trades/mo={f.test_metrics.trades_per_month:.1f} < {r.min_trades_per_month_oos:.1f}"
                )
                break

        if 0 < degradation < r.min_degradation_ratio:
            failures.append(
                f"degradation={degradation:.2f} < {r.min_degradation_ratio:.2f}"
            )

        passed = len(failures) == 0

        return WalkForwardResult(
            folds=folds,
            avg_test_score=avg_test,
            avg_test_sharpe=avg_sharpe,
            pct_positive_folds=pct_positive,
            degradation_ratio=degradation,
            passed=passed,
            failure_reasons=failures,
        )
