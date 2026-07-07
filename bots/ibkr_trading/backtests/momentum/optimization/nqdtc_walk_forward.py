"""NQDTC walk-forward validation with expanding training windows.

Mirrors ApexWalkForwardValidator but uses NQDTC types and 5-min data.
Reuses WalkForwardFold, RobustnessThresholds, WalkForwardResult.
24m train / 6m test / 6m step per backtesting_2.md spec.
"""
from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from backtests.momentum.analysis.metrics import compute_metrics
from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.data.preprocessing import (
    NumpyBars,
    align_daily_to_5m,
    align_higher_tf_to_5m,
    build_numpy_arrays,
    resample_5m_to_1h,
    resample_5m_to_30m,
    resample_5m_to_4h,
    resample_5m_to_daily,
)
from backtests.momentum.optimization.nqdtc_runner import NQDTCOptimizationRunner
from backtests.momentum.optimization.objective import composite_objective
from backtests.momentum.optimization.walk_forward import (
    RobustnessThresholds,
    WalkForwardFold,
    WalkForwardResult,
)

logger = logging.getLogger(__name__)


class NQDTCWalkForwardValidator:
    """Expanding-window walk-forward validation for NQDTC v2.0.

    Default windows: 24m train / 6m test / 6m step.
    """

    def __init__(
        self,
        nqdtc_data: dict,
        base_config: NQDTCBacktestConfig,
        test_window_months: int = 6,
        min_train_months: int = 24,
        purge_days: int = 30,
        n_coarse: int = 300,
        n_refine: int = 100,
        robustness: RobustnessThresholds | None = None,
    ):
        self.nqdtc_data = nqdtc_data
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
        """Generate expanding-window fold boundaries from 5m bar timestamps."""
        five_min_bars = self.nqdtc_data["five_min_bars"]
        times = five_min_bars.times

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

        train_config = NQDTCBacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            data_dir=self.base_config.data_dir,
            track_signals=False,
            track_shadows=False,
            warmup_daily=self.base_config.warmup_daily,
            warmup_30m=self.base_config.warmup_30m,
            warmup_1h=self.base_config.warmup_1h,
            warmup_4h=self.base_config.warmup_4h,
            warmup_5m=self.base_config.warmup_5m,
        )

        optimizer = NQDTCOptimizationRunner(
            base_config=train_config,
            nqdtc_data=train_data,
            n_coarse=self.n_coarse,
            n_refine=self.n_refine,
        )
        opt_result = optimizer.run()
        fold.best_params = opt_result.best_params
        fold.train_score = opt_result.best_score

        # Test phase
        test_config = NQDTCBacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            param_overrides=opt_result.best_params,
            data_dir=self.base_config.data_dir,
            track_signals=False,
            track_shadows=False,
            warmup_daily=self.base_config.warmup_daily,
            warmup_30m=self.base_config.warmup_30m,
            warmup_1h=self.base_config.warmup_1h,
            warmup_4h=self.base_config.warmup_4h,
            warmup_5m=self.base_config.warmup_5m,
        )

        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        engine = NQDTCEngine(test_config.symbols[0], test_config)
        test_result = engine.run(
            test_data["five_min_bars"],
            test_data["thirty_min"],
            test_data["hourly"],
            test_data["four_hour"],
            test_data["daily"],
            test_data["thirty_min_idx_map"],
            test_data["hourly_idx_map"],
            test_data["four_hour_idx_map"],
            test_data["daily_idx_map"],
            daily_es=test_data.get("daily_es"),
            daily_es_idx_map=test_data.get("daily_es_idx_map"),
        )

        if test_result.trades:
            pnls = np.array([t.pnl_dollars for t in test_result.trades])
            risks = np.array([
                abs(t.entry_price - t.initial_stop) * self.base_config.point_value
                for t in test_result.trades
            ])
            holds = np.array([t.bars_held_30m for t in test_result.trades])
            comms = np.array([t.commission for t in test_result.trades])

            test_metrics = compute_metrics(
                trade_pnls=pnls,
                trade_risks=risks,
                trade_hold_hours=holds,
                trade_commissions=comms,
                equity_curve=test_result.equity_curve,
                timestamps=test_result.timestamps,
                initial_equity=test_config.initial_equity,
            )
            fold.test_metrics = test_metrics
            fold.test_score = composite_objective(
                test_metrics, min_trades_per_month=1.0, min_total_trades=20,
            )
            fold.test_trades = test_metrics.total_trades

        return fold

    def _slice_data(self, start, end) -> dict:
        """Slice NQDTC data to a time range, recomputing resampled bars."""
        five_min_bars = self.nqdtc_data["five_min_bars"]
        start_ts = np.datetime64(start, 'ns')
        end_ts = np.datetime64(end, 'ns')

        mask = (five_min_bars.times >= start_ts) & (five_min_bars.times <= end_ts)
        if not mask.any():
            return self.nqdtc_data

        sliced_5m = NumpyBars(
            opens=five_min_bars.opens[mask],
            highs=five_min_bars.highs[mask],
            lows=five_min_bars.lows[mask],
            closes=five_min_bars.closes[mask],
            volumes=five_min_bars.volumes[mask],
            times=five_min_bars.times[mask],
        )

        # Resample from sliced 5-minute bars
        m_df = pd.DataFrame({
            "open": sliced_5m.opens,
            "high": sliced_5m.highs,
            "low": sliced_5m.lows,
            "close": sliced_5m.closes,
            "volume": sliced_5m.volumes,
        }, index=pd.DatetimeIndex(sliced_5m.times))

        thirty_min_df = resample_5m_to_30m(m_df)
        hourly_df = resample_5m_to_1h(m_df)
        four_hour_df = resample_5m_to_4h(m_df)
        daily_df = resample_5m_to_daily(m_df)

        thirty_min = build_numpy_arrays(thirty_min_df)
        hourly = build_numpy_arrays(hourly_df)
        four_hour = build_numpy_arrays(four_hour_df)
        daily = build_numpy_arrays(daily_df)

        thirty_min_idx_map = align_higher_tf_to_5m(m_df, thirty_min_df)
        hourly_idx_map = align_higher_tf_to_5m(m_df, hourly_df)
        four_hour_idx_map = align_higher_tf_to_5m(m_df, four_hour_df)
        daily_idx_map = align_daily_to_5m(m_df, daily_df)

        return {
            "five_min_bars": sliced_5m,
            "thirty_min": thirty_min,
            "hourly": hourly,
            "four_hour": four_hour,
            "daily": daily,
            "thirty_min_idx_map": thirty_min_idx_map,
            "hourly_idx_map": hourly_idx_map,
            "four_hour_idx_map": four_hour_idx_map,
            "daily_idx_map": daily_idx_map,
        }

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
