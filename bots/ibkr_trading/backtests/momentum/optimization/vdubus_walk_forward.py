"""VdubusNQ walk-forward validation with expanding training windows.

Mirrors NQDTCWalkForwardValidator but uses VdubusNQ types and 15m data.
Dual-instrument: NQ 15m (primary) + ES daily (regime).
24m train / 6m test / 6m step per backtesting_3.md spec.
"""
from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from backtests.momentum.analysis.metrics import compute_metrics
from backtests.momentum.config_vdubus import VdubusBacktestConfig
from backtests.momentum.data.preprocessing import (
    NumpyBars,
    align_5m_to_15m,
    align_daily_to_15m,
    align_higher_tf_to_15m,
    build_numpy_arrays,
    resample_15m_to_1h,
    resample_5m_to_15m,
)
from backtests.momentum.optimization.objective import composite_objective
from backtests.momentum.optimization.vdubus_runner import VdubusOptimizationRunner
from backtests.momentum.optimization.walk_forward import (
    RobustnessThresholds,
    WalkForwardFold,
    WalkForwardResult,
)

logger = logging.getLogger(__name__)


class VdubusWalkForwardValidator:
    """Expanding-window walk-forward validation for VdubusNQ v4.0.

    Default windows: 24m train / 6m test / 6m step.
    """

    def __init__(
        self,
        vdubus_data: dict,
        base_config: VdubusBacktestConfig,
        test_window_months: int = 6,
        min_train_months: int = 24,
        purge_days: int = 30,
        n_coarse: int = 300,
        n_refine: int = 100,
        robustness: RobustnessThresholds | None = None,
    ):
        self.vdubus_data = vdubus_data
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
        """Generate expanding-window fold boundaries from 15m bar timestamps."""
        bars_15m = self.vdubus_data["bars_15m"]
        times = bars_15m.times

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

        train_config = VdubusBacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            data_dir=self.base_config.data_dir,
            track_signals=False,
            track_shadows=False,
            warmup_daily_es=self.base_config.warmup_daily_es,
            warmup_1h=self.base_config.warmup_1h,
            warmup_15m=self.base_config.warmup_15m,
            warmup_5m=self.base_config.warmup_5m,
        )

        optimizer = VdubusOptimizationRunner(
            base_config=train_config,
            vdubus_data=train_data,
            n_coarse=self.n_coarse,
            n_refine=self.n_refine,
        )
        opt_result = optimizer.run()
        fold.best_params = opt_result.best_params
        fold.train_score = opt_result.best_score

        # Test phase
        test_config = VdubusBacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            param_overrides=opt_result.best_params,
            data_dir=self.base_config.data_dir,
            track_signals=False,
            track_shadows=False,
            warmup_daily_es=self.base_config.warmup_daily_es,
            warmup_1h=self.base_config.warmup_1h,
            warmup_15m=self.base_config.warmup_15m,
            warmup_5m=self.base_config.warmup_5m,
        )

        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        engine = VdubusEngine(test_config.symbols[0], test_config)
        test_result = engine.run(
            test_data["bars_15m"],
            test_data.get("bars_5m"),
            test_data["hourly"],
            test_data["daily_es"],
            test_data["hourly_idx_map"],
            test_data["daily_es_idx_map"],
            test_data.get("five_to_15_idx_map"),
        )

        if test_result.trades:
            pnls = np.array([t.pnl_dollars for t in test_result.trades])
            risks = np.array([
                abs(t.entry_price - t.initial_stop) * self.base_config.point_value
                for t in test_result.trades
            ])
            holds = np.array([t.bars_held_15m for t in test_result.trades])
            comms = np.array([t.commission for t in test_result.trades])

            test_metrics = compute_metrics(
                trade_pnls=pnls,
                trade_risks=risks,
                trade_hold_hours=holds,
                trade_commissions=comms,
                equity_curve=test_result.equity_curve,
                timestamps=test_result.time_series,
                initial_equity=test_config.initial_equity,
            )
            fold.test_metrics = test_metrics
            fold.test_score = composite_objective(
                test_metrics, min_trades_per_month=1.0, min_total_trades=20,
            )
            fold.test_trades = test_metrics.total_trades

        return fold

    def _slice_data(self, start, end) -> dict:
        """Slice VdubusNQ data to a time range, recomputing resampled bars.

        Handles dual-instrument data: NQ 15m (primary) + ES daily (auxiliary).
        Optionally includes NQ 5m for micro-trigger.
        """
        bars_15m = self.vdubus_data["bars_15m"]
        start_ts = np.datetime64(start, 'ns')
        end_ts = np.datetime64(end, 'ns')

        # Slice 15m NQ bars
        mask_15m = (bars_15m.times >= start_ts) & (bars_15m.times <= end_ts)
        if not mask_15m.any():
            return self.vdubus_data

        sliced_15m = NumpyBars(
            opens=bars_15m.opens[mask_15m],
            highs=bars_15m.highs[mask_15m],
            lows=bars_15m.lows[mask_15m],
            closes=bars_15m.closes[mask_15m],
            volumes=bars_15m.volumes[mask_15m],
            times=bars_15m.times[mask_15m],
        )

        # Resample 15m → 1H
        m_df = pd.DataFrame({
            "open": sliced_15m.opens,
            "high": sliced_15m.highs,
            "low": sliced_15m.lows,
            "close": sliced_15m.closes,
            "volume": sliced_15m.volumes,
        }, index=pd.DatetimeIndex(sliced_15m.times))

        hourly_df = resample_15m_to_1h(m_df)
        hourly = build_numpy_arrays(hourly_df)
        hourly_idx_map = align_higher_tf_to_15m(m_df, hourly_df)

        # Slice ES daily bars
        daily_es_src = self.vdubus_data["daily_es"]
        mask_daily = (daily_es_src.times >= start_ts) & (daily_es_src.times <= end_ts)
        if mask_daily.any():
            daily_es = NumpyBars(
                opens=daily_es_src.opens[mask_daily],
                highs=daily_es_src.highs[mask_daily],
                lows=daily_es_src.lows[mask_daily],
                closes=daily_es_src.closes[mask_daily],
                volumes=daily_es_src.volumes[mask_daily],
                times=daily_es_src.times[mask_daily],
            )
        else:
            daily_es = daily_es_src

        # Build ES daily → 15m alignment
        daily_es_df = pd.DataFrame({
            "open": daily_es.opens,
            "high": daily_es.highs,
            "low": daily_es.lows,
            "close": daily_es.closes,
            "volume": daily_es.volumes,
        }, index=pd.DatetimeIndex(daily_es.times))
        daily_es_idx_map = align_daily_to_15m(m_df, daily_es_df)

        result = {
            "bars_15m": sliced_15m,
            "hourly": hourly,
            "daily_es": daily_es,
            "hourly_idx_map": hourly_idx_map,
            "daily_es_idx_map": daily_es_idx_map,
        }

        # Optional: slice 5m bars for micro-trigger
        bars_5m = self.vdubus_data.get("bars_5m")
        if bars_5m is not None:
            mask_5m = (bars_5m.times >= start_ts) & (bars_5m.times <= end_ts)
            if mask_5m.any():
                sliced_5m = NumpyBars(
                    opens=bars_5m.opens[mask_5m],
                    highs=bars_5m.highs[mask_5m],
                    lows=bars_5m.lows[mask_5m],
                    closes=bars_5m.closes[mask_5m],
                    volumes=bars_5m.volumes[mask_5m],
                    times=bars_5m.times[mask_5m],
                )
                five_df = pd.DataFrame({
                    "open": sliced_5m.opens,
                    "high": sliced_5m.highs,
                    "low": sliced_5m.lows,
                    "close": sliced_5m.closes,
                    "volume": sliced_5m.volumes,
                }, index=pd.DatetimeIndex(sliced_5m.times))
                five_to_15_idx_map = align_5m_to_15m(five_df, m_df)
                result["bars_5m"] = sliced_5m
                result["five_to_15_idx_map"] = five_to_15_idx_map

        return result

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
