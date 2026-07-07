"""Walk-forward validation with expanding training windows."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtests.swing.analysis.metrics import PerformanceMetrics, compute_metrics
from backtests.swing.config import BacktestConfig
from backtests.swing.engine.portfolio_engine import PortfolioData, run_independent
from backtests.swing.optimization.objective import composite_objective
from backtests.swing.optimization.runner import OptimizationRunner, TrialResult
from strategies.swing.atrss.config import SYMBOL_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Result of one walk-forward fold."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, float] = field(default_factory=dict)
    train_score: float = 0.0
    test_score: float = 0.0
    test_metrics: PerformanceMetrics | None = None
    test_trades: int = 0


@dataclass
class RobustnessThresholds:
    """Configurable pass/fail thresholds for walk-forward validation."""

    min_pct_positive_folds: float = 60.0       # % of folds with positive expectancy
    max_drawdown_pct: float = 0.25             # reject if any fold exceeds this DD
    min_degradation_ratio: float = 0.30        # test/train score ratio floor
    min_trades_per_month_oos: float = 1.0      # OOS trade frequency floor
    max_single_instrument_pct: float = 0.70    # max % of trades from one instrument


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    avg_test_score: float = 0.0
    avg_test_sharpe: float = 0.0
    pct_positive_folds: float = 0.0
    degradation_ratio: float = 0.0  # avg test/train score ratio
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)


class WalkForwardValidator:
    """Expanding-window walk-forward validation.

    Example with 5Y data (2019-2024):
    - Fold 1: Train 2019-2021, Test 2022
    - Fold 2: Train 2019-2022, Test 2023
    - Fold 3: Train 2019-2023, Test 2024
    """

    def __init__(
        self,
        data: PortfolioData,
        base_config: BacktestConfig,
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
        # Find data range from the first symbol's hourly timestamps
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

            # Check minimum training period
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

        # Reverse so earliest fold is first
        folds.reverse()
        for i, f in enumerate(folds):
            f.fold_id = i

        return folds

    def _run_fold(self, fold: WalkForwardFold) -> WalkForwardFold:
        """Run optimization on train period, evaluate on test period."""
        # Slice data for training period
        train_data = self._slice_data(fold.train_start, fold.train_end)
        test_data = self._slice_data(fold.test_start, fold.test_end)

        # Optimize on training data
        train_config = BacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            data_dir=self.base_config.data_dir,
            track_shadows=False,
            warmup_daily=self.base_config.warmup_daily,
            warmup_hourly=self.base_config.warmup_hourly,
        )

        optimizer = OptimizationRunner(
            base_config=train_config,
            data=train_data,
            n_coarse=self.n_coarse,
            n_refine=self.n_refine,
        )
        opt_result = optimizer.run()
        fold.best_params = opt_result.best_params
        fold.train_score = opt_result.best_score

        # Evaluate best params on test data
        test_config = BacktestConfig(
            symbols=self.base_config.symbols,
            initial_equity=self.base_config.initial_equity,
            slippage=self.base_config.slippage,
            flags=self.base_config.flags,
            param_overrides=opt_result.best_params,
            data_dir=self.base_config.data_dir,
            track_shadows=False,
            warmup_daily=self.base_config.warmup_daily,
            warmup_hourly=self.base_config.warmup_hourly,
        )

        test_result = run_independent(test_data, test_config)

        # Compute test metrics
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

    def _slice_data(
        self,
        start: datetime,
        end: datetime,
    ) -> PortfolioData:
        """Slice PortfolioData to a time range."""
        sliced = PortfolioData()

        start_ts = np.datetime64(start, 'ns')
        end_ts = np.datetime64(end, 'ns')

        for sym in self.data.hourly:
            h = self.data.hourly[sym]
            mask = (h.times >= start_ts) & (h.times <= end_ts)
            if not mask.any():
                continue

            from backtests.swing.data.preprocessing import NumpyBars
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

            from backtests.swing.data.preprocessing import NumpyBars
            sliced.daily[sym] = NumpyBars(
                opens=d.opens[mask],
                highs=d.highs[mask],
                lows=d.lows[mask],
                closes=d.closes[mask],
                volumes=d.volumes[mask],
                times=d.times[mask],
            )

        # Recompute daily_idx_maps for sliced data
        for sym in sliced.hourly:
            if sym not in sliced.daily:
                continue
            from backtests.swing.data.preprocessing import align_daily_to_hourly
            # Build temporary DataFrames for alignment
            h_times = pd.DatetimeIndex(sliced.hourly[sym].times)
            d_times = pd.DatetimeIndex(sliced.daily[sym].times)
            h_df = pd.DataFrame(index=h_times)
            d_df = pd.DataFrame(index=d_times)
            sliced.daily_idx_maps[sym] = align_daily_to_hourly(h_df, d_df)

        return sliced

    def _aggregate(self, folds: list[WalkForwardFold]) -> WalkForwardResult:
        """Compute summary statistics across folds and enforce robustness constraints."""
        test_scores = [f.test_score for f in folds if f.test_score != 0]
        train_scores = [f.train_score for f in folds if f.train_score > 0]

        avg_test = float(np.mean(test_scores)) if test_scores else 0
        pct_positive = sum(1 for s in test_scores if s > 0) / len(test_scores) * 100 if test_scores else 0

        test_sharpes = [f.test_metrics.sharpe for f in folds if f.test_metrics]
        avg_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0

        # Degradation ratio
        if train_scores and test_scores:
            avg_train = float(np.mean(train_scores))
            degradation = avg_test / avg_train if avg_train > 0 else 0
        else:
            degradation = 0

        # --- Robustness constraint checking ---
        r = self.robustness
        failures: list[str] = []

        # 1. Positive expectancy in most test windows
        if pct_positive < r.min_pct_positive_folds:
            failures.append(
                f"positive_folds={pct_positive:.0f}% < {r.min_pct_positive_folds:.0f}%"
            )

        # 2. Max DD within acceptable band
        for f in folds:
            if f.test_metrics and f.test_metrics.max_drawdown_pct > r.max_drawdown_pct:
                failures.append(
                    f"fold_{f.fold_id}_dd={f.test_metrics.max_drawdown_pct:.1%} > {r.max_drawdown_pct:.1%}"
                )
                break  # one violation is enough

        # 3. Trade frequency not collapsing OOS
        for f in folds:
            if f.test_metrics and f.test_metrics.trades_per_month < r.min_trades_per_month_oos:
                failures.append(
                    f"fold_{f.fold_id}_trades/mo={f.test_metrics.trades_per_month:.1f} < {r.min_trades_per_month_oos:.1f}"
                )
                break

        # 4. Degradation ratio
        if degradation < r.min_degradation_ratio and degradation > 0:
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
