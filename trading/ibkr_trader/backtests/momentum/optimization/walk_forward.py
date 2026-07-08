"""Shared walk-forward validation dataclasses.

Strategy-specific validators (apex_walk_forward, nqdtc_walk_forward,
vdubus_walk_forward) import these from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backtests.momentum.analysis.metrics import PerformanceMetrics


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
