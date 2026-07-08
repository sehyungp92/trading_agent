"""Shared optimization result dataclasses.

Strategy-specific runners (apex_runner, nqdtc_runner, vdubus_runner)
import TrialResult and OptimizationResult from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
