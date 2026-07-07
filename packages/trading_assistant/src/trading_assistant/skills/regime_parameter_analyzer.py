"""Regime-conditional parameter analyzer.

Analyzes whether parameters perform differently across regimes and enables
"in regime X, change Y to Z" proposals. Stratifies trades by market_regime,
runs grid search within each regime's subset, and computes sensitivity scores.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from trading_assistant.schemas.parameter_definition import ParameterDefinition
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.regime_conditional import RegimeParameterAnalysis, RegimeParameterStats


_MIN_TRADES_PER_REGIME = 15
_SENSITIVITY_THRESHOLD = 0.3  # regime_sensitivity > 0.3 → meaningfully different


class RegimeParameterAnalyzer:
    """Analyzes per-regime optimal parameter values."""

    def _stratify_by_regime(
        self, trades: list[TradeEvent],
    ) -> dict[str, list[TradeEvent]]:
        """Group trades by market_regime."""
        groups: dict[str, list[TradeEvent]] = defaultdict(list)
        for t in trades:
            regime = t.market_regime or "unknown"
            groups[regime].append(t)
        return groups

    def _build_grid(self, param: ParameterDefinition, n_points: int = 11) -> list[Any]:
        """Build parameter grid for search."""
        if param.valid_values:
            return list(param.valid_values)
        if param.valid_range:
            lo, hi = param.valid_range
            if param.value_type == "int":
                step = max(1, int((hi - lo) / (n_points - 1)))
                return list(range(int(lo), int(hi) + 1, step))
            return [lo + (hi - lo) * i / (n_points - 1) for i in range(n_points)]
        return []

    def _evaluate_value(
        self, trades: list[TradeEvent], param: ParameterDefinition, value: Any,
    ) -> tuple[float, float, float] | None:
        """Evaluate a parameter value on a set of trades.

        Returns (win_rate, avg_pnl, profit_factor) or None if no trades match.
        Uses strategy_params_at_entry to filter trades where param was at `value`.
        """
        # Group trades by their actual param value
        param_name = param.param_name
        matching: list[TradeEvent] = []
        for t in trades:
            if not t.strategy_params_at_entry:
                continue
            actual = t.strategy_params_at_entry.get(param_name)
            if actual is None:
                continue
            # For boolean params
            if isinstance(value, bool):
                if actual == value:
                    matching.append(t)
            else:
                try:
                    if abs(float(actual) - float(value)) < 1e-6:
                        matching.append(t)
                except (TypeError, ValueError):
                    pass

        if not matching:
            return None  # No trades for this value — skip in grid search

        wins = sum(1 for t in matching if t.pnl > 0)
        pnls = [t.pnl for t in matching]
        wr = wins / n if (n := len(matching)) > 0 else 0.0
        avg_pnl = statistics.mean(pnls) if pnls else 0.0
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )
        return wr, avg_pnl, min(pf, 99.99)

    def _find_regime_optimal(
        self,
        regime_trades: list[TradeEvent],
        param: ParameterDefinition,
        grid: list[Any],
    ) -> tuple[Any, float, float, float]:
        """Find optimal value for a parameter within a regime's trades.

        Returns (optimal_value, win_rate, avg_pnl, profit_factor).
        """
        best_value = param.current_value
        best_pnl = float("-inf")
        best_wr = 0.0
        best_pf = 0.0

        for value in grid:
            result = self._evaluate_value(regime_trades, param, value)
            if result is None:
                continue  # No trades for this value — skip
            wr, avg_pnl, pf = result
            if avg_pnl > best_pnl:
                best_value = value
                best_pnl = avg_pnl
                best_wr = wr
                best_pf = pf

        return best_value, best_wr, best_pnl, best_pf

    def _compute_sensitivity(
        self, optimal_values: list[Any], param: ParameterDefinition,
    ) -> float:
        """Compute regime sensitivity: normalized variance of optimal values.

        Returns 0.0-1.0 where higher means more regime-dependent.
        """
        if len(optimal_values) < 2:
            return 0.0

        # For boolean params: sensitivity = 1.0 if different, 0.0 if same
        if param.value_type == "bool" or param.valid_values:
            unique = set(str(v) for v in optimal_values)
            return 1.0 if len(unique) > 1 else 0.0

        # For numeric params: std(optima) / range
        try:
            numeric = [float(v) for v in optimal_values]
        except (TypeError, ValueError):
            return 0.0

        if param.valid_range:
            range_size = param.valid_range[1] - param.valid_range[0]
            if range_size <= 0:
                return 0.0
            std = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
            return min(std / range_size, 1.0)

        # No valid_range: use coefficient of variation
        mean = statistics.mean(numeric)
        if mean == 0:
            return 0.0
        std = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        return min(std / abs(mean), 1.0)

    def analyze(
        self,
        trades: list[TradeEvent],
        param: ParameterDefinition,
        bot_id: str,
        min_per_regime: int = _MIN_TRADES_PER_REGIME,
    ) -> RegimeParameterAnalysis:
        """Analyze regime-conditional parameter performance.

        For each regime with >= min_per_regime trades, find optimal value.
        Compute regime_sensitivity = std(optimal_values) / range(valid_range).
        """
        if not trades:
            return RegimeParameterAnalysis(
                param_name=param.param_name, bot_id=bot_id,
                current_value=param.current_value,
            )

        regime_groups = self._stratify_by_regime(trades)
        grid = self._build_grid(param)
        if not grid:
            return RegimeParameterAnalysis(
                param_name=param.param_name, bot_id=bot_id,
                current_value=param.current_value,
            )

        regimes_analyzed: list[str] = []
        optimal_per_regime: dict[str, Any] = {}
        regime_stats: list[RegimeParameterStats] = []
        optimal_values: list[Any] = []

        for regime, rtrades in sorted(regime_groups.items()):
            if regime == "unknown" or len(rtrades) < min_per_regime:
                continue

            opt_val, wr, avg_pnl, pf = self._find_regime_optimal(rtrades, param, grid)
            regimes_analyzed.append(regime)
            optimal_per_regime[regime] = opt_val
            optimal_values.append(opt_val)
            regime_stats.append(RegimeParameterStats(
                regime=regime,
                trade_count=len(rtrades),
                optimal_value=opt_val,
                win_rate=round(wr, 4),
                avg_pnl=round(avg_pnl, 4),
                profit_factor=round(pf, 4),
            ))

        sensitivity = self._compute_sensitivity(optimal_values, param)

        # Generate recommendations
        recommendations: list[str] = []
        if sensitivity > _SENSITIVITY_THRESHOLD:
            for regime, opt_val in optimal_per_regime.items():
                if str(opt_val) != str(param.current_value):
                    recommendations.append(
                        f"In {regime} regime: consider {param.param_name}={opt_val} "
                        f"(current: {param.current_value})"
                    )

        return RegimeParameterAnalysis(
            param_name=param.param_name,
            bot_id=bot_id,
            strategy_id=param.strategy_id or "",
            regimes_analyzed=regimes_analyzed,
            optimal_per_regime=optimal_per_regime,
            current_value=param.current_value,
            regime_sensitivity=round(sensitivity, 4),
            regime_stats=regime_stats,
            recommendations=recommendations,
        )
