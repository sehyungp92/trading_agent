# skills/robustness_tester.py
"""Robustness tester — neighborhood stability, regime stability, and safety flags.

From roadmap §4.4:
- Neighborhood test: test params ±10%, ensure performance doesn't collapse
- Regime stability: profitable in at least 3 of 4 regime types
- Safety flags: flat surface → low conviction, spiky → likely overfit, fragile at higher costs
"""
from __future__ import annotations

import statistics

from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.schemas.parameter_space import ParameterSpace, RobustnessConfig
from trading_assistant.schemas.validation_results import RobustnessResult, SafetyFlag
from trading_assistant.skills.backtest_simulator import BacktestSimulator


class RobustnessTester:
    """Tests parameter robustness via neighborhood and regime analysis."""

    def __init__(
        self,
        config: RobustnessConfig,
        space: ParameterSpace,
        simulator: BacktestSimulator,
    ) -> None:
        self._config = config
        self._space = space
        self._sim = simulator

    def test_neighborhood(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        best_params: dict[str, float],
    ) -> dict[str, float]:
        """Test Sharpe at ±neighborhood_pct around best params. Returns label→Sharpe map."""
        scores: dict[str, float] = {}
        pct = self._config.neighborhood_pct

        for pdef in self._space.parameters:
            name = pdef.name
            best_val = best_params.get(name, pdef.current_value)
            for direction in [-1, 0, 1]:
                adjusted = best_val * (1 + direction * pct)
                adjusted = max(pdef.min_value, min(pdef.max_value, adjusted))
                neighbor = dict(best_params)
                neighbor[name] = adjusted
                label = f"{name}={adjusted:.4g}"
                result = self._sim.simulate(trades, missed, neighbor)
                scores[label] = result.sharpe_ratio

        return scores

    def test_regime_stability(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        params: dict[str, float],
    ) -> tuple[dict[str, float], int]:
        """Test PnL by regime. Returns (regime_pnl, profitable_regime_count)."""
        result = self._sim.simulate(trades, missed, params)
        regime_pnl = result.pnl_by_regime
        profitable = sum(1 for v in regime_pnl.values() if v > 0)
        return regime_pnl, profitable

    def evaluate(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        best_params: dict[str, float],
    ) -> RobustnessResult:
        """Full robustness evaluation: neighborhood + regime + score."""
        neighborhood = self.test_neighborhood(trades, missed, best_params)
        regime_pnl, profitable_count = self.test_regime_stability(trades, missed, best_params)

        best_sharpe_result = self._sim.simulate(trades, missed, best_params)
        best_sharpe = best_sharpe_result.sharpe_ratio

        neighborhood_stable = self._is_neighborhood_stable(neighborhood, best_sharpe)
        regime_stable = profitable_count >= self._config.min_profitable_regimes

        # Score: 50 points for neighborhood stability, 50 for regime stability
        score = 0.0
        if neighborhood_stable:
            score += 50.0
        elif neighborhood:
            avg_neighbor = statistics.mean(neighborhood.values()) if neighborhood else 0
            score += max(0, 50 * (avg_neighbor / best_sharpe)) if best_sharpe > 0 else 0

        regime_ratio = profitable_count / self._config.total_regime_types if self._config.total_regime_types > 0 else 0
        score += 50 * regime_ratio

        self.detect_safety_flags(neighborhood, regime_stable, best_sharpe)

        return RobustnessResult(
            neighborhood_scores=neighborhood,
            neighborhood_stable=neighborhood_stable,
            regime_pnl=regime_pnl,
            profitable_regime_count=profitable_count,
            regime_stable=regime_stable,
            robustness_score=min(100, max(0, score)),
        )

    def _is_neighborhood_stable(self, scores: dict[str, float], best_sharpe: float) -> bool:
        """Neighborhood is stable if no neighbor drops by more than 50% from best."""
        if not scores or best_sharpe <= 0:
            return False
        return all(v >= best_sharpe * 0.5 for v in scores.values())

    def detect_safety_flags(
        self,
        neighborhood_scores: dict[str, float],
        regime_stable: bool,
        best_sharpe: float,
    ) -> list[SafetyFlag]:
        """Detect safety flags from robustness test results."""
        flags: list[SafetyFlag] = []
        if not neighborhood_scores:
            return flags

        values = list(neighborhood_scores.values())

        # Flat surface: all scores within 5% of each other → low conviction
        if len(values) >= 2:
            spread = max(values) - min(values)
            mean_val = statistics.mean(values)
            if mean_val > 0 and spread / mean_val < 0.05:
                flags.append(SafetyFlag(
                    flag_type="low_conviction",
                    description="Flat optimization surface: all neighbors within 5% — parameter choice has little impact",
                    severity="high",
                ))

        # Spiky surface: any neighbor drops >50% from best → likely overfit
        if best_sharpe > 0:
            worst_neighbor = min(values)
            if worst_neighbor < best_sharpe * 0.5:
                flags.append(SafetyFlag(
                    flag_type="likely_overfit",
                    description=f"Spiky optimization surface: ±{self._config.neighborhood_pct:.0%} params reduce Sharpe by >{50}%",
                    severity="high",
                ))

        if not regime_stable:
            flags.append(SafetyFlag(
                flag_type="regime_unstable",
                description=f"Not profitable in {self._config.min_profitable_regimes}/{self._config.total_regime_types} regimes",
                severity="medium",
            ))

        return flags
