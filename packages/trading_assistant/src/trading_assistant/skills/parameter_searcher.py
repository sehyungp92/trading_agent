# skills/parameter_searcher.py
"""Autoresearch-style helper loop: explore parameter neighborhoods, pick best.

This local triage score is a legacy/helper composite for proposal screening.
Replay-backed monthly/phased-auto ranking uses immutable score profiles instead.
"""
from __future__ import annotations

import logging
from typing import Any

from trading_assistant.schemas.parameter_definition import ParameterDefinition
from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.schemas.objective_weights import (
    W_CALMAR_NO_PROCESS,
    W_EXPECTANCY_NO_PROCESS,
    W_EXPECTED_R_NO_PROCESS,
    W_INV_DRAWDOWN_NO_PROCESS,
    W_PROFIT_FACTOR_NO_PROCESS,
)
from trading_assistant.schemas.parameter_search import (
    CandidateResult,
    ParameterSearchReport,
    SearchRouting,
)
from trading_assistant.skills.regime_parameter_analyzer import RegimeParameterAnalyzer
from trading_assistant.schemas.parameter_space import ParameterDef, ParameterSpace, RobustnessConfig
from trading_assistant.schemas.simulation_metrics import SimulationMetrics
from trading_assistant.skills.backtest_simulator import BacktestSimulator
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.cost_model import CostModel
from trading_assistant.skills.robustness_tester import RobustnessTester

logger = logging.getLogger(__name__)

# Routing thresholds (module-level, easy to tune)
_APPROVE_IMPROVEMENT = 1.05
_APPROVE_ROBUSTNESS = 70.0
_EXPERIMENT_IMPROVEMENT = 0.95
_EXPERIMENT_ROBUSTNESS = 50.0
_SAFETY_CRITICAL_IMPROVEMENT = 1.10
_COST_SENSITIVITY_MULT = 1.5
_MAX_CANDIDATES = 11
_REGIME_SENSITIVITY_THRESHOLD = 0.3


class ParameterSearcher:
    """Explore parameter neighborhoods and route helper proposals."""

    def __init__(
        self,
        config_registry: ConfigRegistry,
        simulator: BacktestSimulator,
        cost_model: CostModel,
    ) -> None:
        self._registry = config_registry
        self._simulator = simulator
        self._cost_model = cost_model

    def search(
        self,
        suggestion_id: str,
        bot_id: str,
        param: ParameterDefinition,
        proposed_value: Any,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
    ) -> ParameterSearchReport:
        """Explore parameter neighborhood and route to APPROVE/EXPERIMENT/DISCARD."""
        current_value = param.current_value

        if not trades:
            return ParameterSearchReport(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                param_name=param.param_name,
                original_proposed=proposed_value,
                current_value=current_value,
                routing=SearchRouting.DISCARD,
                discard_reason="No trade data available",
            )

        # 1. Compute baseline at current_value
        baseline_metrics = self._simulator.simulate(
            trades, missed, {param.param_name: current_value},
        )
        baseline_composite = self._composite_score(baseline_metrics, baseline_metrics)

        # 2. Build candidate grid
        candidates = self._build_grid(param, proposed_value)

        # 3. Evaluate each candidate
        results: list[CandidateResult] = []
        for value in candidates:
            result = self._evaluate_candidate(
                param, value, trades, missed, baseline_metrics,
            )
            results.append(result)

        # 4. Filter passing candidates, rank by composite_score
        passing = [r for r in results if r.passes_safety]
        passing.sort(key=lambda r: r.composite_score, reverse=True)

        best = passing[0] if passing else None

        # 5. Route
        routing, discard_reason = self._route(
            best, baseline_composite, param.is_safety_critical,
        )

        summary = self._build_summary(
            param, proposed_value, candidates, results, best, routing,
        )

        return ParameterSearchReport(
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            param_name=param.param_name,
            original_proposed=proposed_value,
            current_value=current_value,
            baseline_composite=baseline_composite,
            candidates_tested=len(candidates),
            candidates_passing=len(passing),
            best_value=best.value if best else None,
            best_composite=best.composite_score if best else 0.0,
            best_robustness_score=best.robustness_score if best else 0.0,
            best_cost_sensitivity_sharpe=best.cost_sensitivity_sharpe if best else 0.0,
            routing=routing,
            discard_reason=discard_reason,
            exploration_summary=summary,
        )

    def search_with_regime_analysis(
        self,
        suggestion_id: str,
        bot_id: str,
        param: ParameterDefinition,
        proposed_value: Any,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
    ) -> ParameterSearchReport:
        """Search with regime-conditional enrichment.

        1. Run flat search() first (counterfactual simulation)
        2. Run RegimeParameterAnalyzer (observational: actual params from trades)
        3. If sensitivity > threshold, annotate exploration_summary
        4. If flat DISCARD but regime analysis shows divergent optima → EXPERIMENT
        """
        report = self.search(
            suggestion_id, bot_id, param, proposed_value, trades, missed,
        )

        # Run observational regime analysis — wrapped in try/except so a
        # regime analyzer failure never discards the already-computed flat result.
        try:
            analyzer = RegimeParameterAnalyzer()
            regime_result = analyzer.analyze(trades, param, bot_id)
        except Exception:
            logger.warning(
                "Regime analysis failed for %s; returning flat search result",
                suggestion_id, exc_info=True,
            )
            return report

        report.regime_analysis = regime_result

        if regime_result.regime_sensitivity <= _REGIME_SENSITIVITY_THRESHOLD:
            return report

        # Append regime info to exploration summary
        regime_lines = [
            f"\nRegime sensitivity: {regime_result.regime_sensitivity:.2f}",
        ]
        for rec in regime_result.recommendations:
            regime_lines.append(f"  {rec}")
        report.exploration_summary += "\n".join(regime_lines)

        # Override: flat DISCARD + divergent regime optima → rescue to EXPERIMENT
        if report.routing == SearchRouting.DISCARD and regime_result.regime_stats:
            has_regime_opportunity = any(
                rs.avg_pnl > 0
                and rs.trade_count >= 15
                and str(rs.optimal_value) != str(param.current_value)
                for rs in regime_result.regime_stats
            )
            if has_regime_opportunity:
                report.routing = SearchRouting.EXPERIMENT
                report.discard_reason = ""
                report.exploration_summary += (
                    "\nRouting override: DISCARD→EXPERIMENT "
                    "(regime-conditional improvement detected)"
                )
                logger.info(
                    "Regime override DISCARD→EXPERIMENT for %s: sensitivity=%.2f",
                    suggestion_id, regime_result.regime_sensitivity,
                )

        return report

    def _build_grid(
        self,
        param: ParameterDefinition,
        proposed_value: Any,
    ) -> list[Any]:
        """Build candidate grid from parameter definition."""
        if param.valid_values is not None:
            # Categorical: test all values
            grid = list(param.valid_values)
            if proposed_value not in grid:
                grid.append(proposed_value)
            if param.current_value not in grid:
                grid.append(param.current_value)
            return grid

        if param.valid_range is not None:
            lo, hi = param.valid_range
        else:
            # No range defined — explore ±30% around proposed
            try:
                pval = float(proposed_value)
            except (TypeError, ValueError):
                return [proposed_value, param.current_value]
            lo = pval * 0.7
            hi = pval * 1.3

        try:
            pval = float(proposed_value)
        except (TypeError, ValueError):
            return [proposed_value, param.current_value]

        # Clip exploration window to ±30% of proposed, within valid range
        explore_lo = max(lo, pval * 0.7)
        explore_hi = min(hi, pval * 1.3)

        n_steps = _MAX_CANDIDATES
        if explore_hi <= explore_lo:
            grid_values = [explore_lo]
        else:
            step = (explore_hi - explore_lo) / (n_steps - 1)
            grid_values = [explore_lo + i * step for i in range(n_steps)]

        # Always include proposed and current
        grid_values.append(pval)
        if param.current_value is not None:
            try:
                grid_values.append(float(param.current_value))
            except (TypeError, ValueError):
                pass

        # Integer rounding + dedup
        is_int = param.value_type == "int"
        if is_int:
            grid_values = [round(v) for v in grid_values]

        # Clip to valid_range
        if param.valid_range is not None:
            grid_values = [
                max(lo, min(hi, v)) for v in grid_values
            ]

        # Deduplicate while preserving order
        seen: set[float] = set()
        unique: list[Any] = []
        for v in grid_values:
            key = round(v, 10) if not is_int else v
            if key not in seen:
                seen.add(key)
                unique.append(int(v) if is_int else v)

        return unique

    def _evaluate_candidate(
        self,
        param: ParameterDefinition,
        value: Any,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        baseline_metrics: SimulationMetrics,
    ) -> CandidateResult:
        """Evaluate a single candidate value: backtest + robustness + cost sensitivity."""
        # Backtest at candidate value
        metrics = self._simulator.simulate(
            trades, missed, {param.param_name: value},
        )

        # Robustness: single-param space
        pdef = ParameterDef(
            name=param.param_name,
            min_value=param.valid_range[0] if param.valid_range else float(value) * 0.5,
            max_value=param.valid_range[1] if param.valid_range else float(value) * 1.5,
            step=abs(float(value)) * 0.1 if float(value) != 0 else 0.1,
            current_value=float(value),
        )
        space = ParameterSpace(bot_id=param.bot_id, parameters=[pdef])
        robustness_config = RobustnessConfig(
            neighborhood_pct=0.1,
            min_profitable_regimes=2,
            total_regime_types=4,
        )
        tester = RobustnessTester(robustness_config, space, self._simulator)
        robustness = tester.evaluate(trades, missed, {param.param_name: float(value)})

        # Cost sensitivity: simulate at 1.5x cost multiplier
        cost_metrics = self._simulator.simulate(
            trades, missed, {param.param_name: value},
            cost_multiplier=_COST_SENSITIVITY_MULT,
        )
        cost_sharpe = cost_metrics.sharpe_ratio

        # Safety check
        safety_notes: list[str] = []
        passes = True

        if not robustness.neighborhood_stable:
            safety_notes.append("Neighborhood unstable")
        if cost_sharpe < 0:
            safety_notes.append(f"Negative Sharpe at {_COST_SENSITIVITY_MULT}x cost: {cost_sharpe:.2f}")
            passes = False
        if metrics.sharpe_ratio < 0:
            safety_notes.append(f"Negative Sharpe: {metrics.sharpe_ratio:.2f}")
            passes = False
        if metrics.total_trades < 5:
            safety_notes.append(f"Too few trades: {metrics.total_trades}")
            passes = False

        composite = self._composite_score(metrics, baseline_metrics)

        return CandidateResult(
            value=value,
            metrics=metrics,
            robustness_score=robustness.robustness_score,
            neighborhood_stable=robustness.neighborhood_stable,
            passes_safety=passes,
            safety_notes=safety_notes,
            composite_score=composite,
            cost_sensitivity_sharpe=cost_sharpe,
        )

    @staticmethod
    def _composite_score(
        metrics: SimulationMetrics,
        baseline: SimulationMetrics,
    ) -> float:
        """Ratio-based helper composite for candidate ranking (baseline = 1.0).

        Uses shared weights from schemas.objective_weights (excl. process_quality).
        Output is a ratio, NOT a [0, 1] score — see objective_weights.py for the
        intentional divergence between this, GroundTruthComputer's z-score scale,
        and the monthly immutable score profiles.
        Use ratio_to_unit_scale() for [0, 1] normalization when needed.
        """
        # Baseline guards — avoid division by zero and perverse sign flips
        b_er = baseline.net_pnl if baseline.net_pnl > 0 else 1.0
        b_calmar = baseline.calmar_ratio if baseline.calmar_ratio != 0 else 1.0
        b_pf = baseline.profit_factor if baseline.profit_factor != 0 else 1.0
        b_exp = baseline.expectancy if baseline.expectancy != 0 else 1.0
        b_dd = abs(baseline.max_drawdown_pct) if baseline.max_drawdown_pct != 0 else 1.0

        # net_pnl sign guard: if baseline is non-positive, treat ratio as neutral
        er_imp = metrics.net_pnl / b_er if baseline.net_pnl > 0 else 1.0
        calmar_imp = metrics.calmar_ratio / b_calmar
        pf_imp = metrics.profit_factor / b_pf
        exp_imp = metrics.expectancy / b_exp if baseline.expectancy != 0 else 1.0
        dd_inc = abs(metrics.max_drawdown_pct) / b_dd - 1.0

        return (
            W_EXPECTED_R_NO_PROCESS * er_imp
            + W_CALMAR_NO_PROCESS * calmar_imp
            + W_PROFIT_FACTOR_NO_PROCESS * pf_imp
            + W_EXPECTANCY_NO_PROCESS * exp_imp
            - W_INV_DRAWDOWN_NO_PROCESS * max(0, dd_inc)
        )

    @staticmethod
    def _route(
        best: CandidateResult | None,
        baseline_composite: float,
        is_safety_critical: bool,
    ) -> tuple[SearchRouting, str]:
        """Route based on best candidate vs thresholds."""
        if best is None:
            return SearchRouting.DISCARD, "No candidates passed safety checks"

        # Floor baseline at a small positive value so the ratio remains
        # comparable to the positive-baseline branch. Without this, a
        # non-positive baseline produced raw scores (~0.4) compared against
        # ratio thresholds (~1.05), DISCARDing candidates that clearly
        # improved on a losing baseline.
        effective_baseline = max(baseline_composite, 0.01)
        improvement = best.composite_score / effective_baseline

        threshold = (
            _SAFETY_CRITICAL_IMPROVEMENT if is_safety_critical
            else _APPROVE_IMPROVEMENT
        )

        if improvement >= threshold and best.robustness_score >= _APPROVE_ROBUSTNESS:
            return SearchRouting.APPROVE, ""

        if improvement >= _EXPERIMENT_IMPROVEMENT and best.robustness_score >= _EXPERIMENT_ROBUSTNESS:
            return SearchRouting.EXPERIMENT, ""

        reasons: list[str] = []
        if improvement < _EXPERIMENT_IMPROVEMENT:
            reasons.append(f"improvement {improvement:.2f} below {_EXPERIMENT_IMPROVEMENT}")
        if best.robustness_score < _EXPERIMENT_ROBUSTNESS:
            reasons.append(f"robustness {best.robustness_score:.0f} below {_EXPERIMENT_ROBUSTNESS}")
        return SearchRouting.DISCARD, "; ".join(reasons)

    @staticmethod
    def _build_summary(
        param: ParameterDefinition,
        proposed_value: Any,
        candidates: list[Any],
        results: list[CandidateResult],
        best: CandidateResult | None,
        routing: SearchRouting,
    ) -> str:
        """Human-readable exploration summary for Telegram card."""
        passing = sum(1 for r in results if r.passes_safety)
        lines = [
            f"Searched {len(candidates)} values for {param.param_name}",
            f"{passing}/{len(candidates)} passed safety checks",
        ]
        if best:
            lines.append(
                f"Best: {best.value} (composite={best.composite_score:.3f}, "
                f"robustness={best.robustness_score:.0f})"
            )
            if best.value != proposed_value:
                lines.append(f"Original proposed: {proposed_value} → system found better: {best.value}")
        lines.append(f"Routing: {routing.value}")
        return "\n".join(lines)
