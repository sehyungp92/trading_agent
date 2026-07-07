"""Weekly simulation support."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WeeklySimulationSupport:
    """Weekly simulation support."""

    def _run_weekly_simulations(
        self,
        refinement_report: object,
        week_start: str,
        week_end: str,
    ) -> dict:
        """Run simulation skills for all bots unconditionally.

        Runs FilterSensitivityAnalyzer, CounterfactualSimulator, and
        ExitStrategySimulator for every bot that has data, regardless of
        whether the strategy engine flagged specific issues. This ensures
        healthy bots still get proactive "what if" analysis.

        Strategy engine suggestions are used to enrich context (e.g. extracting
        a specific regime for counterfactual analysis) but do NOT gate whether
        simulations run.
        """
        results: dict = {}

        try:
            from trading_assistant.skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer
            from trading_assistant.skills.counterfactual_simulator import CounterfactualSimulator
            from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator
            from trading_assistant.schemas.exit_simulation import ExitSweepResult

            suggestions = getattr(refinement_report, "suggestions", [])
            counterfactual = CounterfactualSimulator()
            exit_sim = ExitStrategySimulator()

            # Extract regime hints from suggestions (used to enrich counterfactual, not to gate)
            regime_hints: dict[str, str] = {}
            for suggestion in suggestions:
                bot_id = getattr(suggestion, "bot_id", None) or ""
                title = getattr(suggestion, "title", "") or ""
                if "regime" in title.lower() and bot_id:
                    regime_hints[bot_id] = getattr(suggestion, "regime", None) or "ranging"

            # Run all simulations for every known bot
            for bot_id in self._bots:
                trades, missed = self._load_trades_for_week(bot_id, week_start, week_end)

                # FilterSensitivity - needs missed opportunities
                if missed:
                    try:
                        analyzer = FilterSensitivityAnalyzer(bot_id=bot_id, date=week_start)
                        report = analyzer.analyze(missed)
                        results[f"filter_sensitivity_{bot_id}"] = report.model_dump(mode="json")
                    except Exception:
                        logger.warning("FilterSensitivity failed for %s", bot_id)

                # Counterfactual - needs trades or missed
                if trades or missed:
                    try:
                        regime = regime_hints.get(bot_id, "ranging")
                        sim_result = counterfactual.simulate_regime_gate(trades, missed, regime)
                        results[f"counterfactual_{bot_id}"] = sim_result.model_dump(mode="json")
                    except Exception:
                        logger.warning("Counterfactual failed for %s", bot_id)

                # ExitStrategy sweep - test all 12 default configs
                if trades:
                    try:
                        sweep_results = exit_sim.sweep(trades)
                        best = max(sweep_results, key=lambda r: r.improvement)
                        sweep_out = ExitSweepResult(
                            bot_id=bot_id,
                            configs_tested=len(sweep_results),
                            baseline_pnl=best.baseline_pnl,
                            results=sweep_results,
                            best_strategy=best.strategy,
                            best_improvement=best.improvement,
                        )
                        results[f"exit_sweep_{bot_id}"] = sweep_out.model_dump(mode="json")
                    except Exception:
                        logger.warning("ExitStrategy sweep failed for %s", bot_id)

                # FilterInteraction - analyze filter pair co-activation patterns
                if trades or missed:
                    try:
                        from trading_assistant.skills.filter_interaction_analyzer import FilterInteractionAnalyzer

                        fi_analyzer = FilterInteractionAnalyzer(bot_id=bot_id, date=week_start)
                        fi_report = fi_analyzer.analyze(trades, missed)
                        if fi_report.pairs:
                            results[f"filter_interaction_{bot_id}"] = fi_report.model_dump(mode="json")
                    except Exception:
                        logger.warning("FilterInteraction failed for %s", bot_id)

        except Exception:
            logger.warning("Simulation skills import failed - skipping simulations")

        return results
