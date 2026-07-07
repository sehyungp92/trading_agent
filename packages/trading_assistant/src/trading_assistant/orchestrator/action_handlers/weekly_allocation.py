"""Weekly allocation analysis support."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WeeklyAllocationSupport:
    """Weekly allocation analysis support."""

    def _run_allocation_analyses(
        self,
        portfolio_summary: object,
        week_start: str,
        week_end: str,
    ) -> dict:
        """Run portfolio allocation, synergy, and proportion optimization analyses."""
        results: dict = {}

        try:
            from trading_assistant.skills.portfolio_allocator import PortfolioAllocator
            from trading_assistant.skills.synergy_analyzer import SynergyAnalyzer
            from trading_assistant.skills.strategy_proportion_optimizer import StrategyProportionOptimizer

            bot_summaries = getattr(portfolio_summary, "bot_summaries", {})
            if not bot_summaries:
                return results

            # 1. Synergy analysis first (we need the correlation matrix for allocation)
            per_strat = {
                bid: s.per_strategy_summary
                for bid, s in bot_summaries.items()
            }
            synergy = SynergyAnalyzer(week_start, week_end)
            synergy_report = synergy.compute(per_strat)
            results["synergy_analysis"] = synergy_report.model_dump(mode="json")

            # Also compute intra-bot synergy for each bot with multiple strategies
            intra_bot_results: dict = {}
            for bid, strats in per_strat.items():
                if len(strats) >= 2:
                    intra_report = synergy.compute_intra_bot(bid, strats)
                    intra_bot_results[bid] = intra_report.model_dump(mode="json")
            if intra_bot_results:
                results["intra_bot_synergy"] = intra_bot_results

            # 2. Portfolio allocation (cross-bot) with correlation matrix from synergy
            from trading_assistant.skills.allocation_tracker import AllocationTracker
            from trading_assistant.skills.drift_analyzer import DriftAnalyzer

            allocator = PortfolioAllocator(week_start, week_end)
            n_bots = len(bot_summaries)
            tracker = AllocationTracker(self._memory_dir / "findings")
            latest_actuals = tracker.get_latest_actuals()
            if latest_actuals:
                default_pct = 100.0 / n_bots if n_bots > 0 else 0.0
                current = {bid: latest_actuals.get(bid, default_pct) for bid in bot_summaries}
                total = sum(current.values())
                if total > 0 and abs(total - 100.0) > 0.1:
                    current = {bid: pct / total * 100.0 for bid, pct in current.items()}
            else:
                current = {bid: 100.0 / n_bots for bid in bot_summaries} if n_bots > 0 else {}
            bot_correlation = synergy.compute_bot_correlation_matrix(per_strat)
            alloc_report = allocator.compute(bot_summaries, current, correlation_matrix=bot_correlation)
            results["portfolio_allocation"] = alloc_report.model_dump(mode="json")

            # Record allocation snapshot and compute drift trend
            snapshot = DriftAnalyzer.compute_snapshot(alloc_report, current)
            tracker.record_snapshot(snapshot)
            all_snapshots = tracker.load_snapshots()
            drift_trend = DriftAnalyzer.compute_drift_trend(all_snapshots)
            results["allocation_drift"] = {
                "current_snapshot": snapshot.model_dump(mode="json"),
                "trend": drift_trend,
            }

            # 3. Proportion optimization (intra-bot)
            optimizer = StrategyProportionOptimizer(week_start, week_end)
            proportion_report = optimizer.compute(per_strat)
            results["proportion_optimization"] = proportion_report.model_dump(mode="json")

            # 4. Structural analysis
            from trading_assistant.skills.structural_analyzer import StructuralAnalyzer

            structural = StructuralAnalyzer(week_start, week_end)
            structural_report = structural.compute(per_strat)
            results["structural_analysis"] = structural_report.model_dump(mode="json")

            # Write to weekly curated dir for prompt assembler
            weekly_dir = self._curated_dir / "weekly" / week_start
            weekly_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            (weekly_dir / "structural_analysis.json").write_text(
                _json.dumps(results["structural_analysis"], indent=2, default=str),
                encoding="utf-8",
            )

            if "allocation_drift" in results:
                (weekly_dir / "allocation_drift.json").write_text(
                    _json.dumps(results["allocation_drift"], indent=2, default=str),
                    encoding="utf-8",
                )

            # 5. Regime-conditional metrics
            from trading_assistant.analysis.strategy_engine import StrategyEngine as _SE

            engine = _SE(
                week_start=week_start, week_end=week_end,
                threshold_learner=self._threshold_learner,
            )
            trades_by_bot: dict[str, list] = {}
            for bid in bot_summaries:
                trades, _ = self._load_trades_for_week(bid, week_start, week_end)
                trades_by_bot[bid] = trades

            regime_report = engine.compute_regime_conditional_metrics(per_strat, trades_by_bot)
            results["regime_conditional_analysis"] = regime_report.model_dump(mode="json")
            (weekly_dir / "regime_conditional_analysis.json").write_text(
                _json.dumps(results["regime_conditional_analysis"], indent=2, default=str),
                encoding="utf-8",
            )

            # 6. Interaction analysis (swing_multi_01 only)
            if "swing_multi_01" in bot_summaries:
                from trading_assistant.skills.interaction_analyzer import InteractionAnalyzer

                ia = InteractionAnalyzer(week_start, week_end, bot_id="swing_multi_01")
                coord_events = self._load_coordinator_events(week_start, week_end)
                swing_trades = trades_by_bot.get("swing_multi_01", [])
                interaction_report = ia.compute(coord_events, swing_trades)
                results["interaction_analysis"] = interaction_report.model_dump(mode="json")
                (weekly_dir / "interaction_analysis.json").write_text(
                    _json.dumps(results["interaction_analysis"], indent=2, default=str),
                    encoding="utf-8",
                )

        except Exception:
            logger.warning("Allocation analyses failed - skipping")

        return results
