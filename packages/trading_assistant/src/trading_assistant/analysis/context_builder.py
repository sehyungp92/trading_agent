# analysis/context_builder.py
"""Generic context builder for shared policy and corrections loading."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_assistant.analysis.context_sources.common import (
    apply_temporal_window as _apply_temporal_window,
    safe_jsonl as _safe_jsonl,
)
from trading_assistant.analysis.context_sources.learning_context import LearningContextMixin
from trading_assistant.analysis.context_sources.market_context import MarketContextMixin
from trading_assistant.analysis.context_sources.session_recall_context import (
    SessionRecallContextMixin,
)
from trading_assistant.analysis.context_sources.source_facade_context import (
    SourceFacadeContextMixin,
)
from trading_assistant.schemas.memory import MemoryIndex
from trading_assistant.schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

__all__ = ["ContextBuilder", "_apply_temporal_window", "_safe_jsonl"]

_POLICY_FILES = ["agent.md", "trading_rules.md", "soul.md"]
_FINDINGS_MAX_AGE_DAYS = 90
_FINDINGS_MAX_ENTRIES = 50




def _merge_retrieval_profile(base: dict, override: dict) -> dict:
    """Merge a caller-supplied retrieval profile without dropping base context."""
    if not override:
        return base
    merged = dict(base)
    for key in ("tags", "query_terms"):
        values: list[str] = []
        seen: set[str] = set()
        for source in (base.get(key, []), override.get(key, [])):
            for raw in source or []:
                value = str(raw or "").strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        if values:
            merged[key] = values
    for key, value in override.items():
        if key in {"tags", "query_terms"}:
            continue
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged












_DEFAULT_CONTEXT_BUDGET_ITEMS = 15
_EXPANDED_CONTEXT_BUDGET_ITEMS = 25


def _estimate_tokens(value: object) -> int:
    """Estimate token count for a data value.

    Uses a ~4 chars/token heuristic for JSON-serialized data, which is
    conservative for structured data (actual ratio is closer to 3.5 for
    English prose, 4-5 for JSON with keys).
    """
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return max(1, len(text) // 4)


class ContextBuilder(
    SourceFacadeContextMixin,
    SessionRecallContextMixin,
    LearningContextMixin,
    MarketContextMixin,
):
    """Loads shared context (policies, corrections, metadata) used by all assemblers."""

    def __init__(
        self,
        memory_dir: Path,
        curated_dir: Path | None = None,
        run_index: object | None = None,
        evidence_memory: Any | None = None,
    ) -> None:
        self._memory_dir = memory_dir
        self._curated_dir = curated_dir
        self._run_index = run_index
        self._provided_evidence_memory = evidence_memory
        self._strategy_registry = None  # lazy-loaded on first use

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def _get_strategy_registry(self):
        """Lazy-load the StrategyRegistry from data/strategy_profiles.yaml.

        Returns an empty registry on any error so loaders never break the
        analysis pipeline.
        """
        if self._strategy_registry is not None:
            return self._strategy_registry
        try:
            from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry
            self._strategy_registry = load_strategy_registry()
        except Exception:
            from trading_assistant.schemas.strategy_profile import StrategyRegistry
            self._strategy_registry = StrategyRegistry()
        return self._strategy_registry

    def _evidence_memory(self):
        if self._provided_evidence_memory is not None:
            return self._provided_evidence_memory
        from trading_assistant.analysis.evidence_memory import EvidenceMemory

        return EvidenceMemory(
            self._memory_dir,
            run_index=self._run_index,
            strategy_registry=self._get_strategy_registry(),
        )

    def build_system_prompt(self) -> str:
        """Load policy files from memory/policies/v1/ into a system prompt."""
        return self._evidence_memory().policies.build_system_prompt()




    _QUALITY_RANK = {"high": 3, "medium": 2, "low": 1, "insufficient": 0}





    def list_policy_files(self) -> list[str]:
        """List paths to included policy files (for context_files tracking)."""
        return self._evidence_memory().policies.context_files()

    def runtime_metadata(self, bot_configs: dict | None = None) -> dict:
        """Return runtime metadata for the prompt package.

        Args:
            bot_configs: Optional dict of ``{bot_id: BotConfig}`` to include
                per-bot timezone information in the metadata.
        """
        now = datetime.now(timezone.utc)
        meta: dict = {
            "assembled_at": now.isoformat(),
            "timezone": "UTC",
        }
        if bot_configs:
            meta["bot_timezones"] = {
                bid: cfg.timezone if hasattr(cfg, "timezone") else "UTC"
                for bid, cfg in bot_configs.items()
            }
        return meta

    @staticmethod
    def check_data_availability(
        index: MemoryIndex | None, bot_id: str, date: str,
    ) -> dict:
        """Check if curated data exists for a bot on a given date.

        Returns dict with: has_curated (bool), available_dates (list[str]).
        If index is None, returns unknown state.
        """
        if index is None:
            return {"has_curated": None, "available_dates": []}

        bot_dates = index.curated_dates_by_bot.get(bot_id, [])
        return {
            "has_curated": date in bot_dates,
            "available_dates": bot_dates,
        }
























































    # Priority order for context items (highest value first).
    # Items not in this list get lowest priority.
    _CONTEXT_PRIORITY: list[str] = [
        # Core context — always include when available
        "loop_contract",
        "recent_work_log",
        "ground_truth_trend",
        "portfolio_outcomes",
        "portfolio_rolling_metrics",
        "macro_regime_context",
        "self_assessment",
        "convergence_report",
        "strategy_profiles",
        "archetype_expectations",
        "coordination_rules",
        "portfolio_risk_config",
        "last_week_synthesis",
        # Engine-level decomposition and ablation analysis
        "engine_decomposition",
        "ablation_analysis",
        "exit_tier_analysis",
        # Crypto perpetual analysis
        "funding_analysis",
        "grade_analysis",
        "confluence_analysis",
        "leverage_analysis",
        # Learning signals — high value for improvement
        "active_suggestions",
        "rejected_suggestions",
        "recent_proposal_outcomes",
        "monthly_outcomes",
        "outcome_priors",
        "performance_learning",
        "strategy_change_history",
        "focused_run_recall",
        "category_scorecard",
        "regime_stratified_scores",
        "prediction_accuracy_by_metric",
        "outcome_measurements",
        "forecast_meta_analysis",
        "correction_patterns",
        "validation_patterns",
        "active_experiments",
        "backtest_reliability",
        "regime_config_history",
        "transfer_track_record",
        "cycle_effectiveness_trend",
        "suggestion_quality_trend",
        "optimization_allocation",
        "search_signal_summary",
        "search_reports",
        "regime_parameter_analysis",
        "hypothesis_track_record",
        "discoveries",
        "strategy_ideas",
        # Lower-priority learning context
        "outcome_reasonings",
        "recalibrations",
        "threshold_profile",
        "experiment_track_record",
        "consolidated_patterns",
        "spurious_outcomes",
        "pattern_library",
        "failure_log",
        "reliability_summary",
        "instrumentation_readiness",
        "allocation_history",
        "session_history",
    ]

    # Workflow-specific priority overrides.  Keys not listed here fall back
    # to _CONTEXT_PRIORITY.  Each list is ordered highest-to-lowest priority.
    _WORKFLOW_PRIORITIES: dict[str, list[str]] = {
        "weekly_analysis": [
            # Trend and meta-analysis dominate weekly review
            "loop_contract",
            "recent_work_log",
            "ground_truth_trend",
            "convergence_report",
            "self_assessment",
            "last_week_synthesis",
            "outcome_measurements",
            "monthly_outcomes",
            "outcome_priors",
            "performance_learning",
            "recent_proposal_outcomes",
            "strategy_change_history",
            "focused_run_recall",
            "forecast_meta_analysis",
            "category_scorecard",
            "regime_stratified_scores",
            "suggestion_quality_trend",
            "cycle_effectiveness_trend",
            "hypothesis_track_record",
            "transfer_track_record",
            "prediction_accuracy_by_metric",
            "active_suggestions",
            "rejected_suggestions",
            "correction_patterns",
            "validation_patterns",
            "portfolio_outcomes",
            "portfolio_rolling_metrics",
            "macro_regime_context",
            "strategy_profiles",
            "archetype_expectations",
            "coordination_rules",
            "portfolio_risk_config",
            "engine_decomposition",
            "ablation_analysis",
            "exit_tier_analysis",
            "funding_analysis",
            "grade_analysis",
            "confluence_analysis",
            "leverage_analysis",
            "regime_parameter_analysis",
            "outcome_reasonings",
            "recalibrations",
            "discoveries",
            "strategy_ideas",
            "active_experiments",
            "experiment_track_record",
            "backtest_reliability",
            "search_reports",
            "optimization_allocation",
            "search_signal_summary",
            "consolidated_patterns",
            "pattern_library",
            "spurious_outcomes",
            "regime_config_history",
            "threshold_profile",
            "failure_log",
            "reliability_summary",
            "instrumentation_readiness",
            "allocation_history",
            "session_history",
        ],
        "discovery_analysis": [
            # Discovery focuses on novel patterns and gaps in coverage
            "loop_contract",
            "recent_work_log",
            "discoveries",
            "strategy_ideas",
            "pattern_library",
            "ground_truth_trend",
            "outcome_reasonings",
            "consolidated_patterns",
            "hypothesis_track_record",
            "correction_patterns",
            "convergence_report",
            "strategy_profiles",
            "archetype_expectations",
            "macro_regime_context",
            "self_assessment",
        ],
        "outcome_reasoning": [
            # Reasoning about why suggestions worked/failed
            "loop_contract",
            "recent_work_log",
            "outcome_measurements",
            "monthly_outcomes",
            "outcome_priors",
            "performance_learning",
            "recent_proposal_outcomes",
            "strategy_change_history",
            "focused_run_recall",
            "category_scorecard",
            "regime_stratified_scores",
            "active_suggestions",
            "ground_truth_trend",
            "correction_patterns",
            "hypothesis_track_record",
            "transfer_track_record",
            "forecast_meta_analysis",
            "prediction_accuracy_by_metric",
            "self_assessment",
            "convergence_report",
            "strategy_profiles",
        ],
        "triage": [
            # Bug triage needs minimal learning context
            "loop_contract",
            "recent_work_log",
            "ground_truth_trend",
            "convergence_report",
            "self_assessment",
            "strategy_profiles",
            "session_history",
        ],
    }

    def base_package(
        self,
        session_store=None,
        agent_type: str = "",
        bot_configs: dict | None = None,
        context_budget_items: int = _DEFAULT_CONTEXT_BUDGET_ITEMS,
        context_budget_tokens: int = 0,
        strategy_registry=None,
        bot_id: str = "",
        record_retrieval: bool = True,
        retrieval_profile_override: dict | None = None,
    ) -> PromptPackage:
        """Build a PromptPackage pre-filled with system prompt, corrections, and metadata.

        Args:
            session_store: Optional SessionStore for loading session history.
            agent_type: Agent type for session history filtering.
            bot_configs: Optional ``{bot_id: BotConfig}`` for timezone metadata.
            context_budget_items: Max items when token budget is not set.
            context_budget_tokens: When >0, use token-aware budgeting instead
                of item count.  Items are added in priority order until the
                budget is exhausted.
        """
        retrieval_profile = _merge_retrieval_profile(
            self.build_retrieval_profile(agent_type=agent_type, bot_id=bot_id),
            retrieval_profile_override or {},
        )
        loop_contract = self.load_loop_contract_context(agent_type=agent_type)
        recent_work_log = self.load_recent_work_log_entries(
            agent_type=agent_type,
            bot_id=bot_id,
            limit=10,
        )
        failure_log = self.load_failure_log()
        rejected_suggestions = self.load_rejected_suggestions()
        outcome_measurements, low_quality_outcomes = self.load_outcome_measurements()
        allocation_history = self.load_allocation_history()
        consolidated_patterns = self.load_consolidated_patterns()
        data: dict = {}
        if loop_contract:
            data["loop_contract"] = loop_contract
        if recent_work_log:
            data["recent_work_log"] = recent_work_log
        if failure_log:
            data["failure_log"] = failure_log
        if rejected_suggestions:
            data["rejected_suggestions"] = rejected_suggestions
        if outcome_measurements:
            data["outcome_measurements"] = outcome_measurements
        monthly_outcomes = self.load_monthly_outcomes(bot_id=bot_id)
        if monthly_outcomes:
            data["monthly_outcomes"] = monthly_outcomes
        outcome_priors = self.load_outcome_priors(bot_id=bot_id)
        if outcome_priors:
            data["outcome_priors"] = outcome_priors
        performance_learning = self.load_recent_performance_learning_entries(
            bot_id=bot_id,
            limit=10,
        )
        if performance_learning:
            data["performance_learning"] = performance_learning
        if allocation_history:
            data["allocation_history"] = allocation_history
        if consolidated_patterns:
            data["consolidated_patterns"] = consolidated_patterns
        pattern_library = self.load_pattern_library()
        if pattern_library:
            data["pattern_library"] = pattern_library
        correction_patterns = self.load_correction_patterns()
        if correction_patterns:
            data["correction_patterns"] = correction_patterns
        forecast_meta = self.load_forecast_meta()
        if forecast_meta:
            data["forecast_meta_analysis"] = forecast_meta
        active_suggestions = self.load_active_suggestions()
        if active_suggestions:
            data["active_suggestions"] = active_suggestions
        recent_proposal_outcomes = self.load_recent_proposal_outcomes(bot_id=bot_id)
        if recent_proposal_outcomes:
            data["recent_proposal_outcomes"] = recent_proposal_outcomes
        strategy_change_history = self.load_strategy_change_ledger(bot_id=bot_id)
        if strategy_change_history:
            data["strategy_change_history"] = strategy_change_history
        category_scorecard = self.load_category_scorecard()
        if category_scorecard:
            data["category_scorecard"] = category_scorecard
        regime_stratified_scores = self.load_regime_stratified_scores()
        if regime_stratified_scores:
            data["regime_stratified_scores"] = regime_stratified_scores
        prediction_accuracy = self.load_prediction_accuracy()
        if prediction_accuracy:
            data["prediction_accuracy_by_metric"] = prediction_accuracy
        hypothesis_track_record = self.load_hypothesis_track_record()
        if hypothesis_track_record:
            data["hypothesis_track_record"] = hypothesis_track_record
        transfer_track_record = self.load_transfer_track_record()
        if transfer_track_record:
            data["transfer_track_record"] = transfer_track_record
        validation_patterns = self.load_validation_patterns()
        if validation_patterns:
            data["validation_patterns"] = validation_patterns
        threshold_profile = self.load_threshold_profile()
        if threshold_profile:
            data["threshold_profile"] = threshold_profile
        reliability_summary = self.load_reliability_summary()
        if reliability_summary:
            data["reliability_summary"] = reliability_summary
        experiment_track_record = self.load_experiment_track_record()
        if experiment_track_record:
            data["experiment_track_record"] = experiment_track_record
        active_experiments = self.load_active_experiments()
        if active_experiments:
            data["active_experiments"] = active_experiments
        outcome_reasonings = self.load_outcome_reasonings()
        if outcome_reasonings:
            data["outcome_reasonings"] = outcome_reasonings
        recalibrations = self.load_recalibrations()
        if recalibrations:
            data["recalibrations"] = recalibrations
        discoveries = self.load_discoveries()
        if discoveries:
            data["discoveries"] = discoveries
        optimization_allocation = self.load_optimization_allocation()
        if optimization_allocation:
            data["optimization_allocation"] = optimization_allocation
        search_signal_summary = self.load_search_signal_summary()
        if search_signal_summary:
            data["search_signal_summary"] = search_signal_summary
        ground_truth_trend = self.load_ground_truth_trend()
        if ground_truth_trend:
            data["ground_truth_trend"] = ground_truth_trend
        self_assessment = self.build_self_assessment(
            forecast_meta=forecast_meta,
            category_scorecard=category_scorecard,
            correction_patterns=correction_patterns,
            recalibrations=recalibrations,
        )
        if self_assessment:
            data["self_assessment"] = self_assessment
        convergence_report = self.load_convergence_report()
        if convergence_report:
            data["convergence_report"] = convergence_report
        if bot_configs:
            instrumentation = self.load_instrumentation_readiness(
                list(bot_configs.keys()),
            )
            if instrumentation:
                data["instrumentation_readiness"] = instrumentation
        cycle_effectiveness = self.load_cycle_effectiveness()
        if cycle_effectiveness:
            data["cycle_effectiveness_trend"] = cycle_effectiveness
        suggestion_quality_trend = self.load_suggestion_quality_trend(
            value_map=optimization_allocation,
        )
        if suggestion_quality_trend:
            data["suggestion_quality_trend"] = suggestion_quality_trend
        retrospective_synthesis = self.load_retrospective_synthesis()
        if retrospective_synthesis:
            data["last_week_synthesis"] = retrospective_synthesis
        spurious_outcomes = self.load_spurious_outcomes()
        # Merge low-quality outcome measurements into spurious_outcomes
        all_spurious = spurious_outcomes + low_quality_outcomes
        if all_spurious:
            data["spurious_outcomes"] = all_spurious
        strategy_ideas = self.load_strategy_ideas()
        if strategy_ideas:
            data["strategy_ideas"] = strategy_ideas
        search_reports = self.load_search_reports(bot_id=bot_id)
        if search_reports:
            data["search_reports"] = search_reports
        backtest_reliability = self.load_backtest_reliability(bot_id=bot_id)
        if backtest_reliability:
            data["backtest_reliability"] = backtest_reliability
        portfolio_outcomes = self.load_portfolio_outcomes()
        if portfolio_outcomes:
            data["portfolio_outcomes"] = portfolio_outcomes
        portfolio_metrics = self.load_portfolio_metrics()
        if portfolio_metrics:
            data["portfolio_rolling_metrics"] = portfolio_metrics
        macro_regime = self.load_macro_regime_context()
        if macro_regime:
            data["macro_regime_context"] = macro_regime
        regime_config_history = self.load_regime_config_history()
        if regime_config_history:
            data["regime_config_history"] = regime_config_history
        if session_store and agent_type:
            session_history = self.load_session_history(session_store, agent_type)
            if session_history:
                data["session_history"] = session_history

        focused_recall = self.load_focused_recall(
            agent_type=agent_type,
            bot_id=bot_id,
            tags=retrieval_profile.get("tags", []),
        )
        if focused_recall:
            data["focused_run_recall"] = focused_recall

        # Inject similar past runs from RunIndex as fallback when focused
        # provenance-rich recall is unavailable.
        if self._run_index is not None and agent_type:
            if not focused_recall:
                similar_runs = self.load_similar_runs(
                    agent_type=agent_type,
                    bot_id=bot_id,
                    retrieval_profile=retrieval_profile,
                )
                if similar_runs:
                    data["similar_past_runs"] = similar_runs

        # Inject strategy registry data if available
        if strategy_registry and getattr(strategy_registry, "strategies", None):
            data["strategy_profiles"] = {
                sid: profile.model_dump(mode="json", exclude_unset=True)
                for sid, profile in strategy_registry.strategies.items()
            }
            if strategy_registry.coordination.signals or strategy_registry.coordination.cooldown_pairs:
                data["coordination_rules"] = strategy_registry.coordination.model_dump(mode="json")
            if strategy_registry.archetype_expectations:
                data["archetype_expectations"] = {
                    k: v.model_dump(mode="json")
                    for k, v in strategy_registry.archetype_expectations.items()
                }
            if strategy_registry.portfolio.heat_cap_R > 0:
                data["portfolio_risk_config"] = strategy_registry.portfolio.model_dump(mode="json")

        # Engine-level decomposition, ablation analysis, exit tier analysis
        engine_decomposition = self.load_engine_decomposition(bot_id=bot_id)
        if engine_decomposition:
            data["engine_decomposition"] = engine_decomposition
        ablation_analysis = self.load_ablation_analysis(bot_id=bot_id)
        if ablation_analysis:
            data["ablation_analysis"] = ablation_analysis
        exit_tier_analysis = self.load_exit_tier_analysis(bot_id=bot_id)
        if exit_tier_analysis:
            data["exit_tier_analysis"] = exit_tier_analysis
        regime_param_analysis = self.load_regime_parameter_analysis(bot_id=bot_id)
        if regime_param_analysis:
            data["regime_parameter_analysis"] = regime_param_analysis

        # Select workflow-aware priority list (falls back to default)
        active_priority = self._WORKFLOW_PRIORITIES.get(
            agent_type, self._CONTEXT_PRIORITY,
        )

        # Build priority-ordered key list
        priority_order = {k: i for i, k in enumerate(active_priority)}
        priority_set = set(active_priority)
        sorted_prioritized = sorted(
            (k for k in data if k in priority_set),
            key=lambda k: priority_order.get(k, 999),
        )
        unprioritized = [k for k in data if k not in priority_set]
        all_keys_ordered = sorted_prioritized + unprioritized

        total_available = len(data)
        omitted_keys: list[str] = []
        token_estimates: dict[str, int] = {}

        if context_budget_tokens > 0:
            # Token-aware budgeting: add items in priority order, skipping
            # items that don't fit but continuing to try smaller ones.
            budget_keys: list[str] = []
            tokens_used = 0
            for key in all_keys_ordered:
                est = _estimate_tokens(data[key])
                token_estimates[key] = est
                if tokens_used + est <= context_budget_tokens:
                    budget_keys.append(key)
                    tokens_used += est
                else:
                    omitted_keys.append(key)
                    continue  # skip this item but keep trying smaller ones
            if omitted_keys:
                logger.warning(
                    "Token budget (%s): %d/%d tokens used, dropped %d items: %s",
                    agent_type or "default", tokens_used, context_budget_tokens,
                    len(omitted_keys), omitted_keys,
                )
            data = {k: data[k] for k in budget_keys if k in data}
        else:
            # Item-count budgeting (legacy) — adaptive expansion
            if context_budget_items == _DEFAULT_CONTEXT_BUDGET_ITEMS:
                effective_budget = max(context_budget_items, min(total_available, _EXPANDED_CONTEXT_BUDGET_ITEMS))
            else:
                effective_budget = context_budget_items

            if total_available > effective_budget:
                budget_keys = all_keys_ordered[:effective_budget]
                dropped = set(data.keys()) - set(budget_keys)
                omitted_keys = sorted(dropped)
                if dropped:
                    logger.warning(
                        "Context budget (%s): dropped %d low-priority items: %s",
                        agent_type or "default", len(dropped), omitted_keys,
                    )
                data = {k: data[k] for k in budget_keys if k in data}

            # Compute token estimates for manifest (informational)
            for key in data:
                token_estimates[key] = _estimate_tokens(data[key])

        metadata = self.runtime_metadata(bot_configs=bot_configs)
        metadata["_context_budget_manifest"] = {
            "workflow": agent_type or "default",
            "budget_mode": "tokens" if context_budget_tokens > 0 else "items",
            "included": sorted(data.keys()),
            "omitted": omitted_keys,
            "total_available": total_available,
            "token_estimates": token_estimates,
        }

        # Load ranked learning cards (if card store exists)
        learning_cards_text = ""
        try:
            from trading_assistant.skills.learning_card_store import LearningCardStore
            card_store = LearningCardStore(self._memory_dir / "findings")
            ranked = card_store.ranked_for_prompt(
                limit=10,
                bot_id=bot_id,
                workflow=agent_type,
                tags=retrieval_profile.get("tags", []),
            )
            if ranked:
                if record_retrieval:
                    card_store.load().record_retrieval([c.card_id for c in ranked])
                    card_store.save()
                learning_cards_text = "\n\n".join(c.to_prompt_text() for c in ranked)
                metadata["_learning_card_ids"] = [c.card_id for c in ranked]
        except Exception:
            logger.debug("Learning card loading skipped (store not available)")

        playbooks_text = ""
        playbooks = self.load_generated_playbooks(
            workflow=agent_type,
            tags=retrieval_profile.get("tags", []),
        )
        if playbooks:
            playbooks_text = "\n\n".join(playbook["text"] for playbook in playbooks)
            playbook_ids = [playbook["playbook_id"] for playbook in playbooks]
            metadata["_generated_playbook_ids"] = playbook_ids
            try:
                from trading_assistant.skills.playbook_generator import PlaybookGenerator

                if record_retrieval:
                    tracker = PlaybookGenerator(self._memory_dir)
                    for playbook_id in playbook_ids:
                        tracker.record_usage(playbook_id)
            except Exception:
                logger.debug("Generated playbook usage tracking skipped")

        return PromptPackage(
            system_prompt=self.build_system_prompt(),
            corrections=self.load_corrections(),
            context_files=self.list_policy_files(),
            metadata={
                **metadata,
                "_learning_cards_text": learning_cards_text,
                "_generated_playbooks_text": playbooks_text,
                "_retrieval_profile": retrieval_profile,
            },
            data=data,
        )
