# analysis/weekly_prompt_assembler.py
"""Weekly prompt assembler - builds context package for weekly analysis runtime invocation.

Uses deterministic weekly triage for computed summaries and focused analytical
questions. Claude reasons about retrospective accuracy and discovers novel patterns
rather than mechanically reviewing 34 checklist items.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.evidence_memory import EvidenceMemory
from trading_assistant.orchestrator.learning_sufficiency_audit import (
    manifest_runtime_support_source_authoritative,
)
from trading_assistant.schemas.prompt_package import PromptPackage
from trading_assistant.schemas.weekly_focus_rotation import (
    weekly_focus_for_week,
    weekly_focus_payload,
    weekly_focus_rotation_payload,
)

logger = logging.getLogger(__name__)

_FOCUSED_WEEKLY_INSTRUCTIONS = """\
You are analyzing a week of trading data. A deterministic triage system has
pre-computed summaries and identified what deserves your analytical attention.

## COMPUTED WEEK SUMMARY (pre-computed - do NOT regenerate)
{computed_summary}

## ANOMALIES DETECTED
{anomalies}

## RETROSPECTIVE QUESTIONS (about past decisions and predictions)
{retrospective_questions}

## DISCOVERY QUESTIONS (about novel patterns)
{discovery_questions}

## YOUR ANALYTICAL TASKS

For RETROSPECTIVE questions: review what was predicted/suggested, what actually
happened, and WHY the outcome differed from expectations. State lessons learned.

For DISCOVERY questions: look for patterns the automated detectors missed.
State a testable hypothesis, identify evidence for and against, and rate confidence.

## STRATEGY PROPOSALS
When proposing changes:
1. Each suggestion MUST quantify expected Calmar ratio impact (return change + drawdown change)
2. Evidence base required: trade count, time period, statistical significance
3. Check category_scorecard: categories with win_rate < 30% (n>=5) need exceptional evidence
4. Check rejected_suggestions: do NOT re-suggest without new evidence
5. Check hypothesis_track_record: prioritize hypotheses with positive effectiveness
6. Structural proposals MUST include acceptance_criteria with measurable metrics
7. Max 5 suggestions ranked by confidence. Each MUST have a suggestion_id.
8. Check validation_patterns: categories with 3+ blocks in 30 days need explicit differentiation

## PORTFOLIO IMPROVEMENT ASSESSMENT
Review portfolio-level data (if present) and propose at most 2 portfolio-level changes:
- **Family performance trajectory**: compare family_snapshots trends against allocation weights
- **Portfolio rolling metrics**: reference portfolio_rolling_metrics for Sharpe/Sortino/Calmar trends
- **Drawdown correlation risk**: check drawdown_correlation for systemic risk signals
- **Portfolio rule blocks**: review rule_blocks_summary for coordination system effectiveness
- **Strategy engine detector findings**: If refinement_report data is present, it contains
  pre-computed statistical findings from 16 automated detectors. For the top 5
  highest-confidence findings, you MUST state AGREE or DISAGREE with 1-sentence
  reasoning. Do NOT duplicate detector analysis - focus your effort on patterns
  the detectors cannot cover (structural issues, cross-bot interactions, novel market conditions)
- **Allocation analysis**: validate quantitative rationale against your qualitative analysis

Portfolio proposal requirements:
- All proposals must cite specific family/bot data and projected portfolio Calmar impact
- portfolio_allocation proposals require 60+ days of evidence
- portfolio_risk_cap and portfolio_drawdown_tier require 90+ days of evidence
- Never suggest removing drawdown tiers or loosening stop levels
- Maximum 15% allocation change per family per cycle, minimum 5% floor
- Check portfolio_outcomes for past portfolio change track record

## CROSS-BOT TRANSFER
Review transfer_proposals (if present):
- Compatibility > 0.7: recommend with implementation notes
- Compatibility 0.4-0.7: flag as "worth investigating"
- Check transfer_track_record to favor proven patterns

## GROUND TRUTH PERFORMANCE (do not modify this evaluation)
If ground_truth_trend data is present, your job is to improve these composite
scores. Reference specific metric movements when proposing changes. If a bot's
composite is declining, prioritize diagnosis over new proposals.

## YOUR PREDICTION TRACK RECORD
If prediction_accuracy_by_metric data is present, recalibrate accordingly:
- Metrics where your accuracy < 50%: reduce confidence to 0.2-0.4 or skip
- Metrics where your accuracy > 70%: you may use confidence up to 0.8
- Review your worst-performing prediction categories
- Identify systematic biases and state what you will do differently

## DIRECTIONAL BIAS AWARENESS
If forecast_meta_analysis contains directional_bias data:
- "optimistic" bias: you predict improvement more than reality - reduce improve predictions
- "pessimistic" bias: you predict decline more than reality - consider improve scenarios
- Acknowledge your bias before making predictions in affected metrics

## LAST WEEK'S LEARNING SYNTHESIS (what the data shows)
If last_week_synthesis data is present:
- what_worked: Double down on these approaches. Propose similar changes.
- what_failed: Do NOT retry unless conditions demonstrably changed.
- discard: These categories have failed repeatedly - do NOT suggest them.
- lessons: Incorporate these insights into your current analysis.
- ground_truth_deltas: Reference these for performance trajectory.

## STRATEGY IDEAS UNDER REVIEW
If strategy_ideas data is present, for each active idea:
- Assess whether this week's data strengthens or weakens the edge hypothesis
- If evidence is growing (15+ data points, confidence > 0.7), recommend a backtest
- If evidence is contradicted, recommend retiring the idea with an explanation

## ACTIVE EXPERIMENTS
If active_experiments data is present, do NOT propose changes that overlap with
experiments currently in progress - let them complete their observation window.
Reference experiment status when discussing related metrics.

## EXPERIMENT TRACK RECORD
If experiment_track_record data is present, use it to calibrate your confidence
in structural proposals. Categories with high pass rates deserve more aggressive
proposals. Categories with low pass rates need stronger evidence before proposing.

## BACKTEST RELIABILITY
If backtest_reliability data is present, categories with reliability < 0.50
should be addressed with structural changes rather than parameter tuning - historical local backtests are unreliable for those categories.

## OPTIMIZATION ALLOCATION
If optimization_allocation data is present, reference when proposing suggestions:
- Prefer categories with high value_per_suggestion
- Categories with negative value_per_suggestion require exceptional evidence
- Follow _recommendations for shifting effort between categories
- Categories with 0 positive outcomes in 3+ attempts should be avoided

## SEARCH SIGNAL QUALITY
If search_signal_summary data is present:
- approve_rate < 0.3: detector firing on noise - investigate threshold
- approve_rate > 0.7: search productive - more suggestions in this category worthwhile
- Reference specific bot:category approve_rates when proposing parameter changes

## LEARNING SUFFICIENCY AUTHORITY
If learning_sufficiency data is present:
- Treat supported_learning_capabilities as the only capability-specific evidence that can justify high-confidence recommendations.
- Treat blocked_learning_capabilities and top_learning_gaps as instrumentation blockers, not weak trading evidence.
- For blocked capabilities, either label the idea hypothesis-only or recommend the specific instrumentation fix from top_learning_gaps.
- Do not present diagnostics-only evidence as approval-grade evidence.

## CYCLE EFFECTIVENESS TREND
If `cycle_effectiveness_trend` data is present, it shows the normalized
effectiveness score (0.0-1.0) for recent cycles. Use this to calibrate ambition:
- Effectiveness trending up: current approach is working - propose incremental refinements
- Effectiveness trending down: something is off - diagnose before proposing more changes
- Effectiveness plateau: consider targeted experiments to break through

## SUGGESTION QUALITY TREND
If `suggestion_quality_trend` data is present, it shows whether suggestion
generation quality is improving over time (hit rate, high-value category ratio).
- Rising hit_rate: your suggestions are getting better - maintain approach
- Falling hit_rate: recalibrate - check which categories are dragging quality down
- Low high_value_ratio: too many suggestions in low-value categories - shift focus

## CONVERGENCE STATUS (learning loop health)
If `convergence_report` is present, it shows whether the learning system is
improving, degrading, oscillating, or stable across multiple dimensions.
- If OSCILLATING: avoid reversing last week's suggestions - let changes settle
- If DEGRADING: question current approach fundamentals before proposing more changes
- If IMPROVING: maintain current approach, propose incremental refinements only
- Reference specific dimension statuses when justifying confidence levels

## DISCOVERIES AND STRATEGY IDEAS
If `discoveries` and `strategy_ideas` are present:
- Reference discoveries that corroborate or contradict detector findings
- For strategy_ideas with status "under_review": assess edge strength and recommend
  either backtest validation or retirement
- Do not propose structural changes that overlap with active strategy ideas
- If a discovery has been corroborated by 2+ weeks of data, escalate to structural proposal

## OUTCOME REASONING
If outcome_reasonings data is present, reference the causal mechanisms that
drove past successes and failures. Propose changes that leverage proven
mechanisms and avoid mechanisms that have consistently failed.

## HISTORICAL SEARCH RESULTS
If search_reports data is present, treat it as historical read-only context from
the retired local parameter-search path. Reference it only to avoid repeating
discarded parameter ideas. Material parameter or strategy changes must be
validated through the monthly evidence loop.

## MACRO REGIME ANALYSIS
If macro_regime_context data is present in the base package:
- Report current macro regime (G=Recovery, R=Reflation, S=Infl Hedge, D=Defensive),
  confidence, and stress level
- Break down weekly performance by macro regime: P&L, win rate, expectancy per regime
- Cross-reference macro_regime_sensitivity from strategy profiles:
  e.g., DownturnDominator_v1 should be "disabled" in G/R but "full" in S/D
- Evaluate regime config effectiveness: is regime_unit_risk_mult appropriate?
  If losses persist despite reduced sizing - recommend more aggressive reduction
  If winning strongly with heavy reduction - may be too conservative
- If a regime transition occurred this week, measure transition cost (P&L in ±5d window)
- Note: stress_level is observational only (41% FPR). If reporting stress-stratified
  outcomes, caveat that the signal cannot reliably discriminate stress from normal volatility
- Reference regime_config_history for trend analysis: has config been stable or shifting?
- Evaluate whether current regime config values are appropriate and propose adjustments

## INTER-STRATEGY COORDINATION
If coordination_rules data is present:
- Evaluate whether coordination signals fired correctly (e.g., ATRSS entry - AKC_HELIX
  stop tightening). Did the coordination improve or hurt outcomes?
- Check cooldown pair behavior - did cooldowns prevent good setups or correctly block whipsaws?
- Assess direction filter agreement rates and quality of filtered trades
- Review stock_coordination for symbol collision events and sizing adjustments

## ARCHETYPE-RELATIVE EVALUATION
If strategy_profiles and archetype_expectations data are present:
- Evaluate each strategy against its archetype's expected ranges, not universal benchmarks
- Trend-followers with 40% win rate and 2.5R payoff are HEALTHY - do not suggest
  tightening stops to improve win rate
- Breakout strategies with 35% win rate are NORMAL - focus on cost-per-attempt
- Flag strategies performing below archetype floor in their PREFERRED regime as problematic
- Strategies underperforming in ADVERSE regimes is EXPECTED - do not propose changes
- Use portfolio_risk_config to validate that suggestions stay within risk bounds
- For strategies with `sub_engines` in their profile, compare performance across engines
  to identify which engines perform best in each regime/vol state combination.
- For strategies with `entry_types`, compare entry type win rates and payoff ratios.
- Reference the strategy's `analysis_focus` list for priority analytical dimensions.

## ENGINE-LEVEL WEEKLY ANALYSIS
When engine_decomposition data spans multiple days, aggregate across the week:
- Identify engines with declining win rate or profit factor trends over 7 days
- Flag regime-engine combinations where trade count is sufficient but performance is poor
- Proposals targeting a specific engine MUST include `engine` field in structured output

## ABLATION FLAG TRENDS
When ablation_analysis data is present across multiple days:
- Look for flags where the on/off performance gap is consistent across the week
- Flag ablation results with strong statistical significance (p < 0.05)
- Safety-critical flags (stop-loss, circuit breakers, risk caps) require manual review - do NOT propose toggling them autonomously

## EXIT TIER OPTIMIZATION
When exit_tier_analysis data is present:
- Compare actual MFE distributions against configured TP tier targets
- Identify tiers with very low hit rates (< 20%) that may need lowering
- Identify tiers where lowering the target would capture significantly more trades
- For mean_reversion_pullback archetype: high win rate + low payoff is expected - flag if win rate drops below archetype floor or if average loss exceeds 1.5x average win.

## SELF-ASSESSMENT
If self_assessment data is present, READ IT CAREFULLY. This summarizes your known
biases, weak categories, and recurring mistakes. You MUST:
- Acknowledge biases before making predictions in affected metrics
- Avoid or explicitly justify suggestions in weak categories
- Not repeat patterns listed in recurring corrections

## CONSTRAINTS (enforced by validator - violations are automatically stripped)
- Do NOT restate the computed summary - it's above.
- Focus analytical effort on the questions, not on re-summarizing data.
- BLOCKED: NEGATIVE outcome_measurements categories - do NOT re-suggest similar approaches.
- BLOCKED: structural proposals without acceptance_criteria with measurable metrics.
- BLOCKED: hypotheses with effectiveness <= 0 or status="retired" - do NOT re-propose.
- Overconfident predictions are capped by forecast_meta_analysis calibration data.
- outcome_measurements contains only HIGH/MEDIUM quality data. spurious_outcomes
  (if present) had confounding factors (concurrent changes, regime shifts,
  low/insufficient measurement quality) - treat as hypotheses, not evidence.

## STRUCTURED OUTPUT (REQUIRED)
At the END of your analysis, emit a structured data block.
CRITICAL: This block is machine-parsed by the learning system. If you omit it,
your suggestions and predictions are LOST and cannot improve future performance.
Always emit it, even if arrays are empty.
For `strategy_id`: use the exact strategy_id from the strategy registry (e.g.
"TPC", "ATRSS", "AKC_HELIX") when the suggestion or prediction is specific to
one strategy; set to null only when the signal genuinely applies bot-wide
across all that bot's strategies.
<!-- STRUCTURED_OUTPUT
{{
  "predictions": [
    {{"bot_id": "...", "strategy_id": "TPC|ATRSS|...|null", "metric": "pnl|win_rate|drawdown|sharpe", "direction": "improve|decline|stable", "confidence": 0.0-1.0, "timeframe_days": 7, "reasoning": "..."}}
  ],
  "suggestions": [
    {{"suggestion_id": "#abc123", "bot_id": "...", "strategy_id": "TPC|ATRSS|...|null", "category": "exit_timing|filter_threshold|stop_loss|signal|structural|position_sizing|regime_gate|funding_threshold|leverage_cap|confluence_count|setup_grade_filter", "title": "...", "expected_impact": "...", "confidence": 0.0-1.0, "evidence_summary": "...", "proposed_value": 0.5, "target_param": "param_name"}}
  ],
  "structural_proposals": [
    {{"hypothesis_id": "REQUIRED: use id from structural_hypotheses if matching, else null", "bot_id": "...", "title": "...", "description": "...", "reversibility": "easy|moderate|hard", "evidence": "...", "estimated_complexity": "low|medium|high", "acceptance_criteria": [{{"metric": "...", "direction": "improve|not_degrade", "minimum_change": 0.0, "observation_window_days": 14, "minimum_trade_count": 20}}]}}
  ],
  "portfolio_proposals": [
    {{"proposal_type": "allocation_rebalance|risk_cap_change|coordination_change|drawdown_tier_change", "current_config": {{}}, "proposed_config": {{}}, "evidence_summary": "cite specific family metrics, correlation data, and time period", "expected_portfolio_calmar_delta": 0.0, "confidence": 0.0-1.0, "observation_window_days": 30}}
  ]
}}
-->"""

_CRYPTO_WEEKLY_SUPPLEMENT = """
## CRYPTO PERPETUAL WEEKLY ANALYSIS
This bot trades crypto perpetual futures. Apply the following weekly-specific guidance:

**Funding trend**: Week-over-week funding cost trajectory. Is it growing? If funding_pct_of_gross
exceeded 15% any day this week, flag as a structural cost concern.

**Funnel trend**: If funnel_weekly_summary is present, compare setup_to_fill
and fill_to_close conversion against the daily funnel_trends. A worsening
setup-to-confirmation or confirmation-to-entry conversion should be handled as a
funnel/process question before changing risk.

**Health gating**: If health_summary shows degraded or critical days, separate bot/process
health from trading-edge conclusions. Strategy changes should not be justified by periods
with stale feeds, repeated disconnects, or severe alert bursts.

**Grade calibration**: Rolling 7-day A vs B performance. Is the grade split still justified?
If B-grade trades lost money on aggregate this week, recommend disabling B entries.

**Cross-strategy correlation**: If all 3 strategies lost on the same days, the issue is macro/
regime, not per-strategy tuning. Do NOT suggest parameter changes when the root cause is regime
or market conditions - suggest regime gate changes instead.

**Crypto evidence requirements**: Parameter changes on leveraged instruments require more
evidence (leverage amplifies both signal and noise). Minimum 30 trades per grade/confluence
bucket before drawing conclusions. For leverage and risk_pct changes, require 60+ days of data.
"""

# Legacy instructions for when no triage is provided
_WEEKLY_INSTRUCTIONS = _FOCUSED_WEEKLY_INSTRUCTIONS.format(
    computed_summary="(No triage data - compute weekly summary manually from daily reports)",
    anomalies="(No triage - review all weekly data for structural patterns)",
    retrospective_questions="Review past predictions and suggestions. Which were accurate and which were not? Why?",
    discovery_questions="What patterns in this week's data might the automated detectors miss?",
)


class WeeklyPromptAssembler:
    """Assembles the full context package for a weekly analysis agent invocation."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
        bot_configs: dict | None = None,
        strategy_registry=None,
        run_index: object | None = None,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.runs_dir = runs_dir
        self.bot_configs = bot_configs
        self.strategy_registry = strategy_registry
        self._evidence = EvidenceMemory(
            memory_dir,
            run_index=run_index,
            strategy_registry=strategy_registry,
        )
        self._ctx = ContextBuilder(
            memory_dir,
            curated_dir=curated_dir,
            run_index=run_index,
            evidence_memory=self._evidence,
        )

    def assemble(self, triage_report=None, session_store=None) -> PromptPackage:
        """Build the complete weekly prompt package.

        Args:
            triage_report: Optional WeeklyTriageReport from WeeklyTriage. When
                provided, instructions are focused on computed summaries and
                targeted questions. When None, uses fallback instructions.
            session_store: Optional SessionStore for loading session history.
        """
        pkg = self._ctx.base_package(
            session_store=session_store,
            agent_type="weekly_analysis",
            bot_configs=self.bot_configs,
            strategy_registry=self.strategy_registry,
        )
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_data())
        focus = weekly_focus_for_week(self.week_start)
        pkg.data["weekly_focus"] = weekly_focus_payload(self.week_start)
        pkg.data["weekly_focus_rotation"] = weekly_focus_rotation_payload()
        pkg.instructions = self._build_instructions(triage_report)
        pkg.context_files.extend(self._list_data_files())
        pkg.metadata["bot_ids"] = ",".join(self.bots)
        pkg.metadata["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pkg.metadata["weekly_focus_id"] = focus.focus_id
        return pkg

    def _build_instructions(self, triage_report=None) -> str:
        """Build instructions from triage report or use fallback."""
        if triage_report is None:
            instructions = _WEEKLY_INSTRUCTIONS
            if self.strategy_registry and self._has_crypto_strategies():
                instructions += _CRYPTO_WEEKLY_SUPPLEMENT
            return self._weekly_focus_instruction_block() + instructions

        # Format anomalies
        anomaly_lines = []
        for i, a in enumerate(triage_report.anomalies, 1):
            bot_tag = f"[{a.bot_id}] " if a.bot_id else ""
            anomaly_lines.append(
                f"{i}. **[{a.anomaly_type.upper()}]** {bot_tag}"
                f"(severity: {a.severity}) - {a.description}"
            )
        anomalies_text = "\n".join(anomaly_lines) if anomaly_lines else "(No anomalies detected - stable week)"

        # Format retrospective questions
        retro_lines = []
        for i, q in enumerate(triage_report.retrospective_questions, 1):
            retro_lines.append(f"{i}. {q}")
        retro_text = "\n".join(retro_lines)

        # Format discovery questions
        disc_lines = []
        for i, q in enumerate(triage_report.discovery_questions, 1):
            disc_lines.append(f"{i}. {q}")
        disc_text = "\n".join(disc_lines)

        instructions = _FOCUSED_WEEKLY_INSTRUCTIONS.format(
            computed_summary=triage_report.computed_summary,
            anomalies=anomalies_text,
            retrospective_questions=retro_text,
            discovery_questions=disc_text,
        )

        # Append crypto supplement if any bot has crypto perpetual strategies
        if self.strategy_registry and self._has_crypto_strategies():
            instructions += _CRYPTO_WEEKLY_SUPPLEMENT

        return self._weekly_focus_instruction_block() + instructions

    def _has_crypto_strategies(self) -> bool:
        """Check if any bot in scope has crypto perpetual strategies."""
        if not self.strategy_registry or not hasattr(self.strategy_registry, "strategies"):
            return False
        bot_set = set(self.bots)
        for _sid, profile in self.strategy_registry.strategies.items():
            if getattr(profile, "asset_class", "") == "crypto_perpetual":
                if not bot_set or getattr(profile, "bot_id", "") in bot_set:
                    return True
        return False

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        focus = weekly_focus_for_week(self.week_start)
        return (
            f"Produce the weekly summary for {self.week_start} to {self.week_end} "
            f"covering all bots: {bot_list}.\n"
            f"Active weekly focus: {focus.label}. "
            f"Focus on the retrospective and discovery questions. "
            f"Reason about WHY things happened, not just WHAT happened."
        )

    def _weekly_focus_instruction_block(self) -> str:
        focus = weekly_focus_for_week(self.week_start)
        families = ", ".join(focus.portfolio_families)
        strategies = ", ".join(focus.strategy_ids)
        return (
            "## WEEKLY PORTFOLIO FOCUS ROTATION\n"
            f"Active focus: {focus.label}.\n"
            f"Portfolio families: {families}.\n"
            f"Strategy surfaces: {strategies}.\n"
            "This focus is evidence triage for the next monthly_search_brief only. "
            "It must not approve changes, bypass monthly phased-auto, trigger OOS repair, "
            "or satisfy approval gates.\n\n"
        )

    def _load_data(self) -> dict:
        data: dict = {}
        data_load_errors: list[dict] = []

        weekly_dir = self.curated_dir / "weekly" / self.week_start
        for key, filename in [
            ("weekly_summary", "weekly_summary.json"),
            ("refinement_report", "refinement_report.json"),
            ("week_over_week", "week_over_week.json"),
            ("allocation_analysis", "allocation_analysis.json"),
            ("structural_analysis", "structural_analysis.json"),
            ("regime_conditional_analysis", "regime_conditional_analysis.json"),
            ("interaction_analysis", "interaction_analysis.json"),
            ("allocation_drift", "allocation_drift.json"),
        ]:
            path = weekly_dir / filename
            if path.exists():
                loaded = self._safe_load_json(path, data_load_errors)
                if loaded is not None:
                    data[key] = loaded

        data["daily_reports"] = self._load_daily_reports()
        data["portfolio_risk_cards"] = self._load_risk_cards(data_load_errors)
        learning_sufficiency = self._load_learning_sufficiency_summaries(data_load_errors)
        if learning_sufficiency:
            data["learning_sufficiency"] = learning_sufficiency
        funnel_trends = self._load_daily_file("funnel_analysis.json", data_load_errors)
        if funnel_trends:
            data["funnel_trends"] = funnel_trends
            weekly_funnel_summary = self._aggregate_funnel_trends(funnel_trends)
            if weekly_funnel_summary:
                data["funnel_weekly_summary"] = weekly_funnel_summary
            crypto_funnel_trends = [
                record for record in funnel_trends
                if "crypto" in str(record.get("bot_id") or "").lower()
            ]
            if crypto_funnel_trends:
                data["crypto_funnel_trends"] = crypto_funnel_trends
                crypto_funnel_summary = self._aggregate_funnel_trends(crypto_funnel_trends)
                if crypto_funnel_summary:
                    data["crypto_funnel_weekly_summary"] = crypto_funnel_summary
        health_summaries = self._load_daily_file("health_summary.json", data_load_errors)
        if health_summaries:
            data["health_summaries"] = health_summaries
            crypto_health_summaries = [
                record for record in health_summaries
                if "crypto" in str(record.get("bot_id") or "").lower()
            ]
            if crypto_health_summaries:
                data["crypto_health_summaries"] = crypto_health_summaries

        if data_load_errors:
            data["data_load_errors"] = data_load_errors
        return data

    def _load_daily_reports(self) -> list[dict]:
        reports: list[dict] = []
        for date_str in self._week_dates():
            report_paths = self._find_daily_report_paths(date_str)
            if report_paths:
                reports.append({
                    "date": date_str,
                    "content": "\n\n".join(
                        path.read_text(encoding="utf-8") for path in report_paths
                    ),
                })
        return reports

    def _load_risk_cards(self, data_load_errors: list[dict]) -> list[dict]:
        cards: list[dict] = []
        for date_str in self._week_dates():
            card_path = self.curated_dir / date_str / "portfolio_risk_card.json"
            if card_path.exists():
                loaded = self._safe_load_json(card_path, data_load_errors)
                if isinstance(loaded, dict):
                    cards.append(loaded)
        return cards

    def _load_daily_file(self, filename: str, data_load_errors: list[dict]) -> list[dict]:
        records: list[dict] = []
        for date_str in self._week_dates():
            for bot in self.bots:
                path = self.curated_dir / date_str / bot / filename
                if not path.exists():
                    continue
                loaded = self._safe_load_json(path, data_load_errors)
                if isinstance(loaded, dict):
                    records.append({"date": date_str, "bot_id": bot, "data": loaded})
        return records

    def _load_learning_sufficiency_summaries(self, data_load_errors: list[dict]) -> dict:
        manifest_paths = sorted(
            self.runs_dir.rglob("learning_sufficiency_manifest.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )[:20]
        summaries: list[dict] = []
        top_gaps: list[dict] = []
        bot_filter = set(self.bots)
        for path in manifest_paths:
            loaded = self._safe_load_json(path, data_load_errors)
            if not isinstance(loaded, dict):
                continue
            bot_id = str(loaded.get("bot_id") or "")
            if bot_filter and bot_id and bot_id not in bot_filter:
                continue
            strategy_id = str(loaded.get("strategy_id") or "")
            source_authoritative = manifest_runtime_support_source_authoritative(loaded)
            supported = _string_list(loaded.get("supported_learning_capabilities")) if source_authoritative else []
            blocked = _string_list(loaded.get("blocked_learning_capabilities"))
            if not source_authoritative and "runtime_source_classification_violation" not in blocked:
                blocked.append("runtime_source_classification_violation")
            summaries.append({
                "bot_id": bot_id,
                "strategy_id": strategy_id,
                "run_month": loaded.get("run_month", ""),
                "eligibility": loaded.get("eligibility", "") if source_authoritative else "diagnostics_only",
                "supported_learning_capabilities": supported,
                "blocked_learning_capabilities": blocked,
                "manifest_path": str(path),
            })
            for gap in loaded.get("known_gaps", []) or []:
                if not isinstance(gap, dict):
                    continue
                top_gaps.append({
                    "bot_id": bot_id,
                    "strategy_id": strategy_id,
                    "blocked_learning_capability": gap.get("blocked_learning_capability", ""),
                    "expected_learning_value": gap.get("expected_learning_value", ""),
                    "event_type": gap.get("event_type", ""),
                    "missing_field": gap.get("missing_field", ""),
                    "remediation": gap.get("remediation", ""),
                    "manifest_path": str(path),
                })
        if not summaries:
            return {}
        top_gaps = sorted(
            top_gaps,
            key=lambda item: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    str(item.get("expected_learning_value") or "").lower(),
                    4,
                ),
                str(item.get("bot_id") or ""),
                str(item.get("strategy_id") or ""),
                str(item.get("blocked_learning_capability") or ""),
            ),
        )[:10]
        return {
            "scope_count": len(summaries),
            "supported_capabilities_by_scope": [
                {
                    "bot_id": item["bot_id"],
                    "strategy_id": item["strategy_id"],
                    "capabilities": item["supported_learning_capabilities"],
                    "manifest_path": item["manifest_path"],
                }
                for item in summaries
                if item["supported_learning_capabilities"]
            ],
            "blocked_capabilities_by_scope": [
                {
                    "bot_id": item["bot_id"],
                    "strategy_id": item["strategy_id"],
                    "eligibility": item["eligibility"],
                    "capabilities": item["blocked_learning_capabilities"],
                    "manifest_path": item["manifest_path"],
                }
                for item in summaries
                if item["blocked_learning_capabilities"]
            ],
            "top_learning_gaps": top_gaps,
        }

    def _aggregate_funnel_trends(self, records: list[dict]) -> dict:
        stages = [
            "bars_received",
            "indicators_ready",
            "setups_detected",
            "confirmations",
            "entries_attempted",
            "fills",
            "trades_closed",
        ]
        totals = {stage: 0 for stage in stages}
        per_bot: dict[str, dict[str, int]] = {}
        per_strategy: dict[str, dict[str, int]] = {}
        coverage_dates: set[str] = set()

        for record in records:
            data = record.get("data")
            if not isinstance(data, dict):
                continue
            stage_totals = self._normalize_funnel_stage_totals(data.get("stage_totals", {}))
            if not any(stage_totals.values()):
                continue
            coverage_dates.add(str(record.get("date") or ""))
            bot_id = str(record.get("bot_id") or "unknown")
            bot_totals = per_bot.setdefault(bot_id, {stage: 0 for stage in stages})
            self._add_funnel_totals(totals, stage_totals)
            self._add_funnel_totals(bot_totals, stage_totals)

            strategy_breakdown = data.get("per_strategy_breakdown", {})
            if isinstance(strategy_breakdown, dict):
                for strategy_id, breakdown in strategy_breakdown.items():
                    if not isinstance(breakdown, dict):
                        continue
                    strategy_totals = per_strategy.setdefault(
                        str(strategy_id),
                        {stage: 0 for stage in stages},
                    )
                    self._add_funnel_totals(
                        strategy_totals,
                        self._normalize_funnel_stage_totals(breakdown.get("stage_totals", {})),
                    )

        if not any(totals.values()):
            return {}
        return {
            "coverage_days": len({day for day in coverage_dates if day}),
            "bots_with_coverage": sorted(per_bot),
            "stage_totals": totals,
            "conversion_rates": self._weekly_funnel_conversion_rates(totals),
            "per_bot_breakdown": {
                bot_id: {
                    "stage_totals": bot_totals,
                    "conversion_rates": self._weekly_funnel_conversion_rates(bot_totals),
                }
                for bot_id, bot_totals in sorted(per_bot.items())
            },
            "per_strategy_breakdown": {
                strategy_id: {
                    "stage_totals": strategy_totals,
                    "conversion_rates": self._weekly_funnel_conversion_rates(strategy_totals),
                }
                for strategy_id, strategy_totals in sorted(per_strategy.items())
            },
        }

    @staticmethod
    def _normalize_funnel_stage_totals(payload: object) -> dict[str, int]:
        stages = [
            "bars_received",
            "indicators_ready",
            "setups_detected",
            "confirmations",
            "entries_attempted",
            "fills",
            "trades_closed",
        ]
        aliases = {
            "bars_received": "bars_received",
            "signals_generated": "bars_received",
            "indicators_ready": "indicators_ready",
            "setups_detected": "setups_detected",
            "setups_seen": "setups_detected",
            "setups_qualified": "setups_detected",
            "confirmations": "confirmations",
            "confirmations_passed": "confirmations",
            "entries_attempted": "entries_attempted",
            "entries_taken": "entries_attempted",
            "fills": "fills",
            "trades_closed": "trades_closed",
        }
        totals = {stage: 0 for stage in stages}
        if not isinstance(payload, dict):
            return totals
        for key, value in payload.items():
            stage = aliases.get(str(key))
            if not stage:
                continue
            if isinstance(value, dict):
                totals[stage] += sum(
                    int(item or 0)
                    for item in value.values()
                    if isinstance(item, (int, float))
                )
            elif isinstance(value, (int, float)):
                totals[stage] += int(value)
        return totals

    @staticmethod
    def _add_funnel_totals(target: dict[str, int], increment: dict[str, int]) -> None:
        for stage, value in increment.items():
            if stage in target:
                target[stage] += int(value or 0)

    @staticmethod
    def _weekly_funnel_conversion_rates(totals: dict[str, int]) -> dict[str, float]:
        pairs = [
            ("bars_to_indicators", "bars_received", "indicators_ready"),
            ("indicators_to_setups", "indicators_ready", "setups_detected"),
            ("setups_to_confirmations", "setups_detected", "confirmations"),
            ("confirmations_to_entries", "confirmations", "entries_attempted"),
            ("entries_to_fills", "entries_attempted", "fills"),
            ("setup_to_fill", "setups_detected", "fills"),
            ("fill_to_close", "fills", "trades_closed"),
        ]
        rates: dict[str, float] = {}
        for label, before, after in pairs:
            denominator = totals.get(before, 0)
            rates[label] = round(totals.get(after, 0) / denominator, 4) if denominator else 0.0
        return rates

    def _list_data_files(self) -> list[str]:
        files: list[str] = []
        weekly_dir = self.curated_dir / "weekly" / self.week_start
        for name in [
            "weekly_summary.json", "refinement_report.json", "week_over_week.json",
            "allocation_analysis.json", "structural_analysis.json",
            "regime_conditional_analysis.json", "interaction_analysis.json",
            "allocation_drift.json",
        ]:
            path = weekly_dir / name
            if path.exists() and self._json_file_loadable(path):
                files.append(str(path))
        for date_str in self._week_dates():
            for report_path in self._find_daily_report_paths(date_str):
                files.append(str(report_path))
            for bot in self.bots:
                for name in ("funnel_analysis.json", "health_summary.json"):
                    path = self.curated_dir / date_str / bot / name
                    if path.exists() and self._json_file_loadable(path):
                        files.append(str(path))
        for path in sorted(
            self.runs_dir.rglob("learning_sufficiency_manifest.json"),
            key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0.0,
            reverse=True,
        )[:20]:
            if path.exists() and self._json_file_loadable(path):
                files.append(str(path))
        return files

    @staticmethod
    def _safe_load_json(path: Path, errors: list[dict]) -> object | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed weekly JSON %s: %s", path, exc)
            errors.append({"path": str(path), "error": str(exc)})
            return None

    @staticmethod
    def _json_file_loadable(path: Path) -> bool:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return True

    def _week_dates(self) -> list[str]:
        """Generate the 7 date strings for this week."""
        start = datetime.strptime(self.week_start, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    def _find_daily_report_paths(self, date_str: str) -> list[Path]:
        paths: list[Path] = []
        for run_dir in sorted(self.runs_dir.glob(f"daily-{date_str}*")):
            if not run_dir.is_dir():
                continue
            report_path = run_dir / "daily_report.md"
            if report_path.exists():
                paths.append(report_path)
                continue
            fallback = run_dir / "response.md"
            if fallback.exists():
                paths.append(fallback)
        return paths


    def _build_computed_summary(self, triage_report) -> str:
        """Extract computed summary from triage report."""
        if not triage_report:
            return ""
        return triage_report.computed_summary


def _string_list(value: object) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return [str(value)]
