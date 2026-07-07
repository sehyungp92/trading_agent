"""Prompt assembler - builds the context package for the daily analysis runtime invocation.

Uses deterministic triage to pre-process data and generate focused analytical
questions. Claude reasons about 3-5 significant events rather than mechanically
checking 29 items.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, time, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.evidence_memory import EvidenceMemory
from trading_assistant.schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

_CURATED_FILES = [
    "summary.json",
    "winners.json",
    "losers.json",
    "process_failures.json",
    "notable_missed.json",
    "regime_analysis.json",
    "filter_analysis.json",
    "root_cause_summary.json",
    "factor_attribution.json",
    "exit_efficiency.json",
    "hourly_performance.json",
    "slippage_stats.json",
    "excursion_stats.json",
    "overlay_state_summary.json",
    "experiment_data.json",
    "signal_health.json",
    "fill_quality.json",
    "filter_decisions.json",
    "indicator_snapshots.json",
    "orderbook_stats.json",
    "parameter_changes.json",
    "order_lifecycle.json",
    "process_quality.json",
    "applied_regime_config.json",
    "stop_adjustment_analysis.json",
    "execution_latency.json",
    "sizing_analysis.json",
    "param_outcome_correlation.json",
    "portfolio_context.json",
    "market_conditions.json",
    "engine_decomposition.json",
    "ablation_analysis.json",
    "exit_tier_analysis.json",
    "funding_analysis.json",
    "grade_analysis.json",
    "confluence_analysis.json",
    "leverage_analysis.json",
    "funnel_analysis.json",
    "health_summary.json",
]

# Portfolio-level curated files (loaded from curated/{date}/portfolio/)
_PORTFOLIO_CURATED_FILES = [
    "rule_blocks_summary.json",
    "family_snapshots.json",
    "concurrent_position_analysis.json",
    "sector_exposure.json",
    "portfolio_rolling_metrics.json",
    "macro_regime_analysis.json",
]

# Focused instructions that require causal reasoning, not checklist narration.
_FOCUSED_INSTRUCTIONS = """\
You are analyzing today's trading data. A deterministic triage system has already
pre-processed the data and identified what deserves your attention.

## ROUTINE SUMMARY (pre-computed - do NOT regenerate this)
{routine_summary}

## SIGNIFICANT EVENTS REQUIRING YOUR ANALYSIS
{significant_events}

## YOUR ANALYTICAL TASKS

For each significant event above, you MUST:
1. State a hypothesis about WHY it happened
2. Identify confirming AND refuting evidence in the data
3. Check the evidence - does it support or undermine your hypothesis?
4. Rate your confidence (0.0-1.0) with explicit justification

{focus_questions}

## GROUND TRUTH PERFORMANCE (do not modify this evaluation)
If ground_truth_trend data is present, reference specific metric movements when
proposing changes. If a bot's composite is declining, prioritize diagnosis over
new proposals.

## YOUR PREDICTION TRACK RECORD
If prediction_accuracy_by_metric data is present, recalibrate accordingly:
- Metrics where your accuracy < 50%: reduce confidence to 0.2-0.4 or skip predictions
- Metrics where your accuracy > 70%: you may use confidence up to 0.8
- No data yet: use conservative confidence (0.3-0.5)
- Explain what you got wrong last time and why

## DIRECTIONAL BIAS AWARENESS
If forecast_meta_analysis contains directional_bias data:
- "optimistic" bias: you predict improvement more than reality - reduce improve predictions
- "pessimistic" bias: you predict decline more than reality - consider improve scenarios
- Acknowledge your bias before making predictions in affected metrics

## BLOCKED APPROACHES
If last_week_synthesis data contains a "discard" list, do NOT suggest those
approaches - they have failed repeatedly.

## HISTORICAL SEARCH CONTEXT
If search_reports data is present, treat it as historical read-only context from
the retired local parameter-search path. Use it to avoid repeating discarded
ideas, but do not treat it as current approval authority. Material parameter or
strategy changes must go through monthly validation.
If backtest_reliability data is present, categories marked as unreliable should
be handled conservatively and escalated to monthly evidence rather than further
daily parameter tuning.

## BLOCKED SUGGESTION PATTERNS
If validation_patterns data is present, it shows which suggestion categories have
been repeatedly blocked by the validator over the last 30 days. For categories
with 3+ blocks, you MUST either: (a) avoid that category, or (b) explicitly
explain why your proposal differs from what was previously blocked.

## HYPOTHESIS TRACK RECORD
If hypothesis_track_record data is present, prioritize hypotheses with positive
effectiveness scores when making structural proposals. Do NOT propose changes
linked to hypotheses with effectiveness <= 0 or status="retired".

## ACTIVE EXPERIMENTS
If active_experiments data is present, do NOT propose changes that overlap with
experiments currently in progress - let them complete their observation window.
Reference experiment status when discussing related metrics.

## STRATEGY CONTEXT
If strategy_profiles data is present:
- Each strategy has an archetype (trend_follow, breakout, pullback, etc.) with expected
  performance ranges in archetype_expectations
- Distinguish EXPECTED underperformance (strategy in adverse regime) from PROBLEMATIC
  underperformance (strategy in preferred regime but still losing)
- Reference coordination_rules when analyzing multi-strategy bots - did coordination
  signals fire correctly? Did cooldown pairs prevent whipsaws?
- Check portfolio_risk_config bounds before suggesting parameter changes - never suggest
  exceeding heat_cap_R or daily_stop_R limits
- For strategies with `sub_engines` in their profile, compare performance across engines
  to identify which engines perform best in each regime/vol state combination.
- For strategies with `entry_types`, compare entry type win rates and payoff ratios.
- Reference the strategy's `analysis_focus` list for priority analytical dimensions.
- For mean_reversion_pullback archetype: high win rate + low payoff is expected - flag if win rate drops below archetype floor or if average loss exceeds 1.5x average win.

## ENGINE-LEVEL ANALYSIS
If engine_decomposition data is present, it shows per-engine metrics with regime
breakdowns for strategies with multiple sub-engines (e.g., Downturn's REVERSAL,
BREAKDOWN, FADE). When this data is available:
- Identify which engines underperform in which regimes
- Compare engine win rates, profit factors, and exit efficiency
- Proposals targeting a specific engine MUST include `engine` field in structured output
- If one engine dominates losses, consider whether it should be disabled in certain regimes

## ABLATION ANALYSIS
If ablation_analysis data is present, it shows statistical comparisons of boolean
flag on/off states from strategy_params_at_entry. When this data is available:
- Review `flags_with_signal` - these are flags with statistically significant (p < 0.10)
  performance differences between enabled and disabled states
- Proposals to toggle a flag MUST reference the statistical evidence (p-value, PnL delta)
- Include `ablation_flag` field in structured output for flag toggle suggestions
- Do NOT suggest toggling safety-critical flags (circuit breakers, position limits)

## EXIT TIER ANALYSIS
If exit_tier_analysis data is present, it shows TP tier hit rates based on actual
MFE (maximum favorable excursion) data. When this data is available:
- Evaluate whether TP targets match actual MFE distributions
- If a tier has < 20% hit rate, it may be set too aggressively
- If optimal_target differs significantly from current_target, propose adjustment
- Reference stop_placement data when discussing stop levels

## MACRO REGIME CONTEXT
If macro_regime_analysis data is present (in portfolio curated files):
- Current macro regime (G=Recovery, R=Reflation, S=Infl Hedge, D=Defensive)
  drives portfolio-wide sizing and strategy enable/disable decisions
- Check applied_regime_config per bot: active sizing multiplier, directional caps, disabled strategies
- Cross-reference today's performance with macro regime expectations:
  strategies with macro_regime_sensitivity "disabled" in current regime should have zero trades
- Note: stress_level is observational only (41% false positive rate) - record it for diagnostics but do not use it to gate decisions or flag entries
- If regime changed from yesterday: flag as transition event, assess transition cost
- Note: macro regime (G/R/S/D) is distinct from per-trade market_regime (trending_up, ranging, etc.)

## CONVERGENCE STATUS (learning loop health)
If `convergence_report` is present, it shows whether the learning system is
improving, degrading, oscillating, or stable across multiple dimensions.
- If OSCILLATING: avoid reversing last week's suggestions - let changes settle
- If DEGRADING: question current approach fundamentals before proposing more changes
- If IMPROVING: maintain current approach, propose incremental refinements only
- Reference specific dimension statuses when justifying confidence levels

## DISCOVERIES (from automated pattern discovery)
If `discoveries` is present in your data, these are patterns found by a separate
discovery agent scanning raw JSONL data. Reference relevant discoveries when they
corroborate or contradict your analysis. Flag if a discovery seems invalidated by
recent data.

## EXECUTION & SIZING CONTEXT
If execution_latency data is present:
- Identify execution pipeline bottleneck stages
- Correlate latency with slippage - is latency causing worse fills?
- Focus on systematic patterns, not individual outliers

If sizing_analysis data is present:
- Compare sizing model effectiveness across conditions
- Check if risk utilization aligns with signal conviction (strong signals should have larger positions)

If param_outcome_correlation data is present:
- Reference specific parameter ranges correlated with better outcomes
- Cross-reference with parameter_changes for recent drift into/away from optimal ranges
- Require 20+ trades per bucket - do NOT draw conclusions from small samples

If portfolio_context data is present:
- Flag entries during high exposure or with many correlated positions
- Check if crowded entries systematically underperform

If market_conditions data is present:
- Identify specific condition combinations (beyond regime label) that predict outcomes
- Cross-reference with regime analysis for consistency

## SELF-ASSESSMENT
If self_assessment data is present, READ IT CAREFULLY. This summarizes your known
biases, weak categories, and recurring mistakes. You MUST:
- Acknowledge biases before making predictions in affected metrics
- Avoid or explicitly justify suggestions in weak categories
- Not repeat patterns listed in recurring corrections

## CONSTRAINTS (enforced by validator - violations are automatically stripped)
- Do NOT restate the routine summary - it's already computed above.
- Do NOT mechanically review every data file - focus only on what the triage flagged.
- Every suggestion MUST include quantified expected impact (PnL range, drawdown change)
  with evidence base (trade count, time period, statistical significance).
- Check rejected_suggestions: do NOT re-suggest previously rejected items without new evidence.
- Check active_suggestions: do NOT contradict DEPLOYED suggestions - propose revert with evidence.
- BLOCKED by validator: categories with win_rate < 30% (n>=5) in category_scorecard.
  Only propose with exceptional new evidence and explicit justification.
- BLOCKED by validator: structural suggestions with confidence < 0.4.
- BLOCKED by validator: suggestions without quantified expected impact (quantification required).
- Prediction calibration: accuracy < 50% - cap confidence at 0.3; > 70% - up to 0.8.
- outcome_measurements contains only HIGH/MEDIUM quality data. spurious_outcomes
  (if present) had confounding factors - treat as hypotheses, not evidence.

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
  ]
}}
-->"""

_CRYPTO_DAILY_SUPPLEMENT = """
## CRYPTO PERPETUAL ANALYSIS
This bot trades crypto perpetual futures. Apply the following crypto-specific guidance:

**Funding analysis**: Evaluate funding_analysis data - is funding cost eroding edge?
Compare per-direction (longs pay positive funding, shorts receive in positive-rate regimes).
Suggest time-stop tightening if avg_funding_per_hour is high relative to expected R.

**Grade performance**: Compare A vs B grade outcomes in grade_analysis data. Is the grade
differential justified by performance? Flag if B-grade expectancy is negative with 20+ trades.

**Confluence quality**: Review confluence_analysis factor-level data. Which confluences have
highest lift? Suggest raising minimums if clear breakpoints exist in by_count data.

**24/7 market**: Do NOT reference market open/close, pre-market, or extended hours. Crypto
trades continuously. Use Asia/EU/US liquidity overlap windows (00-04/07-11/13-17 UTC) for
time-of-day analysis instead.

**Leverage awareness**: If leverage_analysis data is present, assess liquidation proximity.
Flag any near_liquidation events as critical safety concerns.

**Funnel triage**: If funnel_analysis data is present, identify the top drop-off stage
and cross-reference detector findings before proposing parameter changes. For example,
confirmation leaks plus weak B-grade conversion should point toward confirmation or
confluence tightening, not broad risk reduction.

**Health gating**: If health_summary has HIGH/ERROR/CRITICAL alerts, treat the day as a
process-quality concern first. Do not propose strategy changes during periods with
repeated disconnects, stale feeds, or severe data-flow issues.

**Multi-strategy coordination**: Note max_concurrent and correlated risk limits. Flag if
same-direction trades across strategies caused concentrated losses.
"""

# Legacy instructions kept for backward compatibility (used when no triage is provided)
_INSTRUCTIONS = _FOCUSED_INSTRUCTIONS.format(
    routine_summary="(No triage data - review all bots manually)",
    significant_events="(No triage - review all curated data files for anomalies)",
    focus_questions="Analyze today's trading performance across all bots. Focus on anomalies, "
    "process failures, and actionable improvements.",
)


class DailyPromptAssembler:
    """Assembles the full context package for a daily analysis agent invocation."""

    def __init__(
        self,
        date: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        corrections_lookback_days: int = 30,
        bot_configs: dict | None = None,
        strategy_registry=None,
        run_index: object | None = None,
    ) -> None:
        self.date = date
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.corrections_lookback_days = corrections_lookback_days
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
        """Build the complete prompt package.

        Args:
            triage_report: Optional TriageReport from DailyTriage. When provided,
                instructions are focused on the triage's significant events and
                questions. When None, uses fallback instructions.
            session_store: Optional SessionStore for loading session history.
        """
        pkg = self._ctx.base_package(
            session_store=session_store,
            agent_type="daily_analysis",
            bot_configs=self.bot_configs,
            strategy_registry=self.strategy_registry,
            bot_id=self.bots[0] if len(self.bots) == 1 else "",
        )
        pkg.corrections = self._evidence.findings.load_corrections(
            bot_id=self.bots[0] if len(self.bots) == 1 else "",
            max_age_days=self.corrections_lookback_days,
            as_of=_end_of_report_day(self.date),
        )
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_structured_data(triage_report))
        pkg.instructions = self._build_instructions(triage_report)
        pkg.context_files.extend(self._list_data_files(triage_report))

        # Inject contradiction data if any
        contradictions = self._ctx.load_contradictions(self.date, self.bots, self.curated_dir)
        if contradictions:
            pkg.data["contradictions"] = contradictions

        pkg.metadata["bot_ids"] = ",".join(self.bots)
        pkg.metadata["date"] = self.date
        return pkg

    def _build_instructions(self, triage_report=None) -> str:
        """Build instructions from triage report or use fallback."""
        if triage_report is None:
            instructions = _INSTRUCTIONS
            if self.strategy_registry and self._has_crypto_strategies():
                instructions += _CRYPTO_DAILY_SUPPLEMENT
            return instructions

        # Format significant events
        event_lines = []
        for i, event in enumerate(triage_report.significant_events, 1):
            event_lines.append(
                f"{i}. **[{event.event_type.upper()}]** [{event.bot_id}] "
                f"(severity: {event.severity}) - {event.description}"
            )
        events_text = "\n".join(event_lines) if event_lines else "(No significant events detected - routine day)"

        # Format focus questions
        question_lines = []
        for i, q in enumerate(triage_report.focus_questions, 1):
            question_lines.append(f"{i}. {q}")
        questions_text = "\n".join(question_lines)

        instructions = _FOCUSED_INSTRUCTIONS.format(
            routine_summary=triage_report.routine_summary,
            significant_events=events_text,
            focus_questions=questions_text,
        )

        # Append crypto supplement if any bot has crypto perpetual strategies
        if self.strategy_registry and self._has_crypto_strategies():
            instructions += _CRYPTO_DAILY_SUPPLEMENT

        return instructions

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
        return (
            f"Analyze today's ({self.date}) trading performance for all bots: {bot_list}.\n"
            f"Focus on the significant events identified by triage. Reason about causes, not just symptoms."
        )

    def _load_structured_data(self, triage_report=None) -> dict:
        data: dict = {}
        data_load_errors: list[dict] = []
        date_dir = self.curated_dir / self.date
        findings_dir = self.memory_dir / "findings"

        # Determine which files to load based on triage
        files_to_load = _CURATED_FILES
        if triage_report and triage_report.relevant_data_keys:
            # Always load summary + triage-relevant files
            relevant = set(triage_report.relevant_data_keys)
            relevant.add("summary.json")
            files_to_load = [f for f in _CURATED_FILES if f in relevant]

        for bot in self.bots:
            bot_dir = date_dir / bot
            bot_data: dict = {}
            for filename in files_to_load:
                path = bot_dir / filename
                if path.exists():
                    key = filename.replace(".json", "")
                    loaded = self._safe_load_json(path, data_load_errors)
                    if loaded is not None:
                        bot_data[key] = loaded

            # Inject signal factor rolling trends per bot
            factor_trends = self._ctx.load_signal_factor_history(bot, self.date, findings_dir)
            if factor_trends:
                bot_data["signal_factor_trends"] = factor_trends

            data[bot] = bot_data

        risk_path = date_dir / "portfolio_risk_card.json"
        if risk_path.exists():
            loaded = self._safe_load_json(risk_path, data_load_errors)
            if loaded is not None:
                data["portfolio_risk_card"] = loaded

        # Load portfolio-level curated files
        portfolio_dir = date_dir / "portfolio"
        if portfolio_dir.is_dir():
            for filename in _PORTFOLIO_CURATED_FILES:
                path = portfolio_dir / filename
                if path.exists():
                    key = "portfolio_" + filename.replace(".json", "")
                    loaded = self._safe_load_json(path, data_load_errors)
                    if loaded is not None:
                        data[key] = loaded

        if data_load_errors:
            data["data_load_errors"] = data_load_errors
        return data

    @staticmethod
    def _safe_load_json(path: Path, errors: list[dict]) -> object | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed curated JSON %s: %s", path, exc)
            errors.append({"path": str(path), "error": str(exc)})
            return None

    def _list_data_files(self, triage_report=None) -> list[str]:
        files: list[str] = []
        date_dir = self.curated_dir / self.date

        files_to_list = _CURATED_FILES
        if triage_report and triage_report.relevant_data_keys:
            relevant = set(triage_report.relevant_data_keys)
            relevant.add("summary.json")
            files_to_list = [f for f in _CURATED_FILES if f in relevant]

        for bot in self.bots:
            bot_dir = date_dir / bot
            for filename in files_to_list:
                path = bot_dir / filename
                if path.exists() and self._json_file_loadable(path):
                    files.append(str(path))

        risk_path = date_dir / "portfolio_risk_card.json"
        if risk_path.exists() and self._json_file_loadable(risk_path):
            files.append(str(risk_path))

        # Portfolio-level curated files
        portfolio_dir = date_dir / "portfolio"
        if portfolio_dir.is_dir():
            for filename in _PORTFOLIO_CURATED_FILES:
                path = portfolio_dir / filename
                if path.exists() and self._json_file_loadable(path):
                    files.append(str(path))

        return files

    @staticmethod
    def _json_file_loadable(path: Path) -> bool:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return True


def _end_of_report_day(value: str) -> datetime:
    try:
        report_date = datetime.fromisoformat(value).date()
    except ValueError:
        return datetime.now(timezone.utc)
    return datetime.combine(report_date, time.max, tzinfo=timezone.utc)
