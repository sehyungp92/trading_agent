# analysis/strategy_engine.py
"""Strategy refinement engine - deterministic 4-tier suggestion generator.

Analyzes weekly metrics and produces strategy suggestions. All rules-based,
no LLM calls. The configured analysis provider interprets these in the weekly report prompt.

Tier 1 (Parameter): e.g. stop-loss too tight, threshold misaligned
Tier 2 (Filter): filter cost exceeds benefit over the week
Tier 3 (Strategy Variant): regime mismatch - suggest regime gate
Tier 4 (Hypothesis): reserved for the analysis runtime to synthesize in the weekly report
"""
from __future__ import annotations

from datetime import datetime

from trading_assistant.analysis.detectors.catalog import DEFAULT_DETECTOR_CATALOG
from trading_assistant.analysis.detectors.signal_decay import (
    evaluate_alpha_decay,
    evaluate_component_signal_decay,
    evaluate_factor_correlation_decay,
    evaluate_signal_decay,
)
from trading_assistant.analysis.detectors.time_of_day import evaluate_time_of_day_patterns
from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.events import normalize_strategy_id
from trading_assistant.schemas.regime_conditional import (
    RegimeConditionalReport,
)
from trading_assistant.schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
    RefinementReport,
)
from trading_assistant.schemas.weekly_metrics import (
    BotWeeklySummary,
    FilterWeeklySummary,
    RegimePerformanceTrend,
    StrategyWeeklySummary,
)

# Minimum trade count under which per-bot detectors produce statistical noise.
# Bots with fewer trades than this in a week are excluded from per-bot detector
# calls in build_report so a 1-2 trade sample doesn't generate actionable
# suggestions. Detectors that already enforce their own min_trades (e.g.
# detect_time_of_day_patterns, detect_grade_selectivity) keep their own gates.
_MIN_EVIDENCE_TRADES = 5


class StrategyEngine:
    """Deterministic strategy suggestion generator."""

    _DETECTOR_CATALOG = DEFAULT_DETECTOR_CATALOG
    _ARCHETYPE_DEFAULTS = _DETECTOR_CATALOG.archetype_defaults
    _DETECTOR_TO_CATEGORY = _DETECTOR_CATALOG.categories
    _DECREASE_KEYWORDS = _DETECTOR_CATALOG.decrease_keywords
    _INCREASE_KEYWORDS = _DETECTOR_CATALOG.increase_keywords

    def __init__(
        self,
        week_start: str,
        week_end: str,
        tight_stop_ratio: float = 0.3,
        filter_cost_threshold: float = 0.0,
        regime_loss_threshold: float = 0.0,
        regime_min_weeks: int = 3,
        threshold_learner: object | None = None,
        strategy_registry: object | None = None,
        category_scorecard: object | None = None,
        detector_confidence: dict[str, float] | None = None,
        recent_suggestions: list[dict] | None = None,
        convergence_report: dict | None = None,
        category_value_map: dict | None = None,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.tight_stop_ratio = tight_stop_ratio
        self.filter_cost_threshold = filter_cost_threshold
        self.regime_loss_threshold = regime_loss_threshold
        self.regime_min_weeks = regime_min_weeks
        self._threshold_learner = threshold_learner
        self._strategy_registry = strategy_registry
        self._category_scorecard = category_scorecard
        self._detector_confidence = detector_confidence or {}
        self._recent_suggestions = recent_suggestions or []
        self._convergence_report = convergence_report or {}
        self._category_value_map = category_value_map or {}

    def _get_threshold(
        self,
        detector_name: str,
        threshold_name: str,
        bot_id: str,
        default: float,
    ) -> float:
        """Return learned threshold if available, else default."""
        if self._threshold_learner is None:
            return default
        return self._threshold_learner.get_threshold(
            detector_name, threshold_name, bot_id, default,
        )

    def _archetype_default(self, strategy_id: str, detector: str, param: str) -> float | None:
        """Return archetype-specific default for a detector param, or None."""
        if not self._strategy_registry or not strategy_id:
            return None
        arch = self._strategy_registry.archetype_for_strategy(strategy_id)
        if not arch:
            return None
        arch_str = arch.value if hasattr(arch, "value") else str(arch)
        return self._ARCHETYPE_DEFAULTS.get(detector, {}).get(arch_str, {}).get(param)

    def _resolve_strategy_id(self, bot_id: str) -> str:
        """Resolve the primary strategy_id for a bot_id from registry."""
        if not self._strategy_registry:
            return ""
        strats = self._strategy_registry.strategies_for_bot(bot_id)
        return next(iter(strats)) if len(strats) == 1 else ""

    def _archetype_str(self, strategy_id: str) -> str:
        """Return archetype string for a strategy_id."""
        if not self._strategy_registry or not strategy_id:
            return ""
        arch = self._strategy_registry.archetype_for_strategy(strategy_id)
        return arch.value if arch and hasattr(arch, "value") else str(arch) if arch else ""

    def analyze_parameters(
        self, summary: BotWeeklySummary
    ) -> list[StrategySuggestion]:
        """Tier 1: Detect parameter misalignment from weekly stats."""
        suggestions: list[StrategySuggestion] = []

        # Tight stop detection: avg_loss is small relative to avg_win
        if summary.avg_win > 0 and summary.avg_loss != 0:
            loss_win_ratio = abs(summary.avg_loss) / summary.avg_win
            threshold = self._get_threshold(
                "tight_stop", "tight_stop_ratio", summary.bot_id,
                self.tight_stop_ratio,
            )
            if loss_win_ratio < threshold:
                suggestions.append(
                    StrategySuggestion(
                        tier=SuggestionTier.PARAMETER,
                        bot_id=summary.bot_id,
                        title=f"Stop loss may be too tight on {summary.bot_id}",
                        description=(
                            f"Avg loss (${abs(summary.avg_loss):.0f}) is only "
                            f"{loss_win_ratio:.0%} of avg win (${summary.avg_win:.0f}). "
                            f"Stops may be clipping winners too early. "
                            f"Consider widening stop by 0.5x ATR."
                        ),
                        current_value=f"loss/win_ratio={loss_win_ratio:.2f}",
                        suggested_value="loss/win_ratio>=0.3",
                        evidence_days=7,
                        confidence=0.7,
                        detection_context=DetectionContext(
                            detector_name="tight_stop",
                            bot_id=summary.bot_id,
                            threshold_name="tight_stop_ratio",
                            threshold_value=threshold,
                            observed_value=loss_win_ratio,
                        ),
                    )
                )

        return suggestions

    def analyze_filters(
        self, bot_id: str, filter_summaries: list[FilterWeeklySummary]
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect filters that cost more than they save."""
        suggestions: list[StrategySuggestion] = []

        threshold = self._get_threshold(
            "filter_cost", "filter_cost_threshold", bot_id,
            self.filter_cost_threshold,
        )
        for f in filter_summaries:
            if f.net_impact_pnl < threshold:
                suggestions.append(
                    StrategySuggestion(
                        tier=SuggestionTier.FILTER,
                        bot_id=bot_id,
                        title=f"Relax {f.filter_name} on {bot_id}",
                        description=(
                            f"{f.filter_name} blocked {f.total_blocks} entries this week. "
                            f"Net impact: ${f.net_impact_pnl:.0f} (cost exceeds benefit). "
                            f"Consider relaxing the threshold."
                        ),
                        evidence_days=7,
                        estimated_impact_pnl=abs(f.net_impact_pnl),
                        confidence=max(0.0, min(1.0, f.confidence)),
                        detection_context=DetectionContext(
                            detector_name="filter_cost",
                            bot_id=bot_id,
                            threshold_name="filter_cost_threshold",
                            threshold_value=threshold,
                            observed_value=f.net_impact_pnl,
                        ),
                    )
                )

        return suggestions

    def analyze_regime_fit(
        self, bot_id: str, regime_trends: list[RegimePerformanceTrend],
        trades: list | None = None,
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect consistent losses in a specific regime.

        If trades are provided, includes quantified exclusion impact in the description.
        """
        return self.analyze_regime_fit_quantified(bot_id, regime_trends, trades)

    def compute_regime_exclusion_impact(
        self, bot_id: str, trades: list, regime_to_exclude: str
    ) -> dict:
        """Compute P&L impact of excluding all trades in a specific regime."""
        baseline_pnl = sum(t.pnl for t in trades)
        kept = [t for t in trades if (t.market_regime or "unknown") != regime_to_exclude]
        excluded_pnl = sum(t.pnl for t in kept)
        excluded_count = len(trades) - len(kept)
        return {
            "regime": regime_to_exclude,
            "baseline_pnl": baseline_pnl,
            "excluded_pnl": excluded_pnl,
            "delta_pnl": excluded_pnl - baseline_pnl,
            "excluded_trade_count": excluded_count,
            "total_trade_count": len(trades),
        }

    def analyze_regime_fit_quantified(
        self, bot_id: str, regime_trends: list["RegimePerformanceTrend"],
        trades: list | None = None,
    ) -> list["StrategySuggestion"]:
        """Tier 3: Regime fit analysis with quantified exclusion impact."""
        suggestions: list[StrategySuggestion] = []

        effective_min_weeks = int(self._get_threshold(
            "regime_loss", "regime_min_weeks", bot_id,
            float(self.regime_min_weeks),
        ))
        effective_loss_threshold = self._get_threshold(
            "regime_loss", "regime_loss_threshold", bot_id,
            self.regime_loss_threshold,
        )

        for trend in regime_trends:
            if len(trend.weekly_pnl) < effective_min_weeks:
                continue
            losing_weeks = sum(1 for pnl in trend.weekly_pnl if pnl < effective_loss_threshold)
            if losing_weeks < effective_min_weeks:
                continue

            total_loss = sum(pnl for pnl in trend.weekly_pnl if pnl < 0)
            desc = (
                f"{bot_id} lost in {trend.regime} regime for "
                f"{losing_weeks}/{len(trend.weekly_pnl)} weeks "
                f"(total: ${total_loss:.0f}). "
            )

            if trades:
                impact = self.compute_regime_exclusion_impact(bot_id, trades, trend.regime)
                desc += (
                    f"Excluding {trend.regime} trades would change PnL from "
                    f"${impact['baseline_pnl']:.0f} to ${impact['excluded_pnl']:.0f} "
                    f"(+${impact['delta_pnl']:.0f}, removing {impact['excluded_trade_count']} trades). "
                )

            desc += f"Consider adding a regime gate to disable trading in {trend.regime} conditions."

            suggestions.append(
                StrategySuggestion(
                    tier=SuggestionTier.STRATEGY_VARIANT,
                    bot_id=bot_id,
                    title=f"Add regime gate for {trend.regime} on {bot_id}",
                    description=desc,
                    requires_human_judgment=True,
                    evidence_days=len(trend.weekly_pnl) * 7,
                    confidence=0.5,
                    estimated_impact_pnl=abs(total_loss),
                    detection_context=DetectionContext(
                        detector_name="regime_loss",
                        bot_id=bot_id,
                        threshold_name="regime_min_weeks",
                        threshold_value=float(effective_min_weeks),
                        observed_value=float(losing_weeks),
                    ),
                )
            )

        return suggestions

    def detect_alpha_decay(
        self,
        bot_id: str,
        rolling_sharpe_30d: float,
        rolling_sharpe_60d: float,
        rolling_sharpe_90d: float,
        decay_threshold: float = 0.3,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect declining Sharpe ratio over 30/60/90 day windows."""
        sid = strategy_id or self._resolve_strategy_id(bot_id)
        arch_default = self._archetype_default(sid, "alpha_decay", "decay_threshold")
        base_threshold = arch_default if arch_default is not None else decay_threshold
        effective_threshold = self._get_threshold(
            "alpha_decay", "decay_threshold", bot_id, base_threshold,
        )
        return evaluate_alpha_decay(
            bot_id=bot_id,
            strategy_id=sid,
            strategy_archetype=self._archetype_str(sid),
            rolling_sharpe_30d=rolling_sharpe_30d,
            rolling_sharpe_90d=rolling_sharpe_90d,
            decay_threshold=effective_threshold,
        )
    def detect_signal_decay(
        self,
        bot_id: str,
        signal_outcome_correlation_30d: float,
        signal_outcome_correlation_90d: float,
        decay_threshold: float = 0.2,
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect declining signal-to-outcome correlation."""
        effective_threshold = self._get_threshold(
            "signal_decay", "decay_threshold", bot_id, decay_threshold,
        )
        return evaluate_signal_decay(
            bot_id=bot_id,
            signal_outcome_correlation_30d=signal_outcome_correlation_30d,
            signal_outcome_correlation_90d=signal_outcome_correlation_90d,
            decay_threshold=effective_threshold,
        )
    def detect_exit_timing_issues(
        self,
        bot_id: str,
        avg_exit_efficiency: float,
        premature_exit_pct: float,
        efficiency_threshold: float = 0.5,
        premature_threshold: float = 0.4,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_exit_timing_issues as _detect_exit_timing_issues

        return _detect_exit_timing_issues(self, bot_id=bot_id, avg_exit_efficiency=avg_exit_efficiency, premature_exit_pct=premature_exit_pct, efficiency_threshold=efficiency_threshold, premature_threshold=premature_threshold, strategy_id=strategy_id)

    def detect_correlation_breakdown(
        self,
        correlations: list,  # list[CorrelationSummary]
        threshold: float = 0.7,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_correlation_breakdown as _detect_correlation_breakdown

        return _detect_correlation_breakdown(self, correlations=correlations, threshold=threshold)

    def detect_time_of_day_patterns(
        self,
        bot_id: str,
        hourly_buckets: list,  # list[HourlyBucket]
        min_trades: int = 10,
        loss_threshold: float = 0.35,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect hours with consistently poor performance."""
        sid = strategy_id or self._resolve_strategy_id(bot_id)
        arch_str = self._archetype_str(sid)
        effective_threshold = self._get_threshold(
            "time_of_day", "loss_threshold", bot_id, loss_threshold,
        )
        return evaluate_time_of_day_patterns(
            bot_id=bot_id,
            hourly_buckets=hourly_buckets,
            strategy_id=sid,
            strategy_archetype=arch_str,
            loss_threshold=effective_threshold,
            min_trades=min_trades,
        )

    def detect_drawdown_patterns(
        self,
        bot_id: str,
        largest_single_loss_pct: float,
        max_drawdown_pct: float,
        avg_loss_pct: float,
        concentration_threshold: float = 3.0,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_drawdown_patterns as _detect_drawdown_patterns

        return _detect_drawdown_patterns(self, bot_id=bot_id, largest_single_loss_pct=largest_single_loss_pct, max_drawdown_pct=max_drawdown_pct, avg_loss_pct=avg_loss_pct, concentration_threshold=concentration_threshold)

    def detect_position_sizing_issues(
        self,
        bot_id: str,
        avg_win_pct: float,
        avg_loss_pct: float,
        win_rate: float,
        loss_win_ratio_threshold: float = 1.5,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_position_sizing_issues as _detect_position_sizing_issues

        return _detect_position_sizing_issues(self, bot_id=bot_id, avg_win_pct=avg_win_pct, avg_loss_pct=avg_loss_pct, win_rate=win_rate, loss_win_ratio_threshold=loss_win_ratio_threshold)

    def detect_component_signal_decay(
        self,
        bot_id: str,
        signal_health_data: dict,
        stability_threshold: float = 0.3,
        correlation_threshold: float = 0.05,
        min_trades: int = 5,
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect degraded signal components from signal_health data."""
        effective_stability = self._get_threshold(
            "component_signal_decay", "stability_threshold", bot_id,
            stability_threshold,
        )
        effective_correlation = self._get_threshold(
            "component_signal_decay", "correlation_threshold", bot_id,
            correlation_threshold,
        )
        return evaluate_component_signal_decay(
            bot_id=bot_id,
            signal_health_data=signal_health_data,
            stability_threshold=effective_stability,
            correlation_threshold=effective_correlation,
            min_trades=min_trades,
        )

    def detect_filter_interactions(
        self,
        bot_id: str,
        filter_interactions: list,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_filter_interactions as _detect_filter_interactions

        return _detect_filter_interactions(self, bot_id=bot_id, filter_interactions=filter_interactions)

    def detect_factor_correlation_decay(
        self,
        bot_id: str,
        factor_rolling_data: list[dict],
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect degrading signal factors from rolling 30d analysis.

        Produces HYPOTHESIS suggestions for factors with degrading trend + below_threshold.
        """
        return evaluate_factor_correlation_decay(
            bot_id=bot_id,
            factor_rolling_data=factor_rolling_data,
        )

    def compute_regime_conditional_metrics(
        self,
        per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
        trades_by_bot: dict[str, list],
    ) -> RegimeConditionalReport:
        from trading_assistant.analysis.detectors.regime import compute_regime_conditional_metrics as _compute_regime_conditional_metrics

        return _compute_regime_conditional_metrics(self, per_strategy_summaries=per_strategy_summaries, trades_by_bot=trades_by_bot)

    def detect_microstructure_issues(
        self,
        bot_id: str,
        orderbook_stats: dict,
        spread_threshold_bps: float = 5.0,
        imbalance_threshold: float = 2.0,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_microstructure_issues as _detect_microstructure_issues

        return _detect_microstructure_issues(self, bot_id=bot_id, orderbook_stats=orderbook_stats, spread_threshold_bps=spread_threshold_bps, imbalance_threshold=imbalance_threshold)

    # Macro regime detectors

    def detect_regime_config_effectiveness(
        self,
        bot_id: str,
        macro_regime: str,
        regime_unit_risk_mult: float,
        regime_pnl: float,
        regime_win_rate: float,
        regime_trade_count: int,
        min_trades: int = 10,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.regime import detect_regime_config_effectiveness as _detect_regime_config_effectiveness

        return _detect_regime_config_effectiveness(self, bot_id=bot_id, macro_regime=macro_regime, regime_unit_risk_mult=regime_unit_risk_mult, regime_pnl=regime_pnl, regime_win_rate=regime_win_rate, regime_trade_count=regime_trade_count, min_trades=min_trades, strategy_id=strategy_id)

    def detect_regime_transition_cost(
        self,
        transition_events: list[dict],
        daily_pnl_by_date: dict[str, float],
        window_days: int = 5,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.regime import detect_regime_transition_cost as _detect_regime_transition_cost

        return _detect_regime_transition_cost(self, transition_events=transition_events, daily_pnl_by_date=daily_pnl_by_date, window_days=window_days)

    def detect_stress_entry_pattern(
        self,
        bot_id: str,
        trades_by_stress: dict[str, dict],
        min_trades_per_bucket: int = 5,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.regime import detect_stress_entry_pattern as _detect_stress_entry_pattern

        return _detect_stress_entry_pattern(self, bot_id=bot_id, trades_by_stress=trades_by_stress, min_trades_per_bucket=min_trades_per_bucket, strategy_id=strategy_id)

    def _infer_direction(self, suggestion: StrategySuggestion) -> int:
        """Infer change direction: +1 (increase), -1 (decrease), 0 (unknown).

        Tries numeric comparison first, falls back to keyword analysis.
        """
        # Try numeric comparison: suggested_value vs current_value
        try:
            sv = suggestion.suggested_value
            cv = suggestion.current_value
            if sv and cv:
                sv_f = float(str(sv).split("=")[-1].split(">")[0].split("<")[0].strip())
                cv_f = float(str(cv).split("=")[-1].split(">")[0].split("<")[0].strip())
                if sv_f > cv_f:
                    return 1
                elif sv_f < cv_f:
                    return -1
        except (ValueError, TypeError, IndexError, AttributeError):
            pass

        # Fall back to keyword analysis on title + description
        text = (suggestion.title + " " + suggestion.description).lower()
        for kw in self._INCREASE_KEYWORDS:
            if kw in text:
                return 1
        for kw in self._DECREASE_KEYWORDS:
            if kw in text:
                return -1
        return 0

    def _infer_direction_from_dict(self, rec: dict) -> int:
        """Infer direction from a persisted suggestion dict."""
        # Try proposed_value vs detection_context.threshold_value
        pv = rec.get("proposed_value")
        ctx = rec.get("detection_context") or {}
        cv = ctx.get("threshold_value") or ctx.get("observed_value")
        if pv is not None and cv is not None:
            try:
                if float(pv) > float(cv):
                    return 1
                elif float(pv) < float(cv):
                    return -1
            except (ValueError, TypeError):
                pass

        text = (rec.get("title", "") + " " + rec.get("description", "")).lower()
        for kw in self._INCREASE_KEYWORDS:
            if kw in text:
                return 1
        for kw in self._DECREASE_KEYWORDS:
            if kw in text:
                return -1
        return 0

    def _contradicts_recent(
        self, bot_id: str, detector_name: str, direction: int,
    ) -> bool:
        """Check if a recent suggestion from same detector+bot had opposite direction."""
        if direction == 0 or not self._recent_suggestions:
            return False
        for rec in self._recent_suggestions:
            if rec.get("bot_id") != bot_id:
                continue
            ctx = rec.get("detection_context") or {}
            rec_detector = ctx.get("detector_name", "")
            if rec_detector != detector_name:
                continue
            rec_direction = self._infer_direction_from_dict(rec)
            if rec_direction != 0 and rec_direction != direction:
                return True
        return False

    def detect_execution_bottleneck(
        self,
        bot_id: str,
        latency_stats: dict,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_execution_bottleneck as _detect_execution_bottleneck

        return _detect_execution_bottleneck(self, bot_id=bot_id, latency_stats=latency_stats, strategy_id=strategy_id)

    def detect_sizing_methodology(
        self,
        bot_id: str,
        sizing_data: dict,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_sizing_methodology as _detect_sizing_methodology

        return _detect_sizing_methodology(self, bot_id=bot_id, sizing_data=sizing_data, strategy_id=strategy_id)

    def detect_portfolio_crowding(
        self,
        bot_id: str,
        portfolio_context: dict,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_portfolio_crowding as _detect_portfolio_crowding

        return _detect_portfolio_crowding(self, bot_id=bot_id, portfolio_context=portfolio_context, strategy_id=strategy_id)

    def detect_better_exit_strategies(
        self,
        bot_id: str,
        exit_sweep: dict,
        edge_threshold_pct: float = 0.10,
        min_trades: int = 20,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_better_exit_strategies as _detect_better_exit_strategies

        return _detect_better_exit_strategies(self, bot_id=bot_id, exit_sweep=exit_sweep, edge_threshold_pct=edge_threshold_pct, min_trades=min_trades)

    def detect_filter_sensitivity_findings(
        self,
        bot_id: str,
        sensitivity: dict,
        min_blocks: int = 5,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_filter_sensitivity_findings as _detect_filter_sensitivity_findings

        return _detect_filter_sensitivity_findings(self, bot_id=bot_id, sensitivity=sensitivity, min_blocks=min_blocks)

    def detect_counterfactual_gaps(
        self,
        bot_id: str,
        counterfactual: dict,
        gain_threshold_pct: float = 0.10,
        min_trades: int = 20,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.execution_quality import detect_counterfactual_gaps as _detect_counterfactual_gaps

        return _detect_counterfactual_gaps(self, bot_id=bot_id, counterfactual=counterfactual, gain_threshold_pct=gain_threshold_pct, min_trades=min_trades)

    def build_report(
        self,
        bot_summaries: dict[str, BotWeeklySummary],
        filter_summaries: dict[str, list[FilterWeeklySummary]] | None = None,
        regime_trends: dict[str, list[RegimePerformanceTrend]] | None = None,
        rolling_sharpe: dict[str, dict[str, float]] | None = None,
        signal_correlations: dict[str, dict[str, float]] | None = None,
        hourly_buckets: dict[str, list] | None = None,
        correlation_summaries: list | None = None,
        drawdown_data: dict[str, dict] | None = None,
        signal_health: dict[str, dict] | None = None,
        factor_rolling: dict[str, list[dict]] | None = None,
        filter_interactions: dict[str, list] | None = None,
        orderbook_stats: dict[str, dict] | None = None,
        macro_regime_data: dict | None = None,
        regime_transition_events: list[dict] | None = None,
        daily_pnl_by_date: dict[str, float] | None = None,
        stress_entry_stats: dict[str, dict[str, dict]] | None = None,
        exit_efficiency_data: dict[str, dict] | None = None,
        execution_latency: dict[str, dict] | None = None,
        sizing_data: dict[str, dict] | None = None,
        portfolio_context: dict[str, dict] | None = None,
        param_correlations: dict[str, dict] | None = None,
        funding_data: dict[str, dict] | None = None,
        grade_data: dict[str, dict] | None = None,
        confluence_data: dict[str, dict] | None = None,
        leverage_data: dict[str, dict] | None = None,
        crypto_trade_data: dict[str, list] | None = None,
        exit_sweep: dict[str, dict] | None = None,
        filter_sensitivity: dict[str, dict] | None = None,
        counterfactual: dict[str, dict] | None = None,
    ) -> RefinementReport:
        """Build the complete refinement report across all bots."""
        all_suggestions: list[StrategySuggestion] = []

        # Bots with too few trades this window are excluded from per-bot detector
        # calls; their statistical signals are too noisy to produce actionable
        # suggestions. Detectors with their own min_trades param are unaffected
        # when invoked outside build_report.
        low_evidence_bots: set[str] = {
            bid for bid, s in bot_summaries.items()
            if s.total_trades < _MIN_EVIDENCE_TRADES
        }

        for bot_id, summary in bot_summaries.items():
            if bot_id in low_evidence_bots:
                continue
            all_suggestions.extend(self.analyze_parameters(summary))

            if filter_summaries and bot_id in filter_summaries:
                all_suggestions.extend(
                    self.analyze_filters(bot_id, filter_summaries[bot_id])
                )

            if regime_trends and bot_id in regime_trends:
                all_suggestions.extend(
                    self.analyze_regime_fit(bot_id, regime_trends[bot_id])
                )

        # New detectors
        if rolling_sharpe:
            for bot_id, sharpe in rolling_sharpe.items():
                all_suggestions.extend(self.detect_alpha_decay(
                    bot_id, sharpe.get("30d", 0), sharpe.get("60d", 0), sharpe.get("90d", 0),
                ))

        if signal_correlations:
            for bot_id, corr in signal_correlations.items():
                all_suggestions.extend(self.detect_signal_decay(
                    bot_id, corr.get("30d", 0), corr.get("90d", 0),
                ))

        if hourly_buckets:
            for bot_id, buckets in hourly_buckets.items():
                all_suggestions.extend(self.detect_time_of_day_patterns(bot_id, buckets))

        if correlation_summaries:
            all_suggestions.extend(self.detect_correlation_breakdown(correlation_summaries))

        if drawdown_data:
            for bot_id, dd in drawdown_data.items():
                all_suggestions.extend(self.detect_drawdown_patterns(
                    bot_id,
                    dd.get("largest_single_loss_pct", 0),
                    dd.get("max_drawdown_pct", 0),
                    dd.get("avg_loss_pct", 0),
                ))

        for bot_id, summary in bot_summaries.items():
            if bot_id in low_evidence_bots:
                continue
            if summary.avg_win > 0 and abs(summary.avg_loss) > 0:
                all_suggestions.extend(self.detect_position_sizing_issues(
                    bot_id,
                    avg_win_pct=summary.avg_win,
                    avg_loss_pct=abs(summary.avg_loss),
                    win_rate=summary.win_rate,
                ))

        if signal_health:
            for bot_id, sh_data in signal_health.items():
                all_suggestions.extend(
                    self.detect_component_signal_decay(bot_id, sh_data)
                )

        if factor_rolling:
            for bot_id, factors in factor_rolling.items():
                all_suggestions.extend(
                    self.detect_factor_correlation_decay(bot_id, factors)
                )

        if filter_interactions:
            for bot_id, interactions in filter_interactions.items():
                all_suggestions.extend(
                    self.detect_filter_interactions(bot_id, interactions)
                )

        if orderbook_stats:
            for bot_id, ob_data in orderbook_stats.items():
                all_suggestions.extend(
                    self.detect_microstructure_issues(bot_id, ob_data)
                )

        # Macro regime detectors
        if macro_regime_data:
            macro_regime = macro_regime_data.get("macro_regime", "")
            per_bot_configs = macro_regime_data.get("per_bot_configs", {})
            for bot_id, config in per_bot_configs.items():
                mult = config.get("regime_unit_risk_mult", 1.0)
                if mult < 1.0 and macro_regime:
                    # Per-bot config may carry regime-isolated metrics; fall
                    # back to blended weekly summary (acceptable since macro
                    # regimes persist for years - single-regime weeks are the
                    # common case).
                    r_pnl = config.get("regime_pnl")
                    r_wr = config.get("regime_win_rate")
                    r_tc = config.get("regime_trade_count")
                    summary = bot_summaries.get(bot_id)
                    if r_pnl is None and summary and summary.total_trades > 0:
                        r_pnl = summary.net_pnl
                        r_wr = summary.win_rate
                        r_tc = summary.total_trades
                    if r_pnl is not None and r_tc:
                        all_suggestions.extend(
                            self.detect_regime_config_effectiveness(
                                bot_id=bot_id,
                                macro_regime=macro_regime,
                                regime_unit_risk_mult=mult,
                                regime_pnl=r_pnl,
                                regime_win_rate=r_wr,
                                regime_trade_count=r_tc,
                            )
                        )

        if regime_transition_events:
            all_suggestions.extend(
                self.detect_regime_transition_cost(
                    transition_events=regime_transition_events,
                    daily_pnl_by_date=daily_pnl_by_date or {},
                )
            )

        if stress_entry_stats:
            for bot_id, stress_data in stress_entry_stats.items():
                all_suggestions.extend(
                    self.detect_stress_entry_pattern(bot_id, stress_data)
                )

        if exit_efficiency_data:
            for bot_id, data in exit_efficiency_data.items():
                all_suggestions.extend(
                    self.detect_exit_timing_issues(
                        bot_id,
                        data.get("avg_exit_efficiency", 1.0),
                        data.get("premature_exit_pct", 0.0),
                    )
                )

        if execution_latency:
            for bot_id, stats in execution_latency.items():
                all_suggestions.extend(
                    self.detect_execution_bottleneck(bot_id, stats)
                )

        if sizing_data:
            for bot_id, data in sizing_data.items():
                all_suggestions.extend(
                    self.detect_sizing_methodology(bot_id, data)
                )

        if portfolio_context:
            for bot_id, ctx in portfolio_context.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(
                    self.detect_portfolio_crowding(bot_id, ctx)
                )

        # Crypto perpetual detectors
        if funding_data:
            for bot_id, summary in funding_data.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_funding_impact(bot_id, summary))
        if grade_data:
            for bot_id, summary in grade_data.items():
                all_suggestions.extend(self.detect_grade_selectivity(bot_id, summary))
        if confluence_data:
            for bot_id, summary in confluence_data.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_confluence_quality(bot_id, summary))
        if leverage_data:
            for bot_id, summary in leverage_data.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_leverage_utilization(bot_id, summary))
        if crypto_trade_data:
            for bot_id, trades in crypto_trade_data.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_mtf_alignment_drift(bot_id, trades))
                all_suggestions.extend(self.detect_liquidation_proximity(bot_id, trades))
                all_suggestions.extend(self.detect_symbol_concentration(bot_id, trades))
                all_suggestions.extend(self.detect_session_patterns_24_7(bot_id, trades))
                all_suggestions.extend(self.detect_funding_trend(bot_id, trades))

        # Sim-driven detectors (weekly handler feeds these from filter_sensitivity_analyzer,
        # counterfactual_simulator, exit_strategy_simulator)
        if exit_sweep:
            for bot_id, sweep in exit_sweep.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_better_exit_strategies(bot_id, sweep))
        if filter_sensitivity:
            for bot_id, sens in filter_sensitivity.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_filter_sensitivity_findings(bot_id, sens))
        if counterfactual:
            for bot_id, cf in counterfactual.items():
                if bot_id in low_evidence_bots:
                    continue
                all_suggestions.extend(self.detect_counterfactual_gaps(bot_id, cf))

        # Apply per-detector confidence calibration from outcome data
        if self._detector_confidence:
            calibrated: list[StrategySuggestion] = []
            for s in all_suggestions:
                det_name = ""
                if s.detection_context:
                    det_name = s.detection_context.detector_name
                multiplier = self._detector_confidence.get(det_name, 1.0)
                if multiplier != 1.0 and det_name:
                    adjusted_conf = round(s.confidence * multiplier, 3)
                    s = s.model_copy(update={"confidence": adjusted_conf})
                calibrated.append(s)
            all_suggestions = calibrated

        # Anti-oscillation: filter out suggestions that contradict recent ones
        if self._recent_suggestions:
            filtered: list[StrategySuggestion] = []
            for s in all_suggestions:
                det_name = ""
                if s.detection_context:
                    det_name = s.detection_context.detector_name
                direction = self._infer_direction(s)
                if det_name and direction != 0 and self._contradicts_recent(
                    s.bot_id, det_name, direction,
                ):
                    continue  # Skip contradictory suggestion
                filtered.append(s)
            all_suggestions = filtered

        # If convergence report shows oscillation, dampen all confidence
        if self._convergence_report.get("oscillation_detected"):
            all_suggestions = [
                s.model_copy(update={"confidence": round(s.confidence * 0.7, 3)})
                for s in all_suggestions
            ]

        # Optimization allocation: adjust confidence based on category value
        if self._category_value_map:
            adjusted: list[StrategySuggestion] = []
            for s in all_suggestions:
                det_name_for_cat = ""
                if s.detection_context:
                    det_name_for_cat = s.detection_context.detector_name
                cat = self._DETECTOR_TO_CATEGORY.get(det_name_for_cat, "")
                key = f"{s.bot_id}:{cat}" if cat else ""
                entry = self._category_value_map.get(key, {}) if key else {}
                vps = entry.get("value_per_suggestion") if entry else None
                if entry.get("unexplored"):
                    pass  # neutral treatment - don't penalize unexplored categories
                elif vps is not None and vps != 0:
                    # Scale factor proportionally, clamped to +-10%
                    raw_adj = max(-0.1, min(0.1, vps * 0.5))
                    factor = 1.0 + raw_adj
                    s = s.model_copy(update={
                        "confidence": round(s.confidence * factor, 3),
                    })
                adjusted.append(s)
            all_suggestions = adjusted

        # Suppress suggestions for categories with proven poor track records
        if self._category_scorecard:
            all_suggestions = [
                s for s in all_suggestions
                if not self._should_suppress(s.bot_id, s.tier.value)
            ]

        return RefinementReport(
            week_start=self.week_start,
            week_end=self.week_end,
            suggestions=all_suggestions,
        )

    # --- Crypto perpetual detectors ---

    def detect_funding_impact(
        self, bot_id: str, funding_summary: dict,
        cost_threshold: float = 0.15,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_funding_impact as _detect_funding_impact

        return _detect_funding_impact(self, bot_id=bot_id, funding_summary=funding_summary, cost_threshold=cost_threshold)

    def detect_grade_selectivity(
        self, bot_id: str, grade_summary: dict,
        min_trades: int = 20,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_grade_selectivity as _detect_grade_selectivity

        return _detect_grade_selectivity(self, bot_id=bot_id, grade_summary=grade_summary, min_trades=min_trades)

    def detect_confluence_quality(
        self, bot_id: str, confluence_summary: dict,
        lift_threshold: float = 0.10,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_confluence_quality as _detect_confluence_quality

        return _detect_confluence_quality(self, bot_id=bot_id, confluence_summary=confluence_summary, lift_threshold=lift_threshold)

    def detect_leverage_utilization(
        self, bot_id: str, leverage_summary: dict,
        utilization_warning: float = 0.80,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_leverage_utilization as _detect_leverage_utilization

        return _detect_leverage_utilization(self, bot_id=bot_id, leverage_summary=leverage_summary, utilization_warning=utilization_warning)

    def detect_mtf_alignment_drift(
        self,
        bot_id: str,
        trades: list,
        min_mismatched: int = 5,
        win_rate_gap_threshold: float = 0.15,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_mtf_alignment_drift as _detect_mtf_alignment_drift

        return _detect_mtf_alignment_drift(self, bot_id=bot_id, trades=trades, min_mismatched=min_mismatched, win_rate_gap_threshold=win_rate_gap_threshold)

    def detect_liquidation_proximity(
        self,
        bot_id: str,
        trades: list,
        proximity_threshold: float = 0.70,
        systemic_count: int = 3,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_liquidation_proximity as _detect_liquidation_proximity

        return _detect_liquidation_proximity(self, bot_id=bot_id, trades=trades, proximity_threshold=proximity_threshold, systemic_count=systemic_count)

    def detect_symbol_concentration(
        self,
        bot_id: str,
        trades: list,
        concentration_threshold: float = 0.70,
        min_trades: int = 10,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_symbol_concentration as _detect_symbol_concentration

        return _detect_symbol_concentration(self, bot_id=bot_id, trades=trades, concentration_threshold=concentration_threshold, min_trades=min_trades)

    def detect_session_patterns_24_7(
        self,
        bot_id: str,
        trades: list,
        min_trades: int = 10,
        negative_avg_pnl_threshold: float = 0.0,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_session_patterns_24_7 as _detect_session_patterns_24_7

        return _detect_session_patterns_24_7(self, bot_id=bot_id, trades=trades, min_trades=min_trades, negative_avg_pnl_threshold=negative_avg_pnl_threshold)

    def detect_funding_trend(
        self,
        bot_id: str,
        trades: list,
        cost_threshold: float = 0.15,
        rising_weeks: int = 3,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.crypto import detect_funding_trend as _detect_funding_trend

        return _detect_funding_trend(self, bot_id=bot_id, trades=trades, cost_threshold=cost_threshold, rising_weeks=rising_weeks)

    @staticmethod
    def _trade_get(trade: object, key: str, default: object = None) -> object:
        if isinstance(trade, dict):
            return trade.get(key, default)
        return getattr(trade, key, default)

    @classmethod
    def _trade_float(cls, trade: object, key: str, default: float = 0.0) -> float:
        value = cls._trade_get(trade, key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _trade_leverage(cls, trade: object) -> float:
        sizing = cls._trade_get(trade, "sizing_inputs", None)
        if isinstance(sizing, dict):
            try:
                return float(sizing.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                return 1.0
        return 1.0

    @classmethod
    def _trade_timestamp(cls, trade: object) -> datetime | None:
        raw = cls._trade_get(trade, "exit_time", None) or cls._trade_get(trade, "entry_time", None)
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str) and raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_trade_side(raw: object) -> str:
        value = str(raw or "").strip().lower()
        if value in {"long", "buy", "bull", "bullish"}:
            return "long"
        if value in {"short", "sell", "bear", "bearish"}:
            return "short"
        return ""

    @staticmethod
    def _normalize_bias_direction(raw: object) -> str:
        value = str(raw or "").strip().lower()
        if value in {"long", "bull", "bullish", "up", "trend_up"}:
            return "long"
        if value in {"short", "bear", "bearish", "down", "trend_down"}:
            return "short"
        return ""

    @classmethod
    def _win_rate(cls, trades: list) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for trade in trades if cls._trade_float(trade, "pnl", 0.0) > 0)
        return wins / len(trades)

    @classmethod
    def _avg_pnl(cls, trades: list) -> float:
        if not trades:
            return 0.0
        return sum(cls._trade_float(trade, "pnl", 0.0) for trade in trades) / len(trades)

    @staticmethod
    def _base_symbol(raw: object) -> str:
        symbol = str(raw or "").upper()
        for suffix in ("USDT", "USD", "-PERP", "PERP", "/USD", "/USDT"):
            symbol = symbol.replace(suffix, "")
        return "".join(ch for ch in symbol if ch.isalpha())[:3]

    def _strategy_id_from_trades(self, bot_id: str, trades: list) -> str:
        seen = {
            normalize_strategy_id(bot_id, self._trade_get(trade, "strategy_id", "") or "")
            for trade in trades
            if self._trade_get(trade, "strategy_id", "")
        }
        if len(seen) == 1:
            return next(iter(seen))
        return self._resolve_strategy_id(bot_id)

    def _should_suppress(self, bot_id: str, tier_value: str) -> bool:
        """Check if a (bot_id, tier) pair should be suppressed due to poor track record.

        Maps scorecard categories back to suggestion tiers via CATEGORY_TO_TIER,
        then suppresses when sample_size >= 5 AND win_rate < 0.3 AND avg_pnl_delta < 0.
        """
        if not self._category_scorecard:
            return False
        scores = getattr(self._category_scorecard, "scores", None)
        if not scores:
            return False
        from trading_assistant.schemas.agent_response import CATEGORY_TO_TIER
        for score in scores:
            if score.bot_id != bot_id:
                continue
            mapped_tier = CATEGORY_TO_TIER.get(score.category, score.category)
            if mapped_tier != tier_value:
                continue
            if score.sample_size >= 5 and score.win_rate < 0.3 and score.avg_pnl_delta < 0:
                return True
        return False

    # Portfolio-level detectors (Phase 2)

    def detect_family_imbalance(
        self,
        family_summaries: dict[str, dict],
        family_allocations: dict[str, float],
        min_days: int = 30,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_family_imbalance as _detect_family_imbalance

        return _detect_family_imbalance(self, family_summaries=family_summaries, family_allocations=family_allocations, min_days=min_days)

    def detect_correlation_concentration(
        self,
        correlation_matrix: dict[str, float],
        current_allocations: dict[str, float],
        threshold: float = 0.7,
        weight_threshold: float = 0.4,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_correlation_concentration as _detect_correlation_concentration

        return _detect_correlation_concentration(self, correlation_matrix=correlation_matrix, current_allocations=current_allocations, threshold=threshold, weight_threshold=weight_threshold)

    def detect_drawdown_tier_miscalibration(
        self,
        historical_drawdowns: list[float],
        current_tiers: list[list[float]],
        min_days: int = 90,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_drawdown_tier_miscalibration as _detect_drawdown_tier_miscalibration

        return _detect_drawdown_tier_miscalibration(self, historical_drawdowns=historical_drawdowns, current_tiers=current_tiers, min_days=min_days)

    def detect_coordination_gaps(
        self,
        concurrent_positions: dict,
        existing_coordination: dict | None = None,
        min_co_occurrences: int = 50,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_coordination_gaps as _detect_coordination_gaps

        return _detect_coordination_gaps(self, concurrent_positions=concurrent_positions, existing_coordination=existing_coordination, min_co_occurrences=min_co_occurrences)

    def detect_heat_cap_utilization(
        self,
        daily_heat_series: list[float],
        heat_cap_R: float,
        min_days: int = 30,
    ) -> list[StrategySuggestion]:
        from trading_assistant.analysis.detectors.portfolio import detect_heat_cap_utilization as _detect_heat_cap_utilization

        return _detect_heat_cap_utilization(self, daily_heat_series=daily_heat_series, heat_cap_R=heat_cap_R, min_days=min_days)
