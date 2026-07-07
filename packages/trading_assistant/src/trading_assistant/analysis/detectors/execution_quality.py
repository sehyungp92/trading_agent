"""Execution Quality detector behavior."""

from __future__ import annotations


from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
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
    """Tier 3: Detect systematic premature exits."""
    sid = strategy_id or self._resolve_strategy_id(bot_id)
    arch_default = self._archetype_default(sid, "exit_timing", "efficiency_threshold")
    base_efficiency = arch_default if arch_default is not None else efficiency_threshold
    effective_efficiency = self._get_threshold(
        "exit_timing", "efficiency_threshold", bot_id, base_efficiency,
    )
    effective_premature = self._get_threshold(
        "exit_timing", "premature_threshold", bot_id, premature_threshold,
    )
    if avg_exit_efficiency >= effective_efficiency and premature_exit_pct <= effective_premature:
        return []
    suggestions: list[StrategySuggestion] = []
    arch_str = self._archetype_str(sid)
    if avg_exit_efficiency < effective_efficiency:
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.STRATEGY_VARIANT,
            bot_id=bot_id,
            strategy_id=sid,
            strategy_archetype=arch_str,
            title=f"Premature exits -{bot_id}",
            description=(
                f"Average exit efficiency is {avg_exit_efficiency:.0%} (captures "
                f"{avg_exit_efficiency:.0%} of available move). "
                f"{premature_exit_pct:.0%} of exits are premature. "
                f"Consider trailing stop or wider take-profit."
            ),
            evidence_days=30,
            confidence=0.6,
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="exit_timing",
                bot_id=bot_id,
                threshold_name="efficiency_threshold",
                threshold_value=effective_efficiency,
                observed_value=avg_exit_efficiency,
            ),
        ))
    return suggestions


def detect_filter_interactions(
    self,
    bot_id: str,
    filter_interactions: list,
) -> list[StrategySuggestion]:
    """Tier 2: Generate suggestions from filter interaction analysis.

    Args:
        bot_id: Bot identifier.
        filter_interactions: List of FilterPairInteraction dicts or objects.
    """
    effective_redundancy = self._get_threshold(
        "filter_interactions", "redundancy_threshold", bot_id, 0.5,
    )
    suggestions: list[StrategySuggestion] = []

    for pair in filter_interactions:
        itype = pair.get("interaction_type", "") if isinstance(pair, dict) else getattr(pair, "interaction_type", "")
        if itype == "independent":
            continue

        if isinstance(pair, dict):
            fa = pair.get("filter_a", "")
            fb = pair.get("filter_b", "")
            rec = pair.get("recommendation", "")
            redundancy = pair.get("redundancy_score", 0.0)
        else:
            fa = getattr(pair, "filter_a", "")
            fb = getattr(pair, "filter_b", "")
            rec = getattr(pair, "recommendation", "")
            redundancy = getattr(pair, "redundancy_score", 0.0)

        if itype == "redundant":
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.FILTER,
                bot_id=bot_id,
                title=f"Redundant filters: {fa} + {fb} on {bot_id}",
                description=(
                    f"Filters {fa} and {fb} are redundant "
                    f"(overlap score: {redundancy:.0%}). {rec}"
                ),
                evidence_days=7,
                confidence=min(0.9, redundancy),
                requires_human_judgment=True,
                detection_context=DetectionContext(
                    detector_name="filter_interactions",
                    bot_id=bot_id,
                    threshold_name="redundancy_threshold",
                    threshold_value=effective_redundancy,
                    observed_value=redundancy,
                ),
            ))
        elif itype == "complementary":
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.FILTER,
                bot_id=bot_id,
                title=f"Complementary filters: {fa} + {fb} on {bot_id}",
                description=(
                    f"Filters {fa} and {fb} are complementary. {rec}"
                ),
                evidence_days=7,
                confidence=0.5,
            ))

    return suggestions


def detect_microstructure_issues(
    self,
    bot_id: str,
    orderbook_stats: dict,
    spread_threshold_bps: float = 5.0,
    imbalance_threshold: float = 2.0,
) -> list[StrategySuggestion]:
    """Tier 2: Detect adverse microstructure conditions at entry/exit."""
    effective_spread = self._get_threshold(
        "microstructure", "spread_threshold_bps", bot_id, spread_threshold_bps,
    )
    effective_imbalance = self._get_threshold(
        "microstructure", "imbalance_threshold", bot_id, imbalance_threshold,
    )
    suggestions: list[StrategySuggestion] = []

    by_context = orderbook_stats.get("by_context", {})
    entry_data = by_context.get("entry", {})
    if not entry_data:
        return []

    entry_spread = entry_data.get("spread_stats", {}).get("mean", 0)
    entry_imbalance = entry_data.get("imbalance_stats", {}).get("mean", 1.0)
    entry_count = entry_data.get("count", 0)

    if entry_count < 5:
        return []

    if entry_spread > effective_spread:
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.FILTER,
            bot_id=bot_id,
            title=f"Wide spreads at entry -{bot_id}",
            description=(
                f"Average spread at entry is {entry_spread:.1f} bps "
                f"(threshold: {effective_spread:.1f} bps) across {entry_count} entries. "
                f"Consider adding a spread-width gate or preferring limit orders."
            ),
            evidence_days=7,
            confidence=0.6,
            detection_context=DetectionContext(
                detector_name="microstructure",
                bot_id=bot_id,
                threshold_name="spread_threshold_bps",
                threshold_value=effective_spread,
                observed_value=entry_spread,
            ),
        ))

    if entry_imbalance > effective_imbalance or (entry_imbalance > 0 and entry_imbalance < 1.0 / effective_imbalance):
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.FILTER,
            bot_id=bot_id,
            title=f"Order book imbalance at entry -{bot_id}",
            description=(
                f"Average bid/ask imbalance at entry is {entry_imbalance:.2f} "
                f"across {entry_count} entries. Values far from 1.0 suggest "
                f"positioning against order flow. Review trade direction vs "
                f"book pressure for adverse selection."
            ),
            evidence_days=7,
            confidence=0.5,
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="microstructure",
                bot_id=bot_id,
                threshold_name="imbalance_threshold",
                threshold_value=effective_imbalance,
                observed_value=entry_imbalance,
            ),
        ))

    return suggestions


def detect_execution_bottleneck(
    self,
    bot_id: str,
    latency_stats: dict,
    strategy_id: str = "",
) -> list[StrategySuggestion]:
    """Detect execution pipeline bottlenecks from latency data."""
    stages = latency_stats.get("stages", {})
    bottleneck_stage = latency_stats.get("bottleneck_stage", "")
    latency_corr = latency_stats.get("latency_slippage_correlation")

    # Fire if any stage p95 > 500ms or latency-slippage correlation > 0.3
    high_p95_stages = [
        (name, data["p95_ms"])
        for name, data in stages.items()
        if data.get("p95_ms", 0) > 500
    ]
    corr_issue = latency_corr is not None and latency_corr > 0.3

    if not high_p95_stages and not corr_issue:
        return []

    parts = []
    if high_p95_stages:
        worst = max(high_p95_stages, key=lambda x: x[1])
        parts.append(f"p95 latency {worst[1]:.0f}ms at {worst[0]} stage")
    if corr_issue:
        parts.append(f"latency-slippage correlation {latency_corr:.2f}")

    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        strategy_id=strategy_id,
        title=f"Execution bottleneck -{bot_id}",
        description=f"Execution pipeline shows: {'; '.join(parts)}. "
        f"Bottleneck stage: {bottleneck_stage}.",
        confidence=0.6,
        evidence_days=30,
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="execution_bottleneck",
            bot_id=bot_id,
            threshold_name="p95_ms",
            threshold_value=500.0,
            observed_value=high_p95_stages[0][1] if high_p95_stages else (latency_corr or 0),
        ),
    )]


def detect_sizing_methodology(
    self,
    bot_id: str,
    sizing_data: dict,
    strategy_id: str = "",
) -> list[StrategySuggestion]:
    """Detect sizing methodology issues from sizing analysis data."""
    by_model = sizing_data.get("by_sizing_model", {})
    if not by_model:
        return []

    # Check for low risk efficiency in any model
    low_eff_models = [
        (model, data)
        for model, data in by_model.items()
        if data.get("avg_risk_efficiency") is not None
        and data["avg_risk_efficiency"] < 0.5
        and data.get("trade_count", 0) >= 5
    ]

    # Check for divergent win rates between models
    win_rates = [
        (model, data["win_rate"])
        for model, data in by_model.items()
        if data.get("trade_count", 0) >= 5
    ]
    wr_divergence = 0.0
    if len(win_rates) >= 2:
        rates = [wr for _, wr in win_rates]
        wr_divergence = max(rates) - min(rates)

    if not low_eff_models and wr_divergence <= 0.15:
        return []

    parts = []
    if low_eff_models:
        worst = min(low_eff_models, key=lambda x: x[1].get("avg_risk_efficiency", 0))
        parts.append(f"{worst[0]} model risk_efficiency={worst[1]['avg_risk_efficiency']:.2f}")
    if wr_divergence > 0.15:
        parts.append(f"model win_rate divergence {wr_divergence:.1%}")

    observed = low_eff_models[0][1]["avg_risk_efficiency"] if low_eff_models else wr_divergence

    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        strategy_id=strategy_id,
        title=f"Sizing methodology issue -{bot_id}",
        description=f"Position sizing analysis: {'; '.join(parts)}.",
        confidence=0.6,
        evidence_days=30,
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="sizing_methodology",
            bot_id=bot_id,
            threshold_name="risk_efficiency",
            threshold_value=0.5,
            observed_value=observed,
        ),
    )]


def detect_better_exit_strategies(
    self,
    bot_id: str,
    exit_sweep: dict,
    edge_threshold_pct: float = 0.10,
    min_trades: int = 20,
) -> list[StrategySuggestion]:
    """Tier 3: Emit a suggestion if a sweep variant beats the live exit by >= edge_threshold_pct net PnL.

    Consumes ``ExitSweepResult.model_dump()`` (skills/exit_strategy_simulator.py)
    with shape ``{"baseline_pnl": float, "results": [{"strategy": {...}, "simulated_pnl": float,
    "total_trades": int, ...}], "best_strategy": {...}, "best_improvement": float}``.
    """
    if not exit_sweep:
        return []
    baseline = float(
        exit_sweep.get("baseline_pnl")
        or exit_sweep.get("baseline_net_pnl")
        or 0.0
    )
    if baseline == 0:
        return []
    results = exit_sweep.get("results") or exit_sweep.get("variants") or []
    if not results:
        return []
    # Sample-size guard: use the largest total_trades reported across variants
    total_trades = max(
        (int(r.get("total_trades") or 0) for r in results if isinstance(r, dict)),
        default=0,
    )
    if total_trades < min_trades:
        return []
    # Prefer precomputed best_improvement; fall back to scanning results.
    best_strategy = exit_sweep.get("best_strategy") or {}
    improvement = exit_sweep.get("best_improvement")
    if improvement is None:
        best = max(
            results,
            key=lambda r: float(
                (r.get("simulated_pnl") if isinstance(r, dict) else 0.0) or 0.0
            ),
            default=None,
        )
        if not isinstance(best, dict):
            return []
        best_pnl = float(best.get("simulated_pnl") or best.get("net_pnl") or 0.0)
        base_for_row = float(best.get("baseline_pnl") or baseline)
        improvement = best_pnl - base_for_row
        best_strategy = best.get("strategy") or best_strategy
    edge = float(improvement) / max(abs(baseline), 1e-9)
    if edge < edge_threshold_pct:
        return []
    variant_name = ""
    if isinstance(best_strategy, dict):
        variant_name = (
            best_strategy.get("strategy_type")
            or best_strategy.get("name")
            or ""
        )
    variant_name = variant_name or "candidate"
    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        title=f"Better exit candidate: {variant_name} -{bot_id}",
        description=(
            f"Sweep variant '{variant_name}' improves net PnL by {improvement:+.2f} "
            f"vs baseline {baseline:.2f} ({edge:+.0%}, n={total_trades}). "
            f"Consider running an A/B experiment for this exit configuration."
        ),
        evidence_days=30,
        confidence=min(0.85, 0.5 + edge),
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="exit_sweep_edge",
            bot_id=bot_id,
            threshold_name="edge_threshold_pct",
            threshold_value=edge_threshold_pct,
            observed_value=edge,
            sample_size=total_trades,
        ),
    )]


def detect_filter_sensitivity_findings(
    self,
    bot_id: str,
    sensitivity: dict,
    min_blocks: int = 5,
) -> list[StrategySuggestion]:
    """Tier 2: Emit suggestions for filters that block more value than they save.

    Consumes ``FilterSensitivityReport.model_dump()`` (schemas/filter_sensitivity.py)
    with shape ``{"bot_id": str, "curves": [{"filter_name": str, "current_block_count": int,
    "current_net_impact": float, "blocked_winners": int, "blocked_losers": int, ...}]}``.
    ``current_net_impact`` < 0 means the filter is net value-destroying.
    """
    if not sensitivity:
        return []
    curves = sensitivity.get("curves") or sensitivity.get("filters") or []
    out: list[StrategySuggestion] = []
    for f in curves:
        if not isinstance(f, dict):
            continue
        name = f.get("filter_name") or f.get("name") or ""
        try:
            net_impact = float(
                f.get("current_net_impact")
                if f.get("current_net_impact") is not None
                else f.get("net_value", 0.0)
            )
        except (TypeError, ValueError):
            continue
        blocks = int(f.get("current_block_count") or 0)
        blocked_winners = int(f.get("blocked_winners") or 0)
        if blocks < min_blocks:
            continue
        # Filter is "marginal" when net impact is non-positive AND it has
        # blocked at least one winner (otherwise it might just be a clean
        # safety filter that has nothing to save).
        if net_impact > 0 or blocked_winners == 0:
            continue
        out.append(StrategySuggestion(
            tier=SuggestionTier.FILTER,
            bot_id=bot_id,
            title=f"Filter '{name}' blocks more value than it saves -{bot_id}",
            description=(
                f"Filter '{name}' has net impact {net_impact:+.2f} across "
                f"{blocks} blocks ({blocked_winners} winners blocked). "
                f"Filter sensitivity analysis suggests relaxing or removing it."
            ),
            evidence_days=30,
            confidence=0.55,
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="filter_sensitivity",
                bot_id=bot_id,
                threshold_name="current_net_impact",
                threshold_value=0.0,
                observed_value=net_impact,
                sample_size=blocks,
            ),
        ))
    return out


def detect_counterfactual_gaps(
    self,
    bot_id: str,
    counterfactual: dict,
    gain_threshold_pct: float = 0.10,
    min_trades: int = 20,
) -> list[StrategySuggestion]:
    """Tier 3: Emit a suggestion if a counterfactual scenario would meaningfully improve PnL.

    Consumes ``CounterfactualResult.model_dump()`` (schemas/counterfactual.py)
    with shape ``{"scenario": {"scenario_type": str, "description": str, ...},
    "baseline_pnl": float, "modified_pnl": float, "baseline_trade_count": int, ...}``.
    Tolerates a list-shaped wrapper too in case future code aggregates scenarios.
    """
    if not counterfactual:
        return []
    # Accept either a single result dict or a list of results.
    if isinstance(counterfactual, list):
        return [
            s for c in counterfactual
            for s in self.detect_counterfactual_gaps(
                bot_id, c, gain_threshold_pct, min_trades,
            )
        ]
    baseline = float(counterfactual.get("baseline_pnl") or 0.0)
    modified = float(counterfactual.get("modified_pnl") or 0.0)
    base_count = int(counterfactual.get("baseline_trade_count") or 0)
    if base_count < min_trades or baseline == 0:
        return []
    delta = modified - baseline
    edge = delta / max(abs(baseline), 1e-9)
    if edge < gain_threshold_pct:
        return []
    scenario = counterfactual.get("scenario") or {}
    scen_name = ""
    if isinstance(scenario, dict):
        scen_name = (
            scenario.get("description")
            or scenario.get("scenario_type")
            or ""
        )
    scen_name = scen_name or "scenario"
    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        title=f"Counterfactual gate gain: {scen_name} -{bot_id}",
        description=(
            f"Counterfactual '{scen_name}' projects PnL {modified:+.2f} vs "
            f"baseline {baseline:+.2f} (delta {delta:+.2f}, {edge:+.0%}, n={base_count}). "
            f"Worth running an A/B experiment for this configuration."
        ),
        evidence_days=30,
        confidence=min(0.8, 0.5 + edge),
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="counterfactual_gap",
            bot_id=bot_id,
            threshold_name="gain_threshold_pct",
            threshold_value=gain_threshold_pct,
            observed_value=edge,
            sample_size=base_count,
        ),
    )]
