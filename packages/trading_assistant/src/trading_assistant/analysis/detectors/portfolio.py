"""Portfolio detector behavior."""

from __future__ import annotations


from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
)



def detect_correlation_breakdown(
    self,
    correlations: list,  # list[CorrelationSummary]
    threshold: float = 0.7,
) -> list[StrategySuggestion]:
    """Tier 3: Detect rising cross-bot return correlation (systemic risk)."""
    suggestions: list[StrategySuggestion] = []
    for corr in correlations:
        pair_id = f"{corr.bot_a}+{corr.bot_b}"
        effective_threshold = self._get_threshold(
            "correlation", "threshold", pair_id, threshold,
        )
        if corr.rolling_30d_correlation >= effective_threshold:
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.STRATEGY_VARIANT,
                bot_id=pair_id,
                title=f"High correlation -{corr.bot_a} / {corr.bot_b}",
                description=(
                    f"30d return correlation is {corr.rolling_30d_correlation:.2f}. "
                    f"Same-direction trading {corr.same_direction_pct:.0%} of the time. "
                    f"This increases systemic risk during adverse moves. "
                    f"Consider diversifying signal sources or staggering entry timing."
                ),
                evidence_days=30,
                confidence=min(0.9, corr.rolling_30d_correlation),
                requires_human_judgment=True,
                detection_context=DetectionContext(
                    detector_name="correlation",
                    bot_id=pair_id,
                    threshold_name="threshold",
                    threshold_value=effective_threshold,
                    observed_value=corr.rolling_30d_correlation,
                ),
            ))
    return suggestions


def detect_drawdown_patterns(
    self,
    bot_id: str,
    largest_single_loss_pct: float,
    max_drawdown_pct: float,
    avg_loss_pct: float,
    concentration_threshold: float = 3.0,
) -> list[StrategySuggestion]:
    """Tier 3: Detect concentrated drawdown (single loss dominates)."""
    if avg_loss_pct <= 0:
        return []
    concentration = largest_single_loss_pct / avg_loss_pct
    effective_threshold = self._get_threshold(
        "drawdown_concentration", "concentration_threshold", bot_id,
        concentration_threshold,
    )
    if concentration < effective_threshold:
        return []
    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        title=f"Concentrated drawdown risk -{bot_id}",
        description=(
            f"Largest single loss ({largest_single_loss_pct:.1f}%) is "
            f"{concentration:.1f}x the average loss ({avg_loss_pct:.1f}%). "
            f"Max drawdown: {max_drawdown_pct:.1f}%. "
            f"Consider tighter per-trade risk limits or position sizing adjustments."
        ),
        evidence_days=30,
        confidence=0.65,
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="drawdown_concentration",
            bot_id=bot_id,
            threshold_name="concentration_threshold",
            threshold_value=effective_threshold,
            observed_value=concentration,
        ),
    )]


def detect_position_sizing_issues(
    self,
    bot_id: str,
    avg_win_pct: float,
    avg_loss_pct: float,
    win_rate: float,
    loss_win_ratio_threshold: float = 1.5,
) -> list[StrategySuggestion]:
    """Tier 3: Detect asymmetric position sizing (losses > wins despite positive win rate)."""
    if avg_win_pct <= 0 or win_rate < 0.5:
        return []
    loss_win_ratio = avg_loss_pct / avg_win_pct
    effective_threshold = self._get_threshold(
        "position_sizing", "loss_win_ratio_threshold", bot_id,
        loss_win_ratio_threshold,
    )
    if loss_win_ratio < effective_threshold:
        return []
    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        title=f"Position sizing imbalance -{bot_id}",
        description=(
            f"Average loss ({avg_loss_pct:.1f}%) is {loss_win_ratio:.1f}x "
            f"average win ({avg_win_pct:.1f}%) despite {win_rate:.0%} win rate. "
            f"Risk/reward is asymmetric - consider reducing position size on "
            f"lower-confidence signals or tightening stop placement."
        ),
        evidence_days=30,
        confidence=0.6,
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="position_sizing",
            bot_id=bot_id,
            threshold_name="loss_win_ratio_threshold",
            threshold_value=effective_threshold,
            observed_value=loss_win_ratio,
        ),
    )]


def detect_portfolio_crowding(
    self,
    bot_id: str,
    portfolio_context: dict,
    strategy_id: str = "",
) -> list[StrategySuggestion]:
    """Detect portfolio crowding effects from portfolio context data."""
    crowded_wr = portfolio_context.get("crowded_win_rate")
    uncrowded_wr = portfolio_context.get("uncrowded_win_rate")
    crowding_count = portfolio_context.get("crowding_count", 0)

    if crowded_wr is None or uncrowded_wr is None or crowding_count < 3:
        return []

    wr_gap = uncrowded_wr - crowded_wr
    if wr_gap <= 0.10:
        return []

    return [StrategySuggestion(
        tier=SuggestionTier.STRATEGY_VARIANT,
        bot_id=bot_id,
        strategy_id=strategy_id,
        title=f"Portfolio crowding drag -{bot_id}",
        description=(
            f"Crowded entries (>2 correlated positions) have {wr_gap:.1%} lower "
            f"win rate than uncrowded ({crowded_wr:.1%} vs {uncrowded_wr:.1%}, "
            f"n={crowding_count} crowded trades)."
        ),
        confidence=0.6,
        evidence_days=30,
        requires_human_judgment=True,
        detection_context=DetectionContext(
            detector_name="portfolio_crowding",
            bot_id=bot_id,
            threshold_name="crowding_win_rate_gap",
            threshold_value=0.10,
            observed_value=wr_gap,
            sample_size=int(crowding_count),
        ),
    )]


def detect_family_imbalance(
    self,
    family_summaries: dict[str, dict],
    family_allocations: dict[str, float],
    min_days: int = 30,
) -> list[StrategySuggestion]:
    """Detect families consistently underperforming their allocation weight (2A).

    Args:
        family_summaries: family -{total_net_pnl, trade_count, days, ...}
        family_allocations: family - allocation weight (0-1)
    """
    suggestions: list[StrategySuggestion] = []
    if not family_summaries or not family_allocations:
        return suggestions

    total_pnl = sum(s.get("total_net_pnl", 0.0) for s in family_summaries.values())
    if total_pnl == 0:
        return suggestions

    for family, summary in family_summaries.items():
        alloc_weight = family_allocations.get(family, 0.0)
        if alloc_weight <= 0:
            continue

        days = summary.get("days", 0)
        if days < min_days:
            continue

        family_pnl = summary.get("total_net_pnl", 0.0)
        pnl_share = family_pnl / total_pnl  # safe: total_pnl != 0 guarded above

        # Family PnL share is significantly below its allocation weight
        if alloc_weight > 0.1 and pnl_share < alloc_weight * 0.5:
            # Suggest rebalancing - max 15% shift
            current = alloc_weight
            suggested = max(0.05, current - min(0.15, current * 0.3))
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.PORTFOLIO,
                bot_id="PORTFOLIO",
                title=f"Reduce {family} family allocation",
                description=(
                    f"{family} family contributes {pnl_share:.1%} of PnL but holds "
                    f"{alloc_weight:.1%} allocation over {days} days. "
                    f"Consider reducing from {current:.1%} to {suggested:.1%}."
                ),
                current_value=f"{current:.4f}",
                suggested_value=f"{suggested:.4f}",
                evidence_days=days,
                confidence=min(0.7, 0.4 + (days - min_days) / 100),
                detection_context=DetectionContext(
                    detector_name="detect_family_imbalance",
                    bot_id="PORTFOLIO",
                    threshold_name="alloc_weight",
                    threshold_value=alloc_weight,
                    observed_value=round(pnl_share, 4),
                ),
            ))

    return suggestions


def detect_correlation_concentration(
    self,
    correlation_matrix: dict[str, float],
    current_allocations: dict[str, float],
    threshold: float = 0.7,
    weight_threshold: float = 0.4,
) -> list[StrategySuggestion]:
    """Detect pairs with high correlation holding excessive combined weight (2B).

    Args:
        correlation_matrix: "botA_botB" - correlation coefficient
        current_allocations: bot_id - allocation weight (0-1)
    """
    suggestions: list[StrategySuggestion] = []
    if not correlation_matrix or not current_allocations:
        return suggestions

    for pair_key, corr_val in correlation_matrix.items():
        if corr_val <= threshold:
            continue

        parts = pair_key.split("_", 1)
        if len(parts) != 2:
            continue
        bot_a, bot_b = parts

        weight_a = current_allocations.get(bot_a, 0.0)
        weight_b = current_allocations.get(bot_b, 0.0)
        combined = weight_a + weight_b

        if combined > weight_threshold:
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.PORTFOLIO,
                bot_id="PORTFOLIO",
                title=f"Reduce correlated pair {bot_a}/{bot_b} combined weight",
                description=(
                    f"{bot_a} and {bot_b} have correlation {corr_val:.2f} "
                    f"with combined allocation {combined:.1%} (>{weight_threshold:.0%}). "
                    f"High correlation with high combined weight creates concentration risk."
                ),
                confidence=min(0.8, 0.5 + (corr_val - threshold) * 2),
                detection_context=DetectionContext(
                    detector_name="detect_correlation_concentration",
                    bot_id="PORTFOLIO",
                    threshold_name="correlation_threshold",
                    threshold_value=threshold,
                    observed_value=corr_val,
                ),
            ))

    return suggestions


def detect_drawdown_tier_miscalibration(
    self,
    historical_drawdowns: list[float],
    current_tiers: list[list[float]],
    min_days: int = 90,
) -> list[StrategySuggestion]:
    """Detect drawdown tiers that never trigger or trigger too often (2C).

    Safety: only suggests narrowing, never removing or loosening.

    Args:
        historical_drawdowns: list of daily drawdown percentages (0-100)
        current_tiers: list of [threshold_pct, multiplier] pairs
    """
    suggestions: list[StrategySuggestion] = []
    if len(historical_drawdowns) < min_days or not current_tiers:
        return suggestions

    for tier_idx, tier in enumerate(current_tiers):
        if len(tier) < 2:
            continue
        threshold = tier[0]

        # Count how many days breached this tier
        breaches = sum(1 for dd in historical_drawdowns if dd >= threshold)
        breach_rate = breaches / len(historical_drawdowns)

        if breach_rate == 0.0 and tier_idx > 0:
            # Tier never triggers - may be too loose. Suggest narrowing.
            prev_threshold = current_tiers[tier_idx - 1][0] if tier_idx > 0 else 0
            midpoint = (prev_threshold + threshold) / 2
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.PORTFOLIO,
                bot_id="PORTFOLIO",
                title=f"Tighten drawdown tier {tier_idx + 1} threshold",
                description=(
                    f"Drawdown tier at {threshold}% never triggered in "
                    f"{len(historical_drawdowns)} days. Consider narrowing from "
                    f"{threshold}% to {midpoint:.1f}% (midpoint with tier {tier_idx})."
                ),
                current_value=f"{threshold}",
                suggested_value=f"{midpoint:.1f}",
                evidence_days=len(historical_drawdowns),
                confidence=0.5,
                detection_context=DetectionContext(
                    detector_name="detect_drawdown_tier_miscalibration",
                    bot_id="PORTFOLIO",
                    threshold_name=f"drawdown_tier_{tier_idx}",
                    threshold_value=threshold,
                    observed_value=0.0,
                ),
            ))
        elif breach_rate > 0.3:
            # Tier triggers too often - suggests threshold is too tight
            # But we only narrow, never loosen - so no suggestion here
            pass

    return suggestions


def detect_coordination_gaps(
    self,
    concurrent_positions: dict,
    existing_coordination: dict | None = None,
    min_co_occurrences: int = 50,
) -> list[StrategySuggestion]:
    """Detect strategies that frequently collide without coordination rules (2D).

    Args:
        concurrent_positions: from build_concurrent_position_analysis()
        existing_coordination: CoordinationConfig dict (signals, cooldown_pairs)
    """
    suggestions: list[StrategySuggestion] = []
    if not concurrent_positions:
        return suggestions

    pairs = concurrent_positions.get("pairs", {})
    existing_cooldowns: set[str] = set()
    if existing_coordination:
        for cp in existing_coordination.get("cooldown_pairs", []):
            strats = cp.get("strategies", [])
            if len(strats) >= 2:
                existing_cooldowns.add("_".join(sorted(strats[:2])))

    # observation_days from top-level metadata, or estimate from co-occurrence counts
    obs_days = concurrent_positions.get("observation_days", 0)

    for pair_key, data in pairs.items():
        co_occ = data.get("co_occurrences", 0)
        if co_occ < min_co_occurrences:
            continue

        # Skip if already coordinated
        if pair_key in existing_cooldowns:
            continue

        same_dir = data.get("same_direction_count", 0)
        same_dir_pct = same_dir / co_occ if co_occ > 0 else 0.0

        # Use observation_days from data, or estimate conservatively from co-occurrences
        pair_days = data.get("observation_days", obs_days) or max(co_occ // 2, 0)

        if same_dir_pct > 0.6:
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.PORTFOLIO,
                bot_id="PORTFOLIO",
                title=f"Add coordination for {pair_key}",
                description=(
                    f"Strategies {pair_key} have {co_occ} co-occurrences with "
                    f"{same_dir_pct:.0%} same-direction over ~{pair_days} days. "
                    f"No coordination rule exists. "
                    f"Consider adding a cooldown or direction filter."
                ),
                evidence_days=pair_days,
                confidence=min(0.7, 0.4 + co_occ / 200),
                detection_context=DetectionContext(
                    detector_name="detect_coordination_gaps",
                    bot_id="PORTFOLIO",
                    threshold_name="same_direction_pct",
                    threshold_value=0.6,
                    observed_value=round(same_dir_pct, 3),
                ),
            ))

    return suggestions


def detect_heat_cap_utilization(
    self,
    daily_heat_series: list[float],
    heat_cap_R: float,
    min_days: int = 30,
) -> list[StrategySuggestion]:
    """Detect heat cap consistently too tight or too loose (2E).

    Args:
        daily_heat_series: list of daily peak heat values (in R)
        heat_cap_R: current heat cap setting
    """
    suggestions: list[StrategySuggestion] = []
    if len(daily_heat_series) < min_days or heat_cap_R <= 0:
        return suggestions

    utilization_ratios = [h / heat_cap_R for h in daily_heat_series if heat_cap_R > 0]
    if not utilization_ratios:
        return suggestions

    avg_util = sum(utilization_ratios) / len(utilization_ratios)
    high_util_days = sum(1 for u in utilization_ratios if u > 0.9)
    high_util_pct = high_util_days / len(utilization_ratios)

    if high_util_pct > 0.3:
        # Consistently near cap - opportunity cost
        # Max +10% adjustment
        suggested = round(heat_cap_R * 1.10, 1)
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.PORTFOLIO,
            bot_id="PORTFOLIO",
            title="Increase heat_cap_R - consistently at capacity",
            description=(
                f"Heat utilization exceeds 90% of {heat_cap_R}R on "
                f"{high_util_pct:.0%} of days ({high_util_days}/{len(utilization_ratios)}). "
                f"Avg utilization: {avg_util:.0%}. "
                f"Consider increasing from {heat_cap_R}R to {suggested}R (+10%)."
            ),
            current_value=str(heat_cap_R),
            suggested_value=str(suggested),
            evidence_days=len(daily_heat_series),
            confidence=min(0.7, 0.4 + high_util_pct),
            detection_context=DetectionContext(
                detector_name="detect_heat_cap_utilization",
                bot_id="PORTFOLIO",
                threshold_name="heat_cap_R",
                threshold_value=heat_cap_R,
                observed_value=round(avg_util * heat_cap_R, 2),
            ),
        ))
    elif avg_util < 0.3:
        # Very underutilized - overly conservative
        suggested = round(heat_cap_R * 0.90, 1)
        suggestions.append(StrategySuggestion(
            tier=SuggestionTier.PORTFOLIO,
            bot_id="PORTFOLIO",
            title="Reduce heat_cap_R - significantly underutilized",
            description=(
                f"Heat utilization averages only {avg_util:.0%} of {heat_cap_R}R "
                f"over {len(utilization_ratios)} days. Cap may be overly conservative. "
                f"Consider reducing from {heat_cap_R}R to {suggested}R (-10%)."
            ),
            current_value=str(heat_cap_R),
            suggested_value=str(suggested),
            evidence_days=len(daily_heat_series),
            confidence=0.5,
            detection_context=DetectionContext(
                detector_name="detect_heat_cap_utilization",
                bot_id="PORTFOLIO",
                threshold_name="heat_cap_R",
                threshold_value=heat_cap_R,
                observed_value=round(avg_util * heat_cap_R, 2),
            ),
        ))

    return suggestions
