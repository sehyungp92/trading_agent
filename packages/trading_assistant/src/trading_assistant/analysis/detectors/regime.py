"""Regime detector behavior."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
import statistics

from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.regime_conditional import (
    RegimeAllocation,
    RegimeConditionalReport,
    RegimeDistribution,
    RegimeStrategyMetrics,
)
from trading_assistant.schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary



def compute_regime_conditional_metrics(
    self,
    per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
    trades_by_bot: dict[str, list],
) -> RegimeConditionalReport:
    """Compute regime-conditional performance metrics and allocation suggestions.

    Args:
        per_strategy_summaries: outer key=bot_id, inner key=strategy_id
        trades_by_bot: bot_id -> list of TradeEvent
    """

    # Group trades by (bot_id, strategy_id, regime)
    grouped: dict[tuple[str, str, str], list] = defaultdict(list)
    regime_counts: dict[str, int] = defaultdict(int)
    total_trades = 0

    for bot_id, trades in trades_by_bot.items():
        for t in trades:
            regime = getattr(t, "market_regime", None) or "unknown"
            strat = getattr(t, "strategy_id", "") or "default"
            grouped[(bot_id, strat, regime)].append(t)
            regime_counts[regime] += 1
            total_trades += 1

    # Compute per-group metrics
    metrics: list[RegimeStrategyMetrics] = []
    regime_strategy_sharpes: dict[str, dict[str, float]] = defaultdict(dict)

    for (bot_id, strat_id, regime), trades in grouped.items():
        if not trades:
            continue
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        expectancy = statistics.mean(pnls) if pnls else 0.0
        sharpe = 0.0
        if len(pnls) >= 2:
            std = statistics.stdev(pnls)
            if std > 0:
                sharpe = (statistics.mean(pnls) / std) * math.sqrt(252)

        # Max drawdown from cumulative PnL
        cumsum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumsum += p
            if cumsum > peak:
                peak = cumsum
            dd = (peak - cumsum) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        key = f"{bot_id}:{strat_id}"
        regime_strategy_sharpes[regime][key] = sharpe

        metrics.append(RegimeStrategyMetrics(
            bot_id=bot_id,
            strategy_id=strat_id,
            regime=regime,
            trade_count=len(trades),
            win_rate=round(win_rate, 4),
            expectancy=round(expectancy, 2),
            sharpe=round(sharpe, 4),
            max_drawdown_pct=round(max_dd * 100, 2),
        ))

    # Regime distribution
    regime_dist: list[RegimeDistribution] = []
    for regime, count in regime_counts.items():
        pct = (count / total_trades * 100.0) if total_trades > 0 else 0.0
        regime_dist.append(RegimeDistribution(
            regime=regime,
            pct_of_time=round(pct, 1),
            trade_count=count,
        ))

    # Optimal allocations per regime (inverse-volatility weighted)
    allocations: list[RegimeAllocation] = []
    for regime, strat_sharpes in regime_strategy_sharpes.items():
        if not strat_sharpes:
            continue
        # Use max(sharpe, 0.01) to avoid division by zero; zero/negative -> minimum alloc
        inv_vol = {}
        for key, s in strat_sharpes.items():
            inv_vol[key] = max(s, 0.01)
        total_inv = sum(inv_vol.values())
        alloc = {k: round(v / total_inv * 100.0, 1) for k, v in inv_vol.items()} if total_inv > 0 else {}
        allocations.append(RegimeAllocation(
            regime=regime,
            allocations=alloc,
            rationale=f"Inverse-volatility allocation across {len(alloc)} strategies in {regime} regime",
        ))

    # Generate suggestions for underperforming strategy-regime combos
    suggestions: list[dict] = []
    for m in metrics:
        if m.trade_count >= 10 and m.win_rate < 0.35 and m.expectancy < 0:
            suggestions.append({
                "regime": m.regime,
                "strategy": f"{m.bot_id}:{m.strategy_id}",
                "current_alloc": "equal",
                "suggested_alloc": "reduce",
                "reason": (
                    f"In {m.regime}, {m.bot_id}:{m.strategy_id} has "
                    f"{m.win_rate:.0%} win rate and ${m.expectancy:.0f} expectancy "
                    f"over {m.trade_count} trades. Consider scaling down."
                ),
            })

    return RegimeConditionalReport(
        week_start=self.week_start,
        week_end=self.week_end,
        metrics=metrics,
        optimal_allocations=allocations,
        regime_distribution=regime_dist,
        suggestions=suggestions,
    )


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
    """Tier 3: Compare applied regime config against actual performance.

    If losses persist despite reduced sizing, config isn't aggressive enough.
    If strong win rate with heavy reduction, config may be too conservative.
    """
    suggestions: list[StrategySuggestion] = []
    if regime_trade_count < min_trades:
        return suggestions

    strategy_id = strategy_id or self._resolve_strategy_id(bot_id)

    # Losing despite reduction - not aggressive enough
    if regime_pnl < 0 and regime_unit_risk_mult < 1.0:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            strategy_id=strategy_id,
            tier=SuggestionTier.STRATEGY_VARIANT,
            title=f"Regime sizing too lenient in {macro_regime}",
            description=(
                f"In macro regime {macro_regime} with {regime_unit_risk_mult}x sizing, "
                f"PnL was {regime_pnl:.1f} over {regime_trade_count} trades "
                f"(win rate {regime_win_rate:.1%}). Consider reducing "
                f"regime_unit_risk_mult further."
            ),
            confidence=0.5,
            current_value=str(regime_unit_risk_mult),
            suggested_value=str(round(max(0.1, regime_unit_risk_mult - 0.15), 2)),
            evidence=[
                f"regime={macro_regime}",
                f"pnl={regime_pnl:.1f}",
                f"win_rate={regime_win_rate:.2f}",
                f"mult={regime_unit_risk_mult}",
                f"trades={regime_trade_count}",
            ],
            detection_context=DetectionContext(
                detector_name="regime_config_effectiveness",
                bot_id=bot_id,
                threshold_name="regime_unit_risk_mult",
                threshold_value=regime_unit_risk_mult,
                observed_value=regime_pnl,
            ),
        ))

    # Winning strongly despite heavy reduction - too conservative
    if regime_win_rate > 0.55 and regime_pnl > 0 and regime_unit_risk_mult < 0.7:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            strategy_id=strategy_id,
            tier=SuggestionTier.STRATEGY_VARIANT,
            title=f"Regime sizing too conservative in {macro_regime}",
            description=(
                f"In macro regime {macro_regime} with {regime_unit_risk_mult}x sizing, "
                f"win rate was {regime_win_rate:.1%} with positive PnL ({regime_pnl:.1f}). "
                f"Drawdown protection may be leaving returns on the table."
            ),
            confidence=0.4,
            current_value=str(regime_unit_risk_mult),
            suggested_value=str(round(min(1.0, regime_unit_risk_mult + 0.1), 2)),
            evidence=[
                f"regime={macro_regime}",
                f"win_rate={regime_win_rate:.2f}",
                f"pnl={regime_pnl:.1f}",
                f"mult={regime_unit_risk_mult}",
            ],
            detection_context=DetectionContext(
                detector_name="regime_config_effectiveness",
                bot_id=bot_id,
                threshold_name="regime_unit_risk_mult",
                threshold_value=regime_unit_risk_mult,
                observed_value=regime_win_rate,
            ),
        ))

    return suggestions


def detect_regime_transition_cost(
    self,
    transition_events: list[dict],
    daily_pnl_by_date: dict[str, float],
    window_days: int = 5,
) -> list[StrategySuggestion]:
    """Tier 3: Measure P&L around regime transitions.

    Args:
        transition_events: list of dicts with from_regime, to_regime, date.
        daily_pnl_by_date: {YYYY-MM-DD: total_pnl} for all bots combined.
        window_days: days around transition to measure.
    """
    from datetime import timedelta

    suggestions: list[StrategySuggestion] = []
    if not transition_events or not daily_pnl_by_date:
        return suggestions

    for evt in transition_events:
        trans_date = evt.get("date", "")
        if not trans_date:
            continue
        try:
            dt = datetime.strptime(trans_date, "%Y-%m-%d")
        except ValueError:
            continue

        window_pnl = 0.0
        days_found = 0
        for offset in range(-window_days, window_days + 1):
            d = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
            if d in daily_pnl_by_date:
                window_pnl += daily_pnl_by_date[d]
                days_found += 1

        if days_found < 3:
            continue

        from_r = evt.get("from_regime", "?")
        to_r = evt.get("to_regime", "?")

        if window_pnl < 0:
            desc = (
                f"Regime transition {from_r}->{to_r} on {trans_date} "
                f"had negative P&L ({window_pnl:.1f}) in +/-{window_days}d window. "
                f"Review applied_regime_config changes and whether sizing/disables "
                f"responded correctly to the transition."
            )

            suggestions.append(StrategySuggestion(
                bot_id="portfolio",
                strategy_id="",
                tier=SuggestionTier.STRATEGY_VARIANT,
                title=f"Costly regime transition {from_r}->{to_r}",
                description=desc,
                confidence=0.45,
                evidence=[
                    f"transition={from_r}->{to_r}",
                    f"date={trans_date}",
                    f"window_pnl={window_pnl:.1f}",
                    f"days_with_data={days_found}/{window_days * 2 + 1}",
                ],
                detection_context=DetectionContext(
                    detector_name="regime_transition_cost",
                    bot_id="portfolio",
                    threshold_name="window_pnl",
                    threshold_value=0.0,
                    observed_value=window_pnl,
                ),
            ))

    return suggestions


def detect_stress_entry_pattern(
    self,
    bot_id: str,
    trades_by_stress: dict[str, dict],
    min_trades_per_bucket: int = 5,
    strategy_id: str = "",
) -> list[StrategySuggestion]:
    """Tier 3: Aggregate trade outcomes by stress_level_at_entry buckets.

    Args:
        trades_by_stress: {bucket_name: {trade_count, win_rate, avg_pnl, expectancy}}
            Buckets: "low" (<0.3), "mid" (0.3-0.7), "high" (>0.7).
    """
    suggestions: list[StrategySuggestion] = []
    high = trades_by_stress.get("high", {})
    low = trades_by_stress.get("low", {})

    high_count = high.get("trade_count", 0)
    low_count = low.get("trade_count", 0)

    if high_count < min_trades_per_bucket or low_count < min_trades_per_bucket:
        return suggestions

    high_wr = high.get("win_rate", 0.0)
    low_wr = low.get("win_rate", 0.0)
    high_exp = high.get("expectancy", 0.0)

    strategy_id = strategy_id or self._resolve_strategy_id(bot_id)

    # High-stress entries significantly worse than low-stress
    # NOTE: stress_level has 41% FPR (observational only per reliability guide).
    # Report as diagnostic finding, not as basis for config mutations.
    if low_wr - high_wr > 0.15 and high_exp < 0:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            strategy_id=strategy_id,
            tier=SuggestionTier.STRATEGY_VARIANT,
            title="High-stress entries underperform (diagnostic)",
            description=(
                f"Trades entered during high stress (>0.7) have "
                f"{high_wr:.1%} win rate vs {low_wr:.1%} for low stress (<0.3), "
                f"with negative expectancy ({high_exp:.2f}). "
                f"Caveat: stress_level has 41% false positive rate and cannot "
                f"reliably discriminate stress from normal volatility. "
                f"Investigate whether the underperformance correlates with "
                f"macro regime (G/R/S/D) instead - regime is high-confidence."
            ),
            confidence=0.35,
            evidence=[
                f"high_stress_wr={high_wr:.2f}",
                f"low_stress_wr={low_wr:.2f}",
                f"high_stress_exp={high_exp:.2f}",
                f"high_trades={high_count}",
                f"low_trades={low_count}",
            ],
            detection_context=DetectionContext(
                detector_name="stress_entry_pattern",
                bot_id=bot_id,
                threshold_name="stress_win_rate_gap",
                threshold_value=0.15,
                observed_value=round(low_wr - high_wr, 3),
            ),
        ))

    return suggestions
