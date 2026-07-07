# skills/strategy_proportion_optimizer.py
"""Strategy proportion optimizer — intra-bot risk allocation.

Deterministic pipeline. No LLM calls. Optimizes per-strategy unit risk allocation
within each bot using risk-parity + Sharpe tilt.
"""
from __future__ import annotations

import math
import statistics

from trading_assistant.schemas.proportion_optimization import (
    StrategyAllocationRecommendation,
    IntraBotAllocationReport,
    ProportionOptimizationReport,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary

# Known single-instrument bots for concentration warnings
_SINGLE_INSTRUMENT_BOTS = {"momentum_nq_01"}


class StrategyProportionOptimizer:
    """Optimizes intra-bot strategy proportion allocation."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        min_strategy_pct: float = 0.1,
        max_strategy_pct: float = 3.0,
        max_single_change_pct: float = 0.5,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.min_strategy_pct = min_strategy_pct
        self.max_strategy_pct = max_strategy_pct
        self.max_single_change_pct = max_single_change_pct

    def compute(
        self,
        per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
        current_allocations: dict[str, dict[str, float]] | None = None,
    ) -> ProportionOptimizationReport:
        """Compute proportion optimization for all bots.

        Args:
            per_strategy_summaries: outer=bot_id, inner=strategy_id
            current_allocations: outer=bot_id, inner=strategy_id → current unit_risk_pct
        """
        current_allocations = current_allocations or {}
        bot_reports: list[IntraBotAllocationReport] = []

        for bot_id, strategies in per_strategy_summaries.items():
            if not strategies:
                continue
            current_allocs = current_allocations.get(bot_id)
            report = self._optimize_bot(bot_id, strategies, current_allocs)
            bot_reports.append(report)

        return ProportionOptimizationReport(
            week_start=self.week_start,
            week_end=self.week_end,
            bot_reports=bot_reports,
        )

    def _optimize_bot(
        self,
        bot_id: str,
        strategies: dict[str, StrategyWeeklySummary],
        current_allocs: dict[str, float] | None,
    ) -> IntraBotAllocationReport:
        """Optimize strategy allocation within a single bot."""
        strat_ids = list(strategies.keys())
        special_notes: list[str] = []

        # Single strategy → 100% of bot allocation
        if len(strat_ids) == 1:
            sid = strat_ids[0]
            current = (current_allocs or {}).get(sid, 1.0)
            return IntraBotAllocationReport(
                bot_id=bot_id,
                week_start=self.week_start,
                week_end=self.week_end,
                recommendations=[
                    StrategyAllocationRecommendation(
                        bot_id=bot_id,
                        strategy_id=sid,
                        current_unit_risk_pct=current,
                        suggested_unit_risk_pct=1.0,
                        change_pct=1.0 - current,
                        rationale="single strategy — full allocation",
                    ),
                ],
            )

        # Collect dates and build aligned series
        all_dates: set[str] = set()
        for s in strategies.values():
            all_dates.update(s.daily_pnl.keys())
        dates = sorted(all_dates)

        series: dict[str, list[float]] = {}
        for sid, s in strategies.items():
            series[sid] = [s.daily_pnl.get(d, 0.0) for d in dates]

        # Compute per-strategy metrics
        metrics: dict[str, dict] = {}
        for sid in strat_ids:
            pnl = series[sid]
            n = len(pnl)
            mean_daily = statistics.mean(pnl) if n > 0 else 0.0
            std_daily = statistics.stdev(pnl) if n >= 2 else 0.0
            volatility = std_daily * math.sqrt(252) if std_daily > 0 else 0.0
            ann_return = mean_daily * 252
            sharpe = (mean_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0.0

            metrics[sid] = {
                "annualized_return": ann_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "mean_daily": mean_daily,
                "std_daily": std_daily,
            }

        # Step 1: Risk-parity base (inverse vol)
        # Zero vol with positive return → very stable, give high weight
        raw_weights: dict[str, float] = {}
        max_inv_vol = 0.0
        for sid in strat_ids:
            vol = metrics[sid]["volatility"]
            if vol > 0:
                inv = 1.0 / vol
                raw_weights[sid] = inv
                max_inv_vol = max(max_inv_vol, inv)
            else:
                raw_weights[sid] = 0.0  # placeholder
        # Assign zero-vol strategies 2× max inverse vol (they're the most stable)
        fallback = max(max_inv_vol * 2.0, 1.0)
        for sid in strat_ids:
            if raw_weights[sid] == 0.0:
                raw_weights[sid] = fallback

        total_raw = sum(raw_weights.values())
        if total_raw == 0:
            rp_weights = {sid: 1.0 / len(strat_ids) for sid in strat_ids}
        else:
            rp_weights = {sid: w / total_raw for sid, w in raw_weights.items()}

        # Step 2: Sharpe tilt
        tilted: dict[str, float] = {}
        for sid in strat_ids:
            sharpe = max(metrics[sid]["sharpe"], 0.0)
            tilt = math.sqrt(sharpe) if sharpe > 0 else 0.01
            tilted[sid] = rp_weights[sid] * tilt

        total_tilted = sum(tilted.values())
        if total_tilted == 0:
            norm = {sid: 1.0 / len(strat_ids) for sid in strat_ids}
        else:
            norm = {sid: w / total_tilted for sid, w in tilted.items()}

        # Step 3: Scale to unit_risk_pct range
        # Default: equal allocation sums to max_strategy_pct total
        total_budget = self.max_strategy_pct * len(strat_ids) / len(strat_ids)
        suggested: dict[str, float] = {}
        for sid in strat_ids:
            raw_pct = norm[sid] * total_budget * len(strat_ids)
            # Clamp
            raw_pct = max(self.min_strategy_pct, min(self.max_strategy_pct, raw_pct))
            suggested[sid] = round(raw_pct, 4)

        # Step 4: Compute bot-level Sharpe before/after
        current_allocs = current_allocs or {}
        bot_pnl_current = self._weighted_portfolio_pnl(series, current_allocs, dates)
        bot_pnl_suggested = self._weighted_portfolio_pnl(series, suggested, dates)
        current_sharpe = self._sharpe(bot_pnl_current)
        suggested_sharpe = self._sharpe(bot_pnl_suggested)

        # Step 5: Check NQ concentration
        if bot_id in _SINGLE_INSTRUMENT_BOTS:
            special_notes.append(
                f"NQ concentration warning: all {len(strat_ids)} strategies in {bot_id} trade the same instrument"
            )

        # Step 6: Build recommendations
        recommendations: list[StrategyAllocationRecommendation] = []
        rebalance_needed = False
        for sid in strat_ids:
            current = current_allocs.get(sid, 1.0)
            change = suggested[sid] - current

            if abs(change) > self.max_single_change_pct:
                capped = current + self.max_single_change_pct * (1 if change > 0 else -1)
                suggested[sid] = max(self.min_strategy_pct, min(self.max_strategy_pct, capped))
                change = suggested[sid] - current

            if abs(change) > 0.1:
                rebalance_needed = True

            # Marginal Sharpe
            without_pnl = [
                sum(series[s][i] for s in strat_ids if s != sid)
                for i in range(len(dates))
            ]
            sharpe_without = self._sharpe(without_pnl)
            marginal = suggested_sharpe - sharpe_without

            cap_eff = metrics[sid]["annualized_return"] / max(suggested[sid], 0.01)

            recommendations.append(
                StrategyAllocationRecommendation(
                    bot_id=bot_id,
                    strategy_id=sid,
                    current_unit_risk_pct=current,
                    suggested_unit_risk_pct=suggested[sid],
                    change_pct=round(change, 4),
                    capital_efficiency=round(cap_eff, 2),
                    marginal_sharpe=round(marginal, 4),
                    rationale=self._build_rationale(sid, metrics[sid], suggested[sid]),
                    evidence_period_days=len(dates),
                ),
            )

        return IntraBotAllocationReport(
            bot_id=bot_id,
            week_start=self.week_start,
            week_end=self.week_end,
            recommendations=recommendations,
            current_bot_sharpe=round(current_sharpe, 4),
            suggested_bot_sharpe=round(suggested_sharpe, 4),
            sharpe_change=round(suggested_sharpe - current_sharpe, 4),
            rebalance_needed=rebalance_needed,
            special_notes=special_notes,
        )

    def _weighted_portfolio_pnl(
        self,
        series: dict[str, list[float]],
        weights: dict[str, float],
        dates: list[str],
    ) -> list[float]:
        """Compute weighted portfolio daily PnL."""
        n = len(dates)
        if not weights:
            # Equal weight fallback
            total_weight = len(series)
            return [
                sum(series[sid][i] for sid in series) / max(total_weight, 1)
                for i in range(n)
            ]
        result = []
        for i in range(n):
            day_pnl = sum(
                series[sid][i] * weights.get(sid, 1.0)
                for sid in series
            )
            result.append(day_pnl)
        return result

    @staticmethod
    def _sharpe(daily_pnl: list[float]) -> float:
        if len(daily_pnl) < 2:
            return 0.0
        mean = statistics.mean(daily_pnl)
        std = statistics.stdev(daily_pnl)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _build_rationale(sid: str, metrics: dict, suggested: float) -> str:
        return (
            f"Sharpe={metrics['sharpe']:.2f}, "
            f"vol={metrics['volatility']:.1f}, "
            f"suggested risk={suggested:.2f}%"
        )
