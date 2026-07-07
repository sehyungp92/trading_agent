# skills/portfolio_allocator.py
"""Portfolio allocator — risk-parity + Calmar tilt cross-bot allocation.

Deterministic pipeline. No LLM calls. Produces allocation recommendations
for the weekly analysis prompt.
"""
from __future__ import annotations

import math
import statistics

from trading_assistant.schemas.portfolio_allocation import (
    AllocationConstraints,
    BotAllocationRecommendation,
    PortfolioAllocationReport,
)
from trading_assistant.schemas.weekly_metrics import BotWeeklySummary


class PortfolioAllocator:
    """Computes cross-bot allocation recommendations using risk-parity + Calmar tilt."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        constraints: AllocationConstraints | None = None,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.constraints = constraints or AllocationConstraints()

    def compute(
        self,
        bot_summaries: dict[str, BotWeeklySummary],
        current_allocations: dict[str, float],
        correlation_matrix: dict[str, float] | None = None,
    ) -> PortfolioAllocationReport:
        """Compute allocation recommendations.

        Args:
            bot_summaries: weekly summary per bot
            current_allocations: bot_id → current allocation %
            correlation_matrix: optional "botA_botB" → correlation for diversification penalty
        """
        if not bot_summaries:
            return PortfolioAllocationReport(
                week_start=self.week_start,
                week_end=self.week_end,
            )

        bot_ids = list(bot_summaries.keys())

        # Single bot → 100%
        if len(bot_ids) == 1:
            bid = bot_ids[0]
            current = current_allocations.get(bid, 100.0)
            return PortfolioAllocationReport(
                week_start=self.week_start,
                week_end=self.week_end,
                recommendations=[
                    BotAllocationRecommendation(
                        bot_id=bid,
                        current_allocation_pct=current,
                        suggested_allocation_pct=100.0,
                        change_pct=100.0 - current,
                        rationale="single bot — full allocation",
                    ),
                ],
            )

        # Compute per-bot metrics
        metrics = {}
        for bid, summary in bot_summaries.items():
            metrics[bid] = self._compute_bot_metrics(summary)

        # Step 1: Risk-parity base (inverse volatility)
        raw_weights = {}
        for bid, m in metrics.items():
            vol = m["volatility"]
            raw_weights[bid] = 1.0 / vol if vol > 0 else 0.0

        total_raw = sum(raw_weights.values())
        if total_raw == 0:
            # All zero vol — equal allocation
            rp_weights = {bid: 1.0 / len(bot_ids) for bid in bot_ids}
        else:
            rp_weights = {bid: w / total_raw for bid, w in raw_weights.items()}

        # Step 2: Calmar tilt
        tilted_weights = {}
        for bid in bot_ids:
            calmar = max(metrics[bid]["calmar"], 0.0)
            tilt = math.sqrt(calmar) if calmar > 0 else 0.01
            tilted_weights[bid] = rp_weights[bid] * tilt

        total_tilted = sum(tilted_weights.values())
        if total_tilted == 0:
            norm_weights = {bid: 1.0 / len(bot_ids) for bid in bot_ids}
        else:
            norm_weights = {bid: w / total_tilted * 100.0 for bid, w in tilted_weights.items()}

        # Step 3: Apply correlation penalty if available
        if correlation_matrix:
            norm_weights = self._apply_correlation_penalty(
                norm_weights, correlation_matrix, metrics,
            )

        # Step 4: Apply constraints
        norm_weights = self._apply_constraints(norm_weights)

        # Step 5: Build recommendations
        recommendations = []
        for bid in bot_ids:
            current = current_allocations.get(bid, 0.0)
            suggested = norm_weights[bid]
            change = suggested - current
            cap_eff = metrics[bid]["annualized_return"] / max(suggested, 1.0)
            recommendations.append(
                BotAllocationRecommendation(
                    bot_id=bid,
                    current_allocation_pct=current,
                    suggested_allocation_pct=round(suggested, 2),
                    change_pct=round(change, 2),
                    capital_efficiency=round(cap_eff, 4),
                    calmar_contribution=round(metrics[bid]["calmar"], 4),
                    rationale=self._build_rationale(bid, metrics[bid], suggested, current),
                ),
            )

        # Step 6: Compute portfolio Calmar
        current_calmar = self._portfolio_calmar(metrics, current_allocations)
        suggested_calmar = self._portfolio_calmar(metrics, norm_weights)
        calmar_change = (
            ((suggested_calmar - current_calmar) / abs(current_calmar) * 100.0)
            if current_calmar != 0
            else 0.0
        )

        rebalance_needed = any(abs(r.change_pct) > 3.0 for r in recommendations)

        return PortfolioAllocationReport(
            week_start=self.week_start,
            week_end=self.week_end,
            recommendations=recommendations,
            current_portfolio_calmar=round(current_calmar, 4),
            suggested_portfolio_calmar=round(suggested_calmar, 4),
            calmar_change_pct=round(calmar_change, 2),
            rebalance_needed=rebalance_needed,
        )

    def _compute_bot_metrics(self, summary: BotWeeklySummary) -> dict:
        """Compute annualized return, volatility, max drawdown, Calmar for a bot."""
        daily_pnl = list(summary.daily_pnl.values())
        n_days = len(daily_pnl) if daily_pnl else 1

        total_return = summary.net_pnl
        annualized_return = (total_return / n_days) * 252 if n_days > 0 else 0.0

        if len(daily_pnl) >= 2:
            volatility = statistics.stdev(daily_pnl) * math.sqrt(252)
        else:
            volatility = 0.0

        max_dd = abs(summary.max_drawdown_pct) if summary.max_drawdown_pct != 0 else 0.01

        calmar = annualized_return / max_dd if max_dd > 0 else 0.0

        return {
            "annualized_return": annualized_return,
            "volatility": volatility,
            "max_drawdown": max_dd,
            "calmar": calmar,
        }

    def _apply_correlation_penalty(
        self,
        weights: dict[str, float],
        correlation_matrix: dict[str, float],
        metrics: dict[str, dict],
    ) -> dict[str, float]:
        """Reduce allocation for highly correlated bots."""
        penalized = dict(weights)
        bot_ids = list(weights.keys())
        for i, a in enumerate(bot_ids):
            for b in bot_ids[i + 1:]:
                key1 = f"{a}_{b}"
                key2 = f"{b}_{a}"
                corr = correlation_matrix.get(key1, correlation_matrix.get(key2, 0.0))
                if corr > 0.7:
                    penalty = (corr - 0.7) * 0.5  # scale penalty
                    # Penalize the worse-performing bot
                    calmar_a = metrics[a]["calmar"]
                    calmar_b = metrics[b]["calmar"]
                    worse = b if calmar_a >= calmar_b else a
                    penalized[worse] *= max(1.0 - penalty, 0.5)

        total = sum(penalized.values())
        if total > 0:
            return {bid: w / total * 100.0 for bid, w in penalized.items()}
        return weights

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply min/max allocation and max rebalance constraints."""
        c = self.constraints
        clamped = {}
        for bid, w in weights.items():
            clamped[bid] = max(c.min_allocation_pct, min(c.max_allocation_pct, w))

        # Re-normalize to 100%
        total = sum(clamped.values())
        if total > 0:
            clamped = {bid: w / total * 100.0 for bid, w in clamped.items()}

        return clamped

    def _portfolio_calmar(
        self, metrics: dict[str, dict], allocations: dict[str, float],
    ) -> float:
        """Estimate portfolio Calmar as weighted average of bot Calmars."""
        total_alloc = sum(allocations.values())
        if total_alloc == 0:
            return 0.0
        weighted_calmar = sum(
            metrics[bid]["calmar"] * allocations.get(bid, 0.0)
            for bid in metrics
            if bid in allocations
        )
        return weighted_calmar / total_alloc

    def _build_rationale(
        self, bot_id: str, metrics: dict, suggested: float, current: float,
    ) -> str:
        change = suggested - current
        direction = "increase" if change > 0 else "decrease" if change < 0 else "maintain"
        return (
            f"{direction} allocation: Calmar={metrics['calmar']:.2f}, "
            f"vol={metrics['volatility']:.1f}, "
            f"annualized_return={metrics['annualized_return']:.1f}"
        )
