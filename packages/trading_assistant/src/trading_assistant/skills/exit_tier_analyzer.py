"""Exit tier analyzer — MFE-based TP tier hit-rate analysis.

Analyzes TP tier effectiveness using MFE data. For each TP tier in a strategy's
exit_profile, computes what % of trades had MFE >= tier R-target, and searches
for optimal tier targets that maximize expected PnL.
"""
from __future__ import annotations

import statistics

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.strategy_profile import StrategyRegistry


class ExitTierAnalyzer:
    """Analyzes exit tier hit rates using MFE data."""

    def __init__(self, strategy_registry: StrategyRegistry | None = None) -> None:
        self._registry = strategy_registry

    def _compute_mfe_r(self, trade: TradeEvent) -> float | None:
        """MFE in R-multiples. Uses mfe_r if available, otherwise estimates from pnl ratio."""
        if trade.mfe_r is not None:
            return trade.mfe_r
        # Fallback: if we have pnl in R-multiples (pnl_r), scale MFE proportionally
        if hasattr(trade, "pnl_r") and trade.pnl_r is not None and trade.pnl_pct is not None:
            if trade.pnl_pct != 0 and trade.mfe_pct is not None:
                return trade.mfe_pct / abs(trade.pnl_pct) * abs(trade.pnl_r)
        return None

    def _compute_mae_r(self, trade: TradeEvent) -> float | None:
        """MAE in R-multiples."""
        if trade.mae_r is not None:
            return trade.mae_r
        return None

    def _get_exit_profile(self, bot_id: str) -> dict | None:
        """Get exit_profile for the primary strategy on this bot."""
        if not self._registry:
            return None
        bot_strategies = self._registry.strategies_for_bot(bot_id)
        for _sid, profile in bot_strategies.items():
            ep = getattr(profile, "exit_profile", None)
            if ep:
                return ep.model_dump(mode="json") if hasattr(ep, "model_dump") else ep
        return None

    def _tier_hit_rates(
        self,
        trades: list[TradeEvent],
        tiers: list[dict],
    ) -> list[dict]:
        """For each TP tier, compute what % of trades had MFE >= tier R-target."""
        mfe_values: list[float] = []
        for t in trades:
            mfe_r = self._compute_mfe_r(t)
            if mfe_r is not None:
                mfe_values.append(mfe_r)

        if not mfe_values:
            return []

        n = len(mfe_values)
        tier_stats: list[dict] = []
        for tier in tiers:
            r_target = tier.get("r_target", 0.0)
            tier_name = tier.get("tier_name", f"TP@{r_target}R")
            partial_pct = tier.get("partial_pct", 1.0)

            hits = sum(1 for m in mfe_values if m >= r_target)
            hit_rate = hits / n if n > 0 else 0.0

            # Average MFE for trades that hit this tier
            hitting_mfes = [m for m in mfe_values if m >= r_target]
            avg_mfe_when_hit = statistics.mean(hitting_mfes) if hitting_mfes else 0.0

            tier_stats.append({
                "tier_name": tier_name,
                "r_target": round(r_target, 4),
                "partial_pct": round(partial_pct, 4),
                "hit_count": hits,
                "total_trades": n,
                "hit_rate": round(hit_rate, 4),
                "avg_mfe_when_hit": round(avg_mfe_when_hit, 4),
            })
        return tier_stats

    def _optimal_tier_targets(
        self,
        trades: list[TradeEvent],
        current_tiers: list[dict],
        search_range: tuple[float, float] = (0.5, 4.0),
        n_points: int = 15,
    ) -> list[dict]:
        """Grid search over R-target values for optimal expected PnL."""
        mfe_values: list[float] = []
        pnls: list[float] = []
        for t in trades:
            mfe_r = self._compute_mfe_r(t)
            if mfe_r is not None:
                mfe_values.append(mfe_r)
                pnls.append(t.pnl)

        if not mfe_values:
            return []

        n = len(mfe_values)
        avg_loss = abs(statistics.mean([p for p in pnls if p <= 0])) if any(p <= 0 for p in pnls) else 1.0

        optimizations: list[dict] = []
        for tier in current_tiers:
            tier_name = tier.get("tier_name", "TP")
            current_target = tier.get("r_target", 1.0)
            partial = tier.get("partial_pct", 1.0)

            best_target = current_target
            best_expected = float("-inf")

            lo, hi = search_range
            step = (hi - lo) / (n_points - 1)
            for i in range(n_points):
                candidate = lo + step * i
                hits = sum(1 for m in mfe_values if m >= candidate)
                hit_rate = hits / n
                # Expected PnL per trade = P(hit) * R_target * partial_pct - P(miss) * avg_loss_R
                expected = hit_rate * candidate * partial - (1 - hit_rate) * avg_loss
                if expected > best_expected:
                    best_expected = expected
                    best_target = candidate

            optimizations.append({
                "tier_name": tier_name,
                "current_target": round(current_target, 4),
                "optimal_target": round(best_target, 4),
                "improvement_pct": round(
                    (best_target - current_target) / max(current_target, 0.01) * 100, 2,
                ),
            })
        return optimizations

    def _stop_placement_analysis(self, trades: list[TradeEvent]) -> dict:
        """MAE distribution analysis — what % of eventual winners touch various MAE levels."""
        winners = [t for t in trades if t.pnl > 0]
        if not winners:
            return {}

        mae_values: list[float] = []
        for t in winners:
            mae_r = self._compute_mae_r(t)
            if mae_r is not None:
                mae_values.append(abs(mae_r))

        if not mae_values:
            return {}

        n = len(mae_values)
        # What % of winners had MAE deeper than various thresholds
        thresholds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        levels: list[dict] = []
        for threshold in thresholds:
            deeper = sum(1 for m in mae_values if m >= threshold)
            levels.append({
                "mae_threshold_r": threshold,
                "pct_winners_touched": round(deeper / n, 4) if n > 0 else 0.0,
            })

        return {
            "total_winners_with_mae": n,
            "avg_winner_mae_r": round(statistics.mean(mae_values), 4),
            "median_winner_mae_r": round(statistics.median(mae_values), 4),
            "levels": levels,
        }

    def analyze(
        self,
        trades: list[TradeEvent],
        bot_id: str,
        period: str = "",
    ) -> dict | None:
        """Full exit tier analysis. Returns None if no exit_profile or MFE data."""
        exit_profile = self._get_exit_profile(bot_id)
        if not exit_profile:
            return None

        tiers = exit_profile.get("tiers", [])
        if not tiers:
            return None

        # Check if we have MFE data
        has_mfe = any(
            t.mfe_r is not None or t.mfe_pct is not None
            for t in trades
        )
        if not has_mfe:
            return None

        tier_hit_rates = self._tier_hit_rates(trades, tiers)
        optimal_targets = self._optimal_tier_targets(trades, tiers)
        stop_analysis = self._stop_placement_analysis(trades)

        return {
            "bot_id": bot_id,
            "period": period,
            "tiers": tier_hit_rates,
            "optimal_targets": optimal_targets,
            "stop_placement": stop_analysis,
            "has_chandelier": exit_profile.get("has_chandelier", False),
        }
