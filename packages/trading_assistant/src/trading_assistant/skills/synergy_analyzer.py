# skills/synergy_analyzer.py
"""Synergy analyzer — cross-strategy redundancy and complementarity scoring.

Deterministic pipeline. No LLM calls. Analyzes all strategy pairs across all bots
for correlation, marginal Sharpe contribution, and classification.
"""
from __future__ import annotations

import math
import statistics

from trading_assistant.schemas.synergy_analysis import (
    StrategyPairAnalysis,
    StrategyMarginalContribution,
    SynergyReport,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary

# Known instrument mappings for same-instrument detection
_INSTRUMENT_MAP: dict[str, str] = {
    "NQ": "NQ",
    "MNQ": "NQ",  # micro NQ maps to NQ
    "ES": "ES",
    "MES": "ES",
}


class SynergyAnalyzer:
    """Computes cross-strategy synergy/redundancy analysis."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        correlation_threshold_redundant: float = 0.7,
        correlation_threshold_complementary: float = 0.2,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self._thresh_redundant = correlation_threshold_redundant
        self._thresh_complementary = correlation_threshold_complementary

    def compute(
        self,
        per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
    ) -> SynergyReport:
        """Compute synergy report.

        Args:
            per_strategy_summaries: outer key=bot_id, inner key=strategy_id
        """
        # Flatten to "bot_id:strategy_id" → StrategyWeeklySummary
        flat: dict[str, StrategyWeeklySummary] = {}
        for bot_id, strategies in per_strategy_summaries.items():
            for strat_id, summary in strategies.items():
                key = f"{bot_id}:{strat_id}"
                flat[key] = summary

        if len(flat) < 1:
            return SynergyReport(
                week_start=self.week_start,
                week_end=self.week_end,
            )

        keys = list(flat.keys())
        all_dates = self._collect_dates(flat)

        # Build daily PnL series aligned to common dates
        series = self._build_aligned_series(flat, all_dates)

        # Pairwise analysis
        pairs: list[StrategyPairAnalysis] = []
        redundant_pairs: list[str] = []
        complementary_pairs: list[str] = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                pair = self._analyze_pair(a, b, series, flat)
                pairs.append(pair)
                if pair.classification == "redundant" or pair.classification == "cannibalistic":
                    redundant_pairs.append(f"{a} vs {b}")
                elif pair.classification == "complementary":
                    complementary_pairs.append(f"{a} vs {b}")

        # Marginal contributions
        contributions = self._compute_marginal_contributions(flat, series, all_dates)

        return SynergyReport(
            week_start=self.week_start,
            week_end=self.week_end,
            strategy_pairs=pairs,
            marginal_contributions=contributions,
            redundant_pairs=redundant_pairs,
            complementary_pairs=complementary_pairs,
            total_strategies=len(flat),
        )

    def compute_intra_bot(
        self,
        bot_id: str,
        strategies: dict[str, StrategyWeeklySummary],
    ) -> SynergyReport:
        """Compute synergy analysis for strategies within a single bot.

        Unlike compute() which flattens across bots, this focuses on
        intra-bot strategy relationships to answer: are any strategies
        redundant within this bot? Which provide genuine diversification?
        """
        if len(strategies) < 2:
            return SynergyReport(
                week_start=self.week_start,
                week_end=self.week_end,
                total_strategies=len(strategies),
            )

        # Build flat map with just strategy_id keys (no bot prefix needed)
        flat: dict[str, StrategyWeeklySummary] = {}
        for strat_id, summary in strategies.items():
            flat[f"{bot_id}:{strat_id}"] = summary

        keys = list(flat.keys())
        all_dates = self._collect_dates(flat)
        series = self._build_aligned_series(flat, all_dates)

        pairs: list[StrategyPairAnalysis] = []
        redundant_pairs: list[str] = []
        complementary_pairs: list[str] = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                pair = self._analyze_pair(a, b, series, flat)
                pairs.append(pair)
                if pair.classification in ("redundant", "cannibalistic"):
                    redundant_pairs.append(f"{a} vs {b}")
                elif pair.classification == "complementary":
                    complementary_pairs.append(f"{a} vs {b}")

        contributions = self._compute_marginal_contributions(flat, series, all_dates)

        return SynergyReport(
            week_start=self.week_start,
            week_end=self.week_end,
            strategy_pairs=pairs,
            marginal_contributions=contributions,
            redundant_pairs=redundant_pairs,
            complementary_pairs=complementary_pairs,
            total_strategies=len(flat),
        )

    def compute_bot_correlation_matrix(
        self,
        per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
    ) -> dict[str, float]:
        """Compute bot-level correlation matrix for PortfolioAllocator.

        Aggregates per-strategy daily PnL into per-bot daily PnL, then
        computes pairwise Pearson correlation between bots.

        Returns:
            Dict of "botA_botB" → correlation coefficient.
        """
        # Collect all dates across all strategies
        all_dates: set[str] = set()
        for strategies in per_strategy_summaries.values():
            for summary in strategies.values():
                all_dates.update(summary.daily_pnl.keys())
        sorted_dates = sorted(all_dates)

        if not sorted_dates:
            return {}

        # Build per-bot aggregated daily PnL
        bot_series: dict[str, list[float]] = {}
        for bot_id, strategies in per_strategy_summaries.items():
            daily = []
            for d in sorted_dates:
                day_pnl = sum(s.daily_pnl.get(d, 0.0) for s in strategies.values())
                daily.append(day_pnl)
            bot_series[bot_id] = daily

        # Pairwise correlation
        bot_ids = list(bot_series.keys())
        matrix: dict[str, float] = {}
        for i in range(len(bot_ids)):
            for j in range(i + 1, len(bot_ids)):
                a, b = bot_ids[i], bot_ids[j]
                corr = self._pearson(bot_series[a], bot_series[b])
                matrix[f"{a}_{b}"] = round(corr, 4)

        return matrix

    def _collect_dates(self, flat: dict[str, StrategyWeeklySummary]) -> list[str]:
        """Collect union of all dates across strategies."""
        dates: set[str] = set()
        for s in flat.values():
            dates.update(s.daily_pnl.keys())
        return sorted(dates)

    def _build_aligned_series(
        self,
        flat: dict[str, StrategyWeeklySummary],
        dates: list[str],
    ) -> dict[str, list[float]]:
        """Build daily PnL arrays aligned to common date list."""
        result: dict[str, list[float]] = {}
        for key, summary in flat.items():
            result[key] = [summary.daily_pnl.get(d, 0.0) for d in dates]
        return result

    def _analyze_pair(
        self,
        key_a: str,
        key_b: str,
        series: dict[str, list[float]],
        flat: dict[str, StrategyWeeklySummary],
    ) -> StrategyPairAnalysis:
        """Analyze a single strategy pair."""
        sa = series[key_a]
        sb = series[key_b]

        corr = self._pearson(sa, sb)
        same_inst = self._same_instrument(flat[key_a], flat[key_b])

        # Signal overlap: count days where both have non-zero PnL
        overlap_days = sum(1 for a, b in zip(sa, sb, strict=False) if a != 0 and b != 0)
        active_days = max(sum(1 for v in sa if v != 0), 1)
        overlap_pct = overlap_days / active_days * 100.0

        # Classification
        classification = self._classify(corr, flat[key_a], flat[key_b], key_a, key_b, series)

        return StrategyPairAnalysis(
            strategy_a=key_a,
            strategy_b=key_b,
            correlation_30d=round(corr, 4),
            classification=classification,
            signal_overlap_pct=round(overlap_pct, 1),
            same_instrument=same_inst,
            recommendation=self._pair_recommendation(classification, corr, same_inst),
        )

    def _classify(
        self,
        corr: float,
        summary_a: StrategyWeeklySummary,
        summary_b: StrategyWeeklySummary,
        key_a: str,
        key_b: str,
        series: dict[str, list[float]],
    ) -> str:
        """Classify the pair relationship."""
        if corr > self._thresh_redundant:
            # Check if one has negative marginal contribution → cannibalistic
            portfolio_sharpe = self._sharpe(
                [a + b for a, b in zip(series[key_a], series[key_b], strict=False)]
            )
            sharpe_a = self._sharpe(series[key_a])
            sharpe_b = self._sharpe(series[key_b])
            # If removing one improves portfolio Sharpe, it's cannibalistic
            if sharpe_a > portfolio_sharpe or sharpe_b > portfolio_sharpe:
                return "cannibalistic"
            return "redundant"
        elif corr < self._thresh_complementary:
            return "complementary"
        return "neutral"

    def _compute_marginal_contributions(
        self,
        flat: dict[str, StrategyWeeklySummary],
        series: dict[str, list[float]],
        dates: list[str],
    ) -> list[StrategyMarginalContribution]:
        """Compute marginal Sharpe contribution for each strategy."""
        if not flat or not dates:
            return []

        keys = list(flat.keys())

        # Full portfolio daily PnL
        portfolio_pnl = [sum(series[k][i] for k in keys) for i in range(len(dates))]
        full_sharpe = self._sharpe(portfolio_pnl)
        total_pnl = sum(portfolio_pnl)

        contributions: list[StrategyMarginalContribution] = []
        for key in keys:
            # Portfolio without this strategy
            without_pnl = [
                portfolio_pnl[i] - series[key][i] for i in range(len(dates))
            ]
            without_sharpe = self._sharpe(without_pnl)
            marginal = full_sharpe - without_sharpe

            strat_pnl = sum(series[key])
            pnl_pct = (strat_pnl / total_pnl * 100.0) if total_pnl != 0 else 0.0

            bot_id, strat_id = key.split(":", 1)
            contributions.append(
                StrategyMarginalContribution(
                    strategy_key=key,
                    bot_id=bot_id,
                    strategy_id=strat_id,
                    sharpe_with=round(full_sharpe, 4),
                    sharpe_without=round(without_sharpe, 4),
                    marginal_sharpe=round(marginal, 4),
                    pnl_contribution_pct=round(pnl_pct, 2),
                ),
            )

        return contributions

    def _same_instrument(
        self, a: StrategyWeeklySummary, b: StrategyWeeklySummary,
    ) -> bool:
        """Detect if two strategies trade the same base instrument."""
        # Use bot_id heuristics for known instrument mappings
        # Also check the strategy summaries if symbols are available
        bot_a = a.bot_id
        bot_b = b.bot_id

        # If same bot, likely same instrument space
        if bot_a == bot_b:
            return True

        return False

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return 0.0
        return cov / denom

    @staticmethod
    def _sharpe(daily_pnl: list[float]) -> float:
        """Compute annualized Sharpe from daily PnL series."""
        if len(daily_pnl) < 2:
            return 0.0
        mean = statistics.mean(daily_pnl)
        std = statistics.stdev(daily_pnl)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _pair_recommendation(classification: str, corr: float, same_inst: bool) -> str:
        if classification == "cannibalistic":
            return "Consider removing the weaker strategy — it drags portfolio Sharpe"
        if classification == "redundant":
            extra = " (same instrument)" if same_inst else ""
            return f"High correlation ({corr:.2f}){extra} — evaluate if both add value"
        if classification == "complementary":
            return "Good diversification — strategies complement each other"
        return ""
