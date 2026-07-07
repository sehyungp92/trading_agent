"""Structural analyzer — strategy lifecycle, architecture mismatch, filter ROI.

Deterministic pipeline. No LLM calls. Classifies strategies as growing/mature/decaying,
detects signal-exit architecture mismatches, and computes filter ROI.
"""
from __future__ import annotations

import math
import statistics

from trading_assistant.schemas.structural_analysis import (
    ArchitectureMismatch,
    FilterROI,
    StrategyLifecycleStatus,
    StructuralReport,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary

# Architecture mismatch rules: (signal_type, exit_type) → mismatch info
_MISMATCH_RULES: list[dict] = [
    {
        "signal_type": "momentum",
        "exit_type": "fixed_tp",
        "mismatch_type": "momentum_fixed_tp",
        "current": "momentum signal + fixed take profit",
        "recommended": "momentum signal + trailing stop",
        "evidence": "Momentum signals benefit from trend continuation; fixed TP caps upside.",
    },
    {
        "signal_type": "mean_reversion",
        "exit_type": "trailing",
        "mismatch_type": "mean_reversion_trailing_stop",
        "current": "mean reversion signal + trailing stop",
        "recommended": "mean reversion signal + fixed take profit or time-based exit",
        "evidence": "Mean reversion trades target a known level; trailing stops add whipsaw risk.",
    },
    {
        "signal_type": "breakout",
        "exit_type": "time_based",
        "mismatch_type": "breakout_time_based",
        "current": "breakout signal + time-based exit",
        "recommended": "breakout signal + trailing stop or momentum-based exit",
        "evidence": "Breakouts can run indefinitely; time-based exits cut winning trades short.",
    },
]


class StructuralAnalyzer:
    """Computes structural analysis: lifecycle, mismatches, filter ROI."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        decay_sharpe_threshold: float = 0.3,
        growth_sharpe_threshold: float = 0.2,
        mismatch_rules: list[dict] | None = None,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self._decay_threshold = decay_sharpe_threshold
        self._growth_threshold = growth_sharpe_threshold
        self._mismatch_rules = mismatch_rules or _MISMATCH_RULES

    def compute(
        self,
        per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
        strategy_metadata: dict[str, dict] | None = None,
        filter_data: dict[str, dict] | None = None,
    ) -> StructuralReport:
        """Compute the full structural report.

        Args:
            per_strategy_summaries: outer key=bot_id, inner key=strategy_id
            strategy_metadata: keyed by "bot:strategy" → {"signal_type": ..., "exit_type": ...}
            filter_data: keyed by "bot:strategy" → {"filter_name": {saved: N, cost: N, net_pnl: F}}
        """
        lifecycle_statuses: list[StrategyLifecycleStatus] = []
        growing: list[str] = []
        decaying: list[str] = []

        for bot_id, strategies in per_strategy_summaries.items():
            for strat_id, summary in strategies.items():
                status = self._classify_lifecycle(bot_id, strat_id, summary)
                lifecycle_statuses.append(status)
                key = f"{bot_id}:{strat_id}"
                if status.phase == "growing":
                    growing.append(key)
                elif status.phase == "decaying":
                    decaying.append(key)

        mismatches = self._detect_mismatches(strategy_metadata or {})
        filter_roi = self._compute_filter_roi(filter_data or {})
        proposed = self._build_proposals(lifecycle_statuses, mismatches, filter_roi)

        return StructuralReport(
            week_start=self.week_start,
            week_end=self.week_end,
            lifecycle_statuses=lifecycle_statuses,
            architecture_mismatches=mismatches,
            filter_roi=filter_roi,
            growing_strategies=growing,
            decaying_strategies=decaying,
            proposed_changes=proposed,
        )

    def _classify_lifecycle(
        self, bot_id: str, strategy_id: str, summary: StrategyWeeklySummary,
    ) -> StrategyLifecycleStatus:
        """Classify strategy as growing/mature/decaying from daily PnL series."""
        daily_pnl = summary.daily_pnl
        sorted_dates = sorted(daily_pnl.keys())
        len(sorted_dates)

        # Build Sharpe over 30d/60d/90d windows (use available data, capped)
        sharpe_30d = self._sharpe_window(daily_pnl, sorted_dates, 30)
        sharpe_60d = self._sharpe_window(daily_pnl, sorted_dates, 60)
        sharpe_90d = self._sharpe_window(daily_pnl, sorted_dates, 90)

        # Sharpe trend: (30d - 90d) / 90d ratio
        if sharpe_90d != 0:
            sharpe_trend = (sharpe_30d - sharpe_90d) / abs(sharpe_90d)
        else:
            sharpe_trend = 0.0

        # Classify
        if sharpe_30d > sharpe_90d and (sharpe_30d - sharpe_90d) >= self._growth_threshold:
            phase = "growing"
        elif sharpe_90d > sharpe_30d and (sharpe_90d - sharpe_30d) >= self._decay_threshold:
            phase = "decaying"
        else:
            phase = "mature"

        # Edge half-life estimation (if decaying)
        half_life: float | None = None
        if phase == "decaying" and sharpe_30d > 0 and sharpe_90d > 0:
            # Extrapolate days until Sharpe reaches half of 90d value
            decay_rate = (sharpe_90d - sharpe_30d) / 60.0  # per day
            if decay_rate > 0:
                target = sharpe_30d / 2.0
                half_life = (sharpe_30d - target) / decay_rate

        return StrategyLifecycleStatus(
            bot_id=bot_id,
            strategy_id=strategy_id,
            phase=phase,
            sharpe_30d=round(sharpe_30d, 4),
            sharpe_60d=round(sharpe_60d, 4),
            sharpe_90d=round(sharpe_90d, 4),
            sharpe_trend=round(sharpe_trend, 4),
            edge_half_life_days=round(half_life, 1) if half_life is not None else None,
            trade_count_90d=summary.total_trades,
        )

    def _detect_mismatches(
        self, strategy_metadata: dict[str, dict],
    ) -> list[ArchitectureMismatch]:
        """Check each strategy against mismatch rules."""
        mismatches: list[ArchitectureMismatch] = []

        for key, meta in strategy_metadata.items():
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            bot_id, strat_id = parts

            signal_type = meta.get("signal_type", "").lower()
            exit_type = meta.get("exit_type", "").lower()
            if not signal_type or not exit_type:
                continue

            for rule in self._mismatch_rules:
                if signal_type == rule["signal_type"] and exit_type == rule["exit_type"]:
                    mismatches.append(ArchitectureMismatch(
                        bot_id=bot_id,
                        strategy_id=strat_id,
                        mismatch_type=rule["mismatch_type"],
                        current_setup=rule["current"],
                        recommended_setup=rule["recommended"],
                        evidence=rule["evidence"],
                        confidence=0.7,
                    ))

        return mismatches

    def _compute_filter_roi(
        self, filter_data: dict[str, dict],
    ) -> list[FilterROI]:
        """Compute ROI for each filter from aggregated filter data."""
        results: list[FilterROI] = []

        for key, filters in filter_data.items():
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            bot_id, strat_id = parts

            for filter_name, stats in filters.items():
                saved = stats.get("saved", 0)
                cost = stats.get("cost", 0)
                net_pnl = stats.get("net_pnl", 0.0)
                missed_value = stats.get("missed_value", 0.0)

                roi = net_pnl / missed_value if missed_value != 0 else 0.0

                results.append(FilterROI(
                    bot_id=bot_id,
                    strategy_id=strat_id,
                    filter_name=filter_name,
                    blocks_saved_count=saved,
                    blocks_cost_count=cost,
                    net_pnl_impact=net_pnl,
                    roi=round(roi, 4),
                ))

        return results

    def _build_proposals(
        self,
        lifecycle: list[StrategyLifecycleStatus],
        mismatches: list[ArchitectureMismatch],
        filter_roi: list[FilterROI],
    ) -> list[dict]:
        """Build actionable proposals from lifecycle + mismatch + filter findings."""
        proposals: list[dict] = []

        for status in lifecycle:
            if status.phase == "decaying":
                proposals.append({
                    "strategy": f"{status.bot_id}:{status.strategy_id}",
                    "category": "lifecycle",
                    "proposal": f"Strategy is decaying (30d Sharpe {status.sharpe_30d:.2f} vs 90d {status.sharpe_90d:.2f}). "
                                f"Review signal quality and consider parameter refresh or retirement.",
                    "effort": "medium",
                    "impact": "high",
                    "reversibility": "high",
                })

        for mm in mismatches:
            proposals.append({
                "strategy": f"{mm.bot_id}:{mm.strategy_id}",
                "category": "architecture",
                "proposal": f"Architecture mismatch: {mm.current_setup}. "
                            f"Recommend: {mm.recommended_setup}. {mm.evidence}",
                "effort": "medium",
                "impact": "medium",
                "reversibility": "high",
            })

        for f in filter_roi:
            if f.net_pnl_impact < 0:
                proposals.append({
                    "strategy": f"{f.bot_id}:{f.strategy_id}",
                    "category": "filter",
                    "proposal": f"Filter '{f.filter_name}' has negative ROI ({f.roi:.2f}). "
                                f"Net PnL impact: ${f.net_pnl_impact:.0f}. Consider relaxing or removing.",
                    "effort": "low",
                    "impact": "low",
                    "reversibility": "high",
                })

        return proposals

    @staticmethod
    def _sharpe_window(
        daily_pnl: dict[str, float], sorted_dates: list[str], window: int,
    ) -> float:
        """Compute annualized Sharpe from the last N days of daily PnL."""
        dates = sorted_dates[-window:] if len(sorted_dates) >= window else sorted_dates
        if len(dates) < 2:
            return 0.0
        values = [daily_pnl.get(d, 0.0) for d in dates]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)
