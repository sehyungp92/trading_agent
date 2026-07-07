"""DriftAnalyzer — deterministic allocation drift computation.

Compares recommended allocations against actual allocations and tracks
drift trends over time. Pure computation, no LLM calls.
"""
from __future__ import annotations

from trading_assistant.schemas.allocation_history import AllocationSnapshot, BotAllocationSnapshot
from trading_assistant.schemas.portfolio_allocation import PortfolioAllocationReport


class DriftAnalyzer:
    """Static methods for allocation drift analysis."""

    @staticmethod
    def compute_snapshot(
        alloc_report: PortfolioAllocationReport,
        actual_allocations: dict[str, float],
    ) -> AllocationSnapshot:
        """Build an AllocationSnapshot pairing recommended vs actual per bot.

        Args:
            alloc_report: The portfolio allocator's output with recommendations.
            actual_allocations: Map of bot_id -> actual allocation percentage.

        Returns:
            AllocationSnapshot with per-bot drift and aggregate metrics.
        """
        bot_snaps: list[BotAllocationSnapshot] = []
        for rec in alloc_report.recommendations:
            actual = actual_allocations.get(rec.bot_id, 0.0)
            drift = rec.suggested_allocation_pct - actual
            bot_snaps.append(BotAllocationSnapshot(
                bot_id=rec.bot_id,
                recommended_pct=rec.suggested_allocation_pct,
                actual_pct=actual,
                drift_pct=round(drift, 4),
                abs_drift_pct=round(abs(drift), 4),
            ))

        sum_abs_drift = sum(b.abs_drift_pct for b in bot_snaps)
        # Normalized total drift: sum(|drift|) / 2
        total_drift = round(sum_abs_drift / 2, 4) if bot_snaps else 0.0
        max_drift = round(max((b.abs_drift_pct for b in bot_snaps), default=0.0), 4)

        return AllocationSnapshot(
            date=alloc_report.week_start,
            week_start=alloc_report.week_start,
            week_end=alloc_report.week_end,
            bot_allocations=bot_snaps,
            total_drift_pct=total_drift,
            max_single_drift_pct=max_drift,
        )

    @staticmethod
    def compute_drift_trend(
        snapshots: list[dict],
        weeks: int = 8,
    ) -> dict:
        """Compute drift trend from snapshot history.

        Args:
            snapshots: List of snapshot dicts (from load_snapshots).
            weeks: Number of recent weeks to consider.

        Returns:
            Dict with weekly_drift, trend_direction, avg_drift_pct,
            persistent_drifters.
        """
        if not snapshots:
            return {
                "weekly_drift": [],
                "trend_direction": "stable",
                "avg_drift_pct": 0.0,
                "persistent_drifters": [],
            }

        # Sort by date ascending and take last N weeks
        sorted_snaps = sorted(snapshots, key=lambda s: s.get("date", ""))
        recent = sorted_snaps[-weeks:]

        weekly_drift = [
            {
                "date": s.get("date", ""),
                "total_drift_pct": s.get("total_drift_pct", 0.0),
                "max_single_drift_pct": s.get("max_single_drift_pct", 0.0),
            }
            for s in recent
        ]

        drift_values = [w["total_drift_pct"] for w in weekly_drift]
        avg_drift = round(sum(drift_values) / len(drift_values), 4) if drift_values else 0.0

        # Trend: compare first-half avg vs second-half avg, ±2% threshold
        trend_direction = "stable"
        if len(drift_values) >= 2:
            mid = len(drift_values) // 2
            first_half = drift_values[:mid]
            second_half = drift_values[mid:]
            first_avg = sum(first_half) / len(first_half) if first_half else 0.0
            second_avg = sum(second_half) / len(second_half) if second_half else 0.0
            diff = second_avg - first_avg
            if diff > 2.0:
                trend_direction = "increasing"
            elif diff < -2.0:
                trend_direction = "decreasing"

        # Persistent drifters: bot_ids in >50% of weeks with >5% abs drift
        n_weeks = len(recent)
        bot_high_drift_count: dict[str, int] = {}
        for s in recent:
            for ba in s.get("bot_allocations", []):
                if ba.get("abs_drift_pct", 0.0) > 5.0:
                    bid = ba.get("bot_id", "")
                    if bid:
                        bot_high_drift_count[bid] = bot_high_drift_count.get(bid, 0) + 1

        threshold = n_weeks / 2
        persistent_drifters = sorted(
            bid for bid, count in bot_high_drift_count.items()
            if count > threshold
        )

        return {
            "weekly_drift": weekly_drift,
            "trend_direction": trend_direction,
            "avg_drift_pct": avg_drift,
            "persistent_drifters": persistent_drifters,
        }
