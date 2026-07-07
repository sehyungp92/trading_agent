"""Filter sensitivity analysis — estimates impact of threshold changes.

When margin_pct data is available (from bot-side B4), computes sensitivity curves
showing what happens at +-10/20/30% threshold adjustments.

Without margin_pct, computes net impact and recommendations from outcome data.
"""
from __future__ import annotations

from collections import defaultdict

from trading_assistant.schemas.events import MissedOpportunityEvent
from trading_assistant.schemas.filter_sensitivity import (
    FilterSensitivityCurve,
    FilterSensitivityReport,
    SensitivityPoint,
)


class FilterSensitivityAnalyzer:
    def __init__(self, bot_id: str, date: str) -> None:
        self._bot_id = bot_id
        self._date = date

    def analyze(self, missed: list[MissedOpportunityEvent]) -> FilterSensitivityReport:
        """Analyze filter sensitivity from missed opportunity data."""
        by_filter: dict[str, list[MissedOpportunityEvent]] = defaultdict(list)
        for m in missed:
            if m.blocked_by:
                by_filter[m.blocked_by].append(m)

        curves = []
        for filter_name, events in sorted(by_filter.items()):
            curves.append(self._analyze_filter(filter_name, events))

        return FilterSensitivityReport(
            bot_id=self._bot_id, date=self._date, curves=curves,
        )

    def _analyze_filter(
        self, filter_name: str, events: list[MissedOpportunityEvent]
    ) -> FilterSensitivityCurve:
        outcomes = [(m, m.outcome_24h or 0.0) for m in events]
        winners = sum(1 for _, o in outcomes if o > 0)
        losers = sum(1 for _, o in outcomes if o <= 0)
        net_impact = sum(o for _, o in outcomes)

        # Recommendation
        if net_impact > 0:
            rec = (
                f"Removing {filter_name} would have added ${net_impact:.0f} "
                f"({winners} winners, {losers} losers blocked). Consider relaxing."
            )
        elif net_impact < 0:
            rec = (
                f"{filter_name} saved ${abs(net_impact):.0f} by blocking "
                f"{losers} losing trades. Keep current threshold."
            )
        else:
            rec = f"{filter_name} has neutral impact. Review for simplification."

        # Sensitivity points (only when margin_pct available)
        sensitivity_points = self._compute_sensitivity_points(events)

        return FilterSensitivityCurve(
            filter_name=filter_name,
            bot_id=self._bot_id,
            current_block_count=len(events),
            current_net_impact=net_impact,
            blocked_winners=winners,
            blocked_losers=losers,
            recommendation=rec,
            sensitivity_points=sensitivity_points,
        )

    def _compute_sensitivity_points(
        self, events: list[MissedOpportunityEvent]
    ) -> list[SensitivityPoint]:
        """Compute sensitivity curve from margin_pct data if available."""
        with_margin = [(m, m.margin_pct) for m in events if m.margin_pct is not None]
        if not with_margin:
            return []

        points = []
        for adjustment in [10, 20, 30]:
            # Trades within margin_pct <= adjustment would be captured by widening
            captured = [m for m, margin in with_margin if margin <= adjustment]
            pnl_impact = sum(m.outcome_24h or 0.0 for m in captured)
            points.append(SensitivityPoint(
                threshold_adjustment_pct=float(adjustment),
                estimated_additional_trades=len(captured),
                estimated_pnl_impact=pnl_impact,
            ))

        return points
