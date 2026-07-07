# skills/filter_interaction_analyzer.py
"""Multi-filter interaction analysis — detect redundant or complementary filter pairs.

Analyzes co-activation patterns across trades and missed opportunities
to identify filters that overlap significantly or complement each other.
"""
from __future__ import annotations

from itertools import combinations

from trading_assistant.schemas.events import MissedOpportunityEvent, TradeEvent
from trading_assistant.schemas.filter_interaction import FilterInteractionReport, FilterPairInteraction


def _jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


class FilterInteractionAnalyzer:
    """Analyzes pairwise filter interactions across trades and missed opportunities."""

    def __init__(self, bot_id: str, date: str) -> None:
        self._bot_id = bot_id
        self._date = date

    def analyze(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
    ) -> FilterInteractionReport:
        """Analyze filter pair interactions.

        Args:
            trades: Completed trades with active_filters populated.
            missed: Missed opportunities with blocked_by populated.
        """
        # Collect all filter names
        all_filters: set[str] = set()
        for t in trades:
            all_filters.update(t.active_filters)
        for m in missed:
            if m.blocked_by:
                all_filters.add(m.blocked_by)

        if len(all_filters) < 2:
            return FilterInteractionReport(
                bot_id=self._bot_id,
                date=self._date,
                total_filters_analyzed=len(all_filters),
            )

        # Build per-trade filter membership sets (for Jaccard)
        filter_trade_sets: dict[str, set[int]] = {f: set() for f in all_filters}
        for i, t in enumerate(trades):
            for f in t.active_filters:
                filter_trade_sets[f].add(i)

        # Count missed by each filter
        missed_by: dict[str, int] = {f: 0 for f in all_filters}
        for m in missed:
            if m.blocked_by and m.blocked_by in missed_by:
                missed_by[m.blocked_by] += 1

        # Analyze each pair
        pairs: list[FilterPairInteraction] = []
        sorted_filters = sorted(all_filters)

        for fa, fb in combinations(sorted_filters, 2):
            pair = self._analyze_pair(fa, fb, trades, filter_trade_sets, missed_by)
            if pair.interaction_type != "independent":
                pairs.append(pair)

        # Sort by redundancy score descending
        pairs.sort(key=lambda p: p.redundancy_score, reverse=True)
        flagged = sum(1 for p in pairs if p.interaction_type != "independent")

        return FilterInteractionReport(
            bot_id=self._bot_id,
            date=self._date,
            pairs=pairs,
            total_filters_analyzed=len(all_filters),
            flagged_pairs=flagged,
        )

    def _analyze_pair(
        self,
        fa: str,
        fb: str,
        trades: list[TradeEvent],
        filter_trade_sets: dict[str, set[int]],
        missed_by: dict[str, int],
    ) -> FilterPairInteraction:
        """Analyze a single filter pair."""
        both_trades: list[TradeEvent] = []
        only_a_trades: list[TradeEvent] = []
        only_b_trades: list[TradeEvent] = []

        for t in trades:
            has_a = fa in t.active_filters
            has_b = fb in t.active_filters
            if has_a and has_b:
                both_trades.append(t)
            elif has_a and not has_b:
                only_a_trades.append(t)
            elif has_b and not has_a:
                only_b_trades.append(t)

        # Compute partition stats
        def _stats(tl: list[TradeEvent]) -> tuple[int, float, float]:
            if not tl:
                return 0, 0.0, 0.0
            wins = sum(1 for t in tl if t.pnl > 0)
            return len(tl), wins / len(tl), sum(t.pnl for t in tl)

        n_both, wr_both, pnl_both = _stats(both_trades)
        n_a, wr_a, pnl_a = _stats(only_a_trades)
        n_b, wr_b, pnl_b = _stats(only_b_trades)

        # Redundancy: Jaccard on trade sets + co-activation pattern
        jaccard = _jaccard(filter_trade_sets.get(fa, set()), filter_trade_sets.get(fb, set()))

        # Classify interaction
        interaction, recommendation = self._classify(
            jaccard, n_both, n_a, n_b, wr_both, wr_a, wr_b,
            pnl_both, pnl_a, pnl_b,
            missed_by.get(fa, 0), missed_by.get(fb, 0),
        )

        return FilterPairInteraction(
            filter_a=fa,
            filter_b=fb,
            bot_id=self._bot_id,
            trades_both_active=n_both,
            win_rate_both=round(wr_both, 4),
            pnl_both=round(pnl_both, 2),
            trades_only_a=n_a,
            win_rate_only_a=round(wr_a, 4),
            pnl_only_a=round(pnl_a, 2),
            trades_only_b=n_b,
            win_rate_only_b=round(wr_b, 4),
            pnl_only_b=round(pnl_b, 2),
            missed_by_a=missed_by.get(fa, 0),
            missed_by_b=missed_by.get(fb, 0),
            redundancy_score=round(jaccard, 4),
            interaction_type=interaction,
            recommendation=recommendation,
        )

    @staticmethod
    def _classify(
        jaccard: float,
        n_both: int, n_a: int, n_b: int,
        wr_both: float, wr_a: float, wr_b: float,
        pnl_both: float, pnl_a: float, pnl_b: float,
        missed_a: int, missed_b: int,
    ) -> tuple[str, str]:
        """Classify filter pair interaction and generate recommendation."""
        total = n_both + n_a + n_b
        if total == 0:
            return "independent", ""

        # Redundant: high Jaccard (always co-activate) AND similar blocking patterns
        if jaccard > 0.7:
            recommendation = (
                f"Filters co-activate {jaccard:.0%} of the time. "
                f"Consider consolidating into a single composite filter. "
                f"Missed by A: {missed_a}, by B: {missed_b}."
            )
            return "redundant", recommendation

        # Complementary: low overlap but together they improve outcomes
        # One filter catches what the other misses
        if jaccard < 0.3 and n_a > 0 and n_b > 0:
            # If win rates diverge meaningfully, they're complementary
            if abs(wr_a - wr_b) > 0.15 or (wr_both > wr_a and wr_both > wr_b):
                recommendation = (
                    f"Filters are complementary (overlap {jaccard:.0%}). "
                    f"Win rate when both active: {wr_both:.0%} vs "
                    f"A-only: {wr_a:.0%}, B-only: {wr_b:.0%}. "
                    f"Keep both for coverage."
                )
                return "complementary", recommendation

        return "independent", ""
