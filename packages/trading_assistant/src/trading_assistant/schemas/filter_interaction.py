# schemas/filter_interaction.py
"""Schemas for multi-filter interaction analysis — detect redundant or complementary filter pairs."""
from __future__ import annotations

from pydantic import BaseModel


class FilterPairInteraction(BaseModel):
    """Analysis of how two filters interact across trades and missed opportunities."""

    filter_a: str
    filter_b: str
    bot_id: str
    trades_both_active: int = 0  # trades where both filters passed
    win_rate_both: float = 0.0
    pnl_both: float = 0.0
    trades_only_a: int = 0  # trades where A active, B not present
    win_rate_only_a: float = 0.0
    pnl_only_a: float = 0.0
    trades_only_b: int = 0  # trades where B active, A not present
    win_rate_only_b: float = 0.0
    pnl_only_b: float = 0.0
    missed_by_a: int = 0  # missed opps blocked by A
    missed_by_b: int = 0  # missed opps blocked by B
    redundancy_score: float = 0.0  # 0-1, high = filters overlap significantly
    interaction_type: str = "independent"  # "redundant", "complementary", "independent"
    recommendation: str = ""


class FilterInteractionReport(BaseModel):
    """Report of filter pair interactions for a single bot."""

    bot_id: str
    date: str
    pairs: list[FilterPairInteraction] = []
    total_filters_analyzed: int = 0
    flagged_pairs: int = 0  # pairs with actionable findings
