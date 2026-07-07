# schemas/suggestion_scoring.py
"""Category-level scoring schemas for suggestion track record analysis."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CategoryScore(BaseModel):
    """Track record for a (bot_id, [strategy_id], category) tuple.

    `strategy_id=None` means bot-wide aggregate (existing behavior preserved
    for legacy callers). Non-null `strategy_id` rows are emitted alongside the
    aggregate so prompts can cite per-strategy track records.
    """

    bot_id: str
    strategy_id: Optional[str] = None
    category: str
    win_rate: float = 0.0
    avg_pnl_delta: float = 0.0
    sample_size: int = 0
    confidence_multiplier: float = 1.0


class CategoryScorecard(BaseModel):
    """Aggregated category-level scores across all bots."""

    scores: list[CategoryScore] = []

    def get_score(
        self,
        bot_id: str,
        category: str,
        strategy_id: Optional[str] = None,
    ) -> CategoryScore | None:
        """Return the most specific matching row.

        If `strategy_id` is supplied, prefers a (bot_id, strategy_id, category)
        row; falls back to the bot-wide (bot_id, None, category) row when no
        per-strategy row exists. Existing callers passing only (bot_id, category)
        always hit the bot-wide aggregate row.
        """
        if strategy_id is not None:
            for s in self.scores:
                if (
                    s.bot_id == bot_id
                    and s.strategy_id == strategy_id
                    and s.category == category
                ):
                    return s
        for s in self.scores:
            if s.bot_id == bot_id and s.strategy_id is None and s.category == category:
                return s
        return None
