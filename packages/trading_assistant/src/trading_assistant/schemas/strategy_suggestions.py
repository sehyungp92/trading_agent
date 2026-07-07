# schemas/strategy_suggestions.py
"""Strategy suggestion schemas — 5-tier refinement output models.

Tier 1: Parameter suggestions (automated, high confidence)
Tier 2: Filter adjustments (automated, medium confidence)
Tier 3: Strategy variants (semi-automated, requires human judgment)
Tier 4: New hypotheses (human-led, Claude-assisted)
Tier 5: Portfolio-level changes (allocation, risk caps, coordination, drawdown tiers)
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, computed_field

from trading_assistant.schemas.detection_context import DetectionContext


class SuggestionTier(str, Enum):
    PARAMETER = "parameter"
    FILTER = "filter"
    STRATEGY_VARIANT = "strategy_variant"
    HYPOTHESIS = "hypothesis"
    PORTFOLIO = "portfolio"


class StrategySuggestion(BaseModel):
    """A single strategy refinement suggestion."""

    tier: SuggestionTier
    bot_id: str = ""
    strategy_id: str = ""
    strategy_archetype: str = ""
    archetype_note: str = ""
    title: str
    description: str
    current_value: str = ""
    suggested_value: str = ""
    evidence_days: int = 0
    estimated_impact_pnl: float = 0.0
    confidence: float = 0.0  # 0–1
    simulation_assumptions: list[str] = []
    requires_human_judgment: bool = False
    detection_context: DetectionContext | None = None
    engine: str = ""  # engine tag if targeting specific engine (e.g., "REVERSAL")
    regime_condition: str = ""  # regime name if regime-conditional (e.g., "volatile")


class RefinementReport(BaseModel):
    """Aggregated strategy refinement output for a week."""

    week_start: str
    week_end: str
    suggestions: list[StrategySuggestion] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def suggestions_by_tier(self) -> dict[str, int]:
        counts = Counter(s.tier.value for s in self.suggestions)
        return dict(counts)
