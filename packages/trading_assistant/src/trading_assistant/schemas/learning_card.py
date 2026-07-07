# schemas/learning_card.py
"""LearningCard — structured memory primitive for retrieval-ranked learning signals.

A LearningCard wraps a learning artifact (correction, outcome, hypothesis,
discovery, etc.) with provenance, relevance scoring, and retrieval feedback
metadata.  Cards are the unit of memory that gets ranked and injected into
agent prompts, replacing raw bulk-loaded JSONL entries.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CardType(str, Enum):
    """Taxonomy of learning card origins."""

    CORRECTION = "correction"
    OUTCOME = "outcome"
    HYPOTHESIS = "hypothesis"
    DISCOVERY = "discovery"
    PATTERN = "pattern"
    PREDICTION_VERDICT = "prediction_verdict"
    TRANSFER_RESULT = "transfer_result"
    RECALIBRATION = "recalibration"
    VALIDATOR_BLOCK = "validator_block"
    SYNTHESIS = "synthesis"


class LearningCard(BaseModel):
    """A single, self-contained learning memory primitive."""

    card_id: str = Field(
        default="",
        description="Deterministic ID derived from card_type + source_id.",
    )
    card_type: CardType
    source_id: str = Field(
        description="ID of the originating artifact (suggestion_id, hypothesis_id, etc.).",
    )
    bot_id: str = ""
    title: str = Field(description="One-line human-readable summary.")
    content: str = Field(description="Full content of the learning signal.")
    evidence_summary: str = ""
    tags: list[str] = Field(default_factory=list)

    # Provenance
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_workflow: str = Field(
        default="",
        description="Workflow that produced this card (daily_analysis, weekly_analysis, etc.).",
    )
    source_run_id: str = ""

    # Relevance scoring inputs
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in the learning signal.",
    )
    impact_score: float = Field(
        default=0.0,
        description="Measured impact (positive = helpful, negative = harmful).",
    )
    observation_count: int = Field(
        default=1,
        description="Number of observations supporting this card.",
    )

    # Retrieval feedback (updated over time)
    retrieval_count: int = Field(default=0, description="Times this card was included in a prompt.")
    helpful_count: int = Field(default=0, description="Times marked helpful by downstream analysis.")
    harmful_count: int = Field(default=0, description="Times marked harmful by downstream analysis.")

    # Lifecycle
    superseded_by: str = Field(default="", description="card_id of the card that replaces this one.")
    is_active: bool = True

    def model_post_init(self, __context: Any) -> None:
        if not self.card_id:
            self.card_id = self._compute_id()

    def _compute_id(self) -> str:
        raw = f"{self.card_type.value}:{self.source_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def relevance_score(
        self,
        query_bot_id: str = "",
        query_workflow: str = "",
        query_tags: list[str] | None = None,
        now: datetime | None = None,
    ) -> float:
        """Compute a composite relevance score for retrieval ranking.

        Factors (weighted):
          1. Recency decay      (30%) — half-life 14 days
          2. Impact magnitude    (25%) — abs(impact_score), capped at 1.0
          3. Confidence          (15%) — direct
          4. Retrieval feedback  (15%) — helpful_rate if retrieved 3+ times
          5. Context match       (15%) — bot_id match, workflow match, tag overlap
        """
        now = now or datetime.now(timezone.utc)

        # 1. Recency decay (half-life 14 days)
        age_days = max(0.0, (now - self.created_at).total_seconds() / 86400.0)
        recency = 2.0 ** (-age_days / 14.0)

        # 2. Impact magnitude (absolute, capped at 1.0)
        impact = min(abs(self.impact_score), 1.0)

        # 3. Confidence
        conf = self.confidence

        # 4. Retrieval feedback
        if self.retrieval_count >= 3:
            total_feedback = self.helpful_count + self.harmful_count
            if total_feedback > 0:
                feedback = self.helpful_count / total_feedback
            else:
                feedback = 0.5  # neutral if retrieved but no feedback
        else:
            feedback = 0.5  # neutral prior

        # 5. Context match
        context_score = 0.0
        context_factors = 0
        if query_bot_id:
            context_factors += 1
            if self.bot_id == query_bot_id:
                context_score += 1.0
            elif not self.bot_id:
                context_score += 0.5  # general card — partial match
        if query_workflow:
            context_factors += 1
            if self.source_workflow == query_workflow:
                context_score += 1.0
            elif not self.source_workflow:
                context_score += 0.5  # general card — partial match
        if query_tags:
            context_factors += 1
            overlap = len(set(query_tags) & set(self.tags))
            if self.tags and overlap > 0:
                context_score += overlap / len(set(query_tags))
        context = context_score / max(context_factors, 1)

        return (
            0.30 * recency
            + 0.25 * impact
            + 0.15 * conf
            + 0.15 * feedback
            + 0.15 * context
        )

    def to_prompt_text(self) -> str:
        """Render this card as compact text for prompt injection."""
        parts = [f"[{self.card_type.value.upper()}]"]
        if self.bot_id:
            parts.append(f"[{self.bot_id}]")
        parts.append(self.title)
        lines = [" ".join(parts)]
        if self.content:
            lines.append(self.content)
        if self.evidence_summary:
            lines.append(f"Evidence: {self.evidence_summary}")
        if self.impact_score != 0.0:
            direction = "positive" if self.impact_score > 0 else "negative"
            lines.append(f"Impact: {direction} ({self.impact_score:+.2f})")
        return "\n".join(lines)


class LearningCardIndex(BaseModel):
    """Persistent index of all learning cards (JSONL-backed)."""

    cards: list[LearningCard] = Field(default_factory=list)
    _id_map: dict[str, int] = {}  # card_id -> index in cards list

    def model_post_init(self, __context: Any) -> None:
        self._id_map = {c.card_id: i for i, c in enumerate(self.cards)}

    def add(self, card: LearningCard) -> None:
        """Add or update a card (dedup by card_id). O(1) lookup."""
        if card.card_id in self._id_map:
            self.cards[self._id_map[card.card_id]] = card
        else:
            self._id_map[card.card_id] = len(self.cards)
            self.cards.append(card)

    def get(self, card_id: str) -> LearningCard | None:
        idx = self._id_map.get(card_id)
        if idx is not None and idx < len(self.cards):
            return self.cards[idx]
        return None

    def active_cards(self) -> list[LearningCard]:
        return [c for c in self.cards if c.is_active and not c.superseded_by]

    def ranked(
        self,
        limit: int = 20,
        query_bot_id: str = "",
        query_workflow: str = "",
        query_tags: list[str] | None = None,
    ) -> list[LearningCard]:
        """Return active cards ranked by relevance score."""
        active = self.active_cards()
        scored = [
            (
                card,
                card.relevance_score(
                    query_bot_id=query_bot_id,
                    query_workflow=query_workflow,
                    query_tags=query_tags,
                ),
            )
            for card in active
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [card for card, _ in scored[:limit]]

    def record_retrieval(self, card_ids: list[str]) -> None:
        """Increment retrieval_count for cards included in a prompt."""
        id_set = set(card_ids)
        for card in self.cards:
            if card.card_id in id_set:
                card.retrieval_count += 1

    def record_feedback(
        self, card_id: str, helpful: bool,
    ) -> None:
        """Record whether a retrieved card was helpful or harmful."""
        card = self.get(card_id)
        if card:
            if helpful:
                card.helpful_count += 1
            else:
                card.harmful_count += 1
