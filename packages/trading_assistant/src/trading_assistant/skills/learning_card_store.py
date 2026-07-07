# skills/learning_card_store.py
"""JSONL-backed persistence for LearningCards with factory methods.

Provides:
  - load/save of card index from ``memory/findings/learning_cards.jsonl``
  - Factory methods to create cards from existing synthesis artifacts
  - Bulk conversion from existing JSONL stores (corrections, outcomes, etc.)
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from trading_assistant.schemas.learning_card import (
    CardType,
    LearningCard,
    LearningCardIndex,
)

logger = logging.getLogger(__name__)

_DEFAULT_FILENAME = "learning_cards.jsonl"


def _normalize_lessons(raw) -> list[str]:
    """Normalize lessons_learned which may be a string or list."""
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [l for l in (raw or []) if isinstance(l, str) and l.strip()]


def _slugify_tag(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    chars: list[str] = []
    prev_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            prev_sep = False
        elif not prev_sep:
            chars.append("_")
            prev_sep = True
    return "".join(chars).strip("_")


def _structured_tag(prefix: str, value: str) -> str:
    slug = _slugify_tag(value)
    return f"{prefix}:{slug}" if slug else ""


def _infer_workflow(entry: dict) -> str:
    workflow = str(entry.get("source_workflow", "") or "").strip()
    if workflow:
        return workflow
    run_id = str(
        entry.get("source_run_id")
        or entry.get("run_id")
        or entry.get("source_report_id")
        or ""
    ).strip().lower()
    prefixes = {
        "daily": "daily_analysis",
        "weekly": "weekly_analysis",
        "monthly": "monthly_validation",
        "triage": "triage",
        "discovery": "discovery_analysis",
        "reasoning": "outcome_reasoning",
    }
    for prefix, inferred in prefixes.items():
        if run_id.startswith(prefix):
            return inferred
    return ""


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        normalized = str(tag or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _augment_tags(
    entry: dict,
    *,
    raw_tags: list[str] | None = None,
    categories: list[str] | None = None,
    reasons: list[str] | None = None,
    regimes: list[str] | None = None,
    workflow: str = "",
    bot_id: str = "",
) -> list[str]:
    tags = list(raw_tags or [])
    for category in categories or []:
        category = str(category or "").strip()
        if category:
            tags.append(category)
            structured = _structured_tag("category", category)
            if structured:
                tags.append(structured)
    for reason in reasons or []:
        structured = _structured_tag("reason", str(reason))
        if structured:
            tags.append(structured)
    for regime in regimes or []:
        structured = _structured_tag("regime", str(regime))
        if structured:
            tags.append(structured)
    workflow_tag = _structured_tag("workflow", workflow or _infer_workflow(entry))
    if workflow_tag:
        tags.append(workflow_tag)
    bot_tag = _structured_tag(
        "bot",
        bot_id or entry.get("bot_id", "") or entry.get("target_bot", ""),
    )
    if bot_tag:
        tags.append(bot_tag)
    return _dedupe_tags(tags)


class LearningCardStore:
    """JSONL-backed store for learning cards."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir
        self._path = findings_dir / _DEFAULT_FILENAME
        self._index: LearningCardIndex | None = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self, *, force: bool = False) -> LearningCardIndex:
        """Load cards from JSONL, caching in memory."""
        if self._index is not None and not force:
            return self._index
        index = LearningCardIndex()
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    try:
                        card = LearningCard.model_validate_json(line)
                        index.add(card)
                    except Exception:
                        logger.debug("Skipping malformed learning card line")
        self._index = index
        return index

    def save(self, index: LearningCardIndex | None = None) -> None:
        """Write all cards back to JSONL."""
        idx = index or self._index or LearningCardIndex()
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for card in idx.cards:
                f.write(card.model_dump_json() + "\n")
        self._index = idx

    def add_card(self, card: LearningCard) -> None:
        """Add a card and persist."""
        index = self.load()
        index.add(card)
        self.save(index)

    def ranked_for_prompt(
        self,
        limit: int = 20,
        bot_id: str = "",
        workflow: str = "",
        tags: list[str] | None = None,
    ) -> list[LearningCard]:
        """Return top-ranked active cards for prompt injection."""
        index = self.load()
        return index.ranked(
            limit=limit,
            query_bot_id=bot_id,
            query_workflow=workflow,
            query_tags=tags,
        )

    # -- Factory methods for creating cards from existing artifacts --

    @staticmethod
    def _extract_provenance(entry: dict) -> dict:
        """Extract common provenance fields from a JSONL entry."""
        prov: dict = {}
        if entry.get("source_workflow"):
            prov["source_workflow"] = entry["source_workflow"]
        if entry.get("source_run_id") or entry.get("run_id"):
            prov["source_run_id"] = entry.get("source_run_id") or entry.get("run_id", "")
        for ts_key in ("created_at", "timestamp", "recorded_at"):
            if entry.get(ts_key):
                try:
                    from datetime import datetime
                    prov["created_at"] = datetime.fromisoformat(entry[ts_key])
                except (ValueError, TypeError):
                    pass
                break
        return prov

    @staticmethod
    def card_from_correction(entry: dict) -> LearningCard:
        """Create a card from a corrections.jsonl entry."""
        categories = [c for c in entry.get("categories", []) if c]
        return LearningCard(
            card_type=CardType.CORRECTION,
            source_id=entry.get("id", entry.get("correction_id", "")),
            bot_id=entry.get("bot_id", ""),
            title=entry.get("summary", entry.get("description", "Correction"))[:120],
            content=entry.get("description", entry.get("summary", "")),
            tags=_augment_tags(entry, raw_tags=categories, categories=categories),
            confidence=0.8,
            impact_score=0.3,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_outcome(entry: dict) -> LearningCard:
        """Create a card from an outcomes.jsonl entry."""
        verdict = entry.get("verdict", "unknown")
        impact = 0.5 if verdict == "positive" else (-0.3 if verdict == "negative" else 0.0)
        category = entry.get("category", "")
        regimes = [entry.get("before_regime", ""), entry.get("after_regime", "")]
        return LearningCard(
            card_type=CardType.OUTCOME,
            source_id=entry.get("suggestion_id", ""),
            bot_id=entry.get("bot_id", ""),
            title=f"Outcome: {entry.get('category', 'unknown')} — {verdict}",
            content=entry.get("explanation", ""),
            evidence_summary=entry.get("evidence_base", ""),
            tags=_augment_tags(
                entry,
                raw_tags=[verdict],
                categories=[category],
                regimes=regimes,
            ),
            confidence=0.7 if entry.get("measurement_quality", "").lower() in ("high", "medium") else 0.3,
            impact_score=impact,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_hypothesis(entry: dict) -> LearningCard:
        """Create a card from a hypothesis record."""
        effectiveness = entry.get("effectiveness", 0.0)
        category = entry.get("category", "")
        return LearningCard(
            card_type=CardType.HYPOTHESIS,
            source_id=entry.get("hypothesis_id", entry.get("id", "")),
            bot_id=entry.get("bot_id", ""),
            title=entry.get("title", entry.get("symptom", "Hypothesis"))[:120],
            content=entry.get("description", entry.get("solution", "")),
            tags=_augment_tags(entry, categories=[category]),
            confidence=min(max(effectiveness, 0.0), 1.0),
            impact_score=effectiveness,
            observation_count=entry.get("proposal_count", 1),
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_discovery(entry: dict) -> LearningCard:
        """Create a card from a discoveries.jsonl entry."""
        return LearningCard(
            card_type=CardType.DISCOVERY,
            source_id=entry.get("discovery_id", entry.get("id", "")),
            bot_id=entry.get("bot_id", ""),
            title=entry.get("title", entry.get("pattern", "Discovery"))[:120],
            content=entry.get("description", entry.get("details", "")),
            evidence_summary=entry.get("evidence", ""),
            tags=_augment_tags(entry, raw_tags=entry.get("tags", [])),
            confidence=entry.get("confidence", 0.5),
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_validator_block(entry: dict) -> LearningCard:
        """Create a 'do not repeat' card from a validation_patterns entry."""
        category = entry.get("category", "")
        reason = entry.get("reason", "")
        return LearningCard(
            card_type=CardType.VALIDATOR_BLOCK,
            source_id=entry.get("pattern_id", entry.get("category", "")),
            bot_id=entry.get("bot_id", ""),
            title=f"BLOCKED: {entry.get('category', 'unknown')} — {entry.get('reason', 'blocked')}",
            content=entry.get("reason", entry.get("description", "")),
            tags=_augment_tags(entry, categories=[category], reasons=[reason]),
            confidence=0.9,
            impact_score=-0.5,
            observation_count=entry.get("block_count", 1),
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_validation_log(entry: dict) -> list[LearningCard]:
        """Create cards from a real validation_log.jsonl entry.

        Each entry has ``blocked_details`` — a list of individual blocks.
        Returns one card per blocked detail (may be empty).
        """
        cards: list[LearningCard] = []
        for d in entry.get("blocked_details", []):
            title_raw = d.get("title", "unknown")
            source_hash = hashlib.sha256(title_raw.encode()).hexdigest()[:8]
            category = d.get("category", "")
            reason = d.get("reason", "")
            cards.append(LearningCard(
                card_type=CardType.VALIDATOR_BLOCK,
                source_id=f"vlog:{entry.get('date', '')}:{source_hash}",
                bot_id=d.get("bot_id", ""),
                title=f"BLOCKED: {title_raw}"[:120],
                content=d.get("reason", ""),
                tags=_augment_tags(
                    entry,
                    categories=[category],
                    reasons=[reason],
                    workflow=entry.get("agent_type", ""),
                    bot_id=d.get("bot_id", ""),
                ),
                source_workflow=entry.get("agent_type", "") or _infer_workflow(entry),
                confidence=0.9,
                impact_score=-0.5,
                **LearningCardStore._extract_provenance(entry),
            ))
        return cards

    @staticmethod
    def card_from_outcome_reasoning(entry: dict) -> LearningCard:
        """Create a card from an outcome_reasonings.jsonl entry."""
        lessons = _normalize_lessons(entry.get("lessons_learned", []))
        category = entry.get("category", "")
        return LearningCard(
            card_type=CardType.SYNTHESIS,
            source_id=entry.get("suggestion_id", entry.get("id", "")),
            bot_id=entry.get("bot_id", ""),
            title=f"Outcome reasoning: {entry.get('category', 'unknown')}"[:120],
            content=entry.get("mechanism", "") + (
                "\n" + "; ".join(lessons) if lessons else ""
            ),
            tags=_augment_tags(
                entry,
                raw_tags=(["transferable"] if entry.get("transferable") else []),
                categories=[category],
            ),
            confidence=entry.get("revised_confidence", 0.5),
            impact_score=0.4 if entry.get("genuine_effect", True) else -0.3,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_recalibration(entry: dict) -> LearningCard:
        """Create a card from a recalibrations.jsonl entry."""
        revised = entry.get("revised_confidence", 0.5)
        category = entry.get("category", "")
        return LearningCard(
            card_type=CardType.RECALIBRATION,
            source_id=entry.get("suggestion_id", entry.get("id", "")),
            bot_id=entry.get("bot_id", ""),
            title=f"Recalibration: {entry.get('category', 'unknown')} → {revised:.0%}"[:120],
            content="; ".join(_normalize_lessons(entry.get("lessons_learned", []))) or "Confidence recalibrated",
            tags=_augment_tags(entry, categories=[category]),
            confidence=revised,
            impact_score=0.3,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_spurious_outcome(entry: dict) -> LearningCard:
        """Create a card from a spurious_outcomes.jsonl entry."""
        return LearningCard(
            card_type=CardType.OUTCOME,
            source_id=f"spurious:{entry.get('suggestion_id', entry.get('id', ''))}",
            bot_id=entry.get("bot_id", ""),
            title=f"SPURIOUS: {entry.get('suggestion_id', 'unknown')}"[:120],
            content=entry.get("mechanism", "") + (
                f"\nConfounders: {', '.join(entry.get('confounders', []))}"
                if entry.get("confounders") else ""
            ),
            tags=_augment_tags(entry, raw_tags=["spurious"]),
            confidence=0.7,
            impact_score=-0.3,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_transfer_outcome(entry: dict) -> LearningCard:
        """Create a card from a transfer_outcomes.jsonl entry."""
        verdict = entry.get("verdict", "?")
        impact = 0.4 if verdict == "positive" else (-0.3 if verdict == "negative" else 0.0)
        confidence = 0.7 if entry.get("regime_matched", True) else 0.4
        content_parts = []
        if entry.get("source_bot"):
            content_parts.append(f"Source: {entry['source_bot']}")
        if entry.get("pnl_delta_7d") is not None:
            content_parts.append(f"PnL delta 7d: {entry['pnl_delta_7d']:+.2f}")
        if entry.get("win_rate_delta_7d") is not None:
            content_parts.append(f"Win rate delta 7d: {entry['win_rate_delta_7d']:+.2%}")
        if entry.get("regime_matched") is not None:
            content_parts.append(f"Regime matched: {entry['regime_matched']}")
        raw_tags = ["transfer", verdict]
        if entry.get("regime_matched") is False:
            raw_tags.append("regime_mismatch")
        return LearningCard(
            card_type=CardType.TRANSFER_RESULT,
            source_id=f"transfer:{entry.get('pattern_id', '')}:{entry.get('target_bot', '')}",
            bot_id=entry.get("target_bot", ""),
            title=f"Transfer {verdict}: {entry.get('pattern_id', '')} \u2192 {entry.get('target_bot', '')}"[:120],
            content="; ".join(content_parts) if content_parts else f"Transfer {verdict}",
            tags=_augment_tags(entry, raw_tags=raw_tags, bot_id=entry.get("target_bot", "")),
            confidence=confidence,
            impact_score=impact,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_retrospective(entry: dict) -> LearningCard:
        """Create a card from a retrospective_synthesis.jsonl entry."""
        content_parts = []
        categories: list[str] = []
        for item in entry.get("what_worked", []):
            title = item.get("title", item) if isinstance(item, dict) else str(item)
            content_parts.append(f"Worked: {title}")
        for item in entry.get("what_failed", []):
            title = item.get("title", item) if isinstance(item, dict) else str(item)
            content_parts.append(f"Failed: {title}")
        for item in entry.get("discard", []):
            reason = item.get("reason", item) if isinstance(item, dict) else str(item)
            content_parts.append(f"Discard: {reason}")
            if isinstance(item, dict) and item.get("category"):
                categories.append(item["category"])
        lessons = _normalize_lessons(entry.get("lessons_learned", []))
        for lesson in lessons:
            content_parts.append(f"Lesson: {lesson}")
        return LearningCard(
            card_type=CardType.SYNTHESIS,
            source_id=f"retro:{entry.get('week_start', '')}",
            bot_id="",
            title=f"Retrospective {entry.get('week_start', '')} \u2192 {entry.get('week_end', '')}"[:120],
            content="\n".join(content_parts) if content_parts else "Weekly retrospective",
            tags=_augment_tags(
                entry,
                raw_tags=["retrospective"],
                categories=categories,
                workflow="weekly_analysis",
            ),
            source_workflow="weekly_analysis",
            confidence=0.8,
            impact_score=0.3,
            **LearningCardStore._extract_provenance(entry),
        )

    def ingest_from_existing(self) -> int:
        """Bulk-create cards from existing JSONL stores.

        Scans corrections, outcomes, discoveries, and other learning artifact
        JSONL files and creates cards for entries not already in the index.
        Returns count of new cards.
        """
        index = self.load()
        existing_ids = {c.card_id for c in index.cards}
        new_count = 0

        # Map filename → factory method (one card per entry)
        sources = [
            ("corrections.jsonl", self.card_from_correction),
            ("outcomes.jsonl", self.card_from_outcome),
            ("discoveries.jsonl", self.card_from_discovery),
            ("outcome_reasonings.jsonl", self.card_from_outcome_reasoning),
            ("recalibrations.jsonl", self.card_from_recalibration),
            ("spurious_outcomes.jsonl", self.card_from_spurious_outcome),
            ("hypotheses.jsonl", self.card_from_hypothesis),
            ("transfer_outcomes.jsonl", self.card_from_transfer_outcome),
            ("retrospective_synthesis.jsonl", self.card_from_retrospective),
        ]

        for filename, factory in sources:
            path = self._findings_dir / filename
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    card = factory(entry)
                    if card.card_id not in existing_ids:
                        index.add(card)
                        existing_ids.add(card.card_id)
                        new_count += 1
                except Exception:
                    continue

        # validation_log.jsonl — multi-card expansion (one card per blocked detail)
        vlog_path = self._findings_dir / "validation_log.jsonl"
        if vlog_path.exists():
            for line in vlog_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    for card in self.card_from_validation_log(entry):
                        if card.card_id not in existing_ids:
                            index.add(card)
                            existing_ids.add(card.card_id)
                            new_count += 1
                except Exception:
                    continue

        if new_count:
            self.save(index)
            logger.info("Ingested %d new learning cards from existing stores", new_count)

        return new_count
