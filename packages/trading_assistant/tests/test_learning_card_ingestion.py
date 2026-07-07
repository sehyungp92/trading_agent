"""Tests for expanded learning card ingestion pipeline (Phase A)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_assistant.schemas.learning_card import CardType
from trading_assistant.skills.learning_card_store import LearningCardStore


@pytest.fixture
def findings_dir(tmp_path: Path) -> Path:
    d = tmp_path / "findings"
    d.mkdir()
    return d


class TestCardFromOutcomeReasoning:
    def test_produces_synthesis_card(self):
        entry = {
            "suggestion_id": "s1",
            "bot_id": "bot_a",
            "category": "exit_timing",
            "mechanism": "Exit was too slow",
            "lessons_learned": ["Tighten trailing stop"],
            "transferable": True,
            "genuine_effect": True,
            "revised_confidence": 0.8,
        }
        card = LearningCardStore.card_from_outcome_reasoning(entry)
        assert card.card_type == CardType.SYNTHESIS
        assert card.source_id == "s1"
        assert card.bot_id == "bot_a"
        assert "transferable" in card.tags
        assert card.confidence == 0.8
        assert card.impact_score == 0.4


class TestCardFromRecalibration:
    def test_produces_recalibration_card(self):
        entry = {
            "suggestion_id": "s2",
            "bot_id": "bot_b",
            "category": "position_sizing",
            "revised_confidence": 0.3,
            "lessons_learned": ["Overfit to recent data"],
        }
        card = LearningCardStore.card_from_recalibration(entry)
        assert card.card_type == CardType.RECALIBRATION
        assert card.source_id == "s2"
        assert "Recalibration:" in card.title
        assert "30%" in card.title
        assert card.confidence == 0.3


class TestCardFromSpuriousOutcome:
    def test_uses_spurious_prefix_for_source_id(self):
        entry = {
            "suggestion_id": "s3",
            "bot_id": "bot_c",
            "mechanism": "Coincidental regime shift",
            "confounders": ["vol_spike", "news_event"],
        }
        card = LearningCardStore.card_from_spurious_outcome(entry)
        assert card.card_type == CardType.OUTCOME
        assert card.source_id == "spurious:s3"
        assert "SPURIOUS" in card.title
        assert "spurious" in card.tags
        assert card.impact_score == -0.3
        assert "Confounders:" in card.content


class TestExpandedIngestion:
    def test_ingests_outcome_reasonings(self, findings_dir: Path):
        (findings_dir / "outcome_reasonings.jsonl").write_text(
            json.dumps({"suggestion_id": "or1", "category": "entry", "mechanism": "test"}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1
        cards = store.load().cards
        assert any(c.card_type == CardType.SYNTHESIS for c in cards)

    def test_ingests_recalibrations(self, findings_dir: Path):
        (findings_dir / "recalibrations.jsonl").write_text(
            json.dumps({"suggestion_id": "r1", "category": "sizing", "revised_confidence": 0.4}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1
        cards = store.load().cards
        assert any(c.card_type == CardType.RECALIBRATION for c in cards)

    def test_ingests_hypotheses(self, findings_dir: Path):
        (findings_dir / "hypotheses.jsonl").write_text(
            json.dumps({"hypothesis_id": "h1", "title": "Test hyp", "category": "entry"}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1
        cards = store.load().cards
        assert any(c.card_type == CardType.HYPOTHESIS for c in cards)

    def test_ingests_validation_log(self, findings_dir: Path):
        entry = {
            "date": "2026-04-17",
            "approved_count": 1,
            "blocked_count": 1,
            "blocked_details": [
                {"title": "Tighten trailing stop", "reason": "poor track record in exit_timing", "bot_id": "bot_a"},
            ],
            "timestamp": "2026-04-17T00:00:00+00:00",
        }
        (findings_dir / "validation_log.jsonl").write_text(json.dumps(entry) + "\n")
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1
        cards = store.load().cards
        assert any(c.card_type == CardType.VALIDATOR_BLOCK for c in cards)
        card = [c for c in cards if c.card_type == CardType.VALIDATOR_BLOCK][0]
        assert "Tighten trailing stop" in card.title
        assert card.bot_id == "bot_a"
        assert card.content == "poor track record in exit_timing"

    def test_validation_log_empty_blocked_details(self, findings_dir: Path):
        entry = {"date": "2026-04-17", "approved_count": 3, "blocked_count": 0, "blocked_details": []}
        (findings_dir / "validation_log.jsonl").write_text(json.dumps(entry) + "\n")
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 0

    def test_validation_log_multiple_blocks(self, findings_dir: Path):
        entry = {
            "date": "2026-04-17",
            "approved_count": 0,
            "blocked_count": 2,
            "blocked_details": [
                {"title": "Reduce position size", "reason": "poor track record", "bot_id": "bot_a"},
                {"title": "Change entry signal", "reason": "rejected by user", "bot_id": "bot_b"},
            ],
        }
        (findings_dir / "validation_log.jsonl").write_text(json.dumps(entry) + "\n")
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 2

    def test_dedup_prevents_double_ingestion(self, findings_dir: Path):
        (findings_dir / "corrections.jsonl").write_text(
            json.dumps({"id": "c1", "summary": "Fix"}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        first = store.ingest_from_existing()
        second = store.ingest_from_existing()
        assert first == 1
        assert second == 0
        assert len(store.load().cards) == 1

    def test_missing_files_gracefully_skipped(self, findings_dir: Path):
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 0

    def test_malformed_lines_skipped(self, findings_dir: Path):
        (findings_dir / "corrections.jsonl").write_text(
            "not valid json\n"
            + json.dumps({"id": "c2", "summary": "Good"}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1

    def test_spurious_and_genuine_same_suggestion_id_different_card_ids(self, findings_dir: Path):
        """Spurious and genuine outcomes with same suggestion_id should produce different card_ids."""
        (findings_dir / "outcomes.jsonl").write_text(
            json.dumps({"suggestion_id": "shared_id", "verdict": "positive"}) + "\n"
        )
        (findings_dir / "spurious_outcomes.jsonl").write_text(
            json.dumps({"suggestion_id": "shared_id", "mechanism": "spurious"}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 2
        cards = store.load().cards
        ids = {c.card_id for c in cards}
        assert len(ids) == 2  # No collision

    def test_ingests_transfer_outcomes(self, findings_dir: Path):
        entry = {
            "pattern_id": "p1",
            "source_bot": "bot_a",
            "target_bot": "bot_b",
            "verdict": "positive",
            "pnl_delta_7d": 150.0,
            "win_rate_delta_7d": 0.05,
            "regime_matched": True,
        }
        (findings_dir / "transfer_outcomes.jsonl").write_text(json.dumps(entry) + "\n")
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1
        card = store.load().cards[0]
        assert card.card_type == CardType.TRANSFER_RESULT
        assert card.bot_id == "bot_b"
        assert "positive" in card.title
        assert card.impact_score == 0.4

    def test_ingests_retrospective_synthesis(self, findings_dir: Path):
        entry = {
            "week_start": "2026-04-07",
            "week_end": "2026-04-13",
            "what_worked": [{"title": "Trailing stop improvement"}],
            "what_failed": [{"title": "Aggressive sizing"}],
            "discard": [],
            "lessons_learned": ["Be conservative with sizing"],
        }
        (findings_dir / "retrospective_synthesis.jsonl").write_text(json.dumps(entry) + "\n")
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 1
        card = store.load().cards[0]
        assert card.card_type == CardType.SYNTHESIS
        assert "Retrospective" in card.title
        assert "retrospective" in card.tags
        assert card.bot_id == ""

    def test_returns_total_new_card_count(self, findings_dir: Path):
        (findings_dir / "corrections.jsonl").write_text(
            json.dumps({"id": "c1", "summary": "A"}) + "\n"
            + json.dumps({"id": "c2", "summary": "B"}) + "\n"
        )
        (findings_dir / "discoveries.jsonl").write_text(
            json.dumps({"discovery_id": "d1", "title": "Find"}) + "\n"
        )
        store = LearningCardStore(findings_dir)
        count = store.ingest_from_existing()
        assert count == 3


class TestLessonsLearnedNormalization:
    """Verify string lessons_learned doesn't produce character-split garbage."""

    def test_outcome_reasoning_string_lessons(self):
        entry = {
            "suggestion_id": "s1",
            "category": "exit_timing",
            "mechanism": "Exit too slow",
            "lessons_learned": "Tighten trailing stop for volatile regimes",
            "genuine_effect": True,
            "revised_confidence": 0.7,
        }
        card = LearningCardStore.card_from_outcome_reasoning(entry)
        assert "Tighten trailing stop" in card.content
        # Must NOT be character-split
        assert "T; i; g; h" not in card.content

    def test_recalibration_string_lessons(self):
        entry = {
            "suggestion_id": "s2",
            "category": "sizing",
            "revised_confidence": 0.4,
            "lessons_learned": "Overfit to recent data",
        }
        card = LearningCardStore.card_from_recalibration(entry)
        assert card.content == "Overfit to recent data"
        assert len(card.content.split("; ")) == 1  # Single item, not chars

    def test_outcome_reasoning_empty_string_lessons(self):
        entry = {
            "suggestion_id": "s3",
            "category": "entry",
            "mechanism": "Test mechanism",
            "lessons_learned": "   ",
            "genuine_effect": True,
        }
        card = LearningCardStore.card_from_outcome_reasoning(entry)
        # Empty string should not add garbage
        assert card.content == "Test mechanism"

    def test_recalibration_empty_string_lessons(self):
        entry = {
            "suggestion_id": "s4",
            "category": "entry",
            "revised_confidence": 0.5,
            "lessons_learned": "",
        }
        card = LearningCardStore.card_from_recalibration(entry)
        assert card.content == "Confidence recalibrated"


class TestCardFromTransferOutcome:
    def test_produces_transfer_result_card(self):
        entry = {
            "pattern_id": "p1",
            "source_bot": "bot_a",
            "target_bot": "bot_b",
            "verdict": "positive",
            "pnl_delta_7d": 100.0,
            "win_rate_delta_7d": 0.03,
            "regime_matched": True,
        }
        card = LearningCardStore.card_from_transfer_outcome(entry)
        assert card.card_type == CardType.TRANSFER_RESULT
        assert card.source_id == "transfer:p1:bot_b"
        assert card.bot_id == "bot_b"
        assert "positive" in card.title
        assert card.impact_score == 0.4
        assert card.confidence == 0.7

    def test_negative_verdict_scores(self):
        entry = {"pattern_id": "p2", "target_bot": "bot_c", "verdict": "negative", "regime_matched": False}
        card = LearningCardStore.card_from_transfer_outcome(entry)
        assert card.impact_score == -0.3
        assert card.confidence == 0.4

    def test_neutral_verdict_scores(self):
        entry = {"pattern_id": "p3", "target_bot": "bot_d", "verdict": "neutral"}
        card = LearningCardStore.card_from_transfer_outcome(entry)
        assert card.impact_score == 0.0


class TestCardFromRetrospective:
    def test_produces_synthesis_card(self):
        entry = {
            "week_start": "2026-04-07",
            "week_end": "2026-04-13",
            "what_worked": [{"title": "Trailing stop"}],
            "what_failed": [{"title": "Over-sizing"}],
            "discard": [{"reason": "Stale pattern"}],
            "lessons_learned": ["Size conservatively"],
        }
        card = LearningCardStore.card_from_retrospective(entry)
        assert card.card_type == CardType.SYNTHESIS
        assert card.source_id == "retro:2026-04-07"
        assert card.bot_id == ""
        assert "Retrospective" in card.title
        assert "retrospective" in card.tags
        assert "Worked: Trailing stop" in card.content
        assert "Failed: Over-sizing" in card.content
        assert "Lesson: Size conservatively" in card.content

    def test_string_lessons_learned_normalized(self):
        entry = {
            "week_start": "2026-04-07",
            "week_end": "2026-04-13",
            "lessons_learned": "Single lesson as string",
        }
        card = LearningCardStore.card_from_retrospective(entry)
        assert "Lesson: Single lesson as string" in card.content


class TestValidationLogCards:
    def test_produces_cards_from_blocked_details(self):
        entry = {
            "date": "2026-04-17",
            "approved_count": 1,
            "blocked_count": 1,
            "blocked_details": [
                {"title": "Tighten trailing stop", "reason": "poor track record in exit_timing", "bot_id": "bot_a"},
            ],
            "timestamp": "2026-04-17T00:00:00+00:00",
        }
        cards = LearningCardStore.card_from_validation_log(entry)
        assert len(cards) == 1
        card = cards[0]
        assert card.card_type == CardType.VALIDATOR_BLOCK
        assert "BLOCKED: Tighten trailing stop" == card.title
        assert card.bot_id == "bot_a"
        assert card.content == "poor track record in exit_timing"
        assert card.source_id.startswith("vlog:2026-04-17:")

    def test_empty_blocked_details_returns_empty(self):
        entry = {"date": "2026-04-17", "blocked_details": []}
        cards = LearningCardStore.card_from_validation_log(entry)
        assert cards == []

    def test_missing_blocked_details_returns_empty(self):
        entry = {"date": "2026-04-17"}
        cards = LearningCardStore.card_from_validation_log(entry)
        assert cards == []
