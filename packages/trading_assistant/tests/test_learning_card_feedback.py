from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.schemas.learning_card import CardType, LearningCard, LearningCardIndex
from trading_assistant.skills.learning_card_store import LearningCardStore


def _make_handlers(tmp_path: Path) -> Handlers:
    runner = MagicMock()
    runner.session_store = MagicMock()
    return Handlers(
        agent_runner=runner,
        event_stream=MagicMock(),
        dispatcher=MagicMock(),
        notification_prefs=MagicMock(),
        curated_dir=tmp_path / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1"],
    )


def _make_card(card_id: str, tags: list[str]) -> LearningCard:
    return LearningCard(
        card_id=card_id,
        card_type=CardType.CORRECTION,
        source_id=f"src_{card_id}",
        title=f"Card {card_id}",
        content="test",
        bot_id="bot1",
        source_workflow="daily_analysis",
        tags=tags,
    )


def _seed_cards(findings_dir: Path, cards: list[LearningCard]) -> None:
    findings_dir.mkdir(parents=True, exist_ok=True)
    store = LearningCardStore(findings_dir)
    index = LearningCardIndex(cards=cards)
    store.save(index)


def _approved_suggestion(category: str = "parameter", bot_id: str = "bot1"):
    suggestion = MagicMock()
    suggestion.category = category
    suggestion.bot_id = bot_id
    return suggestion


def _blocked_suggestion(category: str = "parameter", bot_id: str = "bot1", reason: str = "duplicate idea"):
    blocked = MagicMock()
    blocked.reason = reason
    blocked.suggestion = _approved_suggestion(category=category, bot_id=bot_id)
    return blocked


def _make_validation(approved: list | None = None, blocked: list | None = None):
    validation = MagicMock()
    validation.approved_suggestions = approved or []
    validation.blocked_suggestions = blocked or []
    return validation


def _make_package(card_ids: list[str] | None = None):
    package = MagicMock()
    package.metadata = {}
    if card_ids is not None:
        package.metadata["_learning_card_ids"] = card_ids
    return package


class TestLearningCardFeedback:
    def test_matching_approved_category_marks_card_helpful(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        findings = tmp_path / "memory" / "findings"
        _seed_cards(findings, [
            _make_card("c1", ["category:parameter", "bot:bot1"]),
            _make_card("c2", ["category:other", "bot:bot1"]),
        ])

        validation = _make_validation(approved=[_approved_suggestion(category="parameter")])
        package = _make_package(["c1", "c2"])

        handlers._record_learning_card_feedback_targeted(validation, package)

        index = LearningCardStore(findings).load(force=True)
        assert index.get("c1").helpful_count == 1
        assert index.get("c1").harmful_count == 0
        assert index.get("c2").helpful_count == 0

    def test_matching_blocked_reason_marks_card_harmful(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        findings = tmp_path / "memory" / "findings"
        _seed_cards(findings, [
            _make_card("c1", ["reason:duplicate_idea", "bot:bot1"]),
        ])

        validation = _make_validation(blocked=[_blocked_suggestion(reason="duplicate idea")])
        package = _make_package(["c1"])

        handlers._record_learning_card_feedback_targeted(validation, package)

        index = LearningCardStore(findings).load(force=True)
        assert index.get("c1").harmful_count == 1
        assert index.get("c1").helpful_count == 0

    def test_non_matching_cards_are_ignored(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        findings = tmp_path / "memory" / "findings"
        _seed_cards(findings, [
            _make_card("c1", ["category:stop_loss", "bot:bot1"]),
        ])

        validation = _make_validation(approved=[_approved_suggestion(category="parameter")])
        package = _make_package(["c1"])

        handlers._record_learning_card_feedback_targeted(validation, package)

        index = LearningCardStore(findings).load(force=True)
        assert index.get("c1").helpful_count == 0
        assert index.get("c1").harmful_count == 0

    def test_ambiguous_match_is_not_double_counted(self, tmp_path):
        handlers = _make_handlers(tmp_path)
        findings = tmp_path / "memory" / "findings"
        _seed_cards(findings, [
            _make_card("c1", ["category:parameter", "reason:duplicate_idea", "bot:bot1"]),
        ])

        validation = _make_validation(
            approved=[_approved_suggestion(category="parameter")],
            blocked=[_blocked_suggestion(category="parameter", reason="duplicate idea")],
        )
        package = _make_package(["c1"])

        handlers._record_learning_card_feedback_targeted(validation, package)

        index = LearningCardStore(findings).load(force=True)
        assert index.get("c1").helpful_count == 0
        assert index.get("c1").harmful_count == 0

    def test_relevance_score_changes_with_feedback(self):
        card_helpful = LearningCard(
            card_id="h1",
            card_type=CardType.CORRECTION,
            source_id="src_h1",
            title="helpful card",
            content="test",
            helpful_count=5,
            harmful_count=0,
            retrieval_count=6,
        )
        card_neutral = LearningCard(
            card_id="n1",
            card_type=CardType.CORRECTION,
            source_id="src_n1",
            title="neutral card",
            content="test",
            helpful_count=0,
            harmful_count=0,
            retrieval_count=6,
        )

        now = datetime.now(timezone.utc)
        assert card_helpful.relevance_score(now=now) > card_neutral.relevance_score(now=now)
