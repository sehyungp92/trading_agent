# tests/test_suggestion_tracking.py
"""Tests for suggestion tracking schemas."""

from trading_assistant.schemas.suggestion_tracking import (
    SuggestionRecord,
    SuggestionOutcome,
    SuggestionStatus,
)


class TestSuggestionStatus:
    def test_all_statuses_exist(self):
        assert SuggestionStatus.PROPOSED == "proposed"
        assert SuggestionStatus.ACCEPTED == "accepted"
        assert SuggestionStatus.REJECTED == "rejected"
        assert SuggestionStatus.IMPLEMENTED == "implemented"


class TestSuggestionRecord:
    def test_creates_with_required_fields(self):
        rec = SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop by 0.5 ATR",
            tier="parameter",
            source_report_id="weekly-2026-02-24",
        )
        assert rec.status == SuggestionStatus.PROPOSED
        assert rec.suggestion_id == "s001"

    def test_mark_implemented(self):
        rec = SuggestionRecord(
            suggestion_id="s002",
            bot_id="bot1",
            title="Relax volume_gate",
            tier="filter",
            source_report_id="weekly-2026-02-24",
        )
        rec.status = SuggestionStatus.IMPLEMENTED
        assert rec.status == SuggestionStatus.IMPLEMENTED


class TestSuggestionOutcome:
    def test_creates_with_deltas(self):
        outcome = SuggestionOutcome(
            suggestion_id="s001",
            implemented_date="2026-02-25",
            pnl_delta_7d=120.0,
            pnl_delta_30d=340.0,
            win_rate_delta_7d=0.05,
            drawdown_delta_7d=-0.02,
        )
        assert outcome.pnl_delta_7d == 120.0
        assert outcome.drawdown_delta_7d == -0.02

    def test_net_positive_property(self):
        outcome = SuggestionOutcome(
            suggestion_id="s001",
            implemented_date="2026-02-25",
            pnl_delta_7d=120.0,
        )
        assert outcome.net_positive_7d is True

    def test_net_negative_property(self):
        outcome = SuggestionOutcome(
            suggestion_id="s002",
            implemented_date="2026-02-25",
            pnl_delta_7d=-50.0,
        )
        assert outcome.net_positive_7d is False
