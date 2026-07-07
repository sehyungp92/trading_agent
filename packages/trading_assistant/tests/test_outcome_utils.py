# tests/test_outcome_utils.py
"""Tests for the shared outcome utility functions."""
from __future__ import annotations

from trading_assistant.skills._outcome_utils import is_conclusive_outcome, is_positive_outcome


class TestIsPositiveOutcome:
    def test_positive_verdict(self):
        assert is_positive_outcome({"verdict": "positive"}) is True

    def test_negative_verdict(self):
        assert is_positive_outcome({"verdict": "negative"}) is False

    def test_inconclusive_verdict_not_positive(self):
        assert is_positive_outcome({"verdict": "inconclusive"}) is False

    def test_monthly_keep_verdict_is_positive(self):
        assert is_positive_outcome({"verdict": "keep", "outcome_source": "monthly"}) is True

    def test_monthly_rollback_verdict_not_positive(self):
        assert is_positive_outcome({"verdict": "rollback", "outcome_source": "monthly"}) is False

    def test_pnl_delta_fallback_positive(self):
        assert is_positive_outcome({"pnl_delta": 100}) is True

    def test_pnl_delta_fallback_negative(self):
        assert is_positive_outcome({"pnl_delta": -50}) is False

    def test_pnl_delta_7d_fallback(self):
        assert is_positive_outcome({"pnl_delta_7d": 10}) is True

    def test_pnl_delta_30d_fallback(self):
        assert is_positive_outcome({"pnl_delta_30d": 10}) is True

    def test_empty_outcome_not_positive(self):
        assert is_positive_outcome({}) is False


class TestIsConclusiveOutcome:
    def test_positive_is_conclusive(self):
        assert is_conclusive_outcome({"verdict": "positive"}) is True

    def test_negative_is_conclusive(self):
        assert is_conclusive_outcome({"verdict": "negative"}) is True

    def test_inconclusive_excluded(self):
        assert is_conclusive_outcome({"verdict": "inconclusive"}) is False

    def test_insufficient_data_excluded(self):
        assert is_conclusive_outcome({"verdict": "insufficient_data"}) is False

    def test_monthly_watch_excluded(self):
        assert is_conclusive_outcome({"verdict": "watch", "outcome_source": "monthly"}) is False

    def test_empty_verdict_is_conclusive(self):
        """Empty verdict (legacy data) should be treated as conclusive."""
        assert is_conclusive_outcome({"verdict": ""}) is True
        assert is_conclusive_outcome({}) is True

    def test_case_insensitive(self):
        assert is_conclusive_outcome({"verdict": "INCONCLUSIVE"}) is False
        assert is_conclusive_outcome({"verdict": "Insufficient_Data"}) is False

    def test_none_verdict_is_conclusive(self):
        """None verdict (corrupt data) should not crash and should be treated as conclusive."""
        assert is_conclusive_outcome({"verdict": None}) is True
