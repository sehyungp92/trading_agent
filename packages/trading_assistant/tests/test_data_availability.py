"""Tests for MemoryIndex data availability checks (B1)."""
from __future__ import annotations


from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.schemas.memory import MemoryIndex


def test_check_returns_true_when_data_exists():
    index = MemoryIndex(
        curated_dates_by_bot={"bot_alpha": ["2026-03-01", "2026-03-02"]},
    )
    result = ContextBuilder.check_data_availability(index, "bot_alpha", "2026-03-01")
    assert result["has_curated"] is True


def test_check_returns_false_when_data_missing():
    index = MemoryIndex(
        curated_dates_by_bot={"bot_alpha": ["2026-03-01"]},
    )
    result = ContextBuilder.check_data_availability(index, "bot_alpha", "2026-03-05")
    assert result["has_curated"] is False


def test_handles_missing_index_gracefully():
    result = ContextBuilder.check_data_availability(None, "bot_alpha", "2026-03-01")
    assert result["has_curated"] is None
    assert result["available_dates"] == []


def test_available_dates_returned():
    index = MemoryIndex(
        curated_dates_by_bot={"bot_alpha": ["2026-03-01", "2026-03-02", "2026-03-03"]},
    )
    result = ContextBuilder.check_data_availability(index, "bot_alpha", "2026-03-02")
    assert len(result["available_dates"]) == 3


def test_unknown_bot_returns_empty():
    index = MemoryIndex(
        curated_dates_by_bot={"bot_alpha": ["2026-03-01"]},
    )
    result = ContextBuilder.check_data_availability(index, "bot_unknown", "2026-03-01")
    assert result["has_curated"] is False
    assert result["available_dates"] == []
