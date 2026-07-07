"""Tests for soul.md quantification reinforcement in prompts (C1)."""
from __future__ import annotations


from trading_assistant.analysis.prompt_assembler import _INSTRUCTIONS
from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS


def test_daily_prompt_includes_quantification_instruction():
    assert "quantified expected impact" in _INSTRUCTIONS
    assert "drawdown change" in _INSTRUCTIONS
    assert "evidence base" in _INSTRUCTIONS


def test_weekly_prompt_includes_calmar_requirement():
    assert "Calmar ratio impact" in _WEEKLY_INSTRUCTIONS


def test_instruction_text_matches_soul_md_criteria():
    # soul.md says: suggestions must be quantified with evidence
    assert "statistical significance" in _INSTRUCTIONS
    assert "trade count" in _INSTRUCTIONS
    assert "quantification" in _INSTRUCTIONS
