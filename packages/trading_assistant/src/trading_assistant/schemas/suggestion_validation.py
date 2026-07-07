# schemas/suggestion_validation.py
"""Suggestion validation schemas — evidence from backtesting/replay before recording."""
from __future__ import annotations

from pydantic import BaseModel


class ValidationEvidence(BaseModel):
    """Evidence from validating a suggestion against historical data."""
    validated: bool = False
    method: str = "not_testable"  # backtest_replay, regime_analysis, not_testable
    baseline_metrics: dict = {}   # {pnl, win_rate, sharpe, max_dd}
    proposed_metrics: dict = {}   # same keys, with proposed change applied
    improvement_pct: float = 0.0
    sample_size: int = 0
    regime_breakdown: dict = {}   # {regime: {baseline: {...}, proposed: {...}}}
    notes: str = ""


class SuggestionValidationResult(BaseModel):
    """Result of validating a single suggestion."""
    suggestion_id: str = ""
    bot_id: str = ""
    target_param: str = ""
    proposed_value: float | None = None
    evidence: ValidationEvidence = ValidationEvidence()
    degradation_detected: bool = False
    requires_review: bool = False
