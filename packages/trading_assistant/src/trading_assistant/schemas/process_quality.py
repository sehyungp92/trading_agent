# schemas/process_quality.py
"""Process quality scoring schemas — controlled root-cause taxonomy and scoring models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class RootCause(str, Enum):
    """Controlled taxonomy for trade process quality degradation."""
    REGIME_MISMATCH = "regime_mismatch"
    WEAK_SIGNAL = "weak_signal"
    LATE_ENTRY = "late_entry"
    EARLY_EXIT = "early_exit"
    SLIPPAGE_SPIKE = "slippage_spike"
    FILTER_BLOCKED_GOOD = "filter_blocked_good"
    RISK_CAP_HIT = "risk_cap_hit"
    DATA_GAP = "data_gap"
    ORDER_REJECT = "order_reject"
    LATENCY_SPIKE = "latency_spike"


class ScoringDeduction(BaseModel):
    """A single deduction applied to the process quality score."""
    root_cause: RootCause
    points: int
    evidence: str


class ProcessQualityResult(BaseModel):
    """Result of process quality scoring for a single trade."""
    score: int  # 0-100
    root_causes: list[RootCause] = []
    deductions: list[ScoringDeduction] = []
    evidence_refs: list[str] = []
