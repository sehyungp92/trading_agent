"""Shared validation result schemas for leakage and robustness checks."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LeakageAuditEntry(BaseModel):
    """One entry in a temporal leakage audit log."""

    feature_name: str
    computed_at: str
    latest_data_used: str
    passed: bool
    violation: str = ""


class RobustnessResult(BaseModel):
    """Results from neighborhood and regime stability tests."""

    neighborhood_scores: dict[str, float] = Field(default_factory=dict)
    neighborhood_stable: bool = False
    regime_pnl: dict[str, float] = Field(default_factory=dict)
    profitable_regime_count: int = 0
    regime_stable: bool = False
    robustness_score: float = 0.0


class SafetyFlag(BaseModel):
    """A safety warning attached to validation results."""

    flag_type: str
    description: str
    severity: str = "medium"


__all__ = ["LeakageAuditEntry", "RobustnessResult", "SafetyFlag"]
