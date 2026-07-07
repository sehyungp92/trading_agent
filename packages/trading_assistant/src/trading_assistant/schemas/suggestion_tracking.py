# schemas/suggestion_tracking.py
"""Suggestion tracking schemas — record and measure suggestion outcomes.

Closes the loop: suggestion proposed → accepted/rejected → implemented → measured.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SuggestionStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    MERGED = "merged"
    DEPLOYED = "deployed"
    MEASURED = "measured"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"  # deprecated: kept for historical JSONL compatibility; use DEPLOYED


class SuggestionRecord(BaseModel):
    """A single suggestion with lifecycle tracking."""

    suggestion_id: str
    bot_id: str
    strategy_id: Optional[str] = None  # specific strategy; None = bot-wide
    title: str
    tier: str  # parameter | filter | strategy_variant | hypothesis
    category: str = ""  # original category (exit_timing, filter_threshold, etc.)
    source_report_id: str
    description: str = ""
    status: SuggestionStatus = SuggestionStatus.PROPOSED
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    accepted_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    measured_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    rejection_reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hypothesis_id: Optional[str] = None
    approval_request_id: Optional[str] = None
    deployment_id: Optional[str] = None
    pr_url: Optional[str] = None
    outcome_source: str = ""  # early_warning | monthly | follow_up
    outcome_source_history: list[dict] = Field(default_factory=list)
    monthly_outcome_id: Optional[str] = None
    strategy_change_record_id: Optional[str] = None
    target_param: Optional[str] = None
    proposed_value: Optional[float] = None
    expected_impact: str = ""
    detection_context: Optional[dict] = None
    implementation_context: Optional[dict] = None
    proposal_id: Optional[str] = None  # cross-link to ProposalLedger


class SuggestionOutcome(BaseModel):
    """Measured impact of an implemented suggestion."""

    suggestion_id: str
    strategy_id: Optional[str] = None  # carried through from the SuggestionRecord
    implemented_date: str  # YYYY-MM-DD
    pnl_delta_7d: float = 0.0
    pnl_delta_30d: float = 0.0
    win_rate_delta_7d: float = 0.0
    win_rate_delta_30d: float = 0.0
    drawdown_delta_7d: float = 0.0
    drawdown_delta_30d: float = 0.0
    measured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    outcome_source: str = "early_warning"
    monthly_outcome_id: Optional[str] = None
    strategy_change_record_id: Optional[str] = None
    verdict: str = ""

    @property
    def net_positive_7d(self) -> bool:
        return self.pnl_delta_7d > 0
