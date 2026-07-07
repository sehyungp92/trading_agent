# schemas/portfolio_proposal.py
"""Portfolio proposal schemas — structured proposals for portfolio-level config changes."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PortfolioProposalType(str, Enum):
    ALLOCATION_REBALANCE = "allocation_rebalance"
    RISK_CAP_CHANGE = "risk_cap_change"
    COORDINATION_CHANGE = "coordination_change"
    DRAWDOWN_TIER_CHANGE = "drawdown_tier_change"


class PortfolioProposal(BaseModel):
    """A structured portfolio-level change proposal from LLM analysis."""

    proposal_type: PortfolioProposalType
    current_config: dict = {}
    proposed_config: dict = {}
    evidence_summary: str = ""
    expected_portfolio_calmar_delta: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observation_window_days: int = 30
    acceptance_criteria: list[dict] = Field(default_factory=list)
    what_if_result: Optional[dict] = None  # Populated by portfolio_what_if before recording
