"""Schemas for monthly-loop model review output."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from trading_assistant.schemas.agent_response import StructuralProposal
from trading_assistant.schemas.monthly_candidates import MonthlyRiskClassification


class MonthlyModelRouting(str, Enum):
    SMOKE_REPAIR = "smoke_repair"
    PHASED_AUTO = "phased_auto"
    EXPERIMENT = "experiment"
    MANUAL_DESIGN_REVIEW = "manual_design_review"
    HYPOTHESIS_ONLY = "hypothesis_only"


class MonthlyModelCandidateReview(BaseModel):
    candidate_id: str
    recommendation: str = ""
    evidence_paths: list[str] = Field(default_factory=list)
    capability_labels: list[str] = Field(default_factory=list)
    evidence_authority: str = ""
    hypothesized_mechanism: str = ""
    expected_objective_impact: dict[str, float] = Field(default_factory=dict)
    risk_classification: MonthlyRiskClassification = MonthlyRiskClassification.MEDIUM
    replay_or_experiment_plan: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    rollback_plan: str = ""
    routing: MonthlyModelRouting = MonthlyModelRouting.EXPERIMENT
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MonthlyModelReview(BaseModel):
    run_id: str = ""
    bot_id: str = ""
    strategy_id: str = ""
    candidate_reviews: list[MonthlyModelCandidateReview] = Field(default_factory=list)
    structural_proposals: list[StructuralProposal] = Field(default_factory=list)
    rejected_actions: list[str] = Field(default_factory=list)
    raw_report: str = ""
    raw_structured: dict | None = None
    parse_success: bool = True
    fallback_used: bool = False
    dropped_counts: dict[str, int] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MonthlyModelValidationIssue(BaseModel):
    item_type: str
    item_id: str = ""
    severity: str = "error"
    message: str


class MonthlyModelValidationResult(BaseModel):
    valid: bool = False
    issues: list[MonthlyModelValidationIssue] = Field(default_factory=list)
    approval_tiers: dict[str, str] = Field(default_factory=dict)
    actionable_candidate_ids: list[str] = Field(default_factory=list)
    hypothesis_only_ids: list[str] = Field(default_factory=list)
