# schemas/proposal_ledger.py
"""ProposalLedger schemas — unified record of every proposal the system produces.

Every parameter/structural/discovery/portfolio/transfer/instrumentation
proposal becomes a ProposalCandidate with cross-links to the existing trackers
(SuggestionTracker, StructuralExperimentTracker, ExperimentManager, DeploymentMonitor).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from trading_assistant.schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


class ProposalSource(str, Enum):
    LLM_DAILY = "llm_daily"
    LLM_WEEKLY = "llm_weekly"
    DETERMINISTIC = "deterministic"
    MONTHLY_SMOKE_REPAIR = "monthly_smoke_repair"
    MONTHLY_PHASED_AUTO = "monthly_phased_auto"
    MONTHLY_MODEL_REVIEW = "monthly_model_review"
    DISCOVERY = "discovery"
    PARAMETER_SEARCH = "parameter_search"
    STRUCTURAL_EXPERIMENT = "structural"
    PORTFOLIO = "portfolio"
    TRANSFER = "transfer"
    INSTRUMENTATION = "instrumentation"


class ProposalKind(str, Enum):
    PARAMETER_CHANGE = "parameter_change"
    STRUCTURAL_CHANGE = "structural_change"
    ROLLBACK = "rollback"
    NEW_STRATEGY = "new_strategy"
    PORTFOLIO_CHANGE = "portfolio_change"
    SEARCH_SPACE_CHANGE = "search_space_change"
    INSTRUMENTATION_REQUEST = "instrumentation_request"
    BUG_FIX = "bug_fix"


class ProposalCandidate(BaseModel):
    """A single proposal entered into the ledger."""

    proposal_id: str  # deterministic 16-char sha256 prefix
    source: ProposalSource
    kind: ProposalKind
    bot_id: str
    strategy_id: str = ""
    lifecycle_stage: str = ""  # signal/entry/management/exit/portfolio
    hypothesis_id: str = ""
    title: str
    description: str = ""
    expected_mechanism: str = ""
    affected_parameters: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    evaluation_method: str = ""  # parameter_search/experiment/replay/approval
    linked_diagnostics: list[str] = Field(default_factory=list)
    linked_run_id: str = ""
    suggestion_id: str = ""  # cross-link to SuggestionTracker
    experiment_id: str = ""  # cross-link to A/B or structural experiment
    deployment_id: str = ""  # cross-link to DeploymentRecord
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ProposalEvaluation(BaseModel):
    """An evaluation step against a ProposalCandidate."""

    proposal_id: str
    method: str  # parameter_search/experiment/replay/...
    summary: str = ""
    objective_score: float = 0.0
    confidence: float = 0.0
    decision: str  # approve/reject/experiment/defer/instrument
    decision_reason: str = ""
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    evidence_paths: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ProposalOutcome(BaseModel):
    """Measured live outcome for a deployed proposal."""

    proposal_id: str
    deployment_id: str = ""
    objective_delta: float = 0.0
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    verdict: str  # improved/regressed/inconclusive/insufficient_data/positive/negative/neutral
    measurement_path: str = ""  # link to OutcomeMeasurement record
    outcome_source: str = "early_warning"
    monthly_outcome_id: str = ""
    strategy_change_record_id: str = ""
    outcome_source_history: list[dict] = Field(default_factory=list)
    measured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ProposalRecord(BaseModel):
    """Aggregated view of one proposal: candidate + all evaluations + all outcomes."""

    candidate: ProposalCandidate
    evaluations: list[ProposalEvaluation] = Field(default_factory=list)
    outcomes: list[ProposalOutcome] = Field(default_factory=list)

    @property
    def has_terminal_outcome(self) -> bool:
        return any(
            o.verdict
            in {"improved", "regressed", "positive", "negative"}
            for o in self.outcomes
        )

    @property
    def latest_decision(self) -> Optional[str]:
        if not self.evaluations:
            return None
        return self.evaluations[-1].decision
