"""Typed views for the monthly artifact contract.

These schemas describe control-plane views over the backtest artifact index.
The artifact names and base validation rules remain owned by
``schemas.backtest_artifacts``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from trading_assistant.schemas.artifact_authority import ArtifactAuthority
from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.loop_contracts import LoopContract
from trading_assistant.schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyImprovementCandidate,
)
from trading_assistant.schemas.monthly_model_review import (
    MonthlyModelReview,
    MonthlyModelValidationResult,
)
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult


class MonthlyArtifactStatus(str, Enum):
    COMPLETE = "complete"
    MISSING_REQUIRED = "missing_required"
    MALFORMED = "malformed"
    OUTSIDE_ROOT = "outside_root"
    SCOPE_MISMATCH = "scope_mismatch"


class MonthlyArtifactIssue(BaseModel):
    code: str
    message: str
    artifact_name: str = ""
    path: str = ""
    status: MonthlyArtifactStatus
    severity: str = "error"


class MonthlyArtifactView(BaseModel):
    name: str
    path: str = ""
    exists: bool = False
    required: bool = False
    optional: bool = False
    artifact_type: str = ""
    authority: ArtifactAuthority | None = None
    may_satisfy_approval_gate: bool = False


class MonthlyApprovalEvidenceView(BaseModel):
    candidate_id: str
    artifact_paths: list[str] = Field(default_factory=list)
    approval_gate_evidence: list[str] = Field(default_factory=list)
    selected_candidate_count: int = 0
    rejected_candidate_count: int = 0
    selected_candidate_path: str = ""
    rejected_candidates_path: str = ""


class MonthlyVerifierInput(BaseModel):
    monthly_result: MonthlyValidationResult
    artifact_index: BacktestArtifactIndex | None = None
    selected_candidates: list[MonthlyImprovementCandidate] = Field(default_factory=list)
    gate_reports: list[MonthlyCandidateGateReport] = Field(default_factory=list)
    approval_packet: MonthlyApprovalEvidencePacket | None = None
    run_manifest: MonthlyRunManifest | None = None
    model_review: MonthlyModelReview | None = None
    model_validation: MonthlyModelValidationResult | None = None
    model_review_validation_path: str = ""
    deployment_metadata_blockers: list[str] = Field(default_factory=list)
    loop_contract: LoopContract | None = None

    def to_verify_kwargs(self) -> dict[str, Any]:
        return {
            "monthly_result": self.monthly_result,
            "artifact_index": self.artifact_index,
            "selected_candidates": self.selected_candidates,
            "gate_reports": self.gate_reports,
            "approval_packet": self.approval_packet,
            "run_manifest": self.run_manifest,
            "model_review": self.model_review,
            "model_validation": self.model_validation,
            "model_review_validation_path": self.model_review_validation_path,
            "deployment_metadata_blockers": self.deployment_metadata_blockers,
            "loop_contract": self.loop_contract,
        }
