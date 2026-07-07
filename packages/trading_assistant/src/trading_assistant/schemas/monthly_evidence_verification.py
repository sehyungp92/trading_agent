"""Independent monthly evidence verification schemas."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MonthlyEvidenceVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class MonthlyEvidenceRecommendedAction(str, Enum):
    ROUTE_APPROVAL = "route_approval"
    SUPPRESS_APPROVAL = "suppress_approval"
    HUMAN_REVIEW = "human_review"
    NO_ACTION = "no_action"


class MonthlyEvidenceFinding(BaseModel):
    code: str
    severity: str = "error"
    message: str
    evidence_paths: list[str] = Field(default_factory=list)
    remediation: str = ""


class EvidencePathCheck(BaseModel):
    path: str
    known: bool = False
    exists: bool = False
    eligible_for_approval_gate: bool = False
    message: str = ""


class AuthorityCheck(BaseModel):
    name: str
    passed: bool
    message: str = ""


class ApprovalPacketCheck(BaseModel):
    name: str
    passed: bool
    message: str = ""


class MonthlyEvidenceVerification(BaseModel):
    verification_id: str = ""
    run_id: str = ""
    candidate_id: str = ""
    verdict: MonthlyEvidenceVerdict = MonthlyEvidenceVerdict.NEEDS_HUMAN_REVIEW
    blocking_findings: list[MonthlyEvidenceFinding] = Field(default_factory=list)
    non_blocking_findings: list[MonthlyEvidenceFinding] = Field(default_factory=list)
    evidence_path_checks: list[EvidencePathCheck] = Field(default_factory=list)
    authority_checks: list[AuthorityCheck] = Field(default_factory=list)
    approval_packet_checks: list[ApprovalPacketCheck] = Field(default_factory=list)
    recommended_action: MonthlyEvidenceRecommendedAction = MonthlyEvidenceRecommendedAction.HUMAN_REVIEW
    verifier_mode: str = "deterministic_read_only"
    verifier_artifact_path: str = ""
    source_payload: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.verification_id:
            raw = "|".join([
                self.run_id,
                self.candidate_id,
                self.verdict.value,
                str(len(self.blocking_findings)),
            ])
            self.verification_id = "monthly-verifier-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
