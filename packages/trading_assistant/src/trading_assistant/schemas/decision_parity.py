"""Decision-level parity contract for structural monthly candidates."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


DECISION_PARITY_DIMENSIONS = {
    "signals",
    "filters",
    "entries",
    "exits",
    "stops",
    "sizing",
    "risk_caps",
    "order_intent",
}


class DecisionParityStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class DecisionParityCheck(BaseModel):
    dimension: str
    status: DecisionParityStatus = DecisionParityStatus.INSUFFICIENT_DATA
    match_rate: float = 0.0
    mismatch_count: int = 0
    notes: str = ""
    evidence_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "DecisionParityCheck":
        self.dimension = self.dimension.strip().lower()
        self.match_rate = max(0.0, min(float(self.match_rate), 1.0))
        if self.mismatch_count < 0:
            raise ValueError("mismatch_count cannot be negative")
        return self


class DecisionParityReport(BaseModel):
    """Live decision API vs backtest adapter parity for structural changes."""

    run_id: str
    candidate_id: str
    strategy_plugin_id: str = ""
    live_repo_commit_sha: str = ""
    backtest_adapter_commit_sha: str = ""
    checks: list[DecisionParityCheck]
    status: DecisionParityStatus = DecisionParityStatus.INSUFFICIENT_DATA
    min_required_match_rate: float = 1.0
    evidence_paths: list[str] = Field(default_factory=list)
    report_version: str = "decision_parity_report_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_report(self) -> "DecisionParityReport":
        dimensions = {check.dimension for check in self.checks}
        missing = DECISION_PARITY_DIMENSIONS - dimensions
        if missing:
            raise ValueError(
                "decision parity report missing dimensions: "
                + ", ".join(sorted(missing))
            )
        self.min_required_match_rate = max(0.0, min(float(self.min_required_match_rate), 1.0))
        if self.status == DecisionParityStatus.PASS:
            missing_required = [
                name for name in (
                    "strategy_plugin_id",
                    "live_repo_commit_sha",
                    "backtest_adapter_commit_sha",
                    "evidence_paths",
                )
                if not getattr(self, name)
            ]
            if missing_required:
                raise ValueError(
                    "decision parity pass missing required fields: "
                    + ", ".join(sorted(missing_required))
                )
            failed = [
                check.dimension for check in self.checks
                if check.status != DecisionParityStatus.PASS
                or check.match_rate < self.min_required_match_rate
                or check.mismatch_count != 0
            ]
            if failed:
                raise ValueError(
                    "decision parity marked pass but dimensions failed: "
                    + ", ".join(sorted(failed))
                )
            missing_evidence = [
                check.dimension for check in self.checks
                if not check.evidence_paths
            ]
            if missing_evidence:
                raise ValueError(
                    "decision parity pass missing dimension evidence: "
                    + ", ".join(sorted(missing_evidence))
                )
        return self

    @property
    def eligible_for_structural_approval(self) -> bool:
        return self.status == DecisionParityStatus.PASS
