"""Shared human-approval schemas for repo-backed changes."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from trading_assistant.schemas.repo_changes import ChangeKind, FileChange
from trading_assistant.schemas.repo_task import RepoTaskContext
from trading_assistant.schemas.simulation_metrics import SimulationMetrics


class RepoRiskTier(str, Enum):
    AUTO = "auto"
    REQUIRES_APPROVAL = "requires_approval"
    REQUIRES_DOUBLE_APPROVAL = "requires_double_approval"


class BacktestContext(BaseModel):
    """Context for a backtest comparison."""

    suggestion_id: str
    bot_id: str
    param_name: str
    current_value: Any = None
    proposed_value: Any = None
    trade_count: int = 0
    data_days: int = 0


class BacktestComparison(BaseModel):
    """Side-by-side comparison of baseline vs proposed backtest results."""

    context: BacktestContext
    baseline: SimulationMetrics = Field(default_factory=SimulationMetrics)
    proposed: SimulationMetrics = Field(default_factory=SimulationMetrics)
    sharpe_change_pct: float = 0.0
    max_dd_change_pct: float = 0.0
    profit_factor_change_pct: float = 0.0
    win_rate_change_pct: float = 0.0
    passes_safety: bool = False
    safety_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _compute_changes(self) -> BacktestComparison:
        self.sharpe_change_pct = _pct_change(
            self.baseline.sharpe_ratio,
            self.proposed.sharpe_ratio,
        )
        self.max_dd_change_pct = _pct_change(
            self.baseline.max_drawdown_pct,
            self.proposed.max_drawdown_pct,
        )
        self.profit_factor_change_pct = _pct_change(
            self.baseline.profit_factor,
            self.proposed.profit_factor,
        )
        self.win_rate_change_pct = _pct_change(
            self.baseline.win_rate,
            self.proposed.win_rate,
        )
        return self


def _pct_change(old: float, new: float) -> float:
    """Percentage change from old to new. Returns 0 if old is 0."""
    if old == 0:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ApprovalRequest(BaseModel):
    """A request for human approval of a repo-backed change."""

    request_id: str
    suggestion_id: str
    bot_id: str
    strategy_id: str = ""
    change_kind: ChangeKind = ChangeKind.PARAMETER_CHANGE
    title: str = ""
    summary: str = ""
    param_changes: list[dict] = Field(default_factory=list)
    file_changes: list[FileChange] = Field(default_factory=list)
    backtest_summary: Optional[BacktestComparison] = None
    planned_files: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    risk_tier: RepoRiskTier = RepoRiskTier.REQUIRES_APPROVAL
    draft_pr: bool = False
    repo_task: Optional[RepoTaskContext] = None
    implementation_notes: str = ""
    hypothesis_id: Optional[str] = None
    issue_url: Optional[str] = None
    diff_summary: list[str] = Field(default_factory=list)
    monthly_run_id: str = ""
    monthly_run_month: str = ""
    strategy_change_record_id: str = ""
    proposal_id: str = ""
    evidence_paths: list[str] = Field(default_factory=list)
    objective_deltas: dict[str, float] = Field(default_factory=dict)
    risk_classification: str = ""
    rollback_plan: str = ""
    approval_packet_path: str = ""
    machine_readable_payload: dict[str, Any] = Field(default_factory=dict)
    approval_count: int = 0
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    pr_url: Optional[str] = None
    message_id: Optional[int] = None


__all__ = [
    "ApprovalRequest",
    "ApprovalStatus",
    "BacktestComparison",
    "BacktestContext",
    "ChangeKind",
    "RepoRiskTier",
]
