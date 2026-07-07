"""Schemas for deterministic bug triage and structured repair proposals."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from trading_assistant.schemas.repo_changes import FileChange


class BugSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}[self.value]


class BugComplexity(str, Enum):
    OBVIOUS_FIX = "obvious_fix"
    SINGLE_FUNCTION = "single_function"
    MULTI_FILE = "multi_file"
    STATE_DEPENDENT = "state_dependent"
    UNKNOWN = "unknown"


class TriageOutcome(str, Enum):
    KNOWN_FIX = "known_fix"
    NEEDS_INVESTIGATION = "needs_investigation"
    NEEDS_HUMAN = "needs_human"
    QUEUED_FOR_DAILY = "queued_for_daily"
    QUEUED_FOR_WEEKLY = "queued_for_weekly"
    ALERTED = "alerted"


class TriageProposalType(str, Enum):
    BUG_FIX = "bug_fix"
    INVESTIGATION = "investigation"
    HUMAN = "human"


class ErrorCategory(str, Enum):
    CRASH = "crash"
    STUCK_POSITION = "stuck_position"
    CONNECTION_LOST = "connection_lost"
    UNEXPECTED_LOSS = "unexpected_loss"
    REPEATED_ERROR = "repeated_error"
    API_ERROR = "api_error"
    CONFIG_ERROR = "config_error"
    DEPENDENCY = "dependency"
    WARNING = "warning"
    DEPRECATION = "deprecation"
    UNKNOWN = "unknown"


class ErrorEvent(BaseModel):
    """An error event received from a VPS bot sidecar."""

    bot_id: str
    error_type: str
    message: str
    stack_trace: str
    source_file: str = ""
    source_line: int = 0
    severity: Optional[BugSeverity] = None
    category: Optional[ErrorCategory] = None
    context: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriageResult(BaseModel):
    """The outcome of deterministic triage."""

    error_event: ErrorEvent
    severity: BugSeverity
    complexity: BugComplexity
    outcome: TriageOutcome
    suggested_fix: str = ""
    affected_files: list[str] = Field(default_factory=list)
    github_issue_url: str = ""
    pr_url: str = ""
    past_rejections: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriageRepairProposal(BaseModel):
    """Structured output expected from the triage analysis agent."""

    proposal_type: TriageProposalType = TriageProposalType.BUG_FIX
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_files: list[str] = Field(default_factory=list)
    issue_title: str = ""
    issue_body: str = ""
    fix_plan: str = ""
    risk_notes: str = ""
    file_changes: list[FileChange] = Field(default_factory=list)
