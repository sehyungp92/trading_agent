"""Agent capability schemas — defines what each agent type is allowed to do."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class AgentType(str, Enum):
    """Known agent types in the system."""
    DAILY_ANALYSIS = "daily_analysis"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MONTHLY_VALIDATION = "monthly_validation"
    MONTHLY_MODEL_REVIEW = "monthly_model_review"
    BUG_TRIAGE = "bug_triage"
    NOTIFICATION = "notification"
    PR_REVIEW = "pr_review"
    ORCHESTRATOR = "orchestrator"


class CapabilityCheckResult(BaseModel):
    """Result of a capability check."""
    allowed: bool
    agent_type: str
    action: str = ""
    reason: str = ""


class AgentCapability(BaseModel):
    """Defines what an agent type is allowed to do."""
    agent_type: AgentType
    allowed_actions: list[str] = []
    forbidden_actions: list[str] = []
    allowed_read_paths: list[str] = []
    allowed_write_paths: list[str] = []
    forbidden_paths: list[str] = []
    can_execute_shell: bool = False
    can_emit_events: list[str] = []
    max_concurrent_tasks: int = 1
    description: str = ""
