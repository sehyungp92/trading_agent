"""PR review schemas --- multi-stage review pipeline data models.

Every automated PR goes through: code review -> CI -> permission gate ->
trading-specific safety checks -> human review. These schemas track each step.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class PRReviewStep(str, Enum):
    CODE_REVIEW = "code_review"
    CI_CHECK = "ci_check"
    PERMISSION_GATE = "permission_gate"
    TRADING_SAFETY = "trading_safety"
    HUMAN_REVIEW = "human_review"


class PRReviewStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PRReviewStepResult(BaseModel):
    step: PRReviewStep
    status: PRReviewStatus = PRReviewStatus.PENDING
    detail: str = ""
    blocker_reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TradingSafetyCheck(str, Enum):
    POSITION_SIZING_UNCHANGED = "position_sizing_unchanged"
    RISK_LIMITS_ENFORCED = "risk_limits_enforced"
    KILL_SWITCH_WORKS = "kill_switch_works"


class TradingSafetyResult(BaseModel):
    check: TradingSafetyCheck
    passed: bool
    detail: str = ""
    is_intentional_change: bool = False


class PRReviewResult(BaseModel):
    """Aggregated result of a full PR review pipeline run."""

    pr_url: str
    steps: list[PRReviewStepResult] = []
    safety_checks: list[TradingSafetyResult] = []
    rejection_reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_passed(self) -> bool:
        steps_ok = all(s.status == PRReviewStatus.PASSED for s in self.steps)
        safety_ok = all(
            s.passed or s.is_intentional_change for s in self.safety_checks
        )
        return steps_ok and safety_ok
