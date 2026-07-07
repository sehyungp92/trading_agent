"""Report quality gate schemas — Definition of Done for daily reports.

Every daily report must pass all checks before being sent.
"""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class CheckResult(BaseModel):
    """One quality gate check."""

    name: str
    passed: bool
    detail: str = ""


class ReportChecklist(BaseModel):
    """Aggregated quality gate for a daily report."""

    report_id: str
    checks: list[CheckResult] = []
    can_proceed: bool = True
    data_completeness: float = 1.0
    available_bots: list[str] = []
    missing_bots: list[str] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall(self) -> str:
        if all(c.passed for c in self.checks):
            return "PASS"
        return "FAIL"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocking_issues(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]
