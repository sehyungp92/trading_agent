# tests/test_pr_report_builder.py
"""Tests for PR report builder — markdown summary of PR review results."""
from trading_assistant.schemas.pr_review import (
    PRReviewResult,
    PRReviewStep,
    PRReviewStepResult,
    PRReviewStatus,
    TradingSafetyCheck,
    TradingSafetyResult,
)
from trading_assistant.analysis.pr_report_builder import PRReportBuilder


class TestPRReportBuilder:
    def test_builds_markdown_for_passed_review(self):
        result = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/42",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.PASSED),
                PRReviewStepResult(step=PRReviewStep.PERMISSION_GATE, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(check=TradingSafetyCheck.KILL_SWITCH_WORKS, passed=True),
            ],
        )
        md = PRReportBuilder.build_markdown(result)

        assert "PR Review" in md
        assert "PASSED" in md.upper() or "passed" in md.lower()
        assert "pull/42" in md

    def test_builds_markdown_for_failed_review(self):
        result = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/99",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.FAILED, detail="lint errors"),
            ],
            safety_checks=[],
        )
        md = PRReportBuilder.build_markdown(result)

        assert "FAILED" in md.upper() or "failed" in md.lower()
        assert "lint errors" in md

    def test_builds_markdown_for_safety_failures(self):
        result = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/7",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(
                    check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
                    passed=False,
                    detail="Position size formula changed",
                ),
            ],
        )
        md = PRReportBuilder.build_markdown(result)

        assert "position_sizing" in md.lower() or "Position" in md
        assert "changed" in md.lower()

    def test_builds_markdown_for_blocked_permission(self):
        result = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/5",
            steps=[
                PRReviewStepResult(
                    step=PRReviewStep.PERMISSION_GATE,
                    status=PRReviewStatus.BLOCKED,
                    blocker_reason="requires_approval for skills/foo.py",
                ),
            ],
            safety_checks=[],
        )
        md = PRReportBuilder.build_markdown(result)

        assert "blocked" in md.lower() or "BLOCKED" in md
        assert "requires_approval" in md
