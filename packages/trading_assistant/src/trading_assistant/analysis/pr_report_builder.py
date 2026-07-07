# analysis/pr_report_builder.py
"""PR report builder — generates markdown summary of PR review results.

Produces a human-readable summary suitable for Telegram notifications.
"""
from __future__ import annotations

from trading_assistant.schemas.pr_review import PRReviewResult, PRReviewStatus


class PRReportBuilder:
    """Builds markdown reports from PR review results."""

    @staticmethod
    def build_markdown(result: PRReviewResult) -> str:
        lines: list[str] = []

        overall = "PASSED" if result.overall_passed else "FAILED"
        lines.append(f"# PR Review — {overall}")
        lines.append(f"**PR:** {result.pr_url}")
        lines.append("")

        # Step results
        lines.append("## Review Steps")
        for step in result.steps:
            icon = _status_icon(step.status)
            line = f"- {icon} **{step.step.value}**: {step.status.value}"
            if step.detail:
                line += f" — {step.detail}"
            if step.blocker_reason:
                line += f"\n  - Blocker: {step.blocker_reason}"
            lines.append(line)
        lines.append("")

        # Safety checks
        if result.safety_checks:
            lines.append("## Trading Safety Checks")
            for check in result.safety_checks:
                icon = "pass" if check.passed else "FAIL"
                line = f"- [{icon}] **{check.check.value}**"
                if check.detail:
                    line += f": {check.detail}"
                if check.is_intentional_change:
                    line += " (intentional)"
                lines.append(line)
            lines.append("")

        return "\n".join(lines)


def _status_icon(status: PRReviewStatus) -> str:
    return {
        PRReviewStatus.PASSED: "ok",
        PRReviewStatus.FAILED: "FAIL",
        PRReviewStatus.BLOCKED: "BLOCKED",
        PRReviewStatus.PENDING: "...",
    }.get(status, "?")
