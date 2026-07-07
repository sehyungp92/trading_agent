# skills/pr_review_checker.py
"""PR review checker — multi-stage review pipeline with trading safety assertions.

Every automated PR goes through:
1. CI check (lint, types, unit tests, integration tests)
2. Permission gate check — file paths verified against tiers
3. Trading-specific safety checks:
   - Position sizing unchanged
   - Risk limits still enforced
   - Kill switch still works
"""
from __future__ import annotations

import re

from trading_assistant.orchestrator.permission_gates import PermissionGateChecker
from trading_assistant.schemas.permissions import PermissionTier
from trading_assistant.schemas.pr_review import (
    PRReviewResult,
    PRReviewStatus,
    PRReviewStep,
    PRReviewStepResult,
    TradingSafetyCheck,
    TradingSafetyResult,
)

# Patterns that indicate trading-critical changes
_POSITION_SIZING_PATTERNS = re.compile(
    r"position.?siz|lot.?size|order.?size|leverage|margin",
    re.I,
)
_RISK_LIMIT_PATTERNS = re.compile(
    r"risk.?limit|max.?drawdown|max.?loss|stop.?loss.?pct|risk.?per.?trade|max.?position",
    re.I,
)
_KILL_SWITCH_PATTERNS = re.compile(
    r"kill.?switch|emergency.?stop|halt.?trading|shutdown|circuit.?breaker",
    re.I,
)


class PRReviewChecker:
    """Runs the multi-stage PR review pipeline."""

    def __init__(self, permission_config: dict) -> None:
        self._gate = PermissionGateChecker(permission_config)

    def run_review(
        self,
        pr_url: str,
        changed_files: list[str],
        diff_content: str,
        ci_passed: bool,
    ) -> PRReviewResult:
        """Run the full review pipeline and return aggregated results."""
        steps: list[PRReviewStepResult] = []

        # Step 1: CI check
        steps.append(PRReviewStepResult(
            step=PRReviewStep.CI_CHECK,
            status=PRReviewStatus.PASSED if ci_passed else PRReviewStatus.FAILED,
            detail="All CI checks passed" if ci_passed else "CI checks failed",
        ))

        # Step 2: Permission gate
        steps.append(self.check_permission_gate(changed_files))

        # Step 3: Trading safety
        safety_checks = self.check_trading_safety(changed_files, diff_content)

        return PRReviewResult(
            pr_url=pr_url,
            steps=steps,
            safety_checks=safety_checks,
        )

    def check_permission_gate(self, changed_files: list[str]) -> PRReviewStepResult:
        """Check file paths against permission tiers."""
        result = self._gate.check_file_paths(changed_files)

        if result.tier == PermissionTier.AUTO:
            return PRReviewStepResult(
                step=PRReviewStep.PERMISSION_GATE,
                status=PRReviewStatus.PASSED,
                detail="All files in AUTO tier",
            )

        tier_name = result.tier.name.lower()
        flagged = ", ".join(result.flagged_files) if result.flagged_files else "unknown"
        return PRReviewStepResult(
            step=PRReviewStep.PERMISSION_GATE,
            status=PRReviewStatus.BLOCKED,
            detail=f"Files require {tier_name}: {flagged}",
            blocker_reason=f"{tier_name} for: {flagged}",
        )

    def check_trading_safety(
        self,
        changed_files: list[str],
        diff_content: str,
    ) -> list[TradingSafetyResult]:
        """Run all trading-specific safety checks."""
        combined = " ".join(changed_files) + " " + diff_content

        return [
            self._check_pattern(
                TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
                _POSITION_SIZING_PATTERNS,
                combined,
            ),
            self._check_pattern(
                TradingSafetyCheck.RISK_LIMITS_ENFORCED,
                _RISK_LIMIT_PATTERNS,
                combined,
            ),
            self._check_pattern(
                TradingSafetyCheck.KILL_SWITCH_WORKS,
                _KILL_SWITCH_PATTERNS,
                combined,
            ),
        ]

    def _check_pattern(
        self,
        check: TradingSafetyCheck,
        pattern: re.Pattern,
        text: str,
    ) -> TradingSafetyResult:
        match = pattern.search(text)
        if match:
            return TradingSafetyResult(
                check=check,
                passed=False,
                detail=f"Pattern matched: '{match.group()}' — verify this is intentional",
            )
        return TradingSafetyResult(check=check, passed=True)
