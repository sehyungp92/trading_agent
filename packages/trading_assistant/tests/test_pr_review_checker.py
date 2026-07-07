# tests/test_pr_review_checker.py
"""Tests for PR review checker — trading-specific safety assertions."""
from trading_assistant.schemas.pr_review import (
    PRReviewStep,
    PRReviewStepResult,
    PRReviewStatus,
    TradingSafetyCheck,
    TradingSafetyResult,
    PRReviewResult,
)
from trading_assistant.skills.pr_review_checker import PRReviewChecker


class TestPermissionGateCheck:
    def test_auto_tier_passes(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*", "*.md"]},
                "requires_approval": {"file_paths": ["skills/*"]},
                "requires_double_approval": {"file_paths": ["orchestrator/orchestrator_brain.py"]},
            }
        })
        result = checker.check_permission_gate(["tests/test_foo.py", "README.md"])
        assert result.status == PRReviewStatus.PASSED

    def test_requires_approval_blocks(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {"file_paths": ["skills/*"]},
                "requires_double_approval": {},
            }
        })
        result = checker.check_permission_gate(["skills/severity_classifier.py"])
        assert result.status == PRReviewStatus.BLOCKED
        assert "requires_approval" in result.blocker_reason.lower()

    def test_double_approval_blocks(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {},
                "requires_approval": {},
                "requires_double_approval": {"file_paths": ["orchestrator/*"]},
            }
        })
        result = checker.check_permission_gate(["orchestrator/brain.py"])
        assert result.status == PRReviewStatus.BLOCKED
        assert "double" in result.blocker_reason.lower()


class TestTradingSafetyChecks:
    def test_position_sizing_unchanged_passes(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["skills/severity_classifier.py"],
            diff_content="+ def classify(self):\n+     return BugSeverity.HIGH",
        )
        pos_check = _find_safety(result, TradingSafetyCheck.POSITION_SIZING_UNCHANGED)
        assert pos_check.passed is True

    def test_position_sizing_flags_when_touched(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["skills/position_sizer.py"],
            diff_content="- position_size = balance * 0.01\n+ position_size = balance * 0.02",
        )
        pos_check = _find_safety(result, TradingSafetyCheck.POSITION_SIZING_UNCHANGED)
        assert pos_check.passed is False

    def test_risk_limits_unchanged_passes(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["tests/test_foo.py"],
            diff_content="+ assert result == True",
        )
        risk_check = _find_safety(result, TradingSafetyCheck.RISK_LIMITS_ENFORCED)
        assert risk_check.passed is True

    def test_risk_limits_flags_when_touched(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["config/risk_limits.yaml"],
            diff_content="- max_drawdown: 0.05\n+ max_drawdown: 0.10",
        )
        risk_check = _find_safety(result, TradingSafetyCheck.RISK_LIMITS_ENFORCED)
        assert risk_check.passed is False

    def test_kill_switch_unchanged_passes(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["tests/test_foo.py"],
            diff_content="+ pass",
        )
        ks_check = _find_safety(result, TradingSafetyCheck.KILL_SWITCH_WORKS)
        assert ks_check.passed is True

    def test_kill_switch_flags_when_touched(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["orchestrator/kill_switch.py"],
            diff_content="- if should_kill:\n-     kill()\n+ pass",
        )
        ks_check = _find_safety(result, TradingSafetyCheck.KILL_SWITCH_WORKS)
        assert ks_check.passed is False


class TestRunFullReview:
    def test_full_review_all_pass(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {},
                "requires_double_approval": {},
            }
        })
        result = checker.run_review(
            pr_url="https://github.com/user/repo/pull/1",
            changed_files=["tests/test_foo.py"],
            diff_content="+ assert True",
            ci_passed=True,
        )
        assert isinstance(result, PRReviewResult)
        assert result.overall_passed is True

    def test_full_review_ci_fails(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {},
                "requires_double_approval": {},
            }
        })
        result = checker.run_review(
            pr_url="https://github.com/user/repo/pull/2",
            changed_files=["tests/test_foo.py"],
            diff_content="+ assert True",
            ci_passed=False,
        )
        assert result.overall_passed is False


# --- Helpers ---

_MINIMAL_CONFIG = {
    "permission_tiers": {
        "auto": {"file_paths": ["tests/*", "*.md"]},
        "requires_approval": {},
        "requires_double_approval": {},
    }
}


def _find_safety(
    results: list[TradingSafetyResult], check: TradingSafetyCheck,
) -> TradingSafetyResult:
    for r in results:
        if r.check == check:
            return r
    raise ValueError(f"Safety check {check} not found in results")


class TestPRReviewStep:
    def test_all_steps_exist(self):
        assert PRReviewStep.CODE_REVIEW == "code_review"
        assert PRReviewStep.CI_CHECK == "ci_check"
        assert PRReviewStep.PERMISSION_GATE == "permission_gate"
        assert PRReviewStep.TRADING_SAFETY == "trading_safety"
        assert PRReviewStep.HUMAN_REVIEW == "human_review"


class TestPRReviewStatus:
    def test_all_statuses_exist(self):
        assert PRReviewStatus.PENDING == "pending"
        assert PRReviewStatus.PASSED == "passed"
        assert PRReviewStatus.FAILED == "failed"
        assert PRReviewStatus.BLOCKED == "blocked"


class TestPRReviewStepResult:
    def test_creates_passed(self):
        r = PRReviewStepResult(
            step=PRReviewStep.CI_CHECK,
            status=PRReviewStatus.PASSED,
            detail="All checks green",
        )
        assert r.step == PRReviewStep.CI_CHECK
        assert r.status == PRReviewStatus.PASSED
        assert r.detail == "All checks green"
        assert r.blocker_reason == ""

    def test_creates_failed_with_blocker(self):
        r = PRReviewStepResult(
            step=PRReviewStep.PERMISSION_GATE,
            status=PRReviewStatus.FAILED,
            detail="File touches risk_limits.py",
            blocker_reason="requires_double_approval for risk_limits.py",
        )
        assert r.blocker_reason != ""


class TestTradingSafetyCheck:
    def test_all_checks_exist(self):
        assert TradingSafetyCheck.POSITION_SIZING_UNCHANGED == "position_sizing_unchanged"
        assert TradingSafetyCheck.RISK_LIMITS_ENFORCED == "risk_limits_enforced"
        assert TradingSafetyCheck.KILL_SWITCH_WORKS == "kill_switch_works"


class TestTradingSafetyResult:
    def test_creates_with_defaults(self):
        r = TradingSafetyResult(
            check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
            passed=True,
        )
        assert r.passed is True
        assert r.detail == ""
        assert r.is_intentional_change is False

    def test_intentional_change(self):
        r = TradingSafetyResult(
            check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
            passed=False,
            detail="Position size formula changed",
            is_intentional_change=True,
        )
        assert r.is_intentional_change is True


class TestPRReviewResult:
    def test_creates_with_all_steps(self):
        steps = [
            PRReviewStepResult(
                step=PRReviewStep.CODE_REVIEW,
                status=PRReviewStatus.PASSED,
            ),
            PRReviewStepResult(
                step=PRReviewStep.CI_CHECK,
                status=PRReviewStatus.PASSED,
            ),
        ]
        safety = [
            TradingSafetyResult(
                check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
                passed=True,
            ),
        ]
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/42",
            steps=steps,
            safety_checks=safety,
        )
        assert r.pr_url.endswith("/42")
        assert len(r.steps) == 2
        assert len(r.safety_checks) == 1

    def test_overall_passed(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/1",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(check=TradingSafetyCheck.KILL_SWITCH_WORKS, passed=True),
            ],
        )
        assert r.overall_passed is True

    def test_overall_failed_when_step_fails(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/2",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.FAILED),
            ],
            safety_checks=[],
        )
        assert r.overall_passed is False

    def test_overall_failed_when_safety_fails_and_not_intentional(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/3",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(
                    check=TradingSafetyCheck.RISK_LIMITS_ENFORCED,
                    passed=False,
                    is_intentional_change=False,
                ),
            ],
        )
        assert r.overall_passed is False

    def test_overall_passed_when_safety_fails_but_intentional(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/4",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(
                    check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
                    passed=False,
                    is_intentional_change=True,
                ),
            ],
        )
        assert r.overall_passed is True
