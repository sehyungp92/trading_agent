import pytest

from trading_assistant.orchestrator.permission_gates import PermissionGateChecker, PermissionTier


@pytest.fixture
def gates_config() -> dict:
    return {
        "permission_tiers": {
            "auto": {
                "actions": [
                    "open_github_issue",
                    "create_draft_pr",
                    "add_logging",
                    "add_tests",
                    "generate_report",
                ],
                "file_paths": ["docs/*", "tests/*", "*.md"],
            },
            "requires_approval": {
                "actions": [
                    "merge_pr",
                    "change_trading_logic",
                    "change_risk_parameters",
                    "modify_filters",
                ],
                "file_paths": [
                    "strategies/*",
                    "risk/*",
                    "execution/*",
                    "config/trading_*.yaml",
                ],
            },
            "requires_double_approval": {
                "actions": [
                    "change_api_keys",
                    "modify_deployment",
                    "change_kill_switch",
                ],
                "file_paths": [
                    "deploy/*",
                    "infra/*",
                    ".env*",
                    "keys/*",
                    "memory/policies/*",
                ],
            },
        }
    }


@pytest.fixture
def checker(gates_config) -> PermissionGateChecker:
    return PermissionGateChecker(gates_config)


class TestPermissionGateChecker:
    def test_auto_action(self, checker: PermissionGateChecker):
        result = checker.check_action("generate_report")
        assert result.tier == PermissionTier.AUTO
        assert result.allowed is True

    def test_requires_approval_action(self, checker: PermissionGateChecker):
        result = checker.check_action("merge_pr")
        assert result.tier == PermissionTier.REQUIRES_APPROVAL
        assert result.allowed is False

    def test_requires_double_approval_action(self, checker: PermissionGateChecker):
        result = checker.check_action("change_api_keys")
        assert result.tier == PermissionTier.REQUIRES_DOUBLE_APPROVAL
        assert result.allowed is False

    def test_unknown_action_defaults_to_requires_approval(self, checker: PermissionGateChecker):
        result = checker.check_action("unknown_action")
        assert result.tier == PermissionTier.REQUIRES_APPROVAL

    def test_auto_file_path(self, checker: PermissionGateChecker):
        result = checker.check_file_paths(["docs/readme.md", "tests/test_foo.py"])
        assert result.tier == PermissionTier.AUTO
        assert result.allowed is True

    def test_requires_approval_file_path(self, checker: PermissionGateChecker):
        result = checker.check_file_paths(["strategies/ema_cross.py"])
        assert result.tier == PermissionTier.REQUIRES_APPROVAL
        assert result.allowed is False

    def test_mixed_paths_uses_highest_tier(self, checker: PermissionGateChecker):
        result = checker.check_file_paths(["docs/readme.md", "deploy/docker-compose.yml"])
        assert result.tier == PermissionTier.REQUIRES_DOUBLE_APPROVAL

    def test_check_pr_diff(self, checker: PermissionGateChecker):
        """Simulate checking a PR's changed files."""
        changed_files = [
            "strategies/ema_cross.py",
            "tests/test_ema_cross.py",
        ]
        result = checker.check_file_paths(changed_files)
        assert result.tier == PermissionTier.REQUIRES_APPROVAL
        assert "strategies/ema_cross.py" in result.flagged_files

    def test_final_layout_physical_paths_use_logical_policy(self):
        checker = PermissionGateChecker(
            {
                "permission_tiers": {
                    "auto": {"file_paths": []},
                    "requires_approval": {"file_paths": ["skills/*"]},
                    "requires_double_approval": {"file_paths": ["orchestrator/*"]},
                }
            }
        )

        result = checker.check_file_paths(
            [
                "src/trading_assistant/skills/monthly_validation_orchestrator.py",
                "src/trading_assistant/orchestrator/app.py",
            ]
        )

        assert result.tier == PermissionTier.REQUIRES_DOUBLE_APPROVAL
