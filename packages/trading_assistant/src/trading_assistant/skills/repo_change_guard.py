"""Path-scoped safety checks for bot repo edits."""
from __future__ import annotations

from fnmatch import fnmatch

from trading_assistant.orchestrator.permission_gates import PermissionGateChecker, logical_control_path
from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.schemas.permissions import PermissionCheckResult, PermissionTier

DEFAULT_PERMISSION_CONFIG = {
    "permission_tiers": {
        "auto": {
            "actions": ["open_github_issue", "create_draft_pr", "add_logging", "add_tests"],
            "file_paths": ["docs/*", "tests/*", "*.md", "scripts/*", "tools/*", "utils/*"],
        },
        "requires_approval": {
            "actions": ["change_trading_logic", "change_risk_parameters", "modify_filters"],
            "file_paths": [
                "strategies/*",
                "signals/*",
                "filters/*",
                "execution/*",
                "sizing/*",
                "risk/*",
                "trading/*",
                "config/*.py",
                "config/trading_*.yaml",
            ],
        },
        "requires_double_approval": {
            "actions": ["change_kill_switch", "modify_deployment", "change_api_keys"],
            "file_paths": [
                "*kill_switch*",
                "deploy/*",
                "infra/*",
                ".env*",
                "keys/*",
                "secrets/*",
            ],
        },
    },
}


class RepoChangeGuard:
    """Combines bot allowlists with the shared permission-tier checker."""

    def __init__(self, permission_config: dict | None = None) -> None:
        self._checker = PermissionGateChecker(permission_config or DEFAULT_PERMISSION_CONFIG)

    def blocked_paths(
        self,
        profile: BotConfigProfile,
        paths: list[str],
    ) -> list[str]:
        return self._blocked_paths(profile.allowed_edit_paths, paths)

    def check_paths(
        self,
        profile: BotConfigProfile,
        paths: list[str],
    ) -> PermissionCheckResult:
        blocked = self.blocked_paths(profile, paths)
        if blocked:
            return PermissionCheckResult(
                tier=PermissionTier.REQUIRES_DOUBLE_APPROVAL,
                allowed=False,
                flagged_files=blocked,
                reason=f"Paths outside allowed_edit_paths: {', '.join(blocked)}",
            )
        return self._checker.check_file_paths(paths)

    @staticmethod
    def _blocked_paths(
        allowed_edit_paths: list[str],
        paths: list[str],
    ) -> list[str]:
        if not allowed_edit_paths:
            return []
        return [
            path for path in paths
            if not any(fnmatch(logical_control_path(path), pattern) for pattern in allowed_edit_paths)
        ]
