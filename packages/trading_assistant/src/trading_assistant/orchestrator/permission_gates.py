"""Permission gate enforcement for actions and file paths against tiered gates.

Three tiers:
  - auto: system can do without asking
  - requires_approval: human approves via Telegram
  - requires_double_approval: must confirm twice with reason
"""

from __future__ import annotations

from fnmatch import fnmatch

from trading_assistant.schemas.permissions import PermissionCheckResult, PermissionTier


def logical_control_path(path: str) -> str:
    """Map final-layout physical control paths back to policy-logical paths."""

    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    prefix = "src/trading_assistant/"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    package_prefix = "packages/trading_assistant/src/trading_assistant/"
    if normalized.startswith(package_prefix):
        return normalized[len(package_prefix):]
    return normalized


class PermissionGateChecker:
    def __init__(self, config: dict) -> None:
        tiers = config["permission_tiers"]
        self._action_map: dict[str, PermissionTier] = {}
        self._path_rules: list[tuple[str, PermissionTier]] = []

        for tier_name, tier_enum in [
            ("auto", PermissionTier.AUTO),
            ("requires_approval", PermissionTier.REQUIRES_APPROVAL),
            ("requires_double_approval", PermissionTier.REQUIRES_DOUBLE_APPROVAL),
        ]:
            tier_config = tiers.get(tier_name, {})
            for action in tier_config.get("actions", []):
                self._action_map[action] = tier_enum
            for pattern in tier_config.get("file_paths", []):
                self._path_rules.append((pattern, tier_enum))

    def check_action(self, action: str) -> PermissionCheckResult:
        """Check which tier an action falls into."""
        tier = self._action_map.get(action, PermissionTier.REQUIRES_APPROVAL)
        return PermissionCheckResult(
            tier=tier,
            allowed=tier == PermissionTier.AUTO,
            reason=f"Action '{action}' is in tier '{tier.name}'",
        )

    def check_file_paths(self, paths: list[str]) -> PermissionCheckResult:
        """Check file paths against gate rules. Returns the highest (most restrictive) tier."""
        max_tier = PermissionTier.AUTO
        flagged: list[str] = []

        for path in paths:
            path_tier = self._classify_path(path)
            if path_tier > PermissionTier.AUTO:
                flagged.append(path)
            if path_tier > max_tier:
                max_tier = path_tier

        return PermissionCheckResult(
            tier=max_tier,
            allowed=max_tier == PermissionTier.AUTO,
            flagged_files=flagged,
            reason=f"Highest tier: {max_tier.name}",
        )

    def _classify_path(self, path: str) -> PermissionTier:
        """Find the most restrictive tier matching a file path."""
        max_tier = PermissionTier.AUTO
        for pattern, tier in self._path_rules:
            if fnmatch(logical_control_path(path), pattern):
                if tier > max_tier:
                    max_tier = tier
        return max_tier
