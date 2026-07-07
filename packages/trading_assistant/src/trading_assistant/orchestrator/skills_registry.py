"""Skills registry — per-agent-type capability constraints.

Defines and enforces what each agent type is allowed to do: which actions,
which file paths, and whether shell execution is permitted.
Default profiles are provided for all 7 agent types and can be overridden.
"""
from __future__ import annotations

from fnmatch import fnmatch

from trading_assistant.orchestrator.permission_gates import logical_control_path
from trading_assistant.schemas.agent_capabilities import AgentCapability, AgentType, CapabilityCheckResult

# Default capability profiles for all agent types
_DEFAULT_CAPABILITIES: dict[AgentType, AgentCapability] = {
    AgentType.DAILY_ANALYSIS: AgentCapability(
        agent_type=AgentType.DAILY_ANALYSIS,
        allowed_actions=["generate_report", "read_data", "read_memory", "write_report", "emit_event"],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys", "delete_data", "execute_trade"],
        allowed_read_paths=["data/*", "memory/*", "schemas/*", "curated/*"],
        allowed_write_paths=["curated/*", "reports/*"],
        forbidden_paths=["orchestrator/*", ".env", "*.key", "*.pem"],
        can_execute_shell=False,
        can_emit_events=["daily_report_ready"],
        max_concurrent_tasks=1,
        description="Generates daily analysis reports from trade data.",
    ),
    AgentType.WEEKLY_ANALYSIS: AgentCapability(
        agent_type=AgentType.WEEKLY_ANALYSIS,
        allowed_actions=["generate_report", "read_data", "read_memory", "write_report", "emit_event", "read_reports"],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys", "delete_data", "execute_trade"],
        allowed_read_paths=["data/*", "memory/*", "schemas/*", "curated/*", "reports/*"],
        allowed_write_paths=["curated/*", "reports/*"],
        forbidden_paths=["orchestrator/*", ".env", "*.key", "*.pem"],
        can_execute_shell=False,
        can_emit_events=["weekly_report_ready"],
        max_concurrent_tasks=1,
        description="Generates weekly summary reports and strategy refinements.",
    ),
    AgentType.MONTHLY_VALIDATION: AgentCapability(
        agent_type=AgentType.MONTHLY_VALIDATION,
        allowed_actions=["generate_report", "read_data", "read_memory", "write_report", "emit_event"],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys", "delete_data", "execute_trade"],
        allowed_read_paths=["data/*", "memory/*", "schemas/*", "runs/*"],
        allowed_write_paths=["reports/*", "runs/*"],
        forbidden_paths=["orchestrator/*", ".env", "*.key", "*.pem"],
        can_execute_shell=False,
        can_emit_events=["monthly_validation_complete"],
        max_concurrent_tasks=1,
        description="Reviews monthly evidence packages and validation reports.",
    ),
    AgentType.BUG_TRIAGE: AgentCapability(
        agent_type=AgentType.BUG_TRIAGE,
        allowed_actions=[
            "read_source", "read_logs", "open_github_issue", "create_draft_pr",
            "write_report", "emit_event",
        ],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys", "delete_data", "execute_trade"],
        allowed_read_paths=["*.py", "logs/*", "data/*", "tests/*"],
        allowed_write_paths=["triage/*", "reports/*"],
        forbidden_paths=[".env", "*.key", "*.pem"],
        can_execute_shell=False,
        can_emit_events=["triage_complete", "bug_report_ready"],
        max_concurrent_tasks=2,
        description="Triages errors, classifies severity, and creates bug reports.",
    ),
    AgentType.NOTIFICATION: AgentCapability(
        agent_type=AgentType.NOTIFICATION,
        allowed_actions=["send_notification", "read_preferences", "emit_event"],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys", "delete_data", "execute_trade", "write_data"],
        allowed_read_paths=["reports/*", "curated/*", "memory/policies/*"],
        allowed_write_paths=[],
        forbidden_paths=["orchestrator/*", ".env", "*.key", "*.pem", "data/*"],
        can_execute_shell=False,
        can_emit_events=["notification_sent"],
        max_concurrent_tasks=3,
        description="Sends notifications via Telegram, Discord, and email.",
    ),
    AgentType.PR_REVIEW: AgentCapability(
        agent_type=AgentType.PR_REVIEW,
        allowed_actions=[
            "read_source", "read_diff", "write_review", "create_draft_pr",
            "emit_event",
        ],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys", "delete_data", "execute_trade"],
        allowed_read_paths=["*.py", "tests/*", "schemas/*", "docs/*"],
        allowed_write_paths=["reviews/*", "reports/*"],
        forbidden_paths=[".env", "*.key", "*.pem"],
        can_execute_shell=False,
        can_emit_events=["review_complete"],
        max_concurrent_tasks=1,
        description="Reviews pull requests for correctness and trading safety.",
    ),
    AgentType.ORCHESTRATOR: AgentCapability(
        agent_type=AgentType.ORCHESTRATOR,
        allowed_actions=["*"],
        forbidden_actions=["merge_pr", "deploy", "change_api_keys"],
        allowed_read_paths=["*"],
        allowed_write_paths=["*"],
        forbidden_paths=[],
        can_execute_shell=False,
        can_emit_events=["*"],
        max_concurrent_tasks=10,
        description="Central orchestrator — routes events and manages workers.",
    ),
}


class SkillsRegistry:
    """Per-agent-type capability registry with enforcement."""

    def __init__(self, overrides: dict[AgentType, AgentCapability] | None = None) -> None:
        self._capabilities: dict[AgentType, AgentCapability] = dict(_DEFAULT_CAPABILITIES)
        if overrides:
            self._capabilities.update(overrides)

    def check_action(self, agent_type: AgentType | str, action: str) -> CapabilityCheckResult:
        """Check if an agent type is allowed to perform an action."""
        agent_type_str = agent_type.value if isinstance(agent_type, AgentType) else agent_type
        try:
            at = AgentType(agent_type_str) if isinstance(agent_type, str) else agent_type
        except ValueError:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=agent_type_str,
                action=action,
                reason=f"Unknown agent type: {agent_type_str}",
            )

        cap = self._capabilities.get(at)
        if cap is None:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=at.value,
                action=action,
                reason=f"No capability profile for {at.value}",
            )

        # Forbidden check first (overrides allowed)
        if action in cap.forbidden_actions:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=at.value,
                action=action,
                reason=f"Action '{action}' is forbidden for {at.value}",
            )

        # Allowed check (with wildcard support)
        if "*" in cap.allowed_actions or action in cap.allowed_actions:
            return CapabilityCheckResult(
                allowed=True,
                agent_type=at.value,
                action=action,
                reason=f"Action '{action}' is allowed for {at.value}",
            )

        return CapabilityCheckResult(
            allowed=False,
            agent_type=at.value,
            action=action,
            reason=f"Action '{action}' is not in allowed list for {at.value}",
        )

    def check_file_path(
        self, agent_type: AgentType | str, path: str, write: bool = False
    ) -> CapabilityCheckResult:
        """Check if an agent type can access a file path."""
        agent_type_str = agent_type.value if isinstance(agent_type, AgentType) else agent_type
        try:
            at = AgentType(agent_type_str) if isinstance(agent_type, str) else agent_type
        except ValueError:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=agent_type_str,
                action=f"{'write' if write else 'read'}:{path}",
                reason=f"Unknown agent type: {agent_type_str}",
            )

        cap = self._capabilities.get(at)
        if cap is None:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=at.value,
                action=f"{'write' if write else 'read'}:{path}",
                reason=f"No capability profile for {at.value}",
            )

        mode = "write" if write else "read"
        logical_path = logical_control_path(path)

        # Forbidden patterns override everything
        for pattern in cap.forbidden_paths:
            if fnmatch(logical_path, pattern):
                return CapabilityCheckResult(
                    allowed=False,
                    agent_type=at.value,
                    action=f"{mode}:{path}",
                    reason=f"Path '{logical_path}' matches forbidden pattern '{pattern}'",
                )

        # Check allowed patterns
        allowed_patterns = cap.allowed_write_paths if write else cap.allowed_read_paths
        for pattern in allowed_patterns:
            if pattern == "*" or fnmatch(logical_path, pattern):
                return CapabilityCheckResult(
                    allowed=True,
                    agent_type=at.value,
                    action=f"{mode}:{path}",
                    reason=f"Path '{logical_path}' matches allowed {mode} pattern '{pattern}'",
                )

        return CapabilityCheckResult(
            allowed=False,
            agent_type=at.value,
            action=f"{mode}:{path}",
            reason=f"Path '{path}' not in allowed {mode} paths for {at.value}",
        )

    def check_shell(self, agent_type: AgentType | str) -> CapabilityCheckResult:
        """Check if an agent type can execute shell commands."""
        agent_type_str = agent_type.value if isinstance(agent_type, AgentType) else agent_type
        try:
            at = AgentType(agent_type_str) if isinstance(agent_type, str) else agent_type
        except ValueError:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=agent_type_str,
                action="execute_shell",
                reason=f"Unknown agent type: {agent_type_str}",
            )

        cap = self._capabilities.get(at)
        if cap is None:
            return CapabilityCheckResult(
                allowed=False,
                agent_type=at.value,
                action="execute_shell",
                reason=f"No capability profile for {at.value}",
            )

        return CapabilityCheckResult(
            allowed=cap.can_execute_shell,
            agent_type=at.value,
            action="execute_shell",
            reason=f"Shell execution {'allowed' if cap.can_execute_shell else 'denied'} for {at.value}",
        )
