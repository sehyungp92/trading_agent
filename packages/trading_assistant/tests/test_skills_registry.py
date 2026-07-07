"""Tests for SkillsRegistry — per-agent capability enforcement."""
from trading_assistant.schemas.agent_capabilities import AgentCapability, AgentType
from trading_assistant.orchestrator.skills_registry import SkillsRegistry


class TestDefaultCapabilities:
    def test_all_agents_have_defaults(self):
        registry = SkillsRegistry()
        for at in AgentType:
            result = registry.check_action(at, "some_action_that_doesnt_exist")
            # Should get a valid result (not crash), even if denied
            assert result.agent_type == at.value

    def test_daily_analysis_cannot_merge_pr(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.DAILY_ANALYSIS, "merge_pr")
        assert result.allowed is False

    def test_daily_analysis_can_generate_report(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.DAILY_ANALYSIS, "generate_report")
        assert result.allowed is True

    def test_no_agent_has_shell_by_default(self):
        registry = SkillsRegistry()
        for at in AgentType:
            result = registry.check_shell(at)
            assert result.allowed is False


class TestActionChecks:
    def test_allowed_action_returns_true(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.BUG_TRIAGE, "read_source")
        assert result.allowed is True

    def test_forbidden_action_returns_false(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.BUG_TRIAGE, "deploy")
        assert result.allowed is False

    def test_unknown_action_returns_false(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.DAILY_ANALYSIS, "launch_missile")
        assert result.allowed is False

    def test_unknown_agent_type_returns_false(self):
        registry = SkillsRegistry()
        result = registry.check_action("nonexistent_agent", "read_data")
        assert result.allowed is False

    def test_orchestrator_wildcard_allows_most_actions(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.ORCHESTRATOR, "emit_event")
        assert result.allowed is True

    def test_orchestrator_wildcard_still_forbids_deploy(self):
        registry = SkillsRegistry()
        result = registry.check_action(AgentType.ORCHESTRATOR, "deploy")
        assert result.allowed is False


class TestFilePathChecks:
    def test_read_allowed_path(self):
        registry = SkillsRegistry()
        result = registry.check_file_path(AgentType.DAILY_ANALYSIS, "data/trades.json")
        assert result.allowed is True

    def test_read_forbidden_path(self):
        registry = SkillsRegistry()
        result = registry.check_file_path(AgentType.DAILY_ANALYSIS, ".env")
        assert result.allowed is False

    def test_write_allowed_path(self):
        registry = SkillsRegistry()
        result = registry.check_file_path(AgentType.DAILY_ANALYSIS, "reports/daily.json", write=True)
        assert result.allowed is True

    def test_write_to_read_only_path_denied(self):
        registry = SkillsRegistry()
        result = registry.check_file_path(AgentType.NOTIFICATION, "data/trades.json", write=True)
        assert result.allowed is False

    def test_forbidden_overrides_allowed(self):
        registry = SkillsRegistry()
        # DAILY_ANALYSIS has orchestrator/* in forbidden paths
        result = registry.check_file_path(AgentType.DAILY_ANALYSIS, "orchestrator/brain.py")
        assert result.allowed is False

    def test_final_layout_physical_path_uses_logical_policy(self):
        registry = SkillsRegistry()
        result = registry.check_file_path(
            AgentType.DAILY_ANALYSIS,
            "src/trading_assistant/orchestrator/brain.py",
        )
        assert result.allowed is False


class TestShellChecks:
    def test_shell_denied_by_default(self):
        registry = SkillsRegistry()
        result = registry.check_shell(AgentType.MONTHLY_VALIDATION)
        assert result.allowed is False

    def test_custom_override_can_enable_shell(self):
        override = AgentCapability(
            agent_type=AgentType.BUG_TRIAGE,
            allowed_actions=["*"],
            can_execute_shell=True,
        )
        registry = SkillsRegistry(overrides={AgentType.BUG_TRIAGE: override})
        result = registry.check_shell(AgentType.BUG_TRIAGE)
        assert result.allowed is True


class TestOverrides:
    def test_custom_capability_replaces_default(self):
        custom = AgentCapability(
            agent_type=AgentType.DAILY_ANALYSIS,
            allowed_actions=["custom_action"],
            forbidden_actions=[],
        )
        registry = SkillsRegistry(overrides={AgentType.DAILY_ANALYSIS: custom})
        result = registry.check_action(AgentType.DAILY_ANALYSIS, "custom_action")
        assert result.allowed is True
        # Original action should no longer be allowed
        result = registry.check_action(AgentType.DAILY_ANALYSIS, "generate_report")
        assert result.allowed is False
