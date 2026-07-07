"""Tests for provider cooldown tracking."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


from trading_assistant.orchestrator.provider_cooldown import ProviderCooldownTracker
from trading_assistant.schemas.agent_preferences import AgentProvider


class TestRecordAndCheck:
    def test_no_failure_means_not_cooled_down(self):
        tracker = ProviderCooldownTracker()
        assert tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is False

    def test_recent_failure_means_cooled_down(self):
        tracker = ProviderCooldownTracker(cooldown_seconds=300)
        tracker.record_failure(AgentProvider.CLAUDE_MAX)
        assert tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is True

    def test_old_failure_means_not_cooled_down(self):
        tracker = ProviderCooldownTracker(cooldown_seconds=300)
        tracker._failures[AgentProvider.CLAUDE_MAX] = (
            datetime.now(timezone.utc) - timedelta(seconds=400)
        )
        assert tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is False

    def test_different_providers_independent(self):
        tracker = ProviderCooldownTracker(cooldown_seconds=300)
        tracker.record_failure(AgentProvider.CLAUDE_MAX)
        assert tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is True
        assert tracker.is_cooled_down(AgentProvider.CODEX_PRO) is False

    def test_cooldown_seconds_property(self):
        tracker = ProviderCooldownTracker(cooldown_seconds=120)
        assert tracker.cooldown_seconds == 120


class TestClear:
    def test_clear_specific_provider(self):
        tracker = ProviderCooldownTracker()
        tracker.record_failure(AgentProvider.CLAUDE_MAX)
        tracker.record_failure(AgentProvider.CODEX_PRO)
        tracker.clear(AgentProvider.CLAUDE_MAX)
        assert tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is False
        assert tracker.is_cooled_down(AgentProvider.CODEX_PRO) is True

    def test_clear_all(self):
        tracker = ProviderCooldownTracker()
        tracker.record_failure(AgentProvider.CLAUDE_MAX)
        tracker.record_failure(AgentProvider.CODEX_PRO)
        tracker.clear()
        assert tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is False
        assert tracker.is_cooled_down(AgentProvider.CODEX_PRO) is False

    def test_clear_nonexistent_provider_is_noop(self):
        tracker = ProviderCooldownTracker()
        tracker.clear(AgentProvider.OPENROUTER)  # should not raise


class TestActiveCooldowns:
    def test_returns_empty_when_no_failures(self):
        tracker = ProviderCooldownTracker()
        assert tracker.active_cooldowns() == {}

    def test_returns_remaining_seconds(self):
        tracker = ProviderCooldownTracker(cooldown_seconds=300)
        tracker.record_failure(AgentProvider.CLAUDE_MAX)
        cooldowns = tracker.active_cooldowns()
        assert AgentProvider.CLAUDE_MAX in cooldowns
        assert 0 < cooldowns[AgentProvider.CLAUDE_MAX] <= 300

    def test_excludes_expired_cooldowns(self):
        tracker = ProviderCooldownTracker(cooldown_seconds=10)
        tracker._failures[AgentProvider.CLAUDE_MAX] = (
            datetime.now(timezone.utc) - timedelta(seconds=20)
        )
        assert tracker.active_cooldowns() == {}
