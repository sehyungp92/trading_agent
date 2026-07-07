"""Cooldown tracker for provider failures — prevents rapid retry storms."""
from __future__ import annotations

from datetime import datetime, timezone

from trading_assistant.schemas.agent_preferences import AgentProvider

_DEFAULT_COOLDOWN_SECONDS = 300


class ProviderCooldownTracker:
    """Tracks provider failures and enforces cooldown periods before retry."""

    def __init__(self, cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._failures: dict[AgentProvider, datetime] = {}

    @property
    def cooldown_seconds(self) -> int:
        return self._cooldown_seconds

    def record_failure(self, provider: AgentProvider) -> None:
        """Record a failure timestamp for the given provider."""
        self._failures[provider] = datetime.now(timezone.utc)

    def is_cooled_down(self, provider: AgentProvider) -> bool:
        """Return True if the provider is still in cooldown (should NOT be retried)."""
        last_failure = self._failures.get(provider)
        if last_failure is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last_failure).total_seconds()
        return elapsed < self._cooldown_seconds

    def clear(self, provider: AgentProvider | None = None) -> None:
        """Clear cooldown for a specific provider, or all providers."""
        if provider is None:
            self._failures.clear()
        else:
            self._failures.pop(provider, None)

    def active_cooldowns(self) -> dict[AgentProvider, float]:
        """Return providers currently in cooldown with remaining seconds."""
        now = datetime.now(timezone.utc)
        result: dict[AgentProvider, float] = {}
        for provider, failed_at in self._failures.items():
            remaining = self._cooldown_seconds - (now - failed_at).total_seconds()
            if remaining > 0:
                result[provider] = remaining
        return result
