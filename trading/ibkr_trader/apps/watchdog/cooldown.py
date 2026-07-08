"""In-memory cooldown tracker to prevent alert spam."""
from __future__ import annotations

from datetime import datetime, timezone


class CooldownTracker:
    """Track last-fired time per alert key with configurable cooldown."""

    def __init__(self, cooldown_sec: int = 900):
        self._cooldown_sec = cooldown_sec
        self._alerts: dict[str, datetime] = {}

    def should_alert(self, key: str) -> bool:
        """Return True if the key is new or cooldown has expired."""
        now = datetime.now(timezone.utc)
        last = self._alerts.get(key)
        if last is None or (now - last).total_seconds() >= self._cooldown_sec:
            self._alerts[key] = now
            return True
        return False

    def clear(self, key: str) -> bool:
        """Remove key (issue resolved). Returns True if key existed (triggers recovery)."""
        return self._alerts.pop(key, None) is not None
