from __future__ import annotations

from trading_assistant.schemas.notifications import NotificationPayload


class RiskInjectionHook:
    """Appends a risk warning to notifications when drawdown exceeds threshold."""

    def __init__(self, drawdown_threshold_pct: float = 5.0) -> None:
        self._threshold = drawdown_threshold_pct
        self._current_drawdown_pct: float = 0.0

    def set_drawdown(self, drawdown_pct: float) -> None:
        self._current_drawdown_pct = drawdown_pct

    def transform(self, payload: NotificationPayload) -> NotificationPayload:
        if self._current_drawdown_pct >= self._threshold:
            warning = f"\n\n\u26a0\ufe0f RISK WARNING: Portfolio drawdown at {self._current_drawdown_pct:.1f}%"
            return payload.model_copy(update={"body": payload.body + warning})
        return payload
