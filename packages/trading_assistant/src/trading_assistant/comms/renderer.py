# comms/renderer.py
"""Message renderer protocol and plain text implementation.

The MessageRenderer protocol defines how notification payloads get
transformed into channel-specific text. Each channel adapter (Telegram,
Discord, Email) provides its own renderer.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_assistant.schemas.notifications import ControlPanelState, NotificationPayload


@runtime_checkable
class MessageRenderer(Protocol):
    """Protocol for rendering notifications to a specific channel format."""
    def render(self, payload: NotificationPayload) -> str: ...
    def render_control_panel(self, panel: ControlPanelState) -> str: ...
    def render_alert(self, payload: NotificationPayload) -> str: ...
    def render_daily_report(self, payload: NotificationPayload) -> str: ...
    def render_weekly_summary(self, payload: NotificationPayload) -> str: ...


class PlainTextRenderer:
    """Plain text renderer — used for logging and as a fallback."""

    def render(self, payload: NotificationPayload) -> str:
        dispatch = {
            "alert": self.render_alert,
            "daily_report": self.render_daily_report,
            "weekly_summary": self.render_weekly_summary,
        }
        handler = dispatch.get(payload.notification_type)
        if handler:
            return handler(payload)
        return self._render_generic(payload)

    def render_control_panel(self, panel: ControlPanelState) -> str:
        lines: list[str] = []
        lines.append(f"Control Panel — {panel.date}")
        lines.append("")
        lines.append(
            f"Portfolio: +${panel.portfolio_pnl:.0f} ({panel.portfolio_pnl_pct:+.1f}%) "
            f"| DD: {panel.drawdown_pct:.1f}% | Exposure: {panel.exposure_pct:.0f}%"
        )
        lines.append("")
        if panel.daily_report_ready:
            lines.append("Daily report ready")
        if panel.alert_count > 0:
            lines.append(f"{panel.alert_count} alert(s): {panel.alert_summary}")
        if panel.monthly_validation_status:
            lines.append(f"Monthly validation: {panel.monthly_validation_status}")
        lines.append(f"{panel.pending_pr_count} PRs pending")
        lines.append(f"Risk: {panel.risk_status} ({panel.risk_detail})")
        lines.append("")
        for bot in panel.bot_statuses:
            lines.append(
                f"{bot.status_emoji} {bot.bot_id}: +${bot.pnl:.0f} "
                f"({bot.wins}W/{bot.losses}L) — {bot.summary}"
            )
        return "\n".join(lines)

    def render_alert(self, payload: NotificationPayload) -> str:
        priority_label = payload.priority.value.upper()
        lines: list[str] = []
        lines.append(f"[{priority_label}] {payload.title}")
        lines.append("")
        lines.append(payload.body)
        return "\n".join(lines)

    def render_daily_report(self, payload: NotificationPayload) -> str:
        lines: list[str] = []
        lines.append(payload.title)
        lines.append("")
        lines.append(payload.body)
        return "\n".join(lines)

    def render_weekly_summary(self, payload: NotificationPayload) -> str:
        lines: list[str] = []
        lines.append(payload.title)
        lines.append("")
        lines.append(payload.body)
        return "\n".join(lines)

    def _render_generic(self, payload: NotificationPayload) -> str:
        lines: list[str] = []
        lines.append(payload.title)
        lines.append("")
        lines.append(payload.body)
        return "\n".join(lines)
