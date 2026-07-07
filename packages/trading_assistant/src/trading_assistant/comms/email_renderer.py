# comms/email_renderer.py
"""Email HTML renderer — produces HTML email bodies with inline CSS."""
from __future__ import annotations

from trading_assistant.schemas.notifications import (
    ControlPanelState,
    NotificationPayload,
    NotificationPriority,
)

_BASE_STYLE = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
    "font-size: 14px; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;"
)
_HEADER_STYLE = "font-size: 20px; font-weight: bold; color: #1a1a1a; margin-bottom: 16px;"
_CRITICAL_STYLE = "background-color: #fee; border-left: 4px solid #f44; padding: 12px;"


def _body_to_html(body: str) -> str:
    paragraphs = body.split("\n\n")
    parts: list[str] = []
    for p in paragraphs:
        lines = p.strip().replace("\n", "<br>")
        if lines:
            parts.append(f"<p>{lines}</p>")
    return "\n".join(parts)


def _wrap_html(title: str, body_html: str) -> str:
    return (
        f'<html><head><meta charset="utf-8"></head>'
        f'<body style="{_BASE_STYLE}">'
        f'<h1 style="{_HEADER_STYLE}">{title}</h1>'
        f'{body_html}'
        f'</body></html>'
    )


class EmailRenderer:
    """Renders notifications as HTML email bodies."""

    def render(self, payload: NotificationPayload) -> str:
        dispatch = {
            "alert": self.render_alert,
            "daily_report": self.render_daily_report,
            "weekly_summary": self.render_weekly_summary,
        }
        handler = dispatch.get(payload.notification_type)
        if handler:
            return handler(payload)
        return f"{payload.title}\n\n{payload.body}"

    def render_control_panel(self, panel: ControlPanelState) -> str:
        lines: list[str] = [f"Control Panel — {panel.date}"]
        for bot in panel.bot_statuses:
            lines.append(f"{bot.status_emoji} {bot.bot_id}: +${bot.pnl:.0f}")
        return "\n".join(lines)

    def render_alert(self, payload: NotificationPayload) -> str:
        priority = payload.priority.value.upper()
        body_html = f'<div style="{_CRITICAL_STYLE}">{_body_to_html(payload.body)}</div>'
        return _wrap_html(f"[{priority}] {payload.title}", body_html)

    def render_daily_report(self, payload: NotificationPayload) -> str:
        body_html = _body_to_html(payload.body)
        return _wrap_html(f"📊 {payload.title}", body_html)

    def render_weekly_summary(self, payload: NotificationPayload) -> str:
        return self.render_weekly_html(payload)

    def render_weekly_html(self, payload: NotificationPayload) -> str:
        body_html = _body_to_html(payload.body)
        return _wrap_html(f"📈 {payload.title}", body_html)

    def build_subject(self, payload: NotificationPayload) -> str:
        if payload.priority == NotificationPriority.CRITICAL:
            return f"🚨 CRITICAL: {payload.title}"
        return payload.title

    def render_approval_request(self, request) -> str:
        lines = [
            f"Bot: {request.bot_id}",
            f"Kind: {request.change_kind.value}",
            f"Risk: {request.risk_tier.value}",
        ]
        if getattr(request, "strategy_id", ""):
            lines.append(f"Strategy: {request.strategy_id}")
        if getattr(request, "monthly_run_id", ""):
            lines.append(f"Monthly run: {request.monthly_run_id}")
        if getattr(request, "summary", ""):
            lines.append(request.summary)
        if getattr(request, "objective_deltas", None):
            deltas = ", ".join(
                f"{key}={value:+.3f}"
                for key, value in list(request.objective_deltas.items())[:5]
            )
            lines.append(f"Objective deltas: {deltas}")
        if getattr(request, "rollback_plan", ""):
            lines.append(f"Rollback: {request.rollback_plan}")
        if getattr(request, "evidence_paths", None):
            lines.append(f"Evidence paths: {len(request.evidence_paths)}")
        body_html = _body_to_html("\n\n".join(lines))
        return _wrap_html(getattr(request, "title", "") or "Approval Request", body_html)
