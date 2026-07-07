# comms/discord_renderer.py
"""Discord embed renderer for trading notifications."""
from __future__ import annotations

from trading_assistant.schemas.notifications import (
    ControlPanelState,
    NotificationPayload,
    NotificationPriority,
)

_COLORS = {
    NotificationPriority.CRITICAL: 0xFF0000,
    NotificationPriority.HIGH: 0xE67E22,
    NotificationPriority.NORMAL: 0x3498DB,
    NotificationPriority.LOW: 0x95A5A6,
}


class DiscordRenderer:
    """Renders notifications as Discord-style embeds or plain text."""

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
        lines: list[str] = []
        lines.append(f"📊 {panel.date} — Control Panel")
        lines.append(
            f"Portfolio: +${panel.portfolio_pnl:.0f} ({panel.portfolio_pnl_pct:+.1f}%) "
            f"| DD: {panel.drawdown_pct:.1f}% | Exposure: {panel.exposure_pct:.0f}%"
        )
        for bot in panel.bot_statuses:
            lines.append(
                f"{bot.status_emoji} {bot.bot_id}: +${bot.pnl:.0f} "
                f"({bot.wins}W/{bot.losses}L) — {bot.summary}"
            )
        return "\n".join(lines)

    def render_alert(self, payload: NotificationPayload) -> str:
        embed = self.render_alert_embed(payload)
        return f"{embed['title']}\n\n{embed['description']}"

    def render_alert_embed(self, payload: NotificationPayload) -> dict:
        priority = payload.priority
        if priority == NotificationPriority.CRITICAL:
            title = f"🚨 CRITICAL — {payload.title}"
        elif priority == NotificationPriority.HIGH:
            title = f"⚠️ HIGH — {payload.title}"
        else:
            title = f"ℹ️ {payload.title}"
        return {
            "title": title,
            "description": payload.body,
            "color": _COLORS.get(priority, 0x3498DB),
        }

    def render_daily_report(self, payload: NotificationPayload) -> str:
        embed = self.render_daily_report_embed(payload)
        return f"{embed['title']}\n\n{embed['description']}"

    def render_daily_report_embed(self, payload: NotificationPayload) -> dict:
        embed: dict = {
            "title": f"📊 {payload.title}",
            "description": payload.body,
            "color": 0x3498DB,
            "fields": [],
        }
        bot_statuses = payload.data.get("bot_statuses", [])
        status_label = {"green": "OK", "yellow": "WARN", "red": "ALERT"}
        for bot in bot_statuses:
            bot_id = bot.get("bot_id", "?")
            status = bot.get("status", "")
            label = status_label.get(status, "INFO")
            pnl = bot.get("pnl", 0)
            wins = bot.get("wins", 0)
            losses = bot.get("losses", 0)
            summary = bot.get("summary", "")
            embed["fields"].append({
                "name": f"{label} {bot_id}",
                "value": f"+${pnl:.0f} ({wins}W/{losses}L) — {summary}",
                "inline": True,
            })
        return embed

    def render_weekly_summary(self, payload: NotificationPayload) -> str:
        embed = self.render_weekly_summary_embed(payload)
        return f"{embed['title']}\n\n{embed['description']}"

    def render_weekly_summary_embed(self, payload: NotificationPayload) -> dict:
        return {
            "title": f"📈 {payload.title}",
            "description": payload.body,
            "color": 0x2ECC71,
        }

    def render_approval_request(self, request) -> str:
        lines = [
            "Suggestion Approval Request",
            f"Bot: {request.bot_id}",
            f"Kind: {request.change_kind.value}",
            f"Risk: {request.risk_tier.value}",
        ]
        if getattr(request, "strategy_id", ""):
            lines.append(f"Strategy: {request.strategy_id}")
        if getattr(request, "monthly_run_id", ""):
            lines.append(f"Monthly run: {request.monthly_run_id}")
        if getattr(request, "title", ""):
            lines.append(f"Title: {request.title}")
        if getattr(request, "summary", ""):
            lines.extend(["", request.summary])
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
        return "\n".join(lines)
