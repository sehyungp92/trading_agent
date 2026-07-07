# tests/test_discord_renderer.py
"""Tests for Discord embed renderer."""
from trading_assistant.comms.discord_renderer import DiscordRenderer
from trading_assistant.comms.renderer import MessageRenderer
from trading_assistant.schemas.notifications import (
    ControlPanelState,
    BotStatusLine,
    NotificationPayload,
    NotificationPriority,
)


class TestDiscordRendererProtocol:
    def test_implements_message_renderer(self):
        renderer = DiscordRenderer()
        assert isinstance(renderer, MessageRenderer)


class TestDiscordRenderAlert:
    def test_critical_alert_embed(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot3 crash",
            body="RuntimeError in signal handler",
        )
        embed = DiscordRenderer().render_alert_embed(payload)
        assert embed["title"] == "🚨 CRITICAL — Bot3 crash"
        assert embed["color"] == 0xFF0000
        assert "RuntimeError" in embed["description"]

    def test_normal_alert_embed(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.NORMAL,
            title="Slow response",
            body="API latency > 2s",
        )
        embed = DiscordRenderer().render_alert_embed(payload)
        assert embed["color"] == 0x3498DB

    def test_render_alert_returns_string(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Test",
            body="Body",
        )
        text = DiscordRenderer().render_alert(payload)
        assert isinstance(text, str)
        assert "Test" in text


class TestDiscordRenderDailyReport:
    def test_daily_report_embed_has_fields(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report — March 1, 2026",
            body="Portfolio: +$342 (+1.2%)",
            data={
                "bot_statuses": [
                    {"bot_id": "Bot1", "status": "green", "pnl": 210.0, "wins": 4, "losses": 1, "summary": "Strong"},
                    {"bot_id": "Bot2", "status": "red", "pnl": -50.0, "wins": 1, "losses": 3, "summary": "Filter"},
                ],
            },
        )
        embed = DiscordRenderer().render_daily_report_embed(payload)
        assert embed["title"] == "📊 Daily Report — March 1, 2026"
        field_names = [f["name"] for f in embed.get("fields", [])]
        assert any("Bot1" in n for n in field_names)
        assert any("Bot2" in n for n in field_names)

    def test_daily_report_without_bot_data(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report",
            body="Summary here",
        )
        embed = DiscordRenderer().render_daily_report_embed(payload)
        assert embed["title"] == "📊 Daily Report"
        assert embed["description"] == "Summary here"


class TestDiscordRenderWeeklySummary:
    def test_weekly_embed(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly Summary — Feb 24–Mar 1",
            body="Total PnL: +$1,200\n3 bots active",
        )
        embed = DiscordRenderer().render_weekly_summary_embed(payload)
        assert embed["title"] == "📈 Weekly Summary — Feb 24–Mar 1"
        assert "1,200" in embed["description"]

    def test_weekly_embed_color(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly",
            body="Good week",
        )
        embed = DiscordRenderer().render_weekly_summary_embed(payload)
        assert embed["color"] == 0x2ECC71


class TestDiscordRenderControlPanel:
    def test_control_panel_text(self):
        panel = ControlPanelState(
            date="2026-03-01",
            portfolio_pnl=342.0,
            portfolio_pnl_pct=1.2,
            bot_statuses=[
                BotStatusLine(bot_id="Bot1", status="green", pnl=210.0, wins=4, losses=1, summary="Good"),
            ],
        )
        text = DiscordRenderer().render_control_panel(panel)
        assert "Bot1" in text
        assert "342" in text
