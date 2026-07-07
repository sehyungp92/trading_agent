# tests/test_renderer.py
"""Tests for message renderer protocol and plain text renderer."""
from trading_assistant.schemas.notifications import (
    ControlPanelState,
    BotStatusLine,
    NotificationPayload,
    NotificationPriority,
)
from trading_assistant.comms.renderer import MessageRenderer, PlainTextRenderer


class TestMessageRendererProtocol:
    def test_plain_text_renderer_implements_protocol(self):
        renderer = PlainTextRenderer()
        assert isinstance(renderer, MessageRenderer)


class TestPlainTextRenderControlPanel:
    def test_renders_full_panel(self):
        panel = ControlPanelState(
            date="2026-03-01",
            portfolio_pnl=342.0,
            portfolio_pnl_pct=1.2,
            drawdown_pct=-0.3,
            exposure_pct=47.0,
            daily_report_ready=True,
            alert_count=1,
            alert_summary="Bot3 volume filter",
            monthly_validation_status="Bot2 package ready",
            pending_pr_count=0,
            risk_status="OK",
            risk_detail="concentration: 35/100",
            bot_statuses=[
                BotStatusLine(bot_id="Bot1", status="green", pnl=210.0, wins=4, losses=1, summary="Strong trend"),
                BotStatusLine(bot_id="Bot2", status="yellow", pnl=82.0, wins=3, losses=2, summary="Normal losses"),
            ],
        )
        result = PlainTextRenderer().render_control_panel(panel)
        assert "March 1, 2026" in result or "2026-03-01" in result
        assert "+$342" in result or "342" in result
        assert "Bot3 volume filter" in result
        assert "Monthly validation" in result
        assert "Bot1" in result
        assert "Bot2" in result

    def test_renders_empty_panel(self):
        panel = ControlPanelState(date="2026-03-01")
        result = PlainTextRenderer().render_control_panel(panel)
        assert "2026-03-01" in result


class TestPlainTextRenderAlert:
    def test_renders_critical_alert(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot3 crash",
            body="RuntimeError in signal handler",
            data={"bot_id": "bot3"},
        )
        result = PlainTextRenderer().render_alert(payload)
        assert "CRITICAL" in result
        assert "Bot3 crash" in result
        assert "RuntimeError" in result

    def test_renders_medium_alert(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.NORMAL,
            title="Minor issue",
            body="Slow API response",
        )
        result = PlainTextRenderer().render_alert(payload)
        assert "Minor issue" in result


class TestPlainTextRenderDailyReport:
    def test_renders_daily_report_body(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            priority=NotificationPriority.NORMAL,
            title="Daily Report — March 1, 2026",
            body="Portfolio: +$342 (+1.2%) | DD: -0.3%\n\nBot1: +$210 (4W/1L)",
        )
        result = PlainTextRenderer().render_daily_report(payload)
        assert "Daily Report" in result
        assert "+$342" in result or "342" in result
        assert "Bot1" in result

    def test_renders_daily_report_with_data(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report",
            body="Summary",
            data={"bot_statuses": [{"bot_id": "bot1", "pnl": 100}]},
        )
        result = PlainTextRenderer().render_daily_report(payload)
        assert "Daily Report" in result


class TestPlainTextRenderWeeklySummary:
    def test_renders_weekly_summary(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly Summary — Feb 24–Mar 1",
            body="Total PnL: +$1,200\n3 bots active",
        )
        result = PlainTextRenderer().render_weekly_summary(payload)
        assert "Weekly Summary" in result
        assert "$1,200" in result or "1,200" in result


class TestPlainTextRenderGeneric:
    def test_render_dispatches_by_type(self):
        renderer = PlainTextRenderer()
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Test",
            body="Test body",
        )
        result = renderer.render(payload)
        assert "CRITICAL" in result
        assert "Test" in result

    def test_render_unknown_type_falls_back(self):
        renderer = PlainTextRenderer()
        payload = NotificationPayload(
            notification_type="unknown_type",
            title="Something",
            body="Details here",
        )
        result = renderer.render(payload)
        assert "Something" in result
        assert "Details here" in result
