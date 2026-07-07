# tests/test_email_renderer.py
"""Tests for email HTML renderer."""
from trading_assistant.comms.email_renderer import EmailRenderer
from trading_assistant.comms.renderer import MessageRenderer
from trading_assistant.schemas.notifications import (
    NotificationPayload,
    NotificationPriority,
)


class TestEmailRendererProtocol:
    def test_implements_message_renderer(self):
        renderer = EmailRenderer()
        assert isinstance(renderer, MessageRenderer)


class TestEmailRenderWeeklyDigest:
    def test_renders_html_structure(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly Summary — Feb 24–Mar 1",
            body="Total PnL: +$1,200\n3 bots active\n\nBot1: +$800\nBot2: +$400",
        )
        html = EmailRenderer().render_weekly_html(payload)
        assert "<html" in html
        assert "Weekly Summary" in html
        assert "$1,200" in html or "1,200" in html

    def test_renders_with_inline_styles(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly",
            body="Content",
        )
        html = EmailRenderer().render_weekly_html(payload)
        assert "style=" in html

    def test_newlines_become_paragraphs(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly",
            body="Line 1\n\nLine 2",
        )
        html = EmailRenderer().render_weekly_html(payload)
        assert "<p>" in html or "<br" in html


class TestEmailRenderAlert:
    def test_critical_alert_html(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot crash",
            body="RuntimeError",
        )
        text = EmailRenderer().render_alert(payload)
        assert "CRITICAL" in text
        assert "Bot crash" in text

    def test_render_protocol_method(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly",
            body="Content",
        )
        text = EmailRenderer().render_weekly_summary(payload)
        assert "Weekly" in text


class TestEmailRenderDailyReport:
    def test_daily_report_html(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report",
            body="Portfolio: +$342",
        )
        text = EmailRenderer().render_daily_report(payload)
        assert "Daily Report" in text


class TestEmailSubjectLine:
    def test_subject_from_payload(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly Summary — Feb 24–Mar 1",
            body="Content",
        )
        subject = EmailRenderer().build_subject(payload)
        assert "Weekly Summary" in subject

    def test_critical_alert_subject(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot crash",
            body="Details",
        )
        subject = EmailRenderer().build_subject(payload)
        assert "CRITICAL" in subject or "🚨" in subject
