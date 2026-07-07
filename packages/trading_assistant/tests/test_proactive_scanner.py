# tests/test_proactive_scanner.py
"""Tests for the proactive notification scanner."""
import pytest

from trading_assistant.skills.proactive_scanner import ProactiveScanner
from trading_assistant.schemas.notifications import NotificationPriority


class TestMorningScan:
    @pytest.fixture
    def scanner(self):
        return ProactiveScanner()

    def test_no_events_returns_empty(self, scanner):
        result = scanner.morning_scan(events=[], errors=[], unusual_losses=[])
        assert result.has_notifications is False
        assert len(result.payloads) == 0

    def test_errors_produce_alert_payload(self, scanner):
        errors = [
            {"bot_id": "bot1", "error_type": "ConnectionError", "message": "Exchange API timeout", "severity": "HIGH"},
        ]
        result = scanner.morning_scan(events=[], errors=errors, unusual_losses=[])
        assert result.has_notifications is True
        assert len(result.payloads) == 1
        assert result.payloads[0].priority == NotificationPriority.HIGH

    def test_unusual_losses_produce_summary(self, scanner):
        losses = [
            {"bot_id": "bot2", "pnl": -500.0, "reason": "3 consecutive losses in trending market"},
        ]
        result = scanner.morning_scan(events=[], errors=[], unusual_losses=losses)
        assert result.has_notifications is True
        assert "bot2" in result.payloads[0].body.lower() or "bot2" in result.payloads[0].data.get("bot_id", "")

    def test_multiple_items_consolidated(self, scanner):
        errors = [
            {"bot_id": "bot1", "error_type": "ConnectionError", "message": "Timeout", "severity": "HIGH"},
            {"bot_id": "bot3", "error_type": "RuntimeError", "message": "Signal crash", "severity": "CRITICAL"},
        ]
        losses = [
            {"bot_id": "bot2", "pnl": -500.0, "reason": "Unusual drawdown"},
        ]
        result = scanner.morning_scan(events=[], errors=errors, unusual_losses=losses)
        assert result.has_notifications is True
        assert any(p.priority == NotificationPriority.CRITICAL for p in result.payloads)


class TestContinuousScan:
    @pytest.fixture
    def scanner(self):
        return ProactiveScanner()

    def test_no_alerts_returns_empty(self, scanner):
        result = scanner.continuous_scan(alerts=[])
        assert result.has_notifications is False

    def test_critical_alert_produces_immediate_payload(self, scanner):
        from trading_assistant.orchestrator.monitoring import Alert, AlertSeverity
        alerts = [
            Alert(severity=AlertSeverity.CRITICAL, source="heartbeat", message="Bot3 heartbeat stale for 4h"),
        ]
        result = scanner.continuous_scan(alerts=alerts)
        assert result.has_notifications is True
        assert result.payloads[0].priority == NotificationPriority.CRITICAL

    def test_high_alert_produces_payload(self, scanner):
        from trading_assistant.orchestrator.monitoring import Alert, AlertSeverity
        alerts = [
            Alert(severity=AlertSeverity.HIGH, source="stale_task", message="Daily analysis stuck"),
        ]
        result = scanner.continuous_scan(alerts=alerts)
        assert result.has_notifications is True
        assert result.payloads[0].priority == NotificationPriority.HIGH

    def test_low_alerts_batched(self, scanner):
        from trading_assistant.orchestrator.monitoring import Alert, AlertSeverity
        alerts = [
            Alert(severity=AlertSeverity.LOW, source="run_output", message="Missing file A"),
            Alert(severity=AlertSeverity.LOW, source="run_output", message="Missing file B"),
        ]
        result = scanner.continuous_scan(alerts=alerts)
        assert len(result.payloads) <= 1


class TestEveningScan:
    @pytest.fixture
    def scanner(self):
        return ProactiveScanner()

    def test_evening_scan_produces_report_trigger(self, scanner):
        result = scanner.evening_scan(date="2026-03-01", daily_report_ready=True)
        assert result.has_notifications is True
        assert result.payloads[0].notification_type == "daily_report"

    def test_evening_scan_not_ready(self, scanner):
        result = scanner.evening_scan(date="2026-03-01", daily_report_ready=False)
        assert result.has_notifications is True
        assert "pending" in result.payloads[0].body.lower() or "not ready" in result.payloads[0].body.lower()
