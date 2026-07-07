"""Tests for plugin/hook system (M2)."""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.comms.hooks import HookPipeline
from trading_assistant.comms.hooks.risk_injection import RiskInjectionHook
from trading_assistant.comms.hooks.audit_logger import AuditLoggerHook
from trading_assistant.schemas.notifications import NotificationPayload, NotificationPriority


class UppercaseHook:
    """Test hook that uppercases title."""

    def transform(self, payload: NotificationPayload) -> NotificationPayload:
        return payload.model_copy(update={"title": payload.title.upper()})


class FailingHook:
    """Test hook that always fails."""

    def transform(self, payload: NotificationPayload) -> NotificationPayload:
        raise RuntimeError("hook failure")


class TestHookPipeline:
    def test_single_hook(self):
        pipeline = HookPipeline()
        pipeline.add(UppercaseHook())
        payload = NotificationPayload(notification_type="test", title="hello")
        result = pipeline.run(payload)
        assert result.title == "HELLO"

    def test_chained_hooks(self):
        pipeline = HookPipeline()
        pipeline.add(UppercaseHook())
        pipeline.add(UppercaseHook())  # Idempotent
        payload = NotificationPayload(notification_type="test", title="hello")
        result = pipeline.run(payload)
        assert result.title == "HELLO"

    def test_empty_pipeline_passthrough(self):
        pipeline = HookPipeline()
        payload = NotificationPayload(notification_type="test", title="hello")
        result = pipeline.run(payload)
        assert result.title == "hello"

    def test_failing_hook_skipped(self):
        pipeline = HookPipeline()
        pipeline.add(FailingHook())
        pipeline.add(UppercaseHook())
        payload = NotificationPayload(notification_type="test", title="hello")
        result = pipeline.run(payload)
        assert result.title == "HELLO"


class TestRiskInjectionHook:
    def test_adds_warning_above_threshold(self):
        hook = RiskInjectionHook(drawdown_threshold_pct=5.0)
        hook.set_drawdown(7.5)
        payload = NotificationPayload(notification_type="test", body="Daily report ready.")
        result = hook.transform(payload)
        assert "RISK WARNING" in result.body
        assert "7.5%" in result.body

    def test_no_warning_below_threshold(self):
        hook = RiskInjectionHook(drawdown_threshold_pct=5.0)
        hook.set_drawdown(3.0)
        payload = NotificationPayload(notification_type="test", body="All good.")
        result = hook.transform(payload)
        assert "RISK WARNING" not in result.body


class TestAuditLoggerHook:
    def test_logs_to_file(self, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        hook = AuditLoggerHook(log_path=log_path)
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Report Ready",
            priority=NotificationPriority.NORMAL,
        )
        hook.transform(payload)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["notification_type"] == "daily_report"
        assert entry["priority"] == "normal"

    def test_appends_multiple_entries(self, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        hook = AuditLoggerHook(log_path=log_path)
        for i in range(3):
            payload = NotificationPayload(notification_type=f"type-{i}", title=f"T{i}")
            hook.transform(payload)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3
