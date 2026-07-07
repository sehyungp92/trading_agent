"""Tests for dead-letter queue alerting in monitoring."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.orchestrator.monitoring import MonitoringCheck, MonitoringLoop, AlertSeverity


class TestDeadLetterAlerting:
    @pytest.mark.asyncio
    async def test_alerts_when_dead_letters_exist(self):
        queue = AsyncMock()
        queue.count_dead_letters = AsyncMock(return_value=2)
        check = MonitoringCheck(queue=queue)
        alerts = await check.check_dead_letters()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "2" in alerts[0].message

    @pytest.mark.asyncio
    async def test_no_alert_when_empty(self):
        queue = AsyncMock()
        queue.count_dead_letters = AsyncMock(return_value=0)
        check = MonitoringCheck(queue=queue)
        alerts = await check.check_dead_letters()
        assert alerts == []

    @pytest.mark.asyncio
    async def test_critical_when_many_dead_letters(self):
        queue = AsyncMock()
        queue.count_dead_letters = AsyncMock(return_value=10)
        check = MonitoringCheck(queue=queue, dead_letter_critical_threshold=5)
        alerts = await check.check_dead_letters()
        assert alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_dead_letters_included_in_run_all(self):
        queue = AsyncMock()
        queue.count_dead_letters = AsyncMock(return_value=1)
        check = MonitoringCheck(queue=queue)
        loop = MonitoringLoop(checks=[check])
        alerts = await loop.run_all()
        assert any(a.source == "dead_letter" for a in alerts)
