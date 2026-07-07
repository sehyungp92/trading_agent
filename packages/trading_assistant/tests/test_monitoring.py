from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from trading_assistant.orchestrator.monitoring import MonitoringCheck, AlertSeverity
from trading_assistant.orchestrator.task_registry import TaskRegistry
from trading_assistant.schemas.tasks import TaskRecord, TaskStatus


@pytest.fixture
async def registry(tmp_db_path) -> TaskRegistry:
    r = TaskRegistry(db_path=str(tmp_db_path))
    await r.initialize()
    return r


@pytest.fixture
def heartbeat_dir(tmp_path: Path) -> Path:
    d = tmp_path / "heartbeats"
    d.mkdir()
    return d


class TestMonitoringLoop:
    async def test_detects_stale_tasks(self, registry: TaskRegistry):
        task = TaskRecord(id="stale1", type="daily_analysis", agent="claude-code")
        await registry.create(task)
        await registry.update_status("stale1", TaskStatus.RUNNING)

        check = MonitoringCheck(registry=registry, task_timeout_seconds=0)
        alerts = await check.check_stale_tasks()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH
        assert "stale1" in alerts[0].message

    async def test_no_alerts_when_no_stale_tasks(self, registry: TaskRegistry):
        check = MonitoringCheck(registry=registry)
        alerts = await check.check_stale_tasks()
        assert len(alerts) == 0

    def test_detects_missing_heartbeat(self, heartbeat_dir: Path):
        # bot1 reported recently, bot2 has not
        (heartbeat_dir / "bot1.heartbeat").write_text(
            datetime.now(timezone.utc).isoformat()
        )
        (heartbeat_dir / "bot2.heartbeat").write_text(
            (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        )

        check = MonitoringCheck(
            heartbeat_dir=str(heartbeat_dir),
            heartbeat_max_age_seconds=7200,  # 2 hours
        )
        alerts = check.check_heartbeats()
        assert len(alerts) == 1
        assert "bot2" in alerts[0].message
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_no_alert_when_heartbeat_fresh(self, heartbeat_dir: Path):
        (heartbeat_dir / "bot1.heartbeat").write_text(
            datetime.now(timezone.utc).isoformat()
        )
        check = MonitoringCheck(
            heartbeat_dir=str(heartbeat_dir),
            heartbeat_max_age_seconds=7200,
        )
        alerts = check.check_heartbeats()
        assert len(alerts) == 0

    def test_handles_naive_heartbeat_timestamp(self, heartbeat_dir: Path):
        """Naive datetime (no tzinfo) should not crash the heartbeat check."""
        (heartbeat_dir / "bot_naive.heartbeat").write_text("2026-03-01T12:00:00")
        check = MonitoringCheck(
            heartbeat_dir=str(heartbeat_dir),
            heartbeat_max_age_seconds=7200,
        )
        alerts = check.check_heartbeats()
        # Should produce a stale alert, not crash
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "bot_naive" in alerts[0].message

    def test_detects_missing_run_outputs(self, tmp_path: Path):
        run_dir = tmp_path / "runs" / "2026-03-01" / "daily-report"
        run_dir.mkdir(parents=True)
        # Missing expected output file
        check = MonitoringCheck()
        alerts = check.check_run_outputs(
            run_dir=str(run_dir),
            expected_files=["daily_report.md", "report_checklist.json"],
        )
        assert len(alerts) == 2
        assert all(a.severity == AlertSeverity.MEDIUM for a in alerts)
