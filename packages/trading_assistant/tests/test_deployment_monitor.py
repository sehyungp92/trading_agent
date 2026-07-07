# tests/test_deployment_monitor.py
"""Tests for DeploymentMonitor."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_assistant.schemas.deployment_monitoring import (
    DeploymentMetricsSnapshot,
    DeploymentStatus,
)
from trading_assistant.schemas.notifications import NotificationPriority
from trading_assistant.schemas.repo_changes import PRResult
from trading_assistant.skills.deployment_monitor import DeploymentMonitor


@pytest.fixture
def monitor(tmp_path: Path) -> DeploymentMonitor:
    findings = tmp_path / "findings"
    findings.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()
    return DeploymentMonitor(findings_dir=findings, curated_dir=curated)


def _make_snapshot(
    bot_id: str = "bot1",
    avg_pnl: float = 100.0,
    win_rate: float = 0.55,
    max_drawdown_pct: float = 5.0,
    **kwargs,
) -> DeploymentMetricsSnapshot:
    return DeploymentMetricsSnapshot(
        bot_id=bot_id,
        avg_pnl=avg_pnl,
        win_rate=win_rate,
        max_drawdown_pct=max_drawdown_pct,
        **kwargs,
    )


class TestDeploymentMonitor:
    def test_create_deployment(self, monitor: DeploymentMonitor):
        record = monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[{"param_name": "sl_pct", "old_value": 2.0, "new_value": 1.5}],
            pr_number=42,
        )
        assert record.deployment_id == "dep1"
        assert record.status == DeploymentStatus.PENDING_MERGE
        assert record.bot_id == "bot1"
        assert record.pr_number == 42

        found = monitor.get_by_id("dep1")
        assert found is not None
        assert found.deployment_id == "dep1"

    def test_create_deployment_deduplicates(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        second = monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req2",
            pr_url="https://github.com/user/repo/pull/99",
            bot_id="bot2",
            param_changes=[],
        )
        # Should return existing, not create new
        assert second.bot_id == "bot1"
        assert second.pr_url == "https://github.com/user/repo/pull/42"
        all_records = monitor._load_all()
        assert len(all_records) == 1

    @pytest.mark.asyncio
    async def test_check_merge_status_merged(self, tmp_path: Path):
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        mock_pr_builder = MagicMock()
        mock_pr_builder.run_gh_command = AsyncMock(
            return_value=(0, json.dumps({"state": "MERGED", "mergedAt": "2026-03-07T12:00:00Z"}), "")
        )

        mon = DeploymentMonitor(
            findings_dir=findings, curated_dir=curated, pr_builder=mock_pr_builder
        )
        mon.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )

        result = await mon.check_merge_status("dep1")
        assert result is True
        record = mon.get_by_id("dep1")
        assert record.status == DeploymentStatus.MERGED
        assert record.merge_time is not None

    @pytest.mark.asyncio
    async def test_check_merge_status_not_merged(self, tmp_path: Path):
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        mock_pr_builder = MagicMock()
        mock_pr_builder.run_gh_command = AsyncMock(
            return_value=(0, json.dumps({"state": "OPEN"}), "")
        )

        mon = DeploymentMonitor(
            findings_dir=findings, curated_dir=curated, pr_builder=mock_pr_builder
        )
        mon.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )

        result = await mon.check_merge_status("dep1")
        assert result is False
        record = mon.get_by_id("dep1")
        assert record.status == DeploymentStatus.PENDING_MERGE

    def test_record_pre_deploy_metrics(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        snapshot = _make_snapshot(avg_pnl=120.0, win_rate=0.60)
        monitor.record_pre_deploy_metrics("dep1", snapshot)

        record = monitor.get_by_id("dep1")
        assert record.pre_deploy_metrics is not None
        assert record.pre_deploy_metrics.avg_pnl == 120.0
        assert record.pre_deploy_metrics.win_rate == 0.60

    def test_record_post_deploy_metrics(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        snapshot = _make_snapshot(avg_pnl=80.0, win_rate=0.45)
        monitor.record_post_deploy_metrics("dep1", snapshot)

        record = monitor.get_by_id("dep1")
        assert record.post_deploy_metrics is not None
        assert record.post_deploy_metrics.avg_pnl == 80.0

    def test_check_regression_pnl_decline(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        # Pre: avg_pnl = 100, Post: avg_pnl = 30 (70% decline, >50% threshold)
        monitor.record_pre_deploy_metrics("dep1", _make_snapshot(avg_pnl=100.0))
        monitor.record_post_deploy_metrics("dep1", _make_snapshot(avg_pnl=30.0))

        assert monitor.check_regression("dep1") is True
        record = monitor.get_by_id("dep1")
        assert record.regression_detected is True
        assert record.status == DeploymentStatus.REGRESSION_DETECTED
        assert "PnL declined" in record.regression_details

    def test_check_regression_win_rate_decline(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        # Pre: win_rate = 0.60, Post: win_rate = 0.40 (20pp decline, >15pp threshold)
        monitor.record_pre_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, win_rate=0.60)
        )
        monitor.record_post_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, win_rate=0.40)
        )

        assert monitor.check_regression("dep1") is True
        record = monitor.get_by_id("dep1")
        assert record.regression_detected is True
        assert "Win rate declined" in record.regression_details

    def test_check_regression_no_regression(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        # Metrics within normal range -- small changes
        monitor.record_pre_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, win_rate=0.55, max_drawdown_pct=5.0)
        )
        monitor.record_post_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=95.0, win_rate=0.53, max_drawdown_pct=5.5)
        )

        assert monitor.check_regression("dep1") is False
        record = monitor.get_by_id("dep1")
        assert record.regression_detected is False
        assert record.status != DeploymentStatus.REGRESSION_DETECTED

    def test_check_regression_drawdown_worse(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        # Pre: max_drawdown = 5%, Post: max_drawdown = 10% (100% worse, >50% threshold)
        monitor.record_pre_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, max_drawdown_pct=5.0)
        )
        monitor.record_post_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, max_drawdown_pct=10.0)
        )

        assert monitor.check_regression("dep1") is True
        record = monitor.get_by_id("dep1")
        assert "drawdown worsened" in record.regression_details

    @pytest.mark.asyncio
    async def test_create_rollback_pr(self, tmp_path: Path):
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        mock_pr_builder = MagicMock()
        mock_pr_builder.create_pr = AsyncMock(
            return_value=PRResult(
                success=True,
                pr_url="https://github.com/user/repo/pull/99",
                branch_name="codex/rollback-dep1xxxx",
            )
        )

        mon = DeploymentMonitor(
            findings_dir=findings, curated_dir=curated, pr_builder=mock_pr_builder
        )
        mon.create_deployment(
            deployment_id="dep1abcd",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[
                {"param_name": "sl_pct", "old_value": 2.0, "new_value": 1.5, "file_path": "config.yaml"}
            ],
        )
        # Set regression details so the PR body is populated
        record = mon.get_by_id("dep1abcd")
        record.regression_detected = True
        record.regression_details = "Avg PnL declined 60%"
        record.status = DeploymentStatus.REGRESSION_DETECTED
        mon._update_record(record)

        with patch.object(mon, "_get_repo_dir", return_value=tmp_path):
            result = await mon.create_rollback_pr("dep1abcd")
        assert result is not None
        assert result.success is True
        assert result.pr_url == "https://github.com/user/repo/pull/99"

        updated = mon.get_by_id("dep1abcd")
        assert updated.status == DeploymentStatus.ROLLED_BACK
        assert updated.rollback_pr_url == "https://github.com/user/repo/pull/99"

        # Verify the PRRequest was constructed correctly
        call_args = mock_pr_builder.create_pr.call_args
        pr_request = call_args[0][0]
        assert pr_request.branch_name == "codex/rollback-dep1abcd"
        assert "ROLLBACK" in pr_request.title
        # Without file_change_generator, fallback produces descriptive diff_preview
        assert "Revert sl_pct" in pr_request.file_changes[0].diff_preview
        assert "1.5 -> 2.0" in pr_request.file_changes[0].diff_preview

    def test_monitoring_window_expired(self, monitor: DeploymentMonitor):
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        monitor.mark_deployed("dep1")

        # Not yet expired (just deployed)
        assert monitor.check_monitoring_window_expired("dep1") is False

        # Force monitoring_end_time to the past
        record = monitor.get_by_id("dep1")
        record.monitoring_end_time = datetime.now(timezone.utc) - timedelta(hours=1)
        monitor._update_record(record)

        assert monitor.check_monitoring_window_expired("dep1") is True

    def test_get_monitoring_filters_status(self, monitor: DeploymentMonitor):
        # Create deployments with various statuses
        monitor.create_deployment(
            deployment_id="dep_pending",
            approval_request_id="req1",
            pr_url="url1",
            bot_id="bot1",
            param_changes=[],
        )
        monitor.create_deployment(
            deployment_id="dep_deployed",
            approval_request_id="req2",
            pr_url="url2",
            bot_id="bot1",
            param_changes=[],
        )
        monitor.mark_deployed("dep_deployed")

        monitor.create_deployment(
            deployment_id="dep_regression",
            approval_request_id="req3",
            pr_url="url3",
            bot_id="bot1",
            param_changes=[],
        )
        # Manually set to REGRESSION_DETECTED
        record = monitor.get_by_id("dep_regression")
        record.status = DeploymentStatus.REGRESSION_DETECTED
        monitor._update_record(record)

        monitor.create_deployment(
            deployment_id="dep_rolled_back",
            approval_request_id="req4",
            pr_url="url4",
            bot_id="bot1",
            param_changes=[],
        )
        rb = monitor.get_by_id("dep_rolled_back")
        rb.status = DeploymentStatus.ROLLED_BACK
        monitor._update_record(rb)

        # get_monitoring should only return PENDING_MERGE, MERGED, DEPLOYED
        monitoring = monitor.get_monitoring()
        ids = {d.deployment_id for d in monitoring}
        assert "dep_pending" in ids
        assert "dep_deployed" in ids
        assert "dep_regression" not in ids
        assert "dep_rolled_back" not in ids

    def test_jsonl_persistence(self, tmp_path: Path):
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        mon1 = DeploymentMonitor(findings_dir=findings, curated_dir=curated)
        mon1.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[{"param_name": "sl_pct", "old_value": 2.0, "new_value": 1.5}],
            pr_number=42,
        )
        mon1.record_pre_deploy_metrics("dep1", _make_snapshot(avg_pnl=100.0))
        mon1.mark_deployed("dep1")

        # New instance loads from same file
        mon2 = DeploymentMonitor(findings_dir=findings, curated_dir=curated)
        record = mon2.get_by_id("dep1")
        assert record is not None
        assert record.deployment_id == "dep1"
        assert record.status == DeploymentStatus.DEPLOYED
        assert record.pre_deploy_metrics is not None
        assert record.pre_deploy_metrics.avg_pnl == 100.0
        assert record.param_changes[0]["param_name"] == "sl_pct"


class TestCheckDeploymentsHandler:
    """Tests for Handlers._check_deployments() state machine."""

    def _make_handlers(self, monitor, dispatcher=None):
        """Create a Handlers instance with mocked dependencies."""
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.schemas.notifications import NotificationPreferences

        agent_runner = MagicMock()
        event_stream = EventStream()
        disp = dispatcher or MagicMock()
        disp.dispatch = AsyncMock()
        prefs = NotificationPreferences()
        handlers = Handlers(
            agent_runner=agent_runner,
            event_stream=event_stream,
            dispatcher=disp,
            notification_prefs=prefs,
            curated_dir=Path("/tmp/curated"),
            memory_dir=Path("/tmp/memory"),
            runs_dir=Path("/tmp/runs"),
            source_root=Path("/tmp"),
            bots=["bot1"],
            deployment_monitor=monitor,
        )
        return handlers

    @pytest.mark.asyncio
    async def test_check_deployments_handles_merged(self, tmp_path: Path):
        """MERGED state with heartbeat → collect pre-deploy metrics + mark_deployed."""
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        monitor = DeploymentMonitor(findings_dir=findings, curated_dir=curated)
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        # Manually set to MERGED with merge_time in the past
        record = monitor.get_by_id("dep1")
        record.status = DeploymentStatus.MERGED
        record.merge_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        monitor._update_record(record)

        # Set up curated data so collect_metrics_snapshot has something to read
        bot_dir = curated / "2026-03-07" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "trade_count": 50, "win_rate": 0.55, "avg_pnl": 100.0,
            "sharpe_ratio": 1.2, "max_drawdown_pct": 5.0,
        }))

        # Set up heartbeat file with timestamp after merge
        heartbeat_dir = tmp_path / "heartbeats"
        heartbeat_dir.mkdir(parents=True)
        (heartbeat_dir / "bot1.json").write_text(json.dumps({
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }))

        handlers = self._make_handlers(monitor)
        handlers._heartbeat_dir = heartbeat_dir
        await handlers._check_deployments()

        updated = monitor.get_by_id("dep1")
        assert updated.status == DeploymentStatus.DEPLOYED
        assert updated.pre_deploy_metrics is not None
        assert updated.pre_deploy_metrics.avg_pnl == 100.0

    @pytest.mark.asyncio
    async def test_check_deployments_handles_deployed_no_regression(self, tmp_path: Path):
        """DEPLOYED without regression → continue monitoring."""
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        monitor = DeploymentMonitor(findings_dir=findings, curated_dir=curated)
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        # Set up pre-deploy metrics
        monitor.record_pre_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, win_rate=0.55),
        )
        monitor.mark_deployed("dep1")

        # Set up curated data with similar metrics (no regression)
        bot_dir = curated / "2026-03-07" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "trade_count": 45, "win_rate": 0.54, "avg_pnl": 95.0,
            "sharpe_ratio": 1.1, "max_drawdown_pct": 5.5,
        }))

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()
        handlers = self._make_handlers(monitor, dispatcher)
        await handlers._check_deployments()

        updated = monitor.get_by_id("dep1")
        assert updated.regression_detected is False
        # No notification sent for non-regression
        dispatcher.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_deployments_handles_deployed_regression(self, tmp_path: Path):
        """DEPLOYED with regression → rollback PR + notification."""
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        mock_pr_builder = MagicMock()
        mock_pr_builder.create_pr = AsyncMock(
            return_value=PRResult(
                success=True,
                pr_url="https://github.com/user/repo/pull/99",
                branch_name="codex/rollback-dep1",
            )
        )

        monitor = DeploymentMonitor(
            findings_dir=findings, curated_dir=curated, pr_builder=mock_pr_builder,
        )
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[
                {"param_name": "sl_pct", "old_value": 2.0, "new_value": 1.5,
                 "file_path": "config.yaml"},
            ],
        )
        # Pre-deploy: good metrics
        monitor.record_pre_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, win_rate=0.55),
        )
        monitor.mark_deployed("dep1")

        # Curated data shows severe decline (70% PnL drop)
        bot_dir = curated / "2026-03-07" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "trade_count": 50, "win_rate": 0.35, "avg_pnl": 30.0,
            "sharpe_ratio": 0.3, "max_drawdown_pct": 12.0,
        }))

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()
        handlers = self._make_handlers(monitor, dispatcher)
        with patch.object(monitor, "_get_repo_dir", return_value=tmp_path):
            await handlers._check_deployments()

        updated = monitor.get_by_id("dep1")
        assert updated.regression_detected is True
        assert updated.status == DeploymentStatus.ROLLED_BACK
        assert updated.rollback_pr_url == "https://github.com/user/repo/pull/99"
        # Notification sent
        dispatcher.dispatch.assert_called_once()
        payload = dispatcher.dispatch.call_args[0][0]
        assert payload.priority == NotificationPriority.CRITICAL
        assert "Regression Detected" in payload.title

    @pytest.mark.asyncio
    async def test_check_deployments_monitoring_expired(self, tmp_path: Path):
        """Expired monitoring window → clean exit, no regression check."""
        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        monitor = DeploymentMonitor(findings_dir=findings, curated_dir=curated)
        monitor.create_deployment(
            deployment_id="dep1",
            approval_request_id="req1",
            pr_url="https://github.com/user/repo/pull/42",
            bot_id="bot1",
            param_changes=[],
        )
        monitor.record_pre_deploy_metrics(
            "dep1", _make_snapshot(avg_pnl=100.0, win_rate=0.55),
        )
        monitor.mark_deployed("dep1")

        # Force monitoring window to expire
        record = monitor.get_by_id("dep1")
        record.monitoring_end_time = datetime.now(timezone.utc) - timedelta(hours=1)
        monitor._update_record(record)

        handlers = self._make_handlers(monitor)
        await handlers._check_deployments()

        # Deployment should be MONITORING_COMPLETE (not regressed, not rolled back)
        updated = monitor.get_by_id("dep1")
        assert updated.status == DeploymentStatus.MONITORING_COMPLETE
        assert updated.regression_detected is False
