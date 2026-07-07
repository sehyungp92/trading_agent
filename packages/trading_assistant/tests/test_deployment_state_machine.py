"""Tests for deployment monitor state machine fixes (Task 2)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_assistant.schemas.deployment_monitoring import (
    DeploymentMetricsSnapshot,
    DeploymentStatus,
)
from trading_assistant.skills.deployment_monitor import DeploymentMonitor


def _make_snapshot(**kwargs) -> DeploymentMetricsSnapshot:
    defaults = {"bot_id": "bot1", "total_trades": 50, "win_rate": 0.55, "avg_pnl": 100.0}
    defaults.update(kwargs)
    return DeploymentMetricsSnapshot(**defaults)


@pytest.fixture
def monitor(tmp_path) -> DeploymentMonitor:
    findings = tmp_path / "findings"
    findings.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()
    return DeploymentMonitor(findings_dir=findings, curated_dir=curated)


class TestHeartbeatConfirmation:
    """2a: Require heartbeat confirmation before marking DEPLOYED."""

    def test_merged_stays_merged_without_heartbeat(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        record = monitor.get_by_id("dep1")
        record.status = DeploymentStatus.MERGED
        record.merge_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        monitor._update_record(record)

        # No heartbeat → not confirmed
        assert monitor.is_heartbeat_confirmed("dep1", None) is False
        # Status should remain MERGED (handler won't call mark_deployed)
        assert monitor.get_by_id("dep1").status == DeploymentStatus.MERGED

    def test_heartbeat_after_merge_confirms_deployment(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        record = monitor.get_by_id("dep1")
        record.status = DeploymentStatus.MERGED
        record.merge_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        monitor._update_record(record)

        # Heartbeat after merge
        hb_time = datetime.now(timezone.utc)
        assert monitor.is_heartbeat_confirmed("dep1", hb_time) is True
        monitor.mark_deployed("dep1")
        assert monitor.get_by_id("dep1").status == DeploymentStatus.DEPLOYED

    def test_heartbeat_before_merge_does_not_confirm(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        record = monitor.get_by_id("dep1")
        record.status = DeploymentStatus.MERGED
        record.merge_time = datetime.now(timezone.utc)
        monitor._update_record(record)

        # Heartbeat before merge
        old_hb = datetime.now(timezone.utc) - timedelta(hours=1)
        assert monitor.is_heartbeat_confirmed("dep1", old_hb) is False

    def test_merged_timeout_transitions_to_stale(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        record = monitor.get_by_id("dep1")
        record.status = DeploymentStatus.MERGED
        record.merge_time = datetime.now(timezone.utc) - timedelta(hours=7)
        monitor._update_record(record)

        timed_out = monitor.check_merged_timeout("dep1")
        assert timed_out is True
        assert monitor.get_by_id("dep1").status == DeploymentStatus.STALE


class TestPostDeploySnapshots:
    """2b: Accumulate post-deploy metrics instead of overwriting."""

    def test_snapshots_accumulate(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        monitor.mark_deployed("dep1")

        for i in range(5):
            monitor.record_post_deploy_metrics(
                "dep1", _make_snapshot(avg_pnl=100.0 - i * 10),
            )

        record = monitor.get_by_id("dep1")
        assert len(record.post_deploy_snapshots) == 5
        # post_deploy_metrics property returns latest
        assert record.post_deploy_metrics.avg_pnl == 60.0

    def test_regression_uses_worst_snapshot(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        monitor.record_pre_deploy_metrics("dep1", _make_snapshot(avg_pnl=100.0))
        monitor.mark_deployed("dep1")

        # Add good and bad snapshots
        monitor.record_post_deploy_metrics("dep1", _make_snapshot(avg_pnl=95.0))
        monitor.record_post_deploy_metrics("dep1", _make_snapshot(avg_pnl=30.0))  # worst
        monitor.record_post_deploy_metrics("dep1", _make_snapshot(avg_pnl=90.0))

        # Should detect regression based on worst (30.0 vs 100.0 = -70% decline)
        assert monitor.check_regression("dep1") is True

    def test_snapshots_capped_at_48(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        for i in range(55):
            monitor.record_post_deploy_metrics("dep1", _make_snapshot(avg_pnl=float(i)))

        record = monitor.get_by_id("dep1")
        assert len(record.post_deploy_snapshots) == 48


class TestTerminalStates:
    """2c: Terminal states and cleanup."""

    def test_monitoring_complete_on_window_expiry(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        monitor.mark_deployed("dep1")
        record = monitor.get_by_id("dep1")
        record.monitoring_end_time = datetime.now(timezone.utc) - timedelta(hours=1)
        monitor._update_record(record)

        assert monitor.check_monitoring_window_expired("dep1") is True
        assert monitor.get_by_id("dep1").status == DeploymentStatus.MONITORING_COMPLETE

    def test_stale_pending_after_7_days(self, monitor):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        record = monitor.get_by_id("dep1")
        record.created_at = datetime.now(timezone.utc) - timedelta(days=8)
        monitor._update_record(record)

        assert monitor.check_stale_pending("dep1") is True
        assert monitor.get_by_id("dep1").status == DeploymentStatus.STALE

    def test_get_monitoring_excludes_terminal(self, monitor):
        # Create one in each terminal state
        for i, status in enumerate([
            DeploymentStatus.MONITORING_COMPLETE,
            DeploymentStatus.STALE,
            DeploymentStatus.REGRESSION_DETECTED,
            DeploymentStatus.ROLLED_BACK,
        ]):
            monitor.create_deployment(
                deployment_id=f"dep{i}", approval_request_id=f"req{i}",
                pr_url=f"https://github.com/u/r/pull/{i}", bot_id="bot1", param_changes=[],
            )
            record = monitor.get_by_id(f"dep{i}")
            record.status = status
            monitor._update_record(record)

        # Create one PENDING_MERGE (should be included)
        monitor.create_deployment(
            deployment_id="dep_active", approval_request_id="req_active",
            pr_url="https://github.com/u/r/pull/99", bot_id="bot1", param_changes=[],
        )

        monitoring = monitor.get_monitoring()
        assert len(monitoring) == 1
        assert monitoring[0].deployment_id == "dep_active"


class TestRegressionGuard:
    """2d: Guard regression check when pre-deploy metrics missing."""

    def test_no_pre_deploy_metrics_returns_false_with_warning(self, monitor, caplog):
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1", param_changes=[],
        )
        monitor.mark_deployed("dep1")
        monitor.record_post_deploy_metrics("dep1", _make_snapshot(avg_pnl=50.0))

        import logging
        with caplog.at_level(logging.WARNING):
            result = monitor.check_regression("dep1")
        assert result is False
        assert "no baseline metrics" in caplog.text
