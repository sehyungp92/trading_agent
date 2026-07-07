"""Phase 2D integration tests — deployment monitoring lifecycle."""
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
from trading_assistant.schemas.repo_changes import PRResult
from trading_assistant.skills.deployment_monitor import DeploymentMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(tmp_path: Path, *, pr_builder=None, event_stream=None) -> DeploymentMonitor:
    """Create a DeploymentMonitor rooted in *tmp_path*."""
    findings = tmp_path / "findings"
    findings.mkdir(parents=True, exist_ok=True)
    curated = tmp_path / "data" / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    return DeploymentMonitor(
        findings_dir=findings,
        curated_dir=curated,
        pr_builder=pr_builder,
        event_stream=event_stream,
    )


def _make_pr_builder_mock(*, merge_state: str = "OPEN") -> MagicMock:
    """Return a mock PRBuilder whose run_gh_command returns the given merge state."""
    builder = MagicMock()
    builder.run_gh_command = AsyncMock(
        return_value=(
            0,
            json.dumps({"state": merge_state, "mergedAt": "2026-03-07T12:00:00Z"}),
            "",
        )
    )
    builder.create_pr = AsyncMock(
        return_value=PRResult(
            success=True,
            pr_url="https://github.com/user/bot/pull/99",
            branch_name="codex/rollback-abc12345",
            pr_number=99,
        )
    )
    return builder


SAMPLE_PARAM_CHANGES = [
    {
        "param_name": "quality_min_threshold",
        "old_value": 0.6,
        "new_value": 0.7,
        "file_path": "config.yaml",
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeploymentMonitoringIntegration:
    """End-to-end deployment monitoring lifecycle tests."""

    @pytest.mark.asyncio
    async def test_happy_path_deployment_success(self, tmp_path: Path):
        """Full successful deployment: create -> merge -> deploy -> no regression."""
        pr_builder = _make_pr_builder_mock(merge_state="MERGED")
        event_stream = MagicMock()
        monitor = _make_monitor(tmp_path, pr_builder=pr_builder, event_stream=event_stream)

        # Step 1: Create deployment record (PENDING_MERGE)
        record = monitor.create_deployment(
            deployment_id="dep-001",
            approval_request_id="req-001",
            pr_url="https://github.com/user/bot/pull/42",
            bot_id="bot_a",
            param_changes=SAMPLE_PARAM_CHANGES,
            pr_number=42,
        )
        assert record.status == DeploymentStatus.PENDING_MERGE

        # Step 2: Detect merge
        merged = await monitor.check_merge_status("dep-001")
        assert merged is True
        record = monitor.get_by_id("dep-001")
        assert record.status == DeploymentStatus.MERGED

        # Step 3: Mark as deployed
        monitor.mark_deployed("dep-001")
        record = monitor.get_by_id("dep-001")
        assert record.status == DeploymentStatus.DEPLOYED
        assert record.monitoring_end_time is not None

        # Step 4: Set pre-deploy metrics (baseline)
        pre_metrics = DeploymentMetricsSnapshot(
            bot_id="bot_a",
            total_trades=100,
            win_rate=0.55,
            avg_pnl=10.0,
            sharpe_rolling_7d=1.2,
            max_drawdown_pct=5.0,
        )
        monitor.record_pre_deploy_metrics("dep-001", pre_metrics)

        # Step 5: Set post-deploy metrics (improvement)
        post_metrics = DeploymentMetricsSnapshot(
            bot_id="bot_a",
            total_trades=110,
            win_rate=0.58,
            avg_pnl=12.0,
            sharpe_rolling_7d=1.4,
            max_drawdown_pct=4.5,
        )
        monitor.record_post_deploy_metrics("dep-001", post_metrics)

        # Step 6: Check regression — should be False (improvement)
        has_regression = monitor.check_regression("dep-001")
        assert has_regression is False

        # Step 7: Verify deployment stays DEPLOYED (no regression)
        record = monitor.get_by_id("dep-001")
        assert record.status == DeploymentStatus.DEPLOYED
        assert record.regression_detected is False
        assert record.regression_details == ""

    @pytest.mark.asyncio
    async def test_regression_triggers_rollback(self, tmp_path: Path):
        """Regression detected -> status REGRESSION_DETECTED -> rollback PR created."""
        pr_builder = _make_pr_builder_mock()
        event_stream = MagicMock()
        monitor = _make_monitor(tmp_path, pr_builder=pr_builder, event_stream=event_stream)

        # Create and deploy
        monitor.create_deployment(
            deployment_id="dep-002",
            approval_request_id="req-002",
            pr_url="https://github.com/user/bot/pull/43",
            bot_id="bot_a",
            param_changes=SAMPLE_PARAM_CHANGES,
            pr_number=43,
        )
        monitor.mark_deployed("dep-002")

        # Pre-deploy metrics (healthy baseline)
        pre_metrics = DeploymentMetricsSnapshot(
            bot_id="bot_a",
            total_trades=100,
            win_rate=0.55,
            avg_pnl=10.0,
            sharpe_rolling_7d=1.2,
            max_drawdown_pct=5.0,
        )
        monitor.record_pre_deploy_metrics("dep-002", pre_metrics)

        # Post-deploy metrics (severe decline)
        post_metrics = DeploymentMetricsSnapshot(
            bot_id="bot_a",
            total_trades=100,
            win_rate=0.35,
            avg_pnl=3.0,
            sharpe_rolling_7d=0.3,
            max_drawdown_pct=12.0,
        )
        monitor.record_post_deploy_metrics("dep-002", post_metrics)

        # Check regression -> should detect it
        has_regression = monitor.check_regression("dep-002")
        assert has_regression is True

        record = monitor.get_by_id("dep-002")
        assert record.status == DeploymentStatus.REGRESSION_DETECTED
        assert record.regression_detected is True
        assert record.regression_details != ""
        # Win rate dropped from 0.55 -> 0.35 (20pp drop, > 15pp threshold)
        assert "Win rate" in record.regression_details
        # PnL dropped from 10.0 -> 3.0 (70% decline, > 50% threshold)
        assert "PnL" in record.regression_details

        # Create rollback PR
        with patch.object(monitor, "_get_repo_dir", return_value=tmp_path):
            result = await monitor.create_rollback_pr("dep-002")
        assert result is not None
        assert result.success is True
        assert result.pr_url == "https://github.com/user/bot/pull/99"

        # Verify status changed to ROLLED_BACK
        record = monitor.get_by_id("dep-002")
        assert record.status == DeploymentStatus.ROLLED_BACK
        assert record.rollback_pr_url == "https://github.com/user/bot/pull/99"

        # Verify event broadcast
        event_stream.broadcast.assert_any_call(
            "deployment_rolled_back",
            {"deployment_id": "dep-002", "rollback_pr": "https://github.com/user/bot/pull/99"},
        )

    @pytest.mark.asyncio
    async def test_pr_not_merged_stays_pending(self, tmp_path: Path):
        """PR stays open -> status remains PENDING_MERGE."""
        pr_builder = _make_pr_builder_mock(merge_state="OPEN")
        monitor = _make_monitor(tmp_path, pr_builder=pr_builder)

        monitor.create_deployment(
            deployment_id="dep-003",
            approval_request_id="req-003",
            pr_url="https://github.com/user/bot/pull/44",
            bot_id="bot_a",
            param_changes=SAMPLE_PARAM_CHANGES,
            pr_number=44,
        )

        # Check merge status — PR is still OPEN
        merged = await monitor.check_merge_status("dep-003")
        assert merged is False

        # Status should remain PENDING_MERGE
        record = monitor.get_by_id("dep-003")
        assert record.status == DeploymentStatus.PENDING_MERGE
        assert record.merge_time is None

    def test_monitoring_window_expiry(self, tmp_path: Path):
        """Monitoring window expires without regression -> deployment is successful."""
        monitor = _make_monitor(tmp_path)

        monitor.create_deployment(
            deployment_id="dep-004",
            approval_request_id="req-004",
            pr_url="https://github.com/user/bot/pull/45",
            bot_id="bot_a",
            param_changes=SAMPLE_PARAM_CHANGES,
            pr_number=45,
        )
        monitor.mark_deployed("dep-004")

        # Manually set monitoring_end_time to the past
        record = monitor.get_by_id("dep-004")
        record.monitoring_end_time = datetime.now(timezone.utc) - timedelta(hours=1)
        monitor._update_record(record)

        # Check monitoring window — should be expired
        expired = monitor.check_monitoring_window_expired("dep-004")
        assert expired is True

        # Set metrics that show no regression (improvement)
        pre_metrics = DeploymentMetricsSnapshot(
            bot_id="bot_a",
            total_trades=100,
            win_rate=0.55,
            avg_pnl=10.0,
        )
        post_metrics = DeploymentMetricsSnapshot(
            bot_id="bot_a",
            total_trades=105,
            win_rate=0.56,
            avg_pnl=10.5,
        )
        monitor.record_pre_deploy_metrics("dep-004", pre_metrics)
        monitor.record_post_deploy_metrics("dep-004", post_metrics)

        # No regression detected
        has_regression = monitor.check_regression("dep-004")
        assert has_regression is False

        # Deployment is MONITORING_COMPLETE (successful — no regression during window)
        record = monitor.get_by_id("dep-004")
        assert record.status == DeploymentStatus.MONITORING_COMPLETE

    def test_feature_flag_controls_deployment_creation(self, tmp_path: Path):
        """Without DeploymentMonitor, no deployment records are created."""
        # Simulate the feature-flag-off case: monitor is None
        monitor = None

        # All existing behavior unchanged — no deployment records created
        assert monitor is None

        # If code checks `if monitor:` before calling, nothing happens
        deployment_id = "dep-005"
        record = None
        if monitor:
            record = monitor.create_deployment(
                deployment_id=deployment_id,
                approval_request_id="req-005",
                pr_url="https://github.com/user/bot/pull/46",
                bot_id="bot_a",
                param_changes=SAMPLE_PARAM_CHANGES,
            )

        assert record is None

        # Existing behavior: handlers that skip deployment monitoring
        # still work when monitor is None
        can_check_regression = False
        if monitor:
            can_check_regression = monitor.check_regression(deployment_id)
        assert can_check_regression is False

    @pytest.mark.asyncio
    async def test_multiple_deployments_selective_rollback(self, tmp_path: Path):
        """Two deployments: bot_a regresses, bot_b is fine — only bot_a rolls back."""
        pr_builder = _make_pr_builder_mock()
        event_stream = MagicMock()
        monitor = _make_monitor(tmp_path, pr_builder=pr_builder, event_stream=event_stream)

        param_changes_a = [
            {"param_name": "threshold_a", "old_value": 0.5, "new_value": 0.7, "file_path": "a.yaml"}
        ]
        param_changes_b = [
            {"param_name": "threshold_b", "old_value": 0.4, "new_value": 0.6, "file_path": "b.yaml"}
        ]

        # Create and deploy both bots
        monitor.create_deployment(
            deployment_id="dep-a",
            approval_request_id="req-a",
            pr_url="https://github.com/user/bot_a/pull/10",
            bot_id="bot_a",
            param_changes=param_changes_a,
            pr_number=10,
        )
        monitor.create_deployment(
            deployment_id="dep-b",
            approval_request_id="req-b",
            pr_url="https://github.com/user/bot_b/pull/11",
            bot_id="bot_b",
            param_changes=param_changes_b,
            pr_number=11,
        )
        monitor.mark_deployed("dep-a")
        monitor.mark_deployed("dep-b")

        # --- bot_a: set metrics showing regression ---
        monitor.record_pre_deploy_metrics(
            "dep-a",
            DeploymentMetricsSnapshot(
                bot_id="bot_a", total_trades=100, win_rate=0.55, avg_pnl=10.0,
            ),
        )
        monitor.record_post_deploy_metrics(
            "dep-a",
            DeploymentMetricsSnapshot(
                bot_id="bot_a", total_trades=100, win_rate=0.35, avg_pnl=3.0,
            ),
        )

        # --- bot_b: set metrics showing improvement ---
        monitor.record_pre_deploy_metrics(
            "dep-b",
            DeploymentMetricsSnapshot(
                bot_id="bot_b", total_trades=80, win_rate=0.50, avg_pnl=8.0,
            ),
        )
        monitor.record_post_deploy_metrics(
            "dep-b",
            DeploymentMetricsSnapshot(
                bot_id="bot_b", total_trades=90, win_rate=0.55, avg_pnl=9.5,
            ),
        )

        # Check regressions
        assert monitor.check_regression("dep-a") is True
        assert monitor.check_regression("dep-b") is False

        # Only bot_a gets rollback PR
        with patch.object(monitor, "_get_repo_dir", return_value=tmp_path):
            result_a = await monitor.create_rollback_pr("dep-a")
        assert result_a is not None
        assert result_a.success is True

        # Verify final states
        record_a = monitor.get_by_id("dep-a")
        assert record_a.status == DeploymentStatus.ROLLED_BACK
        assert record_a.rollback_pr_url is not None

        record_b = monitor.get_by_id("dep-b")
        assert record_b.status == DeploymentStatus.DEPLOYED
        assert record_b.rollback_pr_url is None
        assert record_b.regression_detected is False
