# tests/test_deployment_monitoring_schemas.py
"""Tests for deployment monitoring schemas."""
from __future__ import annotations

from datetime import datetime, timezone

from trading_assistant.schemas.deployment_monitoring import (
    DeploymentMetricsSnapshot,
    DeploymentRecord,
    DeploymentStatus,
)


class TestDeploymentStatus:
    def test_enum_has_all_expected_values(self):
        expected = {
            "pending_merge",
            "merged",
            "deploying",
            "deployed",
            "regression_detected",
            "rolled_back",
            "monitoring_complete",
            "stale",
        }
        actual = {s.value for s in DeploymentStatus}
        assert actual == expected


class TestDeploymentMetricsSnapshot:
    def test_serialization_roundtrip(self):
        snap = DeploymentMetricsSnapshot(
            bot_id="bot1",
            total_trades=150,
            win_rate=0.62,
            avg_pnl=42.5,
            sharpe_rolling_7d=1.35,
            max_drawdown_pct=-8.2,
        )
        data = snap.model_dump(mode="json")
        restored = DeploymentMetricsSnapshot(**data)
        assert restored.bot_id == "bot1"
        assert restored.total_trades == 150
        assert restored.win_rate == 0.62
        assert restored.avg_pnl == 42.5
        assert restored.sharpe_rolling_7d == 1.35
        assert restored.max_drawdown_pct == -8.2


class TestDeploymentRecord:
    def test_serialization_roundtrip(self):
        record = DeploymentRecord(
            deployment_id="dep-001",
            approval_request_id="req-001",
            pr_url="https://github.com/org/repo/pull/42",
            pr_number=42,
            bot_id="bot1",
            param_changes=[
                {"param_name": "quality_min", "old_value": 0.5, "new_value": 0.6}
            ],
        )
        data = record.model_dump(mode="json")
        restored = DeploymentRecord(**data)
        assert restored.deployment_id == "dep-001"
        assert restored.pr_number == 42
        assert len(restored.param_changes) == 1
        assert restored.param_changes[0]["param_name"] == "quality_min"

    def test_defaults(self):
        record = DeploymentRecord(
            deployment_id="dep-002",
            approval_request_id="req-002",
            pr_url="https://github.com/org/repo/pull/10",
            bot_id="bot2",
        )
        assert record.status == DeploymentStatus.PENDING_MERGE
        assert record.regression_detected is False
        assert record.regression_details == ""
        assert record.param_changes == []
        assert record.pr_number == 0
        assert record.merge_time is None
        assert record.deploy_detected_time is None
        assert record.monitoring_window_hours == 24
        assert record.monitoring_end_time is None
        assert record.pre_deploy_metrics is None
        assert record.post_deploy_metrics is None
        assert record.rollback_pr_url is None
        assert record.created_at is not None

    def test_with_pre_and_post_deploy_metrics(self):
        pre = DeploymentMetricsSnapshot(
            bot_id="bot1",
            total_trades=100,
            win_rate=0.55,
            avg_pnl=30.0,
            sharpe_rolling_7d=1.1,
            max_drawdown_pct=-10.0,
        )
        post = DeploymentMetricsSnapshot(
            bot_id="bot1",
            total_trades=120,
            win_rate=0.60,
            avg_pnl=35.0,
            sharpe_rolling_7d=1.3,
            max_drawdown_pct=-7.5,
        )
        record = DeploymentRecord(
            deployment_id="dep-003",
            approval_request_id="req-003",
            pr_url="https://github.com/org/repo/pull/50",
            bot_id="bot1",
            status=DeploymentStatus.DEPLOYED,
            pre_deploy_metrics=pre,
            post_deploy_snapshots=[post],
        )
        assert record.pre_deploy_metrics is not None
        assert record.post_deploy_metrics is not None
        assert record.pre_deploy_metrics.win_rate == 0.55
        assert record.post_deploy_metrics.win_rate == 0.60
        # Roundtrip preserves nested models
        data = record.model_dump(mode="json")
        restored = DeploymentRecord(**data)
        assert restored.pre_deploy_metrics.total_trades == 100
        assert restored.post_deploy_metrics.total_trades == 120

    def test_with_regression_details(self):
        record = DeploymentRecord(
            deployment_id="dep-004",
            approval_request_id="req-004",
            pr_url="https://github.com/org/repo/pull/55",
            pr_number=55,
            bot_id="bot1",
            status=DeploymentStatus.REGRESSION_DETECTED,
            merge_time=datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc),
            deploy_detected_time=datetime(2026, 3, 5, 12, 30, tzinfo=timezone.utc),
            monitoring_end_time=datetime(2026, 3, 6, 12, 30, tzinfo=timezone.utc),
            regression_detected=True,
            regression_details="Win rate dropped from 0.55 to 0.40 (-27%)",
            rollback_pr_url="https://github.com/org/repo/pull/56",
        )
        assert record.status == DeploymentStatus.REGRESSION_DETECTED
        assert record.regression_detected is True
        assert "Win rate dropped" in record.regression_details
        assert record.rollback_pr_url is not None
        assert record.merge_time is not None
        assert record.deploy_detected_time is not None
        assert record.monitoring_end_time is not None
