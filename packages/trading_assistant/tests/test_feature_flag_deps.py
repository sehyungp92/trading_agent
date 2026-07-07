"""Tests for feature flag dependency handling (Task 7)."""
from __future__ import annotations


import pytest

from trading_assistant.skills.deployment_monitor import DeploymentMonitor


@pytest.fixture
def monitor_no_pr(tmp_path) -> DeploymentMonitor:
    """DeploymentMonitor without pr_builder (simulates autonomous_enabled=False)."""
    findings = tmp_path / "findings"
    findings.mkdir()
    curated = tmp_path / "curated"
    curated.mkdir()
    return DeploymentMonitor(
        findings_dir=findings, curated_dir=curated, pr_builder=None,
    )


class TestRollbackWithoutPRBuilder:
    """7a: Rollback returns clear failure when pr_builder is None."""

    async def test_rollback_fails_clearly_without_pr_builder(self, monitor_no_pr):
        monitor_no_pr.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1",
            param_changes=[{"param_name": "stop_loss", "old_value": 0.02, "new_value": 0.01}],
        )

        result = await monitor_no_pr.create_rollback_pr("dep1")
        assert result is not None
        assert result.success is False
        assert "pr_builder not available" in result.error


class TestGetRepoDirFallback:
    """7b: _get_repo_dir returns None when not resolvable."""

    async def test_rollback_fails_with_unresolvable_repo(self, tmp_path):
        from unittest.mock import MagicMock

        findings = tmp_path / "findings"
        findings.mkdir()
        curated = tmp_path / "curated"
        curated.mkdir()

        mock_pr = MagicMock()
        monitor = DeploymentMonitor(
            findings_dir=findings, curated_dir=curated,
            pr_builder=mock_pr, config_registry=None,
        )
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1", bot_id="bot1",
            param_changes=[],
        )

        result = await monitor.create_rollback_pr("dep1")
        assert result is not None
        assert result.success is False
        assert "repo_dir not resolvable" in result.error

    def test_get_repo_dir_returns_none_without_registry(self, monitor_no_pr):
        assert monitor_no_pr._get_repo_dir("any_bot") is None
