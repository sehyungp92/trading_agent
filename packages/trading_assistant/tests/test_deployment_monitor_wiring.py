# tests/test_deployment_monitor_wiring.py
"""Tests for deployment monitor wiring — Task 17."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


# ── AppConfig ───────────────────────────────────────────────────


class TestAppConfigDeploymentMonitoring:

    def test_feature_flag_default_false(self):
        """deployment_monitoring_enabled defaults to False."""
        config = AppConfig()
        assert config.deployment_monitoring_enabled is False

    def test_reads_from_env(self):
        """Config reads DEPLOYMENT_MONITORING_ENABLED from env."""
        with patch.dict(os.environ, {"DEPLOYMENT_MONITORING_ENABLED": "true"}):
            config = AppConfig.from_env()
            assert config.deployment_monitoring_enabled is True

    def test_reads_false_from_env(self):
        """Config reads DEPLOYMENT_MONITORING_ENABLED=false from env."""
        with patch.dict(os.environ, {"DEPLOYMENT_MONITORING_ENABLED": "false"}):
            config = AppConfig.from_env()
            assert config.deployment_monitoring_enabled is False

    def test_reads_1_from_env(self):
        """Config reads DEPLOYMENT_MONITORING_ENABLED=1 from env."""
        with patch.dict(os.environ, {"DEPLOYMENT_MONITORING_ENABLED": "1"}):
            config = AppConfig.from_env()
            assert config.deployment_monitoring_enabled is True


# ── Scheduler Jobs ──────────────────────────────────────────────


class TestSchedulerDeploymentCheck:

    def test_job_registered_when_fn_provided(self):
        """Scheduler registers deployment_check job when deployment_check_fn is provided."""
        async def _noop():
            pass

        jobs = create_scheduler_jobs(
            config=SchedulerConfig(),
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            deployment_check_fn=_noop,
        )
        job_names = [j["name"] for j in jobs]
        assert "deployment_check" in job_names
        dep_job = next(j for j in jobs if j["name"] == "deployment_check")
        assert dep_job["trigger"] == "interval"
        assert dep_job["seconds"] == 1800  # 30 minutes default

    def test_job_not_registered_when_fn_none(self):
        """Scheduler does NOT register deployment_check job when fn is None."""
        async def _noop():
            pass

        jobs = create_scheduler_jobs(
            config=SchedulerConfig(),
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            deployment_check_fn=None,
        )
        job_names = [j["name"] for j in jobs]
        assert "deployment_check" not in job_names

    def test_interval_configurable(self):
        """Scheduler uses custom interval from SchedulerConfig."""
        async def _noop():
            pass

        cfg = SchedulerConfig(deployment_check_interval_seconds=900)
        jobs = create_scheduler_jobs(
            config=cfg,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            deployment_check_fn=_noop,
        )
        dep_job = next(j for j in jobs if j["name"] == "deployment_check")
        assert dep_job["seconds"] == 900


# ── Handlers deployment_monitor ─────────────────────────────────


def _make_handlers(tmp_path: Path, deployment_monitor=None, **kwargs):
    """Create a minimal Handlers instance for testing."""
    from tests.factories import make_handlers as _factory_make_handlers

    handlers, _, _ = _factory_make_handlers(
        tmp_path,
        bots=["bot1"],
        deployment_monitor=deployment_monitor,
        **kwargs,
    )
    return handlers


class TestHandlersDeploymentMonitor:

    def test_accepts_deployment_monitor_param(self, tmp_path: Path):
        """Handlers constructor accepts deployment_monitor parameter."""
        dm = MagicMock()
        h = _make_handlers(tmp_path, deployment_monitor=dm)
        assert h._deployment_monitor is dm

    @pytest.mark.asyncio
    async def test_check_deployments_no_monitor(self, tmp_path: Path):
        """_check_deployments returns early when no deployment_monitor."""
        h = _make_handlers(tmp_path, deployment_monitor=None)
        await h._check_deployments()  # Should not raise

    @pytest.mark.asyncio
    async def test_check_deployments_processes_pending_merge(self, tmp_path: Path):
        """_check_deployments calls check_merge_status for PENDING_MERGE deployments."""
        from trading_assistant.schemas.deployment_monitoring import DeploymentStatus

        record = MagicMock()
        record.status = DeploymentStatus.PENDING_MERGE
        record.deployment_id = "dep1"

        dm = MagicMock()
        dm.get_monitoring.return_value = [record]
        dm.check_merge_status = AsyncMock()

        h = _make_handlers(tmp_path, deployment_monitor=dm)
        await h._check_deployments()

        dm.check_merge_status.assert_called_once_with("dep1")

    @pytest.mark.asyncio
    async def test_check_deployments_handles_no_active(self, tmp_path: Path):
        """_check_deployments handles no active deployments gracefully."""
        dm = MagicMock()
        dm.get_monitoring.return_value = []

        h = _make_handlers(tmp_path, deployment_monitor=dm)
        await h._check_deployments()  # Should not raise

        dm.get_monitoring.assert_called_once()


# ── App wiring ──────────────────────────────────────────────────


class TestAppDeploymentMonitorWiring:

    def test_feature_flag_off_no_monitor(self, tmp_path: Path):
        """Feature flag off: no DeploymentMonitor created."""
        config = AppConfig(
            data_dir=str(tmp_path),
            deployment_monitoring_enabled=False,
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.deployment_monitor is None

    def test_feature_flag_on_creates_monitor(self, tmp_path: Path):
        """Feature flag on: DeploymentMonitor in app state."""
        config = AppConfig(
            data_dir=str(tmp_path),
            deployment_monitoring_enabled=True,
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.deployment_monitor is not None
        from trading_assistant.skills.deployment_monitor import DeploymentMonitor

        assert isinstance(app.state.deployment_monitor, DeploymentMonitor)

    def test_deployment_monitor_gets_shared_helpers_without_autonomous(self, tmp_path: Path):
        """Deployment monitoring uses shared PR/config helpers even when autonomous is off."""
        config = AppConfig(
            data_dir=str(tmp_path),
            autonomous_enabled=False,
            deployment_monitoring_enabled=True,
            allow_unauthenticated_local=True,
        )
        from trading_assistant.orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.autonomous_pipeline is None
        assert app.state.approval_tracker is not None
        assert app.state.deployment_monitor is not None
        assert app.state.deployment_monitor._pr_builder is not None
        assert app.state.deployment_monitor._config_registry is not None
        assert app.state.deployment_monitor._file_change_generator is not None
