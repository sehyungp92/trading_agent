"""Deployment monitoring schemas — track post-merge deployment lifecycle.

Monitors parameter changes from PR merge through deployment to regression detection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DeploymentStatus(str, Enum):
    PENDING_MERGE = "pending_merge"
    MERGED = "merged"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    REGRESSION_DETECTED = "regression_detected"
    ROLLED_BACK = "rolled_back"
    MONITORING_COMPLETE = "monitoring_complete"
    STALE = "stale"


class DeploymentMetricsSnapshot(BaseModel):
    """Point-in-time performance metrics for regression comparison."""

    bot_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe_rolling_7d: float = 0.0
    max_drawdown_pct: float = 0.0


class DeploymentRecord(BaseModel):
    """Tracks a single parameter change deployment through its lifecycle."""

    deployment_id: str
    approval_request_id: str
    suggestion_id: Optional[str] = None
    pr_url: str
    pr_number: int = 0
    bot_id: str
    param_changes: list[dict[str, Any]] = []  # [{param_name, old_value, new_value}]
    status: DeploymentStatus = DeploymentStatus.PENDING_MERGE
    merge_time: Optional[datetime] = None
    deploy_detected_time: Optional[datetime] = None
    monitoring_window_hours: int = 24
    monitoring_end_time: Optional[datetime] = None
    pre_deploy_metrics: Optional[DeploymentMetricsSnapshot] = None
    post_deploy_snapshots: list[DeploymentMetricsSnapshot] = []
    regression_detected: bool = False
    regression_details: str = ""
    rollback_pr_url: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Lineage / attribution fields (populated by writers; consumed by causal evaluator)
    affected_population: list[str] = Field(default_factory=list)  # event_id refs
    variant_id: Optional[str] = None
    experiment_id: Optional[str] = None
    parameter_set_id: Optional[str] = None
    strategy_version: Optional[str] = None
    config_version: Optional[str] = None
    code_sha: Optional[str] = None

    @property
    def post_deploy_metrics(self) -> DeploymentMetricsSnapshot | None:
        """Backward-compat: return latest post-deploy snapshot."""
        return self.post_deploy_snapshots[-1] if self.post_deploy_snapshots else None
