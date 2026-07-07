"""Deployment monitoring action support."""

from __future__ import annotations

import logging

from trading_assistant.schemas.notifications import NotificationPriority

logger = logging.getLogger(__name__)


class DeploymentMonitoringActions:
    """Deployment monitoring action support."""

    async def _check_deployments(self) -> None:
        """Periodic deployment monitoring check."""
        if not self._deployment_monitor:
            return
        from trading_assistant.schemas.deployment_monitoring import DeploymentStatus

        for deployment in self._deployment_monitor.get_monitoring():
            try:
                if deployment.status == DeploymentStatus.PENDING_MERGE:
                    merged = await self._deployment_monitor.check_merge_status(
                        deployment.deployment_id,
                    )
                    if (
                        merged
                        and self._suggestion_tracker is not None
                        and deployment.suggestion_id
                    ):
                        self._suggestion_tracker.mark_merged(
                            deployment.suggestion_id,
                            pr_url=deployment.pr_url,
                            deployment_id=deployment.deployment_id,
                        )
                    # Also check for stale pending
                    self._deployment_monitor.check_stale_pending(deployment.deployment_id)
                elif deployment.status == DeploymentStatus.MERGED:
                    # Record pre-deploy metrics if not yet captured
                    if deployment.pre_deploy_metrics is None:
                        snapshot = self._deployment_monitor.collect_metrics_snapshot(deployment.bot_id)
                        if snapshot:
                            self._deployment_monitor.record_pre_deploy_metrics(
                                deployment.deployment_id, snapshot,
                            )
                    # Require heartbeat confirmation before marking DEPLOYED
                    latest_hb = self._get_latest_heartbeat_time(deployment.bot_id)
                    if self._deployment_monitor.is_heartbeat_confirmed(
                        deployment.deployment_id, latest_hb,
                    ):
                        self._deployment_monitor.mark_deployed(
                            deployment.deployment_id,
                            detected_at=latest_hb,
                        )
                        if (
                            self._suggestion_tracker is not None
                            and deployment.suggestion_id
                        ):
                            self._suggestion_tracker.mark_deployed(
                                deployment.suggestion_id,
                                deployment_id=deployment.deployment_id,
                            )
                        try:
                            from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger

                            request = (
                                self._approval_tracker.get_by_id(deployment.approval_request_id)
                                if self._approval_tracker is not None
                                else None
                            )
                            StrategyChangeLedger(self._memory_dir / "findings").record_deployment_writeback(
                                record_id=(
                                    request.strategy_change_record_id
                                    if request is not None else ""
                                ),
                                approval_request_id=deployment.approval_request_id,
                                deployment_id=deployment.deployment_id,
                                pr_url=deployment.pr_url,
                                commit_sha=deployment.code_sha or "",
                                deployed_at=latest_hb,
                                config_version=deployment.config_version or "",
                                strategy_version=deployment.strategy_version or "",
                            )
                        except Exception:
                            logger.warning(
                                "Failed to write deployment lineage for %s",
                                deployment.deployment_id,
                            )
                        logger.info(
                            "Deployment %s confirmed by heartbeat, monitoring started",
                            deployment.deployment_id,
                        )
                    elif self._deployment_monitor.check_merged_timeout(deployment.deployment_id):
                        logger.warning(
                            "Deployment %s timed out waiting for heartbeat - STALE",
                            deployment.deployment_id,
                        )
                        await self._notify(
                            notification_type="alert",
                            priority=NotificationPriority.HIGH,
                            title=f"Deployment Stale - {deployment.bot_id}",
                            body=(
                                f"Deployment {deployment.deployment_id} merged but no heartbeat "
                                f"received within 6 hours. Check bot status."
                            ),
                        )
                elif deployment.status == DeploymentStatus.DEPLOYED:
                    snapshot = self._deployment_monitor.collect_metrics_snapshot(deployment.bot_id)
                    if snapshot:
                        self._deployment_monitor.record_post_deploy_metrics(
                            deployment.deployment_id, snapshot,
                        )
                    if self._deployment_monitor.check_monitoring_window_expired(deployment.deployment_id):
                        logger.info(
                            "Deployment %s monitoring complete - no regression",
                            deployment.deployment_id,
                        )
                    elif self._deployment_monitor.check_regression(deployment.deployment_id):
                        logger.warning(
                            "Regression detected for deployment %s",
                            deployment.deployment_id,
                        )
                        result = await self._deployment_monitor.create_rollback_pr(
                            deployment.deployment_id,
                        )
                        record = self._deployment_monitor.get_by_id(deployment.deployment_id)
                        await self._notify(
                            notification_type="alert",
                            priority=NotificationPriority.CRITICAL,
                            title=f"Regression Detected - {deployment.bot_id}",
                            body=(
                                f"Deployment {deployment.deployment_id} shows regression.\n"
                                f"Details: {record.regression_details if record else 'unknown'}\n"
                                f"Rollback PR: {result.pr_url if result and result.success else 'failed'}"
                            ),
                        )
            except Exception:
                logger.exception("Deployment check failed for %s", deployment.deployment_id)
