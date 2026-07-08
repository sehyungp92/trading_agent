"""Cross-cutting handler adapters and loop construction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.schemas.notifications import (
    NotificationPayload,
    NotificationPriority,
)

if TYPE_CHECKING:
    from trading_assistant.schemas.reliability_learning import BugClass

logger = logging.getLogger(__name__)


class CoreHandlerSupport:
    """Cross-cutting handler adapters and loop construction."""

    async def _signal_scheduled_result(
        self,
        action: Action,
        *,
        success: bool,
        error: str = "",
    ) -> None:
        """Mark the originating cron run with the actual handler result."""
        if self._scheduled_run_store is None:
            return
        details = action.details or {}
        marker = details.get("__scheduled_run__")
        if not isinstance(marker, dict):
            return
        try:
            scheduled_for = datetime.fromisoformat(marker["scheduled_for"])
        except (KeyError, ValueError, TypeError):
            return
        try:
            if success:
                await self._scheduled_run_store.mark_completed(
                    marker["job_key"],
                    marker["scope_key"],
                    scheduled_for,
                )
            else:
                await self._scheduled_run_store.mark_failed(
                    marker["job_key"],
                    marker["scope_key"],
                    scheduled_for,
                    error=error or "Scheduled handler did not complete successfully",
                )
        except Exception:
            logger.warning("Failed to mark scheduled run result: %s", marker)

    async def handle_daily_analysis(self, action: Action) -> None:
        await self._daily_analysis_loop.handle(action)

    def set_daily_analysis_loop(self, loop: object) -> None:
        self._daily_analysis_loop = loop

    @property
    def daily_analysis_loop(self) -> object:
        return self._daily_analysis_loop

    def _build_daily_analysis_loop(self):
        from trading_assistant.orchestrator.loops.daily_analysis import (
            DailyAnalysisDependencies,
            DailyAnalysisLoop,
        )

        return DailyAnalysisLoop(
            DailyAnalysisDependencies(
                agent_runner=self._agent_runner,
                event_stream=self._event_stream,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                runs_dir=self._runs_dir,
                bots=self._bots,
                bot_configs=self._bot_configs,
                strategy_registry=self._strategy_registry,
                run_index=self._run_index,
                worker=self._worker,
                brain=self._brain,
                record_run=self._record_run,
                rebuild_daily_curated_from_raw=self._rebuild_daily_curated_from_raw,
                count_daily_trades=self._count_daily_trades,
                check_strategy_registry_drift=self._check_strategy_registry_drift,
                validate_and_annotate=self._validate_and_annotate,
                persist_validator_notes=self._persist_validator_notes,
                refresh_run_index_entry=self._refresh_run_index_entry,
                record_agent_suggestions=self._record_agent_suggestions,
                record_learning_card_feedback_targeted=self._record_learning_card_feedback_targeted,
                run_autonomous_pipeline=self._run_autonomous_pipeline,
                record_predictions=self._record_predictions,
                write_run_report=self._write_run_report,
                notify=self._notify,
                signal_scheduled_result=self._signal_scheduled_result,
            )
        )

    async def handle_monthly_validation(self, action: Action) -> None:
        await self._monthly_validation_loop.handle(action)

    def set_monthly_validation_loop(self, loop: object) -> None:
        self._monthly_validation_loop = loop

    def _build_monthly_validation_loop(self):
        from trading_assistant.orchestrator.loops import MonthlyValidationLoop
        from trading_assistant.orchestrator.loops.monthly_services import (
            MonthlyRunRecorder,
            ScheduledMonthlyProjection,
        )
        from trading_assistant.orchestrator.loops.monthly_validation import (
            MonthlyValidationDependencies,
        )

        recorder = MonthlyRunRecorder(
            run_history_path=self._run_history_path,
            runs_dir=self._runs_dir,
        )
        projection = ScheduledMonthlyProjection(
            scheduled_run_store=self._scheduled_run_store,
            memory_dir=self._memory_dir,
        )
        return MonthlyValidationLoop(
            MonthlyValidationDependencies(
                agent_runner=self._agent_runner,
                event_stream=self._event_stream,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                market_data_root=self._market_data_root,
                backtest_repo_path=self._backtest_repo_path,
                backtest_artifact_root=self._backtest_artifact_root,
                strategy_registry=self._strategy_registry,
                proposal_ledger=self._proposal_ledger,
                approval_tracker=self._approval_tracker,
                monthly_validation_mode=self._monthly_validation_mode,
                monthly_approval_scope_allowlist=self._monthly_approval_scope_allowlist,
                monthly_optimizer_sequence_enabled=self._monthly_optimizer_sequence_enabled,
                monthly_backtest_command=list(self._monthly_backtest_command or []),
                monthly_workflow_contract_path=self._monthly_workflow_contract_path,
                monthly_workflow_contract_version=self._monthly_workflow_contract_version,
                monthly_strategy_plugin_contract_path=self._monthly_strategy_plugin_contract_path,
                market_data_required_coverage_ratio=self._market_data_required_coverage_ratio,
                telemetry_required_lineage_ratio=self._telemetry_required_lineage_ratio,
                backtest_command_timeout_seconds=self._backtest_command_timeout_seconds,
                backtest_max_parallel_strategies=self._backtest_max_parallel_strategies,
                record_run=recorder.record_run,
                write_artifact_index=recorder.write_artifact_index,
                signal_scheduled_result=projection.signal_result,
                project_scheduled_results=projection.project_results,
            )
        )

    async def handle_weekly_analysis(self, action: Action) -> None:
        await self._weekly_analysis_loop.handle(action)

    def set_weekly_analysis_loop(self, loop: object) -> None:
        self._weekly_analysis_loop = loop

    @property
    def weekly_analysis_loop(self) -> object:
        return self._weekly_analysis_loop

    def _build_weekly_analysis_loop(self):
        from trading_assistant.orchestrator.loops.weekly_analysis import (
            WeeklyAnalysisDependencies,
            WeeklyAnalysisLoop,
        )

        return WeeklyAnalysisLoop(
            WeeklyAnalysisDependencies(
                agent_runner=self._agent_runner,
                event_stream=self._event_stream,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                runs_dir=self._runs_dir,
                bots=self._bots,
                bot_configs=self._bot_configs,
                strategy_registry=self._strategy_registry,
                run_index=self._run_index,
                threshold_learner=self._threshold_learner,
                suggestion_tracker=self._suggestion_tracker,
                experiment_manager=self._experiment_manager,
                brain=self._brain,
                record_run=self._record_run,
                load_bot_dailies=self._load_bot_dailies,
                load_weekly_strategy_evidence=self._load_weekly_strategy_evidence,
                run_portfolio_detectors=self._run_portfolio_detectors,
                run_weekly_simulations=self._run_weekly_simulations,
                run_allocation_analyses=self._run_allocation_analyses,
                record_suggestions=self._record_suggestions,
                run_autonomous_pipeline=self._run_autonomous_pipeline,
                ledger_write_candidate=self._ledger_write_candidate,
                validate_and_annotate=self._validate_and_annotate,
                persist_validator_notes=self._persist_validator_notes,
                refresh_run_index_entry=self._refresh_run_index_entry,
                record_agent_suggestions=self._record_agent_suggestions,
                record_learning_card_feedback_targeted=self._record_learning_card_feedback_targeted,
                record_portfolio_proposals=self._record_portfolio_proposals,
                record_predictions=self._record_predictions,
                update_hypothesis_lifecycle=self._update_hypothesis_lifecycle,
                extract_and_record_patterns=self._extract_and_record_patterns,
                write_run_report=self._write_run_report,
                notify=self._notify,
                signal_scheduled_result=self._signal_scheduled_result,
            )
        )

    async def handle_alert(self, action: Action) -> None:
        """Dispatch a CRITICAL alert immediately (bypasses quiet hours)."""
        details = action.details or {}
        await self._notify(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title=f"ALERT: {action.bot_id}",
            body=details.get("message", json.dumps(details)),
        )

    async def handle_heartbeat(self, action: Action) -> None:
        """Write heartbeat timestamp for a bot."""
        self._heartbeat_dir.mkdir(parents=True, exist_ok=True)
        hb_path = self._heartbeat_dir / f"{action.bot_id}.heartbeat"
        hb_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    async def handle_notification(self, action: Action) -> None:
        """Build and dispatch a notification from action details."""
        details = action.details or {}
        await self._notify(
            notification_type=details.get("notification_type", "general"),
            priority=NotificationPriority(details.get("priority", "normal")),
            title=details.get("title", ""),
            body=details.get("body", ""),
        )

    def _record_run(
        self, run_id: str, agent_type: str, status: str,
        started_at: str = "", finished_at: str = "", error: str = "",
        duration_ms: int = 0, metadata: dict | None = None,
    ) -> None:
        """Append a run history entry to the JSONL log."""
        try:
            self._run_history_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "run_id": run_id,
                "agent_type": agent_type,
                "handler": agent_type,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "error": error,
            }
            if metadata:
                entry["metadata"] = metadata
            with open(self._run_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Failed to write run history for %s", run_id)

    async def _notify(
        self,
        notification_type: str,
        priority: NotificationPriority,
        title: str,
        body: str,
    ) -> None:
        """Build a NotificationPayload and dispatch it."""
        payload = NotificationPayload(
            notification_type=notification_type,
            priority=priority,
            title=title,
            body=body,
        )
        hour_utc = datetime.now(timezone.utc).hour
        try:
            results = await self._dispatcher.dispatch(payload, self._notification_prefs, hour_utc)
            for r in results:
                if not r.success:
                    logger.warning(
                        "Notification delivery failed on %s: %s",
                        r.channel.value, r.error,
                    )
        except Exception:
            logger.exception("Notification dispatch failed for %s", notification_type)

    def _severity_to_priority(self, severity) -> NotificationPriority:
        """Map BugSeverity to NotificationPriority."""
        from trading_assistant.schemas.bug_triage import BugSeverity

        return {
            BugSeverity.CRITICAL: NotificationPriority.CRITICAL,
            BugSeverity.HIGH: NotificationPriority.HIGH,
            BugSeverity.MEDIUM: NotificationPriority.NORMAL,
            BugSeverity.LOW: NotificationPriority.LOW,
        }.get(severity, NotificationPriority.NORMAL)

    @staticmethod
    def _map_error_to_bug_class(error_category: str) -> "BugClass":
        """Map error category string to BugClass enum."""
        from trading_assistant.schemas.reliability_learning import BugClass

        mapping = {
            "connection": BugClass.CONNECTION,
            "network": BugClass.CONNECTION,
            "timeout": BugClass.CONNECTION,
            "data": BugClass.DATA_INTEGRITY,
            "data_integrity": BugClass.DATA_INTEGRITY,
            "parse": BugClass.DATA_INTEGRITY,
            "timing": BugClass.TIMING,
            "schedule": BugClass.TIMING,
            "config": BugClass.CONFIG,
            "configuration": BugClass.CONFIG,
            "logic": BugClass.LOGIC,
            "assertion": BugClass.LOGIC,
            "dependency": BugClass.DEPENDENCY,
            "import": BugClass.DEPENDENCY,
            "resource": BugClass.RESOURCE,
            "memory": BugClass.RESOURCE,
            "disk": BugClass.RESOURCE,
        }
        cat_lower = error_category.lower()
        for key, cls in mapping.items():
            if key in cat_lower:
                return cls
        return BugClass.UNKNOWN
