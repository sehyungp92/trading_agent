"""Runtime-owned scheduler job wiring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.config import runtime_scheduler_config_from_app_config
from trading_assistant.orchestrator.runtime_experiments import run_experiment_checks
from trading_assistant.orchestrator.runtime_side_jobs import (
    run_evening_report,
    run_lineage_audit,
    run_market_data_sync,
    run_morning_scan,
)
from trading_assistant.orchestrator.scheduler import (
    ScheduledJobRunner,
    ScheduledJobSpec,
    build_scheduled_job_specs,
    job_specs_to_scheduler_jobs,
)
from trading_assistant.schemas.notifications import NotificationPreferences

logger = logging.getLogger(__name__)

ScheduledCallback = Callable[[datetime | None], Awaitable[None]]


@dataclass(frozen=True)
class RuntimeJobCallbacks:
    outcome_measurement: ScheduledCallback
    memory_consolidation: ScheduledCallback
    transfer_outcome: ScheduledCallback
    reliability_verification: ScheduledCallback
    discovery_analysis: ScheduledCallback
    learning_cycle: ScheduledCallback
    lookup_proposal_id: Callable[..., str]
    record_proposal_outcome: Callable[..., None]
    expire_approvals: Callable[[Any, Any, Any, NotificationPreferences], Awaitable[None]]
    reconcile_linked_subagent_tasks: Callable[[], Awaitable[dict[str, int]]]


@dataclass(frozen=True)
class RuntimeSchedulerWiring:
    job_specs: list[ScheduledJobSpec]
    scheduler_jobs: list[Any]
    runner: ScheduledJobRunner


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _scope_token(scope_key: str) -> str:
    return hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:10]


def _build_scheduled_event(
    *,
    job_key: str,
    scope_key: str,
    scheduled_for: datetime,
    event_type: str,
    bot_id: str,
    payload: dict,
) -> dict:
    scheduled_for = scheduled_for.astimezone(timezone.utc).replace(microsecond=0)
    payload = dict(payload)
    payload.setdefault("__scheduled_run__", {
        "job_key": job_key,
        "scope_key": scope_key,
        "scheduled_for": scheduled_for.isoformat(),
    })
    return {
        "event_id": (
            f"{job_key}-{_scope_token(scope_key)}-"
            f"{scheduled_for.strftime('%Y%m%dT%H%M')}"
        ),
        "bot_id": bot_id,
        "event_type": event_type,
        "payload": json.dumps(payload),
        "exchange_timestamp": scheduled_for.isoformat(),
        "received_at": _utc_now().isoformat(),
    }


def _bot_scope_key(bot_ids: list[str]) -> str:
    if not bot_ids:
        return "global"
    return f"bots:{','.join(sorted(bot_ids))}"


def build_runtime_scheduler_wiring(
    runtime: Any,
    callbacks: RuntimeJobCallbacks,
) -> RuntimeSchedulerWiring:
    """Build tracked scheduled-job specs from runtime collaborators."""
    config = runtime.config
    queue = runtime.queue
    worker = runtime.worker
    monitoring_loop = runtime.monitoring_loop
    db_path = runtime.db_path
    curated_dir = runtime.curated_dir
    memory_dir = runtime.memory_dir
    dispatcher = runtime.dispatcher
    notification_prefs = runtime.notification_preferences
    vps_receiver = runtime.vps_receiver
    approval_tracker = runtime.approval_tracker
    telegram_adapter = runtime.telegram_adapter
    pr_builder = runtime.pr_builder
    config_registry = runtime.config_registry
    event_stream = runtime.event_stream
    deployment_monitor = runtime.deployment_monitor
    threshold_learner = runtime.threshold_learner
    scheduled_run_store = runtime.scheduled_run_store

    async def _worker_job(scheduled_for: datetime | None = None) -> None:
        await worker.drain(
            batch_size=config.worker_batch_size,
            time_budget_seconds=config.worker_drain_seconds,
        )

    async def _monitoring_job(scheduled_for: datetime | None = None) -> None:
        await monitoring_loop.run_all()

    async def _relay_job(scheduled_for: datetime | None = None) -> None:
        if vps_receiver is not None:
            await vps_receiver.poll()

    async def _stale_recovery_job(scheduled_for: datetime | None = None) -> None:
        await queue.recover_stale()
        await callbacks.reconcile_linked_subagent_tasks()

    async def _weekly_summary_trigger(scheduled_for: datetime | None = None) -> None:
        run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
        await queue.enqueue(_build_scheduled_event(
            job_key="weekly_summary",
            scope_key="global",
            scheduled_for=run_at,
            event_type="weekly_summary_trigger",
            bot_id="system",
            payload={
                "week_start": (run_at.date() - timedelta(days=6)).isoformat(),
                "week_end": run_at.date().isoformat(),
            },
        ))

    async def _approval_expiry_job(scheduled_for: datetime | None = None) -> None:
        if approval_tracker is not None:
            await callbacks.expire_approvals(
                approval_tracker,
                telegram_adapter,
                dispatcher,
                notification_prefs,
            )

    async def _pr_review_job(scheduled_for: datetime | None = None) -> None:
        if approval_tracker is None or pr_builder is None:
            return
        try:
            for req in approval_tracker.get_approved_with_prs():
                if not req.pr_url:
                    continue
                profile = (
                    config_registry.get_profile(req.bot_id)
                    if config_registry is not None
                    else None
                )
                repo_dir = Path(profile.repo_dir) if profile else db_path
                status = await pr_builder.check_pr_reviews(req.pr_url, repo_dir)
                if status and status.needs_attention:
                    msg = (
                        "\U0001f50d PR Review Needs Attention\n"
                        f"PR: {status.pr_url}\n"
                        f"State: {status.review_state.value}\n"
                        f"Reviewers: {', '.join(status.reviewers) or 'none'}\n"
                    )
                    if status.actionable_comments:
                        msg += f"Comments: {len(status.actionable_comments)}\n"
                    if telegram_adapter is not None:
                        try:
                            await telegram_adapter.send_message(msg)
                        except Exception:
                            logger.warning("Failed to send PR review notification")
                    event_stream.broadcast("pr_review_needs_attention", {
                        "request_id": req.request_id,
                        "pr_url": req.pr_url,
                        "review_state": status.review_state.value,
                    })
        except Exception:
            logger.exception("PR review check failed")

    async def _deployment_check_job(scheduled_for: datetime | None = None) -> None:
        if deployment_monitor is not None:
            await runtime.handlers._check_deployments()

    async def _threshold_learning_job(scheduled_for: datetime | None = None) -> None:
        if threshold_learner is not None:
            await asyncio.to_thread(threshold_learner.learn_thresholds)

    async def _experiment_check_job(scheduled_for: datetime | None = None) -> None:
        await run_experiment_checks(
            experiment_manager=runtime.experiment_manager,
            structural_experiment_tracker=runtime.structural_experiment_tracker,
            event_stream=event_stream,
            memory_dir=memory_dir,
            curated_dir=curated_dir,
            suggestion_tracker=runtime.suggestion_tracker,
            hypothesis_library=runtime.hypothesis_library,
            telegram_adapter=telegram_adapter,
            lookup_proposal_id=callbacks.lookup_proposal_id,
            record_proposal_outcome=callbacks.record_proposal_outcome,
        )

    async def _market_data_sync_job(scheduled_for: datetime | None = None) -> None:
        await run_market_data_sync(
            config=config,
            db_path=db_path,
            event_stream=event_stream,
            scheduled_for=scheduled_for,
        )

    async def _lineage_audit_job(scheduled_for: datetime | None = None) -> None:
        await run_lineage_audit(
            config=config,
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            proposal_ledger=runtime.proposal_ledger,
            scheduled_for=scheduled_for,
        )

    tracked_daily_fns: list[dict] | None = None
    tracked_morning_fns: list[dict] | None = None
    tracked_evening_fns: list[dict] | None = None
    tracked_monthly_validation_fns: list[dict] | None = None
    tracked_daily_fn = None
    tracked_morning_fn = None
    tracked_evening_fn = None

    if config.bot_configs:
        from trading_assistant.orchestrator.tz_utils import (
            bot_trading_date,
            group_bots_by_analysis_time,
            market_close_utc,
        )

        tracked_daily_fns = []
        tracked_morning_fns = []
        tracked_evening_fns = []
        for time_key, bot_list in group_bots_by_analysis_time(config.bot_configs).items():
            trigger_hour, trigger_minute = (int(value) for value in time_key.split(":"))
            scope_key = _bot_scope_key(bot_list)
            suffix = time_key.replace(":", "")
            cfg0 = config.bot_configs[bot_list[0]]
            close_utc = market_close_utc(cfg0.timezone, cfg0.market_close_local)
            morning_utc = close_utc - timedelta(hours=9)
            evening_utc = close_utc + timedelta(hours=1)

            def _make_daily_trigger(bots: list[str], tz_name: str, scope_key: str):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
                    await queue.enqueue(_build_scheduled_event(
                        job_key="daily_analysis",
                        scope_key=scope_key,
                        scheduled_for=run_at,
                        event_type="daily_analysis_trigger",
                        bot_id="system",
                        payload={
                            "bots": bots,
                            "date": bot_trading_date(tz_name, run_at),
                            "run_scope": scope_key,
                        },
                    ))

                return _trigger

            def _make_morning_trigger(bots: list[str]):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    await run_morning_scan(
                        config=config,
                        queue=queue,
                        curated_dir=curated_dir,
                        dispatcher=dispatcher,
                        notification_preferences=notification_prefs,
                        bot_ids=bots,
                        scheduled_for=scheduled_for,
                    )

                return _trigger

            def _make_evening_trigger(bots: list[str]):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    await run_evening_report(
                        config=config,
                        curated_dir=curated_dir,
                        dispatcher=dispatcher,
                        notification_preferences=notification_prefs,
                        bot_ids=bots,
                        scheduled_for=scheduled_for,
                    )

                return _trigger

            tracked_daily_fns.append({
                "fn": _make_daily_trigger(bot_list, cfg0.timezone, scope_key),
                "hour": trigger_hour,
                "minute": trigger_minute,
                "name_suffix": suffix,
                "scope_key": scope_key,
            })
            tracked_morning_fns.append({
                "fn": _make_morning_trigger(bot_list),
                "hour": morning_utc.hour,
                "minute": morning_utc.minute,
                "name_suffix": suffix,
                "scope_key": scope_key,
            })
            tracked_evening_fns.append({
                "fn": _make_evening_trigger(bot_list),
                "hour": evening_utc.hour,
                "minute": evening_utc.minute,
                "name_suffix": suffix,
                "scope_key": scope_key,
            })
    else:
        scope_key = _bot_scope_key(config.bot_ids)

        async def _global_daily_trigger(scheduled_for: datetime | None = None) -> None:
            run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
            await queue.enqueue(_build_scheduled_event(
                job_key="daily_analysis",
                scope_key=scope_key,
                scheduled_for=run_at,
                event_type="daily_analysis_trigger",
                bot_id="system",
                payload={
                    "bots": config.bot_ids,
                    "date": run_at.strftime("%Y-%m-%d"),
                    "run_scope": scope_key,
                },
            ))

        async def _global_morning_trigger(scheduled_for: datetime | None = None) -> None:
            await run_morning_scan(
                config=config,
                queue=queue,
                curated_dir=curated_dir,
                dispatcher=dispatcher,
                notification_preferences=notification_prefs,
                bot_ids=config.bot_ids or None,
                scheduled_for=scheduled_for,
            )

        async def _global_evening_trigger(scheduled_for: datetime | None = None) -> None:
            await run_evening_report(
                config=config,
                curated_dir=curated_dir,
                dispatcher=dispatcher,
                notification_preferences=notification_prefs,
                bot_ids=config.bot_ids or None,
                scheduled_for=scheduled_for,
            )

        tracked_daily_fn = _global_daily_trigger
        tracked_morning_fn = _global_morning_trigger
        tracked_evening_fn = _global_evening_trigger

    if config.monthly_validation_enabled and config.bot_ids:
        tracked_monthly_validation_fns = []
        approval_allowlist = set(config.monthly_approval_scope_allowlist)
        for bot_id in config.bot_ids:
            scope_key = f"bot:{bot_id}"

            def _make_monthly_validation_trigger(bot_id: str, scope_key: str):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
                    approval_scope = config.monthly_approval_scope_map.get(bot_id, "")
                    approval_allowed = approval_scope in approval_allowlist if approval_scope else False
                    approval_evidence_mode = bool(approval_allowed)
                    payload = {
                        "bot_id": bot_id,
                        "run_month": "",
                        "shadow": not (
                            config.monthly_validation_mode == "approval_gated"
                            and approval_allowed
                        ),
                        "optimizer_sequence_enabled": config.monthly_optimizer_sequence_enabled,
                        "backtest_command": config.monthly_backtest_command,
                        "workflow_contract_path": config.monthly_workflow_contract_path,
                        "workflow_contract_version": config.monthly_workflow_contract_version,
                        "approval_evidence_mode": approval_evidence_mode,
                    }
                    if approval_scope:
                        payload["strategy_id"] = approval_scope
                        payload["approval_scope"] = approval_scope
                    if approval_evidence_mode:
                        payload.update(_monthly_approval_evidence_payload(config))
                    await queue.enqueue(_build_scheduled_event(
                        job_key="monthly_validation",
                        scope_key=scope_key,
                        scheduled_for=run_at,
                        event_type="monthly_validation_trigger",
                        bot_id=bot_id,
                        payload=payload,
                    ))

                return _trigger

            tracked_monthly_validation_fns.append({
                "fn": _make_monthly_validation_trigger(bot_id, scope_key),
                "name_suffix": bot_id,
                "scope_key": scope_key,
            })

    runner = ScheduledJobRunner(scheduled_run_store)
    job_specs = build_scheduled_job_specs(
        config=runtime_scheduler_config_from_app_config(config),
        worker_fn=_worker_job,
        monitoring_fn=_monitoring_job,
        relay_fn=_relay_job,
        daily_analysis_fn=tracked_daily_fn,
        daily_analysis_fns=tracked_daily_fns,
        weekly_analysis_fn=_weekly_summary_trigger,
        stale_event_recovery_fn=_stale_recovery_job,
        morning_scan_fn=tracked_morning_fn,
        evening_report_fn=tracked_evening_fn,
        morning_scan_fns=tracked_morning_fns,
        evening_report_fns=tracked_evening_fns,
        outcome_measurement_fn=callbacks.outcome_measurement,
        memory_consolidation_fn=callbacks.memory_consolidation,
        transfer_outcome_fn=callbacks.transfer_outcome,
        approval_expiry_fn=_approval_expiry_job if approval_tracker else None,
        pr_review_check_fn=_pr_review_job if approval_tracker else None,
        deployment_check_fn=_deployment_check_job if deployment_monitor else None,
        threshold_learning_fn=_threshold_learning_job if threshold_learner else None,
        experiment_check_fn=_experiment_check_job,
        reliability_verification_fn=callbacks.reliability_verification,
        discovery_fn=callbacks.discovery_analysis,
        learning_cycle_fn=callbacks.learning_cycle,
        lineage_audit_fn=_lineage_audit_job if config.monthly_validation_enabled else None,
        market_data_sync_fn=_market_data_sync_job if config.monthly_validation_enabled else None,
        monthly_validation_fn=None,
        monthly_validation_fns=tracked_monthly_validation_fns,
    )
    return RuntimeSchedulerWiring(
        job_specs=job_specs,
        scheduler_jobs=job_specs_to_scheduler_jobs(job_specs, runner),
        runner=runner,
    )


def _monthly_approval_evidence_payload(config: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config.monthly_deployment_metadata_install_report_paths:
        payload["deployment_metadata_install_report_paths"] = list(
            config.monthly_deployment_metadata_install_report_paths
        )
    for source, target in (
        ("monthly_operational_evidence_path", "operational_evidence_path"),
        ("monthly_relay_ingest_evidence_path", "relay_ingest_evidence_path"),
        ("monthly_vps_host_id", "vps_host_id"),
        ("monthly_assistant_host_id", "assistant_host_id"),
    ):
        value = str(getattr(config, source, "") or "")
        if value:
            payload[target] = value
    return payload
