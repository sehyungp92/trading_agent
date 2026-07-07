"""Scheduler helpers and tracked job definitions for periodic orchestrator work."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore

logger = logging.getLogger(__name__)

TrackedScheduledFn = Callable[[datetime | None], Awaitable[None]]
LegacyScheduledFn = Callable[[], Awaitable[None]]

_WEEKDAY_INDEX = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


class ScheduledJobClass(str, Enum):
    STATEFUL = "stateful"
    COALESCED = "coalesced"
    INTERVAL = "interval"


@dataclass
class ScheduledJobSpec:
    name: str
    job_key: str
    trigger: str
    job_class: ScheduledJobClass
    execute: TrackedScheduledFn
    contract_id: str = ""
    scope_key: str = "global"
    hour: int | None = None
    minute: int | None = None
    day_of_week: str | None = None
    day: int | None = None
    seconds: int | None = None
    misfire_grace_time: int | None = None
    coalesce: bool | None = None
    catchup_limit: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # When True, spec.execute() only enqueues a trigger; the downstream
    # handler is responsible for marking the final scheduled-run result with
    # the same scheduled_for. The scheduler marks the run as 'enqueued'
    # until the handler signals success/failure. See P1-3.
    awaits_completion: bool = False

    @property
    def is_cron(self) -> bool:
        return self.trigger == "cron"

    @property
    def is_interval(self) -> bool:
        return self.trigger == "interval"

    @property
    def is_catchup_eligible(self) -> bool:
        return self.is_cron and self.job_class != ScheduledJobClass.INTERVAL and self.catchup_limit > 0


@dataclass
class SchedulerConfig:
    monitoring_interval_minutes: int = 60
    worker_interval_seconds: int = 60
    relay_poll_interval_seconds: int = 300
    daily_analysis_hour: int = 6
    daily_analysis_minute: int = 0
    weekly_analysis_day_of_week: str = "sun"
    weekly_analysis_hour: int = 8
    weekly_analysis_minute: int = 0
    stale_error_sweep_interval_seconds: int = 600
    stale_event_recovery_interval_seconds: int = 900
    morning_scan_hour: int = 7
    morning_scan_minute: int = 0
    evening_report_hour: int = 22
    evening_report_minute: int = 0
    outcome_measurement_day_of_week: str = "sun"
    outcome_measurement_hour: int = 10
    outcome_measurement_minute: int = 0
    memory_consolidation_day_of_week: str = "sun"
    memory_consolidation_hour: int = 9
    memory_consolidation_minute: int = 0
    transfer_outcome_day_of_week: str = "sun"
    transfer_outcome_hour: int = 10
    transfer_outcome_minute: int = 30
    approval_expiry_hour: int = 0
    approval_expiry_minute: int = 0
    pr_review_check_interval_seconds: int = 3600
    deployment_check_interval_seconds: int = 1800
    threshold_learning_day_of_week: str = "sun"
    threshold_learning_hour: int = 9
    threshold_learning_minute: int = 30
    experiment_check_interval_seconds: int = 21600
    reliability_verification_hour: int = 6
    reliability_verification_minute: int = 30
    discovery_day_of_week: str = "sat"
    discovery_hour: int = 3
    discovery_minute: int = 0
    learning_cycle_day_of_week: str = "sun"
    learning_cycle_hour: int = 11
    learning_cycle_minute: int = 0
    lineage_audit_hour: int = 6
    lineage_audit_minute: int = 30
    market_data_sync_day_of_month: int = 1
    market_data_sync_hour: int = 1
    market_data_sync_minute: int = 0
    monthly_validation_day_of_month: int = 2
    monthly_validation_hour: int = 3
    monthly_validation_minute: int = 0


def create_scheduler_jobs(
    config: SchedulerConfig,
    worker_fn: LegacyScheduledFn,
    monitoring_fn: LegacyScheduledFn,
    relay_fn: LegacyScheduledFn,
    daily_analysis_fn: LegacyScheduledFn | None = None,
    weekly_analysis_fn: LegacyScheduledFn | None = None,
    stale_error_sweep_fn: LegacyScheduledFn | None = None,
    stale_event_recovery_fn: LegacyScheduledFn | None = None,
    morning_scan_fn: LegacyScheduledFn | None = None,
    evening_report_fn: LegacyScheduledFn | None = None,
    outcome_measurement_fn: LegacyScheduledFn | None = None,
    memory_consolidation_fn: LegacyScheduledFn | None = None,
    transfer_outcome_fn: LegacyScheduledFn | None = None,
    approval_expiry_fn: LegacyScheduledFn | None = None,
    pr_review_check_fn: LegacyScheduledFn | None = None,
    deployment_check_fn: LegacyScheduledFn | None = None,
    threshold_learning_fn: LegacyScheduledFn | None = None,
    experiment_check_fn: LegacyScheduledFn | None = None,
    reliability_verification_fn: LegacyScheduledFn | None = None,
    discovery_fn: LegacyScheduledFn | None = None,
    learning_cycle_fn: LegacyScheduledFn | None = None,
    lineage_audit_fn: LegacyScheduledFn | None = None,
    market_data_sync_fn: LegacyScheduledFn | None = None,
    monthly_validation_fn: LegacyScheduledFn | None = None,
    daily_analysis_fns: list[dict] | None = None,
    morning_scan_fns: list[dict] | None = None,
    evening_report_fns: list[dict] | None = None,
    monthly_validation_fns: list[dict] | None = None,
) -> list[dict]:
    """Build legacy APScheduler-style job definitions."""
    jobs = [
        {
            "name": "worker",
            "func": worker_fn,
            "trigger": "interval",
            "seconds": config.worker_interval_seconds,
        },
        {
            "name": "monitoring",
            "func": monitoring_fn,
            "trigger": "interval",
            "seconds": config.monitoring_interval_minutes * 60,
        },
        {
            "name": "relay_poll",
            "func": relay_fn,
            "trigger": "interval",
            "seconds": config.relay_poll_interval_seconds,
        },
    ]

    if daily_analysis_fns:
        for i, trigger_def in enumerate(daily_analysis_fns):
            suffix = trigger_def.get("name_suffix", str(i))
            jobs.append({
                "name": f"daily_analysis_{suffix}",
                "func": trigger_def["fn"],
                "trigger": "cron",
                "hour": trigger_def["hour"],
                "minute": trigger_def["minute"],
                "misfire_grace_time": trigger_def.get("misfire_grace_time", 43200),
                "coalesce": trigger_def.get("coalesce", True),
            })
    elif daily_analysis_fn is not None:
        jobs.append({
            "name": "daily_analysis",
            "func": daily_analysis_fn,
            "trigger": "cron",
            "hour": config.daily_analysis_hour,
            "minute": config.daily_analysis_minute,
            "misfire_grace_time": 43200,
            "coalesce": True,
        })

    if weekly_analysis_fn is not None:
        jobs.append({
            "name": "weekly_analysis",
            "func": weekly_analysis_fn,
            "trigger": "cron",
            "day_of_week": config.weekly_analysis_day_of_week,
            "hour": config.weekly_analysis_hour,
            "minute": config.weekly_analysis_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if stale_error_sweep_fn is not None:
        jobs.append({
            "name": "stale_error_sweep",
            "func": stale_error_sweep_fn,
            "trigger": "interval",
            "seconds": config.stale_error_sweep_interval_seconds,
        })

    if stale_event_recovery_fn is not None:
        jobs.append({
            "name": "stale_event_recovery",
            "func": stale_event_recovery_fn,
            "trigger": "interval",
            "seconds": config.stale_event_recovery_interval_seconds,
        })

    if morning_scan_fns:
        for i, trigger_def in enumerate(morning_scan_fns):
            suffix = trigger_def.get("name_suffix", str(i))
            jobs.append({
                "name": f"morning_scan_{suffix}",
                "func": trigger_def["fn"],
                "trigger": "cron",
                "hour": trigger_def["hour"],
                "minute": trigger_def["minute"],
                "misfire_grace_time": trigger_def.get("misfire_grace_time", 14400),
                "coalesce": trigger_def.get("coalesce", True),
            })
    elif morning_scan_fn is not None:
        jobs.append({
            "name": "morning_scan",
            "func": morning_scan_fn,
            "trigger": "cron",
            "hour": config.morning_scan_hour,
            "minute": config.morning_scan_minute,
            "misfire_grace_time": 14400,
            "coalesce": True,
        })

    if evening_report_fns:
        for i, trigger_def in enumerate(evening_report_fns):
            suffix = trigger_def.get("name_suffix", str(i))
            jobs.append({
                "name": f"evening_report_{suffix}",
                "func": trigger_def["fn"],
                "trigger": "cron",
                "hour": trigger_def["hour"],
                "minute": trigger_def["minute"],
                "misfire_grace_time": trigger_def.get("misfire_grace_time", 14400),
                "coalesce": trigger_def.get("coalesce", True),
            })
    elif evening_report_fn is not None:
        jobs.append({
            "name": "evening_report",
            "func": evening_report_fn,
            "trigger": "cron",
            "hour": config.evening_report_hour,
            "minute": config.evening_report_minute,
            "misfire_grace_time": 14400,
            "coalesce": True,
        })

    if outcome_measurement_fn is not None:
        jobs.append({
            "name": "outcome_measurement",
            "func": outcome_measurement_fn,
            "trigger": "cron",
            "day_of_week": config.outcome_measurement_day_of_week,
            "hour": config.outcome_measurement_hour,
            "minute": config.outcome_measurement_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if memory_consolidation_fn is not None:
        jobs.append({
            "name": "memory_consolidation",
            "func": memory_consolidation_fn,
            "trigger": "cron",
            "day_of_week": config.memory_consolidation_day_of_week,
            "hour": config.memory_consolidation_hour,
            "minute": config.memory_consolidation_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if transfer_outcome_fn is not None:
        jobs.append({
            "name": "transfer_outcome_measurement",
            "func": transfer_outcome_fn,
            "trigger": "cron",
            "day_of_week": config.transfer_outcome_day_of_week,
            "hour": config.transfer_outcome_hour,
            "minute": config.transfer_outcome_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if reliability_verification_fn is not None:
        jobs.append({
            "name": "reliability_verification",
            "func": reliability_verification_fn,
            "trigger": "cron",
            "hour": config.reliability_verification_hour,
            "minute": config.reliability_verification_minute,
            "misfire_grace_time": 43200,
            "coalesce": True,
        })

    if approval_expiry_fn is not None:
        jobs.append({
            "name": "approval_expiry",
            "func": approval_expiry_fn,
            "trigger": "cron",
            "hour": config.approval_expiry_hour,
            "minute": config.approval_expiry_minute,
            "misfire_grace_time": 14400,
            "coalesce": True,
        })

    if pr_review_check_fn is not None:
        jobs.append({
            "name": "pr_review_check",
            "func": pr_review_check_fn,
            "trigger": "interval",
            "seconds": config.pr_review_check_interval_seconds,
        })

    if deployment_check_fn is not None:
        jobs.append({
            "name": "deployment_check",
            "func": deployment_check_fn,
            "trigger": "interval",
            "seconds": config.deployment_check_interval_seconds,
        })

    if threshold_learning_fn is not None:
        jobs.append({
            "name": "threshold_learning",
            "func": threshold_learning_fn,
            "trigger": "cron",
            "day_of_week": config.threshold_learning_day_of_week,
            "hour": config.threshold_learning_hour,
            "minute": config.threshold_learning_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if experiment_check_fn is not None:
        jobs.append({
            "name": "experiment_check",
            "func": experiment_check_fn,
            "trigger": "interval",
            "seconds": config.experiment_check_interval_seconds,
        })

    if discovery_fn is not None:
        jobs.append({
            "name": "discovery_analysis",
            "func": discovery_fn,
            "trigger": "cron",
            "day_of_week": config.discovery_day_of_week,
            "hour": config.discovery_hour,
            "minute": config.discovery_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if learning_cycle_fn is not None:
        jobs.append({
            "name": "learning_cycle",
            "func": learning_cycle_fn,
            "trigger": "cron",
            "day_of_week": config.learning_cycle_day_of_week,
            "hour": config.learning_cycle_hour,
            "minute": config.learning_cycle_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if lineage_audit_fn is not None:
        jobs.append({
            "name": "lineage_audit",
            "func": lineage_audit_fn,
            "trigger": "cron",
            "hour": config.lineage_audit_hour,
            "minute": config.lineage_audit_minute,
            "misfire_grace_time": 43200,
            "coalesce": True,
        })

    if market_data_sync_fn is not None:
        jobs.append({
            "name": "market_data_sync",
            "func": market_data_sync_fn,
            "trigger": "cron",
            "day": config.market_data_sync_day_of_month,
            "hour": config.market_data_sync_hour,
            "minute": config.market_data_sync_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    if monthly_validation_fns:
        for i, trigger_def in enumerate(monthly_validation_fns):
            suffix = trigger_def.get("name_suffix", str(i))
            jobs.append({
                "name": f"monthly_validation_{suffix}",
                "func": trigger_def["fn"],
                "trigger": "cron",
                "day": trigger_def.get("day", config.monthly_validation_day_of_month),
                "hour": trigger_def.get("hour", config.monthly_validation_hour),
                "minute": trigger_def.get("minute", config.monthly_validation_minute),
                "misfire_grace_time": trigger_def.get("misfire_grace_time", 172800),
                "coalesce": trigger_def.get("coalesce", True),
            })
    elif monthly_validation_fn is not None:
        jobs.append({
            "name": "monthly_validation",
            "func": monthly_validation_fn,
            "trigger": "cron",
            "day": config.monthly_validation_day_of_month,
            "hour": config.monthly_validation_hour,
            "minute": config.monthly_validation_minute,
            "misfire_grace_time": 172800,
            "coalesce": True,
        })

    return jobs


def build_scheduled_job_specs(
    config: SchedulerConfig,
    worker_fn: TrackedScheduledFn,
    monitoring_fn: TrackedScheduledFn,
    relay_fn: TrackedScheduledFn,
    daily_analysis_fn: TrackedScheduledFn | None = None,
    weekly_analysis_fn: TrackedScheduledFn | None = None,
    stale_error_sweep_fn: TrackedScheduledFn | None = None,
    stale_event_recovery_fn: TrackedScheduledFn | None = None,
    morning_scan_fn: TrackedScheduledFn | None = None,
    evening_report_fn: TrackedScheduledFn | None = None,
    outcome_measurement_fn: TrackedScheduledFn | None = None,
    memory_consolidation_fn: TrackedScheduledFn | None = None,
    transfer_outcome_fn: TrackedScheduledFn | None = None,
    approval_expiry_fn: TrackedScheduledFn | None = None,
    pr_review_check_fn: TrackedScheduledFn | None = None,
    deployment_check_fn: TrackedScheduledFn | None = None,
    threshold_learning_fn: TrackedScheduledFn | None = None,
    experiment_check_fn: TrackedScheduledFn | None = None,
    reliability_verification_fn: TrackedScheduledFn | None = None,
    discovery_fn: TrackedScheduledFn | None = None,
    learning_cycle_fn: TrackedScheduledFn | None = None,
    lineage_audit_fn: TrackedScheduledFn | None = None,
    market_data_sync_fn: TrackedScheduledFn | None = None,
    monthly_validation_fn: TrackedScheduledFn | None = None,
    daily_analysis_fns: list[dict] | None = None,
    morning_scan_fns: list[dict] | None = None,
    evening_report_fns: list[dict] | None = None,
    monthly_validation_fns: list[dict] | None = None,
) -> list[ScheduledJobSpec]:
    """Build the tracked job model used by APScheduler and startup catch-up."""
    specs = [
        ScheduledJobSpec(
            name="worker",
            job_key="worker",
            trigger="interval",
            job_class=ScheduledJobClass.INTERVAL,
            execute=worker_fn,
            seconds=config.worker_interval_seconds,
        ),
        ScheduledJobSpec(
            name="monitoring",
            job_key="monitoring",
            trigger="interval",
            job_class=ScheduledJobClass.INTERVAL,
            execute=monitoring_fn,
            seconds=config.monitoring_interval_minutes * 60,
        ),
        ScheduledJobSpec(
            name="relay_poll",
            job_key="relay_poll",
            trigger="interval",
            job_class=ScheduledJobClass.INTERVAL,
            execute=relay_fn,
            seconds=config.relay_poll_interval_seconds,
        ),
    ]

    _append_cron_specs(
        specs,
        base_name="daily_analysis",
        job_key="daily_analysis",
        job_class=ScheduledJobClass.STATEFUL,
        catchup_limit=7,
        default_fn=daily_analysis_fn,
        default_hour=config.daily_analysis_hour,
        default_minute=config.daily_analysis_minute,
        default_misfire_grace_time=43200,
        default_coalesce=True,
        variant_fns=daily_analysis_fns,
        awaits_completion=True,
    )
    _append_cron_specs(
        specs,
        base_name="weekly_analysis",
        job_key="weekly_summary",
        job_class=ScheduledJobClass.STATEFUL,
        catchup_limit=4,
        default_fn=weekly_analysis_fn,
        default_day_of_week=config.weekly_analysis_day_of_week,
        default_hour=config.weekly_analysis_hour,
        default_minute=config.weekly_analysis_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
        awaits_completion=True,
    )
    _append_interval_spec(
        specs,
        name="stale_error_sweep",
        job_key="stale_error_sweep",
        seconds=config.stale_error_sweep_interval_seconds,
        execute=stale_error_sweep_fn,
    )
    _append_interval_spec(
        specs,
        name="stale_event_recovery",
        job_key="stale_event_recovery",
        seconds=config.stale_event_recovery_interval_seconds,
        execute=stale_event_recovery_fn,
    )

    _append_cron_specs(
        specs,
        base_name="morning_scan",
        job_key="morning_scan",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=morning_scan_fn,
        default_hour=config.morning_scan_hour,
        default_minute=config.morning_scan_minute,
        default_misfire_grace_time=14400,
        default_coalesce=True,
        variant_fns=morning_scan_fns,
    )
    _append_cron_specs(
        specs,
        base_name="evening_report",
        job_key="evening_report",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=evening_report_fn,
        default_hour=config.evening_report_hour,
        default_minute=config.evening_report_minute,
        default_misfire_grace_time=14400,
        default_coalesce=True,
        variant_fns=evening_report_fns,
    )
    _append_cron_specs(
        specs,
        base_name="outcome_measurement",
        job_key="outcome_measurement",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=outcome_measurement_fn,
        default_day_of_week=config.outcome_measurement_day_of_week,
        default_hour=config.outcome_measurement_hour,
        default_minute=config.outcome_measurement_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="memory_consolidation",
        job_key="memory_consolidation",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=memory_consolidation_fn,
        default_day_of_week=config.memory_consolidation_day_of_week,
        default_hour=config.memory_consolidation_hour,
        default_minute=config.memory_consolidation_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="transfer_outcome_measurement",
        job_key="transfer_outcome_measurement",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=transfer_outcome_fn,
        default_day_of_week=config.transfer_outcome_day_of_week,
        default_hour=config.transfer_outcome_hour,
        default_minute=config.transfer_outcome_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="approval_expiry",
        job_key="approval_expiry",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=approval_expiry_fn,
        default_hour=config.approval_expiry_hour,
        default_minute=config.approval_expiry_minute,
        default_misfire_grace_time=14400,
        default_coalesce=True,
    )

    _append_interval_spec(
        specs,
        name="pr_review_check",
        job_key="pr_review_check",
        seconds=config.pr_review_check_interval_seconds,
        execute=pr_review_check_fn,
    )
    _append_interval_spec(
        specs,
        name="deployment_check",
        job_key="deployment_check",
        seconds=config.deployment_check_interval_seconds,
        execute=deployment_check_fn,
    )
    _append_cron_specs(
        specs,
        base_name="threshold_learning",
        job_key="threshold_learning",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=threshold_learning_fn,
        default_day_of_week=config.threshold_learning_day_of_week,
        default_hour=config.threshold_learning_hour,
        default_minute=config.threshold_learning_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_interval_spec(
        specs,
        name="experiment_check",
        job_key="experiment_check",
        seconds=config.experiment_check_interval_seconds,
        execute=experiment_check_fn,
    )
    _append_cron_specs(
        specs,
        base_name="reliability_verification",
        job_key="reliability_verification",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=reliability_verification_fn,
        default_hour=config.reliability_verification_hour,
        default_minute=config.reliability_verification_minute,
        default_misfire_grace_time=43200,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="discovery_analysis",
        job_key="discovery_analysis",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=discovery_fn,
        default_day_of_week=config.discovery_day_of_week,
        default_hour=config.discovery_hour,
        default_minute=config.discovery_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="learning_cycle",
        job_key="learning_cycle",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=learning_cycle_fn,
        default_day_of_week=config.learning_cycle_day_of_week,
        default_hour=config.learning_cycle_hour,
        default_minute=config.learning_cycle_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="lineage_audit",
        job_key="lineage_audit",
        job_class=ScheduledJobClass.COALESCED,
        catchup_limit=1,
        default_fn=lineage_audit_fn,
        default_hour=config.lineage_audit_hour,
        default_minute=config.lineage_audit_minute,
        default_misfire_grace_time=43200,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="market_data_sync",
        job_key="market_data_sync",
        job_class=ScheduledJobClass.STATEFUL,
        catchup_limit=1,
        default_fn=market_data_sync_fn,
        default_day=config.market_data_sync_day_of_month,
        default_hour=config.market_data_sync_hour,
        default_minute=config.market_data_sync_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
    )
    _append_cron_specs(
        specs,
        base_name="monthly_validation",
        job_key="monthly_validation",
        job_class=ScheduledJobClass.STATEFUL,
        catchup_limit=1,
        default_fn=monthly_validation_fn,
        default_day=config.monthly_validation_day_of_month,
        default_hour=config.monthly_validation_hour,
        default_minute=config.monthly_validation_minute,
        default_misfire_grace_time=172800,
        default_coalesce=True,
        variant_fns=monthly_validation_fns,
        awaits_completion=True,
    )

    return specs


def latest_cron_occurrence(spec: ScheduledJobSpec, now: datetime) -> datetime | None:
    if not spec.is_cron or spec.hour is None or spec.minute is None:
        return None

    now = _ensure_utc(now)
    if spec.day is not None:
        candidate = _monthly_candidate(now.year, now.month, spec.day, spec.hour, spec.minute)
        if candidate > now:
            year, month = _previous_month(now.year, now.month)
            candidate = _monthly_candidate(year, month, spec.day, spec.hour, spec.minute)
        return candidate

    if spec.day_of_week is None:
        candidate = datetime(
            now.year,
            now.month,
            now.day,
            spec.hour,
            spec.minute,
            tzinfo=timezone.utc,
        )
        if candidate > now:
            candidate -= timedelta(days=1)
        return candidate

    target_weekday = _weekday_value(spec.day_of_week)
    days_back = (now.weekday() - target_weekday) % 7
    date = (now - timedelta(days=days_back)).date()
    candidate = datetime(
        date.year,
        date.month,
        date.day,
        spec.hour,
        spec.minute,
        tzinfo=timezone.utc,
    )
    if candidate > now:
        candidate -= timedelta(days=7)
    return candidate


def previous_cron_occurrence(spec: ScheduledJobSpec, occurrence: datetime) -> datetime | None:
    if not spec.is_cron:
        return None
    if spec.day is not None:
        year, month = _previous_month(occurrence.year, occurrence.month)
        return _monthly_candidate(year, month, spec.day, spec.hour or 0, spec.minute or 0)
    delta = timedelta(days=7 if spec.day_of_week else 1)
    return _ensure_utc(occurrence) - delta


def recent_cron_occurrences(
    spec: ScheduledJobSpec,
    now: datetime,
    limit: int,
) -> list[datetime]:
    if limit <= 0:
        return []
    occurrence = latest_cron_occurrence(spec, now)
    if occurrence is None:
        return []
    results: list[datetime] = []
    while occurrence is not None and len(results) < limit:
        results.append(occurrence)
        occurrence = previous_cron_occurrence(spec, occurrence)
    results.reverse()
    return results


def job_specs_to_scheduler_jobs(
    specs: list[ScheduledJobSpec],
    runner: ScheduledJobRunner,
) -> list[dict]:
    jobs: list[dict] = []
    for spec in specs:
        job = {
            "name": spec.name,
            "func": runner.build_scheduler_callable(spec),
            "trigger": spec.trigger,
        }
        if spec.seconds is not None:
            job["seconds"] = spec.seconds
        if spec.day_of_week is not None:
            job["day_of_week"] = spec.day_of_week
        if spec.day is not None:
            job["day"] = spec.day
        if spec.hour is not None:
            job["hour"] = spec.hour
        if spec.minute is not None:
            job["minute"] = spec.minute
        if spec.misfire_grace_time is not None:
            job["misfire_grace_time"] = spec.misfire_grace_time
        if spec.coalesce is not None:
            job["coalesce"] = spec.coalesce
        jobs.append(job)
    return jobs


class ScheduledJobRunner:
    """Runs tracked jobs and records cron execution status."""

    def __init__(self, run_store: ScheduledRunStore) -> None:
        self._run_store = run_store

    def build_scheduler_callable(self, spec: ScheduledJobSpec) -> Callable[[], Awaitable[bool]]:
        async def _run() -> bool:
            scheduled_for = None
            if spec.is_cron:
                scheduled_for = latest_cron_occurrence(spec, datetime.now(timezone.utc))
            return await self.run(spec, scheduled_for=scheduled_for)

        return _run

    async def run(
        self,
        spec: ScheduledJobSpec,
        scheduled_for: datetime | None = None,
    ) -> bool:
        if not spec.is_cron or spec.job_class == ScheduledJobClass.INTERVAL:
            await spec.execute(None)
            return True

        occurrence = _ensure_utc(scheduled_for or datetime.now(timezone.utc))
        if await self._run_store.is_completed(spec.job_key, spec.scope_key, occurrence):
            logger.info(
                "Skipping already-completed scheduled job %s (%s @ %s)",
                spec.job_key,
                spec.scope_key,
                occurrence.isoformat(),
            )
            return False

        await self._run_store.mark_started(spec.job_key, spec.scope_key, occurrence)
        try:
            await spec.execute(occurrence)
        except Exception as exc:
            await self._run_store.mark_failed(
                spec.job_key,
                spec.scope_key,
                occurrence,
                error=str(exc),
            )
            raise

        if spec.awaits_completion:
            # Trigger is enqueued; the handler will mark the final result.
            await self._run_store.mark_enqueued(spec.job_key, spec.scope_key, occurrence)
        else:
            await self._run_store.mark_completed(spec.job_key, spec.scope_key, occurrence)
        return True


def _append_interval_spec(
    specs: list[ScheduledJobSpec],
    *,
    name: str,
    job_key: str,
    seconds: int,
    execute: TrackedScheduledFn | None,
) -> None:
    if execute is None:
        return
    specs.append(ScheduledJobSpec(
        name=name,
        job_key=job_key,
        trigger="interval",
        job_class=ScheduledJobClass.INTERVAL,
        execute=execute,
        seconds=seconds,
    ))


def _append_cron_specs(
    specs: list[ScheduledJobSpec],
    *,
    base_name: str,
    job_key: str,
    job_class: ScheduledJobClass,
    catchup_limit: int,
    default_fn: TrackedScheduledFn | None = None,
    default_hour: int | None = None,
    default_minute: int | None = None,
    default_day_of_week: str | None = None,
    default_day: int | None = None,
    default_misfire_grace_time: int | None = None,
    default_coalesce: bool | None = None,
    variant_fns: list[dict] | None = None,
    awaits_completion: bool = False,
) -> None:
    if variant_fns:
        for i, trigger_def in enumerate(variant_fns):
            suffix = trigger_def.get("name_suffix", str(i))
            specs.append(ScheduledJobSpec(
                name=f"{base_name}_{suffix}",
                job_key=job_key,
                trigger="cron",
                job_class=job_class,
                execute=trigger_def["fn"],
                contract_id=trigger_def.get("contract_id", job_key),
                scope_key=trigger_def.get("scope_key", suffix),
                day_of_week=trigger_def.get("day_of_week", default_day_of_week),
                day=trigger_def.get("day", default_day),
                hour=trigger_def.get("hour", default_hour),
                minute=trigger_def.get("minute", default_minute),
                misfire_grace_time=trigger_def.get(
                    "misfire_grace_time",
                    default_misfire_grace_time,
                ),
                coalesce=trigger_def.get("coalesce", default_coalesce),
                catchup_limit=trigger_def.get("catchup_limit", catchup_limit),
                metadata=dict(trigger_def.get("metadata", {})),
                awaits_completion=trigger_def.get("awaits_completion", awaits_completion),
            ))
        return

    if default_fn is None:
        return

    specs.append(ScheduledJobSpec(
        name=base_name,
        job_key=job_key,
        trigger="cron",
        job_class=job_class,
        execute=default_fn,
        contract_id=job_key,
        day_of_week=default_day_of_week,
        day=default_day,
        hour=default_hour,
        minute=default_minute,
        misfire_grace_time=default_misfire_grace_time,
        coalesce=default_coalesce,
        catchup_limit=catchup_limit,
        awaits_completion=awaits_completion,
    ))


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _weekday_value(raw: str) -> int:
    key = raw.strip().lower()[:3]
    if key not in _WEEKDAY_INDEX:
        raise ValueError(f"Unsupported day_of_week value: {raw!r}")
    return _WEEKDAY_INDEX[key]


def _previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _monthly_candidate(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    import calendar

    clamped_day = min(max(1, day), calendar.monthrange(year, month)[1])
    return datetime(year, month, clamped_day, hour, minute, tzinfo=timezone.utc)
