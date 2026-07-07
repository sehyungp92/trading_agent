"""Loader and validators for recurring-loop contracts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.scheduler import (
    ScheduledJobClass,
    ScheduledJobSpec,
)
from trading_assistant.paths import package_root
from trading_assistant.schemas.loop_contracts import LoopContract, LoopStatus, parse_loop_contract_markdown


_SKILL_TRIGGER_RE = re.compile(
    r"\|\s*`(?P<skill>[^`]+)`\s*\|[^|]*\|\s*(?P<trigger>[^|]+)\|",
    re.IGNORECASE,
)
_UTC_RE = re.compile(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*UTC", re.IGNORECASE)
_SCHEDULED_DAILY_RE = re.compile(
    r"Scheduled\s+daily\s+at\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*UTC",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LoopContractIssue:
    am_row: str
    path: str
    field: str
    message: str
    remediation: str

    def format(self) -> str:
        location = f"{self.path}:{self.field}" if self.field else self.path
        return (
            f"{self.am_row} {location} - {self.message}\n"
            f"  remediation: {self.remediation}"
        )


class LoopContractStore:
    """Reads loop contracts from ``memory/loops``."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self.memory_dir = memory_dir or package_root() / "memory"
        self.loop_dir = self.memory_dir / "loops"

    def load_all(self) -> dict[str, LoopContract]:
        contracts: dict[str, LoopContract] = {}
        if not self.loop_dir.exists():
            return contracts
        for path in sorted(self.loop_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            contract = parse_loop_contract_markdown(path)
            contracts[contract.loop_id] = contract
        return contracts

    def get(self, loop_id: str) -> LoopContract | None:
        return self.load_all().get(loop_id)

    def get_for_job_key(self, job_key: str) -> LoopContract | None:
        for contract in self.load_all().values():
            if contract.job_key == job_key and contract.status == LoopStatus.ACTIVE:
                return contract
        return None

    def context_for_prompt(self, loop_id_or_job_key: str) -> dict[str, Any]:
        contract = self.get(loop_id_or_job_key) or self.get_for_job_key(loop_id_or_job_key)
        if contract is None:
            return {}
        return {
            "loop_id": contract.loop_id,
            "job_key": contract.job_key,
            "status": contract.status.value,
            "cadence": contract.schedule.cadence,
            "schedule": contract.schedule.model_dump(mode="json"),
            "authority": contract.authority.model_dump(mode="json"),
            "reads": [item.model_dump(mode="json") for item in contract.reads],
            "writes": [item.model_dump(mode="json") for item in contract.writes],
            "verification": [item.model_dump(mode="json") for item in contract.verification],
            "stopping_criteria": contract.stopping_criteria,
            "purpose": contract.body_sections.get("Purpose", ""),
            "current_focus": contract.body_sections.get("Current focus", ""),
        }


def validate_scheduler_contracts(
    specs: Iterable[ScheduledJobSpec],
    *,
    memory_dir: Path | None = None,
    require_skill_freshness: bool = True,
) -> list[LoopContractIssue]:
    store = LoopContractStore(memory_dir)
    contracts = store.load_all()
    issues: list[LoopContractIssue] = []
    by_job_key = {
        contract.job_key: contract
        for contract in contracts.values()
        if contract.status == LoopStatus.ACTIVE
    }
    for spec in specs:
        if spec.job_class == ScheduledJobClass.INTERVAL:
            continue
        contract_id = spec.contract_id or spec.job_key
        contract = contracts.get(contract_id) or by_job_key.get(spec.job_key)
        if contract is None:
            issues.append(LoopContractIssue(
                am_row="AM-01",
                path=f"ScheduledJobSpec:{spec.name}",
                field="contract_id",
                message=f"{spec.job_key} has no active loop contract",
                remediation=f"Create memory/loops/{contract_id}.md or set a matching contract_id.",
            ))
            continue
        issues.extend(_validate_spec_schedule(spec, contract))
        if contract.authority.may_modify_live_bot_state:
            issues.append(LoopContractIssue(
                am_row="AM-01",
                path=contract.source_path,
                field="authority.may_modify_live_bot_state",
                message="loop contract grants live bot mutation authority",
                remediation="Set may_modify_live_bot_state: false and keep deployment behind approval.",
            ))
        if contract.authority.may_modify_policy_memory:
            issues.append(LoopContractIssue(
                am_row="AM-01",
                path=contract.source_path,
                field="authority.may_modify_policy_memory",
                message="loop contract grants autonomous policy-memory writes",
                remediation="Set may_modify_policy_memory: false; policy memory is human-owned.",
            ))
    if require_skill_freshness:
        issues.extend(validate_skill_trigger_freshness(memory_dir=store.memory_dir))
    return issues


def validate_skill_trigger_freshness(memory_dir: Path) -> list[LoopContractIssue]:
    issues: list[LoopContractIssue] = []
    store = LoopContractStore(memory_dir)
    contracts = store.load_all()
    skills_dir = memory_dir / "skills"
    index_path = skills_dir / "skills_index.md"
    if index_path.exists():
        text = index_path.read_text(encoding="utf-8")
        for match in _SKILL_TRIGGER_RE.finditer(text):
            skill = match.group("skill").strip()
            contract = _contract_for_skill(skill, contracts)
            if contract is None:
                continue
            trigger = match.group("trigger").strip()
            issues.extend(_validate_trigger_text(
                trigger,
                contract,
                path=str(index_path),
                field=f"trigger:{skill}",
            ))
    daily_skill = skills_dir / "daily_analysis.md"
    if daily_skill.exists():
        text = daily_skill.read_text(encoding="utf-8")
        match = _SCHEDULED_DAILY_RE.search(text)
        contract = contracts.get("daily_analysis")
        if match and contract is not None:
            hour = int(match.group("hour"))
            minute = int(match.group("minute"))
            if hour != contract.schedule.hour or minute != contract.schedule.minute:
                issues.append(LoopContractIssue(
                    am_row="AM-02",
                    path=str(daily_skill),
                    field="Trigger",
                    message=(
                        f"daily skill doc says {hour:02d}:{minute:02d} UTC; "
                        f"contract/scheduler says {contract.schedule.hour:02d}:{contract.schedule.minute:02d} UTC"
                    ),
                    remediation="Edit memory/skills/daily_analysis.md Trigger to match daily_analysis contract.",
                ))
    return issues


def _contract_for_skill(
    skill: str,
    contracts: dict[str, LoopContract],
) -> LoopContract | None:
    aliases = {
        "weekly_analysis": "weekly_summary",
        "strategy_refinement": "",
    }
    loop_id = aliases.get(skill, skill)
    return contracts.get(loop_id) if loop_id else None


def _validate_spec_schedule(
    spec: ScheduledJobSpec,
    contract: LoopContract,
) -> list[LoopContractIssue]:
    issues: list[LoopContractIssue] = []
    schedule = contract.schedule
    comparisons = {
        "job_key": (contract.job_key, spec.job_key),
        "trigger": (schedule.trigger.value, spec.trigger),
        "hour": (schedule.hour, spec.hour),
        "minute": (schedule.minute, spec.minute),
        "day_of_week": (_weekday(schedule.day_of_week), _weekday(spec.day_of_week)),
        "day": (schedule.day, spec.day),
        "coalesce": (schedule.coalesce, spec.coalesce),
        "catchup_limit": (schedule.catchup_limit, spec.catchup_limit),
    }
    for field, (contract_value, spec_value) in comparisons.items():
        if contract_value != spec_value:
            issues.append(LoopContractIssue(
                am_row="AM-02",
                path=contract.source_path,
                field=f"schedule.{field}",
                message=(
                    f"contract value {contract_value!r} does not match "
                    f"ScheduledJobSpec {spec_value!r} for {spec.name}"
                ),
                remediation=f"Edit memory/loops/{contract.loop_id}.md or scheduler field {field}.",
            ))
    return issues


def _validate_trigger_text(
    trigger: str,
    contract: LoopContract,
    *,
    path: str,
    field: str,
) -> list[LoopContractIssue]:
    issues: list[LoopContractIssue] = []
    match = _UTC_RE.search(trigger)
    if match and contract.schedule.hour is not None and contract.schedule.minute is not None:
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        if (hour, minute) != (contract.schedule.hour, contract.schedule.minute):
            issues.append(LoopContractIssue(
                am_row="AM-02",
                path=path,
                field=field,
                message=(
                    f"trigger says {hour:02d}:{minute:02d} UTC; "
                    f"contract says {contract.schedule.hour:02d}:{contract.schedule.minute:02d} UTC"
                ),
                remediation=f"Update {path} trigger for {contract.loop_id}.",
            ))
    if contract.schedule.day_of_week:
        day = contract.schedule.day_of_week.lower()[:3]
        if day == "sun" and "sunday" not in trigger.lower() and "sun" not in trigger.lower():
            issues.append(LoopContractIssue(
                am_row="AM-02",
                path=path,
                field=field,
                message=f"trigger omits expected weekly day {contract.schedule.day_of_week}",
                remediation=f"Update {path} trigger to include {contract.schedule.day_of_week}.",
            ))
    return issues


def _weekday(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()[:3]
