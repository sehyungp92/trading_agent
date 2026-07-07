"""Typed loop contracts for recurring orchestrator work."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from trading_assistant.schemas.artifact_authority import ArtifactAuthority


class LoopStatus(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"
    RETIRED = "retired"


class LoopTriggerType(str, Enum):
    CRON = "cron"
    EVENT = "event"
    INTERVAL = "interval"


LoopArtifactAuthority = ArtifactAuthority


class LoopArtifactRef(BaseModel):
    artifact_type: str
    authority: LoopArtifactAuthority = LoopArtifactAuthority.GENERATED
    path_hint: str = ""
    notes: str = ""


class LoopVerificationRequirement(BaseModel):
    requirement_id: str
    description: str = ""
    required_for_approval: bool = False


class LoopAuthority(BaseModel):
    may_create_approval_request: bool = False
    may_modify_policy_memory: bool = False
    may_modify_live_bot_state: bool = False
    may_write_generated_memory: bool = True
    negative_authority: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _guard_trading_authority(self) -> "LoopAuthority":
        if self.may_modify_live_bot_state:
            raise ValueError("loop contracts may not grant live bot mutation authority")
        if self.may_modify_policy_memory:
            raise ValueError("loop contracts may not grant autonomous policy-memory writes")
        required = {
            "no_live_bot_mutation",
            "no_autonomous_policy_memory_write",
        }
        missing = sorted(required - set(self.negative_authority))
        if missing:
            raise ValueError(
                "loop contract missing negative authority: " + ", ".join(missing)
            )
        return self


class LoopSchedule(BaseModel):
    trigger: LoopTriggerType = LoopTriggerType.CRON
    cadence: str = ""
    hour: int | None = None
    minute: int | None = None
    day_of_week: str | None = None
    day: int | None = None
    coalesce: bool | None = None
    catchup_limit: int = 0


class LoopContract(BaseModel):
    loop_id: str
    contract_id: str = ""
    status: LoopStatus = LoopStatus.ACTIVE
    job_key: str
    schedule_source: str = "trading_assistant.orchestrator.scheduler"
    schedule: LoopSchedule = Field(default_factory=LoopSchedule)
    authority: LoopAuthority = Field(default_factory=LoopAuthority)
    reads: list[LoopArtifactRef] = Field(default_factory=list)
    writes: list[LoopArtifactRef] = Field(default_factory=list)
    verification: list[LoopVerificationRequirement] = Field(default_factory=list)
    stopping_criteria: list[str] = Field(default_factory=list)
    body_sections: dict[str, str] = Field(default_factory=dict)
    source_path: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    contract_version: str = "loop_contract_v1"

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_frontmatter(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        schedule = dict(normalized.get("schedule") or {})
        for key in ("trigger_type", "cadence", "hour", "minute", "day", "day_of_week", "coalesce", "catchup_limit"):
            if key in normalized and key not in schedule:
                schedule_key = "trigger" if key == "trigger_type" else key
                schedule[schedule_key] = normalized[key]
        if schedule:
            normalized["schedule"] = schedule
        authority = dict(normalized.get("authority") or {})
        for key in (
            "may_create_approval_request",
            "may_modify_policy_memory",
            "may_modify_live_bot_state",
            "may_write_generated_memory",
            "negative_authority",
        ):
            if key in normalized and key not in authority:
                authority[key] = normalized[key]
        if authority:
            normalized["authority"] = authority
        return normalized

    @model_validator(mode="after")
    def _normalize(self) -> "LoopContract":
        if not self.contract_id:
            self.contract_id = self.loop_id
        required_sections = {
            "Purpose",
            "Current focus",
            "Authority boundary",
            "Inputs",
            "Outputs",
            "Required checks",
            "Failure modes",
            "Escalation path",
            "Backlog",
            "Timeline",
        }
        missing = sorted(required_sections - set(self.body_sections))
        if missing:
            raise ValueError(
                f"{self.loop_id} missing required body sections: {', '.join(missing)}"
            )
        if not self.stopping_criteria:
            raise ValueError(f"{self.loop_id} must declare stopping_criteria")
        return self

    @property
    def approval_verifier_required(self) -> bool:
        return any(item.required_for_approval for item in self.verification)


def parse_loop_contract_markdown(path: Path) -> LoopContract:
    """Parse a markdown contract with YAML frontmatter and required sections."""

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path} is missing YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path} has malformed YAML frontmatter")
    raw_meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(raw_meta, dict):
        raise ValueError(f"{path} frontmatter must be a mapping")
    body = parts[2].strip()
    sections = _parse_sections(body)
    contract = LoopContract.model_validate(
        {
            **raw_meta,
            "body_sections": sections,
            "source_path": str(path),
        }
    )
    return contract


def _parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line.rstrip())
    return {
        heading: "\n".join(lines).strip()
        for heading, lines in sections.items()
    }
