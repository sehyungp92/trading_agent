"""Compact projection records for loop activity."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class LoopSourceRecord(BaseModel):
    kind: str
    id: str
    path: str = ""


class LoopRunLedgerEntry(BaseModel):
    loop_run_id: str = ""
    loop_id: str
    job_key: str
    scope_key: str = "global"
    bot_id: str = ""
    strategy_id: str = ""
    scheduled_for: datetime | None = None
    started_at: str = ""
    completed_at: str = ""
    status: str
    trigger_source: str = "scheduled_run_store"
    task_id: str = ""
    task_retry_count: int = 0
    task_stale: bool = False
    agent_run_id: str = ""
    provider: str = ""
    model: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0
    run_dir: str = ""
    run_index_id: str = ""
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    approval_packet_paths: list[str] = Field(default_factory=list)
    strategy_change_record_ids: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    source_records: list[LoopSourceRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.loop_run_id:
            raw = "|".join([
                self.loop_id,
                self.job_key,
                self.scope_key,
                self.scheduled_for.isoformat() if self.scheduled_for else "",
                self.task_id,
                self.agent_run_id,
            ])
            self.loop_run_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def grouping_key(self) -> tuple[str, str, str]:
        return (self.loop_id, self.bot_id, self.strategy_id)

    def work_log_line(self) -> str:
        when = (
            self.completed_at
            or (self.scheduled_for.isoformat() if self.scheduled_for else "")
            or self.generated_at.isoformat()
        )
        date = when[:10]
        what = self.summary or f"{self.job_key} {self.status}"
        blockers = "; ".join(self.blocking_reasons[:3])
        refs = [*self.evidence_paths[:3], *self.approval_packet_paths[:2]]
        suffixes: list[str] = []
        if blockers:
            suffixes.append(f"Blockers: {blockers}")
        if refs:
            suffixes.append("Refs: " + "; ".join(refs))
        suffix = "\n" + "\n".join(suffixes) if suffixes else ""
        return f"## {date} - {self.loop_id} - {self.status}\nWhat: {what}{suffix}"


def entry_from_dict(data: dict[str, Any]) -> LoopRunLedgerEntry:
    return LoopRunLedgerEntry.model_validate(data)
