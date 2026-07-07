"""Read-only projection helpers for loop-run activity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.scheduled_runs import ScheduledRunRecord
from trading_assistant.schemas.loop_run_ledger import LoopRunLedgerEntry, LoopSourceRecord


_OUTPUT_ARTIFACT_KEYS = {
    "artifact_index",
    "candidate_attempts",
    "runner_observability",
    "model_review_validation",
    "candidate_gate_report",
    "candidate_summary",
}


class LoopRunLedgerStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, *, strict: bool = True) -> list[LoopRunLedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[LoopRunLedgerEntry] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    entries.append(LoopRunLedgerEntry.model_validate(data))
                elif strict:
                    raise ValueError("line is not a JSON object")
            except Exception as exc:
                if strict:
                    raise ValueError(
                        f"{self.path}:{line_no}: invalid loop-run ledger record: {exc}"
                    ) from exc
                continue
        return entries

    def write_projection(self, entries: Iterable[LoopRunLedgerEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = {entry.loop_run_id: entry for entry in self.read()}
        for entry in entries:
            merged[entry.loop_run_id] = entry
        lines = [
            entry.model_dump_json()
            for entry in sorted(
                merged.values(),
                key=lambda item: (
                    item.scheduled_for.isoformat() if item.scheduled_for else "",
                    item.loop_run_id,
                ),
            )
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class RuntimeLoopProjectionWriter:
    """Projection hook for finalized scheduled runs.

    ScheduledRunStore remains the lifecycle authority. This class only mirrors a
    completed/failed scheduled-run record into the compact JSONL projection and
    regenerates the human-readable work log.
    """

    def __init__(self, memory_dir: Path, *, work_log_limit: int = 50) -> None:
        self.memory_dir = Path(memory_dir)
        self.ledger_path = self.memory_dir / "findings" / "loop_run_ledger.jsonl"
        self.work_log_path = self.memory_dir / "work_log.md"
        self.work_log_limit = work_log_limit
        self._projector = LoopRunLedgerProjector()

    def project_record(
        self,
        record: ScheduledRunRecord,
        *,
        run_metadata: dict[str, Any] | None = None,
        monthly_artifacts: dict[str, str] | None = None,
        blocking_reasons: list[str] | None = None,
        approval_packet_paths: list[str] | None = None,
        strategy_change_record_ids: list[str] | None = None,
        proposal_ids: list[str] | None = None,
        summary: str = "",
    ) -> None:
        loop_id = _loop_id_for_job_key(record.job_key)
        entry = self._projector.project_scheduled_run(
            record,
            loop_id=loop_id,
            run_metadata=run_metadata,
            monthly_artifacts=monthly_artifacts,
            blocking_reasons=blocking_reasons,
            approval_packet_paths=approval_packet_paths,
            strategy_change_record_ids=strategy_change_record_ids,
            proposal_ids=proposal_ids,
            summary=summary or _runtime_summary(record, loop_id),
        )
        entry.loop_run_id = _runtime_loop_run_id(record, loop_id)
        LoopRunLedgerStore(self.ledger_path).write_projection([entry])
        from trading_assistant.skills.work_log_projector import WorkLogProjector

        WorkLogProjector(self.ledger_path, self.work_log_path).project(limit=self.work_log_limit)


class LoopRunLedgerProjector:
    """Projects scheduled-run state plus optional links into compact ledger entries."""

    def project_scheduled_run(
        self,
        record: ScheduledRunRecord,
        *,
        loop_id: str | None = None,
        task: Any | None = None,
        run_metadata: dict[str, Any] | None = None,
        monthly_artifacts: dict[str, str] | None = None,
        blocking_reasons: list[str] | None = None,
        approval_packet_paths: list[str] | None = None,
        strategy_change_record_ids: list[str] | None = None,
        proposal_ids: list[str] | None = None,
        summary: str = "",
    ) -> LoopRunLedgerEntry:
        metadata = run_metadata or {}
        artifacts = monthly_artifacts or {}
        task_id = str(getattr(task, "id", "") or metadata.get("task_id") or "")
        run_id = str(metadata.get("run_id") or metadata.get("agent_run_id") or "")
        source_records = [
            LoopSourceRecord(
                kind="scheduled_run",
                id="|".join([
                    record.job_key,
                    record.scope_key,
                    record.scheduled_for.isoformat(),
                ]),
            )
        ]
        if task_id:
            source_records.append(LoopSourceRecord(
                kind="task",
                id=task_id,
                path=str(getattr(task, "run_folder", "") or ""),
            ))
        if run_id:
            source_records.append(LoopSourceRecord(
                kind="run_index",
                id=run_id,
                path=str(metadata.get("run_dir") or ""),
            ))
        for name, path in sorted(artifacts.items()):
            if path:
                source_records.append(LoopSourceRecord(kind="artifact", id=name, path=path))

        task_summary = str(getattr(task, "result_summary", "") or "")
        task_error = str(getattr(task, "error", "") or "")
        reasons = [*(blocking_reasons or [])]
        if record.error:
            reasons.append(record.error)
        if task_error:
            reasons.append(task_error)
        readable_summary = summary or task_summary or _default_summary(record)
        output_artifacts = [
            path for key, path in sorted(artifacts.items())
            if _artifact_key_matches_any(key, _OUTPUT_ARTIFACT_KEYS)
            and path
        ]
        evidence_paths = [
            path for key, path in sorted(artifacts.items())
            if not _artifact_key_matches(key, "approval_packet") and path
        ]
        return LoopRunLedgerEntry(
            loop_id=loop_id or record.job_key,
            job_key=record.job_key,
            scope_key=record.scope_key,
            bot_id=str(metadata.get("bot_id") or _bot_from_scope(record.scope_key)),
            strategy_id=str(metadata.get("strategy_id") or ""),
            scheduled_for=record.scheduled_for,
            started_at=record.started_at,
            completed_at=record.finished_at,
            status=record.status,
            task_id=task_id,
            task_retry_count=int(getattr(task, "retries", 0) or metadata.get("retries") or 0),
            task_stale=bool(metadata.get("task_stale", False)),
            agent_run_id=run_id,
            provider=str(metadata.get("provider") or ""),
            model=str(metadata.get("model") or ""),
            cost_usd=float(metadata.get("cost_usd") or 0.0),
            duration_ms=int(metadata.get("duration_ms") or 0),
            run_dir=str(metadata.get("run_dir") or getattr(task, "run_folder", "") or ""),
            run_index_id=str(metadata.get("run_index_id") or run_id),
            input_artifacts=[path for key, path in sorted(artifacts.items()) if key.endswith("_input")],
            output_artifacts=output_artifacts,
            evidence_paths=evidence_paths,
            blocking_reasons=_dedupe(reasons),
            approval_packet_paths=approval_packet_paths or _list_value(artifacts.get("approval_packet")),
            strategy_change_record_ids=strategy_change_record_ids or [],
            proposal_ids=proposal_ids or [],
            summary=readable_summary,
            source_records=source_records,
        )

    def repeated_blockers(
        self,
        entries: Iterable[LoopRunLedgerEntry],
    ) -> dict[tuple[str, str, str], list[str]]:
        grouped: dict[tuple[str, str, str], list[str]] = {}
        for entry in entries:
            if entry.blocking_reasons:
                grouped.setdefault(entry.grouping_key, []).extend(entry.blocking_reasons)
        return {key: _dedupe(values) for key, values in grouped.items()}


def _default_summary(record: ScheduledRunRecord) -> str:
    if record.status == "failed" and record.error:
        return f"{record.job_key} failed: {record.error}"
    return f"{record.job_key} {record.status}"


def _bot_from_scope(scope_key: str) -> str:
    if scope_key.startswith("bot:"):
        return scope_key.split(":", 1)[1]
    return ""


def _loop_id_for_job_key(job_key: str) -> str:
    aliases = {
        "weekly_analysis": "weekly_summary",
        "triage": "bug_triage",
    }
    return aliases.get(job_key, job_key)


def _runtime_summary(record: ScheduledRunRecord, loop_id: str) -> str:
    if record.status == "failed" and record.error:
        return f"{loop_id} failed: {record.error}"
    return f"{loop_id} {record.status}"


def _runtime_loop_run_id(record: ScheduledRunRecord, loop_id: str) -> str:
    raw = "|".join([
        loop_id,
        record.job_key,
        record.scope_key,
        record.scheduled_for.isoformat(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _list_value(value: object) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    return []


def _artifact_key_matches_any(key: str, names: set[str]) -> bool:
    return any(_artifact_key_matches(key, name) for name in names)


def _artifact_key_matches(key: str, name: str) -> bool:
    return (
        key == name
        or key.startswith(f"{name}_")
        or key.endswith(f"_{name}")
        or f"_{name}_" in key
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
