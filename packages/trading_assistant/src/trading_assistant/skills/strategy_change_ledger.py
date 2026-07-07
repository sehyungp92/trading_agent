"""JSONL-backed StrategyChangeLedger."""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.orchestrator.jsonl_store import (
    jsonl_file_lock,
    read_json_projection,
    write_json_projection,
)
from trading_assistant.schemas.strategy_change_ledger import (
    RollbackStatus,
    StrategyChangeRecord,
    StrategyChangeRecordType,
)

logger = logging.getLogger(__name__)


class StrategyChangeLedger:
    """Append/update ledger for material strategy decisions and outcomes."""

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = Path(store_dir)
        self._path = self._store_dir / "strategy_change_ledger.jsonl"
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, record: StrategyChangeRecord) -> bool:
        """Append a record. Returns False when record_id already exists."""

        with jsonl_file_lock(self._path), self._lock:
            records = self._read_record_map_unlocked()
            if record.record_id in records:
                return False
            self._append_unlocked({"type": "record", "payload": record.model_dump(mode="json")})
            records[record.record_id] = record
            self._write_projection_unlocked(records.values())
        self._refresh_performance_learning_projection()
        return True

    def update(self, record_id: str, **changes) -> bool:
        with jsonl_file_lock(self._path), self._lock:
            records = self._read_record_map_unlocked()
            record = records.get(record_id)
            if record is None:
                return False
            payload = {
                "record_id": record_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **changes,
            }
            self._append_unlocked({"type": "update", "payload": payload})
            merged = record.model_dump(mode="json")
            merged.update(payload)
            records[record_id] = StrategyChangeRecord.model_validate(merged)
            self._write_projection_unlocked(records.values())
        self._refresh_performance_learning_projection()
        return True

    def record_monthly_review(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        run_id: str,
        run_month: str,
        monthly_status: str,
        evidence_paths: list[str],
        decision_reason: str,
        objective_deltas: dict[str, float] | None = None,
        source_proposal_ids: list[str] | None = None,
        source_suggestion_ids: list[str] | None = None,
        created_at: datetime | None = None,
    ) -> StrategyChangeRecord:
        record = self.build_monthly_review_record(
            bot_id=bot_id,
            strategy_id=strategy_id,
            run_id=run_id,
            run_month=run_month,
            monthly_status=monthly_status,
            evidence_paths=evidence_paths,
            objective_deltas=objective_deltas or {},
            decision_reason=decision_reason,
            source_proposal_ids=source_proposal_ids or [],
            source_suggestion_ids=source_suggestion_ids or [],
            created_at=created_at,
        )
        if self.record(record):
            return record
        self.update(
            record.record_id,
            run_id=run_id,
            run_month=run_month,
            monthly_status=monthly_status,
            evidence_paths=evidence_paths,
            objective_deltas=objective_deltas or {},
            decision_reason=decision_reason,
            source_proposal_ids=source_proposal_ids or [],
            source_suggestion_ids=source_suggestion_ids or [],
        )
        return self.get_by_id(record.record_id) or record

    def build_monthly_review_record(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        run_id: str,
        run_month: str,
        monthly_status: str,
        evidence_paths: list[str],
        decision_reason: str,
        objective_deltas: dict[str, float] | None = None,
        source_proposal_ids: list[str] | None = None,
        source_suggestion_ids: list[str] | None = None,
        created_at: datetime | None = None,
    ) -> StrategyChangeRecord:
        """Build the deterministic monthly-review record without appending it."""

        return StrategyChangeRecord(
            bot_id=bot_id,
            strategy_id=strategy_id,
            record_type=StrategyChangeRecordType.MONTHLY_REVIEW,
            run_id=run_id,
            run_month=run_month,
            monthly_status=monthly_status,
            evidence_paths=evidence_paths,
            objective_deltas=objective_deltas or {},
            decision_reason=decision_reason,
            source_proposal_ids=source_proposal_ids or [],
            source_suggestion_ids=source_suggestion_ids or [],
            created_at=created_at or datetime.now(timezone.utc),
        )

    def record_proposed_change(self, record: StrategyChangeRecord) -> bool:
        if record.record_type != StrategyChangeRecordType.PROPOSED_CHANGE:
            record.record_type = StrategyChangeRecordType.PROPOSED_CHANGE
        return self.record(record)

    def record_deployed_change(
        self,
        record_id: str,
        *,
        deployment_id: str,
        commit_sha: str = "",
        deployed_at: datetime | None = None,
        config_version: str = "",
        strategy_version: str = "",
    ) -> bool:
        changes = {
            "record_type": StrategyChangeRecordType.DEPLOYED_CHANGE.value,
            "deployment_id": deployment_id,
            "commit_sha": commit_sha or None,
            "deployed_at": (deployed_at or datetime.now(timezone.utc)).isoformat(),
        }
        if config_version:
            changes["new_config_version"] = config_version
        if strategy_version:
            changes["strategy_version"] = strategy_version
        return self.update(record_id, **changes)

    def record_deployment_writeback(
        self,
        *,
        record_id: str = "",
        approval_request_id: str = "",
        deployment_id: str,
        pr_url: str = "",
        commit_sha: str = "",
        deployed_at: datetime | None = None,
        config_version: str = "",
        strategy_version: str = "",
    ) -> bool:
        """Link approval/PR/deployment lineage into the strategy change record."""
        target_id = record_id or self._find_record_id_by_approval(approval_request_id)
        if not target_id:
            return False
        changes = {
            "record_type": StrategyChangeRecordType.DEPLOYED_CHANGE.value,
            "deployment_id": deployment_id,
            "deployed_at": (deployed_at or datetime.now(timezone.utc)).isoformat(),
        }
        if approval_request_id:
            changes["approval_request_id"] = approval_request_id
        if pr_url:
            changes["pr_url"] = pr_url
        if commit_sha:
            changes["commit_sha"] = commit_sha
        if config_version:
            changes["new_config_version"] = config_version
        if strategy_version:
            changes["strategy_version"] = strategy_version
        return self.update(target_id, **changes)

    def record_one_month_verdict(self, record_id: str, verdict: dict) -> bool:
        return self.update(record_id, monthly_verdict=verdict)

    def record_follow_up_verdict(self, record_id: str, verdict: dict) -> bool:
        return self.update(record_id, follow_up_verdict=verdict)

    def record_rollback(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        evidence_paths: list[str],
        decision_reason: str,
        rollback_status: RollbackStatus = RollbackStatus.RECOMMENDED,
    ) -> StrategyChangeRecord:
        record = StrategyChangeRecord(
            bot_id=bot_id,
            strategy_id=strategy_id,
            record_type=StrategyChangeRecordType.ROLLBACK,
            evidence_paths=evidence_paths,
            decision_reason=decision_reason,
            rollback_status=rollback_status,
        )
        self.record(record)
        return record

    def record_no_change_decision(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        run_id: str,
        run_month: str,
        evidence_paths: list[str],
        decision_reason: str,
    ) -> StrategyChangeRecord:
        record = StrategyChangeRecord(
            bot_id=bot_id,
            strategy_id=strategy_id,
            record_type=StrategyChangeRecordType.NO_CHANGE,
            run_id=run_id,
            run_month=run_month,
            evidence_paths=evidence_paths,
            decision_reason=decision_reason,
            monthly_status="no_change",
        )
        self.record(record)
        return record

    def get_recent(self, days: int = 180) -> list[StrategyChangeRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            record for record in self._read_records()
            if _aware(record.updated_at) >= cutoff
        ]

    def get_for_strategy(self, bot_id: str, strategy_id: str, days: int = 365) -> list[StrategyChangeRecord]:
        return [
            record for record in self.get_recent(days=days)
            if record.bot_id == bot_id and record.strategy_id == strategy_id
        ]

    def projected_records(
        self,
        *,
        bot_id: str = "",
        strategy_id: str = "",
        days: int = 365,
        limit: int | None = None,
    ) -> list[dict]:
        """Return normalized row-shaped records for JSONL consumers.

        The ledger is event-sourced on disk. Consumers that need a row-like
        contract should use this projection instead of reading the JSONL file
        directly and accidentally depending on the event wrapper.
        """

        records = self.get_recent(days=days)
        rows: list[dict] = []
        for record in records:
            if bot_id and record.bot_id not in ("", bot_id):
                continue
            if strategy_id and record.strategy_id not in ("", strategy_id):
                continue
            rows.append(project_strategy_change_record(record))
        return rows[:limit] if limit is not None else rows

    def get_by_id(self, record_id: str) -> StrategyChangeRecord | None:
        with jsonl_file_lock(self._path), self._lock:
            return self._read_record_map_unlocked().get(record_id)

    def _find_record_id_by_approval(self, approval_request_id: str) -> str:
        if not approval_request_id:
            return ""
        for record in self._read_records():
            if record.approval_request_id == approval_request_id:
                return record.record_id
        return ""

    def _append(self, event: dict) -> None:
        with jsonl_file_lock(self._path), self._lock:
            self._append_unlocked(event)

    def _append_unlocked(self, event: dict) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _refresh_performance_learning_projection(self) -> None:
        try:
            from trading_assistant.skills.performance_learning_ledger import (
                PerformanceLearningRefreshMarkerError,
                refresh_performance_learning_projection,
            )

            refresh_performance_learning_projection(self._store_dir)
        except PerformanceLearningRefreshMarkerError:
            raise
        except Exception:
            logger.warning("Failed to refresh performance-learning projection", exc_info=True)

    def _read_records(self) -> list[StrategyChangeRecord]:
        with jsonl_file_lock(self._path), self._lock:
            records = self._read_record_map_unlocked()
        return sorted(records.values(), key=lambda record: record.updated_at, reverse=True)

    def _read_record_map_unlocked(self) -> dict[str, StrategyChangeRecord]:
        projection = self._read_projection_unlocked()
        if projection:
            return projection

        records: dict[str, StrategyChangeRecord] = {}
        if not self._path.exists():
            return records
        for line_no, line in enumerate(self._path.open("r", encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                etype = event.get("type")
                payload = event.get("payload") or {}
                if etype == "record":
                    record = StrategyChangeRecord.model_validate(payload)
                    records[record.record_id] = record
                elif etype == "update":
                    record_id = payload.get("record_id")
                    if record_id and record_id in records:
                        merged = records[record_id].model_dump(mode="json")
                        merged.update(payload)
                        records[record_id] = StrategyChangeRecord.model_validate(merged)
                elif "record_id" in event and "record_type" in event:
                    record = StrategyChangeRecord.model_validate(event)
                    records[record.record_id] = record
            except Exception:
                logger.warning("Skipping malformed strategy-change ledger line %d", line_no)
        self._write_projection_unlocked(records.values())
        return records

    def _read_projection_unlocked(self) -> dict[str, StrategyChangeRecord]:
        if not self._path.exists():
            return {}
        projection_path = Path(str(self._path) + ".index.json")
        if not projection_path.exists() or projection_path.stat().st_mtime < self._path.stat().st_mtime:
            return {}
        records: dict[str, StrategyChangeRecord] = {}
        for record_id, payload in read_json_projection(self._path).items():
            try:
                record = StrategyChangeRecord.model_validate(payload)
            except Exception:
                return {}
            records[record_id] = record
        return records

    def _write_projection_unlocked(self, records: Iterable[StrategyChangeRecord]) -> None:
        write_json_projection(
            self._path,
            key_field="record_id",
            records=[
                record.model_dump(mode="json")
                for record in records
                if isinstance(record, StrategyChangeRecord)
            ],
        )

    def compact(self) -> int:
        """Rewrite the event log to one latest-state record per strategy change."""
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl

        with jsonl_file_lock(self._path), self._lock:
            records = self._read_record_map_unlocked()
            ordered = sorted(records.values(), key=lambda record: record.updated_at)
            events = [
                {"type": "record", "payload": record.model_dump(mode="json")}
                for record in ordered
            ]
            atomic_rewrite_jsonl(self._path, events)
            self._write_projection_unlocked(ordered)
            return len(events)


def project_strategy_change_record(record: StrategyChangeRecord) -> dict:
    """Flatten a strategy-change record for recall/search consumers."""

    payload = record.model_dump(mode="json")
    mutation = payload.get("mutation_diff") or {}
    if not isinstance(mutation, dict):
        mutation = {}
    family = str(
        mutation.get("mutation_family")
        or mutation.get("family")
        or mutation.get("change_kind")
        or payload.get("monthly_status")
        or payload.get("record_type")
        or ""
    )
    category = str(
        mutation.get("category")
        or mutation.get("change_kind")
        or family
        or ""
    )
    payload.update({
        "record_type": record.record_type.value,
        "rollback_status": record.rollback_status.value,
        "proposal_ids": list(record.source_proposal_ids),
        "suggestion_ids": list(record.source_suggestion_ids),
        "mutation_family": family,
        "category": category,
        "recorded_at": record.updated_at.isoformat(),
        "approval_status": (
            "approval_request_linked"
            if record.approval_request_id else payload.get("monthly_status", "")
        ),
    })
    return payload


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
