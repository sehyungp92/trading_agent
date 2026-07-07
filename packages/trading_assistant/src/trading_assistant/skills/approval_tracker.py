# skills/approval_tracker.py
"""Approval tracker — JSONL-backed approval request lifecycle.

Tracks parameter change approval requests through PENDING → APPROVED/REJECTED/EXPIRED.
Same persistence pattern as SuggestionTracker.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from trading_assistant.schemas.approval import ApprovalRequest, ApprovalStatus, RepoRiskTier
from trading_assistant.orchestrator.jsonl_store import (
    jsonl_file_lock,
    read_json_projection,
    write_json_projection,
)

logger = logging.getLogger(__name__)


class ApprovalTracker:
    """JSONL-backed approval request lifecycle tracker."""

    def __init__(self, storage_path: Path) -> None:
        self._path = Path(storage_path)
        self._lock = threading.RLock()

    def create_request(self, request: ApprovalRequest) -> ApprovalRequest:
        """Persist a new approval request. Deduplicates by request_id."""
        with jsonl_file_lock(self._path), self._lock:
            existing = self._load_all()
            existing_ids = {r.request_id for r in existing}
            if request.request_id in existing_ids:
                return request
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(request.model_dump(mode="json"), default=str) + "\n")
            existing.append(request)
            write_json_projection(
                self._path,
                key_field="request_id",
                records=[record.model_dump(mode="json") for record in existing],
            )
            return request

    def approve(self, request_id: str, resolved_by: str = "telegram") -> ApprovalRequest:
        """Transition a PENDING request to APPROVED."""
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            target = None
            for r in requests:
                if r.request_id == request_id:
                    target = r
                    break
            if target is None:
                raise ValueError(f"Request {request_id} not found")
            if target.status != ApprovalStatus.PENDING:
                raise ValueError(f"Request {request_id} is {target.status.value}, not PENDING")
            if (
                target.risk_tier == RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
                and target.approval_count == 0
            ):
                target.approval_count = 1
                self._save_all(requests)
                return target
            target.approval_count += 1
            target.status = ApprovalStatus.APPROVED
            target.resolved_at = datetime.now(timezone.utc)
            target.resolved_by = resolved_by
            self._save_all(requests)
            return target

    def reject(self, request_id: str, reason: str = "", resolved_by: str = "telegram") -> ApprovalRequest:
        """Transition a PENDING request to REJECTED."""
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            target = None
            for r in requests:
                if r.request_id == request_id:
                    target = r
                    break
            if target is None:
                raise ValueError(f"Request {request_id} not found")
            if target.status != ApprovalStatus.PENDING:
                raise ValueError(f"Request {request_id} is {target.status.value}, not PENDING")
            target.status = ApprovalStatus.REJECTED
            target.resolved_at = datetime.now(timezone.utc)
            target.resolved_by = resolved_by
            target.rejection_reason = reason or "rejected via Telegram"
            self._save_all(requests)
        return target

    def expire_old(self, max_age_days: int = 7) -> list[str]:
        """Expire PENDING requests older than max_age_days. Returns expired IDs."""
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            expired_ids: list[str] = []
            for r in requests:
                if r.status == ApprovalStatus.PENDING and r.created_at < cutoff:
                    r.status = ApprovalStatus.EXPIRED
                    r.resolved_at = datetime.now(timezone.utc)
                    expired_ids.append(r.request_id)
            if expired_ids:
                self._save_all(requests)
            return expired_ids

    def get_pending(self) -> list[ApprovalRequest]:
        with jsonl_file_lock(self._path):
            return [r for r in self._load_all() if r.status == ApprovalStatus.PENDING]

    def find_pending_for_suggestion(self, suggestion_id: str) -> ApprovalRequest | None:
        for request in self.get_pending():
            if request.suggestion_id == suggestion_id:
                return request
        return None

    def find_latest_for_suggestion(self, suggestion_id: str) -> ApprovalRequest | None:
        with jsonl_file_lock(self._path):
            matches = [
                request for request in self._load_all()
                if request.suggestion_id == suggestion_id
            ]
        if not matches:
            return None
        return max(matches, key=lambda request: request.created_at)

    def get_approved_with_prs(self) -> list[ApprovalRequest]:
        """Return APPROVED requests that have a non-empty pr_url (for review monitoring)."""
        with jsonl_file_lock(self._path):
            return [
                r for r in self._load_all()
                if r.status == ApprovalStatus.APPROVED and r.pr_url
            ]

    def get_by_id(self, request_id: str) -> ApprovalRequest | None:
        with jsonl_file_lock(self._path):
            for r in self._load_all():
                if r.request_id == request_id:
                    return r
        return None

    def set_pr_url(self, request_id: str, pr_url: str) -> None:
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            for r in requests:
                if r.request_id == request_id:
                    r.pr_url = pr_url
                    break
            self._save_all(requests)

    def set_pr_result(
        self,
        request_id: str,
        pr_url: str,
        diff_summary: list[str] | None = None,
    ) -> None:
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            for request in requests:
                if request.request_id == request_id:
                    request.pr_url = pr_url
                    if diff_summary is not None:
                        request.diff_summary = diff_summary
                    break
            self._save_all(requests)

    def set_issue_url(self, request_id: str, issue_url: str) -> None:
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            for request in requests:
                if request.request_id == request_id:
                    request.issue_url = issue_url
                    break
            self._save_all(requests)

    def set_message_id(self, request_id: str, message_id: int) -> None:
        """Store the Telegram message_id for later editing."""
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            for r in requests:
                if r.request_id == request_id:
                    r.message_id = message_id
                    break
            self._save_all(requests)

    def revert_to_pending(self, request_id: str) -> None:
        """Revert an APPROVED request back to PENDING (rollback on PR failure)."""
        with jsonl_file_lock(self._path), self._lock:
            requests = self._load_all()
            for r in requests:
                if r.request_id == request_id:
                    r.status = ApprovalStatus.PENDING
                    r.resolved_at = None
                    r.resolved_by = None
                    break
            self._save_all(requests)

    def _load_all(self) -> list[ApprovalRequest]:
        if not self._path.exists():
            return []
        projection = read_json_projection(self._path)
        projection_path = Path(str(self._path) + ".index.json")
        if (
            projection
            and projection_path.exists()
            and projection_path.stat().st_mtime >= self._path.stat().st_mtime
        ):
            return [ApprovalRequest(**record) for record in projection.values()]
        requests: list[ApprovalRequest] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    requests.append(ApprovalRequest(**json.loads(line)))
                except Exception:
                    logger.warning("Skipping malformed approval record")
        write_json_projection(
            self._path,
            key_field="request_id",
            records=[request.model_dump(mode="json") for request in requests],
        )
        return requests

    def _save_all(self, requests: list[ApprovalRequest]) -> None:
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl
        atomic_rewrite_jsonl(self._path, requests)
        write_json_projection(
            self._path,
            key_field="request_id",
            records=[request.model_dump(mode="json") for request in requests],
        )
