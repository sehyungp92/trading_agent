# skills/suggestion_tracker.py
"""SuggestionTracker — records suggestions, tracks status, and measures outcomes.

Storage: Two JSONL files in store_dir:
  - suggestions.jsonl — one record per suggestion with lifecycle status
  - outcomes.jsonl — measured impacts of implemented suggestions
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.orchestrator.jsonl_store import (
    jsonl_file_lock,
    read_json_projection,
    write_json_projection,
)
from trading_assistant.schemas.suggestion_tracking import (
    SuggestionOutcome,
    SuggestionRecord,
    SuggestionStatus,
)


class SuggestionTracker:
    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._suggestions_path = store_dir / "suggestions.jsonl"
        self._outcomes_path = store_dir / "outcomes.jsonl"
        self._lock = threading.RLock()

    def record(self, suggestion: SuggestionRecord) -> bool:
        """Record a suggestion. Returns False if suggestion_id already exists (dedup)."""
        with jsonl_file_lock(self._suggestions_path), self._lock:
            existing = self._load_suggestions()
            existing_ids = {s.get("suggestion_id") for s in existing}
            if suggestion.suggestion_id in existing_ids:
                return False
            self._store_dir.mkdir(parents=True, exist_ok=True)
            with open(self._suggestions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(suggestion.model_dump(mode="json"), default=str) + "\n")
            existing.append(suggestion.model_dump(mode="json"))
            self._write_suggestion_projection(existing)
            return True

    def reject(self, suggestion_id: str, reason: str = "") -> None:
        self._update_status(suggestion_id, SuggestionStatus.REJECTED, reason=reason)

    def accept(
        self,
        suggestion_id: str,
        approval_request_id: str | None = None,
    ) -> None:
        self._update_status(
            suggestion_id,
            SuggestionStatus.ACCEPTED,
            approval_request_id=approval_request_id,
        )

    def mark_merged(
        self,
        suggestion_id: str,
        pr_url: str | None = None,
        deployment_id: str | None = None,
    ) -> None:
        self._update_status(
            suggestion_id,
            SuggestionStatus.MERGED,
            pr_url=pr_url,
            deployment_id=deployment_id,
        )

    def mark_deployed(
        self,
        suggestion_id: str,
        deployment_id: str | None = None,
    ) -> None:
        self._update_status(
            suggestion_id,
            SuggestionStatus.DEPLOYED,
            deployment_id=deployment_id,
        )

    def mark_measured(
        self,
        suggestion_id: str,
        *,
        source: str = "early_warning",
        outcome_id: str = "",
        strategy_change_record_id: str = "",
        final: bool = True,
    ) -> None:
        """Record measurement provenance and optionally close the lifecycle.

        ``source="early_warning"`` is retained as the legacy default. Monthly
        validation callers should pass ``source="monthly"`` or ``"follow_up"``.
        Set ``final=False`` for lightweight early warnings that should not mark
        material strategy/config changes as finally measured.
        """
        if final:
            self._update_status(
                suggestion_id,
                SuggestionStatus.MEASURED,
                outcome_source=source,
                outcome_id=outcome_id,
                strategy_change_record_id=strategy_change_record_id,
            )
        else:
            self._update_measurement_source(
                suggestion_id,
                source=source,
                outcome_id=outcome_id,
                strategy_change_record_id=strategy_change_record_id,
            )

    def implement(self, suggestion_id: str) -> None:
        """Legacy helper retained for historical tooling.

        .. deprecated::
            Use ``accept()`` followed by ``mark_deployed()`` instead.
        """
        import warnings

        warnings.warn(
            "implement() is deprecated; use accept() + mark_deployed()",
            DeprecationWarning,
            stacklevel=2,
        )
        self._update_status(suggestion_id, SuggestionStatus.IMPLEMENTED)

    def record_outcome(self, outcome: SuggestionOutcome) -> None:
        with jsonl_file_lock(self._outcomes_path), self._lock:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            with open(self._outcomes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome.model_dump(mode="json"), default=str) + "\n")

    def load_all(self) -> list[dict]:
        with jsonl_file_lock(self._suggestions_path):
            return self._load_suggestions()

    def load_outcomes(self) -> list[dict]:
        with jsonl_file_lock(self._outcomes_path):
            return self._read_jsonl(self._outcomes_path)

    def get_rejected(self, bot_id: str | None = None) -> list[dict]:
        suggestions = self.load_all()
        rejected = [s for s in suggestions if s.get("status") == SuggestionStatus.REJECTED.value]
        if bot_id:
            rejected = [s for s in rejected if s.get("bot_id") == bot_id]
        return rejected

    def get_deployed_portfolio_count(self) -> int:
        """Count DEPLOYED suggestions with bot_id='PORTFOLIO'.

        Used to enforce concurrent deployment limit (max 1 portfolio change at a time).
        """
        suggestions = self.load_all()
        return sum(
            1 for s in suggestions
            if s.get("bot_id") == "PORTFOLIO"
            and s.get("status") == SuggestionStatus.DEPLOYED.value
        )

    def get_last_portfolio_proposal_date(self, proposal_type: str = "") -> str | None:
        """Get the date of the most recent portfolio proposal (any status).

        Args:
            proposal_type: Optional filter by category (e.g., 'portfolio_allocation').

        Returns:
            ISO date string of most recent proposal, or None if no history.
        """
        suggestions = self.load_all()
        portfolio = [
            s for s in suggestions
            if s.get("bot_id") == "PORTFOLIO"
        ]
        if proposal_type:
            portfolio = [s for s in portfolio if s.get("category") == proposal_type]
        if not portfolio:
            return None
        # Find most recent by proposed_at, created_at, or timestamp
        dates = []
        for s in portfolio:
            for key in ("proposed_at", "created_at", "timestamp", "accepted_at"):
                val = s.get(key)
                if val:
                    dates.append(val)
                    break
        return max(dates) if dates else None

    def get_recent_by_bot(self, bot_id: str, weeks: int = 4) -> list[dict]:
        """Return non-rejected suggestions for bot_id within time window."""
        return self.get_recent_grouped([bot_id], weeks=weeks).get(bot_id, [])

    def get_recent_grouped(
        self, bot_ids: list[str], weeks: int = 4,
    ) -> dict[str, list[dict]]:
        """Return non-rejected suggestions grouped by bot_id within time window.

        Reads suggestions.jsonl once and groups in memory — replaces N+1 callers
        that loop ``get_recent_by_bot`` per bot.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
        wanted = set(bot_ids)
        result: dict[str, list[dict]] = {bid: [] for bid in wanted}
        for s in self.load_all():
            bid = s.get("bot_id")
            if bid not in wanted:
                continue
            if s.get("status") == SuggestionStatus.REJECTED.value:
                continue
            ts_str = s.get("proposed_at", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            result[bid].append(s)
        return result

    def _update_status(
        self,
        suggestion_id: str,
        status: SuggestionStatus,
        reason: str = "",
        approval_request_id: str | None = None,
        deployment_id: str | None = None,
        pr_url: str | None = None,
        outcome_source: str = "",
        outcome_id: str = "",
        strategy_change_record_id: str = "",
    ) -> None:
        with jsonl_file_lock(self._suggestions_path), self._lock:
            records = self._load_suggestions()
            now = datetime.now(timezone.utc).isoformat()
            for rec in records:
                if rec["suggestion_id"] == suggestion_id:
                    rec["status"] = status.value
                    if status == SuggestionStatus.ACCEPTED:
                        rec["accepted_at"] = now
                    elif status == SuggestionStatus.MERGED:
                        rec["merged_at"] = now
                    elif status == SuggestionStatus.DEPLOYED:
                        rec["deployed_at"] = now
                    elif status == SuggestionStatus.MEASURED:
                        rec["measured_at"] = now
                    if reason:
                        rec["rejection_reason"] = reason
                    if approval_request_id:
                        rec["approval_request_id"] = approval_request_id
                    if deployment_id:
                        rec["deployment_id"] = deployment_id
                    if pr_url:
                        rec["pr_url"] = pr_url
                    if outcome_source:
                        self._stamp_outcome_source(
                            rec,
                            source=outcome_source,
                            outcome_id=outcome_id,
                            strategy_change_record_id=strategy_change_record_id,
                            measured_at=now,
                        )
                    if status in {
                        SuggestionStatus.REJECTED,
                        SuggestionStatus.MEASURED,
                    }:
                        rec["resolved_at"] = now
            self._save_suggestions(records)

    def _update_measurement_source(
        self,
        suggestion_id: str,
        *,
        source: str,
        outcome_id: str = "",
        strategy_change_record_id: str = "",
    ) -> None:
        with jsonl_file_lock(self._suggestions_path), self._lock:
            records = self._load_suggestions()
            now = datetime.now(timezone.utc).isoformat()
            for rec in records:
                if rec["suggestion_id"] == suggestion_id:
                    self._stamp_outcome_source(
                        rec,
                        source=source,
                        outcome_id=outcome_id,
                        strategy_change_record_id=strategy_change_record_id,
                        measured_at=now,
                    )
                    break
            self._save_suggestions(records)

    def _load_suggestions(self) -> list[dict]:
        if not self._suggestions_path.exists():
            return []
        projection = read_json_projection(self._suggestions_path)
        projection_path = Path(str(self._suggestions_path) + ".index.json")
        if (
            projection
            and projection_path.exists()
            and projection_path.stat().st_mtime >= self._suggestions_path.stat().st_mtime
        ):
            return list(projection.values())
        records = self._read_jsonl(self._suggestions_path)
        self._write_suggestion_projection(records)
        return records

    def _save_suggestions(self, records: list[dict]) -> None:
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl

        atomic_rewrite_jsonl(self._suggestions_path, records)
        self._write_suggestion_projection(records)

    def _write_suggestion_projection(self, records: list[dict]) -> None:
        write_json_projection(
            self._suggestions_path,
            key_field="suggestion_id",
            records=records,
        )

    @staticmethod
    def _stamp_outcome_source(
        rec: dict,
        *,
        source: str,
        outcome_id: str = "",
        strategy_change_record_id: str = "",
        measured_at: str = "",
    ) -> None:
        rec["outcome_source"] = source
        if outcome_id:
            rec["monthly_outcome_id"] = outcome_id
        if strategy_change_record_id:
            rec["strategy_change_record_id"] = strategy_change_record_id
        history = rec.get("outcome_source_history")
        if not isinstance(history, list):
            history = []
        duplicate = bool(outcome_id) and any(
            item.get("source") == source
            and item.get("outcome_id") == outcome_id
            and item.get("strategy_change_record_id") == strategy_change_record_id
            for item in history
            if isinstance(item, dict)
        )
        if not duplicate:
            history.append({
                "source": source,
                "outcome_id": outcome_id,
                "strategy_change_record_id": strategy_change_record_id,
                "measured_at": measured_at,
            })
        rec["outcome_source_history"] = history

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        import logging
        _logger = logging.getLogger(__name__)
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    _logger.warning(
                        "Skipping malformed JSON at %s:%d", path.name, line_no,
                    )
                    continue
        return records
