"""Finding and ledger prompt evidence sources."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from trading_assistant.analysis.context_sources.common import (
    apply_temporal_window,
    filter_by_bot,
    filter_inactive_strategies,
    safe_jsonl,
)


class FindingMemorySource:
    _QUALITY_RANK = {"high": 3, "medium": 2, "low": 1, "insufficient": 0}

    def __init__(self, memory_dir: Path, *, strategy_registry: object | None = None) -> None:
        self._findings_dir = Path(memory_dir) / "findings"
        self._strategy_registry = strategy_registry

    def load_corrections(
        self,
        bot_id: str = "",
        *,
        max_age_days: int = 90,
        as_of: datetime | None = None,
    ) -> list[dict]:
        entries = safe_jsonl(self._findings_dir / "corrections.jsonl")
        return apply_temporal_window(
            filter_by_bot(entries, bot_id),
            max_age_days=max_age_days,
            now=as_of,
        )

    def load_failure_log(self, bot_id: str = "") -> list[dict]:
        entries = safe_jsonl(self._findings_dir / "failure-log.jsonl")
        entries = filter_by_bot(entries, bot_id)
        entries = filter_inactive_strategies(entries, self._strategy_registry)
        return apply_temporal_window(entries)

    def load_rejected_suggestions(self) -> list[dict]:
        entries = [
            rec for rec in safe_jsonl(self._findings_dir / "suggestions.jsonl")
            if rec.get("status") == "rejected"
        ]
        entries = filter_inactive_strategies(entries, self._strategy_registry)
        return apply_temporal_window(entries)

    def load_outcome_measurements(
        self,
        min_quality: str = "medium",
    ) -> tuple[list[dict], list[dict]]:
        entries = safe_jsonl(self._findings_dir / "outcomes.jsonl")
        if not entries:
            return [], []
        seen: dict[str | int, dict] = {}
        for entry in entries:
            seen[entry.get("suggestion_id") or id(entry)] = entry
        min_rank = self._QUALITY_RANK.get(min_quality.lower(), 2)
        reliable: list[dict] = []
        low_quality: list[dict] = []
        for entry in seen.values():
            entry.setdefault("outcome_source", "early_warning")
            quality = (entry.get("measurement_quality") or "").lower()
            target = (
                reliable
                if not quality or self._QUALITY_RANK.get(quality, 2) >= min_rank
                else low_quality
            )
            target.append(entry)
        reliable = filter_inactive_strategies(reliable, self._strategy_registry)
        low_quality = filter_inactive_strategies(low_quality, self._strategy_registry)
        return apply_temporal_window(reliable), apply_temporal_window(low_quality)

    def load_monthly_outcomes(
        self,
        bot_id: str = "",
        days: int = 365,
        max_entries: int = 30,
    ) -> list[dict]:
        entries = safe_jsonl(self._findings_dir / "monthly_outcomes.jsonl")
        if bot_id:
            entries = [entry for entry in entries if entry.get("bot_id") == bot_id]
        return apply_temporal_window(entries, max_age_days=days, max_entries=max_entries)

    def load_outcome_priors(self, bot_id: str = "", max_entries: int = 30) -> list[dict]:
        entries = safe_jsonl(self._findings_dir / "outcome_priors.jsonl")
        if bot_id:
            entries = [entry for entry in entries if entry.get("bot_id") == bot_id]
        entries.sort(key=lambda entry: entry.get("updated_at", ""), reverse=True)
        return entries[:max_entries]

    def load_allocation_history(self) -> list[dict]:
        return apply_temporal_window(safe_jsonl(self._findings_dir / "allocation_history.jsonl"))

    def load_recent_work_log_entries(
        self,
        *,
        agent_type: str = "",
        bot_id: str = "",
        limit: int = 10,
    ) -> list[dict]:
        entries = safe_jsonl(self._findings_dir / "loop_run_ledger.jsonl")
        if agent_type:
            aliases = {
                "weekly_analysis": "weekly_summary",
                "monthly_model_review": "monthly_validation",
                "triage": "bug_triage",
            }
            loop_id = aliases.get(agent_type, agent_type)
            entries = [
                entry for entry in entries
                if entry.get("loop_id") in {loop_id, agent_type}
                or entry.get("job_key") in {loop_id, agent_type}
            ]
        if bot_id:
            entries = [
                entry for entry in entries
                if not entry.get("bot_id") or entry.get("bot_id") == bot_id
            ]
        entries.sort(
            key=lambda entry: (
                entry.get("completed_at")
                or entry.get("scheduled_for")
                or entry.get("generated_at")
                or ""
            ),
            reverse=True,
        )
        return [
            {
                "loop_run_id": entry.get("loop_run_id", ""),
                "loop_id": entry.get("loop_id", ""),
                "status": entry.get("status", ""),
                "scheduled_for": entry.get("scheduled_for", ""),
                "summary": entry.get("summary", ""),
                "blocking_reasons": entry.get("blocking_reasons", [])[:3],
                "evidence_paths": entry.get("evidence_paths", [])[:5],
                "approval_packet_paths": entry.get("approval_packet_paths", [])[:3],
            }
            for entry in entries[:limit]
        ]
