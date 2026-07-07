"""JSONL-backed cost tracker for agent invocations."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.cost_tracking import CostRecord, CostSummary

logger = logging.getLogger(__name__)


class CostTracker:
    """Append-only JSONL cost log with query helpers."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, entry: CostRecord) -> None:
        """Append a cost record to the JSONL log."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def _load(self, days: int | None = None) -> list[CostRecord]:
        """Load records, optionally filtered to the last N days."""
        if not self._path.exists():
            return []
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days) if days else None
        )
        records: list[CostRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = CostRecord(**json.loads(line))
                if cutoff and rec.timestamp < cutoff:
                    continue
                records.append(rec)
            except Exception:
                logger.debug("Skipping malformed cost record: %s", line[:100])
        return records

    def summary(self, days: int | None = None) -> CostSummary:
        """Return aggregated cost summary for the given window."""
        records = self._load(days)
        by_provider: dict[str, float] = {}
        by_workflow: dict[str, float] = {}
        total_cost = 0.0
        total_duration = 0
        success_count = 0
        fail_count = 0
        for r in records:
            total_cost += r.cost_usd
            total_duration += r.duration_ms
            if r.success:
                success_count += 1
            else:
                fail_count += 1
            by_provider[r.provider] = by_provider.get(r.provider, 0.0) + r.cost_usd
            if r.workflow:
                by_workflow[r.workflow] = by_workflow.get(r.workflow, 0.0) + r.cost_usd
        return CostSummary(
            total_cost_usd=total_cost,
            total_invocations=len(records),
            successful_invocations=success_count,
            failed_invocations=fail_count,
            total_duration_ms=total_duration,
            by_provider=by_provider,
            by_workflow=by_workflow,
        )

    def by_provider(self, days: int | None = None) -> dict[str, float]:
        """Return cost breakdown by provider."""
        return self.summary(days).by_provider

    def by_workflow(self, days: int | None = None) -> dict[str, float]:
        """Return cost breakdown by workflow."""
        return self.summary(days).by_workflow
