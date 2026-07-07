# skills/reliability_tracker.py
"""ReliabilityTracker — JSONL-backed tracker for bug fix interventions.

Records interventions, detects recurrences, auto-verifies fixes after
observation windows, and computes reliability summaries.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.reliability_learning import (
    BugClass,
    InterventionStatus,
    ReliabilityIntervention,
    ReliabilityScorecard,
    ReliabilitySummary,
)

logger = logging.getLogger(__name__)


class ReliabilityTracker:
    def __init__(self, store_dir: Path) -> None:
        self._path = store_dir / "reliability_interventions.jsonl"

    def _load_all(self) -> list[ReliabilityIntervention]:
        if not self._path.exists():
            return []
        records: list[ReliabilityIntervention] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(ReliabilityIntervention(**json.loads(line)))
                except Exception:
                    pass
        return records

    def _save_all(self, records: list[ReliabilityIntervention]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.model_dump(mode="json"), default=str) + "\n")

    @staticmethod
    def generate_id(bot_id: str, bug_class: str, date: str) -> str:
        """Generate a deterministic intervention ID."""
        raw = f"{bot_id}:{bug_class}:{date}"
        return "rel_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    def record_intervention(self, intervention: ReliabilityIntervention) -> bool:
        """Record a new intervention. Returns False if ID already exists."""
        records = self._load_all()
        if any(r.intervention_id == intervention.intervention_id for r in records):
            return False
        records.append(intervention)
        self._save_all(records)
        return True

    def record_recurrence(
        self, bot_id: str, bug_class: BugClass, error_category: str,
    ) -> str | None:
        """Match an error against open interventions and record recurrence.

        Returns the intervention_id if matched, None otherwise.
        """
        records = self._load_all()
        matched_id = None
        for r in records:
            if (
                r.status == InterventionStatus.OPEN
                and r.bot_id == bot_id
                and r.bug_class == bug_class
                and (not error_category or r.error_category == error_category)
            ):
                r.recurrence_count += 1
                r.last_recurrence_at = datetime.now(timezone.utc)
                r.status = InterventionStatus.RECURRED
                matched_id = r.intervention_id
                break
        if matched_id:
            self._save_all(records)
        return matched_id

    def verify_completed(self) -> list[str]:
        """Auto-verify open interventions past their observation window.

        Interventions with 0 recurrences → VERIFIED.
        Interventions with recurrences stay RECURRED.
        Returns list of verified intervention IDs.
        """
        records = self._load_all()
        verified_ids: list[str] = []
        changed = False
        for r in records:
            if r.status != InterventionStatus.OPEN:
                continue
            if r.is_within_observation:
                continue
            # Past observation window
            if r.recurrence_count == 0:
                r.status = InterventionStatus.VERIFIED
                r.verified_at = datetime.now(timezone.utc)
                verified_ids.append(r.intervention_id)
                changed = True
            else:
                r.status = InterventionStatus.RECURRED
                changed = True
        if changed:
            self._save_all(records)
        return verified_ids

    def compute_summary(self) -> ReliabilitySummary:
        """Compute reliability summary across all bug classes."""
        records = self._load_all()
        if not records:
            return ReliabilitySummary()

        # Group by bug_class
        by_class: dict[str, list[ReliabilityIntervention]] = {}
        for r in records:
            by_class.setdefault(r.bug_class.value, []).append(r)

        scorecards: dict[str, ReliabilityScorecard] = {}
        chronic: list[str] = []
        total_open = 0

        for cls, interventions in by_class.items():
            count = len(interventions)
            verified = sum(1 for i in interventions if i.status == InterventionStatus.VERIFIED)
            total_recurrences = sum(i.recurrence_count for i in interventions)
            recurrence_rate = total_recurrences / count if count else 0.0
            avg_obs = sum(i.observation_window_days for i in interventions) / count if count else 0.0
            open_count = sum(1 for i in interventions if i.status == InterventionStatus.OPEN)
            total_open += open_count

            scorecards[cls] = ReliabilityScorecard(
                bug_class=BugClass(cls),
                intervention_count=count,
                verified_count=verified,
                recurrence_rate=round(recurrence_rate, 3),
                avg_observation_days=round(avg_obs, 1),
            )

            if total_recurrences >= 3:
                chronic.append(cls)

        return ReliabilitySummary(
            scorecards_by_class=scorecards,
            chronic_bug_classes=chronic,
            total_open=total_open,
        )
