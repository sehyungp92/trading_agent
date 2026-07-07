# skills/structural_experiment_tracker.py
"""StructuralExperimentTracker — JSONL-backed tracker for structural experiments.

Records experiments with acceptance criteria, manages lifecycle
(proposed → active → passed/failed/abandoned), and computes track records.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.structural_experiment import (
    ExperimentRecord,
    ExperimentStatus,
)

logger = logging.getLogger(__name__)


class StructuralExperimentTracker:
    def __init__(self, store_dir: Path) -> None:
        self._path = store_dir / "structural_experiments.jsonl"

    def _load_all(self) -> list[ExperimentRecord]:
        if not self._path.exists():
            return []
        records: list[ExperimentRecord] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(ExperimentRecord(**json.loads(line)))
                except Exception:
                    pass
        return records

    def _save_all(self, records: list[ExperimentRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.model_dump(mode="json"), default=str) + "\n")

    def record_experiment(self, experiment: ExperimentRecord) -> bool:
        """Record a new experiment. Returns False if ID already exists."""
        records = self._load_all()
        if any(r.experiment_id == experiment.experiment_id for r in records):
            return False
        records.append(experiment)
        self._save_all(records)
        return True

    def activate(self, experiment_id: str) -> bool:
        """Set experiment status to ACTIVE with activation timestamp."""
        records = self._load_all()
        for r in records:
            if r.experiment_id == experiment_id:
                if r.status != ExperimentStatus.PROPOSED:
                    return False
                r.status = ExperimentStatus.ACTIVE
                r.activated_at = datetime.now(timezone.utc)
                self._save_all(records)
                return True
        return False

    def resolve(
        self,
        experiment_id: str,
        criteria_met: list[bool],
        actual_values: list[float],
        notes: str = "",
    ) -> bool:
        """Resolve an experiment: PASSED if all criteria met, FAILED otherwise."""
        records = self._load_all()
        for r in records:
            if r.experiment_id == experiment_id:
                if r.status != ExperimentStatus.ACTIVE:
                    return False
                r.criteria_met = criteria_met
                r.actual_values = actual_values
                r.resolution_notes = notes
                r.resolved_at = datetime.now(timezone.utc)
                r.status = (
                    ExperimentStatus.PASSED
                    if criteria_met and all(criteria_met)
                    else ExperimentStatus.FAILED
                )
                self._save_all(records)
                return True
        return False

    def abandon(self, experiment_id: str, reason: str = "") -> bool:
        """Abandon an experiment."""
        records = self._load_all()
        for r in records:
            if r.experiment_id == experiment_id:
                if r.status in (
                    ExperimentStatus.PASSED,
                    ExperimentStatus.FAILED,
                    ExperimentStatus.ABANDONED,
                ):
                    return False
                r.status = ExperimentStatus.ABANDONED
                r.resolution_notes = reason
                r.resolved_at = datetime.now(timezone.utc)
                self._save_all(records)
                return True
        return False

    def get_active_experiments(self) -> list[ExperimentRecord]:
        """Return all active experiments."""
        return [r for r in self._load_all() if r.status == ExperimentStatus.ACTIVE]

    def find_by_suggestion_id(self, suggestion_id: str) -> ExperimentRecord | None:
        """Find an experiment by its linked suggestion_id."""
        for r in self._load_all():
            if r.suggestion_id == suggestion_id:
                return r
        return None

    def get_evaluable_experiments(self) -> list[ExperimentRecord]:
        """Return active experiments past their observation window."""
        return [r for r in self._load_all() if r.is_evaluable]

    def get_failed_experiments(self) -> list[ExperimentRecord]:
        """Return all experiments with FAILED or ABANDONED status."""
        return [
            r for r in self._load_all()
            if r.status in (ExperimentStatus.FAILED, ExperimentStatus.ABANDONED)
        ]

    def compute_track_record(self) -> dict:
        """Compute pass/fail/abandon counts and pass rate."""
        records = self._load_all()
        passed = sum(1 for r in records if r.status == ExperimentStatus.PASSED)
        failed = sum(1 for r in records if r.status == ExperimentStatus.FAILED)
        abandoned = sum(1 for r in records if r.status == ExperimentStatus.ABANDONED)
        active = sum(1 for r in records if r.status == ExperimentStatus.ACTIVE)
        proposed = sum(1 for r in records if r.status == ExperimentStatus.PROPOSED)
        resolved = passed + failed
        pass_rate = passed / resolved if resolved else 0.0
        return {
            "total": len(records),
            "passed": passed,
            "failed": failed,
            "abandoned": abandoned,
            "active": active,
            "proposed": proposed,
            "pass_rate": round(pass_rate, 3),
        }
