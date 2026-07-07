# skills/proposal_ledger.py
"""ProposalLedger — append-only JSONL store for unified proposal records.

Every parameter/structural/discovery/portfolio proposal flows through this
ledger so the weekly LearningCycle can ask:
  - what proposals were produced?
  - which were evaluated, accepted, deployed?
  - what live outcomes did they produce?

Storage shape: one JSONL line per event in `proposal_ledger.jsonl`. Each line is
``{"type": "candidate"|"evaluation"|"outcome", "payload": {...}}``. Reading
groups events by ``proposal_id`` into ``ProposalRecord``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from trading_assistant.schemas.proposal_ledger import (
    ProposalCandidate,
    ProposalEvaluation,
    ProposalKind,
    ProposalOutcome,
    ProposalRecord,
    ProposalSource,
)

logger = logging.getLogger(__name__)


def make_proposal_id(
    source: ProposalSource,
    bot_id: str,
    kind: ProposalKind,
    title: str,
    proposed_at: datetime | None = None,
    strategy_id: str = "",
    link_key: str = "",
) -> str:
    """Deterministic 16-char proposal_id (sha256 prefix).

    Same (source, bot_id, kind, normalized title, day) → same id, so the
    ledger naturally deduplicates re-runs of the same handler within a day.
    """
    when = proposed_at or datetime.now(timezone.utc)
    raw = "|".join([
        source.value,
        bot_id,
        kind.value,
        strategy_id.strip(),
        link_key.strip(),
        title.strip().lower(),
        when.date().isoformat(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ProposalLedger:
    """JSONL-backed append-only ledger of proposals, evaluations, and outcomes."""

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._path = store_dir / "proposal_ledger.jsonl"
        self._lock = threading.Lock()
        # In-memory candidate-ID cache; lazy-loaded on first use, kept in sync
        # with appends. Single-process orchestrator → no cross-process invalidation
        # required. Reset to None to force re-scan on next access.
        self._candidate_ids: set[str] | None = None

    def _load_candidate_ids(self) -> set[str]:
        if self._candidate_ids is None:
            self._candidate_ids = self._existing_candidate_ids()
        return self._candidate_ids

    def record_candidate(self, candidate: ProposalCandidate) -> bool:
        """Append a candidate; returns False if a candidate with the same id exists."""
        with self._lock:
            ids = self._load_candidate_ids()
            if candidate.proposal_id in ids:
                return False
            self._append({"type": "candidate", "payload": candidate.model_dump(mode="json")})
            ids.add(candidate.proposal_id)
        self._refresh_performance_learning_projection()
        return True

    def record_evaluation(
        self, proposal_id: str, evaluation: ProposalEvaluation,
    ) -> bool:
        """Append an evaluation event for an existing candidate."""
        with self._lock:
            if proposal_id not in self._load_candidate_ids():
                return False
            payload = evaluation.model_dump(mode="json")
            payload["proposal_id"] = proposal_id  # ensure consistency
            self._append({"type": "evaluation", "payload": payload})
        self._refresh_performance_learning_projection()
        return True

    def record_outcome(self, proposal_id: str, outcome: ProposalOutcome) -> bool:
        """Append an outcome event for an existing candidate."""
        with self._lock:
            if proposal_id not in self._load_candidate_ids():
                return False
            payload = outcome.model_dump(mode="json")
            payload["proposal_id"] = proposal_id
            self._append({"type": "outcome", "payload": payload})
        self._refresh_performance_learning_projection()
        return True

    def get_by_id(self, proposal_id: str) -> Optional[ProposalRecord]:
        for rec in self._iter_records():
            if rec.candidate.proposal_id == proposal_id:
                return rec
        return None

    def list_by_bot(
        self,
        bot_id: str,
        lifecycle_stage: str | None = None,
        kind: ProposalKind | None = None,
    ) -> list[ProposalRecord]:
        out: list[ProposalRecord] = []
        for rec in self._iter_records():
            if rec.candidate.bot_id != bot_id:
                continue
            if lifecycle_stage and rec.candidate.lifecycle_stage != lifecycle_stage:
                continue
            if kind and rec.candidate.kind != kind:
                continue
            out.append(rec)
        return out

    def list_recent(self, days: int = 30) -> list[ProposalRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            rec for rec in self._iter_records()
            if _ensure_aware(rec.candidate.proposed_at) >= cutoff
        ]

    def list_open(self) -> list[ProposalRecord]:
        """Candidates that have not reached a terminal outcome verdict."""
        return [rec for rec in self._iter_records() if not rec.has_terminal_outcome]

    def list_all(self) -> list[ProposalRecord]:
        return list(self._iter_records())

    def _append(self, event: dict) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
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

    def _existing_candidate_ids(self) -> set[str]:
        ids: set[str] = set()
        for event in self._iter_events():
            if event.get("type") == "candidate":
                pid = event.get("payload", {}).get("proposal_id")
                if pid:
                    ids.add(pid)
        return ids

    def _iter_events(self) -> list[dict]:
        if not self._path.exists():
            return []
        events: list[dict] = []
        with open(self._path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d", self._path.name, line_no,
                    )
        return events

    def _iter_records(self) -> list[ProposalRecord]:
        records: dict[str, ProposalRecord] = {}
        for event in self._iter_events():
            etype = event.get("type")
            payload = event.get("payload") or {}
            try:
                if etype == "candidate":
                    candidate = ProposalCandidate(**payload)
                    records.setdefault(
                        candidate.proposal_id,
                        ProposalRecord(candidate=candidate),
                    )
                elif etype == "evaluation":
                    pid = payload.get("proposal_id")
                    if pid and pid in records:
                        records[pid].evaluations.append(ProposalEvaluation(**payload))
                elif etype == "outcome":
                    pid = payload.get("proposal_id")
                    if pid and pid in records:
                        records[pid].outcomes.append(ProposalOutcome(**payload))
            except Exception:  # pragma: no cover - tolerate malformed event rows
                logger.warning("Skipping malformed ledger event: %s", etype, exc_info=True)
        return list(records.values())


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
