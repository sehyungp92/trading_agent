"""Focused run-recall schemas for provenance-rich prompt context."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class FocusedRecallCard(BaseModel):
    """Compact evidence summary for a prior run or learning-review card."""

    recall_id: str = ""
    run_id: str = ""
    date: str = ""
    workflow: str = ""
    bot_id: str = ""
    strategy_id: str = ""
    reason_for_retrieval: str = ""
    proposal_or_finding_summary: str = ""
    validator_gate_status: str = ""
    approval_status: str = ""
    deployment_status: str = ""
    outcome_status: str = ""
    evidence_paths: list[str] = Field(default_factory=list)
    supersession_notes: str = ""
    contradiction_notes: str = ""
    how_this_matters_now: str = ""
    source: str = "focused_recall"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.recall_id:
            raw = "|".join([
                self.source,
                self.run_id,
                self.workflow,
                self.bot_id,
                self.strategy_id,
                ",".join(self.evidence_paths[:3]),
            ])
            self.recall_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_prompt_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)
