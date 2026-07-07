"""Append-only candidate attempt ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trading_assistant_backtest.contract_models import (
    CandidateAttemptRecord,
    CandidateAttemptState,
    CandidateWorkspaceManifest,
    OptimizerStage,
    sanitize_workspace_key,
)


class CandidateAttemptStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CandidateAttemptRecord) -> CandidateAttemptRecord:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    def load(self) -> list[CandidateAttemptRecord]:
        if not self.path.exists():
            return []
        records: list[CandidateAttemptRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(CandidateAttemptRecord.model_validate(json.loads(line)))
        return records

    def claim(
        self,
        *,
        run_id: str,
        candidate_id: str,
        workspace: CandidateWorkspaceManifest,
        manifest_id: str,
        stage: OptimizerStage = OptimizerStage.PHASED_AUTO,
    ) -> CandidateAttemptRecord:
        attempt_number = 1 + max(
            (
                record.attempt_number
                for record in self.load()
                if record.run_id == run_id and record.candidate_id == candidate_id
            ),
            default=0,
        )
        attempt_id = "attempt-" + sanitize_workspace_key(
            f"{run_id}:{candidate_id}:{attempt_number}"
        )
        return self.append(
            CandidateAttemptRecord(
                attempt_id=attempt_id,
                run_id=run_id,
                candidate_id=candidate_id,
                workspace_key=workspace.workspace_key,
                workspace_path=workspace.workspace_path,
                state=CandidateAttemptState.CLAIMED,
                stage=stage,
                attempt_number=attempt_number,
                manifest_id=manifest_id,
            )
        )

    def transition(
        self,
        attempt_id: str,
        state: CandidateAttemptState,
        *,
        reason: str = "",
    ) -> CandidateAttemptRecord:
        latest = {record.attempt_id: record for record in self.load()}.get(attempt_id)
        if latest is None:
            raise ValueError(f"unknown attempt_id: {attempt_id}")
        return self.append(
            latest.model_copy(
                update={
                    "state": state,
                    "reason": reason,
                    "updated_at": datetime.now(UTC),
                }
            )
        )
