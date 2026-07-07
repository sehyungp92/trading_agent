"""Structured repair requests for blocked monthly validation runs."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MonthlyRepairClassification(str, Enum):
    DATA = "data"
    TELEMETRY = "telemetry"
    ARTIFACT_CONTRACT = "artifact_contract"
    STRATEGY_PLUGIN = "strategy_plugin"
    DATA_CHECKSUM = "data_checksum"
    SCHEMA_VERSION = "schema_version"
    RUNNER_TIMEOUT = "runner_timeout"
    STALE_ARTIFACTS = "stale_artifacts"
    PATH_CONTAINMENT = "path_containment"
    DECISION_PARITY = "decision_parity"
    REPLAY_PARITY = "replay_parity"
    CANDIDATE_GENERATION = "candidate_generation"
    MODEL_REVIEW = "model_review"
    BACKTEST_RUNNER = "backtest_runner"
    UNKNOWN = "unknown"


class MonthlyRepairRequest(BaseModel):
    """Evidence-bound operator action request for incomplete monthly evidence."""

    repair_request_id: str = ""
    run_id: str
    bot_id: str
    strategy_id: str
    run_month: str
    classification: MonthlyRepairClassification = MonthlyRepairClassification.UNKNOWN
    missing_artifact_keys: list[str] = Field(default_factory=list)
    malformed_artifacts: list[str] = Field(default_factory=list)
    blocking_gates: list[str] = Field(default_factory=list)
    owner_component: str = ""
    repair_command_hints: list[str] = Field(default_factory=list)
    retry_eligible: bool = False
    evidence_paths: list[str] = Field(default_factory=list)
    human_summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.repair_request_id:
            raw = "|".join([
                self.run_id,
                self.classification.value,
                ",".join(self.blocking_gates),
                ",".join(self.missing_artifact_keys),
                ",".join(self.malformed_artifacts),
            ])
            self.repair_request_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
