"""Monthly validation result schemas."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from trading_assistant.schemas.assistant_lineage import AssistantLineage, assistant_lineage_from_fields
from trading_assistant.schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


class MonthlyValidationStatus(str, Enum):
    KEEP = "keep"
    WATCH = "watch"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    EXPERIMENT = "experiment"
    INSUFFICIENT_DATA = "insufficient_data"
    INSUFFICIENT_LINEAGE = "insufficient_lineage"
    UNSUPPORTED_NO_REPLAY_PLUGIN = "unsupported_no_replay_plugin"
    NO_CHANGE = "no_change"


class GapAttributionCategory(str, Enum):
    UNDER_TRADING = "under_trading"
    OUTLIER_LOSS = "outlier_loss"
    BROAD_DEGRADATION = "broad_degradation"
    EXECUTION_DRIFT = "execution_drift"
    SLIPPAGE_COST_DRIFT = "slippage_cost_drift"
    DATA_GAP = "data_gap"
    REGIME_MISMATCH = "regime_mismatch"
    HARMFUL_ACCEPTED_MUTATION = "harmful_accepted_mutation"
    FILTER_OVERREACH = "filter_overreach"
    ENTRY_SIGNAL_DECAY = "entry_signal_decay"
    EXIT_MISMATCH = "exit_mismatch"
    PORTFOLIO_CORRELATION_CROWDING = "portfolio_correlation_crowding"
    OPPORTUNITY_SCARCITY = "opportunity_scarcity"
    NONE = "none"


class GapAttribution(BaseModel):
    primary_category: GapAttributionCategory = GapAttributionCategory.NONE
    supporting_categories: list[GapAttributionCategory] = Field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    evidence_paths: list[str] = Field(default_factory=list)


class MonthlyValidationResult(BaseModel):
    run_id: str
    run_month: str
    bot_id: str
    strategy_id: str
    status: MonthlyValidationStatus
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    telemetry_manifest_path: str = ""
    learning_sufficiency_manifest_path: str = ""
    learning_sufficiency_status: str = ""
    supported_learning_capabilities: list[str] = Field(default_factory=list)
    blocked_learning_capabilities: list[str] = Field(default_factory=list)
    learning_sufficiency_gate_paths: list[str] = Field(default_factory=list)
    learning_sufficiency_blocking_reasons: list[str] = Field(default_factory=list)
    strategy_discovery_packet_path: str = ""
    market_data_manifest_path: str = ""
    run_manifest_path: str = ""
    artifact_index_path: str = ""
    replay_parity_path: str = ""
    gap_attribution: GapAttribution = GapAttribution()
    monthly_report_path: str = ""
    strategy_change_record_id: str = ""
    blocking_reasons: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    candidate_summary_path: str = ""
    candidate_gate_report_path: str = ""
    approval_packet_paths: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    approval_request_ids: list[str] = Field(default_factory=list)
    selected_candidate_count: int = 0
    rejected_candidate_count: int = 0
    gate_passed_candidate_count: int = 0
    approval_ready_candidate_count: int = 0
    model_review_path: str = ""
    model_review_validation_path: str = ""
    monthly_evidence_verification_paths: list[str] = Field(default_factory=list)
    model_review_valid: bool | None = None
    model_review_issues: list[str] = Field(default_factory=list)
    model_review_provider: str = ""
    model_review_model: str = ""
    model_review_runtime: str = ""
    model_review_cost_usd: float = 0.0
    optimizer_sequence_result_path: str = ""
    optimizer_sequence_status: str = ""
    adopted_candidate_id: str = ""
    optimizer_no_adoption_reason: str = ""
    repair_request_path: str = ""
    repair_required: bool = False
    proposed_strategy_change_record_ids: list[str] = Field(default_factory=list)
    shadow: bool = True
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive_assistant_lineage(self) -> "MonthlyValidationResult":
        derived = assistant_lineage_from_fields(
            proposal_ids=self.proposal_ids,
            strategy_change_record_ids=[
                self.strategy_change_record_id,
                *self.proposed_strategy_change_record_ids,
            ],
        )
        if not self.assistant_lineage.has_any():
            self.assistant_lineage = derived
        else:
            self.assistant_lineage = AssistantLineage.model_validate({
                **derived.model_dump(),
                **self.assistant_lineage.model_dump(),
                "proposal_ids": [*derived.proposal_ids, *self.assistant_lineage.proposal_ids],
                "strategy_change_record_ids": [
                    *derived.strategy_change_record_ids,
                    *self.assistant_lineage.strategy_change_record_ids,
                ],
            })
        return self
