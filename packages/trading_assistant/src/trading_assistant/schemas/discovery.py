# schemas/discovery.py
"""Discovery schemas — novel patterns found by Claude from raw trade data."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TradeReference(BaseModel):
    """Reference to a specific trade that supports a discovery."""
    date: str = ""
    bot_id: str = ""
    trade_id: str = ""
    pnl: float = 0.0
    regime: str = ""
    signal_strength: float = 0.0
    note: str = ""


class Discovery(BaseModel):
    """A novel pattern discovered by the analysis agent."""
    discovery_id: str = ""
    pattern_description: str
    evidence: list[TradeReference] = []
    proposed_root_cause: str = ""  # existing taxonomy or "novel"
    testable_hypothesis: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detector_coverage: str = ""  # which automated detector relates, or "novel"
    bot_id: str = ""
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class StrategyIdea(BaseModel):
    """A novel strategy concept grounded in evidence from discovery analysis."""
    idea_id: str = ""  # deterministic from hash of description
    title: str = ""  # short name (e.g., "Regime-Filtered ORB Reversal")
    description: str = ""  # how the strategy works
    edge_hypothesis: str = ""  # why it should work (grounded in data)
    evidence: list[TradeReference] = []  # supporting data points
    entry_logic: str = ""  # when to enter
    exit_logic: str = ""  # when to exit
    applicable_regimes: list[str] = []  # which market regimes
    applicable_bots: list[str] = []  # which bots could run it
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "proposed"  # proposed → under_review → testing → adopted → retired
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_discoveries: list[str] = []  # discovery_ids that informed this


class DiscoveryReport(BaseModel):
    """Collection of discoveries from a single agent invocation."""
    run_id: str = ""
    date: str = ""
    discoveries: list[Discovery] = []
    strategy_ideas: list[StrategyIdea] = []
    data_scope: str = ""  # e.g., "30d raw trades for bot_x"


class StrategyDiscoveryCluster(BaseModel):
    """Recurring diagnostics-only opportunity cluster for new-strategy ideation."""

    cluster_id: str = ""
    source: str = ""  # missed_opportunity | denominator_snapshot
    bot_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    regime: str = ""
    setup_key: str = ""
    support_count: int = 0
    missed_count: int = 0
    denominator_count: int = 0
    control_count: int = 0
    estimated_after_cost_pnl: float = 0.0
    estimated_after_cost_pnl_source: str = ""
    after_cost_status: str = "diagnostic_estimate"
    evidence: list[TradeReference] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    control_slice: dict[str, Any] = Field(default_factory=dict)
    replay_plan: str = ""
    shadow_plan: str = ""

    @model_validator(mode="after")
    def _derive_cluster_id(self) -> "StrategyDiscoveryCluster":
        self.evidence_paths = _dedupe(self.evidence_paths)
        self.support_count = max(0, self.support_count)
        self.missed_count = max(0, self.missed_count)
        self.denominator_count = max(0, self.denominator_count)
        self.control_count = max(0, self.control_count)
        if not self.cluster_id:
            raw = "|".join([
                self.source,
                self.bot_id,
                self.strategy_id,
                self.symbol,
                self.regime,
                self.setup_key,
                str(self.support_count),
            ])
            self.cluster_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self


class StrategyDiscoveryPacket(BaseModel):
    """Diagnostics-only packet grounding new-strategy discovery proposals."""

    packet_id: str = ""
    run_id: str = ""
    run_month: str = ""
    bot_id: str = ""
    strategy_id: str = ""
    authority: str = "diagnostics_only"
    evidence_authority: str = "diagnostics_only"
    approval_gate_eligible: bool = False
    learning_sufficiency_manifest_path: str = ""
    supported_learning_capabilities: list[str] = Field(default_factory=list)
    blocked_learning_capabilities: list[str] = Field(default_factory=list)
    missed_opportunity_clusters: list[StrategyDiscoveryCluster] = Field(default_factory=list)
    denominator_clusters: list[StrategyDiscoveryCluster] = Field(default_factory=list)
    control_slices: list[dict[str, Any]] = Field(default_factory=list)
    after_cost_estimates: list[dict[str, Any]] = Field(default_factory=list)
    replay_or_shadow_plan: str = ""
    new_strategy_proposal_requirements: list[str] = Field(default_factory=lambda: [
        "cite_recurring_opportunity_cluster",
        "cite_control_slice",
        "cite_after_cost_estimate",
        "provide_replay_or_shadow_plan",
        "remain_diagnostics_only_until_replay_and_approval_ready_bridge_gates_pass",
    ])
    evidence_paths: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    packet_version: str = "strategy_discovery_packet_v1"

    @model_validator(mode="after")
    def _normalize(self) -> "StrategyDiscoveryPacket":
        self.authority = "diagnostics_only"
        self.evidence_authority = "diagnostics_only"
        self.approval_gate_eligible = False
        self.supported_learning_capabilities = _dedupe(self.supported_learning_capabilities)
        self.blocked_learning_capabilities = _dedupe(self.blocked_learning_capabilities)
        self.evidence_paths = _dedupe([
            *self.evidence_paths,
            *([self.learning_sufficiency_manifest_path] if self.learning_sufficiency_manifest_path else []),
            *[
                path
                for cluster in [*self.missed_opportunity_clusters, *self.denominator_clusters]
                for path in cluster.evidence_paths
            ],
        ])
        if not self.control_slices:
            self.control_slices = [
                cluster.control_slice
                for cluster in [*self.missed_opportunity_clusters, *self.denominator_clusters]
                if cluster.control_slice
            ]
        if not self.after_cost_estimates:
            self.after_cost_estimates = [
                {
                    "cluster_id": cluster.cluster_id,
                    "source": cluster.source,
                    "estimated_after_cost_pnl": cluster.estimated_after_cost_pnl,
                    "estimated_after_cost_pnl_source": cluster.estimated_after_cost_pnl_source,
                    "after_cost_status": cluster.after_cost_status,
                }
                for cluster in [*self.missed_opportunity_clusters, *self.denominator_clusters]
            ]
        if not self.replay_or_shadow_plan:
            self.replay_or_shadow_plan = (
                "Replay recurring clusters against held-out windows, then run scheduled "
                "shadow cycles before any approval-ready bridge promotion."
            )
        if not self.packet_id:
            raw = "|".join([
                self.run_id,
                self.run_month,
                self.bot_id,
                self.strategy_id,
                ",".join(cluster.cluster_id for cluster in self.missed_opportunity_clusters),
                ",".join(cluster.cluster_id for cluster in self.denominator_clusters),
            ])
            self.packet_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
