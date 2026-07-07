"""Phase 0 event schemas — Pydantic models for trade instrumentation.

These define the data contracts between VPS bots, the relay, and the orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from trading_assistant.schemas.assistant_lineage import AssistantLineage, assistant_lineage_from_fields
from trading_assistant.schemas.canonical_envelope import CANONICAL_ENVELOPE_PAYLOAD_FIELDS


_CRYPTO_STRATEGY_ID_ALIASES = {
    "momentum": "MomentumPullback_M15",
    "trend": "InstitutionalAnchor_H1",
    "breakout": "VolumeProfileBreakout_M30",
}


def normalize_strategy_id(bot_id: str, strategy_id: object) -> str:
    """Map crypto_trader internal strategy keys to assistant profile IDs."""
    value = str(strategy_id or "")
    if bot_id == "crypto_trader":
        return _CRYPTO_STRATEGY_ID_ALIASES.get(value, value)
    return value


def _has_value(value: object) -> bool:
    return value not in (None, "")


def _copy_identity_fields(target: dict, source: dict) -> None:
    for key in CANONICAL_ENVELOPE_PAYLOAD_FIELDS:
        value = source.get(key)
        if _has_value(value) and not _has_value(target.get(key)):
            target[key] = value


def _first_identity_value(data: dict, key: str) -> object:
    value = data.get(key)
    if _has_value(value):
        return value
    for container_key in ("metadata", "event_metadata", "lineage"):
        container = data.get(container_key)
        if isinstance(container, dict):
            value = container.get(key)
            if _has_value(value):
                return value
    return ""


def _copy_metadata_identity(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    source = dict(data)
    payload = source.get("payload")
    if isinstance(payload, str):
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError:
            parsed_payload = None
        if isinstance(parsed_payload, dict):
            payload = parsed_payload
    if isinstance(payload, dict):
        normalized = dict(payload)
        _copy_identity_fields(normalized, source)
    else:
        normalized = dict(source)

    for container_key in ("metadata", "event_metadata", "lineage"):
        container = normalized.get(container_key)
        if isinstance(container, dict):
            _copy_identity_fields(normalized, container)

    bot_id = _first_identity_value(normalized, "bot_id")
    if _has_value(bot_id) and not _has_value(normalized.get("bot_id")):
        normalized["bot_id"] = bot_id

    assistant_strategy_id = _first_identity_value(normalized, "assistant_strategy_id")
    source_strategy_id = _first_identity_value(normalized, "strategy_id")
    if _has_value(assistant_strategy_id):
        normalized["assistant_strategy_id"] = str(assistant_strategy_id)
        normalized["strategy_id"] = str(assistant_strategy_id)
    elif _has_value(source_strategy_id):
        normalized["strategy_id"] = str(source_strategy_id)

    if not _has_value(normalized.get("pair")) and _has_value(normalized.get("symbol")):
        normalized["pair"] = normalized["symbol"]

    if normalized.get("strategy_id"):
        normalized["strategy_id"] = normalize_strategy_id(
            str(normalized.get("bot_id", "")), normalized["strategy_id"],
        )
    return normalized


def _normalize_strategy_keyed_dict(bot_id: str, value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {
        normalize_strategy_id(bot_id, key): item
        for key, item in value.items()
    }


def _merge_assistant_lineage(current: AssistantLineage, derived: AssistantLineage) -> AssistantLineage:
    if not current.has_any():
        return derived
    return AssistantLineage.model_validate({
        **derived.model_dump(),
        **current.model_dump(),
        "weekly_signal_ids": [*derived.weekly_signal_ids, *current.weekly_signal_ids],
        "proposal_ids": [*derived.proposal_ids, *current.proposal_ids],
        "suggestion_ids": [*derived.suggestion_ids, *current.suggestion_ids],
        "hypothesis_ids": [*derived.hypothesis_ids, *current.hypothesis_ids],
        "strategy_change_record_ids": [
            *derived.strategy_change_record_ids,
            *current.strategy_change_record_ids,
        ],
    })


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING = "TRAILING"
    TIMEOUT = "TIMEOUT"
    MANUAL = "MANUAL"
    FUNDING_ADVERSE = "FUNDING_ADVERSE"
    BREAKEVEN = "BREAKEVEN"
    TIME_STOP = "TIME_STOP"
    INVALIDATION = "INVALIDATION"


class EventMetadata(BaseModel):
    """Attached to every event for traceability and clock alignment."""

    bot_id: str
    exchange_timestamp: datetime
    local_timestamp: datetime
    data_source_id: str = ""
    data_source: str = ""
    event_type: str
    payload_key: str = ""
    schema_version: str = ""
    priority: str | int | None = None
    payload_hash: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    account_alias: str = ""
    strategy_id: str = ""
    assistant_strategy_id: str = ""
    logical_event_id: str = ""
    deployment_id: str = ""
    config_version: str = ""
    code_sha: str = ""
    event_ref: str = ""
    source: dict[str, Any] | str | None = None
    source_stream: str = ""
    lineage: dict[str, Any] = Field(default_factory=dict)
    bar_id: Optional[str] = None
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])

    @model_validator(mode="before")
    @classmethod
    def _from_reference_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("data_source_id") and normalized.get("data_source"):
            normalized["data_source_id"] = normalized["data_source"]
        if not normalized.get("assistant_strategy_id"):
            assistant_strategy_id = _first_identity_value(normalized, "assistant_strategy_id")
            if _has_value(assistant_strategy_id):
                normalized["assistant_strategy_id"] = assistant_strategy_id
        if normalized.get("strategy_id"):
            normalized["strategy_id"] = normalize_strategy_id(
                str(normalized.get("bot_id", "")), normalized["strategy_id"],
            )
        return normalized

    @computed_field  # type: ignore[prop-decorator]
    @property
    def event_id(self) -> str:
        raw = f"{self.bot_id}|{self.exchange_timestamp.isoformat()}|{self.event_type}|{self.payload_key}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clock_skew_ms(self) -> int:
        delta = self.exchange_timestamp - self.local_timestamp
        return int(delta.total_seconds() * 1000)


class MarketSnapshot(BaseModel):
    snapshot_id: str = ""
    symbol: str = ""
    timestamp: Optional[datetime] = None
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread_bps: float = 0.0
    last_trade_price: float = 0.0
    volume_1m: float = 0.0
    atr_14: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0


class AfterCostStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNAVAILABLE = "unavailable"


class AfterCostOutcome(BaseModel):
    gross_pnl: float = 0.0
    fees: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    slippage_bps: float = 0.0
    spread_cost: float = 0.0
    funding_or_borrow: float = 0.0
    net_pnl: float = 0.0
    realized_net_r: float | None = None
    cost_model_version: str = ""
    execution_model_version: str = ""
    status: AfterCostStatus = AfterCostStatus.UNAVAILABLE
    net_pnl_source: str = ""


class TradeEvent(BaseModel):
    """A completed trade emitted by a bot."""

    model_config = {"extra": "ignore"}

    trade_id: str
    bot_id: str
    strategy_id: str = ""  # identifies strategy within multi-strategy bots
    assistant_strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    account_alias: str = ""
    schema_version: str = ""
    priority: str | int | None = None
    payload_hash: str = ""
    logical_event_id: str = ""
    revision: int | None = None
    event_ref: str = ""
    payload_key: str = ""
    data_source: str = ""
    data_source_id: str = ""
    source: dict[str, Any] | str | None = None
    source_stream: str = ""
    lineage: dict[str, Any] | None = None
    pair: str
    event_metadata: Optional[EventMetadata] = None
    market_snapshot: Optional[MarketSnapshot] = None

    side: str  # LONG | SHORT
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    gross_pnl: float | None = None
    net_pnl: float | None = None
    net_pnl_source: str = ""
    after_cost_status: str = ""
    after_cost_outcome: AfterCostOutcome | None = None
    cost_model_version: str = ""
    execution_model_version: str = ""

    entry_signal: str = ""
    entry_signal_strength: float = 0.0
    exit_reason: str = ""
    market_regime: str = ""
    active_filters: list[str] = []
    blocked_by: Optional[str] = None

    atr_at_entry: float = 0.0
    volume_24h: float = 0.0
    spread_at_entry: float = 0.0
    funding_rate: float = 0.0
    open_interest_delta: float = 0.0

    process_quality_score: int = 100
    root_causes: list[str] = []
    evidence_refs: list[str] = []

    signal_factors: list[dict] | None = None
    post_exit_1h_price: float | None = None
    post_exit_4h_price: float | None = None

    # Intra-trade excursion tracking (populated by bots that implement bar-by-bar MFE/MAE)
    mfe_price: float | None = None
    mae_price: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    exit_efficiency: float | None = None  # actual_pnl_pct / mfe_pct

    # 1.5: momentum_nq_01 per-bar signal component values
    signal_evolution: list[dict] | None = None
    # 2.6: momentum_nq_01 order fill details
    entry_fill_details: dict | None = None
    exit_fill_details: dict | None = None

    # Cross-repo execution join keys from canonical bot envelopes
    entry_decision_id: str = ""
    exit_decision_id: str = ""
    entry_signal_id: str = ""
    entry_bar_id: str = ""
    exit_bar_id: str = ""
    entry_order_ids: list[str] = Field(default_factory=list)
    exit_order_ids: list[str] = Field(default_factory=list)
    order_ids: list[str] = Field(default_factory=list)
    entry_fill_ids: list[str] = Field(default_factory=list)
    exit_fill_ids: list[str] = Field(default_factory=list)
    fill_ids: list[str] = Field(default_factory=list)
    entry_order_event_refs: list[str] = Field(default_factory=list)
    exit_order_event_refs: list[str] = Field(default_factory=list)
    client_order_ids: list[str] = Field(default_factory=list)
    exchange_order_ids: list[str] = Field(default_factory=list)
    intent_id: str = ""
    idempotency_key: str = ""
    order_id: str = ""
    entry_order_id: str = ""
    exit_order_id: str = ""
    client_order_id: str = ""
    broker_order_id: str = ""
    original_order_id: str = ""
    oms_order_id: str = ""
    kis_order_id: str = ""
    kis_order_date: str = ""
    kis_exec_id: str = ""
    decision_id: str = ""
    decision_ref: dict[str, Any] | None = None
    action_ref: dict[str, Any] | None = None
    provisional_order_ref: str = ""
    portfolio_decision_ref: dict[str, Any] | None = None
    portfolio_rule_event_id: str = ""
    risk_decision_id: str = ""
    artifact_hash: str = ""
    source_artifact_hash: str = ""
    source_fingerprint: str = ""
    candidate_hash: str = ""
    resource_plan_hash: str = ""
    kis_resource_plan_hash: str = ""
    portfolio_policy_hash: str = ""
    state_hash: str = ""
    plan_hash: str = ""
    snapshot_id: str = ""
    runtime_join: dict[str, Any] | None = None
    join_completeness: dict[str, Any] | None = None

    # stock_trader execution quality fields
    fees_paid: float = 0.0
    tax: float = 0.0
    spread_cost: float = 0.0
    borrow_cost: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    entry_latency_ms: float = 0.0

    # stock_trader session/drawdown context
    session_type: str = ""  # e.g. "regular", "pre_market", "extended"
    drawdown_pct: float = 0.0

    # stock_trader signal tracing (links trade back to originating signal)
    signal_id: str = ""

    # stock_trader filter detail (passed_filters = filters the signal cleared;
    # distinct from active_filters which lists all filters that were active)
    passed_filters: list[str] | None = None
    filter_decisions: list[dict] | None = None

    # Macro regime context (from portfolio-level HMM classifier)
    macro_regime: str = ""  # G/R/S/D active at trade time
    stress_level_at_entry: float = 0.0  # P(stress) at trade time

    # Execution pipeline timing {signal_detected_at, intent_created_at, risk_checked_at, order_submitted_at, fill_received_at}
    execution_timestamps: dict | None = None
    # Position sizing decision context {target_risk_pct, account_equity, volatility_basis, sizing_model, unit_risk_usd, ...multipliers}
    sizing_inputs: dict | None = None
    # Full dict of active strategy parameter values at trade execution time
    strategy_params_at_entry: dict | None = None
    # Portfolio state at entry {exposure, direction, correlated_positions}
    portfolio_state_at_entry: dict | None = None
    # Comprehensive market condition dict from bot (supplements atr_at_entry, volume_24h, etc.)
    market_conditions_at_entry: dict | None = None

    # Crypto perpetual fields (populated by bots that trade leveraged perpetuals)
    notional_usd: float = 0.0
    price_pnl_gross: float = 0.0
    total_fees: float = 0.0
    price_pnl_after_funding: float = 0.0
    realized_pnl_net: float = 0.0
    r_multiple: float | None = None
    realized_r_net: float | None = None
    geometric_r: float | None = None
    commission: float = 0.0
    slippage_pct: float = 0.0
    latency_ms: float = 0.0
    liquidity: str = ""
    funding_paid: float = 0.0  # cumulative funding cost during trade hold
    setup_grade: str = ""  # A/B/C or A+/A/B from bot's confluence scoring
    confluences: list[str] = []  # list of confluence factors that fired at entry
    bias_direction: str = ""  # bullish/bearish/neutral from higher-TF analysis
    entry_method: str = ""  # on_close/on_breakout/limit_entry

    # Post-exit move percentages (backfilled by bot, supplements post_exit_1h_price/4h_price)
    post_exit_1h_move_pct: float | None = None
    post_exit_4h_move_pct: float | None = None
    post_exit_backfill_status: str = ""  # pending | partial | complete

    # Deployment / experiment lineage (optional — populated when a change is in flight)
    deployment_id: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    parameter_set_id: str | None = None
    proposal_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_id: str = ""
    suggestion_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str = ""
    hypothesis_ids: list[str] = Field(default_factory=list)
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    strategy_version: str | None = None
    config_version: str | None = None
    signal_generation_version: str | None = None
    code_sha: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_reference_payload(cls, data: Any) -> Any:
        data = _copy_metadata_identity(data)
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("trade_id"):
            normalized["trade_id"] = (
                normalized.get("logical_event_id")
                or normalized.get("event_id")
                or ""
            )
        market_context = normalized.get("market_context")
        if isinstance(market_context, dict):
            if not normalized.get("bias_direction") and market_context.get("bias_direction"):
                normalized["bias_direction"] = market_context["bias_direction"]
            if not normalized.get("setup_grade") and market_context.get("setup_grade"):
                normalized["setup_grade"] = market_context["setup_grade"]
            if not normalized.get("confluences") and market_context.get("setup_confluences"):
                normalized["confluences"] = market_context["setup_confluences"]
        return normalized

    @model_validator(mode="after")
    def _derive_after_cost_outcome(self) -> "TradeEvent":
        gross = self.gross_pnl
        if gross is None:
            gross = self.price_pnl_gross if self.price_pnl_gross else self.pnl
            self.gross_pnl = gross

        explicit_net = self.net_pnl is not None
        fees = float(self.total_fees or self.fees_paid or 0.0)
        commission = float(self.commission or 0.0)
        tax = float(self.tax or 0.0)
        spread_cost = float(self.spread_cost or 0.0)
        slippage_bps = float(self.entry_slippage_bps or 0.0) + float(self.exit_slippage_bps or 0.0)
        if not slippage_bps and self.slippage_pct:
            slippage_bps = float(self.slippage_pct) * 10_000.0
        funding_or_borrow = float(self.funding_paid or self.borrow_cost or 0.0)

        if explicit_net:
            net = float(self.net_pnl or 0.0)
            source = self.net_pnl_source or "observed_net_pnl"
            status = AfterCostStatus.OBSERVED
        elif self.realized_pnl_net:
            net = float(self.realized_pnl_net)
            source = self.net_pnl_source or "observed_realized_pnl_net"
            status = AfterCostStatus.OBSERVED
        elif fees or commission or tax or spread_cost or slippage_bps or funding_or_borrow:
            net = float(self.pnl) - fees - commission - tax - spread_cost - funding_or_borrow
            source = self.net_pnl_source or "inferred_from_cost_fields"
            status = AfterCostStatus.INFERRED
        else:
            net = float(self.pnl)
            source = self.net_pnl_source or "gross_as_net_fallback"
            status = AfterCostStatus.UNAVAILABLE

        self.net_pnl = net
        self.net_pnl_source = source
        self.after_cost_status = self.after_cost_status or status.value
        self.assistant_lineage = _merge_assistant_lineage(
            self.assistant_lineage,
            assistant_lineage_from_fields(
                proposal_ids=[self.proposal_id, *self.proposal_ids],
                suggestion_ids=[self.suggestion_id, *self.suggestion_ids],
                hypothesis_ids=[self.hypothesis_id, *self.hypothesis_ids],
                experiment_id=self.experiment_id or "",
                variant_id=self.variant_id or "",
                parameter_set_id=self.parameter_set_id or "",
                deployment_id=self.deployment_id or "",
            ),
        )
        if self.after_cost_outcome is None:
            self.after_cost_outcome = AfterCostOutcome(
                gross_pnl=float(gross or 0.0),
                fees=fees,
                commission=commission,
                tax=tax,
                slippage_bps=slippage_bps,
                spread_cost=spread_cost,
                funding_or_borrow=funding_or_borrow,
                net_pnl=net,
                realized_net_r=self.realized_r_net or self.r_multiple,
                cost_model_version=self.cost_model_version,
                execution_model_version=self.execution_model_version,
                status=status,
                net_pnl_source=source,
            )
        return self


class MissedOpportunityEvent(BaseModel):
    model_config = {"extra": "ignore"}

    event_metadata: Optional[EventMetadata] = None
    market_snapshot: Optional[MarketSnapshot] = None
    bot_id: str
    strategy_id: str = ""  # identifies strategy within multi-strategy bots
    assistant_strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    account_alias: str = ""
    schema_version: str = ""
    priority: str | int | None = None
    payload_hash: str = ""
    opportunity_id: str = ""
    logical_event_id: str = ""
    revision: int = 0
    supersedes_event_id: str = ""
    event_ref: str = ""
    payload_key: str = ""
    data_source: str = ""
    data_source_id: str = ""
    source: dict[str, Any] | str | None = None
    source_stream: str = ""
    lineage: dict[str, Any] | None = None
    pair: str
    symbol: str = ""
    timeframe: str = ""
    bar_id: str = ""
    signal: str
    signal_strength: float = 0.0
    signal_id: str = ""
    decision_id: str = ""
    portfolio_rule_event_id: str = ""
    risk_decision_id: str = ""
    blocking_rule_type: str = ""
    blocked_scope: str = ""
    resource_conflict_type: str = ""
    blocked_by: str = ""
    hypothetical_entry: float = 0.0
    outcome_1h: Optional[float] = None
    outcome_4h: Optional[float] = None
    outcome_24h: Optional[float] = None
    would_have_hit_tp: Optional[bool] = None
    would_have_hit_sl: Optional[bool] = None
    confidence: float = 0.0
    assumption_tags: list[str] = []
    margin_pct: float | None = None  # how close to filter threshold (requires bot B4)

    # stock_trader extras
    block_reason: str = ""  # freetext explanation (vs blocked_by = filter name)
    backfill_status: str = ""  # e.g. "completed", "pending", "failed"
    simulation_confidence: float = 0.0  # counterfactual sim confidence (vs confidence = signal confidence)

    # Deployment / experiment lineage (optional — populated when a change is in flight)
    deployment_id: str | None = None
    experiment_id: str | None = None
    variant_id: str | None = None
    parameter_set_id: str | None = None
    proposal_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_id: str = ""
    suggestion_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str = ""
    hypothesis_ids: list[str] = Field(default_factory=list)
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    strategy_version: str | None = None
    config_version: str | None = None
    signal_generation_version: str | None = None
    code_sha: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_reference_payload(cls, data: Any) -> Any:
        return _copy_metadata_identity(data)

    @model_validator(mode="after")
    def _derive_assistant_lineage(self) -> "MissedOpportunityEvent":
        self.assistant_lineage = _merge_assistant_lineage(
            self.assistant_lineage,
            assistant_lineage_from_fields(
                proposal_ids=[self.proposal_id, *self.proposal_ids],
                suggestion_ids=[self.suggestion_id, *self.suggestion_ids],
                hypothesis_ids=[self.hypothesis_id, *self.hypothesis_ids],
                experiment_id=self.experiment_id or "",
                variant_id=self.variant_id or "",
                parameter_set_id=self.parameter_set_id or "",
                deployment_id=self.deployment_id or "",
            ),
        )
        return self


class DailySnapshot(BaseModel):
    date: str  # YYYY-MM-DD
    bot_id: str
    family_id: str = ""
    portfolio_id: str = ""
    account_alias: str = ""
    schema_version: str = ""
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_rolling_30d: float = 0.0
    sortino_rolling_30d: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    exposure_pct: float = 0.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    regime_breakdown: dict = {}
    error_count: int = 0
    uptime_pct: float = 100.0
    avg_process_quality: float = 100.0
    root_cause_distribution: dict = {}
    per_strategy_summary: dict = {}
    strategy_summaries: dict = {}
    family_summary: dict | None = None
    portfolio_summary: dict | None = None
    overlay_state_summary: dict | None = None
    experiment_breakdown: dict | None = None  # 1.4: swing_multi_01 per-experiment A/B stats
    lineage_summary: dict | None = None
    lineage_gap: bool = False

    # Macro regime context (from portfolio-level HMM classifier)
    regime_context: dict | None = None  # RegimeContext snapshot (macro_regime, confidence, stress, etc.)
    applied_regime_config: dict | None = None  # Active regime-adjusted portfolio config

    calmar_rolling_30d: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _from_reference_payload(cls, data: Any) -> Any:
        data = _copy_metadata_identity(data)
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        bot_id = str(normalized.get("bot_id", ""))
        if not normalized.get("per_strategy_summary") and isinstance(
            normalized.get("strategy_summaries"), dict,
        ):
            normalized["per_strategy_summary"] = normalized["strategy_summaries"]
        normalized["per_strategy_summary"] = _normalize_strategy_keyed_dict(
            bot_id, normalized.get("per_strategy_summary", {}),
        )
        return normalized


class PipelineFunnelSnapshot(BaseModel):
    """Crypto strategy pipeline funnel snapshot.

    Accepts both the current crypto_trader JSONL shape
    (strategy_id/timestamp/period_start/period_end/funnel/assessment) and the
    flatter assistant-side shape used by earlier planning notes.
    """

    model_config = {"extra": "ignore"}

    event_metadata: Optional[EventMetadata] = None
    metadata: dict[str, Any] | None = None
    bot_id: str = ""
    strategy_id: str = ""
    date: str = ""
    timestamp: str = ""
    period_start: str = ""
    period_end: str = ""
    funnel: dict[str, Any] = Field(default_factory=dict)
    signals_generated: int = 0
    setups_qualified: int = 0
    confirmations_passed: int = 0
    entries_taken: int = 0
    wins: int = 0
    losses: int = 0
    per_strategy_breakdown: dict[str, Any] = Field(default_factory=dict)
    per_symbol_breakdown: dict[str, Any] = Field(default_factory=dict)
    assessment: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_reference_payload(cls, data: Any) -> Any:
        return _copy_metadata_identity(data)


class HealthReportSnapshot(BaseModel):
    """Crypto bot health report snapshot."""

    model_config = {"extra": "ignore"}

    event_metadata: Optional[EventMetadata] = None
    metadata: dict[str, Any] | None = None
    bot_id: str = ""
    timestamp: str = ""
    report: dict[str, Any] = Field(default_factory=dict)
    uptime_pct: float = 0.0
    last_event_age_sec: float = 0.0
    queue_depth: int = 0
    funding_drift_per_symbol: dict[str, Any] = Field(default_factory=dict)
    websocket_disconnects_24h: int = 0
    error_count_24h: int = 0
    severity: str = ""
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_reference_payload(cls, data: Any) -> Any:
        return _copy_metadata_identity(data)


class RegimeTransitionEvent(BaseModel):
    """Emitted when the macro regime classifier changes state (G/R/S/D)."""

    bot_id: str
    event_metadata: Optional[EventMetadata] = None
    from_regime: str  # e.g. "G"
    to_regime: str  # e.g. "S"
    regime_confidence: float = 0.0  # confidence in new regime
    stress_level: float = 0.0  # stress at transition (observational, 41% FPR)
    timestamp: Optional[datetime] = None
