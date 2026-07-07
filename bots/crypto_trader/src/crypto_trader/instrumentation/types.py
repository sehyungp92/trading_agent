"""Instrumentation event types for trading assistant integration."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from crypto_trader.instrumentation.lineage import stable_hash
from crypto_trader.instrumentation.strategy_ids import assistant_strategy_id


ASSISTANT_EVENT_SCHEMA_VERSION = "assistant_event_v1"


def _event_payload_hash(payload: dict[str, Any]) -> str:
    return stable_hash(payload, length=32)


_CORE_CONTEXT_FIELDS = (
    "bot_id",
    "family_id",
    "portfolio_id",
    "account_alias",
    "strategy_id",
    "assistant_strategy_id",
    "exchange_timestamp",
    "local_timestamp",
    "deployment_id",
    "config_version",
    "code_sha",
)
_DIRECT_JOIN_FIELDS = (
    "decision_id",
    "decision_ref",
    "action_ref",
    "portfolio_rule_event_id",
    "risk_decision_id",
    "intent_id",
    "client_order_id",
    "trade_id",
)
_ORDER_ID_CANDIDATES = (
    "order_id",
    "client_order_id",
    "broker_order_id",
    "exchange_order_id",
)
_FILL_ID_CANDIDATES = ("fill_id", "fill_event_id", "exchange_fill_id")
_OPTIONAL_REPLAY_FIELDS = frozenset((*_DIRECT_JOIN_FIELDS, "order_id", "fill_id"))
_ENVELOPE_REPLAY_FIELDS = (*_CORE_CONTEXT_FIELDS, *_DIRECT_JOIN_FIELDS, "order_id", "fill_id")


def _has_context_value(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, (dict, list, tuple, set)) and not value:
        return False
    return True


def _first_present(*values: Any, default: Any = "") -> Any:
    for value in values:
        if _has_context_value(value):
            return value
    return default


def _context_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _set_missing_context(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if key not in payload or payload.get(key) in (None, ""):
        payload[key] = value


def _metadata_to_dict(metadata: "EventMetadata | dict[str, Any] | None") -> dict[str, Any]:
    if isinstance(metadata, EventMetadata):
        return metadata.to_dict()
    return _context_dict(metadata)


def _lineage_context(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    envelope: dict[str, Any],
    lineage: dict[str, Any] | None,
) -> dict[str, Any]:
    for value in (
        lineage,
        payload.get("lineage"),
        metadata.get("lineage"),
        envelope.get("lineage"),
    ):
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _value_from_sources(key: str, *sources: dict[str, Any]) -> Any:
    return _first_present(*(source.get(key) for source in sources))


def _value_from_candidate_keys(
    keys: tuple[str, ...],
    *sources: dict[str, Any],
) -> Any:
    return _first_present(*(source.get(key) for source in sources for key in keys))


def _canonical_context(
    event_type: str,
    payload: dict[str, Any],
    *,
    metadata: "EventMetadata | dict[str, Any] | None" = None,
    envelope: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
    logical_event_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata_dict = _metadata_to_dict(metadata)
    envelope_dict = _context_dict(envelope)
    lineage_dict = _lineage_context(payload, metadata_dict, envelope_dict, lineage)
    sources = (payload, metadata_dict, envelope_dict, lineage_dict)

    context = {
        "event_id": _value_from_sources("event_id", *sources),
        "logical_event_id": _first_present(
            _value_from_sources("logical_event_id", *sources),
            logical_event_id,
            _value_from_sources("event_id", *sources),
        ),
        "event_type": _first_present(
            event_type,
            _value_from_sources("event_type", *sources),
        ),
    }
    for key in _CORE_CONTEXT_FIELDS:
        context[key] = _value_from_sources(key, *sources)
    if not context["assistant_strategy_id"] and context["strategy_id"]:
        context["assistant_strategy_id"] = assistant_strategy_id(str(context["strategy_id"]))
    if not context["exchange_timestamp"]:
        context["exchange_timestamp"] = payload.get("timestamp", "")
    for key in _DIRECT_JOIN_FIELDS:
        context[key] = _value_from_sources(key, *sources[:-1])
    context["order_id"] = _value_from_candidate_keys(_ORDER_ID_CANDIDATES, *sources[:-1])
    context["fill_id"] = _value_from_candidate_keys(_FILL_ID_CANDIDATES, *sources[:-1])
    return context, metadata_dict, lineage_dict


def _payload_with_canonical_identity(
    event_type: str,
    payload: dict[str, Any],
    *,
    metadata: "EventMetadata | dict[str, Any] | None" = None,
    envelope: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
    logical_event_id: str = "",
) -> dict[str, Any]:
    """Return payload with canonical identity and replay join keys duplicated."""
    enriched = dict(payload)
    context, metadata_dict, lineage_dict = _canonical_context(
        event_type,
        enriched,
        metadata=metadata,
        envelope=envelope,
        lineage=lineage,
        logical_event_id=logical_event_id,
    )
    if metadata_dict:
        enriched.setdefault("metadata", metadata_dict)
    if lineage_dict:
        enriched.setdefault("lineage", lineage_dict)
    for key, value in context.items():
        if key in _OPTIONAL_REPLAY_FIELDS and value in (None, ""):
            continue
        _set_missing_context(enriched, key, value)
    return enriched


def _copy_payload_context_to_envelope(
    envelope: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    for key in _ENVELOPE_REPLAY_FIELDS:
        value = payload.get(key)
        if _has_context_value(value) and not _has_context_value(envelope.get(key)):
            envelope[key] = value


def _metadata_aliases(metadata: "EventMetadata", lineage: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata_dict = metadata.to_dict()
    aliases = {
        "event_id": metadata.event_id,
        "event_type": metadata.event_type,
        "bot_id": metadata.bot_id,
        "family_id": metadata.family_id,
        "portfolio_id": metadata.portfolio_id,
        "account_alias": metadata.account_alias,
        "strategy_id": metadata.strategy_id,
        "assistant_strategy_id": assistant_strategy_id(metadata.strategy_id),
        "exchange_timestamp": metadata.exchange_timestamp.isoformat(),
        "local_timestamp": metadata.local_timestamp.isoformat(),
        "schema_version": metadata.schema_version,
        "lineage": lineage if lineage is not None else dict(metadata.lineage),
    }
    if metadata.bar_id:
        aliases["bar_id"] = metadata.bar_id
    if metadata.config_version:
        aliases["config_version"] = metadata.config_version
    if metadata.deployment_id:
        aliases["deployment_id"] = metadata.deployment_id
    if metadata.code_sha:
        aliases["code_sha"] = metadata.code_sha
    return {"metadata": metadata_dict, **aliases}


# ---------------------------------------------------------------------------
# Root cause taxonomy (21 values matching reference)
# ---------------------------------------------------------------------------

ROOT_CAUSE_TAXONOMY = frozenset({
    "regime_mismatch",
    "weak_signal",
    "strong_signal",
    "late_entry",
    "early_exit",
    "premature_stop",
    "slippage_spike",
    "good_execution",
    "filter_blocked_good",
    "filter_saved_bad",
    "risk_cap_hit",
    "data_gap",
    "order_reject",
    "latency_spike",
    "correlation_crowding",
    "funding_adverse",
    "funding_favorable",
    "regime_aligned",
    "normal_loss",
    "normal_win",
    "exceptional_win",
})


# ---------------------------------------------------------------------------
# Filter / gate decision
# ---------------------------------------------------------------------------

@dataclass
class FilterDecision:
    """Per-gate evaluation detail."""
    filter_name: str
    passed: bool
    threshold: float | None = None
    actual_value: float | None = None
    margin_pct: float | None = None
    reason: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Market context snapshot
# ---------------------------------------------------------------------------

@dataclass
class MarketContext:
    """Indicator snapshot at signal time."""
    atr: float = 0.0
    adx: float = 0.0
    rsi: float | None = None
    ema_fast: float = 0.0
    ema_mid: float = 0.0
    ema_slow: float = 0.0
    volume_ma: float = 0.0
    funding_rate: float = 0.0
    # Strategy-specific (all optional)
    bias_direction: str | None = None
    bias_strength: float | None = None
    regime_tier: str | None = None
    regime_direction: str | None = None
    h4_context_direction: str | None = None
    h4_context_strength: str | None = None
    setup_grade: str | None = None
    setup_confluences: list[str] = field(default_factory=list)
    setup_room_r: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Signal factor
# ---------------------------------------------------------------------------

@dataclass
class SignalFactor:
    """What drove the entry signal."""
    factor: str
    value: float

    def to_dict(self) -> dict:
        return {"factor": self.factor, "value": self.value}


# ---------------------------------------------------------------------------
# Event metadata
# ---------------------------------------------------------------------------

@dataclass
class EventMetadata:
    """Deterministic event identity."""
    event_id: str
    bot_id: str
    strategy_id: str
    exchange_timestamp: datetime
    event_type: str = ""
    payload_key: str = ""
    local_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    clock_skew_ms: float | None = None
    data_source: str = "runtime"
    bar_id: str = ""
    schema_version: str = ASSISTANT_EVENT_SCHEMA_VERSION
    family_id: str = "crypto_perps"
    portfolio_id: str = "default"
    account_alias: str = "default"
    config_version: str = ""
    deployment_id: str = ""
    code_sha: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    @staticmethod
    def create(
        bot_id: str,
        strategy_id: str,
        exchange_ts: datetime,
        event_type: str,
        payload_key: str,
        *,
        local_ts: datetime | None = None,
        data_source: str = "runtime",
        bar_id: str = "",
        lineage: dict[str, Any] | None = None,
        family_id: str = "crypto_perps",
        portfolio_id: str = "default",
        account_alias: str = "default",
        config_version: str = "",
        deployment_id: str = "",
        code_sha: str = "",
    ) -> EventMetadata:
        raw = f"{bot_id}|{strategy_id}|{exchange_ts.isoformat()}|{event_type}|{payload_key}"
        event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        local_timestamp = local_ts or datetime.now(timezone.utc)
        clock_skew_ms = None
        try:
            clock_skew_ms = (local_timestamp - exchange_ts).total_seconds() * 1000
        except Exception:
            clock_skew_ms = None
        return EventMetadata(
            event_id=event_id,
            bot_id=bot_id,
            strategy_id=strategy_id,
            exchange_timestamp=exchange_ts,
            event_type=event_type,
            payload_key=payload_key,
            local_timestamp=local_timestamp,
            clock_skew_ms=clock_skew_ms,
            data_source=data_source,
            bar_id=bar_id,
            family_id=family_id,
            portfolio_id=portfolio_id,
            account_alias=account_alias,
            config_version=config_version,
            deployment_id=deployment_id,
            code_sha=code_sha,
            lineage=dict(lineage or {}),
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload_key": self.payload_key,
            "bot_id": self.bot_id,
            "family_id": self.family_id,
            "portfolio_id": self.portfolio_id,
            "account_alias": self.account_alias,
            "strategy_id": self.strategy_id,
            "assistant_strategy_id": assistant_strategy_id(self.strategy_id),
            "exchange_timestamp": self.exchange_timestamp.isoformat(),
            "local_timestamp": self.local_timestamp.isoformat(),
            "clock_skew_ms": self.clock_skew_ms,
            "data_source": self.data_source,
            "bar_id": self.bar_id,
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "deployment_id": self.deployment_id,
            "code_sha": self.code_sha,
            "lineage": dict(self.lineage),
            "trace_id": self.trace_id,
        }


@dataclass
class GenericInstrumentationEvent:
    """Canonical assistant event wrapper for non-legacy instrumentation types."""

    metadata: EventMetadata
    payload: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    logical_event_id: str = ""
    priority: str = "normal"
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        return self.metadata.event_type

    def to_dict(self) -> dict[str, Any]:
        lineage = dict(self.lineage or self.metadata.lineage)
        payload = _payload_with_canonical_identity(
            self.metadata.event_type,
            self.payload,
            metadata=self.metadata,
            lineage=lineage,
            logical_event_id=self.logical_event_id or self.metadata.event_id,
        )
        payload_hash = _event_payload_hash(payload)
        envelope = {
            "schema_version": ASSISTANT_EVENT_SCHEMA_VERSION,
            "event_id": self.metadata.event_id,
            "logical_event_id": self.logical_event_id or self.metadata.event_id,
            "event_type": self.metadata.event_type,
            "bot_id": self.metadata.bot_id,
            "family_id": self.metadata.family_id,
            "portfolio_id": self.metadata.portfolio_id,
            "account_alias": self.metadata.account_alias,
            "strategy_id": self.metadata.strategy_id,
            "assistant_strategy_id": assistant_strategy_id(self.metadata.strategy_id),
            "symbol": payload.get("symbol") or payload.get("pair", ""),
            "exchange_timestamp": self.metadata.exchange_timestamp.isoformat(),
            "local_timestamp": self.metadata.local_timestamp.isoformat(),
            "config_version": self.metadata.config_version,
            "deployment_id": self.metadata.deployment_id,
            "code_sha": self.metadata.code_sha,
            "payload_hash": payload_hash,
            "priority": self.priority,
            "lineage": lineage,
            "source": dict(self.source),
            "payload": payload,
        }
        _copy_payload_context_to_envelope(
            envelope,
            payload,
        )
        return envelope


def canonical_event_envelope(
    event_type: str,
    payload: dict[str, Any],
    *,
    bot_id: str = "",
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap legacy payloads in the canonical assistant event envelope."""
    if payload.get("schema_version") == ASSISTANT_EVENT_SCHEMA_VERSION and "payload" in payload:
        canonical = dict(payload)
        canonical_event_type = str(canonical.get("event_type") or event_type)
        canonical_payload = canonical.get("payload")
        canonical_payload = (
            dict(canonical_payload)
            if isinstance(canonical_payload, dict)
            else {}
        )
        canonical["payload"] = _payload_with_canonical_identity(
            canonical_event_type,
            canonical_payload,
            metadata=canonical_payload.get("metadata"),
            envelope=canonical,
            lineage=_context_dict(canonical.get("lineage")),
            logical_event_id=str(canonical.get("logical_event_id") or ""),
        )
        _copy_payload_context_to_envelope(
            canonical,
            canonical["payload"],
        )
        canonical["payload_hash"] = _event_payload_hash(canonical["payload"])
        existing_source = canonical.get("source")
        merged_source = dict(existing_source) if isinstance(existing_source, dict) else {}
        if source:
            merged_source.update(source)
        if merged_source:
            canonical["source"] = merged_source
        return canonical

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    event_id = str(payload.get("event_id") or metadata.get("event_id") or stable_hash(payload))
    logical_event_id = str(payload.get("logical_event_id") or event_id)
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        lineage = metadata.get("lineage") if isinstance(metadata.get("lineage"), dict) else {}
    strategy_id = str(payload.get("strategy_id") or metadata.get("strategy_id") or "")
    assistant_id = str(
        payload.get("assistant_strategy_id")
        or metadata.get("assistant_strategy_id")
        or assistant_strategy_id(strategy_id)
    )
    envelope = {
        "schema_version": ASSISTANT_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "logical_event_id": logical_event_id,
        "event_type": event_type,
        "bot_id": str(payload.get("bot_id") or metadata.get("bot_id") or bot_id),
        "family_id": str(payload.get("family_id") or metadata.get("family_id") or lineage.get("family_id", "")),
        "portfolio_id": str(payload.get("portfolio_id") or metadata.get("portfolio_id") or lineage.get("portfolio_id", "")),
        "account_alias": str(payload.get("account_alias") or metadata.get("account_alias") or lineage.get("account_alias", "")),
        "strategy_id": strategy_id,
        "assistant_strategy_id": assistant_id,
        "symbol": str(payload.get("symbol") or payload.get("pair") or ""),
        "exchange_timestamp": (
            payload.get("exchange_timestamp")
            or metadata.get("exchange_timestamp")
            or payload.get("timestamp")
        ),
        "local_timestamp": (
            payload.get("local_timestamp")
            or metadata.get("local_timestamp")
            or datetime.now(timezone.utc).isoformat()
        ),
        "config_version": str(payload.get("config_version") or metadata.get("config_version") or lineage.get("config_version", "")),
        "deployment_id": str(payload.get("deployment_id") or metadata.get("deployment_id") or lineage.get("deployment_id", "")),
        "code_sha": str(payload.get("code_sha") or metadata.get("code_sha") or lineage.get("code_sha", "")),
        "payload_hash": "",
        "priority": payload.get("priority", "normal"),
        "lineage": lineage,
        "source": dict(source or {}),
    }
    canonical_payload = _payload_with_canonical_identity(
        event_type,
        payload,
        metadata=metadata,
        envelope=envelope,
        lineage=lineage,
        logical_event_id=logical_event_id,
    )
    _copy_payload_context_to_envelope(
        envelope,
        canonical_payload,
    )
    envelope["payload_hash"] = _event_payload_hash(canonical_payload)
    envelope["payload"] = canonical_payload
    return envelope


# ---------------------------------------------------------------------------
# Instrumented trade event
# ---------------------------------------------------------------------------

@dataclass
class InstrumentedTradeEvent:
    """Full trade record with context."""
    metadata: EventMetadata
    lineage: dict[str, Any] = field(default_factory=dict)
    logical_event_id: str = ""
    # Identity
    trade_id: str = ""
    pair: str = ""
    side: str = ""
    entry_decision_id: str = ""
    exit_decision_id: str = ""
    entry_signal_id: str = ""
    entry_bar_id: str = ""
    exit_bar_id: str = ""
    entry_order_ids: list[str] = field(default_factory=list)
    exit_order_ids: list[str] = field(default_factory=list)
    entry_fill_ids: list[str] = field(default_factory=list)
    exit_fill_ids: list[str] = field(default_factory=list)
    client_order_ids: list[str] = field(default_factory=list)
    exchange_order_ids: list[str] = field(default_factory=list)
    intent_id: str = ""
    decision_ref: dict[str, Any] = field(default_factory=dict)
    action_ref: dict[str, Any] = field(default_factory=dict)
    portfolio_decision_ref: dict[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""
    resource_plan_hash: str = ""
    runtime_join: dict[str, Any] = field(default_factory=dict)
    # Timing
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Prices & P&L
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    pnl: float = 0.0
    price_pnl_gross: float = 0.0
    total_fees: float = 0.0
    price_pnl_after_funding: float = 0.0
    realized_pnl_net: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float | None = None
    realized_r_net: float | None = None
    geometric_r: float | None = None
    commission: float = 0.0
    funding_paid: float = 0.0
    slippage_pct: float | None = None
    latency_ms: float | None = None
    liquidity: str | None = None
    # Signal
    entry_signal: str = ""
    entry_signal_strength: float = 0.0
    setup_grade: str = ""
    exit_reason: str = ""
    confluences: list[str] = field(default_factory=list)
    entry_method: str = ""
    signal_factors: list[SignalFactor] = field(default_factory=list)
    # Gate pipeline
    filter_decisions: list[FilterDecision] = field(default_factory=list)
    passed_filters: list[str] = field(default_factory=list)
    active_filters: list[str] = field(default_factory=list)
    # Market context at entry
    market_context: MarketContext | None = None
    # Excursion
    mfe_r: float | None = None
    mae_r: float | None = None
    exit_efficiency: float | None = None
    # Post-exit tracking (backfilled in live mode)
    post_exit_1h_move_pct: float | None = None
    post_exit_4h_move_pct: float | None = None
    # Quality
    process_quality_score: int = 100
    root_causes: list[str] = field(default_factory=list)
    # Snapshots at entry
    strategy_params_at_entry: dict = field(default_factory=dict)
    sizing_inputs: dict = field(default_factory=dict)
    portfolio_state_at_entry: dict | None = None
    portfolio_rule_event_id: str = ""
    risk_decision_id: str = ""

    def to_dict(self) -> dict:
        lineage = dict(self.lineage or self.metadata.lineage)
        notional_usd = self.entry_price * self.position_size
        d: dict[str, Any] = {
            **_metadata_aliases(self.metadata, lineage),
            "logical_event_id": self.logical_event_id or self.trade_id or self.metadata.event_id,
            "trade_id": self.trade_id,
            "pair": self.pair,
            "symbol": self.pair,
            "side": self.side,
            "entry_decision_id": self.entry_decision_id,
            "exit_decision_id": self.exit_decision_id,
            "entry_signal_id": self.entry_signal_id,
            "entry_bar_id": self.entry_bar_id,
            "exit_bar_id": self.exit_bar_id,
            "entry_order_ids": list(self.entry_order_ids),
            "exit_order_ids": list(self.exit_order_ids),
            "entry_fill_ids": list(self.entry_fill_ids),
            "exit_fill_ids": list(self.exit_fill_ids),
            "client_order_ids": list(self.client_order_ids),
            "exchange_order_ids": list(self.exchange_order_ids),
            "intent_id": self.intent_id,
            "decision_ref": dict(self.decision_ref),
            "action_ref": dict(self.action_ref),
            "portfolio_decision_ref": dict(self.portfolio_decision_ref),
            "artifact_hash": self.artifact_hash,
            "resource_plan_hash": self.resource_plan_hash,
            "runtime_join": dict(self.runtime_join),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "time_in_trade_seconds": max(
                0.0,
                (self.exit_time - self.entry_time).total_seconds(),
            ),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "notional_usd": notional_usd,
            "pnl": self.pnl,
            "price_pnl_gross": self.price_pnl_gross,
            "total_fees": self.total_fees,
            "price_pnl_after_funding": (
                self.price_pnl_after_funding
                if self.price_pnl_after_funding != 0.0
                else self.price_pnl_gross - self.funding_paid
            ),
            "realized_pnl_net": self.realized_pnl_net,
            "pnl_pct": self.pnl_pct,
            "r_multiple": self.r_multiple,
            "realized_r_net": self.realized_r_net,
            "geometric_r": self.geometric_r,
            "commission": self.commission,
            "funding_paid": self.funding_paid,
            "slippage_pct": self.slippage_pct,
            "latency_ms": self.latency_ms,
            "liquidity": self.liquidity,
            "entry_signal": self.entry_signal,
            "entry_signal_strength": self.entry_signal_strength,
            "setup_grade": self.setup_grade,
            "exit_reason": self.exit_reason,
            "confluences": self.confluences,
            "entry_method": self.entry_method,
            "signal_factors": [sf.to_dict() for sf in self.signal_factors],
            "filter_decisions": [fd.to_dict() for fd in self.filter_decisions],
            "passed_filters": self.passed_filters,
            "active_filters": self.active_filters,
            "bias_direction": (
                self.market_context.bias_direction
                if self.market_context is not None
                else None
            ),
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "exit_efficiency": self.exit_efficiency,
            "post_exit_1h_move_pct": self.post_exit_1h_move_pct,
            "post_exit_4h_move_pct": self.post_exit_4h_move_pct,
            "process_quality_score": self.process_quality_score,
            "root_causes": self.root_causes,
            "strategy_params_at_entry": self.strategy_params_at_entry,
            "sizing_inputs": self.sizing_inputs,
            "portfolio_state_at_entry": self.portfolio_state_at_entry,
            "portfolio_rule_event_id": self.portfolio_rule_event_id,
            "risk_decision_id": self.risk_decision_id,
        }
        return d


# ---------------------------------------------------------------------------
# Missed opportunity event
# ---------------------------------------------------------------------------

@dataclass
class MissedOpportunityEvent:
    """Blocked signals with backfill slots."""
    metadata: EventMetadata
    lineage: dict[str, Any] = field(default_factory=dict)
    opportunity_id: str = ""
    logical_event_id: str = ""
    revision: int = 0
    supersedes_event_id: str = ""
    pair: str = ""
    symbol: str = ""
    timeframe: str = ""
    bar_id: str = ""
    decision_id: str = ""
    signal_id: str = ""
    signal: str = ""
    signal_strength: float = 0.0
    blocked_by: str = ""
    block_reason: str = ""
    blocking_rule_type: str = "strategy_filter"
    margin_pct: float | None = None
    hypothetical_entry: float = 0.0
    simulation_policy: dict[str, Any] = field(default_factory=dict)
    market_context: MarketContext | None = None
    filter_decisions: list[FilterDecision] = field(default_factory=list)
    portfolio_rule_event_id: str = ""
    # Backfilled later
    outcome_1h: float | None = None
    outcome_4h: float | None = None
    outcome_24h: float | None = None
    would_have_hit_tp: bool | None = None
    would_have_hit_sl: bool | None = None
    backfill_status: str = "pending"

    def bump_revision(self, event_type: str = "missed_opportunity") -> None:
        """Give a mutable missed opportunity update a new event id."""
        self.revision += 1
        self.supersedes_event_id = self.metadata.event_id
        logical = self.logical_event_id or self.opportunity_id or self.metadata.event_id
        payload_key = f"{logical}:revision:{self.revision}"
        self.metadata = EventMetadata.create(
            bot_id=self.metadata.bot_id,
            strategy_id=self.metadata.strategy_id,
            exchange_ts=self.metadata.exchange_timestamp,
            event_type=event_type,
            payload_key=payload_key,
            local_ts=datetime.now(timezone.utc),
            data_source=self.metadata.data_source,
            bar_id=self.metadata.bar_id,
            lineage=self.metadata.lineage,
            family_id=self.metadata.family_id,
            portfolio_id=self.metadata.portfolio_id,
            account_alias=self.metadata.account_alias,
            config_version=self.metadata.config_version,
            deployment_id=self.metadata.deployment_id,
            code_sha=self.metadata.code_sha,
        )

    def to_dict(self) -> dict:
        lineage = dict(self.lineage or self.metadata.lineage)
        symbol = self.symbol or self.pair
        return {
            **_metadata_aliases(self.metadata, lineage),
            "opportunity_id": self.opportunity_id or self.logical_event_id or self.metadata.event_id,
            "logical_event_id": self.logical_event_id or self.opportunity_id or self.metadata.event_id,
            "revision": self.revision,
            "supersedes_event_id": self.supersedes_event_id,
            "pair": self.pair,
            "symbol": symbol,
            "timeframe": self.timeframe,
            "bar_id": self.bar_id or self.metadata.bar_id,
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "signal": self.signal,
            "signal_strength": self.signal_strength,
            "blocked_by": self.blocked_by,
            "block_reason": self.block_reason,
            "blocking_rule_type": self.blocking_rule_type,
            "margin_pct": self.margin_pct,
            "hypothetical_entry": self.hypothetical_entry,
            "simulation_policy": dict(self.simulation_policy),
            "market_context": self.market_context.to_dict() if self.market_context else None,
            "filter_decisions": [fd.to_dict() for fd in self.filter_decisions],
            "portfolio_rule_event_id": self.portfolio_rule_event_id,
            "outcome_1h": self.outcome_1h,
            "outcome_4h": self.outcome_4h,
            "outcome_24h": self.outcome_24h,
            "would_have_hit_tp": self.would_have_hit_tp,
            "would_have_hit_sl": self.would_have_hit_sl,
            "backfill_status": self.backfill_status,
        }


# ---------------------------------------------------------------------------
# Daily snapshot
# ---------------------------------------------------------------------------

@dataclass
class DailySnapshot:
    """End-of-day aggregate (live mode only)."""
    metadata: EventMetadata
    lineage: dict[str, Any] = field(default_factory=dict)
    date: str = ""
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_rolling_30d: float = 0.0
    sortino_rolling_30d: float = 0.0
    calmar_rolling_30d: float = 0.0
    exposure_pct: float = 0.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    avg_process_quality: float = 0.0
    root_cause_distribution: dict[str, int] = field(default_factory=dict)
    per_strategy_summary: dict[str, dict] = field(default_factory=dict)
    family_summary: dict[str, Any] = field(default_factory=dict)
    portfolio_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            **_metadata_aliases(self.metadata, dict(self.lineage or self.metadata.lineage)),
            "date": self.date,
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_rolling_30d": self.sharpe_rolling_30d,
            "sortino_rolling_30d": self.sortino_rolling_30d,
            "calmar_rolling_30d": self.calmar_rolling_30d,
            "exposure_pct": self.exposure_pct,
            "missed_count": self.missed_count,
            "missed_would_have_won": self.missed_would_have_won,
            "avg_process_quality": self.avg_process_quality,
            "root_cause_distribution": self.root_cause_distribution,
            "per_strategy_summary": self.per_strategy_summary,
            "family_summary": self.family_summary,
            "portfolio_summary": self.portfolio_summary,
        }


# ---------------------------------------------------------------------------
# Error event
# ---------------------------------------------------------------------------

@dataclass
class ErrorEvent:
    """Error telemetry."""
    metadata: EventMetadata
    lineage: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    message: str = ""
    stack_trace: str = ""
    severity: str = "low"
    component: str = ""
    symbol: str = ""
    order_id: str = ""
    fill_id: str = ""
    decision_id: str = ""
    recovery_action: str = ""

    def to_dict(self) -> dict:
        return {
            **_metadata_aliases(self.metadata, dict(self.lineage or self.metadata.lineage)),
            "error_type": self.error_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "severity": self.severity,
            "component": self.component,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "fill_id": self.fill_id,
            "decision_id": self.decision_id,
            "recovery_action": self.recovery_action,
        }


# ---------------------------------------------------------------------------
# Pipeline funnel snapshot
# ---------------------------------------------------------------------------

@dataclass
class PipelineFunnelSnapshot:
    """Periodic pipeline funnel snapshot for a strategy."""
    strategy_id: str
    timestamp: str
    period_start: str
    period_end: str
    funnel: dict = field(default_factory=dict)
    assessment: str = "normal"
    metadata: EventMetadata | None = None
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        metadata = payload.pop("metadata", None)
        lineage = payload.pop("lineage", self.lineage)
        payload["assistant_strategy_id"] = assistant_strategy_id(self.strategy_id)
        payload["event_type"] = "pipeline_funnel"
        if self.metadata is not None:
            payload = {
                **_metadata_aliases(self.metadata, dict(self.lineage or self.metadata.lineage)),
                **payload,
            }
        else:
            payload["lineage"] = lineage
        return payload


# ---------------------------------------------------------------------------
# Health report snapshot
# ---------------------------------------------------------------------------

@dataclass
class HealthReportSnapshot:
    """Periodic system health report."""
    timestamp: str
    report: dict = field(default_factory=dict)
    metadata: EventMetadata | None = None
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("metadata", None)
        lineage = payload.pop("lineage", self.lineage)
        payload["event_type"] = "heartbeat"
        if self.metadata is not None:
            return {
                **_metadata_aliases(self.metadata, dict(self.lineage or self.metadata.lineage)),
                **payload,
            }
        payload["lineage"] = lineage
        return payload
