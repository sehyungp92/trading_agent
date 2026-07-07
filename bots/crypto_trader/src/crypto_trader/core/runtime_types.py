"""Canonical runtime contracts for live/backtest parity.

These types are additive: current feeds, strategies, and brokers still use the
existing ``Bar``, ``Order``, ``Fill``, and ``Trade`` models. The migration map is
intentionally narrow for now: ``Bar`` adapts through ``MarketEvent.from_bar()``,
``Trade`` adapts through ``TradeOutcome.from_trade()``, and future execution
adapters can map orders/fills into ``OrderIntent`` and ``ExecutionReport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from crypto_trader.core.models import Bar, Order, OrderStatus, OrderType, Side, TimeFrame, Trade
from crypto_trader.core.market_time import candle_times_from_timestamp, ensure_utc


class TimestampPolicy(Enum):
    """How a source timestamp should be interpreted."""

    OPEN_TIME = "open_time"
    CLOSE_TIME = "close_time"


class ExecutionReportKind(Enum):
    """Adapter-neutral execution report categories."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RESTING = "resting"
    PARTIAL_FILL = "partial_fill"
    FILL = "fill"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(slots=True)
class DecisionContext:
    """Mutable context for one strategy-visible callback."""

    decision_id: str
    strategy_id: str
    symbol: str
    timeframe: TimeFrame
    decision_time: datetime
    decision_key: str
    action: str = "no_order"
    order_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_order(self) -> None:
        self.order_count += 1
        self.action = "order"

    def to_decision_event(self) -> "DecisionEvent":
        return DecisionEvent(
            decision_id=self.decision_id,
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            timeframe=self.timeframe,
            decision_time=self.decision_time,
            decision_key=self.decision_key,
            action=self.action,
            metadata={**self.metadata, "order_count": self.order_count},
        )


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


def _enum(value: Enum | None) -> str | None:
    return value.value if value is not None else None


def _intent_id(
    strategy_id: str,
    symbol: str,
    decision_id: str,
    context: "DecisionContext | None",
) -> str:
    seq = context.order_count + 1 if context is not None else 1
    prefix = strategy_id or "unknown"
    decision_part = decision_id or "manual"
    return f"{prefix}:{symbol}:{decision_part}:intent:{seq}"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """Normalized completed market bar with explicit availability time."""

    symbol: str
    timeframe: TimeFrame
    open_time: datetime
    close_time: datetime
    available_at: datetime
    source: str
    timestamp_policy: TimestampPolicy
    open: float
    high: float
    low: float
    close: float
    volume: float
    raw_timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bar(
        cls,
        bar: Bar,
        *,
        source: str,
        timestamp_policy: TimestampPolicy = TimestampPolicy.OPEN_TIME,
    ) -> "MarketEvent":
        times = candle_times_from_timestamp(
            bar.timestamp,
            bar.timeframe,
            timestamp_policy=timestamp_policy.value,
        )

        return cls(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            open_time=times.open_time,
            close_time=times.close_time,
            available_at=times.available_at,
            source=source,
            timestamp_policy=timestamp_policy,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            raw_timestamp=ensure_utc(bar.timestamp),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "available_at": self.available_at.isoformat(),
            "source": self.source,
            "timestamp_policy": self.timestamp_policy.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "raw_timestamp": _iso(self.raw_timestamp),
            "metadata": dict(self.metadata),
        }

    def to_bar(self) -> Bar:
        """Return the legacy strategy-visible bar using canonical open time."""
        return Bar(
            timestamp=self.open_time,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            timeframe=self.timeframe,
        )


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """Strategy decision record suitable for replay comparison."""

    decision_id: str
    strategy_id: str
    symbol: str
    timeframe: TimeFrame
    decision_time: datetime
    decision_key: str = ""
    action: str = ""
    signal_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "decision_id": self.decision_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "decision_time": self.decision_time.isoformat(),
            "decision_key": self.decision_key,
            "action": self.action,
            "signal_context": dict(self.signal_context),
            "metadata": dict(self.metadata),
        }
        if self.metadata.get("bar_id"):
            payload["bar_id"] = self.metadata["bar_id"]
        return payload


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Adapter-neutral order request produced by a strategy decision."""

    intent_id: str
    strategy_id: str
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    decision_id: str = ""
    client_order_id: str = ""
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    time_in_force: str = "GTC"
    ttl_bars: int | None = None
    oca_group: str | None = None
    bracket_group: str | None = None
    risk_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_order(
        cls,
        order: Order,
        context: DecisionContext | None = None,
    ) -> "OrderIntent":
        """Adapt a legacy strategy order into a canonical intent."""
        metadata = dict(order.metadata)
        strategy_id = str(metadata.get("strategy_id") or (context.strategy_id if context else ""))
        client_order_id = str(metadata.get("client_order_id") or order.order_id or "")
        decision_id = str(metadata.get("decision_id") or (context.decision_id if context else ""))
        context_intent_id = (
            _intent_id(strategy_id, order.symbol, decision_id, context)
            if context is not None
            else ""
        )
        intent_id = str(
            metadata.get("intent_id")
            or context_intent_id
            or client_order_id
            or _intent_id(strategy_id, order.symbol, decision_id, context)
        )
        risk_metadata = {
            key: value for key, value in metadata.items()
            if key.startswith("risk") or key in {"risk_R", "leverage", "stop_distance"}
        }
        return cls(
            intent_id=intent_id,
            strategy_id=strategy_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            qty=order.qty,
            decision_id=decision_id,
            client_order_id=client_order_id,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            reduce_only=bool(metadata.get("reduce_only", False)),
            time_in_force=order.time_in_force,
            ttl_bars=order.ttl_bars,
            oca_group=order.oca_group,
            bracket_group=metadata.get("bracket_group"),
            risk_metadata=risk_metadata,
            metadata={**metadata, "tag": order.tag},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": self.qty,
            "decision_id": self.decision_id,
            "client_order_id": self.client_order_id,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "reduce_only": self.reduce_only,
            "time_in_force": self.time_in_force,
            "ttl_bars": self.ttl_bars,
            "oca_group": self.oca_group,
            "bracket_group": self.bracket_group,
            "risk_metadata": dict(self.risk_metadata),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Adapter-neutral order lifecycle report."""

    report_id: str
    kind: ExecutionReportKind
    timestamp: datetime
    symbol: str
    side: Side | None = None
    client_order_id: str = ""
    exchange_order_id: str = ""
    fill_id: str | None = None
    order_status: OrderStatus | None = None
    qty: float = 0.0
    filled_qty: float = 0.0
    fill_price: float | None = None
    commission: float = 0.0
    liquidity: str | None = None
    reject_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "kind": self.kind.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": _enum(self.side),
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "fill_id": self.fill_id,
            "order_status": _enum(self.order_status),
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "liquidity": self.liquidity,
            "reject_reason": self.reject_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Owned strategy position state at a point in time."""

    strategy_id: str
    symbol: str
    timestamp: datetime
    direction: Side | None = None
    qty: float = 0.0
    avg_entry: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    mfe_r: float | None = None
    mae_r: float | None = None
    open_orders: list[str] = field(default_factory=list)
    risk_metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": _enum(self.direction),
            "qty": self.qty,
            "avg_entry": self.avg_entry,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid,
            "funding_paid": self.funding_paid,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "open_orders": list(self.open_orders),
            "risk_metadata": dict(self.risk_metadata),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    """Canonical realized or terminal trade economics."""

    trade_id: str
    symbol: str
    direction: Side
    entry_price: float
    exit_price: float
    qty: float
    entry_time: datetime
    exit_time: datetime
    price_pnl_gross: float
    total_fees: float
    funding_paid: float
    realized_pnl_net: float
    initial_risk_amount: float | None = None
    geometric_r: float | None = None
    realized_r_net: float | None = None
    exit_reason: str = ""
    source_trade_pnl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trade(cls, trade: Trade) -> "TradeOutcome":
        """Adapt current ``Trade`` economics without changing their semantics."""
        funding_paid = trade.funding_paid or 0.0
        total_fees = trade.commission or 0.0
        price_pnl_gross = trade.pnl + funding_paid
        return cls(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            qty=trade.qty,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            price_pnl_gross=price_pnl_gross,
            total_fees=total_fees,
            funding_paid=funding_paid,
            realized_pnl_net=price_pnl_gross - funding_paid - total_fees,
            geometric_r=trade.r_multiple,
            realized_r_net=trade.realized_r_multiple,
            exit_reason=trade.exit_reason,
            source_trade_pnl=trade.pnl,
            metadata={"signal_variant": trade.signal_variant},
        )

    @property
    def price_pnl_after_funding(self) -> float:
        return self.price_pnl_gross - self.funding_paid

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "price_pnl_gross": self.price_pnl_gross,
            "price_pnl_after_funding": self.price_pnl_after_funding,
            "total_fees": self.total_fees,
            "funding_paid": self.funding_paid,
            "realized_pnl_net": self.realized_pnl_net,
            "initial_risk_amount": self.initial_risk_amount,
            "geometric_r": self.geometric_r,
            "realized_r_net": self.realized_r_net,
            "exit_reason": self.exit_reason,
            "source_trade_pnl": self.source_trade_pnl,
            "metadata": dict(self.metadata),
        }
