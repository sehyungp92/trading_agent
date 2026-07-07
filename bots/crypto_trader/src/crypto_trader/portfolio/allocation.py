"""Strategy ownership read models for shared-symbol live portfolios."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any

from crypto_trader.core.models import Position, Side
from crypto_trader.portfolio.state import OpenRisk

_QTY_EPS = 1e-8


@dataclass(frozen=True, slots=True)
class StrategyPositionAllocation:
    position_instance_id: str
    strategy_id: str
    symbol: str
    direction: Side
    allocated_qty: float
    avg_entry: float
    open_risk_R: float
    entry_time: datetime
    entry_order_ids: list[str]
    entry_fill_ids: list[str]
    exit_order_ids: list[str]
    exit_fill_ids: list[str]
    source: str = "lifecycle"
    confidence: str = "exact"
    status: str = "OPEN"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_instance_id": self.position_instance_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "allocated_qty": self.allocated_qty,
            "avg_entry": self.avg_entry,
            "open_risk_R": self.open_risk_R,
            "risk_r": self.open_risk_R,
            "entry_time": self.entry_time.isoformat(),
            "entry_order_ids": list(self.entry_order_ids),
            "entry_fill_ids": list(self.entry_fill_ids),
            "exit_order_ids": list(self.exit_order_ids),
            "exit_fill_ids": list(self.exit_fill_ids),
            "source": self.source,
            "confidence": self.confidence,
            "status": self.status,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class ExchangeNetPosition:
    symbol: str
    direction: Side
    qty: float
    avg_entry: float
    unrealized_pnl: float
    liquidation_price: float | None
    observed_at: datetime
    source: str = "exchange"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "qty": self.qty,
            "avg_entry": self.avg_entry,
            "unrealized_pnl": self.unrealized_pnl,
            "liquidation_price": self.liquidation_price,
            "observed_at": self.observed_at.isoformat(),
            "source": self.source,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class AllocationResidual:
    symbol: str
    direction: Side
    net_exchange_qty: float
    allocated_qty: float
    unallocated_qty: float
    unknown_allocation: bool
    status: str = "DRIFT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "net_exchange_qty": self.net_exchange_qty,
            "allocated_qty": self.allocated_qty,
            "unallocated_qty": self.unallocated_qty,
            "unknown_allocation": self.unknown_allocation,
            "status": self.status,
        }


def derive_strategy_position_allocations(
    lifecycle_entries: list[Any] | None = None,
    open_risks: list[OpenRisk] | None = None,
    *,
    observed_at: datetime | None = None,
) -> list[StrategyPositionAllocation]:
    allocations: dict[str, StrategyPositionAllocation] = {}
    now = observed_at or datetime.now(timezone.utc)

    for raw in lifecycle_entries or []:
        entry = _plain(raw)
        qty = _float(entry.get("qty"))
        if abs(qty) <= _QTY_EPS:
            continue
        metadata = dict(entry.get("metadata") or {})
        allocation = StrategyPositionAllocation(
            position_instance_id=str(entry.get("position_instance_id") or ""),
            strategy_id=str(entry.get("strategy_id") or ""),
            symbol=str(entry.get("symbol") or ""),
            direction=_side(entry.get("direction")),
            allocated_qty=abs(qty),
            avg_entry=_float(entry.get("avg_entry")),
            open_risk_R=_float(metadata.get("risk_R") or metadata.get("risk_r")),
            entry_time=_datetime(entry.get("entry_time"), now),
            entry_order_ids=_list(metadata.get("entry_order_ids")),
            entry_fill_ids=_list(metadata.get("entry_fill_ids")),
            exit_order_ids=_list(metadata.get("exit_order_ids")),
            exit_fill_ids=_list(metadata.get("exit_fill_ids")),
            source="lifecycle",
            confidence="exact",
            status="OPEN",
            metadata=metadata,
        )
        if allocation.position_instance_id:
            allocations[allocation.position_instance_id] = allocation

    for risk in open_risks or []:
        position_instance_id = (
            risk.position_instance_id
            or risk.risk_id
            or risk.intent_id
            or risk.client_order_id
            or risk.order_id
        )
        if not position_instance_id or position_instance_id in allocations:
            continue
        qty = risk.filled_qty if risk.filled_qty > 0 else risk.order_qty
        confidence = "recovered" if risk.filled_qty > 0 else "inferred"
        allocations[position_instance_id] = StrategyPositionAllocation(
            position_instance_id=position_instance_id,
            strategy_id=risk.strategy_id,
            symbol=risk.symbol,
            direction=risk.direction,
            allocated_qty=max(0.0, qty),
            avg_entry=0.0,
            open_risk_R=risk.risk_R,
            entry_time=_datetime(risk.entry_time, now),
            entry_order_ids=_compact([risk.order_id, risk.client_order_id, risk.exchange_order_id]),
            entry_fill_ids=list(risk.applied_fill_ids),
            exit_order_ids=[],
            exit_fill_ids=[],
            source="portfolio_state",
            confidence=confidence,
            status="OPEN" if qty > _QTY_EPS else "DRIFT",
            metadata={
                "intent_id": risk.intent_id,
                "risk_id": risk.risk_id,
            },
        )

    return sorted(
        allocations.values(),
        key=lambda item: (item.symbol, item.strategy_id, item.position_instance_id),
    )


def exchange_net_positions(
    positions: list[Position],
    *,
    observed_at: datetime | None = None,
) -> list[ExchangeNetPosition]:
    now = observed_at or datetime.now(timezone.utc)
    return [
        ExchangeNetPosition(
            symbol=position.symbol,
            direction=position.direction,
            qty=abs(position.qty),
            avg_entry=position.avg_entry,
            unrealized_pnl=position.unrealized_pnl,
            liquidation_price=position.liquidation_price,
            observed_at=now,
            metadata=dict(position.metadata),
        )
        for position in positions
        if abs(position.qty) > _QTY_EPS
    ]


def allocation_residuals(
    exchange_positions: list[ExchangeNetPosition],
    allocations: list[StrategyPositionAllocation],
    *,
    epsilon: float = _QTY_EPS,
) -> list[AllocationResidual]:
    allocated: dict[tuple[str, Side], float] = {}
    for allocation in allocations:
        key = (allocation.symbol, allocation.direction)
        allocated[key] = allocated.get(key, 0.0) + allocation.allocated_qty

    residuals: list[AllocationResidual] = []
    seen: set[tuple[str, Side]] = set()
    for net in exchange_positions:
        key = (net.symbol, net.direction)
        seen.add(key)
        allocated_qty = allocated.get(key, 0.0)
        residual = net.qty - allocated_qty
        if abs(residual) <= epsilon:
            continue
        residuals.append(AllocationResidual(
            symbol=net.symbol,
            direction=net.direction,
            net_exchange_qty=net.qty,
            allocated_qty=allocated_qty,
            unallocated_qty=residual,
            unknown_allocation=residual > epsilon,
        ))
    for (symbol, direction), allocated_qty in sorted(allocated.items(), key=lambda item: (item[0][0], item[0][1].value)):
        if (symbol, direction) in seen or allocated_qty <= epsilon:
            continue
        residuals.append(AllocationResidual(
            symbol=symbol,
            direction=direction,
            net_exchange_qty=0.0,
            allocated_qty=allocated_qty,
            unallocated_qty=-allocated_qty,
            unknown_allocation=False,
        ))
    return residuals


def admin_correct_unknown_allocation(
    residual: AllocationResidual,
    *,
    strategy_id: str,
    position_instance_id: str,
    avg_entry: float,
    entry_time: datetime | None = None,
    reason: str = "",
) -> StrategyPositionAllocation:
    return StrategyPositionAllocation(
        position_instance_id=position_instance_id,
        strategy_id=strategy_id,
        symbol=residual.symbol,
        direction=residual.direction,
        allocated_qty=abs(residual.unallocated_qty),
        avg_entry=avg_entry,
        open_risk_R=0.0,
        entry_time=entry_time or datetime.now(timezone.utc),
        entry_order_ids=[],
        entry_fill_ids=[],
        exit_order_ids=[],
        exit_fill_ids=[],
        source="admin_correction",
        confidence="recovered",
        status="OPEN",
        metadata={"correction_reason": reason},
    )


def _plain(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value or {})


def _side(value: Any) -> Side:
    if isinstance(value, Side):
        return value
    return Side(str(value))


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _datetime(value: Any, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return default
    return default


def _list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _compact(values: list[Any]) -> list[str]:
    return [str(value) for value in values if value]
