"""Live position lifecycle ledger built from owned fills."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from crypto_trader.core.models import Fill, Side, Trade
from crypto_trader.core.order_semantics import entry_position_instance_id
from crypto_trader.live.oms_store import fill_identity


@dataclass(slots=True)
class LivePositionLedgerEntry:
    strategy_id: str
    symbol: str
    direction: Side
    position_instance_id: str
    qty: float
    avg_entry: float
    entry_time: datetime
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    closed_qty: float = 0.0
    realized_price_pnl: float = 0.0
    funding_paid: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PositionLifecycleLedger:
    """Accumulate live position economics from fills instead of final fill only."""

    def __init__(self) -> None:
        self._positions: dict[tuple[str, str, Side], LivePositionLedgerEntry] = {}

    def apply_fill(self, strategy_id: str, fill: Fill) -> Trade | None:
        """Apply a fill and return a completed trade when the position closes."""
        entry_key = (strategy_id, fill.symbol, fill.side)
        existing = self._positions.get(entry_key)
        if fill.tag == "entry" or existing is not None:
            return self._apply_entry_or_add(strategy_id, fill, entry_key, existing)

        exit_direction = Side.SHORT if fill.side == Side.LONG else Side.LONG
        key = (strategy_id, fill.symbol, exit_direction)
        position = self._positions.get(key)
        if position is None:
            return None
        return self._apply_exit(position, fill, key)

    def open_positions(self) -> list[LivePositionLedgerEntry]:
        return list(self._positions.values())

    def snapshot(self) -> list[LivePositionLedgerEntry]:
        return self.open_positions()

    def restore(self, entries: list[dict]) -> None:
        self._positions.clear()
        for raw in entries:
            direction = raw["direction"]
            if not isinstance(direction, Side):
                direction = Side(direction)
            entry_time = raw["entry_time"]
            if not isinstance(entry_time, datetime):
                entry_time = datetime.fromisoformat(entry_time)
            entry = LivePositionLedgerEntry(
                strategy_id=raw["strategy_id"],
                symbol=raw["symbol"],
                direction=direction,
                position_instance_id=raw["position_instance_id"],
                qty=float(raw["qty"]),
                avg_entry=float(raw["avg_entry"]),
                entry_time=entry_time,
                entry_commission=float(raw.get("entry_commission", 0.0)),
                exit_commission=float(raw.get("exit_commission", 0.0)),
                closed_qty=float(raw.get("closed_qty", 0.0)),
                realized_price_pnl=float(raw.get("realized_price_pnl", 0.0)),
                funding_paid=float(raw.get("funding_paid", 0.0)),
                metadata=dict(raw.get("metadata") or {}),
            )
            self._positions[(entry.strategy_id, entry.symbol, entry.direction)] = entry

    def _apply_entry_or_add(
        self,
        strategy_id: str,
        fill: Fill,
        key: tuple[str, str, Side],
        existing: LivePositionLedgerEntry | None,
    ) -> None:
        if existing is None:
            self._positions[key] = LivePositionLedgerEntry(
                strategy_id=strategy_id,
                symbol=fill.symbol,
                direction=fill.side,
                position_instance_id=str(
                    fill.raw.get("position_instance_id")
                    or entry_position_instance_id(strategy_id, fill.symbol, fill.side, fill.timestamp)
                ),
                qty=fill.qty,
                avg_entry=fill.fill_price,
                entry_time=fill.timestamp,
                entry_commission=fill.commission,
                metadata=_fill_metadata(fill, "entry"),
            )
            return None

        total_qty = existing.qty + fill.qty
        if total_qty > 0:
            existing.avg_entry = (
                existing.avg_entry * existing.qty + fill.fill_price * fill.qty
            ) / total_qty
        existing.qty = total_qty
        existing.entry_commission += fill.commission
        if fill.timestamp < existing.entry_time:
            existing.entry_time = fill.timestamp
        _merge_fill_metadata(existing.metadata, fill, "entry")
        return None

    def _apply_exit(
        self,
        position: LivePositionLedgerEntry,
        fill: Fill,
        key: tuple[str, str, Side],
    ) -> Trade | None:
        close_qty = min(position.qty, fill.qty)
        if position.direction == Side.LONG:
            price_pnl = close_qty * (fill.fill_price - position.avg_entry)
        else:
            price_pnl = close_qty * (position.avg_entry - fill.fill_price)

        position.realized_price_pnl += price_pnl
        position.exit_commission += fill.commission
        position.closed_qty += close_qty
        position.qty -= close_qty
        _merge_fill_metadata(position.metadata, fill, "exit")

        if position.qty > 1e-12:
            return None

        del self._positions[key]
        commission = position.entry_commission + position.exit_commission
        avg_exit_price = fill.fill_price
        if position.closed_qty > 0:
            if position.direction == Side.LONG:
                avg_exit_price = position.avg_entry + (
                    position.realized_price_pnl / position.closed_qty
                )
            else:
                avg_exit_price = position.avg_entry - (
                    position.realized_price_pnl / position.closed_qty
                )
        trade = Trade(
            trade_id=f"live_{position.position_instance_id}:{int(fill.timestamp.timestamp() * 1000)}",
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.avg_entry,
            exit_price=avg_exit_price,
            qty=position.closed_qty,
            entry_time=position.entry_time,
            exit_time=fill.timestamp,
            pnl=position.realized_price_pnl - position.funding_paid,
            r_multiple=None,
            commission=commission,
            bars_held=0,
            setup_grade=None,
            exit_reason=fill.tag or "exchange_fill",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=position.funding_paid,
            mae_r=None,
            mfe_r=None,
        )
        setattr(trade, "instrumentation_context", {
            "position_instance_id": position.position_instance_id,
            "entry_fill_ids": list(position.metadata.get("entry_fill_ids", [])),
            "exit_fill_ids": list(position.metadata.get("exit_fill_ids", [])),
            "entry_order_ids": list(position.metadata.get("entry_order_ids", [])),
            "exit_order_ids": list(position.metadata.get("exit_order_ids", [])),
        })
        return trade


def _fill_metadata(fill: Fill, role: str) -> dict[str, list[str]]:
    metadata: dict[str, list[str]] = {
        "entry_fill_ids": [],
        "exit_fill_ids": [],
        "entry_order_ids": [],
        "exit_order_ids": [],
    }
    _merge_fill_metadata(metadata, fill, role)
    return metadata


def _merge_fill_metadata(metadata: dict[str, Any], fill: Fill, role: str) -> None:
    role_key = "entry" if role == "entry" else "exit"
    _append_unique(metadata.setdefault(f"{role_key}_fill_ids", []), fill_identity(fill))
    order_ids = metadata.setdefault(f"{role_key}_order_ids", [])
    for order_id in (fill.order_id, fill.exchange_order_id):
        if order_id:
            _append_unique(order_ids, str(order_id))


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
