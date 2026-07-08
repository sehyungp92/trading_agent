"""Broker adapter protocol — implemented by SimBroker and (future) HyperliquidBroker."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from crypto_trader.core.models import Fill, Order, Position


@runtime_checkable
class BrokerAdapter(Protocol):
    """Unified broker interface for backtesting and live trading."""

    def submit_order(self, order: Order) -> str:
        """Submit an order. Returns the assigned order_id."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order. Returns True if successfully cancelled."""
        ...

    def cancel_all(self, symbol: str = "") -> int:
        """Cancel all open orders, optionally filtered by symbol. Returns count cancelled."""
        ...

    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol, or None if flat."""
        ...

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        """Get all open (pending/working) orders, optionally filtered by symbol."""
        ...

    def get_equity(self) -> float:
        """Get current account equity (mark-to-market)."""
        ...

    def get_fills_since(self, since: datetime) -> list[Fill]:
        """Get fills since a given timestamp."""
        ...
