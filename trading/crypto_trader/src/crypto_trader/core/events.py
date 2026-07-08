"""Typed event system for decoupled communication."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import structlog

from crypto_trader.core.models import Bar, Fill, Trade

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Base event with timestamp."""
    timestamp: datetime


@dataclass
class BarEvent(Event):
    """Emitted after a bar is processed."""
    bar: Bar


@dataclass
class FillEvent(Event):
    """Emitted when an order is filled."""
    fill: Fill


@dataclass
class PositionClosedEvent(Event):
    """Emitted when a round-trip trade completes."""
    trade: Trade


@dataclass
class CanonicalRuntimeEvent(Event):
    """Canonical parity event emitted beside legacy runtime callbacks."""

    stream: str
    payload: dict[str, Any]


@dataclass
class InstrumentedTradeEmitted(Event):
    """Emitted when an instrumented trade event is produced."""
    event: Any  # InstrumentedTradeEvent (avoid circular import)


@dataclass
class MissedOpportunityEmitted(Event):
    """Emitted when a missed opportunity event is produced."""
    event: Any  # MissedOpportunityEvent (avoid circular import)


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

class EventBus:
    """Simple synchronous publish-subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type[Event], handler: Callable[[Any], None]) -> None:
        """Register a handler for a specific event type."""
        self._handlers[event_type].append(handler)

    def emit(self, event: Event) -> None:
        """Dispatch an event to all registered handlers."""
        for handler in self._handlers.get(type(event), []):
            try:
                handler(event)
            except Exception:
                log.exception("event_handler.error", event_type=type(event).__name__)
