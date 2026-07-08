"""Cross-strategy coordinator — monitors fills and positions across strategies.

Implements:
  Rule 1: ATRSS entry fill on symbol X -> tighten Helix stop to breakeven on X
  Rule 2: has_atrss_position() for Helix size boost (1.25x when ATRSS active same direction)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Lightweight position record for coordination tracking."""
    qty: int
    direction: str          # "LONG" or "SHORT"
    entry_price: float = 0.0


class StrategyCoordinator:
    """Monitors cross-strategy events and emits coordination signals.

    Wired into OMS fill callbacks to track positions across all strategies.
    Strategies query the coordinator for cross-strategy state (e.g., whether
    ATRSS has an active position on a given symbol).
    """

    def __init__(self, bus, repo=None):
        self._bus = bus
        self._repo = repo
        # (strategy_id, symbol) → OpenPosition
        self._position_book: dict[tuple[str, str], OpenPosition] = {}
        self._on_action: Callable | None = None

    def set_action_logger(self, callback: Callable) -> None:
        """Register a callback for logging coordination actions."""
        self._on_action = callback

    def log_action(self, **kwargs) -> None:
        """Delegate to registered action logger. Never crashes."""
        if self._on_action is None:
            return
        try:
            self._on_action(**kwargs)
        except Exception as e:
            logger.warning("Coordinator action logging failed: %s", e)

    def on_fill(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        role: str,
        price: float = 0.0,
    ) -> None:
        """Called on every fill across all strategies.

        Args:
            strategy_id: Which strategy received the fill
            symbol: Instrument symbol (e.g. "QQQ")
            side: "BUY" or "SELL"
            role: "ENTRY", "EXIT", "STOP", "TP"
            price: Fill price
        """
        # Rule 1: ATRSS entry fill → check for open Helix on same symbol
        if strategy_id == "ATRSS" and role == "ENTRY":
            helix_key = ("AKC_HELIX", symbol)
            helix_pos = self._position_book.get(helix_key)
            if helix_pos and helix_pos.qty > 0:
                logger.info(
                    "Rule 1: ATRSS entry on %s — emitting TIGHTEN_STOP_BE to Helix",
                    symbol,
                )
                self._bus.emit_coordination_event(
                    target_strategy="AKC_HELIX",
                    event_type="TIGHTEN_STOP_BE",
                    symbol=symbol,
                )
                self.log_action(
                    action="tighten_stop_be",
                    trigger_strategy="ATRSS",
                    target_strategy="AKC_HELIX",
                    symbol=symbol,
                    rule="rule_1",
                    details={"fill_price": price, "direction": side},
                    outcome="emitted",
                )

    def on_position_update(
        self,
        strategy_id: str,
        symbol: str,
        qty: int,
        direction: str,
        entry_price: float = 0.0,
    ) -> None:
        """Track positions for coordination rules.

        Called after entry fills (qty > 0) and after exit fills (qty == 0).
        """
        key = (strategy_id, symbol)
        if qty > 0:
            self._position_book[key] = OpenPosition(
                qty=qty, direction=direction, entry_price=entry_price,
            )
        else:
            self._position_book.pop(key, None)

    def has_atrss_position(
        self, symbol: str, direction: Optional[str] = None
    ) -> bool:
        """Rule 2: Check if ATRSS has an active position on symbol.

        Used by Helix to decide whether to apply 1.25x size boost.
        Only boosts when directions match (same-direction confirmed).
        """
        pos = self._position_book.get(("ATRSS", symbol))
        if pos is None or pos.qty <= 0:
            return False
        if direction and pos.direction != direction:
            return False
        return True

    def get_position(
        self, strategy_id: str, symbol: str
    ) -> Optional[OpenPosition]:
        """Get a specific strategy's position on a symbol."""
        return self._position_book.get((strategy_id, symbol))

    def get_all_positions(self) -> dict[tuple[str, str], OpenPosition]:
        """Return full position book (for logging/debug)."""
        return dict(self._position_book)
