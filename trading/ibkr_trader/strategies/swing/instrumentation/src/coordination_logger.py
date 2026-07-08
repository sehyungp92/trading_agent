"""Coordination Action Logger — structured logging for cross-strategy decisions.

Captures StrategyCoordinator actions (stop tightening, size boosts, overlay
signal changes) as JSONL events that flow through the sidecar pipeline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .event_metadata import create_event_metadata
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.coordination_logger")


@dataclass
class CoordinationEvent:
    """Single cross-strategy coordination action."""
    timestamp: str
    event_metadata: dict
    action: str               # "tighten_stop_be" | "size_boost" | "overlay_signal_change"
    trigger_strategy: str     # e.g. "ATRSS"
    target_strategy: str      # e.g. "AKC_HELIX"
    symbol: str
    rule: str                 # "rule_1" | "rule_2" | "ema_crossover"
    details: dict             # rule-specific: {old_stop, new_stop, direction, ...}
    outcome: str              # "applied" | "skipped_already_tighter" | "emitted"

    def to_dict(self) -> dict:
        return asdict(self)


class CoordinationLogger:
    """Writes coordination events to daily JSONL files.

    Follows the same pattern as TradeLogger: one file per day, append-only,
    never crashes the trading system.

    Usage::

        cl = CoordinationLogger(config)
        cl.log_action(
            action="tighten_stop_be",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="QQQ",
            rule="rule_1",
            details={"old_stop": 480.0, "new_stop": 485.0},
            outcome="applied",
        )
    """

    def __init__(self, config: dict) -> None:
        self.bot_id = config.get("bot_id", "swing_multi_01")
        self.data_dir = Path(config.get("data_dir", "instrumentation/data")) / "coordination"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_source_id = config.get("data_source_id", "ibkr_execution")
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )
        self._event_count = 0

    _ORDER_MODIFICATION_ACTIONS = frozenset({
        "tighten_stop_be", "loosen_stop", "adjust_size", "size_boost",
        "force_exit", "cancel_order",
    })

    _ORDER_ACTION_MAP = {
        "tighten_stop_be": "MODIFY",
        "loosen_stop": "MODIFY",
        "adjust_size": "MODIFY",
        "size_boost": "MODIFY",
        "force_exit": "NEW",
        "cancel_order": "CANCEL",
    }

    def log_action(
        self,
        action: str,
        trigger_strategy: str,
        target_strategy: str,
        symbol: str,
        rule: str,
        details: Optional[dict] = None,
        outcome: str = "applied",
        order_logger: object = None,
        related_order_id: str = "",
        related_trade_id: str = "",
    ) -> Optional[CoordinationEvent]:
        """Log a coordination action and optionally emit an OrderEvent. Never raises."""
        try:
            now = datetime.now(timezone.utc)
            metadata = create_event_metadata(
                bot_id=self.bot_id,
                event_type="coordinator_action",
                payload_key=f"{action}_{symbol}_{now.isoformat()}",
                exchange_timestamp=now,
                data_source_id=self.data_source_id,
                lineage=self._lineage,
            )

            event = CoordinationEvent(
                timestamp=now.isoformat(),
                event_metadata=metadata.to_dict(),
                action=action,
                trigger_strategy=trigger_strategy,
                target_strategy=target_strategy,
                symbol=symbol,
                rule=rule,
                details=details or {},
                outcome=outcome,
            )

            self._write_event(event)

            # If the action resulted in an order modification, also log an OrderEvent
            if (
                outcome == "applied"
                and order_logger is not None
                and action in self._ORDER_MODIFICATION_ACTIONS
            ):
                try:
                    order_logger.log_order(
                        order_id=related_order_id or f"coord_{self._event_count}",
                        pair=symbol,
                        side="",
                        order_type="STOP" if "stop" in action else "MARKET",
                        status="SUBMITTED",
                        requested_qty=0,
                        strategy_id=target_strategy,
                        order_action=self._ORDER_ACTION_MAP.get(action, "MODIFY"),
                        coordinator_triggered=True,
                        coordinator_rule=rule,
                        modification_details=details,
                    )
                except Exception as e:
                    logger.debug("OrderEvent emission from coordinator failed: %s", e)

            if not hasattr(self, "_event_count"):
                self._event_count = 0
            self._event_count += 1
            return event

        except Exception as e:
            logger.warning("CoordinationLogger.log_action failed: %s", e)
            return None

    def _write_event(self, event: CoordinationEvent) -> None:
        """Append event to daily JSONL file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"coordination_{today}.jsonl"
            payload = enrich_payload(
                event.to_dict(),
                lineage=self._lineage,
                event_type="coordinator_action",
                scope="family",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write coordination event: %s", e)
