"""Order-level event logging for tracking order lifecycle."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger(__name__)


@dataclass
class OrderEvent:
    """A single order lifecycle event, including coordinator-triggered modifications."""

    # Identity (base — shared across all bots)
    order_id: str
    bot_id: str
    pair: str
    side: str = ""                         # LONG | SHORT

    # Order details
    order_type: str = ""                   # MARKET | LIMIT | STOP | STOP_LIMIT
    status: str = ""                       # SUBMITTED | FILLED | PARTIAL_FILL | REJECTED | CANCELLED
    requested_qty: float = 0.0
    filled_qty: float = 0.0
    requested_price: Optional[float] = None
    fill_price: Optional[float] = None
    slippage_bps: Optional[float] = None

    # Context
    reject_reason: str = ""
    timestamp: str = ""
    latency_ms: Optional[float] = None
    related_trade_id: str = ""

    # Experiment tracking
    experiment_id: str = ""
    experiment_variant: str = ""

    # swing_trader-specific fields
    strategy_id: str = ""                  # ATRSS | AKC_HELIX | OVERLAY
    order_action: str = "NEW"              # NEW | MODIFY | CANCEL
    coordinator_triggered: bool = False    # True if coordinator rule caused this order
    coordinator_rule: str = ""             # which rule (e.g., "tighten_stop_be", "size_boost")
    modification_details: Optional[dict] = None  # for MODIFY: what changed
    overlay_state: Optional[dict] = None   # current overlay state at time of order
    drawdown_tier: str = ""                # NORMAL | CAUTION | DANGER | HALT
    market_session: str = ""               # PRE | RTH | ETH_POST

    # Standard metadata
    event_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class OrderLogger:
    """Writes OrderEvent records to JSONL files."""

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "orders"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_id = config.get("experiment_id", "")
        self._experiment_variant = config.get("experiment_variant", "")
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )
        self._event_count = 0

    def log_order(
        self,
        order_id: str,
        pair: str,
        side: str,
        order_type: str,
        status: str,
        requested_qty: float,
        filled_qty: float = 0.0,
        requested_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        reject_reason: str = "",
        latency_ms: Optional[float] = None,
        related_trade_id: str = "",
        strategy_id: str = "",
        order_action: str = "NEW",
        coordinator_triggered: bool = False,
        coordinator_rule: str = "",
        modification_details: Optional[dict] = None,
        overlay_state: Optional[dict] = None,
        drawdown_tier: str = "",
        market_session: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> OrderEvent:
        """Record an order lifecycle event."""
        now = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = now.isoformat() if isinstance(now, datetime) else str(now)

        slippage_bps = None
        if fill_price is not None and requested_price is not None and requested_price > 0:
            slippage_bps = round(
                abs(fill_price - requested_price) / requested_price * 10_000, 2
            )

        from .event_metadata import create_event_metadata
        meta = create_event_metadata(
            bot_id=self.bot_id,
            event_type="order",
            payload_key=f"{order_id}:{status}:{order_action}",
            exchange_timestamp=now,
            data_source_id="ibkr_execution",
            bar_id=bar_id,
            lineage=self._lineage,
        )

        self._event_count += 1

        event = OrderEvent(
            order_id=order_id,
            bot_id=self.bot_id,
            pair=pair,
            side=side,
            order_type=order_type,
            status=status,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            requested_price=requested_price,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            reject_reason=reject_reason,
            timestamp=ts_str,
            latency_ms=latency_ms,
            related_trade_id=related_trade_id,
            experiment_id=self._experiment_id,
            experiment_variant=self._experiment_variant,
            strategy_id=strategy_id,
            order_action=order_action,
            coordinator_triggered=coordinator_triggered,
            coordinator_rule=coordinator_rule,
            modification_details=modification_details,
            overlay_state=overlay_state,
            drawdown_tier=drawdown_tier,
            market_session=market_session,
            event_metadata=meta.to_dict() if hasattr(meta, "to_dict") else meta,
        )

        self._write_event(event)
        return event

    def _write_event(self, event: OrderEvent) -> None:
        try:
            date_str = event.timestamp[:10] if event.timestamp else datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"orders_{date_str}.jsonl"
            payload = enrich_payload(
                event.to_dict(),
                lineage=self._lineage,
                event_type="order",
                scope="oms",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write OrderEvent %s", event.order_id)
