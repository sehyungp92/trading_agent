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

logger = logging.getLogger("instrumentation.order_logger")


@dataclass
class OrderEvent:
    """A single order lifecycle event (submit, fill, partial, reject, cancel)."""

    # Identity
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

    # Futures-specific fields
    strategy_type: str = ""                # helix | nqdtc | vdubus
    session: str = ""                      # ETH | RTH
    contract_month: str = ""               # e.g., "2026-06"
    order_book_depth: Optional[dict] = None

    # Standard metadata
    event_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class OrderLogger:
    """Writes OrderEvent records to JSONL files."""

    def __init__(self, config: dict, strategy_type: str = ""):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "orders"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_id = config.get("experiment_id", "") or ""
        self._experiment_variant = config.get("experiment_variant", "") or ""
        self._strategy_type = strategy_type
        self._data_source_id = config.get("data_source_id", "ibkr_cme_nq")
        self._lineage = lineage_from_config(
            config,
            family_id="stock",
            strategy_id=config.get("strategy_id", ""),
        )

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
        strategy_type: str = "",
        session: str = "",
        contract_month: str = "",
        order_book_depth: Optional[dict] = None,
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
            payload_key=f"{order_id}:{status}",
            exchange_timestamp=now,
            data_source_id=self._data_source_id,
            bar_id=bar_id,
            lineage=self._lineage,
        )

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
            strategy_type=strategy_type or self._strategy_type,
            session=session,
            contract_month=contract_month,
            order_book_depth=order_book_depth,
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
