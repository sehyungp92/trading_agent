"""Order book context logger — emits OrderBookContext JSONL events."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import LineageContext

logger = logging.getLogger("instrumentation.orderbook_logger")


@dataclass
class OrderBookContext:
    """Point-in-time order book state, captured at trade entry/exit or signal eval."""

    bot_id: str
    pair: str
    timestamp: str
    best_bid: float
    best_ask: float
    spread_bps: float = 0.0
    bid_depth_10bps: float = 0.0
    ask_depth_10bps: float = 0.0
    bid_levels: Optional[list[dict]] = None  # top 10 levels
    ask_levels: Optional[list[dict]] = None
    trade_context: Optional[str] = None  # "entry", "exit", "signal_eval"
    related_trade_id: Optional[str] = None
    event_id: str = ""

    @property
    def imbalance_ratio(self) -> float:
        if self.ask_depth_10bps <= 0:
            return 0.0
        return round(self.bid_depth_10bps / self.ask_depth_10bps, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["imbalance_ratio"] = self.imbalance_ratio
        return d


class OrderBookLogger:
    """Writes OrderBookContext events to daily JSONL files."""

    def __init__(self, data_dir: str | Path, bot_id: str, lineage: LineageContext | dict | None = None) -> None:
        self._data_dir = Path(data_dir) / "orderbook"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id
        self._lineage = lineage or {"bot_id": bot_id}

    def log_context(
        self,
        pair: str,
        best_bid: float,
        best_ask: float,
        trade_context: Optional[str] = None,
        related_trade_id: Optional[str] = None,
        bid_depth_10bps: float = 0.0,
        ask_depth_10bps: float = 0.0,
        bid_levels: Optional[list[dict]] = None,
        ask_levels: Optional[list[dict]] = None,
        exchange_timestamp: Optional[datetime] = None,
    ) -> OrderBookContext:
        """Record an order book context snapshot."""
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.0
        spread_bps = round((best_ask - best_bid) / mid * 10000, 2) if mid > 0 else 0.0

        raw = f"{self._bot_id}|{ts_str}|orderbook_context|{pair}|{trade_context or ''}|{related_trade_id or ''}"
        event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        ctx = OrderBookContext(
            bot_id=self._bot_id,
            pair=pair,
            timestamp=ts_str,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            bid_depth_10bps=bid_depth_10bps,
            ask_depth_10bps=ask_depth_10bps,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            trade_context=trade_context,
            related_trade_id=related_trade_id,
            event_id=event_id,
        )

        self._write(ctx)
        return ctx

    def _write(self, ctx: OrderBookContext) -> None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self._data_dir / f"orderbook_{today}.jsonl"
            payload = enrich_payload(
                ctx.to_dict(),
                lineage=self._lineage,
                event_type="orderbook_context",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write order book context: %s", e)
