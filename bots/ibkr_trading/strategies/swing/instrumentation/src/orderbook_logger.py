"""Order book context logging — captures bid/ask depth at trade events.

Writes OrderBookContext events to JSONL for liquidity analysis.
When IBKR depth data is available, includes top-5 levels and 10bps depth.
Degrades gracefully to zeros when depth data is unavailable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .event_metadata import create_event_metadata
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.orderbook_logger")


@dataclass
class OrderBookContext:
    """Point-in-time order book snapshot."""

    bot_id: str
    pair: str
    timestamp: str
    best_bid: float
    best_ask: float
    spread_bps: float = 0.0
    bid_depth_10bps: float = 0.0
    ask_depth_10bps: float = 0.0
    bid_levels: Optional[list[dict]] = None  # top 5: [{"price": 100.0, "size": 50}, ...]
    ask_levels: Optional[list[dict]] = None  # top 5
    trade_context: Optional[str] = None      # "entry", "exit", "signal_eval"
    related_trade_id: Optional[str] = None
    event_id: str = ""
    event_metadata: dict = field(default_factory=dict)

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

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "orderbook"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_source_id = config.get("data_source_id", "ibkr_execution")
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )

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
        now = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = now.isoformat() if isinstance(now, datetime) else str(now)

        # Compute spread in basis points
        mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0
        spread_bps = round((best_ask - best_bid) / mid * 10_000, 2) if mid > 0 else 0.0

        meta = create_event_metadata(
            bot_id=self.bot_id,
            event_type="orderbook_context",
            payload_key=f"{pair}:{trade_context or 'snapshot'}:{ts_str}",
            exchange_timestamp=now if isinstance(now, datetime) else datetime.now(timezone.utc),
            data_source_id=self.data_source_id,
            lineage=self._lineage,
        )

        ctx = OrderBookContext(
            bot_id=self.bot_id,
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
            event_id=meta.event_id,
            event_metadata=meta.to_dict(),
        )

        self._write_event(ctx)
        return ctx

    def _write_event(self, ctx: OrderBookContext) -> None:
        try:
            date_str = ctx.timestamp[:10] if ctx.timestamp else (
                datetime.now(timezone.utc).strftime("%Y-%m-%d")
            )
            filepath = self.data_dir / f"orderbook_{date_str}.jsonl"
            payload = enrich_payload(
                ctx.to_dict(),
                lineage=self._lineage,
                event_type="orderbook_context",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write OrderBookContext")
