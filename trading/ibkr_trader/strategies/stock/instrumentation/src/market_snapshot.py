import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.market_snapshot")


@dataclass
class MarketSnapshot:
    """
    Point-in-time capture of market state for a single symbol.
    Referenced by trade events and missed opportunity events.
    """
    snapshot_id: str
    symbol: str
    timestamp: str                # exchange time, ISO 8601
    bid: float
    ask: float
    mid: float                    # (bid + ask) / 2
    spread_bps: float             # (ask - bid) / mid * 10000
    last_trade_price: float
    volume_1m: Optional[float] = None
    volume_5m: Optional[float] = None
    volume_24h: Optional[float] = None
    atr_14: Optional[float] = None
    funding_rate: Optional[float] = None   # N/A for equity futures
    open_interest: Optional[float] = None  # N/A for equity futures
    mark_price: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class MarketSnapshotService:
    """
    Captures and stores market snapshots for NQ/MNQ futures via IBKR data.

    Usage:
        service = MarketSnapshotService(config, data_provider)
        snapshot = service.capture_now("NQ")  # on-demand for trade events
    """

    def __init__(self, config: dict, data_provider=None):
        """
        Args:
            config: from instrumentation_config.yaml
            data_provider: object with get_bid_ask(symbol) and get_atr(symbol) methods,
                or None for standalone use (degraded snapshots only)
        """
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "snapshots"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.interval = config.get("market_snapshots", {}).get("interval_seconds", 60)
        self.symbols = config.get("market_snapshots", {}).get("symbols", [])
        self._data_provider = data_provider
        self.data_provider = data_provider
        self.data_source_id = config.get("data_source_id", "ibkr_us_equities")
        self._lineage = lineage_from_config(
            config,
            family_id="stock",
            strategy_id=config.get("strategy_id", ""),
        )
        self._cache: Dict[str, MarketSnapshot] = {}

    def set_data_provider(self, data_provider) -> None:
        self._data_provider = data_provider
        self.data_provider = data_provider

    def _compute_snapshot_id(self, symbol: str, timestamp: str) -> str:
        raw = f"{symbol}|{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def capture_now(self, symbol: str) -> MarketSnapshot:
        """
        Capture a snapshot immediately. Call this when a trade or signal occurs.
        Returns the snapshot and caches it.

        Adapted for IBKR NQ/MNQ futures:
        - Bid/ask from IB market data subscription (cached in engine._bid/_ask)
        - ATR from strategy's computed indicator arrays
        - No funding rate or open interest (equity futures)
        """
        try:
            bid = 0.0
            ask = 0.0
            last = 0.0
            atr_14 = None

            provider = self._data_provider
            if provider is not None:
                # Get bid/ask from the data provider
                # The data provider interface:
                #   get_bid_ask(symbol) -> (bid, ask)
                #   get_last_price(symbol) -> float
                #   get_atr(symbol) -> Optional[float]
                try:
                    bid, ask = provider.get_bid_ask(symbol)
                except (AttributeError, TypeError):
                    pass

                try:
                    last = provider.get_last_price(symbol)
                except (AttributeError, TypeError):
                    if bid > 0 and ask > 0:
                        last = (bid + ask) / 2

                try:
                    atr_14 = provider.get_atr(symbol)
                except (AttributeError, TypeError):
                    pass

            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 and bid > 0 and ask > 0 else 0.0

            now = datetime.now(timezone.utc)
            ts_str = now.isoformat()

            snapshot = MarketSnapshot(
                snapshot_id=self._compute_snapshot_id(symbol, ts_str),
                symbol=symbol,
                timestamp=ts_str,
                bid=bid,
                ask=ask,
                mid=round(mid, 4),
                spread_bps=round(spread_bps, 2),
                last_trade_price=last,
                atr_14=atr_14,
            )

            self._cache[symbol] = snapshot
            self._write_snapshot(snapshot)
            return snapshot

        except Exception as e:
            logger.warning("Snapshot capture failed for %s: %s", symbol, e)
            now = datetime.now(timezone.utc)
            ts_str = now.isoformat()
            degraded = MarketSnapshot(
                snapshot_id=self._compute_snapshot_id(symbol, ts_str),
                symbol=symbol,
                timestamp=ts_str,
                bid=0, ask=0, mid=0, spread_bps=0,
                last_trade_price=0,
            )
            self._write_snapshot(degraded)
            return degraded

    def get_latest(self, symbol: str) -> Optional[MarketSnapshot]:
        """Return the most recent cached snapshot for a symbol."""
        return self._cache.get(symbol)

    def _write_snapshot(self, snapshot: MarketSnapshot):
        """Append snapshot to daily JSONL file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"snapshots_{today}.jsonl"
            payload = enrich_payload(
                snapshot.to_dict(),
                lineage=self._lineage,
                event_type="market_snapshot",
                scope="strategy",
            )
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write snapshot: %s", e)

    def run_periodic(self):
        """Capture snapshots for all configured symbols."""
        for symbol in self.symbols:
            self.capture_now(symbol)

    def cleanup_old_files(self, max_age_days: int = 30):
        """Delete snapshot files older than max_age_days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        for filepath in self.data_dir.glob("snapshots_*.jsonl"):
            try:
                date_str = filepath.stem.replace("snapshots_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    filepath.unlink()
            except (ValueError, OSError):
                pass
