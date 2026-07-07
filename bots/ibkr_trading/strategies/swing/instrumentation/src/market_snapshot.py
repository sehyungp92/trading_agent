"""Market Snapshot Service — captures point-in-time market state.

Adapted for Interactive Brokers via ib_async.  The bot uses
``reqHistoricalDataAsync`` for OHLC bars; bid/ask are not currently
subscribed so they default to 0 (degraded but safe).
"""
from __future__ import annotations

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
    """Point-in-time capture of market state for a single symbol."""
    snapshot_id: str              # deterministic: hash(symbol + timestamp)
    symbol: str                   # e.g. "QQQ"
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
    funding_rate: Optional[float] = None   # N/A for equities/futures
    open_interest: Optional[float] = None
    mark_price: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class MarketSnapshotService:
    """Captures and stores market snapshots.

    Usage::

        service = MarketSnapshotService(config, data_provider)
        snapshot = service.capture_now("QQQ")  # on-demand for trade events
    """

    def __init__(self, config: dict, data_provider=None):
        """
        Args:
            config: from instrumentation_config.yaml
            data_provider: dict mapping symbol -> dict with keys like
                ``last_price``, ``daily_bars`` (list of OHLC bar objects),
                ``hourly_bars``, etc.  Or an object with ``get_ticker`` /
                ``get_ohlcv`` methods.  The capture method adapts to both.
        """
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "snapshots"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.interval = config.get("market_snapshots", {}).get("interval_seconds", 60)
        self.symbols: list[str] = config.get("market_snapshots", {}).get("symbols", [])
        self.data_provider = data_provider
        self.data_source_id = "ibkr_historical"
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )
        self._cache: Dict[str, MarketSnapshot] = {}

    def _compute_snapshot_id(self, symbol: str, timestamp: str) -> str:
        raw = f"{symbol}|{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def capture_now(self, symbol: str) -> MarketSnapshot:
        """Capture a snapshot immediately.

        Adapts to the IBKR data provider — uses last bar close as price proxy
        since live bid/ask subscription is not currently active.
        """
        try:
            bid = 0.0
            ask = 0.0
            last = 0.0
            volume_24h = None
            atr_14 = None
            volume_1m = None
            volume_5m = None

            if self.data_provider is not None:
                if hasattr(self.data_provider, "get_ticker"):
                    # Generic dict-based provider (for tests / future adapters)
                    ticker = self.data_provider.get_ticker(symbol)
                    bid = float(ticker.get("bid", 0))
                    ask = float(ticker.get("ask", 0))
                    last = float(ticker.get("last", 0))
                    volume_24h = float(ticker.get("quoteVolume", 0) or ticker.get("volume", 0))
                elif isinstance(self.data_provider, dict):
                    # Strategy engine passes cached bar data
                    sym_data = self.data_provider.get(symbol, {})
                    last = float(sym_data.get("last_price", 0))
                    bid = float(sym_data.get("bid", 0))
                    ask = float(sym_data.get("ask", 0))
                    volume_24h = sym_data.get("volume_24h")

                # ATR from candle data
                try:
                    atr_14 = self._compute_atr(symbol, period=14)
                except Exception:
                    pass

                # Volume from recent candles
                try:
                    volume_1m, volume_5m = self._compute_recent_volume(symbol)
                except Exception:
                    pass

            mid = (bid + ask) / 2 if bid and ask else last
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 and bid > 0 and ask > 0 else 0.0

            now = datetime.now(timezone.utc)
            ts_str = now.isoformat()

            snapshot = MarketSnapshot(
                snapshot_id=self._compute_snapshot_id(symbol, ts_str),
                symbol=symbol,
                timestamp=ts_str,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_bps=round(spread_bps, 2),
                last_trade_price=last,
                volume_1m=volume_1m,
                volume_5m=volume_5m,
                volume_24h=volume_24h,
                atr_14=atr_14,
                funding_rate=None,      # N/A for IBKR equities/futures
                open_interest=None,
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

    def _write_snapshot(self, snapshot: MarketSnapshot) -> None:
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

    def _compute_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Compute ATR from the bot's cached candle data."""
        if self.data_provider is None:
            return None

        candles = None
        if hasattr(self.data_provider, "get_ohlcv"):
            candles = self.data_provider.get_ohlcv(symbol, timeframe="1h", limit=period + 1)
        elif isinstance(self.data_provider, dict):
            sym_data = self.data_provider.get(symbol, {})
            candles = sym_data.get("hourly_bars")

        if not candles or len(candles) < period + 1:
            return None

        trs = []
        for i in range(1, len(candles)):
            if isinstance(candles[i], (list, tuple)):
                high, low, prev_close = candles[i][2], candles[i][3], candles[i - 1][4]
            else:
                high = getattr(candles[i], "high", 0)
                low = getattr(candles[i], "low", 0)
                prev_close = getattr(candles[i - 1], "close", 0)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        return sum(trs[-period:]) / period if trs else None

    def _compute_recent_volume(self, symbol: str):
        """Compute 1m and 5m volume from recent candles."""
        if self.data_provider is None:
            return None, None

        candles_1m = None
        if hasattr(self.data_provider, "get_ohlcv"):
            candles_1m = self.data_provider.get_ohlcv(symbol, timeframe="1m", limit=5)

        if not candles_1m:
            return None, None

        if isinstance(candles_1m[-1], (list, tuple)):
            volume_1m = float(candles_1m[-1][5])
            volume_5m = sum(float(c[5]) for c in candles_1m[-5:]) if len(candles_1m) >= 5 else None
        else:
            volume_1m = float(getattr(candles_1m[-1], "volume", 0))
            volume_5m = (
                sum(float(getattr(c, "volume", 0)) for c in candles_1m[-5:])
                if len(candles_1m) >= 5 else None
            )
        return volume_1m, volume_5m

    def run_periodic(self) -> None:
        """Capture snapshots for all configured symbols."""
        for symbol in self.symbols:
            self.capture_now(symbol)

    def cleanup_old_files(self, max_age_days: int = 30) -> None:
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
